"""Simple regime-rotation 'DAGs': one state variable -> risk-on/off.

Each rule is the smallest possible causal DAG:  state_t -> exposure_t ->
return_{t+1}.  Risk-on holds BTC; risk-off rotates into stables (0% return —
conservative; a real stable yield would only help).  Decision uses data
through day t, exposure is applied to day t+1's return (one-day lag), with a
switching cost on every exposure change.

The point is breadth: test many dead-simple rules, see which beat or improve
on buy-and-hold BTC, and placebo-test the survivors so we don't fool
ourselves (the lesson of every other experiment here).

Rule families (all causal — trailing windows / rolling quantiles only, no
fitted thresholds chosen on the full sample):
  always_in        exposure 1 (= BH BTC; the benchmark)
  vol_target       exposure = clip(target/trailing_vol, 0, 1), continuous
  vol_off          0 when trailing vol > its rolling q-pctile, else 1
  vix_off          0 when VIX > its rolling q-pctile, else 1
  trend_N          0 when price < its N-day MA, else 1 (classic trend filter)
  funding_off      0 when aggregate perp funding < 0, else 1
  dd_off           0 when drawdown from trailing peak < -x%, else 1
  trend+vol etc.   AND-combinations (risk-on only if both agree)

Scoring vs BH BTC: annualized return, Sharpe, Sortino, max drawdown, Calmar,
% time in market, #switches.  A rule "wins on risk" if it raises Sharpe or
Calmar even at lower total return (legitimate for a risk overlay); it "wins
outright" if it also matches/beats total return.

Run:  python -m causal_portfolio.experiments.regime_rotation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import (
    ANNUALIZATION, calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio,
)

logger = logging.getLogger("cpcm.experiments.regime_rotation")


# ── signal builders (all causal) ────────────────────────────────────

def _price(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def _trailing_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std()


def _rolling_pctile_flag(s: pd.Series, q: float, window: int = 252) -> pd.Series:
    """1 when s is at/below its trailing q-quantile (risk-on), else 0.

    Trailing quantile is causal: the threshold at t uses only s[..t].
    """
    thresh = s.rolling(window, min_periods=60).quantile(q)
    return (s <= thresh).astype(float)


def _drawdown(returns: pd.Series) -> pd.Series:
    eq = _price(returns)
    return eq / eq.cummax() - 1.0


# ── exposure rules: return a Series of target BTC exposure in [0,1] ──

def build_rules(
    btc: pd.Series, vix: pd.Series | None, funding: pd.Series | None,
    *, vol_window: int = 20, q: float = 0.70,
) -> dict[str, pd.Series]:
    idx = btc.index
    vol = _trailing_vol(btc, vol_window)
    price = _price(btc)
    rules: dict[str, pd.Series] = {}

    rules["always_in"] = pd.Series(1.0, index=idx)

    # Continuous vol targeting: scale exposure to a trailing-median vol budget.
    tgt = vol.rolling(252, min_periods=60).median()
    rules["vol_target"] = (tgt / (vol + 1e-12)).clip(0, 1)

    # Risk-off when trailing vol is in its upper tail.
    rules["vol_off_70"] = 1.0 - _rolling_pctile_flag(-vol, q, 252)  # high vol -> 0
    rules["vol_off_80"] = 1.0 - _rolling_pctile_flag(-vol, 0.80, 252)

    # Trend filters (price below moving average -> risk-off).
    for n in (50, 100, 200):
        rules[f"trend_{n}"] = (price > price.rolling(n, min_periods=n // 2)
                               .mean()).astype(float)

    # Drawdown stop.
    dd = _drawdown(btc)
    rules["dd_off_20"] = (dd > -0.20).astype(float)
    rules["dd_off_30"] = (dd > -0.30).astype(float)

    if vix is not None:
        v = vix.reindex(idx).ffill()
        rules["vix_off_70"] = 1.0 - _rolling_pctile_flag(-v, q, 252)

    if funding is not None:
        f = funding.reindex(idx).ffill()
        rules["funding_off"] = (f >= 0).astype(float)

    # AND-combinations: risk-on only if BOTH agree.
    if "trend_200" in rules:
        rules["trend200_and_vol"] = rules["trend_200"] * rules["vol_off_70"]
    if "vix_off_70" in rules and "trend_200" in rules:
        rules["trend200_and_vix"] = rules["trend_200"] * rules["vix_off_70"]
    return rules


# ── backtest a single exposure path ─────────────────────────────────

@dataclass
class RotResult:
    name: str
    ann_return: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    pct_in_market: float
    n_switches: int
    daily: pd.Series


def backtest_rule(
    btc: pd.Series, exposure: pd.Series, *, fee_bps_oneway: float = 5.0,
) -> RotResult:
    e = exposure.reindex(btc.index).shift(1).fillna(0.0).clip(0, 1)
    strat = e * btc.fillna(0.0)
    turn = e.diff().abs().fillna(e.abs())
    strat = strat - turn * (fee_bps_oneway / 1e4)
    daily = strat.dropna()
    return RotResult(
        name="",
        ann_return=float(daily.mean() * ANNUALIZATION),
        sharpe=float(sharpe_ratio(daily.values)),
        sortino=float(sortino_ratio(daily.values)),
        max_dd=float(max_drawdown(daily.values)),
        calmar=float(calmar_ratio(daily.values)),
        pct_in_market=float((e > 0.01).mean()),
        n_switches=int((e.diff().abs() > 0.01).sum()),
        daily=daily,
    )


def placebo_p(
    btc: pd.Series, exposure: pd.Series, real: RotResult, *,
    metric: str = "sharpe", n: int = 200, seed: int = 0, **kw,
) -> float:
    """Share of circular time-shifts of the exposure path that match/beat the
    real rule on `metric`. High p => the TIMING carries no information (the
    rule's edge is just average exposure level, reproducible at random)."""
    rng = np.random.default_rng(seed)
    e = exposure.reindex(btc.index).fillna(0.0).clip(0, 1).values
    real_v = getattr(real, metric)
    hits = 0
    for _ in range(n):
        k = int(rng.integers(30, len(e) - 30))
        shifted = pd.Series(np.roll(e, k), index=btc.index)
        r = backtest_rule(btc, shifted, **kw)
        hits += getattr(r, metric) >= real_v
    return hits / n


# ── report ──────────────────────────────────────────────────────────

def render_markdown(results: list[RotResult], placebos: dict[str, dict],
                    bh: RotResult, args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# Simple regime-rotation rules (state → risk-on/off → return)\n"]
    L.append("Each rule rotates BTC↔stables (0% when out) on a single causal "
             "state variable; decision at t close, applied to t+1 return, "
             f"{args['fee_bps_oneway']:.0f}bp one-way switching cost. Sorted "
             "by Calmar (return / max drawdown) — the metric a risk overlay "
             "should win on.\n")
    L.append(f"- Asset: BTC | window `{args['start']}` → `{args['end']}` "
             f"({bh.daily.shape[0]} days)")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | "
             "%in mkt | switches | placebo p (Sharpe) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda r: -r.calmar):
        pp = placebos.get(r.name, {}).get("sharpe")
        pps = f"{pp:.2f}" if pp is not None else "—"
        tag = " ⟵ BH" if r.name == "always_in" else ""
        L.append(f"| {r.name}{tag} | {r.ann_return:+.0%} | {r.sharpe:.2f} | "
                 f"{r.sortino:.2f} | {r.max_dd:.0%} | {r.calmar:.2f} | "
                 f"{r.pct_in_market:.0%} | {r.n_switches} | {pps} |")
    L.append("")

    # Verdict
    better_calmar = [r for r in results
                     if r.name != "always_in" and r.calmar > bh.calmar]
    placebo_robust = [r for r in better_calmar
                      if (placebos.get(r.name, {}).get("sharpe") or 1.0) < 0.10
                      and r.sharpe > bh.sharpe]
    L.append("## Verdict\n")
    L.append(f"Buy-and-hold BTC: ann {bh.ann_return:+.0%}, Sharpe "
             f"{bh.sharpe:.2f}, maxDD {bh.max_dd:.0%}, Calmar {bh.calmar:.2f}.\n")
    if placebo_robust:
        L.append("Rules that beat BH on Sharpe AND whose timing survives the "
                 "placebo (p<0.10):")
        for r in sorted(placebo_robust, key=lambda r: -r.sharpe):
            L.append(f"- **{r.name}** — Sharpe {r.sharpe:.2f} vs {bh.sharpe:.2f}, "
                     f"maxDD {r.max_dd:.0%} vs {bh.max_dd:.0%} "
                     f"(placebo p={placebos[r.name]['sharpe']:.2f})")
    else:
        L.append("**No rule both beats BH on Sharpe and survives the placebo.** "
                 f"{len(better_calmar)} rule(s) improve Calmar (mostly by "
                 "cutting drawdown via lower average exposure), but their "
                 "TIMING does not beat a random-timed overlay — i.e. the "
                 "drawdown relief comes from being out of the market on "
                 "average, not from skillful regime calls.")
    L.append("\n*Note: trend filters typically improve Calmar by truncating "
             "bear markets; check the placebo column to see whether that is "
             "timing skill or just reduced exposure.*")
    return "\n".join(L)


def main() -> None:
    import argparse
    from pathlib import Path
    from causal_portfolio.data import get_loader
    from causal_portfolio.factors.builder import MACRO_SERIES

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--fee-bps-oneway", type=float, default=5.0)
    p.add_argument("--n-placebo", type=int, default=200)
    p.add_argument("--out", default="causal_portfolio/docs/regime_rotation.md")
    args = p.parse_args()

    loader = get_loader()
    returns = loader.load_returns(["btc"], args.start, args.end)
    returns.index = pd.to_datetime(returns.index)
    btc = returns["btc_return"]

    macro = loader.load_macro(MACRO_SERIES, args.start, args.end)
    vix = None
    if macro is not None and len(macro):
        macro.index = pd.to_datetime(macro.index)
        for c in macro.columns:
            if c.upper() == "VIXCLS":
                vix = macro[c]
                break

    funding = None
    try:
        import duckdb
        con = duckdb.connect("causal_portfolio/data/cpcm_local.duckdb",
                             read_only=True)
        fdf = con.sql("""
            select cast(time at time zone 'UTC' as date) as day,
                   sum(value) as f
            from asset_metrics_best where metric='funding_rate_8h'
            group by 1""").df()
        con.close()
        funding = pd.Series(fdf["f"].values,
                            index=pd.to_datetime(fdf["day"])).sort_index()
    except Exception as e:
        logger.info("funding unavailable: %s", e)

    rules = build_rules(btc, vix, funding)
    results = []
    for name, exp in rules.items():
        r = backtest_rule(btc, exp, fee_bps_oneway=args.fee_bps_oneway)
        r.name = name
        results.append(r)
        logger.info("%-18s ann=%+.0f%% sharpe=%.2f maxDD=%.0f%% calmar=%.2f "
                    "in=%.0f%%", name, r.ann_return * 100, r.sharpe,
                    r.max_dd * 100, r.calmar, r.pct_in_market * 100)

    bh = next(r for r in results if r.name == "always_in")
    # Placebo only the rules that beat BH on Calmar (cheaper, and the only
    # ones we'd consider).
    placebos = {}
    for r in results:
        if r.name != "always_in" and (r.sharpe > bh.sharpe
                                      or r.calmar > bh.calmar):
            placebos[r.name] = {"sharpe": placebo_p(
                btc, rules[r.name], r, metric="sharpe",
                n=args.n_placebo, fee_bps_oneway=args.fee_bps_oneway)}
            logger.info("placebo %-18s p(Sharpe)=%.2f", r.name,
                        placebos[r.name]["sharpe"])

    md = render_markdown(results, placebos, bh, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
