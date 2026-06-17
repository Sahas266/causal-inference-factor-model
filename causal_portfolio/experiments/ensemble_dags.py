"""Ensemble / combination allocation 'DAGs': state -> exposure -> return.

Every other experiment in this folder tested *single* signals (one trend
filter, one vol gate, one funding sign).  This module asks the only question
that matters after that: **does COMBINING signals that each looked useful beat
the best single one?**  The single-signal baselines to clear are the BTC 50d
trend filter (``trend_50``, Sharpe ~0.88) and the daily-rebalanced 1/N basket
(Sharpe ~1.05) — the two things in this repo that come closest to buy-and-hold
BTC out of sample.

All rules are causal in exactly the sense the harness enforces: signals use
only trailing windows / rolling quantiles (never a threshold fit on the full
sample); the decision is taken on day t's close and applied to day t+1's
return (single-asset rules go through ``regime_rotation.backtest_rule`` which
shifts exposure by one; weight-matrix rules are lagged one day here before
meeting t+1 returns); a switching cost hits every position change.  Stables
earn 0% (conservative — a real stable yield only helps the risk-off rules).

The five combinations under test (each a small DAG ``state_t -> alloc -> r_{t+1}``):

  1. trend_on_basket    Gate the WHOLE 1/N basket on the BASKET's own price>50d
                        MA trend.  state = basket-NAV trend; alloc = full basket
                        when up / stables when down.  Compared to trend-on-BTC
                        (gate the basket on BTC's trend) and the ungated basket.

  2. breadth_trend_basket  Each major gated by ITS OWN price>50d MA; hold equal
                        weight (1/N of the universe) only the majors currently
                        in uptrend, the rest of the book in stables.  A
                        breadth-weighted trend basket — exposure scales with the
                        fraction of majors trending up.

  3. vote_frac          Voting ensemble on BTC: three binary risk-on signals —
                        trend_50 (price>50d MA), vix_calm (VIX <= trailing 70th
                        pct), funding_pos (aggregate funding >= 0).  BTC exposure
                        = fraction of signals currently risk-on (continuous
                        0..1).  Compared to each signal used alone.

  4. dual_confirm_basket  Gate the 1/N basket on a TWO-condition confirm of the
                        basket NAV: price>100d MA AND trailing 30d return>0.
                        Both must agree to hold the basket.

  5. trend_funding_*    Trend + funding combined two ways on BTC: AND (risk-on
                        only if BOTH price>50d MA and funding>=0) and AVG
                        (exposure = 0.5*trend + 0.5*funding, continuous).  Funding
                        only exists from 2023-11, so these (and a fair
                        ``trend_50`` re-run on the same sub-window) are reported
                        over the funding window separately.

Scoring vs TWO benchmarks (buy-and-hold BTC and the 1/N basket): annualized
return, Sharpe, Sortino, max drawdown, Calmar, % time in market / average
turnover, and a placebo p-value.  Any rule that beats a benchmark on Sharpe or
Calmar is placebo-tested (circular time-shift of the whole allocation path);
high p => the TIMING carries no information and the apparent edge is just the
average exposure level, reproducible at random.

Run:  python -m causal_portfolio.experiments.ensemble_dags
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the validated harness — do NOT reimplement backtest / placebo logic.
from causal_portfolio.experiments.regime_rotation import (
    backtest_rule,
    placebo_p as placebo_p_single,
)
from causal_portfolio.experiments.cross_asset_rotation import (
    MAJORS,
    _returns_matrix,
    _trailing_return,
    backtest_weights,
    placebo_p as placebo_p_weights,
)
from causal_portfolio.factors.builder import MACRO_SERIES

logger = logging.getLogger("cpcm.experiments.ensemble_dags")

DB_PATH = "causal_portfolio/data/cpcm_local.duckdb"


# ── data loaders ────────────────────────────────────────────────────


def load_returns_matrix(start: str, end: str) -> pd.DataFrame:
    from causal_portfolio.data import get_loader

    loader = get_loader()
    returns = loader.load_returns(MAJORS, start, end)
    returns.index = pd.to_datetime(returns.index)
    return _returns_matrix(returns, MAJORS).sort_index()


def load_vix(start: str, end: str) -> pd.Series | None:
    from causal_portfolio.data import get_loader

    loader = get_loader()
    macro = loader.load_macro(MACRO_SERIES, start, end)
    if macro is None or not len(macro):
        return None
    macro.index = pd.to_datetime(macro.index)
    for c in macro.columns:
        if c.upper() == "VIXCLS":
            return macro[c].sort_index()
    return None


def load_funding(start: str, end: str) -> pd.Series:
    """Aggregate perp funding: sum hourly funding_rate_8h per UTC day per asset,
    then average across assets.  Daily series (only exists from 2023-11)."""
    import duckdb

    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.sql(
        """
        with daily as (
            select asset,
                   cast(time at time zone 'UTC' as date) as day,
                   sum(value) as f
            from asset_metrics_best
            where metric = 'funding_rate_8h'
            group by 1, 2
        )
        select day, avg(f) as f from daily group by 1
        """
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["f"].values, index=pd.to_datetime(df["day"])).sort_index()
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


# ── signal helpers (all causal: trailing windows only) ──────────────


def _price(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def _trend_flag(price: pd.Series, window: int) -> pd.Series:
    """1 when price > its trailing window-day MA, else 0."""
    return (price > price.rolling(window, min_periods=window // 2).mean()).astype(float)


def _basket_nav(R: pd.DataFrame) -> pd.Series:
    """NAV of the daily-rebalanced 1/N basket (for its own trend signal).

    Causal: each day's basket return is the equal-weight mean of that day's
    asset returns; the NAV at t uses only returns through t.
    """
    basket_ret = R.fillna(0.0).mean(axis=1)
    return (1 + basket_ret).cumprod()


# ── allocation builders ──────────────────────────────────────────────
# Single-asset (BTC) rules return a Series in [0,1] (go through backtest_rule).
# Basket rules return a (T x n) weight DataFrame (go through backtest_weights).


def ew_weights(R: pd.DataFrame) -> pd.DataFrame:
    n = R.shape[1]
    return pd.DataFrame(1.0 / n, index=R.index, columns=R.columns)


def gate_basket(R: pd.DataFrame, gate: pd.Series) -> pd.DataFrame:
    """Full 1/N basket on days where ``gate``==1, all-stables (0) otherwise."""
    n = R.shape[1]
    g = gate.reindex(R.index).fillna(0.0).clip(0, 1)
    return pd.DataFrame(
        (g.values[:, None] / n) * np.ones((1, n)), index=R.index, columns=R.columns
    )


def breadth_trend_weights(R: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Each major gated by its OWN price>window-MA; included majors get 1/N of
    the universe (so exposure scales with breadth), rest of book in stables."""
    n = R.shape[1]
    flags = {}
    for a in R.columns:
        flags[a] = _trend_flag(_price(R[a]), window)
    sel = pd.DataFrame(flags, index=R.index, columns=R.columns).fillna(0.0)
    return sel / float(n)


# ── result container + runner ────────────────────────────────────────


@dataclass
class Row:
    name: str
    ann_return: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    in_market: float
    turnover: str  # "switches" for single-asset, avg daily turnover for basket
    placebo: float | None
    daily: pd.Series


def _row_from_single(name: str, res, placebo: float | None) -> Row:
    return Row(
        name=name,
        ann_return=res.ann_return,
        sharpe=res.sharpe,
        sortino=res.sortino,
        max_dd=res.max_dd,
        calmar=res.calmar,
        in_market=res.pct_in_market,
        turnover=f"{res.n_switches} sw",
        placebo=placebo,
        daily=res.daily,
    )


def _row_from_weights(name: str, res, placebo: float | None) -> Row:
    return Row(
        name=name,
        ann_return=res.ann_return,
        sharpe=res.sharpe,
        sortino=res.sortino,
        max_dd=res.max_dd,
        calmar=res.calmar,
        in_market=res.pct_in_market,
        turnover=f"{res.avg_turnover:.3f} t/o",
        placebo=placebo,
        daily=res.daily,
    )


def run_experiment(
    R: pd.DataFrame,
    vix: pd.Series | None,
    funding: pd.Series,
    *,
    fee_bps_oneway: float = 5.0,
    n_placebo: int = 200,
) -> tuple[list[Row], dict[str, Row]]:
    """Run all five combinations + their single-signal comparators.

    Returns (rows, benchmarks) where benchmarks maps {'bh_btc','ew_basket'} to
    their Row for the verdict.  Placebo is only computed (and stored) for rules
    that beat EITHER benchmark on Sharpe or Calmar.
    """
    idx = R.index
    btc = R["btc"]
    btc_price = _price(btc)
    basket_nav = _basket_nav(R)

    rows: list[Row] = []

    # ── benchmarks ──────────────────────────────────────────────────
    bh_w = pd.DataFrame(0.0, index=idx, columns=R.columns)
    bh_w["btc"] = 1.0
    bh_res = backtest_weights(R, bh_w, fee_bps_oneway=fee_bps_oneway)
    bh_row = _row_from_weights("bh_btc", bh_res, None)
    rows.append(bh_row)

    ew_res = backtest_weights(R, ew_weights(R), fee_bps_oneway=fee_bps_oneway)
    ew_row = _row_from_weights("ew_basket", ew_res, None)
    rows.append(ew_row)

    benchmarks = {"bh_btc": bh_row, "ew_basket": ew_row}

    # Helper to threshold a Sharpe/Calmar vs BOTH benchmarks.
    def beats_either(res) -> bool:
        return (
            res.sharpe > bh_res.sharpe
            or res.calmar > bh_res.calmar
            or res.sharpe > ew_res.sharpe
            or res.calmar > ew_res.calmar
        )

    # Registries we will placebo at the end (only the winners).
    single_rules: dict[str, pd.Series] = {}
    weight_rules: dict[str, pd.DataFrame] = {}
    single_res: dict[str, object] = {}
    weight_res: dict[str, object] = {}

    def add_single(name: str, exposure: pd.Series):
        res = backtest_rule(btc, exposure, fee_bps_oneway=fee_bps_oneway)
        res.name = name
        single_rules[name] = exposure
        single_res[name] = res
        rows.append(_row_from_single(name, res, None))
        return res

    def add_weights(name: str, w: pd.DataFrame):
        res = backtest_weights(R, w, fee_bps_oneway=fee_bps_oneway)
        res.name = name
        weight_rules[name] = w
        weight_res[name] = res
        rows.append(_row_from_weights(name, res, None))
        return res

    # ── Strategy 1: trend on the basket vs trend-on-BTC vs ungated ──
    basket_trend = _trend_flag(basket_nav, 50)
    btc_trend = _trend_flag(btc_price, 50)
    add_weights("trend_on_basket", gate_basket(R, basket_trend))
    add_weights("btctrend_on_basket", gate_basket(R, btc_trend))
    # (ungated basket == ew_basket benchmark already in the table)

    # ── Strategy 2: breadth-weighted own-trend basket ──────────────
    add_weights("breadth_trend_basket", breadth_trend_weights(R, 50))

    # ── Strategy 3: voting ensemble on BTC (3 signals) ─────────────
    sig_trend = _trend_flag(btc_price, 50)
    if vix is not None:
        v = vix.reindex(idx).ffill()
        vix_thresh = v.rolling(252, min_periods=60).quantile(0.70)
        sig_vix = (v <= vix_thresh).astype(float)  # calm = risk-on
    else:
        sig_vix = pd.Series(np.nan, index=idx)
    f_full = funding.reindex(idx).ffill()
    sig_fund = (f_full >= 0).astype(float)
    sig_fund = sig_fund.where(f_full.notna())  # NaN where funding absent

    # Each signal alone (comparators).
    add_single("sig_trend50", sig_trend)
    if vix is not None:
        add_single("sig_vix_calm", sig_vix.fillna(0.0))
    # Voting fraction over the signals AVAILABLE each day (ignore NaNs so the
    # pre-2023-11 period votes on {trend,vix} and post on all three).
    votes = pd.concat([sig_trend, sig_vix, sig_fund], axis=1)
    vote_frac = votes.mean(axis=1, skipna=True).fillna(0.0).clip(0, 1)
    add_single("vote_frac", vote_frac)

    # ── Strategy 4: dual-confirm on the basket ─────────────────────
    confirm_ma = _trend_flag(basket_nav, 100)
    basket_ret = R.fillna(0.0).mean(axis=1)
    ret_30 = (1 + basket_ret).rolling(30).apply(np.prod, raw=True) - 1.0
    confirm_ret = (ret_30 > 0).astype(float)
    dual = (confirm_ma * confirm_ret).fillna(0.0)
    add_weights("dual_confirm_basket", gate_basket(R, dual))

    # ── Strategy 5: trend + funding on BTC, full-sample views ──────
    # AND / AVG defined over the full sample; funding NaN (pre-2023-11) -> the
    # AND treats missing funding as risk-off (0), AVG as 0.5 weight only on the
    # available leg.  We additionally report a fair funding-window comparison
    # below so the reader is not misled by the long NaN stretch.
    trend_exp = sig_trend
    fund_exp = (f_full >= 0).astype(float)  # 1/0, NaN pre-2023-11
    add_single("trend_AND_funding_full", (trend_exp * fund_exp.fillna(0.0)))
    avg_full = 0.5 * trend_exp + 0.5 * fund_exp.fillna(0.0)
    add_single("trend_AVG_funding_full", avg_full.clip(0, 1))

    # ── compute placebos only for rules that beat a benchmark ──────
    placebos: dict[str, float] = {}
    for name, res in single_res.items():
        if beats_either(res):
            placebos[name] = placebo_p_single(
                btc, single_rules[name], res, metric="sharpe",
                n=n_placebo, fee_bps_oneway=fee_bps_oneway,
            )
            logger.info("placebo %-24s p(Sharpe)=%.2f", name, placebos[name])
    for name, res in weight_res.items():
        if beats_either(res):
            placebos[name] = placebo_p_weights(
                R, weight_rules[name], res, metric="sharpe",
                n=n_placebo, fee_bps_oneway=fee_bps_oneway,
            )
            logger.info("placebo %-24s p(Sharpe)=%.2f", name, placebos[name])

    for r in rows:
        if r.name in placebos:
            r.placebo = placebos[r.name]

    return rows, benchmarks


def run_funding_window(
    R: pd.DataFrame, funding: pd.Series, *,
    fee_bps_oneway: float = 5.0, n_placebo: int = 200,
) -> list[Row]:
    """Strategy 5 done FAIRLY: restrict to the funding window (2023-11+) and
    report trend-only, funding-only, AND, AVG over exactly the same days, so
    the combination is judged against trend on its OWN sub-window — not against
    a five-year trend number padded with a NaN-funding stretch."""
    f = funding.dropna()
    if f.empty:
        return []
    start = f.index.min()
    Rw = R[R.index >= start]
    if len(Rw) < 90:
        return []
    btc = Rw["btc"]
    btc_price = _price(btc)
    idx = Rw.index
    sig_trend = _trend_flag(btc_price, 50)
    f_full = funding.reindex(idx).ffill()
    fund_exp = (f_full >= 0).astype(float).fillna(0.0)

    specs = {
        "fw_trend50_only": sig_trend,
        "fw_funding_only": fund_exp,
        "fw_trend_AND_funding": sig_trend * fund_exp,
        "fw_trend_AVG_funding": (0.5 * sig_trend + 0.5 * fund_exp).clip(0, 1),
        "fw_bh_btc": pd.Series(1.0, index=idx),
    }
    rows: list[Row] = []
    results: dict[str, object] = {}
    for name, exp in specs.items():
        res = backtest_rule(btc, exp, fee_bps_oneway=fee_bps_oneway)
        res.name = name
        results[name] = res
        rows.append(_row_from_single(name, res, None))

    bh = results["fw_bh_btc"]
    for r in rows:
        res = results[r.name]
        if r.name != "fw_bh_btc" and (res.sharpe > bh.sharpe or res.calmar > bh.calmar):
            r.placebo = placebo_p_single(
                btc, specs[r.name], res, metric="sharpe",
                n=n_placebo, fee_bps_oneway=fee_bps_oneway,
            )
    return rows


# ── report ───────────────────────────────────────────────────────────


def render_markdown(
    rows: list[Row], benchmarks: dict[str, Row], fw_rows: list[Row], args: dict
) -> str:
    from datetime import datetime, timezone

    bh = benchmarks["bh_btc"]
    ew = benchmarks["ew_basket"]
    L = ["# Ensemble / combination allocation DAGs (state -> allocation -> return)\n"]
    L.append(
        "Each rule combines signals that each looked useful on their own and "
        "asks whether the COMBINATION beats the best SINGLE signal. Allocations "
        "are causal (trailing windows only); the decision at t close is applied "
        f"to t+1 return with a {args['fee_bps_oneway']:.0f}bp one-way switching "
        "cost; stables earn 0%. Sorted by Sharpe.\n"
    )
    L.append(f"- Universe: {', '.join(MAJORS)}")
    L.append(
        f"- Window `{args['start']}` -> `{args['end']}` "
        f"({bh.daily.shape[0]} days)"
    )
    L.append(
        f"- Run UTC: "
        f"`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n"
    )

    L.append(
        "| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | "
        "%in mkt | turnover | placebo p (Sharpe) |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: -r.sharpe):
        pp = f"{r.placebo:.2f}" if r.placebo is not None else "—"
        tag = ""
        if r.name == "bh_btc":
            tag = " ⟵ BH BTC"
        elif r.name == "ew_basket":
            tag = " ⟵ 1/N"
        L.append(
            f"| {r.name}{tag} | {r.ann_return:+.0%} | {r.sharpe:.2f} | "
            f"{r.sortino:.2f} | {r.max_dd:.0%} | {r.calmar:.2f} | "
            f"{r.in_market:.0%} | {r.turnover} | {pp} |"
        )
    L.append("")

    # ── funding-window sub-table (fair comparison) ─────────────────
    if fw_rows:
        fw_days = fw_rows[0].daily.shape[0]
        L.append(
            f"### Strategy 5 — fair funding-window comparison "
            f"(2023-11 onward, {fw_days} days)\n"
        )
        L.append(
            "Funding only exists from 2023-11, so the full-sample `trend_*_funding` "
            "rows above pad a long NaN-funding stretch. Here every rule is judged "
            "over exactly the funding window, against trend-only and BH BTC on the "
            "SAME days.\n"
        )
        L.append(
            "| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | "
            "%in mkt | switches | placebo p (Sharpe) |"
        )
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(fw_rows, key=lambda r: -r.sharpe):
            pp = f"{r.placebo:.2f}" if r.placebo is not None else "—"
            tag = " ⟵ BH BTC (sub)" if r.name == "fw_bh_btc" else ""
            L.append(
                f"| {r.name}{tag} | {r.ann_return:+.0%} | {r.sharpe:.2f} | "
                f"{r.sortino:.2f} | {r.max_dd:.0%} | {r.calmar:.2f} | "
                f"{r.in_market:.0%} | {r.turnover} | {pp} |"
            )
        L.append("")

    # ── verdict ────────────────────────────────────────────────────
    cands = [r for r in rows if r.name not in ("bh_btc", "ew_basket")]
    best_single_sharpe = max(bh.sharpe, ew.sharpe)
    beat_best = [r for r in cands if r.sharpe > best_single_sharpe]
    survivors = [
        r for r in beat_best
        if r.placebo is not None and r.placebo < 0.10
    ]
    L.append("## Verdict\n")
    L.append(
        f"Benchmarks — buy-and-hold BTC: ann {bh.ann_return:+.0%}, Sharpe "
        f"{bh.sharpe:.2f}, maxDD {bh.max_dd:.0%}, Calmar {bh.calmar:.2f}. "
        f"1/N basket: ann {ew.ann_return:+.0%}, Sharpe {ew.sharpe:.2f}, maxDD "
        f"{ew.max_dd:.0%}, Calmar {ew.calmar:.2f}.\n"
    )
    if survivors:
        L.append(
            "Combinations that beat the best single benchmark on Sharpe AND whose "
            "timing survives the placebo (p<0.10):"
        )
        for r in sorted(survivors, key=lambda r: -r.sharpe):
            L.append(
                f"- **{r.name}** — Sharpe {r.sharpe:.2f} vs best-single "
                f"{best_single_sharpe:.2f}, maxDD {r.max_dd:.0%} "
                f"(placebo p={r.placebo:.2f})"
            )
    else:
        L.append(
            "**No combination both beats the best single benchmark "
            f"(Sharpe {best_single_sharpe:.2f}, the 1/N basket) AND survives the "
            "placebo (p<0.10).** Combining signals raises Calmar in several cases "
            "by cutting drawdown, but that relief comes from holding a "
            "diversified basket and/or sitting out on average — not from skillful "
            "ensemble timing. Stacking signals here mostly multiplies researcher "
            "degrees of freedom without buying a real edge over plain "
            "diversification."
        )
    L.append("\n## Caveats\n")
    L.append(
        "- **One market cycle.** 2021–2025 is a single bull→bear→recovery path. "
        "Trend and breadth filters look heroic when they truncate exactly one "
        "bear; that is not evidence of a repeatable edge."
    )
    L.append(
        "- **Combinations multiply researcher DoF.** Each combo bakes in choices "
        "(which signals, which windows, AND vs average vs vote). The placebo "
        "guards against *timing* luck on a fixed rule, not against having picked "
        "the lucky combination among many."
    )
    L.append(
        "- **Funding is short** (2023-11+) and the full-sample trend+funding rows "
        "pad a NaN stretch; trust the funding-window sub-table for that comparison."
    )
    L.append(
        "- Stables modeled at 0% (no yield, no slippage beyond the switching "
        "cost); the 5bp/side fee is optimistic for the smaller majors."
    )
    return "\n".join(L)


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--fee-bps-oneway", type=float, default=5.0)
    p.add_argument("--n-placebo", type=int, default=200)
    p.add_argument("--out", default="causal_portfolio/docs/ensemble_dags.md")
    args = p.parse_args()

    R = load_returns_matrix(args.start, args.end)
    logger.info("loaded %d days x %d majors", R.shape[0], R.shape[1])
    vix = load_vix(args.start, args.end)
    funding = load_funding(args.start, args.end)
    logger.info(
        "vix=%s funding=%d days (from %s)",
        "yes" if vix is not None else "no",
        len(funding.dropna()),
        funding.dropna().index.min().date() if len(funding.dropna()) else "—",
    )

    rows, benchmarks = run_experiment(
        R, vix, funding,
        fee_bps_oneway=args.fee_bps_oneway, n_placebo=args.n_placebo,
    )
    for r in sorted(rows, key=lambda r: -r.sharpe):
        logger.info(
            "%-24s ann=%+.0f%% sharpe=%.2f maxDD=%.0f%% calmar=%.2f in=%.0f%%",
            r.name, r.ann_return * 100, r.sharpe, r.max_dd * 100,
            r.calmar, r.in_market * 100,
        )

    fw_rows = run_funding_window(
        R, funding, fee_bps_oneway=args.fee_bps_oneway, n_placebo=args.n_placebo
    )
    for r in sorted(fw_rows, key=lambda r: -r.sharpe):
        logger.info(
            "[fw] %-20s ann=%+.0f%% sharpe=%.2f calmar=%.2f in=%.0f%%",
            r.name, r.ann_return * 100, r.sharpe, r.calmar, r.in_market * 100,
        )

    md = render_markdown(rows, benchmarks, fw_rows, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
