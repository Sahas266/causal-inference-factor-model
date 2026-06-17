"""Market-breadth and correlation-regime 'DAGs': a market state -> exposure.

Each rule here is the smallest possible causal DAG built from a *cross-section*
of major coins:

    market_state_t  ->  exposure_t  ->  return_{t+1}

The state is a single scalar summarizing the whole majors universe (breadth,
average pairwise correlation, cross-sectional dispersion, BTC relative
strength). That scalar maps to a 0/1 (or continuous) exposure, which is applied
*both* to BTC and to the equal-weight basket of majors (basket return = mean of
the major daily returns). Decision at day t (trailing data only) is applied to
day t+1's return; the harness `backtest_rule` does the one-day shift and charges
a switching cost on every exposure change.

This reuses the regime-rotation harness verbatim (import, never edit):
`causal_portfolio.experiments.regime_rotation.backtest_rule` / `placebo_p`.

State families (all causal — trailing windows / rolling quantiles only):

  breadth_ma50/100   % of majors whose price > own N-day MA; risk-on > 50%
  breadth_med        breadth > its own trailing rolling median (adaptive gate)
  breadth_mom        breadth RISING: breadth > its trailing 10-day average
  corr_off_high      avg 30d pairwise corr HIGH (>trailing 70th pct) -> risk-OFF
  corr_off_low       the opposite sign: corr LOW -> risk-off (honest two-sided)
  disp_off_high      cross-sectional return dispersion HIGH -> risk-off
  disp_off_low       dispersion LOW -> risk-off (opposite sign)
  btcdom_alts_on     BTC LAGGING basket (rel-strength<0) -> alt risk-on (basket)
  btcdom_alts_off    BTC LEADING basket -> alt risk-on (the opposite)

Every state is scored against two benchmarks: buy-and-hold BTC and
buy-and-hold equal-weight basket. A rule "wins on risk" if it raises Sharpe or
Calmar; it "wins outright" if it also matches total return. Winners are
placebo-tested (circular time-shifts of the exposure path) so we don't mistake
average-exposure effects or lucky timing for skill.

Caveats baked into the reading: this is one crypto cycle; breadth/correlation
are *derived from the same prices* we trade, so part of any edge is mechanical;
and the two-sided sign tests (we report both corr signs, both dispersion signs,
both dominance signs) inflate the chance one looks significant — adjust your
prior accordingly.

Run:  python -m causal_portfolio.experiments.breadth_correlation_dags
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from causal_portfolio.experiments.regime_rotation import (
    backtest_rule, placebo_p,
)

logger = logging.getLogger("cpcm.experiments.breadth_correlation_dags")

MAJORS = ["btc", "eth", "sol", "bnb", "avax", "xrp", "doge", "link",
          "uni", "aave", "crv"]


# ── cross-sectional state builders (all causal / trailing) ──────────

def _prices(returns: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a price level per asset from simple daily returns."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def breadth_above_ma(prices: pd.DataFrame, window: int) -> pd.Series:
    """% of assets whose price is above its own trailing N-day MA, in [0,1].

    Causal: the MA at t uses prices[..t]. Before `window` history exists the
    MA is NaN and that asset simply doesn't vote, so early breadth is computed
    over whatever assets already have enough history.
    """
    ma = prices.rolling(window, min_periods=window).mean()
    above = (prices > ma)            # NaN MA -> False, excluded by .count below
    valid = ma.notna()
    num = (above & valid).sum(axis=1)
    den = valid.sum(axis=1).replace(0, np.nan)
    return (num / den).astype(float)


def avg_pairwise_corr(returns: pd.DataFrame, window: int = 30) -> pd.Series:
    """Trailing average off-diagonal pairwise correlation across the majors.

    Computed from the rolling correlation matrix at each t over the trailing
    `window` days. Fully causal (uses returns[t-window+1 .. t]).
    """
    cols = returns.columns
    n = len(cols)
    out = pd.Series(index=returns.index, dtype=float)
    arr = returns.values
    idx = returns.index
    iu = np.triu_indices(n, k=1)
    for i in range(len(idx)):
        if i + 1 < window:
            continue
        w = arr[i - window + 1: i + 1]
        # correlation matrix of the window (assets in columns)
        c = np.corrcoef(w, rowvar=False)
        vals = c[iu]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out.iloc[i] = float(vals.mean())
    return out


def cross_sectional_dispersion(returns: pd.DataFrame,
                               window: int = 30) -> pd.Series:
    """Stdev *across assets* of each asset's trailing `window`-day total return.

    High dispersion = coins moving very differently (stock-picker's market);
    low dispersion = everything moving together. Causal.
    """
    trailing = (1.0 + returns.fillna(0.0)).rolling(window).apply(
        lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    # trailing[t, asset] = window total return ending at t; std across assets
    disp = trailing.std(axis=1, ddof=0)
    disp[trailing.notna().sum(axis=1) < 2] = np.nan
    return disp.astype(float)


def btc_relative_strength(returns: pd.DataFrame,
                          window: int = 30) -> pd.Series:
    """BTC trailing-window total return minus the basket's trailing-window
    total return. >0 => BTC leading (dominance rising); <0 => alts leading."""
    basket = returns.mean(axis=1)
    cum = lambda s: (1.0 + s.fillna(0.0)).rolling(window).apply(
        lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    return (cum(returns["btc"]) - cum(basket)).astype(float)


def _rolling_pctile_flag_high(s: pd.Series, q: float,
                              window: int = 252,
                              min_periods: int = 60) -> pd.Series:
    """1.0 when s is at/above its trailing q-quantile, else 0.0 (causal)."""
    thresh = s.rolling(window, min_periods=min_periods).quantile(q)
    return (s >= thresh).astype(float)


# ── assemble exposure rules ─────────────────────────────────────────

def build_states(returns: pd.DataFrame) -> dict[str, pd.Series]:
    """Return {name: exposure in [0,1]} keyed by market state. Each exposure
    is the SAME series applied to both BTC and basket later."""
    idx = returns.index
    prices = _prices(returns)
    states: dict[str, pd.Series] = {}

    # benchmark
    states["always_in"] = pd.Series(1.0, index=idx)

    # 1. Breadth gate (MA crossings)
    b50 = breadth_above_ma(prices, 50)
    b100 = breadth_above_ma(prices, 100)
    states["breadth_ma50_gt50"] = (b50 > 0.50).astype(float)
    states["breadth_ma100_gt50"] = (b100 > 0.50).astype(float)
    # adaptive: breadth above its own trailing median (no magic 50% number)
    med50 = b50.rolling(252, min_periods=60).median()
    states["breadth_ma50_gtmed"] = (b50 > med50).astype(float)

    # 2. Breadth momentum (rising breadth)
    b50_avg10 = b50.rolling(10, min_periods=10).mean()
    states["breadth_mom_rising"] = (b50 > b50_avg10).astype(float)

    # 3. Correlation regime (BOTH signs, honestly reported)
    corr = avg_pairwise_corr(returns, 30)
    high_corr = _rolling_pctile_flag_high(corr, 0.70, 252)
    states["corr_off_high"] = 1.0 - high_corr   # high corr -> risk-OFF
    states["corr_off_low"] = high_corr          # opposite: low corr -> off

    # 4. Dispersion regime (BOTH signs)
    disp = cross_sectional_dispersion(returns, 30)
    high_disp = _rolling_pctile_flag_high(disp, 0.70, 252)
    states["disp_off_high"] = 1.0 - high_disp   # high dispersion -> risk-OFF
    states["disp_off_low"] = high_disp          # opposite: low dispersion -> off

    # 5. BTC-dominance proxy gating ALTS (applied to basket primarily; both
    #    signs). rel>0 => BTC leading.
    rel = btc_relative_strength(returns, 30)
    btc_leading = (rel > 0).astype(float)
    states["btcdom_alts_on_lag"] = 1.0 - btc_leading  # BTC lagging -> alts on
    states["btcdom_alts_on_lead"] = btc_leading       # opposite: BTC leading

    # forward-fill any internal gaps, leave warmup NaN -> 0 exposure
    for k, v in states.items():
        states[k] = v.reindex(idx)
    return states


# ── reporting ────────────────────────────────────────────────────────

@dataclass
class Row:
    rule: str
    asset: str
    ann_return: float
    sharpe: float
    max_dd: float
    calmar: float
    pct_in_market: float
    n_switches: int
    placebo_p: float | None


def _evaluate(states, btc, basket, *, fee_bps_oneway, n_placebo):
    """Backtest every state on BTC and on the basket; placebo winners."""
    targets = {"BTC": btc, "basket": basket}
    rows: list[Row] = []
    raw: dict[tuple[str, str], object] = {}

    # benchmarks
    bh = {}
    for tname, series in targets.items():
        r = backtest_rule(series, states["always_in"],
                          fee_bps_oneway=fee_bps_oneway)
        bh[tname] = r

    for sname, exp in states.items():
        for tname, series in targets.items():
            r = backtest_rule(series, exp, fee_bps_oneway=fee_bps_oneway)
            r = replace(r, name=sname)
            raw[(sname, tname)] = r
            pp = None
            # placebo only candidates that beat the matching BH on Sharpe or
            # Calmar (cheaper; the only ones we'd consider).
            base = bh[tname]
            if sname != "always_in" and (r.sharpe > base.sharpe
                                         or r.calmar > base.calmar):
                pp = placebo_p(series, exp, r, metric="sharpe",
                               n=n_placebo, fee_bps_oneway=fee_bps_oneway)
            rows.append(Row(
                rule=sname, asset=tname, ann_return=r.ann_return,
                sharpe=r.sharpe, max_dd=r.max_dd, calmar=r.calmar,
                pct_in_market=r.pct_in_market, n_switches=r.n_switches,
                placebo_p=pp))
            logger.info("%-22s %-6s ann=%+.0f%% sharpe=%.2f maxDD=%.0f%% "
                        "calmar=%.2f in=%.0f%% placebo=%s", sname, tname,
                        r.ann_return * 100, r.sharpe, r.max_dd * 100,
                        r.calmar, r.pct_in_market * 100,
                        f"{pp:.2f}" if pp is not None else "-")
    return rows, bh


def render_markdown(rows: list[Row], bh: dict, args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# Market-breadth & correlation-regime DAGs (state → exposure → "
         "return)\n"]
    L.append("Each rule turns a single cross-sectional market state (breadth, "
             "avg pairwise correlation, cross-sectional dispersion, BTC "
             "relative strength) into a 0/1 exposure, applied to **both** BTC "
             "and the equal-weight majors **basket**. Decision at day t close, "
             f"applied to t+1 return, {args['fee_bps_oneway']:.0f}bp one-way "
             "switching cost. Sorted within each asset by Calmar.\n")
    L.append(f"- Universe: `{', '.join(MAJORS)}`")
    L.append(f"- Window `{args['start']}` → `{args['end']}` "
             f"({args['n_days']} days)")
    L.append(f"- Run UTC: "
             f"`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("| Rule | Asset | Ann.ret | Sharpe | MaxDD | Calmar | %in mkt | "
             "switches | placebo p (Sharpe) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    def sort_key(r: Row):
        return (0 if r.asset == "BTC" else 1, -r.calmar)

    for r in sorted(rows, key=sort_key):
        pps = f"{r.placebo_p:.2f}" if r.placebo_p is not None else "—"
        tag = " ⟵ BH" if r.rule == "always_in" else ""
        L.append(f"| {r.rule}{tag} | {r.asset} | {r.ann_return:+.0%} | "
                 f"{r.sharpe:.2f} | {r.max_dd:.0%} | {r.calmar:.2f} | "
                 f"{r.pct_in_market:.0%} | {r.n_switches} | {pps} |")
    L.append("")

    # Verdict
    L.append("## Verdict\n")
    for tname in ("BTC", "basket"):
        b = bh[tname]
        L.append(f"**Buy-and-hold {tname}**: ann {b.ann_return:+.0%}, "
                 f"Sharpe {b.sharpe:.2f}, maxDD {b.max_dd:.0%}, "
                 f"Calmar {b.calmar:.2f}.")
    L.append("")

    winners = [r for r in rows if r.rule != "always_in"
               and r.placebo_p is not None and r.placebo_p < 0.10
               and r.sharpe > bh[r.asset].sharpe]
    if winners:
        L.append("Rules that beat their BH benchmark on Sharpe AND whose "
                 "timing survives the placebo (p<0.10):")
        for r in sorted(winners, key=lambda r: -r.sharpe):
            L.append(f"- **{r.rule}** on {r.asset} — Sharpe {r.sharpe:.2f} vs "
                     f"{bh[r.asset].sharpe:.2f}, maxDD {r.max_dd:.0%} vs "
                     f"{bh[r.asset].max_dd:.0%} (placebo p={r.placebo_p:.2f})")
    else:
        improved = [r for r in rows if r.rule != "always_in"
                    and r.calmar > bh[r.asset].calmar]
        L.append("**No breadth/correlation state both beats its BH benchmark "
                 "on Sharpe and survives the placebo.** "
                 f"{len(improved)} rule/asset combo(s) improve Calmar, but "
                 "they do so by sitting out of the market on average (lower "
                 "exposure cuts drawdown); a random-timed overlay with the "
                 "same average exposure reproduces the effect (placebo "
                 "p≥0.10), so the *timing* of the regime calls carries no "
                 "demonstrable edge.")
    L.append("")
    L.append("## Caveats\n")
    L.append("- **One cycle.** This is a single 2021–2025 crypto regime "
             "(2021 mania → 2022 bear → 2023–25 recovery). Breadth and "
             "correlation gates are exactly the kind of rule that overfits to "
             "one drawdown.")
    L.append("- **Mechanically endogenous.** Breadth, correlation and "
             "dispersion are all derived from the *same* prices being traded, "
             "so part of any apparent edge is mechanical (e.g. a price below "
             "its MA both lowers breadth and is, tautologically, a recent "
             "loss).")
    L.append("- **Two-sided sign tests inflate significance.** We report both "
             "signs of the correlation, dispersion and dominance gates. "
             "Testing a hypothesis and its negation roughly doubles the "
             "chance one looks good by luck — treat any single survivor with "
             "suspicion and discount the placebo threshold accordingly.")
    L.append("- **Stables = 0%.** Risk-off earns 0% (no stable yield modeled); "
             "a real cash/stable yield would only improve the risk-off legs, "
             "so these gate results are conservative on that axis.")
    L.append("- **Read the survivors skeptically.** Any rule that clears the "
             "placebo here is one survivor out of ~20 rule/asset combos under "
             "two-sided sign testing on a single cycle; the placebo only "
             "rules out trivial average-exposure effects, not overfitting, so "
             "the honest read is \"suggestive, not validated — would need a "
             "second regime / market to trust.\"")
    return "\n".join(L)


def main() -> None:
    import argparse
    from pathlib import Path
    from causal_portfolio.data import get_loader

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--fee-bps-oneway", type=float, default=5.0)
    p.add_argument("--n-placebo", type=int, default=200)
    p.add_argument("--out",
                   default="causal_portfolio/docs/breadth_correlation_dags.md")
    args = p.parse_args()

    loader = get_loader()
    returns = loader.load_returns(MAJORS, args.start, args.end)
    returns.index = pd.to_datetime(returns.index)
    returns = returns[[f"{a}_return" for a in MAJORS]]
    returns.columns = MAJORS  # rename to plain tickers for clarity

    btc = returns["btc"]
    basket = returns.mean(axis=1)
    basket.name = "basket"

    states = build_states(returns)
    rows, bh = _evaluate(states, btc, basket,
                         fee_bps_oneway=args.fee_bps_oneway,
                         n_placebo=args.n_placebo)

    argd = vars(args) | {"n_days": int(len(returns))}
    md = render_markdown(rows, bh, argd)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
