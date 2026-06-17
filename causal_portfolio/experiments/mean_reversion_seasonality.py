"""Simple mean-reversion and calendar/seasonality rules on BTC (and the basket).

The cousin of `regime_rotation.py`: instead of risk-on/off on a macro state,
these are textbook *mean-reversion* and *calendar* rules — the kind a retail
"buy the dip" trader reaches for. Each produces a daily BTC exposure in [0,1];
risk-off goes to stables (0% — conservative, a real stable yield only helps).
All signals are CAUSAL (trailing windows / rolling stats only); the decision
through day t is applied to t+1's return (`backtest_rule` shifts exposure by 1),
with a switching cost on every exposure change.

These churn far more than the regime rules, so we report net Sharpe at 5/10/20
bp one-way costs, and placebo-test (circular time-shift) every rule that beats
buy-and-hold BTC on Sharpe or Calmar — the only honest way to tell a real
timing edge from "happens to be in the market less / at the right average level."

Rules (BTC unless noted):
  bh                buy-and-hold BTC (exposure 1; the benchmark)
  rsi_reversal      long when RSI(14)<30, stables when RSI>70, else hold prior
  rsi_dipbuyer      long only when RSI(14)<30, else stables (long-only variant)
  st_reversal       after a < trailing-2σ down day, hold long next day; after a
                    > +2σ up day, go to stables; else hold prior state
  bollinger         long when price < (mean - 2·std) lower band, exit to stables
                    when price >= the rolling mean
  dow_seasonality   trailing-1y mean return by weekday; hold BTC only on weekdays
                    whose trailing-mean return is positive
  turn_of_month     hold BTC only on the last trading day of the month through
                    the first ~3 of the next, stables otherwise

The single best mean-reversion rule is also applied to the equal-weight basket
to check it isn't BTC-specific.

Run:  python -m causal_portfolio.experiments.mean_reversion_seasonality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reuse the harness verbatim — do NOT reimplement the backtest / placebo.
from causal_portfolio.experiments.regime_rotation import (
    RotResult, backtest_rule, placebo_p,
)

logger = logging.getLogger("cpcm.experiments.mean_reversion_seasonality")


# ── helpers (all causal) ─────────────────────────────────────────────

def _price(returns: pd.Series) -> pd.Series:
    """Synthetic price level from simple returns (starts at 1.0)."""
    return (1 + returns.fillna(0)).cumprod()


def _rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI(window) on the synthetic price, computed causally.

    Uses an exponential (Wilder) average of up/down moves; the value at t uses
    only data through t. RSI in [0, 100]: <30 oversold, >70 overbought.
    """
    price = _price(returns)
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0 (only gains), RSI saturates at 100.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def _state_from_entry_exit(
    idx: pd.Index, entry: pd.Series, exit_: pd.Series,
) -> pd.Series:
    """Carry a binary position forward: 1 on `entry`, 0 on `exit_`, else hold.

    Both `entry` and `exit_` are boolean Series aligned to `idx`. If both fire
    on the same day, exit wins (conservative). Position starts flat (0). Fully
    causal: the state at t depends only on entry/exit events up to and at t.
    """
    e = entry.reindex(idx).fillna(False).to_numpy(dtype=bool)
    x = exit_.reindex(idx).fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(idx), dtype=float)
    pos = 0.0
    for i in range(len(idx)):
        if x[i]:
            pos = 0.0
        elif e[i]:
            pos = 1.0
        out[i] = pos
    return pd.Series(out, index=idx)


# ── strategy exposure builders ───────────────────────────────────────

def rule_rsi_reversal(returns: pd.Series, *, window: int = 14,
                      low: float = 30.0, high: float = 70.0) -> pd.Series:
    """Long when RSI<low (oversold), stables when RSI>high (overbought),
    hold prior state in between."""
    rsi = _rsi(returns, window)
    entry = rsi < low
    exit_ = rsi > high
    return _state_from_entry_exit(returns.index, entry, exit_)


def rule_rsi_dipbuyer(returns: pd.Series, *, window: int = 14,
                      low: float = 30.0) -> pd.Series:
    """Long-only dip buyer: exposure 1 only while RSI<low, else stables."""
    rsi = _rsi(returns, window)
    return (rsi < low).astype(float)


def rule_short_term_reversal(returns: pd.Series, *, window: int = 20,
                             k: float = 2.0) -> pd.Series:
    """After a down day beyond trailing -kσ (capitulation), hold long next day;
    after an up day beyond +kσ (blowoff), go to stables; else hold prior state.

    The trailing σ is a rolling std of returns through day t-1 (shifted), so the
    threshold the day-t move is judged against uses no contemporaneous info."""
    sigma = returns.rolling(window, min_periods=window).std().shift(1)
    entry = returns < (-k * sigma)
    exit_ = returns > (k * sigma)
    return _state_from_entry_exit(returns.index, entry, exit_)


def rule_bollinger(returns: pd.Series, *, window: int = 20,
                   k: float = 2.0) -> pd.Series:
    """Long when price < lower Bollinger band (mean - k·std of price), exit to
    stables when price >= the rolling mean. Bands use trailing price stats."""
    price = _price(returns)
    ma = price.rolling(window, min_periods=window).mean()
    sd = price.rolling(window, min_periods=window).std()
    lower = ma - k * sd
    entry = price < lower
    exit_ = price >= ma
    return _state_from_entry_exit(returns.index, entry, exit_)


def rule_dow_seasonality(returns: pd.Series, *, lookback: int = 365) -> pd.Series:
    """Hold BTC only on weekdays whose TRAILING `lookback`-day mean return is
    positive. The seasonal effect is re-estimated each day on a trailing window
    (never the full sample), so it's causal.

    Returns the exposure Series; the favored-weekday breakdown is available via
    `dow_favored_summary`."""
    idx = returns.index
    dow = idx.dayofweek  # 0=Mon .. 6=Sun
    # Trailing per-weekday mean: for each day t, mean of same-weekday returns in
    # (t-lookback, t-1]. We compute it groupwise with a shifted rolling mean so
    # day t never sees its own return.
    r = returns.fillna(0.0)
    exposure = np.zeros(len(idx), dtype=float)
    s = pd.Series(r.to_numpy(), index=np.arange(len(idx)))
    dow_arr = np.asarray(dow)
    # Precompute, per weekday, a trailing mean indexed back onto the timeline.
    # Window in *calendar* days; same weekday repeats every 7 days, so a 365d
    # lookback ~= 52 same-weekday observations.
    approx_obs = max(4, lookback // 7)
    trailing_mean = pd.Series(np.nan, index=np.arange(len(idx)))
    for wd in range(7):
        pos = np.where(dow_arr == wd)[0]
        if len(pos) == 0:
            continue
        vals = s.iloc[pos]
        # Trailing mean of this weekday's returns, EXCLUDING the current obs.
        tm = vals.shift(1).rolling(approx_obs, min_periods=4).mean()
        trailing_mean.iloc[pos] = tm.to_numpy()
    exposure = (trailing_mean.to_numpy() > 0).astype(float)
    return pd.Series(exposure, index=idx)


def dow_favored_summary(returns: pd.Series, *, lookback: int = 365) -> pd.Series:
    """Fraction of days each weekday ended up favored (in-market) by the
    trailing-mean rule. Reported in the markdown so we can see WHICH weekdays
    the rule picks."""
    idx = returns.index
    exp = rule_dow_seasonality(returns, lookback=lookback)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    out = {}
    for wd in range(7):
        mask = idx.dayofweek == wd
        if mask.sum() == 0:
            out[names[wd]] = float("nan")
        else:
            out[names[wd]] = float(exp[mask].mean())
    return pd.Series(out)


def rule_turn_of_month(returns: pd.Series, *, days_before: int = 1,
                       days_after: int = 3) -> pd.Series:
    """Hold BTC across the turn of the month: the last `days_before` trading day(s)
    of a month through the first `days_after` of the next; stables otherwise.

    Pure calendar — no fitting at all, so trivially causal."""
    idx = returns.index
    exposure = pd.Series(0.0, index=idx)
    period = idx.to_period("M")
    is_last = np.zeros(len(idx), dtype=bool)
    is_first = np.zeros(len(idx), dtype=bool)
    # Position within each month.
    for _, grp in pd.Series(np.arange(len(idx)), index=period).groupby(level=0):
        locs = grp.to_numpy()
        # last `days_before` of this month
        for j in locs[-days_before:]:
            is_last[j] = True
        # first `days_after` of this month
        for j in locs[:days_after]:
            is_first[j] = True
    exposure[is_last | is_first] = 1.0
    return exposure


# ── orchestration ────────────────────────────────────────────────────

def build_rules(returns: pd.Series) -> dict[str, pd.Series]:
    """All BTC mean-reversion + seasonality exposures, keyed by name."""
    idx = returns.index
    rules: dict[str, pd.Series] = {}
    rules["bh"] = pd.Series(1.0, index=idx)
    rules["rsi_reversal"] = rule_rsi_reversal(returns)
    rules["rsi_dipbuyer"] = rule_rsi_dipbuyer(returns)
    rules["st_reversal"] = rule_short_term_reversal(returns)
    rules["bollinger"] = rule_bollinger(returns)
    rules["dow_seasonality"] = rule_dow_seasonality(returns)
    rules["turn_of_month"] = rule_turn_of_month(returns)
    return rules


# Net Sharpe at multiple cost levels, for the cost-sensitivity columns.
COST_LEVELS = (5.0, 10.0, 20.0)


def evaluate(
    asset_returns: pd.Series, rules: dict[str, pd.Series], *,
    n_placebo: int = 200, fee_bps_oneway: float = 5.0,
) -> tuple[list[RotResult], dict[str, dict], dict[str, dict]]:
    """Backtest every rule at the headline fee, compute net Sharpe at each cost
    level, and placebo-test the rules that beat BH on Sharpe or Calmar.

    Returns (results_at_headline_fee, net_sharpe_by_rule, placebos_by_rule)."""
    results: list[RotResult] = []
    net_sharpe: dict[str, dict] = {}
    for name, exp in rules.items():
        r = backtest_rule(asset_returns, exp, fee_bps_oneway=fee_bps_oneway)
        r.name = name
        results.append(r)
        net_sharpe[name] = {
            bps: backtest_rule(asset_returns, exp, fee_bps_oneway=bps).sharpe
            for bps in COST_LEVELS
        }
        logger.info(
            "%-16s ann=%+.0f%% sharpe=%.2f maxDD=%.0f%% calmar=%.2f in=%.0f%% "
            "switches=%d", name, r.ann_return * 100, r.sharpe, r.max_dd * 100,
            r.calmar, r.pct_in_market * 100, r.n_switches)

    bh = next(r for r in results if r.name == "bh")
    placebos: dict[str, dict] = {}
    for r in results:
        if r.name != "bh" and (r.sharpe > bh.sharpe or r.calmar > bh.calmar):
            p = placebo_p(asset_returns, rules[r.name], r, metric="sharpe",
                          n=n_placebo, fee_bps_oneway=fee_bps_oneway)
            placebos[r.name] = {"sharpe": p}
            logger.info("placebo %-16s p(Sharpe)=%.2f", r.name, p)
    return results, net_sharpe, placebos


# ── report ───────────────────────────────────────────────────────────

_LABELS = {
    "bh": "buy-and-hold BTC",
    "rsi_reversal": "RSI(14) reversal (long <30, exit >70)",
    "rsi_dipbuyer": "RSI(14) dip-buyer (long only <30)",
    "st_reversal": "short-term reversal (buy <-2σ, sell >+2σ)",
    "bollinger": "Bollinger reversion (buy < lower band, exit at mean)",
    "dow_seasonality": "day-of-week seasonality (trailing-1y)",
    "turn_of_month": "turn-of-month (last 1 + first 3 days)",
}


def render_markdown(
    btc_results: list[RotResult], btc_net: dict[str, dict],
    btc_placebos: dict[str, dict], dow_favored: pd.Series,
    basket_block: dict | None, args: dict,
) -> str:
    from datetime import datetime, timezone

    bh = next(r for r in btc_results if r.name == "bh")
    L = ["# Mean-reversion & calendar/seasonality rules on BTC\n"]
    L.append(
        "Textbook 'buy-the-dip' mean-reversion plus calendar/seasonality rules, "
        "each as a daily BTC exposure in [0,1] (stables = 0% when out). Signals "
        "are causal (trailing windows only); decision at t close applied to t+1, "
        "switching cost charged on every exposure change. Seasonality is "
        "estimated on a TRAILING 1-year window, never the full sample. Sorted by "
        "Calmar.\n")
    L.append(f"- Asset: BTC | window `{args['start']}` -> `{args['end']}` "
             f"({bh.daily.shape[0]} days)")
    L.append(f"- Headline switching cost: `{args['fee_bps_oneway']:.0f}` bp "
             "one-way; net-Sharpe columns show 5 / 10 / 20 bp sensitivity")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("| Rule | Ann.ret | Sharpe | MaxDD | Calmar | %in mkt | switches | "
             "placebo p | net Sharpe 5bp | 10bp | 20bp |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(btc_results, key=lambda r: -r.calmar):
        pp = btc_placebos.get(r.name, {}).get("sharpe")
        pps = f"{pp:.2f}" if pp is not None else "—"
        ns = btc_net.get(r.name, {})
        tag = " <- BH" if r.name == "bh" else ""
        L.append(
            f"| {r.name}{tag} | {r.ann_return:+.0%} | {r.sharpe:.2f} | "
            f"{r.max_dd:.0%} | {r.calmar:.2f} | {r.pct_in_market:.0%} | "
            f"{r.n_switches} | {pps} | {ns.get(5.0, float('nan')):.2f} | "
            f"{ns.get(10.0, float('nan')):.2f} | {ns.get(20.0, float('nan')):.2f} |")
    L.append("")

    # Day-of-week breakdown.
    L.append("## Day-of-week: which weekdays the rule favors\n")
    L.append("Fraction of each weekday the trailing-1y rule held BTC "
             "(higher = that weekday's trailing mean was usually positive):\n")
    L.append("| " + " | ".join(dow_favored.index) + " |")
    L.append("|" + "---:|" * len(dow_favored))
    L.append("| " + " | ".join(f"{v:.0%}" for v in dow_favored.values) + " |")
    favored = [d for d, v in dow_favored.items() if v >= 0.5]
    L.append(f"\nMost-favored weekdays (held >=50% of the time): "
             f"**{', '.join(favored) if favored else 'none'}**.\n")

    # Verdict.
    def survives(r: RotResult) -> bool:
        p = btc_placebos.get(r.name, {}).get("sharpe")
        net10 = btc_net.get(r.name, {}).get(10.0, -9)
        return (r.name != "bh" and r.sharpe > bh.sharpe and p is not None
                and p < 0.10 and net10 > bh.sharpe)

    winners = [r for r in btc_results if survives(r)]
    L.append("## Verdict\n")
    L.append(f"Buy-and-hold BTC: ann {bh.ann_return:+.0%}, Sharpe {bh.sharpe:.2f}, "
             f"maxDD {bh.max_dd:.0%}, Calmar {bh.calmar:.2f}.\n")
    if winners:
        L.append("Rules that beat BH on Sharpe, survive the placebo (p<0.10), "
                 "AND still beat BH net of realistic (>=10bp) costs:")
        for r in sorted(winners, key=lambda r: -r.sharpe):
            L.append(f"- **{r.name}** — Sharpe {r.sharpe:.2f} vs {bh.sharpe:.2f}, "
                     f"net@10bp {btc_net[r.name][10.0]:.2f}, "
                     f"placebo p={btc_placebos[r.name]['sharpe']:.2f}")
    else:
        better_sharpe = [r for r in btc_results
                         if r.name != "bh" and r.sharpe > bh.sharpe]
        if not better_sharpe:
            L.append("**No rule clears even the first bar.** Not one beats "
                     f"buy-and-hold BTC on gross Sharpe ({bh.sharpe:.2f}) — so "
                     "none warranted placebo testing. Every rule de-risks "
                     "(lower maxDD via lower average exposure) but at the cost "
                     "of giving up so much upside that risk-adjusted return "
                     "falls below BH. Mean-reversion does not time BTC's regime "
                     "here.")
        else:
            L.append("**No rule clears all three bars** (beat BH on Sharpe + "
                     "survive the placebo + still beat BH net of >=10bp costs). "
                     f"{len(better_sharpe)} rule(s) beat BH on gross Sharpe, but "
                     "they either fail the placebo (the *timing* carries no "
                     "information — a random-timed overlay with the same average "
                     "exposure does as well) or their edge is eaten by switching "
                     "costs once churn is priced realistically.")
    L.append("")

    # Basket robustness.
    if basket_block is not None:
        bb = basket_block
        L.append("## Robustness: best mean-reversion rule on the EW basket\n")
        L.append(f"Applied **{bb['rule']}** (best BTC mean-reversion rule by "
                 "Calmar) to the equal-weight basket of "
                 f"{bb['n_assets']} coins, to check it isn't BTC-specific.\n")
        L.append("| Rule | Ann.ret | Sharpe | MaxDD | Calmar | %in mkt | "
                 "switches | placebo p | net 5/10/20bp |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label, r in (("basket BH", bb["bh"]), (f"basket {bb['rule']}", bb["rule_res"])):
            pp = bb["placebo"] if "basket " + bb["rule"] in label or label.endswith(bb["rule"]) else None
            pps = f"{pp:.2f}" if (pp is not None and not label.startswith("basket BH")) else "—"
            ns = bb["net"] if not label.startswith("basket BH") else bb["bh_net"]
            L.append(
                f"| {label} | {r.ann_return:+.0%} | {r.sharpe:.2f} | "
                f"{r.max_dd:.0%} | {r.calmar:.2f} | {r.pct_in_market:.0%} | "
                f"{r.n_switches} | {pps} | "
                f"{ns[5.0]:.2f}/{ns[10.0]:.2f}/{ns[20.0]:.2f} |")
        verdict = ("the rule's behavior carries over to the basket"
                   if bb["rule_res"].sharpe > bb["bh"].sharpe
                   else "the rule does NOT beat basket BH either — not a "
                        "BTC-specific fluke, just a weak edge everywhere")
        L.append(f"\nResult: {verdict}.\n")

    # Caveats.
    L.append("## Caveats\n")
    L.append("- **These churn.** Mean-reversion rules flip in/out far more than "
             "the regime overlays; every switch pays the spread, so the 10/20bp "
             "columns matter more here than anywhere else in this repo.")
    L.append("- **Mean-reversion is fragile and regime-dependent.** 'Buy the dip' "
             "works in choppy/ranging markets and gets run over in trends "
             "(crashes keep crashing, melt-ups keep melting up). A single sample "
             "can flatter or bury it depending on which regime dominated.")
    L.append("- **One market cycle.** 2021-2025 is essentially one bull-bear-bull "
             "crypto cycle; weekday/turn-of-month effects estimated on it are "
             "thin and unstable, and any apparent seasonality may not persist.")
    L.append("- **Stables = 0%.** Out-of-market days earn nothing here; a real "
             "stable yield would lift every rule's return modestly but not change "
             "the risk-adjusted ranking vs BH.")
    L.append("- **Placebo = circular time-shift.** It asks whether the *timing* "
             "beats a random-timed overlay with the same average exposure; a high "
             "p means the rule's only 'edge' is being in the market a certain "
             "fraction of the time, not skill.")
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
                   default="causal_portfolio/docs/mean_reversion_seasonality.md")
    args = p.parse_args()

    basket_assets = ["btc", "eth", "sol", "bnb", "avax",
                     "xrp", "doge", "link", "uni", "aave"]
    loader = get_loader()
    returns = loader.load_returns(basket_assets, args.start, args.end)
    returns.index = pd.to_datetime(returns.index)
    btc = returns["btc_return"].astype(float)

    # ── BTC ──
    rules = build_rules(btc)
    results, net_sharpe, placebos = evaluate(
        btc, rules, n_placebo=args.n_placebo,
        fee_bps_oneway=args.fee_bps_oneway)
    dow_favored = dow_favored_summary(btc)

    # ── Best mean-reversion rule -> basket (exclude calendar rules; "best" by
    #    Calmar among the four mean-reversion families). ──
    mr_names = {"rsi_reversal", "rsi_dipbuyer", "st_reversal", "bollinger"}
    mr_results = [r for r in results if r.name in mr_names]
    best = max(mr_results, key=lambda r: r.calmar)
    logger.info("best BTC mean-reversion rule by Calmar: %s", best.name)

    basket_cols = [f"{a}_return" for a in basket_assets]
    basket = returns[basket_cols].mean(axis=1).astype(float)
    basket_rule_exp = build_rules(basket)[best.name]
    basket_bh = backtest_rule(basket, pd.Series(1.0, index=basket.index),
                              fee_bps_oneway=args.fee_bps_oneway)
    basket_bh.name = "bh"
    basket_res = backtest_rule(basket, basket_rule_exp,
                               fee_bps_oneway=args.fee_bps_oneway)
    basket_res.name = best.name
    basket_net = {bps: backtest_rule(basket, basket_rule_exp,
                                     fee_bps_oneway=bps).sharpe
                  for bps in COST_LEVELS}
    basket_bh_net = {bps: backtest_rule(
        basket, pd.Series(1.0, index=basket.index),
        fee_bps_oneway=bps).sharpe for bps in COST_LEVELS}
    basket_placebo = placebo_p(basket, basket_rule_exp, basket_res,
                               metric="sharpe", n=args.n_placebo,
                               fee_bps_oneway=args.fee_bps_oneway)
    logger.info("basket %-16s sharpe=%.2f (BH %.2f) placebo p=%.2f",
                best.name, basket_res.sharpe, basket_bh.sharpe, basket_placebo)

    basket_block = {
        "rule": best.name, "n_assets": len(basket_assets),
        "bh": basket_bh, "rule_res": basket_res,
        "net": basket_net, "bh_net": basket_bh_net, "placebo": basket_placebo,
    }

    md = render_markdown(results, net_sharpe, placebos, dow_favored,
                         basket_block, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
