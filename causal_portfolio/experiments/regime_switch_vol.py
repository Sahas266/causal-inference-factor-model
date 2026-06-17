"""Volatility-regime-conditional strategy switching ('regime DAGs').

Every other experiment in this folder applies ONE allocation rule to the whole
sample.  This one asks the user's specific question: does it help to **detect a
volatility regime (calm vs volatile) and apply a DIFFERENT rule per regime** —
the classic "de-risk / mean-revert in volatile, ride momentum in calm" idea?

The honesty bar is identical to `regime_rotation` and `dag_strategies_findings`:

  * **Causal regime** — the volatility state at day *t* uses only trailing data.
    20d trailing realized vol of BTC (or VIX) is bucketed by its *trailing*
    rolling quantiles (calm = below 33rd pct of the trailing window, volatile =
    above 67th, mid in between).  Quantiles are computed on a rolling window,
    NEVER the full sample, so the label at *t* could have been known at *t*.
  * **Causal menu rules** — every per-regime candidate (hold, trend_50,
    vol_target, flat, mean_revert, momentum) is itself a trailing-only signal in
    [0, 1] on BTC; stables earn 0%.
  * **One-day lag + cost** — the composed exposure is decided on *t*'s close and
    applied to *t+1*'s return with a 5bp one-way switching cost.  This is done by
    handing the final composed exposure path to the validated
    `regime_rotation.backtest_rule`, which performs the shift / cost itself.

Two ways to assign a menu rule to each regime:

  1. **Pre-registered economic mapping** (fixed BEFORE looking at results).  The
     user's hypothesis encoded directly: a de-risk / mean-revert rule in the
     volatile regime, momentum / hold in calm.  Several fixed mappings are
     declared as constants below.

  2. **Adaptive causal selection** (no leak).  At each monthly rebalance, for the
     CURRENT regime, pick the menu rule with the best *trailing-window* Sharpe
     **restricted to days in that same regime**, using only past data, and apply
     it forward until the next rebalance.  Ranking never uses full-sample or
     future per-regime performance.

Evaluation vs THREE references:
  * buy-and-hold BTC (the unbeaten direction benchmark),
  * `trend_50` (the best UNCONDITIONAL rule in this repo, Sharpe ~0.88), and
  * a **regime-shuffle placebo**: circular-shift the regime-label series, rebuild
    the composed exposure with the SAME per-regime rules, re-run; p = share of
    shuffles with Sharpe >= the real strategy.  This isolates whether the
    *regime timing* carries information, separately from the per-regime rules.

A volatility-regime switch "works" only if it beats trend_50 AND its
regime-shuffle placebo p < 0.10.

Run:  python -m causal_portfolio.experiments.regime_switch_vol
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reuse the validated harness — do NOT reimplement backtest / cost logic.
from causal_portfolio.experiments.regime_rotation import backtest_rule, RotResult
from causal_portfolio.backtest.metrics import sharpe_ratio

logger = logging.getLogger("cpcm.experiments.regime_switch_vol")


# ── causal menu of per-regime rules (exposure in [0,1] on BTC) ───────

def _price(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def _trailing_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std()


def _rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI on the reconstructed price (trailing only)."""
    price = _price(returns)
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def build_menu(btc: pd.Series, *, vol_window: int = 20) -> dict[str, pd.Series]:
    """The per-regime candidate rules.  Each is a causal exposure path in [0,1].

    hold          always 1 (= BH BTC inside the regime)
    trend_50      1 when price > trailing 50d MA, else 0
    vol_target    clip(trailing-median vol / trailing vol, 0, 1), continuous
    flat          always 0 (de-risk to stables)
    mean_revert   1 when RSI14 < 30 (oversold dip-buy), else 0
    momentum      1 when trailing 30d return > 0, else 0
    """
    idx = btc.index
    price = _price(btc)
    vol = _trailing_vol(btc, vol_window)
    menu: dict[str, pd.Series] = {}
    menu["hold"] = pd.Series(1.0, index=idx)
    menu["trend_50"] = (price > price.rolling(50, min_periods=25).mean()).astype(float)
    tgt = vol.rolling(252, min_periods=60).median()
    menu["vol_target"] = (tgt / (vol + 1e-12)).clip(0, 1)
    menu["flat"] = pd.Series(0.0, index=idx)
    menu["mean_revert"] = (_rsi(btc, 14) < 30).astype(float)
    menu["momentum"] = (price / price.shift(30) - 1.0 > 0).astype(float)
    return menu


# ── causal volatility-regime labels (trailing rolling quantiles) ─────

def vol_regime(
    state: pd.Series, *, n_states: int = 3, window: int = 252,
    lo: float = 0.33, hi: float = 0.67, min_periods: int = 90,
) -> pd.Series:
    """Bucket a state series into 'calm' / 'normal' / 'volatile' by its TRAILING
    rolling quantiles.  The threshold at day t uses only state[..t], so the label
    is knowable at t (causal).  2-state collapses 'normal' into the boundary:
    calm below the trailing median, volatile above.

    Returns an object Series of {'calm','normal','volatile'} (NaN until warmup).
    """
    if n_states == 2:
        med = state.rolling(window, min_periods=min_periods).median()
        lab = pd.Series(np.where(state <= med, "calm", "volatile"), index=state.index,
                        dtype=object)
        lab[med.isna()] = np.nan
        return lab
    q_lo = state.rolling(window, min_periods=min_periods).quantile(lo)
    q_hi = state.rolling(window, min_periods=min_periods).quantile(hi)
    lab = pd.Series("normal", index=state.index, dtype=object)
    lab[state <= q_lo] = "calm"
    lab[state >= q_hi] = "volatile"
    lab[q_lo.isna() | q_hi.isna()] = np.nan
    return lab


# ── compose a per-regime exposure path ───────────────────────────────

def compose_fixed(
    labels: pd.Series, menu: dict[str, pd.Series], mapping: dict[str, str],
) -> pd.Series:
    """Build the composed exposure: on each day, use the menu rule assigned to
    that day's regime label.  Days before the regime warms up (NaN label) sit in
    stables (exposure 0) — conservative, and they predate any usable signal."""
    idx = labels.index
    exp = pd.Series(0.0, index=idx)
    for regime, rule_name in mapping.items():
        mask = (labels == regime)
        exp.loc[mask] = menu[rule_name].reindex(idx).loc[mask]
    exp.loc[labels.isna()] = 0.0
    return exp.fillna(0.0).clip(0, 1)


def compose_adaptive(
    btc: pd.Series, labels: pd.Series, menu: dict[str, pd.Series], *,
    rebalance: int = 21, lookback: int = 365, min_regime_days: int = 20,
    fee_bps_oneway: float = 5.0,
) -> tuple[pd.Series, dict]:
    """Adaptive causal selection.  Walk forward; every `rebalance` days, for EACH
    regime independently, score each menu rule by its trailing-`lookback`-day
    Sharpe **restricted to past days in that regime**, and assign the best rule to
    that regime until the next rebalance.  Selection uses only data strictly
    before the rebalance date, so there is no look-ahead.

    Returns (composed exposure path, info dict with #reselections etc.).
    """
    idx = btc.index
    n = len(idx)
    exp = pd.Series(0.0, index=idx)
    # Per-regime per-rule realized daily return inside the regime (for scoring):
    # rule exposure shifted 1d * btc return, evaluated only on regime days.
    menu_daily = {name: (e.shift(1).fillna(0.0).clip(0, 1) * btc.fillna(0.0))
                  for name, e in menu.items()}
    current: dict[str, str] = {}
    selections = []
    regimes = [r for r in ("calm", "normal", "volatile")]
    for start in range(0, n, rebalance):
        t0 = idx[start]
        # Re-select using only the trailing window strictly before t0.
        win_mask = (idx >= t0 - pd.Timedelta(days=lookback)) & (idx < t0)
        win_labels = labels[win_mask]
        for regime in regimes:
            reg_days = win_labels.index[win_labels == regime]
            if len(reg_days) < min_regime_days:
                # not enough history in this regime yet -> conservative flat
                current[regime] = current.get(regime, "flat")
                continue
            best_name, best_sh = "flat", -np.inf
            for name, daily in menu_daily.items():
                r = daily.reindex(reg_days).dropna().values
                if len(r) < min_regime_days:
                    continue
                sh = sharpe_ratio(r)
                if sh > best_sh:
                    best_name, best_sh = name, sh
            current[regime] = best_name
        end = min(start + rebalance, n)
        block = idx[start:end]
        block_labels = labels.reindex(block)
        for regime in regimes:
            mask = (block_labels == regime)
            rule = current.get(regime, "flat")
            exp.loc[block[mask.values]] = menu[rule].reindex(block).loc[block[mask.values]]
        exp.loc[block[block_labels.isna().values]] = 0.0
        selections.append((t0, dict(current)))
    info = {"n_reselections": len(selections), "selections": selections,
            "final_map": dict(current)}
    return exp.fillna(0.0).clip(0, 1), info


# ── regime-shuffle placebo ───────────────────────────────────────────

def regime_shuffle_p(
    btc: pd.Series, labels: pd.Series, menu: dict[str, pd.Series],
    real: RotResult, build_exposure, *, n: int = 200, seed: int = 0,
    metric: str = "sharpe", fee_bps_oneway: float = 5.0,
) -> float:
    """Circular-shift the REGIME LABEL series, rebuild the composed exposure with
    the SAME per-regime rule logic, re-run.  p = share of shuffles whose `metric`
    matches/beats the real strategy.  Isolates whether the *regime timing* (when
    we call calm vs volatile) carries information, holding the menu rules fixed.

    `build_exposure(shuffled_labels) -> exposure Series` lets the caller reuse
    either the fixed mapping or the adaptive selector with shuffled labels.
    """
    rng = np.random.default_rng(seed)
    lab_vals = labels.values
    real_v = getattr(real, metric)
    hits = 0
    valid = 0
    for _ in range(n):
        k = int(rng.integers(30, len(lab_vals) - 30))
        shuffled = pd.Series(np.roll(lab_vals, k), index=labels.index, dtype=object)
        exp = build_exposure(shuffled)
        r = backtest_rule(btc, exp, fee_bps_oneway=fee_bps_oneway)
        if np.isfinite(getattr(r, metric)):
            valid += 1
            hits += getattr(r, metric) >= real_v
    return hits / max(valid, 1)


# ── pre-registered economic mappings (FIXED before results) ──────────
# The user's hypothesis: de-risk / mean-revert in the volatile regime, ride
# momentum / hold in calm.  Declared as constants so they cannot be tuned to
# results.

MAPPINGS_3STATE: dict[str, dict[str, str]] = {
    # calm -> ride, normal -> trend, volatile -> de-risk / mean-revert
    "momo_calm__flat_vol":     {"calm": "momentum", "normal": "trend_50", "volatile": "flat"},
    "hold_calm__flat_vol":     {"calm": "hold",     "normal": "trend_50", "volatile": "flat"},
    "momo_calm__revert_vol":   {"calm": "momentum", "normal": "trend_50", "volatile": "mean_revert"},
    "hold_calm__voltgt_vol":   {"calm": "hold",     "normal": "hold",     "volatile": "vol_target"},
    "momo_calm__voltgt_vol":   {"calm": "momentum", "normal": "trend_50", "volatile": "vol_target"},
    # adversarial control: the WRONG way round (de-risk in calm, momentum in
    # volatile) — should look worse if the hypothesis has any content.
    "INVERTED_flat_calm__momo_vol": {"calm": "flat", "normal": "trend_50", "volatile": "momentum"},
}

MAPPINGS_2STATE: dict[str, dict[str, str]] = {
    "momo_calm__flat_vol":   {"calm": "momentum", "volatile": "flat"},
    "hold_calm__flat_vol":   {"calm": "hold",     "volatile": "flat"},
    "momo_calm__revert_vol": {"calm": "momentum", "volatile": "mean_revert"},
    "hold_calm__voltgt_vol": {"calm": "hold",     "volatile": "vol_target"},
    "INVERTED_flat_calm__momo_vol": {"calm": "flat", "volatile": "momentum"},
}


# ── per-regime dwell / switch diagnostics ────────────────────────────

@dataclass
class StratResult:
    name: str
    regime_def: str
    mode: str            # "fixed" or "adaptive"
    res: RotResult
    placebo_p: float | None
    dwell: dict[str, float]
    n_regime_switches: int


def _dwell(labels: pd.Series) -> dict[str, float]:
    vc = labels.dropna().value_counts(normalize=True)
    return {k: float(vc.get(k, 0.0)) for k in ("calm", "normal", "volatile")}


def _regime_switches(labels: pd.Series) -> int:
    l = labels.dropna()
    return int((l != l.shift(1)).sum() - 1) if len(l) else 0


# ── driver ───────────────────────────────────────────────────────────

def run(
    btc: pd.Series, vix: pd.Series | None, *,
    vol_window: int = 20, regime_window: int = 252, lo: float = 0.33,
    hi: float = 0.67, fee_bps_oneway: float = 5.0, n_placebo: int = 200,
    rebalance: int = 21, lookback: int = 365,
) -> dict:
    menu = build_menu(btc, vol_window=vol_window)

    # Reference strategies.
    bh = backtest_rule(btc, menu["hold"], fee_bps_oneway=fee_bps_oneway)
    bh.name = "BH_BTC"
    trend = backtest_rule(btc, menu["trend_50"], fee_bps_oneway=fee_bps_oneway)
    trend.name = "trend_50 (unconditional)"

    # Regime definitions: BTC realized vol (2 & 3 state) and VIX (3 state).
    btc_vol = _trailing_vol(btc, vol_window)
    regime_defs: dict[str, tuple[pd.Series, dict[str, dict[str, str]]]] = {
        "btcvol_3state": (vol_regime(btc_vol, n_states=3, window=regime_window,
                                     lo=lo, hi=hi), MAPPINGS_3STATE),
        "btcvol_2state": (vol_regime(btc_vol, n_states=2, window=regime_window),
                          MAPPINGS_2STATE),
    }
    if vix is not None:
        v = vix.reindex(btc.index).ffill()
        regime_defs["vix_3state"] = (
            vol_regime(v, n_states=3, window=regime_window, lo=lo, hi=hi),
            MAPPINGS_3STATE)

    results: list[StratResult] = []
    for rdef, (labels, mappings) in regime_defs.items():
        dwell = _dwell(labels)
        rsw = _regime_switches(labels)

        # 1) Fixed pre-registered mappings.
        for mname, mapping in mappings.items():
            exp = compose_fixed(labels, menu, mapping)
            res = backtest_rule(btc, exp, fee_bps_oneway=fee_bps_oneway)
            res.name = mname
            # Placebo only the configs that beat trend_50 on Sharpe (the bar).
            pp = None
            if res.sharpe > trend.sharpe:
                pp = regime_shuffle_p(
                    btc, labels, menu, res,
                    lambda sl, m=mapping: compose_fixed(sl, menu, m),
                    n=n_placebo, fee_bps_oneway=fee_bps_oneway)
            results.append(StratResult(mname, rdef, "fixed", res, pp, dwell, rsw))
            logger.info("[%s/fixed] %-30s ann=%+.0f%% sh=%.2f maxDD=%.0f%% p=%s",
                        rdef, mname, res.ann_return * 100, res.sharpe,
                        res.max_dd * 100, f"{pp:.2f}" if pp is not None else "-")

        # 2) Adaptive causal selection.
        exp_a, info = compose_adaptive(
            btc, labels, menu, rebalance=rebalance, lookback=lookback,
            fee_bps_oneway=fee_bps_oneway)
        res_a = backtest_rule(btc, exp_a, fee_bps_oneway=fee_bps_oneway)
        res_a.name = "ADAPTIVE_trailing_sharpe"
        pp_a = None
        if res_a.sharpe > trend.sharpe:
            pp_a = regime_shuffle_p(
                btc, labels, menu, res_a,
                lambda sl: compose_adaptive(
                    btc, sl, menu, rebalance=rebalance, lookback=lookback,
                    fee_bps_oneway=fee_bps_oneway)[0],
                n=n_placebo, fee_bps_oneway=fee_bps_oneway)
        results.append(StratResult("ADAPTIVE_trailing_sharpe", rdef, "adaptive",
                                    res_a, pp_a, dwell, rsw))
        logger.info("[%s/adaptive] ann=%+.0f%% sh=%.2f maxDD=%.0f%% p=%s | final=%s",
                    rdef, res_a.ann_return * 100, res_a.sharpe,
                    res_a.max_dd * 100, f"{pp_a:.2f}" if pp_a is not None else "-",
                    info["final_map"])

    return {"bh": bh, "trend": trend, "results": results, "regime_defs": regime_defs}


# ── report ───────────────────────────────────────────────────────────

def render_markdown(out: dict, args: dict) -> str:
    from datetime import datetime, timezone
    bh, trend = out["bh"], out["trend"]
    results: list[StratResult] = out["results"]

    L = ["# Volatility-regime-conditional strategy switching\n"]
    L.append("Detect a **volatility regime** (calm / normal / volatile) from a "
             "trailing-only signal, then apply a **different rule per regime** — "
             "the user's *\"de-risk / mean-revert in volatile, momentum in calm\"* "
             "hypothesis.  Same honesty bar as the rest of the DAG work: causal "
             "regime labels (trailing rolling quantiles), causal menu rules, "
             "one-day lag, "
             f"{args['fee_bps_oneway']:.0f}bp one-way cost.  A switch **works** only "
             "if it beats `trend_50` AND survives the regime-shuffle placebo "
             "(p<0.10).\n")
    L.append(f"- Asset: BTC | window `{args['start']}` → `{args['end']}` "
             f"({bh.daily.shape[0]} days)")
    L.append(f"- Regime: 20d realized vol / VIX, bucketed by trailing "
             f"{args['regime_window']}d rolling quantiles "
             f"(calm<{args['lo']:.2f}, volatile>{args['hi']:.2f})")
    L.append(f"- Adaptive: reselect every {args['rebalance']}d on trailing "
             f"{args['lookback']}d per-regime Sharpe")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    # Reference block.
    L.append("## References\n")
    L.append("| Strategy | Ann.ret | Sharpe | Sortino | MaxDD | Calmar |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in (bh, trend):
        L.append(f"| {r.name} | {r.ann_return:+.0%} | {r.sharpe:.2f} | "
                 f"{r.sortino:.2f} | {r.max_dd:.0%} | {r.calmar:.2f} |")
    L.append("")

    # Per-regime-def tables.
    for rdef in dict.fromkeys(s.regime_def for s in results):
        rows = [s for s in results if s.regime_def == rdef]
        d = rows[0].dwell
        L.append(f"## Regime def: `{rdef}` "
                 f"(dwell calm {d['calm']:.0%} / normal {d['normal']:.0%} / "
                 f"volatile {d['volatile']:.0%}, {rows[0].n_regime_switches} regime "
                 f"switches)\n")
        L.append("| Mapping (mode) | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | "
                 "%in mkt | switches | shuffle p |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in sorted(rows, key=lambda s: -s.res.sharpe):
            r = s.res
            pp = f"{s.placebo_p:.2f}" if s.placebo_p is not None else "—"
            tag = " (adaptive)" if s.mode == "adaptive" else ""
            L.append(f"| {s.name}{tag} | {r.ann_return:+.0%} | {r.sharpe:.2f} | "
                     f"{r.sortino:.2f} | {r.max_dd:.0%} | {r.calmar:.2f} | "
                     f"{r.pct_in_market:.0%} | {r.n_switches} | {pp} |")
        L.append("")

    # Verdict.
    L.append("## Verdict\n")
    L.append(f"Bar to clear: **trend_50** (Sharpe {trend.sharpe:.2f}, "
             f"maxDD {trend.max_dd:.0%}) AND regime-shuffle placebo p<0.10. "
             f"BH BTC reference Sharpe {bh.sharpe:.2f}.\n")
    survivors = [s for s in results
                 if s.res.sharpe > trend.sharpe
                 and s.placebo_p is not None and s.placebo_p < 0.10]
    beat_trend = [s for s in results if s.res.sharpe > trend.sharpe]
    if survivors:
        L.append("Configs that beat trend_50 on Sharpe AND survive the "
                 "regime-shuffle placebo (p<0.10):")
        for s in sorted(survivors, key=lambda s: -s.res.sharpe):
            L.append(f"- **{s.regime_def} / {s.name}** ({s.mode}) — Sharpe "
                     f"{s.res.sharpe:.2f} vs {trend.sharpe:.2f}, maxDD "
                     f"{s.res.max_dd:.0%}, shuffle p={s.placebo_p:.2f}")
    else:
        L.append("**No volatility-regime-conditional configuration both beats "
                 "trend_50 on Sharpe and survives the regime-shuffle placebo.**")
        if beat_trend:
            L.append(f"\n{len(beat_trend)} config(s) edged trend_50 on Sharpe "
                     "in-sample, but their regime *timing* does not beat a "
                     "circularly-shuffled regime label (placebo p≥0.10) — i.e. the "
                     "numbers come from the average per-regime exposure mix, not "
                     "from skill at calling calm vs volatile.")
        else:
            L.append("\nNothing even edges trend_50 in-sample: applying a "
                     "de-risk/mean-revert rule only in volatile regimes and "
                     "momentum/hold in calm does not improve on the plain "
                     "unconditional trend filter.")
    L.append("\n### Caveats")
    L.append("- **One market cycle** (2021–2025: a single bull→bear→recovery). "
             "Volatility in crypto clusters around BOTH crashes and parabolic "
             "rallies, so a 'volatile' label does not cleanly mean 'go down' — "
             "de-risking in high vol can cut upside as much as downside.")
    L.append("- **Multiple testing**: several fixed mappings × 3 regime "
             "definitions × an adaptive selector were tried; the best in-sample "
             "number is upward-biased. The placebo guards regime-timing luck on a "
             "*fixed* config, not the luck of having picked the best config.")
    L.append("- **Stables at 0%**; **no leverage** (exposure capped at 1, so every "
             "overlay can only de-risk — structurally disadvantaged on raw return "
             "in a bull).")
    L.append("- The `INVERTED_*` row is an adversarial control (de-risk in calm, "
             "momentum in volatile); if it is not clearly worse than its non-"
             "inverted twin, the regime label is not carrying directional info.")
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
    p.add_argument("--vol-window", type=int, default=20)
    p.add_argument("--regime-window", type=int, default=252)
    p.add_argument("--lo", type=float, default=0.33)
    p.add_argument("--hi", type=float, default=0.67)
    p.add_argument("--rebalance", type=int, default=21)
    p.add_argument("--lookback", type=int, default=365)
    p.add_argument("--out", default="causal_portfolio/docs/regime_switch_vol.md")
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

    out = run(
        btc, vix, vol_window=args.vol_window, regime_window=args.regime_window,
        lo=args.lo, hi=args.hi, fee_bps_oneway=args.fee_bps_oneway,
        n_placebo=args.n_placebo, rebalance=args.rebalance,
        lookback=args.lookback)

    md = render_markdown(out, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
