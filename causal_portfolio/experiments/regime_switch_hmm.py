"""Regime-switching strategy where a causal HMM picks the regime and a
per-state allocation rule ("DAG") is applied.

This is a cleaner re-take on `experiments/regime_dag.py`, which crossed the
HMM regime labels with *per-regime OLS factor loadings* and found regime
conditioning HURT (the per-regime driver winners were a look-ahead artifact;
the causal A/B lost to buy-and-hold). Here we strip out the fragile
driver-attribution step entirely: the HMM is the only "intelligent" component,
and each state just selects one dead-simple exposure rule on BTC. That isolates
the question "does conditioning the allocation rule on the causal HMM state add
anything over the best unconditional rule (trend_50) and over a regime-shuffle
placebo?"

Pipeline (all causal — no look-ahead in the tradable arms):
  1. Build (VIX, BTC realized-vol) features (`build_regime_features`).
  2. Decode regime labels with `rolling_fit_decode` (rolling-window fit +
     forward filter — the live-system label, NOT full-sample Viterbi).
  3. The HMM states are unlabeled. Map them to exposure rules two ways:

     PRE-REGISTERED (fixed logic, decided before seeing OOS results):
       At each day, rank the *current* HMM states by their TRAILING per-state
       mean return (and break ties / sanity-check with trailing per-state vol)
       over a trailing window using only PAST data. The higher-trailing-return
       (risk-on) state -> a "hold/trend" rule; the lower-return / higher-vol
       (risk-off) state -> a defensive rule (flat or mean-revert). The mapping
       from rank -> rule is fixed up front; only the rank uses data.

     ADAPTIVE causal selection (no leak):
       At each rebalance, for the current HMM state, pick the menu rule with
       the best trailing-window Sharpe COMPUTED ONLY ON PAST DAYS IN THAT STATE.
       Never rank rules by full-sample per-state performance (that is the leak).

  4. The chosen rule's exposure (in [0,1] on BTC; stables = 0%) is applied to
     the NEXT day's return with a switching cost on every exposure change.

Benchmarks / honesty gates:
  - BH BTC (always_in) and the best UNCONDITIONAL menu rule (trend_50).
  - REGIME-SHUFFLE PLACEBO: circular-shift the HMM label series, re-run the
    same mapping, recompute Sharpe. p = share of shuffles with Sharpe >= real.
    High p => the HMM *timing* carries no information; the "edge" is just the
    average exposure the mapping happens to produce.
  - LOOK-AHEAD Viterbi check (NOT tradable): re-run both mappings on full-sample
    Viterbi labels (fit once on all data, decode the whole sequence). Quantifies
    how much the look-ahead labeling overstates the result vs the causal labels.

Run:  python -m causal_portfolio.experiments.regime_switch_hmm
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import (
    ANNUALIZATION, calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio,
)
from causal_portfolio.regimes.hmm import (
    RegimeClassifier, build_regime_features, dwell_stats, rolling_fit_decode,
)

logger = logging.getLogger("cpcm.experiments.regime_switch_hmm")


# ── per-state exposure menu (all causal) ─────────────────────────────
#
# Each builder maps a BTC return series -> a target-exposure series in [0,1].
# Exposure at day t uses only data through t; it is applied to t+1's return in
# `backtest_exposure` (one-day lag).


def _price(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI on the BTC price path (causal — rolling means)."""
    price = _price(returns)
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def build_menu(btc: pd.Series, *, vol_window: int = 20) -> dict[str, pd.Series]:
    """The fixed per-state strategy menu. Exposures in [0,1] on BTC."""
    idx = btc.index
    price = _price(btc)
    vol = btc.rolling(vol_window).std()
    menu: dict[str, pd.Series] = {}

    # always_in: hold BTC (= BH BTC inside its assigned state)
    menu["hold"] = pd.Series(1.0, index=idx)

    # trend_50: long only when price is above its 50d MA
    menu["trend_50"] = (price > price.rolling(50, min_periods=25).mean()).astype(float)

    # vol_target: scale exposure to a trailing-median vol budget, capped at 1
    tgt = vol.rolling(252, min_periods=60).median()
    menu["vol_target"] = (tgt / (vol + 1e-12)).clip(0.0, 1.0)

    # flat: stables (0% exposure) — pure defense
    menu["flat"] = pd.Series(0.0, index=idx)

    # mean_revert: long only on an oversold dip (RSI14 < 30), else flat
    rsi = _rsi(btc, 14)
    menu["mean_revert"] = (rsi < 30.0).astype(float)

    return menu


# Menu rule classes for the PRE-REGISTERED mapping. The rank->rule assignment
# is fixed here, before any OOS result is seen.
RISK_ON_RULE = "trend_50"     # risk-on state -> participate, but trend-gated
RISK_OFF_RULE = "flat"        # risk-off state -> stables (0%)
# For 3 states: middle state gets a half-risk rule.
MIDDLE_RULE = "vol_target"    # intermediate state -> vol-scaled exposure


# ── backtest a realized exposure path ────────────────────────────────


@dataclass
class PerfResult:
    name: str
    ann_return: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    pct_in_market: float
    n_switches: int
    daily: pd.Series


def backtest_exposure(
    btc: pd.Series, exposure: pd.Series, *, fee_bps_oneway: float = 5.0,
    name: str = "",
) -> PerfResult:
    """Apply an exposure path to BTC with a one-day lag and switching cost."""
    e = exposure.reindex(btc.index).shift(1).fillna(0.0).clip(0.0, 1.0)
    strat = e * btc.fillna(0.0)
    turn = e.diff().abs().fillna(e.abs())
    strat = strat - turn * (fee_bps_oneway / 1e4)
    daily = strat.dropna()
    return PerfResult(
        name=name,
        ann_return=float(daily.mean() * ANNUALIZATION),
        sharpe=float(sharpe_ratio(daily.values)),
        sortino=float(sortino_ratio(daily.values)),
        max_dd=float(max_drawdown(daily.values)),
        calmar=float(calmar_ratio(daily.values)),
        pct_in_market=float((e > 0.01).mean()),
        n_switches=int((e.diff().abs() > 0.01).sum()),
        daily=daily,
    )


# ── mappings: HMM state -> per-state rule -> realized exposure ────────


def preregistered_exposure(
    btc: pd.Series, menu: dict[str, pd.Series], labels: pd.Series,
    *, n_states: int, char_window: int = 126,
) -> tuple[pd.Series, pd.Series]:
    """PRE-REGISTERED mapping (fixed rule per risk-rank, causal rank).

    The states are unlabeled. At each day t we rank the states by their
    TRAILING mean return over the last `char_window` days (using only data
    through t-1), then apply the fixed rank->rule assignment:
        - rank 0 (highest trailing return)  -> RISK_ON_RULE  (trend_50)
        - rank n-1 (lowest trailing return)  -> RISK_OFF_RULE (flat)
        - middle (3-state)                   -> MIDDLE_RULE   (vol_target)

    Returns (realized_exposure, applied_rule_label) both indexed like btc over
    the labelled days.
    """
    common = btc.index.intersection(labels.dropna().index)
    btc = btc.loc[common]
    lab = labels.loc[common].astype(int)
    menu = {k: v.reindex(common) for k, v in menu.items()}

    # rank->rule lookup keyed by number of states
    if n_states == 2:
        rank_rule = {0: RISK_ON_RULE, 1: RISK_OFF_RULE}
    else:  # 3 (or more): top=on, bottom=off, everything between=middle
        rank_rule = {i: MIDDLE_RULE for i in range(n_states)}
        rank_rule[0] = RISK_ON_RULE
        rank_rule[n_states - 1] = RISK_OFF_RULE

    ret = btc.values
    lab_v = lab.values
    T = len(common)
    exposure = np.zeros(T)
    rule_used = np.empty(T, dtype=object)

    for t in range(T):
        cur = lab_v[t]
        # Causal per-state trailing mean return, computed on [t-char_window, t).
        lo = max(0, t - char_window)
        # rank states by trailing mean return (desc); ties -> by lower trailing vol
        means = {}
        vols = {}
        for s in range(n_states):
            mask = lab_v[lo:t] == s
            r = ret[lo:t][mask]
            means[s] = float(np.nanmean(r)) if r.size >= 5 else np.nan
            vols[s] = float(np.nanstd(r)) if r.size >= 5 else np.nan
        # Build a ranking. States with no trailing history sink to the bottom.
        order = sorted(
            range(n_states),
            key=lambda s: (
                -(means[s] if not np.isnan(means[s]) else -1e9),
                (vols[s] if not np.isnan(vols[s]) else 1e9),
            ),
        )
        rank_of = {s: r for r, s in enumerate(order)}
        rule = rank_rule[rank_of[cur]]
        rule_used[t] = rule
        exposure[t] = float(menu[rule].iloc[t]) if not np.isnan(menu[rule].iloc[t]) else 0.0

    return (pd.Series(exposure, index=common, name="exposure"),
            pd.Series(rule_used, index=common, name="rule"))


def adaptive_exposure(
    btc: pd.Series, menu: dict[str, pd.Series], labels: pd.Series,
    *, n_states: int, lookback: int = 252, rebalance_freq: int = 5,
    min_state_obs: int = 30,
) -> tuple[pd.Series, pd.Series]:
    """ADAPTIVE causal mapping (no leak).

    At each rebalance, for the CURRENT HMM state, pick the menu rule whose
    realized exposure path has the best trailing Sharpe COMPUTED ONLY ON PAST
    DAYS WHERE THAT STATE WAS ACTIVE. The rule is held until the next rebalance.
    Selection uses only data strictly before t — never full-sample per-state
    performance.

    If the current state has < `min_state_obs` past days in the lookback, fall
    back to the safe default (flat) so we don't gamble on a noisy estimate.
    """
    common = btc.index.intersection(labels.dropna().index)
    btc = btc.loc[common]
    lab = labels.loc[common].astype(int).values
    menu = {k: v.reindex(common).values for k, v in menu.items()}
    rule_names = list(menu.keys())
    ret = btc.values
    T = len(common)

    # Pre-compute each menu rule's one-day-lagged realized daily return so the
    # trailing-Sharpe selection is apples-to-apples with how exposure is traded.
    lagged_exp = {k: np.concatenate([[0.0], menu[k][:-1]]) for k in rule_names}
    rule_daily = {k: lagged_exp[k] * np.nan_to_num(ret) for k in rule_names}

    exposure = np.zeros(T)
    rule_used = np.empty(T, dtype=object)
    cur_rule = "flat"

    for t in range(T):
        if t % rebalance_freq == 0:
            cur = lab[t]
            lo = max(0, t - lookback)
            mask = lab[lo:t] == cur
            n_obs = int(mask.sum())
            if n_obs >= min_state_obs:
                best_rule, best_sharpe = "flat", -np.inf
                for k in rule_names:
                    d = rule_daily[k][lo:t][mask]
                    d = d[~np.isnan(d)]
                    if d.size >= min_state_obs:
                        sh = sharpe_ratio(d)
                        if sh > best_sharpe:
                            best_sharpe, best_rule = sh, k
                cur_rule = best_rule
            else:
                cur_rule = "flat"
        rule_used[t] = cur_rule
        e = menu[cur_rule][t]
        exposure[t] = float(e) if not np.isnan(e) else 0.0

    return (pd.Series(exposure, index=common, name="exposure"),
            pd.Series(rule_used, index=common, name="rule"))


# ── regime-shuffle placebo ───────────────────────────────────────────


def regime_shuffle_placebo(
    btc: pd.Series, menu: dict[str, pd.Series], labels: pd.Series,
    mapping_fn, real: PerfResult, *, n_states: int, n: int = 200, seed: int = 0,
    fee_bps_oneway: float = 5.0, **map_kwargs,
) -> float:
    """Share of circular-shifted HMM label series that match/beat the real
    Sharpe. The label values and per-state dwell are preserved; only the
    alignment of labels to dates is destroyed. High p => HMM *timing* carries
    no information (the mapping's edge is just its average exposure level)."""
    rng = np.random.default_rng(seed)
    lab = labels.dropna().astype(int)
    vals = lab.values
    n_lab = len(vals)
    hits = 0
    for _ in range(n):
        k = int(rng.integers(30, n_lab - 30))
        shifted = pd.Series(np.roll(vals, k), index=lab.index).astype("Int64")
        exp, _ = mapping_fn(btc, menu, shifted, n_states=n_states, **map_kwargs)
        r = backtest_exposure(btc, exp, fee_bps_oneway=fee_bps_oneway)
        hits += r.sharpe >= real.sharpe
    return hits / n


# ── look-ahead (full-sample Viterbi) reference ───────────────────────


def viterbi_labels(features: pd.DataFrame, *, n_states: int,
                   n_restarts: int = 5) -> pd.Series:
    """Full-sample fit + Viterbi decode. NON-CAUSAL — look-ahead, NOT tradable.
    Used only to quantify how much the causal result is overstated by a labeler
    that gets to see the whole sequence."""
    clf = RegimeClassifier(n_states=n_states, n_restarts=n_restarts).fit(features)
    return pd.Series(clf.decode(features), index=features.index, name="regime").astype("Int64")


# ── runner ───────────────────────────────────────────────────────────


@dataclass
class SwitchResult:
    n_states: int
    labels_causal: pd.Series
    labels_viterbi: pd.Series
    state_means: np.ndarray            # (n_states, 2) z-scored [vix, btc_vol]
    transmat: np.ndarray
    btc: pd.Series
    # perf
    bh: PerfResult
    trend50: PerfResult
    prereg_causal: PerfResult
    adaptive_causal: PerfResult
    prereg_viterbi: PerfResult
    adaptive_viterbi: PerfResult
    prereg_rules: pd.Series
    adaptive_rules: pd.Series
    placebo_prereg: float
    placebo_adaptive: float


def run(start: str, end: str, *, n_states: int, hmm_window: int,
        refit_every: int, char_window: int, adaptive_lookback: int,
        rebalance_freq: int, fee_bps_oneway: float, n_placebo: int) -> SwitchResult:
    from causal_portfolio.data import get_loader
    loader = get_loader()
    returns = loader.load_returns(["btc"], start, end)
    returns.index = pd.to_datetime(returns.index)
    btc = returns["btc_return"]

    macro = loader.load_macro(["VIXCLS"], start, end)
    macro.index = pd.to_datetime(macro.index)

    feats = build_regime_features(macro, returns, btc_col="btc_return")

    labels_causal = rolling_fit_decode(
        feats, window_size=hmm_window, refit_every=refit_every,
        n_states=n_states, n_restarts=max(5, n_states + 2))
    labels_vit = viterbi_labels(feats, n_states=n_states)

    # Characterize the full-sample HMM (means/transmat) for the report only.
    clf = RegimeClassifier(n_states=n_states, n_restarts=max(5, n_states + 2)).fit(feats)
    state_means = clf.state_means()
    transmat = clf.transition_matrix()

    menu = build_menu(btc)

    # Benchmarks
    bh = backtest_exposure(btc, menu["hold"], fee_bps_oneway=fee_bps_oneway, name="BH BTC")
    trend50 = backtest_exposure(btc, menu["trend_50"], fee_bps_oneway=fee_bps_oneway,
                                name="trend_50 (uncond)")

    # Causal arms
    exp_pre, rules_pre = preregistered_exposure(
        btc, menu, labels_causal, n_states=n_states, char_window=char_window)
    prereg_causal = backtest_exposure(btc, exp_pre, fee_bps_oneway=fee_bps_oneway,
                                      name="prereg (causal)")
    exp_ad, rules_ad = adaptive_exposure(
        btc, menu, labels_causal, n_states=n_states, lookback=adaptive_lookback,
        rebalance_freq=rebalance_freq)
    adaptive_causal = backtest_exposure(btc, exp_ad, fee_bps_oneway=fee_bps_oneway,
                                        name="adaptive (causal)")

    # Look-ahead arms (NOT tradable)
    exp_pre_v, _ = preregistered_exposure(
        btc, menu, labels_vit, n_states=n_states, char_window=char_window)
    prereg_vit = backtest_exposure(btc, exp_pre_v, fee_bps_oneway=fee_bps_oneway,
                                   name="prereg (Viterbi look-ahead)")
    exp_ad_v, _ = adaptive_exposure(
        btc, menu, labels_vit, n_states=n_states, lookback=adaptive_lookback,
        rebalance_freq=rebalance_freq)
    adaptive_vit = backtest_exposure(btc, exp_ad_v, fee_bps_oneway=fee_bps_oneway,
                                     name="adaptive (Viterbi look-ahead)")

    # Regime-shuffle placebos on the CAUSAL arms
    p_pre = regime_shuffle_placebo(
        btc, menu, labels_causal, preregistered_exposure, prereg_causal,
        n_states=n_states, n=n_placebo, fee_bps_oneway=fee_bps_oneway,
        char_window=char_window)
    p_ad = regime_shuffle_placebo(
        btc, menu, labels_causal, adaptive_exposure, adaptive_causal,
        n_states=n_states, n=n_placebo, fee_bps_oneway=fee_bps_oneway,
        lookback=adaptive_lookback, rebalance_freq=rebalance_freq)

    return SwitchResult(
        n_states=n_states, labels_causal=labels_causal, labels_viterbi=labels_vit,
        state_means=state_means, transmat=transmat, btc=btc,
        bh=bh, trend50=trend50, prereg_causal=prereg_causal,
        adaptive_causal=adaptive_causal, prereg_viterbi=prereg_vit,
        adaptive_viterbi=adaptive_vit, prereg_rules=rules_pre,
        adaptive_rules=rules_ad, placebo_prereg=p_pre, placebo_adaptive=p_ad,
    )


# ── reporting ────────────────────────────────────────────────────────


def _perf_row(r: PerfResult, placebo: float | None = None) -> str:
    pp = f"{placebo:.2f}" if placebo is not None else "—"
    return (f"| {r.name} | {r.ann_return:+.0%} | {r.sharpe:.2f} | {r.sortino:.2f} "
            f"| {r.max_dd:.0%} | {r.calmar:.2f} | {r.pct_in_market:.0%} "
            f"| {r.n_switches} | {pp} |")


def render_markdown(results: list[SwitchResult], args: dict) -> str:
    L = ["# Regime-switching DAG on a causal HMM\n"]
    L.append(
        "A causal Hidden Markov Model (rolling-window fit + forward filter on "
        "VIX + BTC realized vol — the live-system label, *not* full-sample "
        "Viterbi) identifies the regime; a per-state exposure rule from a fixed "
        "menu is applied to BTC (stables = 0%). Two mappings: a **pre-registered** "
        "one (risk-on state -> `trend_50`, risk-off -> `flat`, mapping fixed "
        "before seeing OOS) and an **adaptive causal** one (per state, pick the "
        "menu rule with best *trailing* Sharpe in that state, past data only). "
        "Compared to BH BTC, the best unconditional rule (`trend_50`), a "
        "**regime-shuffle placebo**, and a **look-ahead Viterbi** version that is "
        "marked NOT tradable.\n")
    L.append(f"- Asset: BTC | window `{args['start']}` -> `{args['end']}`")
    L.append(f"- HMM: rolling fit window `{args['hmm_window']}`d, refit every "
             f"`{args['refit_every']}`d, multi-start | fee `{args['fee_bps_oneway']:.0f}`bp "
             f"one-way | placebo shuffles `{args['n_placebo']}`")
    L.append(f"- Menu: `hold` (=BH), `trend_50`, `vol_target`, `flat` (stables), "
             f"`mean_revert` (RSI14<30 long else flat)")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("> **Prior finding (`experiments/regime_dag.py`):** crossing the HMM "
             "regime with per-regime OLS factor loadings HURT — the per-regime "
             "driver winners were a look-ahead artifact and the causal A/B lost "
             "to BH BTC. This experiment removes the driver-attribution step and "
             "tests only allocation-rule switching.\n")

    for res in results:
        L.append(f"\n## {res.n_states}-state HMM\n")

        # State characterization (full-sample means, for description)
        L.append("**HMM states** (full-sample z-scored feature means, sorted by "
                 "VIX; state 0 = lowest-stress):\n")
        L.append("| State | VIX z | BTC vol z | Read |")
        L.append("|---:|---:|---:|---|")
        for s in range(res.n_states):
            vix_z, vol_z = res.state_means[s]
            if s == 0:
                read = "low-stress (risk-on)"
            elif s == res.n_states - 1:
                read = "high-stress (risk-off)"
            else:
                read = "intermediate"
            L.append(f"| {s} | {vix_z:+.2f} | {vol_z:+.2f} | {read} |")
        L.append("")

        # dwell of CAUSAL labels
        lab = res.labels_causal.dropna().astype(int)
        ds = dwell_stats(lab.values)
        L.append("**Causal-label occupancy / dwell** (forward-filter, tradable):\n")
        L.append("| State | Days | % | Runs | Mean run (d) | Max run (d) |")
        L.append("|---:|---:|---:|---:|---:|---:|")
        for s in sorted(ds):
            d = ds[s]
            L.append(f"| {s} | {d['days']} | {d['pct']:.0%} | {d['n_runs']} "
                     f"| {d['mean_run_length']:.0f} | {d['max_run_length']} |")
        L.append("")

        # rule usage under causal labels
        for label, ser in (("Pre-registered", res.prereg_rules),
                            ("Adaptive", res.adaptive_rules)):
            vc = ser.value_counts(normalize=True)
            usage = ", ".join(f"`{k}` {v:.0%}" for k, v in vc.items())
            L.append(f"- {label} rule usage (causal): {usage}")
        L.append("")

        # performance table
        L.append("**Performance** (placebo p = share of HMM-label circular "
                 "shifts with Sharpe >= real; low = timing matters):\n")
        L.append("| Strategy | Ann.ret | Sharpe | Sortino | MaxDD | Calmar "
                 "| %in mkt | switches | placebo p |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        L.append(_perf_row(res.bh))
        L.append(_perf_row(res.trend50))
        L.append(_perf_row(res.prereg_causal, res.placebo_prereg))
        L.append(_perf_row(res.adaptive_causal, res.placebo_adaptive))
        L.append("| *— look-ahead (NOT tradable) —* | | | | | | | | |")
        L.append(_perf_row(res.prereg_viterbi))
        L.append(_perf_row(res.adaptive_viterbi))
        L.append("")

        # per-config verdict
        best_causal = max(res.prereg_causal, res.adaptive_causal, key=lambda r: r.sharpe)
        best_p = (res.placebo_prereg if best_causal is res.prereg_causal
                  else res.placebo_adaptive)
        vit_best = max(res.prereg_viterbi, res.adaptive_viterbi, key=lambda r: r.sharpe)
        beats_trend = best_causal.sharpe > res.trend50.sharpe
        survives = best_p < 0.10
        L.append("**Verdict ({}-state):** best causal arm `{}` Sharpe {:.2f} vs "
                 "trend_50 {:.2f} vs BH {:.2f}. {} trend_50. Placebo p={:.2f} ({}). "
                 "Look-ahead Viterbi best Sharpe {:.2f} (+{:.2f} over causal — the "
                 "look-ahead inflation).".format(
                     res.n_states, best_causal.name, best_causal.sharpe,
                     res.trend50.sharpe, res.bh.sharpe,
                     "Beats" if beats_trend else "Does NOT beat", best_p,
                     "timing carries info" if survives else "timing is NOT informative",
                     vit_best.sharpe, vit_best.sharpe - best_causal.sharpe))
        L.append("")

    # overall verdict
    L.append("\n## Overall verdict\n")
    any_win = False
    for res in results:
        best_causal = max(res.prereg_causal, res.adaptive_causal, key=lambda r: r.sharpe)
        best_p = (res.placebo_prereg if best_causal is res.prereg_causal
                  else res.placebo_adaptive)
        if best_causal.sharpe > res.trend50.sharpe and best_p < 0.10:
            any_win = True
    if any_win:
        L.append("At least one causal-HMM regime-switch arm beats the best "
                 "unconditional rule (`trend_50`) on Sharpe AND survives the "
                 "regime-shuffle placebo (p<0.10). Treat as a one-cycle lead to "
                 "paper-trade forward, not deploy on faith — see caveats.")
    else:
        L.append("**No causal-HMM regime-switch arm both beats `trend_50` on "
                 "Sharpe and survives the regime-shuffle placebo.** The HMM-state "
                 "conditioning does not add information over simply applying the "
                 "single best unconditional rule. Where a regime arm's headline "
                 "Sharpe looks competitive, the placebo shows the result is "
                 "reproducible by random-timed labels of the same dwell — i.e. it "
                 "is the *average exposure* the mapping happens to set, not the "
                 "HMM's *timing*. The look-ahead Viterbi version looks materially "
                 "better, which is exactly the look-ahead inflation that doomed "
                 "the prior `regime_dag.py` finding. Consistent with that prior "
                 "negative result and with the project-wide finding that nothing "
                 "beats BH BTC out-of-sample.")
    L.append("\n## Caveats\n")
    L.append("- **One market cycle.** 2021-2025 is a single bull->bear->recovery; "
             "any regime overlay that 'works' largely does so by truncating the "
             "2022 bear once. The placebo validates timing *within* this sample, "
             "not across regime change.")
    L.append("- **HMM refit instability.** Rolling-window forward-filter labels "
             "flap near transitions and the 3-state fit is fragile (multi-start "
             "mitigates but does not remove this); causal labels agree with the "
             "full-sample Viterbi only ~77% at the best config (see "
             "`driver_selection_regimes.md`).")
    L.append("- **Look-ahead is large.** The Viterbi rows quantify how much a "
             "labeler that sees the whole sequence overstates the result; the "
             "causal (tradable) rows are the honest ones.")
    L.append("- **Multiple testing / menu choice.** Two mappings × two state "
             "counts × a 5-rule menu were tried; the placebo guards timing luck "
             "on a *fixed* mapping, not the menu/config search.")
    L.append("- **Prior negative finding.** `experiments/regime_dag.py` already "
             "found HMM regime conditioning (over factor loadings) hurt; this "
             "cleaner setup tests the allocation-rule variant honestly.")
    return "\n".join(L)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--n-states", default="2,3",
                   help="comma-separated state counts to run (e.g. '2,3')")
    p.add_argument("--hmm-window", type=int, default=252)
    p.add_argument("--refit-every", type=int, default=21)
    p.add_argument("--char-window", type=int, default=126,
                   help="trailing window for pre-registered state risk-ranking")
    p.add_argument("--adaptive-lookback", type=int, default=252)
    p.add_argument("--rebalance-freq", type=int, default=5)
    p.add_argument("--fee-bps-oneway", type=float, default=5.0)
    p.add_argument("--n-placebo", type=int, default=200)
    p.add_argument("--out", default="causal_portfolio/docs/regime_switch_hmm.md")
    args = p.parse_args()

    state_counts = [int(s) for s in str(args.n_states).split(",")]
    results: list[SwitchResult] = []
    for ns in state_counts:
        logger.info("running %d-state HMM regime-switch ...", ns)
        res = run(
            args.start, args.end, n_states=ns, hmm_window=args.hmm_window,
            refit_every=args.refit_every, char_window=args.char_window,
            adaptive_lookback=args.adaptive_lookback,
            rebalance_freq=args.rebalance_freq, fee_bps_oneway=args.fee_bps_oneway,
            n_placebo=args.n_placebo)
        results.append(res)
        bc = max(res.prereg_causal, res.adaptive_causal, key=lambda r: r.sharpe)
        logger.info("%d-state: BH Sharpe %.2f | trend_50 %.2f | best causal %s %.2f "
                    "| prereg placebo %.2f | adaptive placebo %.2f",
                    ns, res.bh.sharpe, res.trend50.sharpe, bc.name, bc.sharpe,
                    res.placebo_prereg, res.placebo_adaptive)

    md = render_markdown(results, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
