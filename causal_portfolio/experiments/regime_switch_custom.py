"""Custom-regime DAG switching: a *different* allocation rule per regime.

This is regime-SWITCHING (not the single-state gates of `regime_rotation` /
`breadth_correlation_dags`). The pipeline is:

    custom_regime_t  ->  pick a menu rule for that regime  ->  exposure_t
                      ->  return_{t+1}

A "DAG" here is: a regime label (categorical market state), and a mapping
{regime -> menu rule}. At each day the active regime selects which exposure
rule is in force; that rule's exposure is applied to the next day's return.

Everything is causal. Regimes are labeled at t from data through t-1 (we shift
the label by one extra day so the state at decision time uses only *past*
information), and the harness (`regime_rotation.backtest_rule`) applies the
chosen exposure to t+1 with a switching cost.

CUSTOM REGIME DEFINITIONS (the novelty — not plain trend/vol/HMM):
  drawdown   "at highs" (within X% of trailing peak) vs "in drawdown" (>X% off)
  trending   efficiency ratio  |Δprice over N| / Σ|daily Δ|  high=trend, low=range
  corr       trailing-30d avg pairwise correlation of majors, high vs low
  breadth    % of majors above their own 50d MA, high vs low

PER-REGIME STRATEGY MENU (each is a causal exposure series in [0,1]):
  hold         always 1 (full risk)
  trend_50     1 when price > 50d MA else 0
  vol_target   clip(trailing-median-vol / trailing-vol, 0, 1)
  flat         always 0 (stables, 0% return)
  mean_revert  1 when RSI14 < 30 (oversold dip-buy) else 0
  momentum     1 when trailing-30d return > 0 else 0

ASSIGNMENT METHODS:
  1. PRE-REGISTERED economic mapping (fixed before seeing results):
     - trending  -> {trend: momentum, range: mean_revert}
     - drawdown  -> {at_highs: hold, in_drawdown: flat}
     - corr      -> {high_corr: flat (de-risk), low_corr: hold}
     - breadth   -> {high_breadth: hold, low_breadth: flat}
  2. ADAPTIVE causal selection (NO leak): at each rebalance, for the CURRENT
     regime, pick the menu rule with the best trailing-window Sharpe IN THAT
     REGIME using only past data, then apply it forward. Rules are NEVER ranked
     by full-sample per-regime performance.

EVALUATION:
  - vs BH BTC, BH basket (for basket rules), and best unconditional rule trend_50
  - REGIME-SHUFFLE PLACEBO: circular-shift the regime label series, rebuild the
    exposure under the SAME mapping, re-run. p = share of shuffles with
    Sharpe >= the real strategy. High p => the regime *labels* carry no timing
    information (the result is reproducible with scrambled regimes).

Run:  python -m causal_portfolio.experiments.regime_switch_custom
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import sharpe_ratio
from causal_portfolio.experiments.regime_rotation import backtest_rule

logger = logging.getLogger("cpcm.experiments.regime_switch_custom")

MAJORS = ["btc", "eth", "sol", "bnb", "avax", "xrp", "doge", "link",
          "uni", "aave", "crv"]


# ════════════════════════════════════════════════════════════════════
# Menu of per-regime exposure rules (all causal, in [0,1])
# ════════════════════════════════════════════════════════════════════

def _price(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI on a reconstructed price (causal: uses price[..t])."""
    delta = price.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    roll_dn = down.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def build_menu(btc: pd.Series) -> dict[str, pd.Series]:
    """{rule_name -> exposure in [0,1]} on BTC. All causal/trailing."""
    idx = btc.index
    price = _price(btc)
    vol = btc.rolling(20).std()
    menu: dict[str, pd.Series] = {}

    menu["hold"] = pd.Series(1.0, index=idx)
    menu["flat"] = pd.Series(0.0, index=idx)
    menu["trend_50"] = (price > price.rolling(50, min_periods=25).mean()
                        ).astype(float)
    tgt = vol.rolling(252, min_periods=60).median()
    menu["vol_target"] = (tgt / (vol + 1e-12)).clip(0, 1)
    rsi = _rsi(price, 14)
    menu["mean_revert"] = (rsi < 30).astype(float)
    roc30 = price / price.shift(30) - 1.0
    menu["momentum"] = (roc30 > 0).astype(float)

    for k, v in menu.items():
        menu[k] = v.reindex(idx).fillna(0.0)
    return menu


# ════════════════════════════════════════════════════════════════════
# Custom regime label builders (categorical, causal)
# ════════════════════════════════════════════════════════════════════
#
# Each builder returns an *integer-coded* regime Series plus a list of the
# regime names in code order. Labels are causal: a label at t is derived only
# from data through t (we apply an extra one-day shift downstream so the
# decision uses the label known at the prior close).

def regime_drawdown(btc: pd.Series, threshold: float = 0.10) -> tuple[pd.Series, list[str]]:
    """At-highs (drawdown shallower than -threshold) vs in-drawdown."""
    eq = _price(btc)
    dd = eq / eq.cummax() - 1.0
    lab = (dd <= -threshold).astype(int)  # 0 at_highs, 1 in_drawdown
    return lab, ["at_highs", "in_drawdown"]


def efficiency_ratio(btc: pd.Series, window: int = 20) -> pd.Series:
    """Kaufman efficiency ratio: |net change| / sum |daily change| over N.

    High (-> 1) = clean trend; low (-> 0) = choppy/ranging. Causal."""
    price = _price(btc)
    net = (price - price.shift(window)).abs()
    path = price.diff().abs().rolling(window).sum()
    return (net / (path + 1e-12)).clip(0, 1)


def regime_trending(btc: pd.Series, window: int = 20,
                    q: float = 0.50) -> tuple[pd.Series, list[str]]:
    """Trending vs ranging via efficiency ratio above/below its trailing median.

    Threshold is a trailing rolling quantile (causal), not a fixed magic number.
    """
    er = efficiency_ratio(btc, window)
    thresh = er.rolling(252, min_periods=60).quantile(q)
    lab = (er > thresh).astype(int)  # 0 ranging, 1 trending
    return lab, ["ranging", "trending"]


def avg_pairwise_corr(returns: pd.DataFrame, window: int = 30) -> pd.Series:
    """Trailing avg off-diagonal pairwise correlation across majors (causal)."""
    n = len(returns.columns)
    out = pd.Series(index=returns.index, dtype=float)
    arr = returns.values
    iu = np.triu_indices(n, k=1)
    for i in range(len(returns.index)):
        if i + 1 < window:
            continue
        w = arr[i - window + 1: i + 1]
        c = np.corrcoef(w, rowvar=False)
        vals = c[iu]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out.iloc[i] = float(vals.mean())
    return out


def regime_correlation(returns: pd.DataFrame, window: int = 30,
                       q: float = 0.70) -> tuple[pd.Series, list[str]]:
    """High vs low trailing average pairwise correlation of the majors.

    High corr (>= trailing q-quantile) is the risk-off / crash-correlation state.
    """
    corr = avg_pairwise_corr(returns, window)
    thresh = corr.rolling(252, min_periods=60).quantile(q)
    lab = (corr >= thresh).astype(int)  # 0 low_corr, 1 high_corr
    return lab, ["low_corr", "high_corr"]


def regime_breadth(returns: pd.DataFrame, ma: int = 50,
                   level: float = 0.50) -> tuple[pd.Series, list[str]]:
    """% of majors above their own 50d MA, above vs below `level` (50%)."""
    prices = (1.0 + returns.fillna(0.0)).cumprod()
    ma_lvl = prices.rolling(ma, min_periods=ma).mean()
    above = (prices > ma_lvl) & ma_lvl.notna()
    den = ma_lvl.notna().sum(axis=1).replace(0, np.nan)
    breadth = (above.sum(axis=1) / den).astype(float)
    lab = (breadth >= level).astype(int)  # 0 low_breadth, 1 high_breadth
    return lab, ["low_breadth", "high_breadth"]


# ════════════════════════════════════════════════════════════════════
# Build a switching exposure path from (regime labels, mapping)
# ════════════════════════════════════════════════════════════════════

def switch_exposure(labels: pd.Series, names: list[str],
                    mapping: dict[str, str],
                    menu: dict[str, pd.Series]) -> pd.Series:
    """Compose the exposure series: in each regime, use mapping[regime]'s rule.

    `labels` is integer-coded (causal as built). We shift it by 1 so the regime
    active at the *decision* close was known at the prior close — defense in
    depth against using same-day information to pick the rule. The downstream
    backtest_rule shifts once more for the return application.
    """
    idx = labels.index
    lab = labels.reindex(idx)
    exposure = pd.Series(np.nan, index=idx)
    for code, rname in enumerate(names):
        rule = mapping[rname]
        mask = (lab == code)
        exposure[mask] = menu[rule].reindex(idx)[mask]
    # shift the *label-driven selection* by one day (causal label use), warmup -> 0
    return exposure.shift(1).reindex(idx).fillna(0.0).clip(0, 1)


# ── adaptive causal selection ───────────────────────────────────────

def adaptive_exposure(labels: pd.Series, names: list[str],
                      menu: dict[str, pd.Series], target: pd.Series,
                      *, lookback: int = 365, rebalance: int = 21,
                      min_obs: int = 30) -> tuple[pd.Series, dict]:
    """For the current regime at each rebalance, pick the menu rule with the
    best trailing-window Sharpe IN THAT REGIME using only past data, apply fwd.

    NEVER ranks rules by full-sample per-regime performance: at decision day d,
    only target returns and labels with index < d enter the Sharpe computation.
    Selection is re-evaluated every `rebalance` days; between rebalances the
    chosen rule per regime is held fixed. Returns (exposure, diagnostics).
    """
    idx = labels.index
    lab = labels.reindex(idx)
    tr = target.reindex(idx).fillna(0.0)
    rule_names = list(menu.keys())
    # Pre-shift each rule's exposure to its realized next-day contribution, so a
    # trailing Sharpe over a window reflects what that rule WOULD have earned.
    rule_pnl = {rn: (menu[rn].reindex(idx).shift(1).fillna(0.0) * tr)
                for rn in rule_names}

    exposure = pd.Series(np.nan, index=idx)
    n = len(idx)
    # current chosen rule per regime code; recomputed at rebalances
    choice: dict[int, str | None] = {c: None for c in range(len(names))}
    picks_log: list[tuple] = []

    for i in range(n):
        if i % rebalance == 0:
            # recompute the best trailing rule per regime using data < i only
            hist = slice(0, i)
            lab_hist = lab.iloc[hist]
            for code in range(len(names)):
                in_reg = (lab_hist == code)
                if in_reg.sum() < min_obs:
                    choice[code] = "hold"  # default until enough regime history
                    continue
                best_rn, best_sh = "hold", -np.inf
                for rn in rule_names:
                    pnl = rule_pnl[rn].iloc[hist][in_reg.values]
                    # only the most recent `lookback` in-regime obs
                    pnl = pnl.iloc[-lookback:]
                    if len(pnl) < min_obs:
                        continue
                    sh = sharpe_ratio(pnl.values)
                    if sh > best_sh:
                        best_sh, best_rn = sh, rn
                choice[code] = best_rn
            picks_log.append((idx[i], dict(choice)))
        code_now = lab.iloc[i]
        if pd.isna(code_now):
            continue
        rn = choice.get(int(code_now)) or "hold"
        exposure.iloc[i] = float(menu[rn].reindex(idx).iloc[i])

    # shift by one (use the label/selection known at the prior close)
    exp = exposure.shift(1).reindex(idx).fillna(0.0).clip(0, 1)
    return exp, {"picks": picks_log, "n_rebalances": len(picks_log)}


# ════════════════════════════════════════════════════════════════════
# Metrics + placebo
# ════════════════════════════════════════════════════════════════════

@dataclass
class Row:
    name: str
    asset: str
    method: str
    ann_return: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    pct_in_market: float
    n_switches: int
    placebo_p: float | None
    daily: pd.Series | None = None


def _result_to_row(name, asset, method, r, placebo_p_val=None) -> Row:
    return Row(name=name, asset=asset, method=method,
               ann_return=r.ann_return, sharpe=r.sharpe, sortino=r.sortino,
               max_dd=r.max_dd, calmar=r.calmar,
               pct_in_market=r.pct_in_market, n_switches=r.n_switches,
               placebo_p=placebo_p_val, daily=r.daily)


def regime_shuffle_placebo(target: pd.Series, labels: pd.Series, names: list[str],
                           menu: dict[str, pd.Series], mapping: dict[str, str],
                           real: object, *, n: int = 200, seed: int = 0,
                           fee_bps_oneway: float = 5.0,
                           metric: str = "sharpe") -> float:
    """Circular-shift the REGIME LABELS, rebuild exposure under the same mapping,
    re-run. p = share of shuffles whose Sharpe >= the real strategy.

    This is the right placebo for switching: it scrambles *which regime is when*
    while preserving the menu rules and mapping, so a low p means the regime
    *timing* (not the menu rules themselves) carries the edge.
    """
    rng = np.random.default_rng(seed)
    lab = labels.reindex(target.index)
    real_v = getattr(real, metric)
    arr = lab.values.astype(float)
    hits = 0
    for _ in range(n):
        k = int(rng.integers(30, len(arr) - 30))
        shifted = pd.Series(np.roll(arr, k), index=target.index)
        exp = switch_exposure(shifted, names, mapping, menu)
        r = backtest_rule(target, exp, fee_bps_oneway=fee_bps_oneway)
        hits += getattr(r, metric) >= real_v
    return hits / n


def adaptive_shuffle_placebo(target: pd.Series, labels: pd.Series, names: list[str],
                             menu: dict[str, pd.Series], real: object, *,
                             n: int = 100, seed: int = 0,
                             fee_bps_oneway: float = 5.0, metric: str = "sharpe",
                             **adapt_kw) -> float:
    """Same idea for the adaptive method: shuffle regime labels, re-run the
    full causal adaptive selection on the scrambled labels."""
    rng = np.random.default_rng(seed)
    lab = labels.reindex(target.index)
    real_v = getattr(real, metric)
    arr = lab.values.astype(float)
    hits = 0
    for _ in range(n):
        k = int(rng.integers(30, len(arr) - 30))
        shifted = pd.Series(np.roll(arr, k), index=target.index)
        exp, _ = adaptive_exposure(shifted, names, menu, target, **adapt_kw)
        r = backtest_rule(target, exp, fee_bps_oneway=fee_bps_oneway)
        hits += getattr(r, metric) >= real_v
    return hits / n


# ════════════════════════════════════════════════════════════════════
# Pre-registered economic mappings (FIXED before results)
# ════════════════════════════════════════════════════════════════════

PRE_REGISTERED = {
    "drawdown":    {"at_highs": "hold", "in_drawdown": "flat"},
    "drawdown_mr": {"at_highs": "hold", "in_drawdown": "mean_revert"},
    "trending":    {"trending": "momentum", "ranging": "mean_revert"},
    "trending_alt": {"trending": "trend_50", "ranging": "flat"},
    "corr":        {"high_corr": "flat", "low_corr": "hold"},
    "corr_vt":     {"high_corr": "vol_target", "low_corr": "hold"},
    "breadth":     {"high_breadth": "hold", "low_breadth": "flat"},
    "breadth_mr":  {"high_breadth": "hold", "low_breadth": "mean_revert"},
}

# which regime builder each mapping uses
MAPPING_REGIME = {
    "drawdown": "drawdown", "drawdown_mr": "drawdown",
    "trending": "trending", "trending_alt": "trending",
    "corr": "corr", "corr_vt": "corr",
    "breadth": "breadth", "breadth_mr": "breadth",
}


# ════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════

def run(returns: pd.DataFrame, *, fee_bps_oneway: float = 5.0,
        n_placebo: int = 200, n_placebo_adaptive: int = 80,
        dd_threshold: float = 0.10):
    """Build regimes, run both assignment methods on BTC and basket, placebo."""
    idx = returns.index
    btc = returns["btc"]
    basket = returns.mean(axis=1)
    basket.name = "basket"

    # regime label sets (causal)
    regimes = {
        "drawdown": regime_drawdown(btc, dd_threshold),
        "trending": regime_trending(btc, 20, 0.50),
        "corr": regime_correlation(returns, 30, 0.70),
        "breadth": regime_breadth(returns, 50, 0.50),
    }

    targets = {"BTC": btc, "basket": basket}
    menus = {"BTC": build_menu(btc), "basket": build_menu(basket)}

    # benchmarks
    bh = {t: backtest_rule(s, pd.Series(1.0, index=idx),
                           fee_bps_oneway=fee_bps_oneway)
          for t, s in targets.items()}
    # unconditional trend_50 benchmark on each target
    trend50 = {t: backtest_rule(s, build_menu(s)["trend_50"],
                                fee_bps_oneway=fee_bps_oneway)
               for t, s in targets.items()}

    rows: list[Row] = []

    # ── 1. PRE-REGISTERED mappings ──────────────────────────────────
    for mname, mapping in PRE_REGISTERED.items():
        rname = MAPPING_REGIME[mname]
        labels, names = regimes[rname]
        for tname, series in targets.items():
            menu = menus[tname]
            exp = switch_exposure(labels, names, mapping, menu)
            r = backtest_rule(series, exp, fee_bps_oneway=fee_bps_oneway)
            r = replace(r, name=mname)
            pp = None
            base = bh[tname]
            if r.sharpe > base.sharpe or r.calmar > base.calmar:
                pp = regime_shuffle_placebo(
                    series, labels, names, menu, mapping, r,
                    n=n_placebo, fee_bps_oneway=fee_bps_oneway)
            rows.append(_result_to_row(f"prereg:{mname}", tname,
                                       "pre-registered", r, pp))
            logger.info("prereg %-14s %-6s ann=%+.0f%% sh=%.2f maxDD=%.0f%% "
                        "cal=%.2f in=%.0f%% placebo=%s", mname, tname,
                        r.ann_return * 100, r.sharpe, r.max_dd * 100, r.calmar,
                        r.pct_in_market * 100,
                        f"{pp:.2f}" if pp is not None else "-")

    # ── 2. ADAPTIVE causal selection ────────────────────────────────
    for rname, (labels, names) in regimes.items():
        for tname, series in targets.items():
            menu = menus[tname]
            exp, diag = adaptive_exposure(labels, names, menu, series)
            r = backtest_rule(series, exp, fee_bps_oneway=fee_bps_oneway)
            r = replace(r, name=rname)
            pp = None
            base = bh[tname]
            if r.sharpe > base.sharpe or r.calmar > base.calmar:
                pp = adaptive_shuffle_placebo(
                    series, labels, names, menu, r,
                    n=n_placebo_adaptive, fee_bps_oneway=fee_bps_oneway)
            rows.append(_result_to_row(f"adaptive:{rname}", tname,
                                       "adaptive", r, pp))
            logger.info("adapt  %-14s %-6s ann=%+.0f%% sh=%.2f maxDD=%.0f%% "
                        "cal=%.2f in=%.0f%% rebals=%d placebo=%s", rname, tname,
                        r.ann_return * 100, r.sharpe, r.max_dd * 100, r.calmar,
                        r.pct_in_market * 100, diag["n_rebalances"],
                        f"{pp:.2f}" if pp is not None else "-")

    # per-regime dwell / switches diagnostics
    dwell = {}
    for rname, (labels, names) in regimes.items():
        vc = labels.value_counts(normalize=True).sort_index()
        switches = int((labels.diff().abs() > 0).sum())
        dwell[rname] = {
            "names": names,
            "share": {names[int(k)]: float(v) for k, v in vc.items()
                      if int(k) < len(names)},
            "switches": switches,
        }

    return rows, bh, trend50, dwell


# ════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════

def render_markdown(rows: list[Row], bh: dict, trend50: dict, dwell: dict,
                    args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# Custom-regime DAG switching (regime → per-regime rule → return)\n"]
    L.append("Each strategy assigns a **different** exposure rule per regime "
             "(menu: hold / trend_50 / vol_target / flat / mean_revert / "
             "momentum). Two assignment methods: a **pre-registered** economic "
             "mapping fixed before results, and an **adaptive** causal selection "
             "(at each rebalance, pick the menu rule with best trailing-window "
             "Sharpe in the current regime, past data only). Exposure applied to "
             f"BTC and the equal-weight majors basket; decision at t close → t+1 "
             f"return, {args['fee_bps_oneway']:.0f}bp one-way cost.\n")
    L.append(f"- Universe: `{', '.join(MAJORS)}`")
    L.append(f"- Window `{args['start']}` → `{args['end']}` "
             f"({args['n_days']} days)")
    L.append(f"- Drawdown regime threshold: {args['dd_threshold']:.0%}")
    L.append(f"- Run UTC: "
             f"`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    # Benchmarks
    L.append("## Benchmarks\n")
    L.append("| Benchmark | Ann.ret | Sharpe | Sortino | MaxDD | Calmar |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for tname in ("BTC", "basket"):
        b = bh[tname]
        L.append(f"| BH {tname} | {b.ann_return:+.0%} | {b.sharpe:.2f} | "
                 f"{b.sortino:.2f} | {b.max_dd:.0%} | {b.calmar:.2f} |")
        t = trend50[tname]
        L.append(f"| trend_50 on {tname} | {t.ann_return:+.0%} | {t.sharpe:.2f} "
                 f"| {t.sortino:.2f} | {t.max_dd:.0%} | {t.calmar:.2f} |")
    L.append("")

    # Regime diagnostics
    L.append("## Regime dwell / switches (full-sample label distribution)\n")
    L.append("| Regime def | States (share) | Switches |")
    L.append("|---|---|---:|")
    for rname, d in dwell.items():
        shares = ", ".join(f"{k} {v:.0%}" for k, v in d["share"].items())
        L.append(f"| {rname} | {shares} | {d['switches']} |")
    L.append("")

    # Main results
    L.append("## Results — switching strategies\n")
    L.append("Sorted within asset by Sharpe. `placebo p` = regime-shuffle "
             "(circular-shift the regime labels, re-run under the same mapping); "
             "high p ⇒ regime *timing* carries no edge. `vs trend_50` compares "
             "Sharpe to the unconditional trend_50 benchmark on the same asset.\n")
    L.append("| Strategy | Method | Asset | Ann.ret | Sharpe | Sortino | MaxDD | "
             "Calmar | %in | switches | placebo p | vs trend_50 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    def sort_key(r: Row):
        return (0 if r.asset == "BTC" else 1, -r.sharpe)

    for r in sorted(rows, key=sort_key):
        pps = f"{r.placebo_p:.2f}" if r.placebo_p is not None else "—"
        t_sh = trend50[r.asset].sharpe
        delta = r.sharpe - t_sh
        mark = "beats" if delta > 0.02 else ("ties" if abs(delta) <= 0.02
                                             else "below")
        L.append(f"| {r.name} | {r.method} | {r.asset} | {r.ann_return:+.0%} | "
                 f"{r.sharpe:.2f} | {r.sortino:.2f} | {r.max_dd:.0%} | "
                 f"{r.calmar:.2f} | {r.pct_in_market:.0%} | {r.n_switches} | "
                 f"{pps} | {mark} ({delta:+.2f}) |")
    L.append("")

    # Verdict
    L.append("## Verdict\n")
    L.append(f"- **BH BTC**: Sharpe {bh['BTC'].sharpe:.2f}, maxDD "
             f"{bh['BTC'].max_dd:.0%}. **trend_50 (BTC)**: Sharpe "
             f"{trend50['BTC'].sharpe:.2f}. **BH basket**: Sharpe "
             f"{bh['basket'].sharpe:.2f}. **trend_50 (basket)**: Sharpe "
             f"{trend50['basket'].sharpe:.2f}.\n")

    # survivors: beat trend_50 on the same asset AND survive shuffle placebo
    survivors = [r for r in rows
                 if r.placebo_p is not None and r.placebo_p < 0.10
                 and r.sharpe > trend50[r.asset].sharpe + 0.02]
    if survivors:
        L.append("**Strategies that beat trend_50 (same asset) on Sharpe AND "
                 "survive the regime-shuffle placebo (p<0.10):**")
        for r in sorted(survivors, key=lambda r: -r.sharpe):
            L.append(f"- **{r.name}** ({r.method}, {r.asset}) — Sharpe "
                     f"{r.sharpe:.2f} vs trend_50 {trend50[r.asset].sharpe:.2f}, "
                     f"maxDD {r.max_dd:.0%}, placebo p={r.placebo_p:.2f}")
    else:
        beat_trend = [r for r in rows
                      if r.sharpe > trend50[r.asset].sharpe + 0.02]
        L.append("**No custom-regime switching strategy both beats trend_50 on "
                 "Sharpe (same asset) and survives the regime-shuffle placebo.** "
                 f"{len(beat_trend)} strategy/asset combo(s) edge out trend_50 on "
                 "Sharpe, but either fail the shuffle placebo (their edge is "
                 "reproducible with scrambled regime labels ⇒ it comes from the "
                 "menu rules / average exposure, not regime *timing*) or do not "
                 "clear it cleanly. The regime label adds churn, not skill.")
    L.append("")

    L.append("## Caveats\n")
    L.append("- **One cycle.** 2021–2025 is a single bull→bear→recovery; any "
             "regime gate that helps does so largely by truncating the *one* "
             "2022 bear. The placebo validates timing *within* this sample, not "
             "across regime change.")
    L.append("- **Mechanically endogenous regimes.** Drawdown, efficiency "
             "ratio, correlation and breadth are all derived from the *same* "
             "prices being traded, so part of any edge is tautological (a price "
             "below its peak both defines 'drawdown' and is a recent loss).")
    L.append("- **Multiple testing.** 8 pre-registered mappings × 2 assets + 4 "
             "adaptive regimes × 2 assets were tried; the best in-sample numbers "
             "are upward-biased. The shuffle placebo guards against regime-timing "
             "luck on a *fixed* mapping, not against having picked the lucky "
             "mapping/regime.")
    L.append("- **Funding regime omitted from the headline.** Funding data only "
             "starts 2023-11 (~2 yr), too short for a 252-day rolling-quantile "
             "regime over the full window; run `--with-funding` to include it on "
             "the 2023-11+ subsample, but treat it as exploratory.")
    L.append("- **Stables = 0%.** Risk-off legs earn 0% (no stable yield "
             "modeled); a real cash yield would only lift the flat-rule regimes "
             "slightly and does not change rankings vs the benchmarks.")
    L.append("- **No leverage.** Exposure capped at 1, so every overlay can only "
             "de-risk — structurally disadvantaged on raw return in a bull "
             "market.")
    return "\n".join(L)


# ── optional funding regime (2023-11+ subsample, exploratory) ───────

def load_funding(db_path: str = "causal_portfolio/data/cpcm_local.duckdb"
                 ) -> pd.Series | None:
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        fdf = con.sql("""
            select cast(time at time zone 'UTC' as date) as day,
                   avg(value) as f
            from asset_metrics_best where metric='funding_rate_8h'
            group by 1 order by 1""").df()
        con.close()
        return pd.Series(fdf["f"].values,
                         index=pd.to_datetime(fdf["day"])).sort_index()
    except Exception as e:
        logger.info("funding unavailable: %s", e)
        return None


def regime_funding(funding: pd.Series, idx: pd.DatetimeIndex,
                   q: float = 0.50) -> tuple[pd.Series, list[str]]:
    """Aggregate perp funding elevated (>= trailing median) vs low/negative.

    Elevated funding = crowded longs / risk-on positioning (often precedes
    pullbacks); low/negative = risk-off. Causal trailing-median threshold.
    """
    f = funding.reindex(idx).ffill()
    thresh = f.rolling(180, min_periods=30).quantile(q)
    lab = (f >= thresh).astype(int)  # 0 low_funding, 1 high_funding
    return lab, ["low_funding", "high_funding"]


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

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
    p.add_argument("--n-placebo-adaptive", type=int, default=80)
    p.add_argument("--dd-threshold", type=float, default=0.10)
    p.add_argument("--with-funding", action="store_true",
                   help="also test a funding regime on the 2023-11+ subsample")
    p.add_argument("--out",
                   default="causal_portfolio/docs/regime_switch_custom.md")
    args = p.parse_args()

    loader = get_loader()
    returns = loader.load_returns(MAJORS, args.start, args.end)
    returns.index = pd.to_datetime(returns.index)
    returns = returns[[f"{a}_return" for a in MAJORS]]
    returns.columns = MAJORS

    rows, bh, trend50, dwell = run(
        returns, fee_bps_oneway=args.fee_bps_oneway,
        n_placebo=args.n_placebo, n_placebo_adaptive=args.n_placebo_adaptive,
        dd_threshold=args.dd_threshold)

    # optional funding regime (exploratory, short subsample)
    if args.with_funding:
        funding = load_funding()
        if funding is not None:
            sub = returns.loc[returns.index >= "2023-11-01"]
            btc_sub = sub["btc"]
            basket_sub = sub.mean(axis=1)
            f_labels, f_names = regime_funding(funding, sub.index, 0.50)
            f_map = {"high_funding": "flat", "low_funding": "hold"}
            for tname, series in (("BTC", btc_sub), ("basket", basket_sub)):
                menu = build_menu(series)
                exp = switch_exposure(f_labels, f_names, f_map, menu)
                r = backtest_rule(series, exp, fee_bps_oneway=args.fee_bps_oneway)
                logger.info("FUNDING(2023-11+) %-6s ann=%+.0f%% sh=%.2f "
                            "maxDD=%.0f%% in=%.0f%%", tname, r.ann_return * 100,
                            r.sharpe, r.max_dd * 100, r.pct_in_market * 100)

    argd = vars(args) | {"n_days": int(len(returns))}
    md = render_markdown(rows, bh, trend50, dwell, argd)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
