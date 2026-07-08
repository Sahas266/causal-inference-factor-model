"""Weekly/monthly-horizon factor -> return: the longer-horizon experiment.

The daily-horizon program ended in a structural null (on-chain factors are
downstream of daily returns, `causal_directions_summary.md`) and the intraday
escape hatch is closed (`funding_intraday.md`). The remaining open avenue is
longer horizons: maybe factor state predicts the NEXT WEEK's return even
though it says nothing about tomorrow's.

Pre-registered design (kept deliberately small — one spec per factor, no
sign or lookback sweeps):

1. Resample daily factors to weekly (Friday close, `.last()`), align each
   factor value at week t with the forward 1-week return over (t, t+1w] —
   non-overlapping. Two targets: BTC and the equal-weight 9-asset basket.
2. For each of the 7 core CPCM factors (skipping any all-NaN) plus vixcls
   and t10y2y: OLS of the forward weekly return on the (full-sample-affine
   z-scored) factor, Newey-West HAC t-stats (maxlags 4). <=18 regressions;
   Bonferroni applied across all of them.
3. Monthly variant: same machinery on month-end resampling (~60 obs —
   reported but flagged underpowered).
4. Trading overlay ONLY for factors whose weekly HAC p < 0.05 pre-Bonferroni:
   long/flat gate with the direction AND the gate center estimated on the
   FIRST HALF of the weekly grid only, evaluated on the SECOND HALF only.
   5bp one-way costs per exposure change, weekly rebalance, vs buy-and-hold
   on the same second-half grid; circular-shift placebo (n=200) when the
   overlay beats BH on Sharpe (annualization 52).

Causality contract (unit-tested in tests/test_weekly_horizon.py): the factor
value at week t uses only daily data <= t; the return earned is (t, t+1w].
Factor z-scores are full-sample affine (documented acceptable — absorbed by
the regression); every threshold used in trading is first-half-only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_portfolio.backtest.metrics import max_drawdown, sharpe_ratio
from causal_portfolio.data import get_loader
from causal_portfolio.factors.builder import (
    FACTOR_SOURCE_ASSETS,
    GLOBAL_FACTORS,
    MACRO_SERIES,
    PANEL_METRICS,
    build_all_factors,
)

DEFAULT_ASSETS = ["btc", "eth", "sol", "bnb", "avax", "uni", "aave", "link", "doge"]
MACRO_TESTED = ["vixcls", "t10y2y"]
START, END = "2021-01-01", "2025-12-31"
FEE_BPS_ONEWAY = 5.0
PLACEBO_N = 200
HAC_MAXLAGS = 4
PERIODS_PER_YEAR = {"W-FRI": 52, "ME": 12}


# ── data / alignment ─────────────────────────────────────────────────

def load_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily factors (build_all_factors) and daily per-asset returns."""
    loader = get_loader()
    panel_assets = list(dict.fromkeys(DEFAULT_ASSETS + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, START, END)
    returns = loader.load_returns(DEFAULT_ASSETS, START, END)
    returns.columns = [c.removesuffix("_return") for c in returns.columns]
    macro = loader.load_macro(MACRO_SERIES, START, END)
    factors = build_all_factors(panel, macro)
    return factors, returns


def build_grid(
    factors: pd.DataFrame, returns: pd.DataFrame, freq: str = "W-FRI"
) -> pd.DataFrame:
    """One row per period-end t: factor state using only daily data <= t,
    forward returns over (t, t+1 period] for BTC and the EW basket."""
    f = factors.resample(freq).last()  # last daily observation <= period end t

    per_asset = (1.0 + returns).resample(freq).prod() - 1.0
    n_obs = returns.notna().resample(freq).sum()
    per_asset = per_asset.where(n_obs > 0)  # empty bins compound to 0 -> NaN
    basket = per_asset.mean(axis=1)

    grid = f.copy()
    grid["fwd_btc"] = per_asset["btc"].shift(-1)
    grid["fwd_basket"] = basket.shift(-1)
    return grid


# ── regression ───────────────────────────────────────────────────────

def hac_reg(factor: pd.Series, fwd: pd.Series, maxlags: int = HAC_MAXLAGS) -> dict:
    """Forward return ~ standardized factor, Newey-West HAC. Beta in bps/sigma.

    Standardization is full-sample affine (documented acceptable): it only
    rescales beta, never the t/p."""
    data = pd.DataFrame({"x": factor, "y": fwd}).dropna()
    if len(data) < 20 or data["x"].std() < 1e-12:
        return {"n": len(data), "beta_bps": np.nan, "t": np.nan, "p": np.nan}
    x = (data["x"] - data["x"].mean()) / data["x"].std()
    fit = sm.OLS(data["y"], sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return {
        "n": int(fit.nobs),
        "beta_bps": float(fit.params["x"] * 1e4),
        "t": float(fit.tvalues["x"]),
        "p": float(fit.pvalues["x"]),
    }


def run_regressions(grid: pd.DataFrame, factor_names: list[str]) -> list[dict]:
    rows = []
    for f in factor_names:
        for target in ("fwd_btc", "fwd_basket"):
            rows.append(
                {"factor": f, "target": target, **hac_reg(grid[f], grid[target])}
            )
    return rows


# ── trading overlay (split-sample) ──────────────────────────────────

@dataclass
class RuleResult:
    name: str
    ann_return: float
    sharpe: float
    max_dd: float
    pct_in_market: float
    n_switches: int
    net: pd.Series


def backtest(fwd: pd.Series, exposure: pd.Series, ppy: int, name: str) -> RuleResult:
    e = exposure.astype(float).clip(0, 1).fillna(0.0)
    net = (e * fwd - e.diff().abs().fillna(e.abs()) * FEE_BPS_ONEWAY / 1e4).dropna()
    return RuleResult(
        name=name,
        ann_return=float(net.mean() * ppy),
        sharpe=float(sharpe_ratio(net.values, annualization=ppy)),
        max_dd=float(max_drawdown(net.values)),
        pct_in_market=float((e.loc[net.index] > 0.01).mean()),
        n_switches=int((e.loc[net.index].diff().abs() > 0.01).sum()),
        net=net,
    )


def placebo_p(fwd: pd.Series, exposure: pd.Series, real: RuleResult,
              ppy: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    e = exposure.astype(float).clip(0, 1).fillna(0.0).values
    hits = 0
    for _ in range(PLACEBO_N):
        k = int(rng.integers(8, len(e) - 8))
        shifted = pd.Series(np.roll(e, k), index=exposure.index)
        hits += backtest(fwd, shifted, ppy, "shift").sharpe >= real.sharpe
    return hits / PLACEBO_N


def run_overlay(grid: pd.DataFrame, factor: str, target: str) -> dict:
    """Long/flat gate. Sign and gate center fit on the FIRST HALF of the
    weekly grid only; performance evaluated on the SECOND HALF only."""
    ppy = 52
    data = grid[[factor, target]].dropna()
    half = len(data) // 2
    first, second = data.iloc[:half], data.iloc[half:]

    sign = np.sign(hac_reg(first[factor], first[target])["beta_bps"]) or 1.0
    center = first[factor].mean()  # first-half-only threshold

    exposure = (sign * (second[factor] - center) >= 0).astype(float)
    rule = backtest(second[target], exposure, ppy, f"{factor}_gate")
    bh = backtest(second[target], pd.Series(1.0, index=second.index), ppy, "bh")
    p = placebo_p(second[target], exposure, rule, ppy) if rule.sharpe > bh.sharpe else None
    return {
        "factor": factor, "target": target, "sign": int(sign),
        "eval_start": str(second.index[0].date()),
        "eval_end": str(second.index[-1].date()),
        "rule": rule, "bh": bh, "placebo_p": p,
    }


# ── report ───────────────────────────────────────────────────────────

def render_regs(rows: list[dict], label: str, bonf_alpha: float) -> str:
    lines = [
        f"### {label} (Bonferroni threshold p < {bonf_alpha:.4f} for {len(rows)} tests)",
        "",
        "| factor | target | n | beta (bps/sigma) | HAC t | p | p<0.05 | Bonf |",
        "|---|---|--:|--:|--:|--:|:-:|:-:|",
    ]
    for r in rows:
        sig = "*" if r["p"] < 0.05 else ""
        bonf = "**" if r["p"] < bonf_alpha else ""
        lines.append(
            f"| {r['factor']} | {r['target'][4:]} | {r['n']} "
            f"| {r['beta_bps']:+.1f} | {r['t']:+.2f} | {r['p']:.3f} | {sig} | {bonf} |"
        )
    return "\n".join(lines)


def render_overlay(o: dict) -> str:
    lines = [
        f"### overlay: {o['factor']} -> {o['target'][4:]}  "
        f"(sign={o['sign']:+d} from first half; eval {o['eval_start']} -> {o['eval_end']})",
        "",
        "| rule | ann | sharpe | maxDD | in-mkt | switches | placebo p |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for r in (o["bh"], o["rule"]):
        p = o["placebo_p"] if r is o["rule"] else None
        lines.append(
            f"| {r.name} | {r.ann_return:+.1%} | {r.sharpe:.2f} | {r.max_dd:.1%} "
            f"| {r.pct_in_market:.0%} | {r.n_switches} "
            f"| {'-' if p is None else f'{p:.2f}'} |"
        )
    return "\n".join(lines)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    factors, returns = load_daily()
    tested = [f for f in GLOBAL_FACTORS + MACRO_TESTED if f in factors.columns]
    skipped = [f for f in GLOBAL_FACTORS + MACRO_TESTED if f not in factors.columns]
    if skipped:
        print(f"skipped (all-NaN): {skipped}\n")

    weekly = build_grid(factors, returns, "W-FRI")
    monthly = build_grid(factors, returns, "ME")
    print(f"weekly grid: {weekly.index[0].date()} -> {weekly.index[-1].date()}, "
          f"{len(weekly)} weeks; monthly: {len(monthly)} months\n")

    wk_rows = run_regressions(weekly, tested)
    print(render_regs(wk_rows, "Weekly", 0.05 / len(wk_rows)))
    print()
    mo_rows = run_regressions(monthly, tested)
    print(render_regs(mo_rows, "Monthly (underpowered, ~60 obs)", 0.05 / len(mo_rows)))
    print()

    hits = {(r["factor"], r["target"]) for r in wk_rows if r["p"] < 0.05}
    if not hits:
        print("No weekly regression reaches p < 0.05 pre-Bonferroni. "
              "Per the pre-registration, no trading overlays are run.")
    for factor, target in sorted(hits):
        # Stability diagnostic for every hit: persistent-level regressors can
        # produce large HAC t-stats spuriously; a real effect should hold in
        # both halves with the same sign.
        data = weekly[[factor, target]].dropna()
        half = len(data) // 2
        h1 = hac_reg(data[factor].iloc[:half], data[target].iloc[:half])
        h2 = hac_reg(data[factor].iloc[half:], data[target].iloc[half:])
        rho = float(data[factor].autocorr())
        print(f"diagnostic {factor}->{target[4:]}: weekly AR(1)={rho:.2f}; "
              f"1st half beta={h1['beta_bps']:+.1f} (t={h1['t']:+.2f}), "
              f"2nd half beta={h2['beta_bps']:+.1f} (t={h2['t']:+.2f})")
        print(render_overlay(run_overlay(weekly, factor, target)))
        print()


if __name__ == "__main__":
    main()
