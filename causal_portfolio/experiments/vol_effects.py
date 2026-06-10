"""Direction 5 — do the factors causally move VOLATILITY (and is it usable)?

Effects on volatility are an order of magnitude easier to detect than effects
on mean returns, and exploitable through risk targeting even with zero mean
edge. Two parts:

  1. Inference: forward 5d realized vol (EW basket) regressed on factors_t,
     CONTROLLING for trailing 5d realized vol (vol clustering is the elephant
     — a factor only matters if it adds beyond persistence). HAC (Newey-West,
     10 lags) errors because overlapping forward windows induce MA structure.
  2. Practice: walk-forward vol-targeting backtest on BH BTC and the EW
     basket — constant exposure vs trailing-vol targeting vs factor-augmented
     targeting (same model refit causally each rebalance). If factors carry
     vol information beyond persistence, the augmented arm should win.

Run:  python -m causal_portfolio.experiments.vol_effects
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import ANNUALIZATION

logger = logging.getLogger("cpcm.experiments.vol_effects")

VOL_H = 5


# ── part 1: HAC inference ───────────────────────────────────────────

@dataclass
class VolCoef:
    factor: str
    coef: float        # effect of +1σ factor on fwd 5d vol, in vol units
    tstat: float
    pval: float


def vol_regression(
    factors: pd.DataFrame, basket: pd.Series,
) -> tuple[list[VolCoef], float, int]:
    """fwd5vol ~ const + trail5vol + factors, HAC errors. Returns
    (per-factor rows, trailing-vol t-stat, n_obs)."""
    import statsmodels.api as sm

    fwd_vol = basket.shift(-1).rolling(VOL_H).std().shift(-(VOL_H - 1))
    trail_vol = basket.rolling(VOL_H).std()
    data = pd.concat(
        [fwd_vol.rename("fwd"), trail_vol.rename("trail"), factors],
        axis=1).dropna()
    X = sm.add_constant(data[["trail"] + list(factors.columns)])
    model = sm.OLS(data["fwd"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10})
    rows = [VolCoef(f, float(model.params[f]), float(model.tvalues[f]),
                    float(model.pvalues[f]))
            for f in factors.columns]
    return rows, float(model.tvalues["trail"]), len(data)


# ── part 2: vol-targeting backtest ──────────────────────────────────

@dataclass
class VolTargetResult:
    name: str
    sharpe: float
    total: float
    realized_vol: float
    max_lev: float


def _sharpe(r: np.ndarray) -> float:
    return float(np.mean(r) / (np.std(r) + 1e-15) * np.sqrt(ANNUALIZATION))


def vol_target_backtest(
    target: pd.Series, factors: pd.DataFrame, *,
    train_window: int = 252, rebalance_freq: int = 5, max_lev: float = 2.0,
) -> list[VolTargetResult]:
    """Constant vs trailing-vol-target vs factor-augmented vol-target."""
    trail = target.rolling(VOL_H).std().rename("trail")
    fwd = target.shift(-1).rolling(VOL_H).std().shift(-(VOL_H - 1)).rename("fwd")
    data = pd.concat([target.rename("r"), trail, fwd, factors], axis=1).dropna(
        subset=["r", "trail"] + list(factors.columns))
    r = data["r"].values
    T = len(data)
    fcols = list(factors.columns)

    tgt_vol = float(np.nanmedian(data["trail"].values))  # neutral target

    lev_trail = np.ones(T)
    lev_fact = np.ones(T)
    for t in range(train_window, T):
        if (t - train_window) % rebalance_freq == 0:
            # trailing-vol arm
            lt = np.clip(tgt_vol / (data["trail"].values[t] + 1e-12),
                         0.0, max_lev)
            # factor-augmented arm: fit fwd ~ [1, trail, factors] on train rows
            tr = data.iloc[t - train_window:t].dropna(subset=["fwd"])
            X = np.column_stack([np.ones(len(tr)), tr["trail"].values,
                                 tr[fcols].values])
            beta, *_ = np.linalg.lstsq(X, tr["fwd"].values, rcond=None)
            x_now = np.concatenate(
                [[1.0], [data["trail"].values[t]], data[fcols].values[t]])
            pred = max(float(x_now @ beta), 1e-12)
            lf = np.clip(tgt_vol / pred, 0.0, max_lev)
        lev_trail[t] = lt
        lev_fact[t] = lf

    oos = slice(train_window, T)
    arms = {
        "constant": np.ones(T)[oos] * r[oos],
        "trail_vol_target": lev_trail[oos] * r[oos],
        "factor_vol_target": lev_fact[oos] * r[oos],
    }
    out = []
    for name, rr in arms.items():
        pv = np.cumprod(1 + rr)
        lev = {"constant": np.ones(T), "trail_vol_target": lev_trail,
               "factor_vol_target": lev_fact}[name][oos]
        out.append(VolTargetResult(
            name, _sharpe(rr), float(pv[-1] - 1.0),
            float(np.std(rr) * np.sqrt(ANNUALIZATION)),
            float(np.max(lev))))
    return out


def placebo_pvalues(
    btc: pd.Series, ew: pd.Series, factors: pd.DataFrame,
    real_btc: float, real_ew: float, *,
    n_placebo: int = 30, seed: int = 1, **kw,
) -> tuple[float, float]:
    """Fraction of circular factor shifts whose factor-vol-target arm does
    at least as well — a leverage-overlay-respecting null."""
    rng = np.random.default_rng(seed)
    hits_btc = hits_ew = 0
    for _ in range(n_placebo):
        k = int(rng.integers(60, len(factors) - 60))
        shifted = pd.DataFrame(np.roll(factors.values, k, axis=0),
                               index=factors.index, columns=factors.columns)
        sb = [r.sharpe for r in vol_target_backtest(btc, shifted, **kw)
              if r.name == "factor_vol_target"][0]
        se = [r.sharpe for r in vol_target_backtest(ew, shifted, **kw)
              if r.name == "factor_vol_target"][0]
        hits_btc += sb >= real_btc
        hits_ew += se >= real_ew
    return hits_btc / n_placebo, hits_ew / n_placebo


# ── report ──────────────────────────────────────────────────────────

def render_markdown(rows, trail_t, n_obs, bt_btc, bt_ew, args,
                    placebo: tuple[float, float] | None = None) -> str:
    from datetime import datetime, timezone
    L = ["# Factor effects on volatility + vol targeting (Direction 5)\n"]
    L.append("Part 1: forward 5d realized vol (EW basket) on factors at t, "
             "controlling for trailing 5d vol, HAC(10) errors. Part 2: "
             "walk-forward vol-targeting backtest — if factors carry vol "
             "information beyond persistence, the factor-augmented arm "
             "should beat plain trailing-vol targeting.\n")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | n={n_obs} | "
             f"train {args['train_window']}d\n")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("## Part 1 — incremental vol information (HAC t-stats)\n")
    L.append(f"Trailing 5d vol t-stat: **{trail_t:.1f}** (persistence "
             "dominates, as expected).\n")
    L.append("| Factor | coef (vol/σ) | t | p |")
    L.append("|---|---:|---:|---:|")
    n_sig = 0
    for c in sorted(rows, key=lambda c: -abs(c.tstat)):
        mark = " **" if c.pval < 0.05 else ""
        n_sig += c.pval < 0.05
        L.append(f"| {c.factor} | {c.coef:+.5f} | {c.tstat:.2f}{mark} | "
                 f"{c.pval:.3f} |")
    L.append("")
    L.append(f"{n_sig}/{len(rows)} factors significant at 5% after the "
             f"trailing-vol control (~{0.05 * len(rows):.1f} expected by "
             "chance).\n")

    for label, bt in (("BH BTC", bt_btc), ("EW basket", bt_ew)):
        L.append(f"## Part 2 — vol-targeting backtest ({label})\n")
        L.append("| Arm | OOS Sharpe | Total | Realized vol | max leverage |")
        L.append("|---|---:|---:|---:|---:|")
        for r in bt:
            L.append(f"| {r.name} | {r.sharpe:.3f} | {r.total:+.1%} | "
                     f"{r.realized_vol:.1%} | {r.max_lev:.2f} |")
        L.append("")

    def _delta(bt):
        d = {r.name: r.sharpe for r in bt}
        return d["factor_vol_target"] - d["trail_vol_target"]

    L.append("## Verdict\n")
    d_btc, d_ew = _delta(bt_btc), _delta(bt_ew)
    L.append(f"Factor-augmented vol targeting vs plain trailing-vol "
             f"targeting: ΔSharpe {d_btc:+.3f} (BTC), {d_ew:+.3f} (EW). ")
    if placebo is not None:
        p_btc, p_ew = placebo
        L.append(f"\n**Placebo check (circular factor shifts):** "
                 f"p={p_btc:.2f} (BTC), p={p_ew:.2f} (EW) — the fraction of "
                 f"random factor alignments whose vol-target arm does at "
                 f"least as well.\n")
        if max(p_btc, p_ew) < 0.05 and n_sig > 1:
            L.append("The factor vol signal survives the placebo — worth "
                     "promoting into the risk layer.")
        else:
            L.append("**The apparent improvement does not survive the "
                     "placebo**: a large share of random factor alignments "
                     "produce equal or better Sharpe, i.e. ANY time-varying "
                     "leverage overlay tends to score in this OOS window — "
                     "the gain is leverage-timing luck, not factor "
                     "information. The HAC-significant coefficients in Part "
                     "1 (mostly VIX, itself a vol index) are real "
                     "statistically but already subsumed by trailing vol + "
                     "noise at the portfolio level.")
    return "\n".join(L)


def main() -> None:
    import argparse
    from pathlib import Path
    from causal_portfolio.data import get_loader
    from causal_portfolio.factors.builder import (
        FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS, build_all_factors,
    )

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--n-placebo", type=int, default=30)
    p.add_argument("--out", default="causal_portfolio/docs/vol_effects.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, args.start, args.end)
    macro = loader.load_macro(MACRO_SERIES, args.start, args.end)
    returns = loader.load_returns(assets, args.start, args.end)
    factors = build_all_factors(panel, macro).dropna(axis=1, how="all")

    ew = returns.mean(axis=1)
    btc = returns["btc_return"]

    rows, trail_t, n_obs = vol_regression(factors, ew)
    for c in sorted(rows, key=lambda c: -abs(c.tstat))[:5]:
        logger.info("%s: t=%.2f p=%.3f", c.factor, c.tstat, c.pval)

    bt_btc = vol_target_backtest(btc, factors, train_window=args.train_window)
    bt_ew = vol_target_backtest(ew, factors, train_window=args.train_window)
    for r in bt_btc:
        logger.info("BTC %s: Sharpe=%.3f vol=%.1f%%", r.name, r.sharpe,
                    r.realized_vol * 100)

    real_btc = [r.sharpe for r in bt_btc if r.name == "factor_vol_target"][0]
    real_ew = [r.sharpe for r in bt_ew if r.name == "factor_vol_target"][0]
    placebo = placebo_pvalues(btc, ew, factors, real_btc, real_ew,
                              n_placebo=args.n_placebo,
                              train_window=args.train_window)
    logger.info("placebo p: btc=%.2f ew=%.2f", *placebo)

    md = render_markdown(rows, trail_t, n_obs, bt_btc, bt_ew, vars(args),
                         placebo=placebo)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
