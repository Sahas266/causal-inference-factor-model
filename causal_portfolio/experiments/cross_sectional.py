"""Direction 1 — cross-sectional, market-neutral causal test.

Daily AGGREGATE crypto returns are ~one market factor plus noise, and nothing
in this repo beats holding that factor (BH BTC). This experiment changes the
question: do the CPCM factors predict RELATIVE returns once the market factor
is hedged out?

Design (causal end-to-end):
  - Residual returns: resid_{i,t} = r_{i,t} - beta_{i,t-1} * r_{btc,t}, with
    beta estimated on a trailing window ENDING t-1 (no look-ahead in the
    hedge ratio).
  - Prediction: per-asset OLS of resid on LAG-1 factors over a trailing
    train window; predictions use factors known at t.
  - Portfolio: cross-sectionally demeaned predictions, scaled to gross 1 —
    dollar-neutral long/short of beta-hedged residuals.
  - The factors are GLOBAL (identical across assets each day), so the
    cross-section is differentiated only through per-asset loadings: this
    tests whether differential factor SENSITIVITIES predict relative returns.

Honesty checks:
  - Newey-West t-stat on the mean L/S return.
  - Placebo: N circular time-shifts of the factor panel -> empirical p-value
    for the realized Sharpe (kills any "structure-free" artifact).
  - Walk-forward folds, not one number.

Run:  python -m causal_portfolio.experiments.cross_sectional
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import ANNUALIZATION

logger = logging.getLogger("cpcm.experiments.cross_sectional")


# ── residual returns ────────────────────────────────────────────────

def beta_hedged_residuals(
    returns: pd.DataFrame, market_col: str = "btc_return",
    beta_window: int = 90,
) -> pd.DataFrame:
    """r_i - beta_i * r_mkt with beta from a trailing window ending t-1."""
    mkt = returns[market_col]
    out = {}
    for col in returns.columns:
        if col == market_col:
            continue
        cov = returns[col].rolling(beta_window).cov(mkt)
        var = mkt.rolling(beta_window).var()
        beta = (cov / var).shift(1)          # known at t-1
        out[col] = returns[col] - beta * mkt
    return pd.DataFrame(out, index=returns.index)


# ── walk-forward L/S backtest on residuals ──────────────────────────

@dataclass
class XSResult:
    ls_returns: pd.Series       # daily L/S portfolio return (residual space)
    sharpe: float
    nw_tstat: float
    mean_ic: float
    ic_tstat: float
    fold_sharpes: list[float]


def _newey_west_t(x: np.ndarray, lags: int = 5) -> float:
    """t-stat of mean(x) with Newey-West (Bartlett) HAC variance."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return 0.0
    xc = x - x.mean()
    s = xc @ xc / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (xc[k:] @ xc[:-k]) / n
    se = np.sqrt(s / n)
    return float(x.mean() / (se + 1e-30))


def run_xs(
    residuals: pd.DataFrame, factors: pd.DataFrame, *,
    train_window: int = 252, rebalance_freq: int = 5, n_folds: int = 4,
) -> XSResult:
    """Walk-forward long/short on beta-hedged residuals from lag-1 factors."""
    F_lag = factors.shift(1)
    joined = pd.concat([residuals, F_lag], axis=1).dropna()
    R = joined[residuals.columns].values     # (T, n)
    F = joined[factors.columns].values       # (T, m) — already lagged
    T, n = R.shape

    ls = np.full(T, np.nan)
    w = np.zeros(n)
    for t in range(train_window, T):
        if (t - train_window) % rebalance_freq == 0:
            R_tr, F_tr = R[t - train_window:t], F[t - train_window:t]
            X = np.column_stack([np.ones(train_window), F_tr])
            beta, *_ = np.linalg.lstsq(X, R_tr, rcond=None)  # (m+1, n)
            pred = np.concatenate([[1.0], F[t]]) @ beta       # (n,)
            pred = pred - pred.mean()                        # dollar-neutral
            gross = np.abs(pred).sum()
            w = pred / gross if gross > 1e-12 else np.zeros(n)
        ls[t] = float(w @ R[t])

    ls_s = pd.Series(ls, index=joined.index).dropna()
    sharpe = float(ls_s.mean() / (ls_s.std() + 1e-15) * np.sqrt(ANNUALIZATION))
    nw_t = _newey_west_t(ls_s.values)

    # Daily cross-sectional IC of prediction sign vs realized residual:
    # recompute predictions on rebalance days for the IC series.
    ics = []
    for t in range(train_window, T, rebalance_freq):
        R_tr, F_tr = R[t - train_window:t], F[t - train_window:t]
        X = np.column_stack([np.ones(train_window), F_tr])
        beta, *_ = np.linalg.lstsq(X, R_tr, rcond=None)
        pred = np.concatenate([[1.0], F[t]]) @ beta
        fwd = R[t:min(t + rebalance_freq, T)].sum(axis=0)
        if np.std(pred) > 1e-12 and np.std(fwd) > 1e-12:
            ics.append(float(pd.Series(pred).corr(pd.Series(fwd),
                                                  method="spearman")))
    ics_a = np.asarray([i for i in ics if not np.isnan(i)])
    mean_ic = float(ics_a.mean()) if len(ics_a) else 0.0
    ic_t = float(mean_ic / (ics_a.std() / np.sqrt(len(ics_a)) + 1e-15)) \
        if len(ics_a) > 2 else 0.0

    folds = np.array_split(ls_s.values, n_folds)
    fold_sh = [float(np.mean(f) / (np.std(f) + 1e-15) * np.sqrt(ANNUALIZATION))
               for f in folds if len(f) > 20]
    return XSResult(ls_s, sharpe, nw_t, mean_ic, ic_t, fold_sh)


def placebo_pvalue(
    residuals: pd.DataFrame, factors: pd.DataFrame, real_sharpe: float, *,
    n_placebo: int = 50, seed: int = 0, **kw,
) -> tuple[float, list[float]]:
    """Empirical p-value of the realized Sharpe vs circular factor shifts.

    Circular shifts preserve each factor's autocorrelation and the
    cross-asset return structure while destroying any genuine factor-return
    alignment — a structure-respecting null.
    """
    rng = np.random.default_rng(seed)
    T = len(factors)
    sharpes = []
    for _ in range(n_placebo):
        k = int(rng.integers(60, T - 60))
        shifted = pd.DataFrame(np.roll(factors.values, k, axis=0),
                               index=factors.index, columns=factors.columns)
        res = run_xs(residuals, shifted, **kw)
        sharpes.append(res.sharpe)
    p = float(np.mean([abs(s) >= abs(real_sharpe) for s in sharpes]))
    return p, sharpes


# ── report ──────────────────────────────────────────────────────────

def render_markdown(full: XSResult, singles: dict[str, XSResult],
                    placebo_p: float, placebo_sharpes: list[float],
                    args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# Cross-sectional market-neutral test (Direction 1)\n"]
    L.append("Beta-hedged residual returns (trailing β known at t-1), "
             "predicted from lag-1 global factors via per-asset OLS, traded "
             "dollar-neutral. The market factor — the thing BH BTC owns — is "
             "hedged out, so this isolates whether differential factor "
             "sensitivities carry relative-return information.\n")
    L.append(f"- Assets: `{', '.join(args['assets'])}` (BTC = hedge leg)")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | train "
             f"{args['train_window']}d | rebalance {args['rebalance_freq']}d "
             f"| β window {args['beta_window']}d")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("## All factors jointly\n")
    L.append("| Metric | Value |")
    L.append("|---|---:|")
    L.append(f"| OOS L/S Sharpe | {full.sharpe:.3f} |")
    L.append(f"| Newey-West t (mean) | {full.nw_tstat:.2f} |")
    L.append(f"| Mean rank IC (5d fwd) | {full.mean_ic:.4f} |")
    L.append(f"| IC t-stat | {full.ic_tstat:.2f} |")
    L.append(f"| Fold Sharpes | {', '.join(f'{s:.2f}' for s in full.fold_sharpes)} |")
    L.append(f"| Placebo p-value ({len(placebo_sharpes)} circular shifts) "
             f"| {placebo_p:.2f} |")
    ps = np.asarray(placebo_sharpes)
    L.append(f"| Placebo Sharpe range | [{ps.min():.2f}, {ps.max():.2f}] "
             f"(median {np.median(ps):.2f}) |")
    L.append("")

    L.append("## Single-factor L/S\n")
    L.append("| Factor | Sharpe | NW t | mean IC | IC t |")
    L.append("|---|---:|---:|---:|---:|")
    for name, r in sorted(singles.items(), key=lambda kv: -kv[1].sharpe):
        L.append(f"| {name} | {r.sharpe:.3f} | {r.nw_tstat:.2f} | "
                 f"{r.mean_ic:.4f} | {r.ic_tstat:.2f} |")
    L.append("")

    sig = (abs(full.nw_tstat) > 2 and placebo_p < 0.05)
    L.append("## Verdict\n")
    if sig:
        L.append(f"The joint L/S is statistically distinguishable from the "
                 f"placebo null (NW t={full.nw_tstat:.2f}, placebo "
                 f"p={placebo_p:.2f}). Worth pursuing: check costs, "
                 f"capacity, and per-fold stability before any stronger claim.")
    else:
        L.append(f"**No reliable cross-sectional signal.** Joint L/S Sharpe "
                 f"{full.sharpe:.3f} (NW t={full.nw_tstat:.2f}) is "
                 f"indistinguishable from the circular-shift placebo "
                 f"(p={placebo_p:.2f}). Hedging out the market factor does "
                 f"not reveal relative-return predictability from the CPCM "
                 f"factor loadings at the daily horizon.")
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
    p.add_argument("--rebalance-freq", type=int, default=5)
    p.add_argument("--beta-window", type=int, default=90)
    p.add_argument("--n-placebo", type=int, default=50)
    p.add_argument("--out", default="causal_portfolio/docs/cross_sectional_causal.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, args.start, args.end)
    macro = loader.load_macro(MACRO_SERIES, args.start, args.end)
    returns = loader.load_returns(assets, args.start, args.end)
    factors = build_all_factors(panel, macro).dropna(axis=1, how="all")

    residuals = beta_hedged_residuals(returns, beta_window=args.beta_window)
    kw = dict(train_window=args.train_window,
              rebalance_freq=args.rebalance_freq)

    full = run_xs(residuals, factors, **kw)
    logger.info("joint L/S Sharpe=%.3f NW t=%.2f IC=%.4f",
                full.sharpe, full.nw_tstat, full.mean_ic)

    singles = {}
    for f in factors.columns:
        singles[f] = run_xs(residuals, factors[[f]], **kw)

    pp, ps = placebo_pvalue(residuals, factors, full.sharpe,
                            n_placebo=args.n_placebo, **kw)
    logger.info("placebo p=%.2f", pp)

    md = render_markdown(full, singles, pp, ps, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Sharpe={full.sharpe:.3f} NWt={full.nw_tstat:.2f} "
          f"placebo_p={pp:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
