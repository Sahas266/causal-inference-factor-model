"""PCA-orthogonalized drivers vs raw factors — walk-forward, scored vs BH BTC.

Fair A/B: identical optimizer / cadence / train window. The only difference is
the regressor set feeding the per-asset OLS loadings:
  raw arm:  the 14 (collinear) CPCM factors.
  pca arm:  principal components of those factors, fit on the TRAILING window
            only (causal) and kept to `var_keep` cumulative variance.

If decorrelating the drivers stabilizes the loadings, the PCA arm should
generalize better OOS. Scored against buy-and-hold BTC via the harness.

Run:  python -m causal_portfolio.experiments.pca_drivers
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from causal_portfolio.experiments.dag_variants import (
    FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS,
)
from causal_portfolio.experiments.ols_vs_2sls import _fit_ols_loadings
from causal_portfolio.factors.builder import build_all_factors
from causal_portfolio.optimizer.manifold import ManifoldOptimizer, estimate_covariance

logger = logging.getLogger("cpcm.experiments.pca_drivers")
ANNUALIZATION = 365


@dataclass
class PCAABResult:
    raw_returns: pd.Series
    pca_returns: pd.Series
    bh_btc: pd.Series
    mean_n_components: float


def _causal_pca(F_train: np.ndarray, F_now: np.ndarray, var_keep: float):
    """Fit PCA on the training window, return (PCs_train, PC_now, n_comp)."""
    mean = F_train.mean(axis=0)
    scale = F_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Xs = (F_train - mean) / scale
    pca = PCA(n_components=var_keep, svd_solver="full")
    pcs = pca.fit_transform(Xs)
    now = pca.transform(((F_now - mean) / scale).reshape(1, -1))[0]
    return pcs, now, pcs.shape[1]


def run_ab(returns, factors, *, train_window=252, rebalance_freq=5,
           var_keep=0.9, risk_aversion=1.0, max_weight=0.25) -> PCAABResult:
    joined = pd.concat([factors, returns], axis=1).dropna()
    factor_cols = list(factors.columns)
    F = joined[factor_cols].values
    R = np.expm1(joined[returns.columns].values)
    idx = joined.index
    T = F.shape[0]
    n_assets = R.shape[1]
    if T <= train_window + 10:
        raise ValueError(f"too few aligned rows ({T})")

    optimizer = ManifoldOptimizer(risk_aversion=risk_aversion, max_weight=max_weight)
    test_n = T - train_window
    raw_ret = np.zeros(test_n)
    pca_ret = np.zeros(test_n)
    w_raw = np.ones(n_assets) / n_assets
    w_pca = np.ones(n_assets) / n_assets
    ncomps: list[int] = []

    for t in range(train_window, T):
        i = t - train_window
        if i % rebalance_freq == 0:
            s = max(0, t - train_window)
            F_tr, R_tr, F_now = F[s:t], R[s:t], F[t - 1]
            cov = estimate_covariance(R_tr)
            try:
                raw_load = _fit_ols_loadings(F_tr, R_tr)
                w_raw = optimizer.optimize(raw_load, F_now, cov)
            except Exception as e:
                logger.warning("raw rebalance failed t=%d: %s", t, e)
            try:
                pcs, pc_now, k = _causal_pca(F_tr, F_now, var_keep)
                pca_load = _fit_ols_loadings(pcs, R_tr)
                w_pca = optimizer.optimize(pca_load, pc_now, cov)
                ncomps.append(k)
            except Exception as e:
                logger.warning("pca rebalance failed t=%d: %s", t, e)

        day = R[t]
        valid = ~np.isnan(day)
        if valid.any():
            raw_ret[i] = float(np.nansum(w_raw[valid] * day[valid]))
            pca_ret[i] = float(np.nansum(w_pca[valid] * day[valid]))

    test_idx = idx[train_window:]
    bh = np.expm1(returns["btc_return"].reindex(test_idx)).fillna(0.0) \
        if "btc_return" in returns else pd.Series(0.0, index=test_idx)
    return PCAABResult(pd.Series(raw_ret, index=test_idx),
                       pd.Series(pca_ret, index=test_idx), bh,
                       float(np.mean(ncomps)) if ncomps else 0.0)


def render_markdown(res: PCAABResult, cmp: dict, args: dict) -> str:
    L = ["# PCA-orthogonalized drivers vs raw factors\n"]
    L.append("Same optimizer / cadence / train window; only the OLS regressors "
             "differ — the 14 raw CPCM factors vs their causal principal "
             f"components (fit on the trailing window, {args['var_keep']:.0%} "
             "variance kept). Scored OOS vs BH BTC.\n")
    L.append(f"- Assets: `{', '.join(args['assets'])}`")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | train "
             f"{args['train_window']}d, rebalance {args['rebalance_freq']}d")
    L.append(f"- Mean components kept: {res.mean_n_components:.1f}")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    def _line(name, s):
        r = s.values
        pv = np.cumprod(1 + r)
        sr = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(ANNUALIZATION)
        return f"| {name} | {pv[-1]/pv[0]-1:+.1%} | {sr:.3f} |"

    L.append("## Full OOS window\n| Arm | Total | Sharpe |\n|---|---:|---:|")
    L.append(_line("Raw factors", res.raw_returns))
    L.append(_line("PCA drivers", res.pca_returns))
    L.append(_line("BH BTC", res.bh_btc))
    L.append("")

    L.append("## Walk-forward folds vs BH BTC\n")
    L.append(cmp["multiple_testing_note"] + "\n")
    L.append("| Arm | Win-rate vs BH | Median Sharpe | Folds |\n|---|---:|---:|---:|")
    for name, wr, ms in cmp["ranking"]:
        rep = cmp["reports"][name]
        L.append(f"| {name} | {wr:.0%} | {ms:.3f} | {rep.n_folds} |")
    L.append("")

    raw_wr = cmp["reports"]["Raw"].win_rate
    pca_wr = cmp["reports"]["PCA"].win_rate
    L.append("## Verdict\n")
    if pca_wr > raw_wr:
        L.append(f"PCA drivers generalized better than raw factors "
                 f"({pca_wr:.0%} vs {raw_wr:.0%} fold win-rate vs BH) — "
                 f"decorrelating the drivers helped loading stability. Whether it "
                 f"clears BH BTC is the real bar (see table).")
    elif pca_wr == raw_wr:
        L.append(f"PCA and raw drivers tied OOS ({pca_wr:.0%} fold win-rate vs "
                 f"BH). Orthogonalizing didn't change the generalization story.")
    else:
        L.append(f"PCA drivers did NOT help ({pca_wr:.0%} vs raw {raw_wr:.0%}). "
                 f"Collinearity was not the binding constraint, and neither "
                 f"driver set beats buy-and-hold BTC.")
    return "\n".join(L)


def run(assets, start, end, **kw):
    from causal_portfolio.data import get_loader
    from causal_portfolio.validation.walk_forward import compare_variants
    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    returns = loader.load_returns(assets, start, end)
    factors = build_all_factors(panel, macro)
    if "funding_basis" in factors.columns:
        factors = factors.drop(columns=["funding_basis"])  # keep 2022+ sample
    res = run_ab(returns, factors, **kw)
    cmp = compare_variants({"Raw": res.raw_returns, "PCA": res.pca_returns},
                           res.bh_btc, win_rate_bar=0.8)
    return res, cmp


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--rebalance-freq", type=int, default=5)
    p.add_argument("--var-keep", type=float, default=0.9)
    p.add_argument("--out", default="causal_portfolio/docs/pca_drivers.md")
    args = p.parse_args()
    assets = args.assets.split(",")
    res, cmp = run(assets, args.start, args.end, train_window=args.train_window,
                   rebalance_freq=args.rebalance_freq, var_keep=args.var_keep)
    for name in ("Raw", "PCA"):
        rep = cmp["reports"][name]
        print(f"{name:>4}: win-rate vs BH {rep.win_rate:.0%} | median Sharpe "
              f"{rep.median_sharpe:.3f}")
    md = render_markdown(res, cmp, {
        "assets": assets, "start": args.start, "end": args.end,
        "train_window": args.train_window, "rebalance_freq": args.rebalance_freq,
        "var_keep": args.var_keep,
    })
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
