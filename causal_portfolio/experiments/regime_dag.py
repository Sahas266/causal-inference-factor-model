"""Step 7 (exploratory) — Regime × DAG.

Question: do the factor->return loadings differ by market regime, and does
estimating them regime-conditionally generalize OOS better than a single pooled
model? This crosses the causal DAG estimation (factor loadings) with the HMM
regime labels (risk-on / risk-off from VIX + BTC realized vol).

Design (fair A/B, same optimizer/cadence/factor pool):
  - Causal regime labels via rolling_fit_decode (forward-filter only — no
    look-ahead).
  - At each rebalance t with current regime r_t:
      pooled arm:  OLS loadings on the trailing `train_window` days.
      regime arm:  OLS loadings on the most recent `regime_lookback` PAST days
                   whose label == r_t (>= min_regime_obs, else fall back to the
                   pooled fit). Covariance is estimated on the same rows.
  - Both -> ManifoldOptimizer weights, held to the next rebalance.

Estimation uses OLS (Steps 5-6 showed the declared instruments are too weak for
2SLS to separate from OLS, so the regime lever — not OLS-vs-2SLS — is what is
varied here). funding_basis is excluded by default so the sample spans
2022-2025 (it only starts 2023-11, and regime-splitting a short sample starves
the folds).

Exploratory: judged on OOS fold win-rate vs BH BTC, same honest bar as
everything else.

Run:  python -m causal_portfolio.experiments.regime_dag
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.experiments.dag_variants import (
    FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS,
)
from causal_portfolio.experiments.ols_vs_2sls import _fit_ols_loadings
from causal_portfolio.factors.builder import build_all_factors
from causal_portfolio.optimizer.manifold import ManifoldOptimizer, estimate_covariance
from causal_portfolio.regimes.hmm import build_regime_features, rolling_fit_decode

logger = logging.getLogger("cpcm.experiments.regime_dag")

ANNUALIZATION = 365


@dataclass
class RegimeABResult:
    pooled_returns: pd.Series
    regime_returns: pd.Series
    bh_btc: pd.Series
    regime_labels: pd.Series
    regime_fit_count: int      # rebalances that used a regime-specific fit
    pooled_fallback_count: int  # rebalances that fell back to pooled
    n_states: int


def run_ab(
    returns: pd.DataFrame, factors: pd.DataFrame, regime_labels: pd.Series,
    *, train_window: int = 252, rebalance_freq: int = 5,
    regime_lookback: int = 504, min_regime_obs: int = 60,
    risk_aversion: float = 1.0, max_weight: float = 0.25,
) -> tuple[pd.Series, pd.Series, pd.Series, int, int]:
    joined = pd.concat([factors, returns], axis=1).dropna()
    joined = joined.join(regime_labels.rename("regime"), how="inner").dropna()
    factor_cols = list(factors.columns)
    F = joined[factor_cols].values
    R = np.expm1(joined[returns.columns].values)   # log -> simple
    reg = joined["regime"].astype(int).values
    idx = joined.index
    T = F.shape[0]
    n_assets = R.shape[1]
    if T <= train_window + 10:
        raise ValueError(f"too few aligned rows ({T})")

    optimizer = ManifoldOptimizer(risk_aversion=risk_aversion, max_weight=max_weight)
    test_n = T - train_window
    pooled_ret = np.zeros(test_n)
    regime_ret = np.zeros(test_n)
    w_pool = np.ones(n_assets) / n_assets
    w_reg = np.ones(n_assets) / n_assets
    regime_fits = 0
    fallbacks = 0

    for t in range(train_window, T):
        i = t - train_window
        if i % rebalance_freq == 0:
            s = max(0, t - train_window)
            F_now = F[t - 1]
            # pooled
            try:
                load = _fit_ols_loadings(F[s:t], R[s:t])
                w_pool = optimizer.optimize(load, F_now, estimate_covariance(R[s:t]))
            except Exception as e:
                logger.warning("pooled rebalance failed t=%d: %s", t, e)
            # regime-conditional
            cur = reg[t - 1]
            lo = max(0, t - regime_lookback)
            mask = reg[lo:t] == cur
            rows = np.nonzero(mask)[0] + lo
            if rows.size >= min_regime_obs:
                try:
                    load_r = _fit_ols_loadings(F[rows], R[rows])
                    w_reg = optimizer.optimize(load_r, F_now, estimate_covariance(R[rows]))
                    regime_fits += 1
                except Exception as e:
                    logger.warning("regime rebalance failed t=%d: %s", t, e)
                    w_reg = w_pool
                    fallbacks += 1
            else:
                w_reg = w_pool
                fallbacks += 1

        day = R[t]
        valid = ~np.isnan(day)
        if valid.any():
            pooled_ret[i] = float(np.nansum(w_pool[valid] * day[valid]))
            regime_ret[i] = float(np.nansum(w_reg[valid] * day[valid]))

    test_idx = idx[train_window:]
    if "btc_return" in returns:
        bh = np.expm1(returns["btc_return"].reindex(test_idx)).fillna(0.0)
    else:
        bh = pd.Series(0.0, index=test_idx)
    return (pd.Series(pooled_ret, index=test_idx),
            pd.Series(regime_ret, index=test_idx), bh, regime_fits, fallbacks)


def run(assets, start, end, *, n_states, train_window, rebalance_freq,
        regime_lookback, min_regime_obs, exclude_funding=True):
    from causal_portfolio.data import get_loader
    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    returns = loader.load_returns(assets, start, end)
    factors = build_all_factors(panel, macro)
    if exclude_funding and "funding_basis" in factors.columns:
        factors = factors.drop(columns=["funding_basis"])

    feats = build_regime_features(macro, returns)
    labels = rolling_fit_decode(
        feats, window_size=train_window, refit_every=63,
        n_states=n_states, n_restarts=max(3, n_states + 1))

    pooled, regime, bh, rfits, fb = run_ab(
        returns, factors, labels, train_window=train_window,
        rebalance_freq=rebalance_freq, regime_lookback=regime_lookback,
        min_regime_obs=min_regime_obs)
    return RegimeABResult(pooled, regime, bh, labels, rfits, fb, n_states)


def render_markdown(res: RegimeABResult, cmp: dict, args: dict) -> str:
    from causal_portfolio.regimes.hmm import dwell_stats

    L = ["# Regime × DAG (exploratory)\n"]
    L.append("Crosses the causal factor loadings with HMM regime labels "
             "(risk-on/off from VIX + BTC realized vol, forward-filter = causal). "
             "Pooled OLS loadings vs regime-conditional OLS loadings (fit only on "
             "past days sharing the current regime), same optimizer / cadence / "
             "factor pool, scored OOS vs BH BTC.\n")
    L.append(f"- Assets: `{', '.join(args['assets'])}`")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | {args['n_states']} "
             f"regimes | train {args['train_window']}d, rebalance "
             f"{args['rebalance_freq']}d, regime lookback {args['regime_lookback']}d")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    lbls = res.regime_labels.dropna().astype(int).values
    if len(lbls):
        ds = dwell_stats(lbls)
        L.append("## Regime occupancy (causal labels)\n")
        L.append("| Regime | Days | % | Mean run (d) |")
        L.append("|---|---:|---:|---:|")
        for st in sorted(ds):
            d = ds[st]
            L.append(f"| {st} | {d['days']} | {d['pct']:.0%} | {d['mean_run_length']:.0f} |")
        L.append("")
    L.append(f"Regime-specific fits used at {res.regime_fit_count} rebalances; "
             f"fell back to pooled at {res.pooled_fallback_count} (insufficient "
             f"same-regime history).\n")

    def _line(name, s):
        r = s.values
        pv = np.cumprod(1 + r)
        tot = pv[-1] / pv[0] - 1
        sr = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(ANNUALIZATION)
        return f"| {name} | {tot:+.1%} | {sr:.3f} |"

    L.append("## Full OOS window\n")
    L.append("| Arm | Total | Sharpe |")
    L.append("|---|---:|---:|")
    L.append(_line("Pooled", res.pooled_returns))
    L.append(_line("Regime", res.regime_returns))
    L.append(_line("BH BTC", res.bh_btc))
    L.append("")

    L.append("## Walk-forward folds vs BH BTC\n")
    L.append(cmp["multiple_testing_note"] + "\n")
    L.append("| Arm | Win-rate vs BH | Median Sharpe | Folds |")
    L.append("|---|---:|---:|---:|")
    for name, wr, ms in cmp["ranking"]:
        rep = cmp["reports"][name]
        L.append(f"| {name} | {wr:.0%} | {ms:.3f} | {rep.n_folds} |")
    L.append("")

    pooled_wr = cmp["reports"]["Pooled"].win_rate
    regime_wr = cmp["reports"]["Regime"].win_rate
    L.append("## Verdict\n")
    if regime_wr > pooled_wr and regime_wr >= 0.8:
        L.append(f"Regime-conditional loadings beat BH BTC in {regime_wr:.0%} of "
                 f"OOS folds, above pooled ({pooled_wr:.0%}) — a genuine "
                 f"exploratory signal worth a deeper look (validate stability + "
                 f"costs before trusting it).")
    elif regime_wr > pooled_wr:
        L.append(f"Regime conditioning edged pooled OOS ({regime_wr:.0%} vs "
                 f"{pooled_wr:.0%} fold win-rate vs BH) but neither clears a "
                 f"robust bar against simply holding BTC. Marginal at best.")
    else:
        L.append(f"Regime conditioning did NOT help: {regime_wr:.0%} vs pooled "
                 f"{pooled_wr:.0%} fold win-rate vs BH. Splitting the sample by "
                 f"regime cut estimation data without a generalizing payoff — and "
                 f"neither arm beats buy-and-hold BTC. Consistent with every prior "
                 f"result in this project.")
    return "\n".join(L)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--rebalance-freq", type=int, default=5)
    p.add_argument("--regime-lookback", type=int, default=504)
    p.add_argument("--min-regime-obs", type=int, default=60)
    p.add_argument("--out", default="causal_portfolio/docs/regime_dag_search.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    res = run(assets, args.start, args.end, n_states=args.n_states,
              train_window=args.train_window, rebalance_freq=args.rebalance_freq,
              regime_lookback=args.regime_lookback, min_regime_obs=args.min_regime_obs)

    from causal_portfolio.validation.walk_forward import compare_variants
    cmp = compare_variants(
        {"Pooled": res.pooled_returns, "Regime": res.regime_returns},
        res.bh_btc, win_rate_bar=0.8)

    print("\n" + "=" * 70)
    for name in ("Pooled", "Regime"):
        rep = cmp["reports"][name]
        print(f"{name:>7}: win-rate vs BH {rep.win_rate:.0%} | "
              f"median Sharpe {rep.median_sharpe:.3f} | folds {rep.n_folds}")
    print(f"regime fits={res.regime_fit_count} fallbacks={res.pooled_fallback_count}")

    md = render_markdown(res, cmp, {
        "assets": assets, "start": args.start, "end": args.end,
        "n_states": args.n_states, "train_window": args.train_window,
        "rebalance_freq": args.rebalance_freq, "regime_lookback": args.regime_lookback,
    })
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
