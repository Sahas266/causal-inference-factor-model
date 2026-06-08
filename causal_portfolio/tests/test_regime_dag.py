"""Smoke tests for the Step-7 Regime x DAG experiment (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_portfolio.experiments.regime_dag import run_ab


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n)
    factors = pd.DataFrame(
        {"liq_flow": rng.normal(size=n),
         "chain_congestion": rng.normal(size=n),
         "vixcls": rng.normal(size=n)},
        index=idx,
    )
    returns = pd.DataFrame(
        {"btc_return": rng.normal(scale=0.02, size=n),
         "eth_return": rng.normal(scale=0.03, size=n),
         "sol_return": rng.normal(scale=0.04, size=n)},
        index=idx,
    )
    # alternating regime blocks of ~30 days
    labels = pd.Series((np.arange(n) // 30) % 2, index=idx, name="regime")
    return returns, factors, labels


def test_run_ab_shapes_and_counts():
    returns, factors, labels = _synthetic()
    pooled, regime, bh, rfits, fb = run_ab(
        returns, factors, labels,
        train_window=200, rebalance_freq=10,
        regime_lookback=400, min_regime_obs=40)
    assert len(pooled) == len(regime) == len(bh)
    assert rfits + fb > 0
    # bh aligned to the test index
    assert bh.index.equals(pooled.index)


def test_regime_fallback_when_starved():
    returns, factors, labels = _synthetic()
    # Impossible min -> always fall back to pooled => regime == pooled returns
    pooled, regime, bh, rfits, fb = run_ab(
        returns, factors, labels,
        train_window=200, rebalance_freq=10,
        regime_lookback=400, min_regime_obs=10_000)
    assert rfits == 0 and fb > 0
    np.testing.assert_allclose(pooled.values, regime.values, atol=1e-12)
