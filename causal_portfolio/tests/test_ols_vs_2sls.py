"""Tests for the Step-5 OLS-vs-2SLS experiment helpers.

The key guard: prove the first-stage F gate CAN detect a strong instrument and
that 2SLS then differs from OLS. This rules out the "instruments look dead due
to a bug" explanation for the empirical finding that the real instruments all
fail the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_portfolio.experiments.ols_vs_2sls import (
    _GateInfo, _fit_2sls_loadings, _fit_ols_loadings, _partial_first_stage_f,
    run_ab,
)


def test_partial_f_strong_instrument():
    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(size=n)
    exog = rng.normal(size=(n, 2))
    factor = 2.0 * z + 0.5 * exog[:, 0] + rng.normal(scale=0.1, size=n)  # z is strong
    f = _partial_first_stage_f(factor, z, exog)
    assert f > 100.0  # very strong relevance


def test_partial_f_dead_instrument():
    rng = np.random.default_rng(1)
    n = 500
    z = (rng.random(n) < 0.02).astype(float)  # sparse spike, ~uncorrelated
    exog = rng.normal(size=(n, 2))
    factor = rng.normal(size=n)
    f = _partial_first_stage_f(factor, z, exog)
    assert f < 10.0  # fails the weak-instrument gate


def test_ols_loadings_recover_known_beta():
    rng = np.random.default_rng(2)
    n, m, n_assets = 600, 3, 2
    F = rng.normal(size=(n, m))
    true_beta = np.array([[0.5, -0.2], [0.1, 0.3], [-0.4, 0.05]])  # (m, n_assets)
    R = F @ true_beta + rng.normal(scale=0.01, size=(n, n_assets))
    load = _fit_ols_loadings(F, R)
    np.testing.assert_allclose(load.beta_, true_beta, atol=0.02)


def test_2sls_degrades_to_ols_when_instrument_weak():
    rng = np.random.default_rng(3)
    n, m, n_assets = 400, 2, 2
    F = rng.normal(size=(n, m))
    R = F @ np.array([[0.3, 0.1], [-0.2, 0.4]]) + rng.normal(scale=0.01, size=(n, n_assets))
    Z = (rng.random(n) < 0.02).astype(float).reshape(-1, 1)  # dead instrument
    gate = _GateInfo()
    ols_load = _fit_ols_loadings(F, R)
    tsls_load = _fit_2sls_loadings(
        F, R, Z, endog_idx=[0], instrument_of={0: 0},
        f_threshold=10.0, gate=gate, iv_names={0: "dead"})
    # Instrument failed the gate -> identical to OLS
    np.testing.assert_allclose(tsls_load.beta_, ols_load.beta_, atol=1e-9)
    assert gate.passed.get("dead", 0) == 0


def test_2sls_differs_from_ols_with_strong_instrument():
    rng = np.random.default_rng(4)
    n, n_assets = 800, 1
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    e = rng.normal(size=n) + 0.8 * v          # endogeneity
    x = 0.8 * z + v                            # strong first stage
    F = x.reshape(-1, 1)
    R = (1.0 + 0.5 * x + e).reshape(-1, n_assets)
    Z = z.reshape(-1, 1)
    gate = _GateInfo()
    ols_load = _fit_ols_loadings(F, R)
    tsls_load = _fit_2sls_loadings(
        F, R, Z, endog_idx=[0], instrument_of={0: 0},
        f_threshold=10.0, gate=gate, iv_names={0: "z"})
    assert gate.passed.get("z", 0) == 1
    # OLS β biased upward by endogeneity; 2SLS closer to the true 0.5
    assert abs(tsls_load.beta_[0, 0] - 0.5) < abs(ols_load.beta_[0, 0] - 0.5)


def test_run_ab_smoke():
    rng = np.random.default_rng(5)
    n = 400
    idx = pd.date_range("2022-01-01", periods=n)
    factors = pd.DataFrame(
        {"liq_flow": rng.normal(size=n), "vixcls": rng.normal(size=n)}, index=idx)
    instruments = pd.DataFrame({"gas_spike": (rng.random(n) < 0.05).astype(float)}, index=idx)
    returns = pd.DataFrame(
        {"btc_return": rng.normal(scale=0.02, size=n),
         "eth_return": rng.normal(scale=0.03, size=n)}, index=idx)
    ab = run_ab(returns, factors, instruments, {"liq_flow": "gas_spike"},
                train_window=200, rebalance_freq=10, f_threshold=10.0)
    assert len(ab.ols_returns) == len(ab.tsls_returns)
    assert "gas_spike" in ab.gate.seen
