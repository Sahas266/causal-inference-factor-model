"""Smoke tests for the regime-conditional backtester.

Uses small synthetic data to verify the engine runs end-to-end in both
modes ('hard' and 'moe'), returns a populated result, and the shape
contracts hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn", reason="hmmlearn not installed")

from causal_portfolio.backtest.regime_engine import (
    RegimeConditionalBacktester,
    RegimeBacktestResult,
)
from causal_portfolio.optimizer.manifold import ManifoldOptimizer


def _synthetic_inputs(T: int = 800, seed: int = 0):
    """Build small synthetic returns/factor_panel/macro with both regime features."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T)
    # Two synthetic regimes in VIX
    vix = np.concatenate([
        rng.normal(15, 2, T // 2),
        rng.normal(28, 4, T - T // 2),
    ])
    rng.shuffle(vix)  # not strictly contiguous but doesn't matter for smoke
    macro = pd.DataFrame({
        "vixcls": vix,
        "dgs10": rng.normal(4.0, 0.5, T),
        "t10y2y": rng.normal(0.0, 0.3, T),
    }, index=idx)

    # Returns: 3 assets
    returns = pd.DataFrame(
        rng.normal(0, 0.02, (T, 3)),
        index=idx,
        columns=["btc_return", "eth_return", "sol_return"],
    )

    # Factor panel: 6 plausible candidates
    factors = pd.DataFrame(
        rng.normal(0, 1, (T, 6)),
        index=idx,
        columns=["liq_flow", "chain_congestion", "mev_pressure",
                 "cex_dex_flow", "stable_flow", "staking_yield"],
    )
    return returns, factors, macro


def test_regime_engine_hard_smoke():
    """Hard-switch mode should run end-to-end without errors."""
    returns, factors, macro = _synthetic_inputs(T=800)
    opt = ManifoldOptimizer(risk_aversion=1.0, max_weight=0.5)
    bt = RegimeConditionalBacktester(
        optimizer=opt, mode="hard", n_states=2, m=3,
        hmm_window=200, hmm_refit_every=50,
        rebalance_freq=10, train_window=150, min_obs_per_regime=30,
        hmm_n_restarts=2,
    )
    result = bt.run(returns, factors, macro)
    assert isinstance(result, RegimeBacktestResult)
    # Some rebalances happened
    assert len(result.rebalance_dates) > 0
    # Returns series non-empty
    assert len(result.returns_series) > 0
    # Weight history aligned
    assert result.weights_history.shape[0] == len(result.returns_series)
    assert result.weights_history.shape[1] == 3  # 3 assets


def test_regime_engine_moe_smoke():
    """MoE mode should also run end-to-end."""
    returns, factors, macro = _synthetic_inputs(T=800, seed=1)
    opt = ManifoldOptimizer(risk_aversion=1.0, max_weight=0.5)
    bt = RegimeConditionalBacktester(
        optimizer=opt, mode="moe", n_states=2, m=3,
        hmm_window=200, hmm_refit_every=50,
        rebalance_freq=10, train_window=150, min_obs_per_regime=30,
        hmm_n_restarts=2,
    )
    result = bt.run(returns, factors, macro)
    assert isinstance(result, RegimeBacktestResult)
    assert len(result.rebalance_dates) > 0


def test_regime_engine_records_regime_labels():
    """The result should carry the regime label history."""
    returns, factors, macro = _synthetic_inputs(T=800)
    opt = ManifoldOptimizer(risk_aversion=1.0, max_weight=0.5)
    bt = RegimeConditionalBacktester(
        optimizer=opt, mode="hard", n_states=2, m=3,
        hmm_window=200, hmm_refit_every=50,
        rebalance_freq=10, train_window=150, min_obs_per_regime=30,
        hmm_n_restarts=2,
    )
    result = bt.run(returns, factors, macro)
    # Labels exist and cover the test portion
    assert len(result.regime_labels) > 0
    # Labels are in [0, n_states)
    unique = set(int(x) for x in result.regime_labels if x >= 0)
    assert unique.issubset({0, 1})
    # Posterior history shape
    assert result.posterior_history.shape == (len(result.regime_labels), 2)


def test_invalid_mode_raises():
    opt = ManifoldOptimizer(risk_aversion=1.0)
    with pytest.raises(ValueError, match="mode"):
        RegimeConditionalBacktester(optimizer=opt, mode="bogus")


def test_short_input_raises():
    returns, factors, macro = _synthetic_inputs(T=100)  # too short
    opt = ManifoldOptimizer(risk_aversion=1.0)
    bt = RegimeConditionalBacktester(
        optimizer=opt, mode="hard", n_states=2,
        hmm_window=200, train_window=150,
    )
    with pytest.raises(ValueError, match="T >"):
        bt.run(returns, factors, macro)
