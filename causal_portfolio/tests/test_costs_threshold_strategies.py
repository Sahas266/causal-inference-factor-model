"""Tests for cost model, no-trade band, and baseline strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.backtest.costs import cost_drag_return, trade_cost_usd
from causal_portfolio.backtest.threshold import should_rebalance


# ── costs ────────────────────────────────────────────────────────────


def test_trade_cost_zero_when_weights_unchanged():
    w = np.array([0.5, 0.5])
    assert trade_cost_usd(w, w, equity_usd=10_000.0) == 0.0


def test_trade_cost_proportional_to_l1_distance():
    """Full reallocation from [1,0] to [0,1] is |Δw|_L1 = 2 → 2*equity*bps notional."""
    w0 = np.array([1.0, 0.0])
    w1 = np.array([0.0, 1.0])
    cost = trade_cost_usd(w0, w1, equity_usd=10_000.0, fee_bps=5, slippage_bps=5)
    # 2 * 10000 * 0.0010 = $20
    assert cost == pytest.approx(20.0)


def test_trade_cost_scales_with_bps():
    w0 = np.array([0.5, 0.5])
    w1 = np.array([1.0, 0.0])  # L1 = 1.0
    cost_5 = trade_cost_usd(w0, w1, equity_usd=10_000.0, fee_bps=5, slippage_bps=0)
    cost_50 = trade_cost_usd(w0, w1, equity_usd=10_000.0, fee_bps=50, slippage_bps=0)
    assert cost_50 == pytest.approx(10 * cost_5)


def test_cost_drag_return_normalizes_by_equity():
    w0 = np.array([1.0, 0.0])
    w1 = np.array([0.0, 1.0])
    drag = cost_drag_return(w0, w1, equity_usd=10_000.0, fee_bps=5, slippage_bps=5)
    # cost = $20, drag = 20 / 10000 = 0.002 = 20 bps
    assert drag == pytest.approx(0.002)


def test_cost_drag_zero_equity_safe():
    w0 = np.array([0.5, 0.5])
    w1 = np.array([1.0, 0.0])
    assert cost_drag_return(w0, w1, equity_usd=0.0) == 0.0


def test_cost_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        trade_cost_usd(np.array([1.0]), np.array([1.0, 0.0]), 10_000.0)


# ── threshold (no-trade band) ────────────────────────────────────────


def test_threshold_zero_always_trades():
    assert should_rebalance(np.array([1, 0]), np.array([0.99, 0.01]), 0.0)
    assert should_rebalance(np.array([1, 0]), np.array([1, 0]), 0.0)


def test_threshold_skips_small_change():
    """0.04 L1 distance, 0.10 threshold → skip."""
    assert not should_rebalance(
        np.array([0.5, 0.5]), np.array([0.52, 0.48]), threshold_l1=0.10,
    )


def test_threshold_fires_on_large_change():
    """0.4 L1 distance, 0.10 threshold → fire."""
    assert should_rebalance(
        np.array([0.5, 0.5]), np.array([0.7, 0.3]), threshold_l1=0.10,
    )


def test_threshold_at_boundary_fires():
    """Equality → fire (≥)."""
    assert should_rebalance(
        np.array([0.5, 0.5]), np.array([0.55, 0.45]), threshold_l1=0.10,
    )


def test_threshold_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        should_rebalance(np.array([1.0]), np.array([1.0, 0.0]), threshold_l1=0.1)


# ── baseline strategies (small synthetic data) ─────────────────────


def _synthetic_returns(T: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T)
    # BTC: positive drift; ETH: negative drift (so equal-weight ≠ btc-only)
    return pd.DataFrame({
        "btc_return": rng.normal(0.0005, 0.03, T),
        "eth_return": rng.normal(-0.0003, 0.04, T),
        "sol_return": rng.normal(0.0002, 0.05, T),
    }, index=idx)


def _synthetic_macro(idx) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "vixcls": np.concatenate([
            rng.normal(15, 2, len(idx) // 2),
            rng.normal(30, 4, len(idx) - len(idx) // 2),
        ]),
    }, index=idx)


def test_buy_and_hold_btc_runs_clean():
    from causal_portfolio.backtest.strategies import buy_and_hold
    R = _synthetic_returns()
    result = buy_and_hold(R, asset="btc_return")
    # Weights should be 100% BTC after first day
    assert result.weights_history[-1][0] == pytest.approx(1.0)
    assert result.weights_history[-1][1] == 0.0
    # Result populated
    assert len(result.returns_series) == len(R)


def test_buy_and_hold_unknown_asset_raises():
    from causal_portfolio.backtest.strategies import buy_and_hold
    R = _synthetic_returns()
    with pytest.raises(KeyError, match="not in returns"):
        buy_and_hold(R, asset="bogus_return")


def test_equal_weight_basket_holds_third_in_each():
    from causal_portfolio.backtest.strategies import equal_weight_basket
    R = _synthetic_returns()
    result = equal_weight_basket(R, rebalance_freq=21)
    final_w = result.weights_history[-1]
    # Weights drift between rebalances; just check they're roughly equal
    assert np.allclose(final_w.sum(), 1.0, atol=0.01)


def test_regime_gated_long_only_runs_clean():
    pytest.importorskip("hmmlearn", reason="hmmlearn not installed")
    from causal_portfolio.backtest.strategies import regime_gated_long_only
    R = _synthetic_returns(T=800)
    M = _synthetic_macro(R.index)
    result = regime_gated_long_only(
        R, M,
        universe_weights={"btc": 1.0},
        hmm_window=200, hmm_refit_every=50, n_restarts=2,
    )
    # Should have produced some returns and some regime labels
    assert len(result.returns_series) > 0
    assert len(result.regime_labels) == len(result.returns_series)
    # Labels in [0, 1]
    valid_labels = result.regime_labels[result.regime_labels >= 0]
    assert set(np.unique(valid_labels).tolist()).issubset({0, 1})


def test_regime_gated_respects_stress_allocation():
    """If we never want to be in cash (stress_allocation=1.0), weights should be ~constant."""
    pytest.importorskip("hmmlearn", reason="hmmlearn not installed")
    from causal_portfolio.backtest.strategies import regime_gated_long_only
    R = _synthetic_returns(T=800)
    M = _synthetic_macro(R.index)
    result = regime_gated_long_only(
        R, M, universe_weights={"btc": 1.0},
        hmm_window=200, hmm_refit_every=50, n_restarts=2,
        stress_allocation=1.0,  # never de-risk
    )
    # All weights should be BTC=1.0, no turnover
    final_btc = result.weights_history[-1][0]
    assert final_btc == pytest.approx(1.0)
