"""Tests for the regime dashboard panel logic.

`contiguous_runs` is pure and tested directly. `analyze_regimes` is exercised
against the local DuckDB (real data) for the common configs, with the heavy
HMM kept small/fast via n_restarts=2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn", reason="hmmlearn not installed")

from causal_portfolio.regimes.dashboard_panel import (
    analyze_regimes,
    contiguous_runs,
)


# ── contiguous_runs (pure) ──────────────────────────────────────────


def test_contiguous_runs_basic():
    dates = pd.date_range("2024-01-01", periods=8)
    labels = np.array([0, 0, 0, 1, 1, 0, 0, 1])
    runs = list(contiguous_runs(dates, labels))
    assert len(runs) == 4
    assert runs[0] == (dates[0], dates[2], 0)
    assert runs[1] == (dates[3], dates[4], 1)
    assert runs[2] == (dates[5], dates[6], 0)
    assert runs[3] == (dates[7], dates[7], 1)


def test_contiguous_runs_single_regime():
    dates = pd.date_range("2024-01-01", periods=4)
    labels = np.array([0, 0, 0, 0])
    runs = list(contiguous_runs(dates, labels))
    assert runs == [(dates[0], dates[3], 0)]


def test_contiguous_runs_empty():
    assert list(contiguous_runs(pd.DatetimeIndex([]), np.array([]))) == []


# ── analyze_regimes (integration, local DuckDB) ────────────────────

ASSETS = ["btc", "eth", "sol"]
START, END = "2022-01-01", "2025-12-31"


def test_analyze_two_state_causal():
    r = analyze_regimes(ASSETS, START, END, n_states=2, causal=True,
                        n_restarts=2, hmm_window=504, hmm_refit_every=63)
    assert r.n_states == 2
    assert r.causal is True
    assert len(r.dates) == len(r.labels) == len(r.market_curve)
    assert r.transition_matrix.shape == (2, 2)
    # Rows of a transition matrix sum to ~1
    np.testing.assert_allclose(r.transition_matrix.sum(axis=1), [1, 1], atol=1e-6)
    # Per-regime structures present for each state
    assert set(r.per_regime_drivers.keys()) == {0, 1}
    assert set(r.per_regime_returns.keys()) == {0, 1}
    # State means: 2 states x 2 features (vix, btc_vol)
    assert r.state_means.shape == (2, 2)
    # Params that produced the result are stored on it (dashboard captions
    # read these, never the live sidebar widgets)
    assert r.params["n_states"] == 2
    assert r.params["causal"] is True
    assert r.params["hmm_window"] == 504
    assert r.params["assets"] == ASSETS


def test_analyze_two_state_noncausal_differs_or_matches():
    """Non-causal labels should run and produce a full-sample labeling."""
    r = analyze_regimes(ASSETS, START, END, n_states=2, causal=False,
                        n_restarts=2)
    assert r.causal is False
    # Non-causal labels cover the whole feature window (no warmup drop)
    assert len(r.labels) > 500
    # Both regimes should be visited on this multi-year sample
    assert set(np.unique(r.labels).tolist()) == {0, 1}


def test_analyze_single_state_is_baseline():
    r = analyze_regimes(ASSETS, START, END, n_states=1, causal=True, n_restarts=2)
    assert r.n_states == 1
    assert set(np.unique(r.labels).tolist()) == {0}
    # One regime → drivers/returns keyed only by 0
    assert set(r.per_regime_drivers.keys()) == {0}
    # Per-regime driver winner for the single regime == global pick
    assert r.per_regime_drivers[0]["winner"] is not None


def test_analyze_three_state():
    r = analyze_regimes(ASSETS, START, END, n_states=3, causal=True,
                        n_restarts=3, hmm_window=504, hmm_refit_every=63)
    assert r.n_states == 3
    assert r.transition_matrix.shape == (3, 3)
    assert r.state_means.shape == (3, 2)


def test_analyze_requires_btc():
    with pytest.raises(ValueError, match="BTC"):
        analyze_regimes(["eth", "sol"], START, END, n_states=2, n_restarts=2)


def test_per_regime_returns_have_expected_keys():
    r = analyze_regimes(ASSETS, START, END, n_states=2, causal=True, n_restarts=2)
    for s in range(2):
        stat = r.per_regime_returns[s]
        assert set(stat.keys()) == {"ann_return", "ann_vol", "sharpe", "days"}
