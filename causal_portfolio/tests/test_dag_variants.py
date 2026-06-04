"""Tests for DAG-variant driver construction.

The _build_drivers function is the core of the experiment: it turns a
DagVariant spec into the (returns, drivers, dates, names) the backtester
consumes. We test lag application, pool filtering, and combo selection on
synthetic factor data — no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.experiments.dag_variants import DagVariant, _build_drivers


def _synthetic():
    idx = pd.date_range("2024-01-01", periods=300)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({
        "btc_return": rng.normal(0, 0.02, 300),
        "eth_return": rng.normal(0, 0.03, 300),
    }, index=idx)
    # 2 global + 2 macro factors (names must match builder's GLOBAL/MACRO sets)
    factors = pd.DataFrame({
        "liq_flow": rng.normal(0, 1, 300),
        "mev_pressure": rng.normal(0, 1, 300),
        "vixcls": rng.normal(0, 1, 300),
        "dff": rng.normal(0, 1, 300),
    }, index=idx)
    return returns, factors


def test_pool_filtering_explicit():
    returns, factors = _synthetic()
    v = DagVariant("t", ("liq_flow", "mev_pressure"), 0, 0, None)
    _, D, _, names = _build_drivers(v, returns, factors)
    assert set(names) == {"liq_flow", "mev_pressure"}
    assert D.shape[1] == 2


def test_pool_none_uses_all():
    returns, factors = _synthetic()
    v = DagVariant("t", None, 0, 0, None)
    _, D, _, names = _build_drivers(v, returns, factors)
    assert set(names) == {"liq_flow", "mev_pressure", "vixcls", "dff"}


def test_macro_lag_drops_rows():
    """Lagging macro by 1 should drop exactly one row (the first NaN)."""
    returns, factors = _synthetic()
    v_nolag = DagVariant("a", None, 0, 0, None)
    v_lag = DagVariant("b", None, 0, 1, None)
    R0, _, d0, _ = _build_drivers(v_nolag, returns, factors)
    R1, _, d1, _ = _build_drivers(v_lag, returns, factors)
    assert len(d1) == len(d0) - 1  # one row lost to the lag


def test_lag_actually_shifts_values():
    """A lag-1 macro driver at row t should equal the unlagged value at t-1."""
    returns, factors = _synthetic()
    v = DagVariant("b", ("vixcls",), 0, 1, None)
    _, D, dates, names = _build_drivers(v, returns, factors)
    assert names == ["vixcls"]
    # D[0] corresponds to dates[0]; its value = original vixcls at the prior day
    orig = factors["vixcls"]
    first_date = pd.Timestamp(dates[0])
    prior_date = orig.index[orig.index.get_loc(first_date) - 1]
    assert D[0, 0] == pytest.approx(orig.loc[prior_date])


def test_combo_selects_subset():
    returns, factors = _synthetic()
    v = DagVariant("t", None, 0, 0, 2)
    _, D, _, names = _build_drivers(v, returns, factors)
    assert len(names) == 2
    assert D.shape[1] == 2


def test_empty_pool_raises():
    returns, factors = _synthetic()
    v = DagVariant("t", ("nonexistent_factor",), 0, 0, None)
    with pytest.raises(ValueError, match="no factors available"):
        _build_drivers(v, returns, factors)
