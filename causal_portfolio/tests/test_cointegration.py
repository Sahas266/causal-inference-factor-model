"""Tests for cointegration primitives + spread signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_portfolio.analysis.cointegration import (
    find_cointegrated_pairs, spread_positions, spread_series, johansen_rank,
)


def _cointegrated_logprices(n=500, seed=0):
    rng = np.random.default_rng(seed)
    # common stochastic trend
    trend = np.cumsum(rng.normal(0, 0.02, n))
    log_b = 4.0 + trend
    # A = const + beta*B + stationary noise  => A,B cointegrated
    log_a = 1.0 + 1.5 * log_b + rng.normal(0, 0.05, n)
    # an unrelated random walk (not cointegrated with either)
    log_c = 3.0 + np.cumsum(rng.normal(0, 0.02, n))
    idx = pd.date_range("2022-01-01", periods=n)
    return pd.DataFrame({"a": log_a, "b": log_b, "c": log_c}, index=idx)


def test_finds_the_cointegrated_pair():
    lp = _cointegrated_logprices()
    pairs = find_cointegrated_pairs(lp, adf_pvalue=0.05)
    found = {frozenset((p.a, p.b)) for p in pairs}
    assert frozenset(("a", "b")) in found
    # recovered hedge ratio near the true 1.5 (for a~b orientation)
    ab = [p for p in pairs if frozenset((p.a, p.b)) == frozenset(("a", "b"))][0]
    # depending on orientation beta is ~1.5 or ~1/1.5
    assert (abs(ab.beta - 1.5) < 0.3) or (abs(ab.beta - 1 / 1.5) < 0.15)


def test_rejects_noncointegrated():
    rng = np.random.default_rng(1)
    n = 400
    lp = pd.DataFrame({
        "x": 3 + np.cumsum(rng.normal(0, 0.02, n)),
        "y": 3 + np.cumsum(rng.normal(0, 0.02, n)),  # independent RW
    }, index=pd.date_range("2022-01-01", periods=n))
    pairs = find_cointegrated_pairs(lp, adf_pvalue=0.01)
    assert frozenset(("x", "y")) not in {frozenset((p.a, p.b)) for p in pairs}


def test_spread_series_is_residual():
    la = np.array([2.0, 3.0, 4.0])
    lb = np.array([1.0, 2.0, 3.0])
    spr = spread_series(la, lb, beta=1.0, const=1.0)
    np.testing.assert_allclose(spr, [0.0, 0.0, 0.0])


def test_spread_positions_band():
    z = np.array([0.0, 2.0, 1.0, 0.3, -2.0, -0.6, -0.4])
    pos = spread_positions(z, entry=1.5, exit=0.5)
    # z=2 -> short(-1); 1.0 hold(-1); 0.3 exit(0); -2 long(+1); -0.6 hold(+1); -0.4 exit(0)
    np.testing.assert_array_equal(pos, [0, -1, -1, 0, 1, 1, 0])


def test_johansen_detects_rank():
    lp = _cointegrated_logprices()
    out = johansen_rank(lp[["a", "b"]])
    assert out["rank"] >= 1
