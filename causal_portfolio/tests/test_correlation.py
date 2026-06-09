"""Tests for correlation-distance + HRP allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_portfolio.analysis.correlation import (
    correlation_distance, hrp_weights,
)


def test_correlation_distance_bounds():
    corr = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    d = correlation_distance(corr)
    assert np.allclose(np.diag(d), 0.0)
    assert abs(d[0, 1] - 0.0) < 1e-9     # perfectly correlated -> distance 0
    assert abs(d[0, 2] - 1.0) < 1e-9     # perfectly anti-correlated -> distance 1


def test_hrp_weights_sum_to_one_long_only():
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.normal(0, 0.02, (300, 5)),
                     columns=list("abcde"))
    w = hrp_weights(R)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()
    assert set(w.index) == set("abcde")


def test_hrp_downweights_redundant_cluster():
    """Two near-duplicate assets should together not dominate a third independent one."""
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.02, 600)
    a = base + rng.normal(0, 0.001, 600)
    b = base + rng.normal(0, 0.001, 600)   # a,b highly correlated (redundant)
    c = rng.normal(0, 0.02, 600)           # independent
    R = pd.DataFrame({"a": a, "b": b, "c": c})
    w = hrp_weights(R)
    # the lone independent asset should get a meaningful share, not be crowded out
    assert w["c"] > 0.25
    # a and b individually should each be <= c (their cluster is split)
    assert w["a"] <= w["c"] + 1e-9 and w["b"] <= w["c"] + 1e-9


def test_hrp_single_asset():
    R = pd.DataFrame({"only": np.random.default_rng(2).normal(0, 0.02, 100)})
    w = hrp_weights(R)
    assert abs(w["only"] - 1.0) < 1e-9
