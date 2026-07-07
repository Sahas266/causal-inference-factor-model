"""Validate the Python estimator port against exact Rust fixtures.

The fixtures (tests/fixtures/rust_estimator_fixtures.json) were emitted by
causal_model/crates/cpcm-estimate/tests/emit_fixtures.rs. Here we reproduce the
IDENTICAL inputs (including the Rust LCG) and assert the Python OLS/2SLS output
matches the Rust output to ~1e-6 — i.e. the port is numerically faithful.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal_portfolio.scm.estimators import (
    add_intercept, ols, tsls, durbin_watson,
)

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "rust_estimator_fixtures.json").read_text()
)


def _lcg_dataset(n: int = 1000):
    """Reproduce the Rust deterministic LCG dataset (tsls.rs seed 42)."""
    state = 42
    mask = (1 << 64) - 1
    u32_max = float(0xFFFFFFFF)

    def nxt() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1) & mask
        return ((state >> 33) / u32_max) - 0.5

    z = np.empty(n); x = np.empty(n); y = np.empty(n)
    for i in range(n):
        z_i = nxt() * 4.0
        v_i = nxt()
        e_i = nxt() + 0.8 * v_i
        x_i = 0.3 * z_i + v_i
        y_i = 1.0 + 0.5 * x_i + e_i
        z[i], x[i], y[i] = z_i, x_i, y_i
    return z, x, y


# ── input parity: confirm our LCG matches Rust's first 5 rows ─────────


def test_lcg_matches_rust_inputs():
    z, x, y = _lcg_dataset(1000)
    fx = FIXTURES["tsls_canonical"]
    np.testing.assert_allclose(z[:5], fx["data_z5"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(x[:5], fx["data_x5"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(y[:5], fx["data_y5"], rtol=0, atol=1e-9)


# ── OLS parity ───────────────────────────────────────────────────────


def test_ols_simple_matches_rust():
    n = 200
    x = np.array([i / 10.0 for i in range(n)])
    y = 3.0 + 2.0 * x
    res = ols(y, add_intercept(x), ["intercept", "x"])
    fx = FIXTURES["ols_simple"]
    np.testing.assert_allclose(res.coefficients, fx["coef"], rtol=0, atol=1e-6)
    assert abs(res.r_squared - fx["r2"]) < 1e-9


def test_ols_multiple_matches_rust():
    n = 300
    x = np.empty((n, 2))
    y = np.empty(n)
    for i in range(n):
        x1 = i / 50.0
        x2 = ((i * 7 + 3) % 100) / 20.0
        x[i] = [x1, x2]
        y[i] = 1.0 + 2.0 * x1 - 0.5 * x2
    res = ols(y, add_intercept(x), ["intercept", "x1", "x2"])
    np.testing.assert_allclose(res.coefficients, FIXTURES["ols_multi"]["coef"],
                               rtol=0, atol=1e-6)


# ── 2SLS parity ──────────────────────────────────────────────────────


def test_tsls_matches_rust_exactly():
    z, x, y = _lcg_dataset(1000)
    n = 1000
    res = tsls(
        y, x.reshape(-1, 1), np.ones((n, 1)), z.reshape(-1, 1),
        ["x"], ["intercept"], ["z"],
    )
    fx = FIXTURES["tsls_canonical"]
    np.testing.assert_allclose(res.coefficients, fx["coef"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.std_errors, fx["se"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.hac_std_errors, fx["hac_se"], rtol=1e-6, atol=0)
    np.testing.assert_allclose(res.first_stage_f, fx["first_stage_f"], rtol=1e-6, atol=0)
    assert abs(res.r_squared - fx["r2"]) < 1e-6
    # Hausman now uses the corrected 2SLS SEs (fixture regenerated for the fix)
    assert res.hausman_stat is not None
    assert abs(res.hausman_stat - fx["hausman_stat"]) < 1e-4
    # Just-identified (m == k1) → no Sargan
    assert res.sargan_stat is None


def test_tsls_recovers_causal_effect_beats_ols():
    """2SLS β on x should be closer to the true 0.5 than the biased OLS β."""
    z, x, y = _lcg_dataset(1000)
    n = 1000
    res = tsls(y, x.reshape(-1, 1), np.ones((n, 1)), z.reshape(-1, 1),
               ["x"], ["intercept"], ["z"])
    beta_x = res.coefficients[0]          # coef on endogenous x
    ols_res = ols(y, add_intercept(x), ["intercept", "x"])
    ols_beta_x = ols_res.coefficients[1]
    assert abs(beta_x - 0.5) < abs(ols_beta_x - 0.5)
    assert res.first_stage_f[0] > 5.0


# ── diagnostics sanity (mirror Rust diagnostics tests) ───────────────


def test_durbin_watson_alternating():
    assert durbin_watson(np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])) > 1.5


def test_durbin_watson_monotone():
    assert durbin_watson(np.array([0.1, 0.2, 0.3, 0.4, 0.5])) < 1.0


def test_overidentified_produces_sargan():
    """Sargan runs whenever m > k1 (df = m - k1, instruments minus endogenous)."""
    z, x, y = _lcg_dataset(500)
    n = 500
    z2 = z + 0.5 * np.sin(np.arange(n))
    Z = np.column_stack([z, z2])  # m=2 > k1=1 now suffices (df bug fixed)
    res = tsls(y, x.reshape(-1, 1), np.ones((n, 1)), Z,
               ["x"], ["intercept"], ["z1", "z2"])
    assert res.sargan_stat is not None
    assert res.sargan_p is not None
    assert 0.0 <= res.sargan_p <= 1.0


def test_hausman_none_when_no_usable_terms():
    """Exogenous x: 2SLS ≈ OLS, corrected SEs can be <= OLS SEs → terms with
    var_diff <= 0 are skipped; all skipped → (None, None) instead of the old
    1e-15-clamp explosion."""
    rng = np.random.default_rng(7)
    n = 400
    z = rng.standard_normal(n)
    x = z.copy()          # x IS the instrument: 2SLS == OLS exactly
    y = 1.0 + 0.5 * x + rng.standard_normal(n) * 0.1
    res = tsls(y, x.reshape(-1, 1), np.ones((n, 1)), z.reshape(-1, 1),
               ["x"], ["intercept"], ["z"])
    # identical fits → var_diff == 0 → skipped → no Hausman
    assert res.hausman_stat is None and res.hausman_p is None


def test_partial_f_matches_reference_implementation():
    """Library partial F must agree with the experiment harness's
    _partial_first_stage_f (ols_vs_2sls.py) — same math, different code path."""
    from causal_portfolio.experiments.ols_vs_2sls import _partial_first_stage_f
    rng = np.random.default_rng(11)
    n = 300
    exog = rng.standard_normal((n, 2))
    z = rng.standard_normal(n)
    x = 0.4 * z + 0.8 * exog[:, 0] + rng.standard_normal(n) * 0.5
    y = 0.3 * x + rng.standard_normal(n)
    x_exog = add_intercept(exog)
    res = tsls(y, x.reshape(-1, 1), x_exog, z.reshape(-1, 1),
               ["x"], ["intercept", "e1", "e2"], ["z"])
    f_ref = _partial_first_stage_f(x, z, exog)
    np.testing.assert_allclose(res.first_stage_f[0], f_ref, rtol=1e-8)
