"""Tests for solvers/v1_linear.py — V1 Ridge solver."""

import numpy as np
import pytest

from causal_portfolio.solvers.v1_linear import V1LinearSolver


@pytest.fixture
def linear_data():
    """Synthetic linear data: returns = drivers @ beta + noise."""
    rng = np.random.default_rng(42)
    T = 300
    m = 3
    n_assets = 5

    drivers = rng.standard_normal((T, m))
    beta_true = rng.standard_normal((m, n_assets))
    noise = rng.standard_normal((T, n_assets)) * 0.1
    returns = drivers @ beta_true + noise

    return drivers, returns, beta_true


class TestV1LinearSolver:
    def test_fit_sets_fitted(self, linear_data):
        drivers, returns, _ = linear_data
        solver = V1LinearSolver(alpha=0.01)
        solver.fit(drivers, returns)
        assert solver.is_fitted

    def test_not_fitted_raises(self):
        solver = V1LinearSolver()
        with pytest.raises(RuntimeError, match="not fitted"):
            solver.predict(np.zeros(3))

    def test_recovers_beta(self, linear_data):
        """With low noise and low regularization, should recover true beta."""
        drivers, returns, beta_true = linear_data
        solver = V1LinearSolver(alpha=0.01)
        solver.fit(drivers, returns)

        # beta_ is (m, n_assets), compare with beta_true (m, n_assets)
        np.testing.assert_allclose(solver.beta_, beta_true, atol=0.15)

    def test_predict_shape(self, linear_data):
        drivers, returns, _ = linear_data
        solver = V1LinearSolver()
        solver.fit(drivers, returns)

        # Single observation
        pred = solver.predict(drivers[0])
        assert pred.shape == (1, returns.shape[1])

        # Batch
        pred = solver.predict(drivers[:10])
        assert pred.shape == (10, returns.shape[1])

    def test_jacobian_shape(self, linear_data):
        drivers, returns, _ = linear_data
        solver = V1LinearSolver()
        solver.fit(drivers, returns)

        J = solver.jacobian(drivers[0])
        assert J.shape == (returns.shape[1], drivers.shape[1])

    def test_jacobian_constant(self, linear_data):
        """For linear solver, Jacobian should be the same at any point."""
        drivers, returns, _ = linear_data
        solver = V1LinearSolver()
        solver.fit(drivers, returns)

        J1 = solver.jacobian(drivers[0])
        J2 = solver.jacobian(drivers[50])
        np.testing.assert_array_equal(J1, J2)

    def test_diagnostics(self, linear_data):
        drivers, returns, _ = linear_data
        solver = V1LinearSolver()
        solver.fit(drivers, returns)

        diag = solver.diagnostics
        assert "mean_r2" in diag
        assert "r2_per_asset" in diag
        assert diag["mean_r2"] > 0.8  # low noise, should have high R²

    def test_residual_covariance_shape(self, linear_data):
        drivers, returns, _ = linear_data
        solver = V1LinearSolver()
        solver.fit(drivers, returns)

        assert solver.residual_cov_.shape == (returns.shape[1], returns.shape[1])

    def test_handles_nan_rows(self):
        rng = np.random.default_rng(0)
        drivers = rng.standard_normal((100, 2))
        returns = rng.standard_normal((100, 3))
        # Insert some NaN rows
        drivers[10, 0] = np.nan
        returns[20, 1] = np.nan

        solver = V1LinearSolver()
        solver.fit(drivers, returns)
        assert solver.is_fitted
        assert solver.diagnostics["n_observations"] == 98

    def test_insufficient_data_raises(self):
        solver = V1LinearSolver()
        with pytest.raises(ValueError, match="Insufficient"):
            solver.fit(np.zeros((3, 2)), np.zeros((3, 5)))
