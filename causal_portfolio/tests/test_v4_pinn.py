"""Tests for solvers/v4_pinn.py — V4 PINN solver."""

import numpy as np
import pytest

from causal_portfolio.solvers.v4_pinn import V4PINNSolver


@pytest.fixture
def nonlinear_data():
    """Synthetic nonlinear data: returns = sin(F1) + F2^2 + F3."""
    rng = np.random.default_rng(42)
    T = 500
    m = 3
    n_assets = 4

    drivers = rng.standard_normal((T, m)) * 0.5

    # Nonlinear mapping with different functions per asset
    returns = np.column_stack([
        np.sin(drivers[:, 0]) + 0.5 * drivers[:, 1],
        drivers[:, 1] ** 2 - drivers[:, 2],
        np.tanh(drivers[:, 0] + drivers[:, 2]),
        0.3 * drivers[:, 0] * drivers[:, 1],
    ]) + rng.standard_normal((T, n_assets)) * 0.05

    return drivers, returns


class TestV4PINNSolver:
    def test_fit_and_predict(self, nonlinear_data):
        drivers, returns = nonlinear_data
        solver = V4PINNSolver(m_drivers=3, n_assets=4)
        solver.fit(drivers, returns, n_epochs=100, patience=20)
        assert solver.is_fitted

        pred = solver.predict(drivers[:5])
        assert pred.shape == (5, 4)

    def test_predict_single(self, nonlinear_data):
        drivers, returns = nonlinear_data
        solver = V4PINNSolver(m_drivers=3, n_assets=4)
        solver.fit(drivers, returns, n_epochs=50, patience=10)

        pred = solver.predict(drivers[0])
        assert pred.shape == (1, 4)

    def test_jacobian_shape(self, nonlinear_data):
        drivers, returns = nonlinear_data
        solver = V4PINNSolver(m_drivers=3, n_assets=4)
        solver.fit(drivers, returns, n_epochs=50, patience=10)

        J = solver.jacobian(drivers[0])
        assert J.shape == (4, 3)  # (n_assets, m_drivers)

    def test_jacobian_varies(self, nonlinear_data):
        """For nonlinear model, Jacobian should differ at different points."""
        drivers, returns = nonlinear_data
        solver = V4PINNSolver(m_drivers=3, n_assets=4)
        solver.fit(drivers, returns, n_epochs=100, patience=20)

        J1 = solver.jacobian(drivers[0])
        J2 = solver.jacobian(drivers[100])
        assert not np.allclose(J1, J2, atol=1e-3)

    def test_learns_mapping(self, nonlinear_data):
        """Should achieve reasonable MSE on nonlinear data."""
        drivers, returns = nonlinear_data
        solver = V4PINNSolver(m_drivers=3, n_assets=4, hidden_dim=32)
        solver.fit(
            drivers, returns,
            n_epochs=300, lr=1e-3, lambda_J=0.001, patience=50,
        )

        pred = solver.predict(drivers[:400])  # train portion
        mse = np.mean((pred - returns[:400]) ** 2)
        assert mse < 0.1, f"MSE too high: {mse:.4f}"

    def test_train_history(self, nonlinear_data):
        drivers, returns = nonlinear_data
        solver = V4PINNSolver(m_drivers=3, n_assets=4)
        solver.fit(drivers, returns, n_epochs=20, patience=50)

        history = solver.train_history
        assert len(history) == 20
        assert "train_loss" in history[0]
        assert "val_loss" in history[0]

    def test_not_fitted_raises(self):
        solver = V4PINNSolver(m_drivers=3, n_assets=4)
        with pytest.raises(RuntimeError, match="not fitted"):
            solver.predict(np.zeros(3))

    def test_early_stopping(self, nonlinear_data):
        """With very small patience, should stop early."""
        drivers, returns = nonlinear_data
        solver = V4PINNSolver(m_drivers=3, n_assets=4)
        solver.fit(drivers, returns, n_epochs=1000, patience=5)

        # Should have stopped well before 1000
        assert len(solver.train_history) < 1000
