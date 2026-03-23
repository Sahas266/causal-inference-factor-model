"""Tests for filters/ekf.py — Kalman Filter."""

import numpy as np
import pytest

from causal_portfolio.filters.ekf import CPCMKalmanFilter


@pytest.fixture
def var1_data():
    """Synthetic VAR(1) process with known parameters."""
    rng = np.random.default_rng(42)
    m = 3
    T = 500

    # Stable transition matrix (spectral radius < 1)
    A_true = np.array([
        [0.8, 0.1, 0.0],
        [0.0, 0.7, 0.1],
        [0.1, 0.0, 0.6],
    ])
    Q_true = np.eye(m) * 0.1
    R_true = np.eye(m) * 0.05

    # Generate true states
    states = np.zeros((T, m))
    states[0] = rng.standard_normal(m)
    for t in range(1, T):
        states[t] = A_true @ states[t - 1] + rng.multivariate_normal(
            np.zeros(m), Q_true
        )

    # Generate noisy observations
    observations = states + rng.multivariate_normal(np.zeros(m), R_true, size=T)

    return states, observations, A_true, Q_true, R_true


class TestCPCMKalmanFilter:
    def test_fit_dynamics(self, var1_data):
        _, observations, _, _, _ = var1_data
        kf = CPCMKalmanFilter(m=3)
        kf.fit_dynamics(observations)
        assert kf._fitted
        assert kf.A.shape == (3, 3)
        assert kf.Q.shape == (3, 3)

    def test_spectral_radius_stable(self, var1_data):
        _, observations, _, _, _ = var1_data
        kf = CPCMKalmanFilter(m=3)
        kf.fit_dynamics(observations)
        assert kf._spectral_radius() < 1.0

    def test_filter_output_shape(self, var1_data):
        _, observations, _, _, _ = var1_data
        kf = CPCMKalmanFilter(m=3)
        kf.fit_dynamics(observations)
        filtered, covs = kf.filter(observations)
        assert filtered.shape == observations.shape
        assert covs.shape == (len(observations), 3, 3)

    def test_filter_reduces_noise(self, var1_data):
        """Filtered states should be closer to true states than raw obs."""
        states, observations, _, _, _ = var1_data
        kf = CPCMKalmanFilter(m=3)
        kf.fit_dynamics(observations)
        filtered, _ = kf.filter(observations)

        # MSE: filtered vs true should be < raw vs true
        mse_raw = np.mean((observations[50:] - states[50:]) ** 2)
        mse_filtered = np.mean((filtered[50:] - states[50:]) ** 2)
        assert mse_filtered < mse_raw, (
            f"Filtered MSE ({mse_filtered:.4f}) should be < raw MSE ({mse_raw:.4f})"
        )

    def test_predict_grows_uncertainty(self, var1_data):
        _, observations, _, _, _ = var1_data
        kf = CPCMKalmanFilter(m=3)
        kf.fit_dynamics(observations)
        filtered, covs = kf.filter(observations)

        preds, pred_covs = kf.predict(
            filtered[-1], covs[-1], n_steps=10
        )
        # Uncertainty should grow with horizon
        traces = [np.trace(pred_covs[t]) for t in range(10)]
        assert all(traces[i] <= traces[i + 1] for i in range(9))

    def test_predict_shape(self, var1_data):
        _, observations, _, _, _ = var1_data
        kf = CPCMKalmanFilter(m=3)
        kf.fit_dynamics(observations)
        filtered, covs = kf.filter(observations)

        preds, pred_covs = kf.predict(filtered[-1], covs[-1], n_steps=5)
        assert preds.shape == (5, 3)
        assert pred_covs.shape == (5, 3, 3)

    def test_handles_nan_observations(self, var1_data):
        _, observations, _, _, _ = var1_data
        kf = CPCMKalmanFilter(m=3)
        kf.fit_dynamics(observations)

        # Insert NaN gaps
        obs_with_gaps = observations.copy()
        obs_with_gaps[100:110] = np.nan

        filtered, covs = kf.filter(obs_with_gaps)
        # Should still produce output (predict-only during gaps)
        assert not np.isnan(filtered).any()

    def test_innovation_diagnostics(self, var1_data):
        _, observations, _, _, _ = var1_data
        kf = CPCMKalmanFilter(m=3)
        kf.fit_dynamics(observations)

        diag = kf.innovation_diagnostics(observations)
        # Normalized innovations should be approximately N(0,1)
        assert all(abs(m) < 0.3 for m in diag["mean"]), (
            f"Innovation mean should be ~0, got {diag['mean']}"
        )
        assert all(0.5 < s < 2.0 for s in diag["std"]), (
            f"Innovation std should be ~1, got {diag['std']}"
        )

    def test_not_fitted_raises(self):
        kf = CPCMKalmanFilter(m=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            kf.filter(np.zeros((10, 3)))

    def test_insufficient_data_raises(self):
        kf = CPCMKalmanFilter(m=3)
        with pytest.raises(ValueError, match="Need at least"):
            kf.fit_dynamics(np.zeros((3, 3)))
