"""Extended Kalman Filter for CPCM latent driver state estimation.

Estimates the true driver state from noisy observations using a
VAR(1) state-space model:

    State transition:  x_t = A * x_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation:       y_t = H * x_t + v_t,       v_t ~ N(0, R)

For the linear case (H=I, linear transition), this reduces to a
standard Kalman Filter. The "Extended" structure allows future
nonlinear extensions (regime-switching VAR, etc.).

Reference: Section 3.4 of arXiv:2509.09585v2
"""

import logging

import numpy as np
from numpy.linalg import inv

logger = logging.getLogger("cpcm.filters")


class CPCMKalmanFilter:
    """Kalman Filter for CPCM driver state estimation."""

    def __init__(self, m: int):
        """
        Args:
            m: State dimension (number of drivers).
        """
        self.m = m
        self.A: np.ndarray | None = None  # (m, m) transition matrix
        self.Q: np.ndarray | None = None  # (m, m) process noise covariance
        self.R: np.ndarray | None = None  # (m, m) observation noise covariance
        self.H = np.eye(m)                # (m, m) observation matrix (identity)
        self._fitted = False

    def fit_dynamics(
        self,
        drivers_history: np.ndarray,
        obs_noise_scale: float = 0.1,
    ) -> None:
        """Estimate VAR(1) dynamics and noise covariances from data.

        Fits x_t = A * x_{t-1} + w_t via OLS, then estimates Q from
        residuals and sets R as a fraction of Q.

        Args:
            drivers_history: (T, m) array of historical driver observations.
            obs_noise_scale: R = obs_noise_scale * diag(var(drivers)).
        """
        # Drop NaN rows
        valid = ~np.isnan(drivers_history).any(axis=1)
        X = drivers_history[valid]

        if len(X) < self.m + 5:
            raise ValueError(f"Need at least {self.m + 5} observations, got {len(X)}")

        # OLS for VAR(1): x_t = A * x_{t-1}
        X_lag = X[:-1]  # (T-1, m)
        X_cur = X[1:]   # (T-1, m)

        # A = (X_cur' X_lag) (X_lag' X_lag)^{-1}
        XtX = X_lag.T @ X_lag + np.eye(self.m) * 1e-6  # ridge for stability
        XtY = X_lag.T @ X_cur
        self.A = np.linalg.solve(XtX, XtY).T  # (m, m)

        # Process noise from residuals
        residuals = X_cur - X_lag @ self.A.T
        self.Q = np.cov(residuals, rowvar=False)

        # Observation noise: fraction of marginal variance
        self.R = np.diag(np.var(X, axis=0)) * obs_noise_scale

        self._fitted = True
        logger.info(
            f"EKF dynamics fitted: A spectral radius={self._spectral_radius():.4f}"
        )

    def filter(
        self,
        observations: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run forward Kalman filter pass.

        Args:
            observations: (T, m) array of noisy driver observations.
            x0: Initial state estimate. Defaults to first observation.
            P0: Initial state covariance. Defaults to Q.

        Returns:
            filtered_states: (T, m) filtered state estimates x_{t|t}.
            covariances: (T, m, m) state covariance matrices P_{t|t}.
        """
        self._check_fitted()
        T = len(observations)
        m = self.m

        # Initialize
        if x0 is None:
            x0 = observations[0].copy()
        if P0 is None:
            P0 = self.Q.copy()

        filtered_states = np.zeros((T, m))
        covariances = np.zeros((T, m, m))

        x = x0.copy()
        P = P0.copy()

        for t in range(T):
            y = observations[t]

            if np.isnan(y).any():
                # Missing observation: predict only (no update)
                x = self.A @ x
                P = self.A @ P @ self.A.T + self.Q
            else:
                # ── Predict ──
                x_pred = self.A @ x
                P_pred = self.A @ P @ self.A.T + self.Q

                # ── Update ──
                innovation = y - self.H @ x_pred
                S = self.H @ P_pred @ self.H.T + self.R  # innovation covariance
                K = P_pred @ self.H.T @ inv(S)           # Kalman gain

                x = x_pred + K @ innovation
                P = (np.eye(m) - K @ self.H) @ P_pred

            filtered_states[t] = x
            covariances[t] = P

        return filtered_states, covariances

    def predict(
        self,
        x_current: np.ndarray,
        P_current: np.ndarray,
        n_steps: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast n steps ahead from current state.

        Returns:
            predictions: (n_steps, m) predicted states.
            uncertainties: (n_steps, m, m) predicted covariances.
        """
        self._check_fitted()
        predictions = np.zeros((n_steps, self.m))
        uncertainties = np.zeros((n_steps, self.m, self.m))

        x = x_current.copy()
        P = P_current.copy()

        for t in range(n_steps):
            x = self.A @ x
            P = self.A @ P @ self.A.T + self.Q
            predictions[t] = x
            uncertainties[t] = P

        return predictions, uncertainties

    def innovation_diagnostics(
        self,
        observations: np.ndarray,
    ) -> dict:
        """Compute filter diagnostics.

        Returns:
            Dictionary with innovation statistics (should be ~N(0,1) if
            model is well-specified).
        """
        self._check_fitted()
        T = len(observations)
        innovations = np.zeros((T, self.m))

        x = observations[0].copy()
        P = self.Q.copy()

        for t in range(1, T):
            y = observations[t]
            if np.isnan(y).any():
                x = self.A @ x
                P = self.A @ P @ self.A.T + self.Q
                continue

            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q
            S = self.H @ P_pred @ self.H.T + self.R

            innovation = y - self.H @ x_pred
            # Normalized innovation
            S_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(S)))
            innovations[t] = S_inv_sqrt @ innovation

            K = P_pred @ self.H.T @ inv(S)
            x = x_pred + K @ (y - self.H @ x_pred)
            P = (np.eye(self.m) - K @ self.H) @ P_pred

        valid_innov = innovations[1:]  # skip first
        return {
            "mean": valid_innov.mean(axis=0),
            "std": valid_innov.std(axis=0),
            "autocorr_lag1": np.array([
                np.corrcoef(valid_innov[:-1, i], valid_innov[1:, i])[0, 1]
                for i in range(self.m)
            ]),
        }

    def _spectral_radius(self) -> float:
        """Spectral radius of A (should be < 1 for stability)."""
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Filter not fitted. Call fit_dynamics() first.")
