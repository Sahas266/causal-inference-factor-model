"""V1 Linear Solver: Ridge regression with Ledoit-Wolf covariance shrinkage.

The simplest CPCM solver variant. Maps drivers to returns via:
    returns = drivers @ beta + epsilon

The Jacobian dA/dF is constant (= beta), making this the baseline
against which V4 PINN is compared.

Reference: Section 4.1 of arXiv:2509.09585v2
"""

import logging

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

from .base import CPCMSolver

logger = logging.getLogger("cpcm.solvers")


class V1LinearSolver(CPCMSolver):
    """Ridge OLS solver with Ledoit-Wolf residual covariance shrinkage."""

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Ridge regularization strength.
        """
        self.alpha = alpha
        self.beta_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.residual_cov_: np.ndarray | None = None
        self._fitted = False

    def fit(self, drivers: np.ndarray, returns: np.ndarray) -> None:
        """Fit Ridge regression: returns = drivers @ beta + intercept + epsilon.

        Args:
            drivers: (T, m) driver values.
            returns: (T, n_assets) asset returns.
        """
        # Drop rows with any NaN
        valid = ~(np.isnan(drivers).any(axis=1) | np.isnan(returns).any(axis=1))
        D = drivers[valid]
        R = returns[valid]

        if len(D) < max(D.shape[1], R.shape[1]) + 5:
            raise ValueError(
                f"Insufficient data: {len(D)} obs for {D.shape[1]} drivers "
                f"and {R.shape[1]} assets"
            )

        # Ridge regression
        reg = Ridge(alpha=self.alpha, fit_intercept=True)
        reg.fit(D, R)
        self.beta_ = reg.coef_.T  # (m, n_assets) → transpose to standard form
        self.intercept_ = reg.intercept_  # (n_assets,)

        # Residual covariance with Ledoit-Wolf shrinkage
        residuals = R - reg.predict(D)
        lw = LedoitWolf().fit(residuals)
        self.residual_cov_ = lw.covariance_

        self._fitted = True
        self._n_assets = R.shape[1]
        self._m_drivers = D.shape[1]
        self._diagnostics = self._compute_diagnostics(D, R, reg)

        logger.info(
            f"V1 fitted: {self._m_drivers} drivers → {self._n_assets} assets, "
            f"mean R²={self._diagnostics['mean_r2']:.4f}"
        )

    def predict(self, drivers: np.ndarray) -> np.ndarray:
        """Predict expected returns: drivers @ beta.T + intercept."""
        self._check_fitted()
        if drivers.ndim == 1:
            drivers = drivers.reshape(1, -1)
        return drivers @ self.beta_ + self.intercept_

    def jacobian(self, drivers: np.ndarray) -> np.ndarray:
        """Return constant Jacobian = beta.T (n_assets, m)."""
        self._check_fitted()
        return self.beta_.T  # (n_assets, m)

    @property
    def diagnostics(self) -> dict:
        """Fitting diagnostics: R² per asset, mean R², residual std."""
        self._check_fitted()
        return self._diagnostics

    def _compute_diagnostics(self, D, R, reg) -> dict:
        predicted = reg.predict(D)
        residuals = R - predicted

        # Per-asset R²
        ss_res = (residuals ** 2).sum(axis=0)
        ss_tot = ((R - R.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1 - ss_res / np.where(ss_tot > 1e-15, ss_tot, 1.0)

        return {
            "r2_per_asset": r2,
            "mean_r2": float(r2.mean()),
            "residual_std": float(residuals.std()),
            "n_observations": len(D),
        }

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Solver not fitted. Call fit() first.")
