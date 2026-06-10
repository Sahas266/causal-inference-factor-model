"""Manifold-constrained portfolio optimization.

Projects portfolio weights onto the tangent space of the driver-return
Jacobian manifold, then solves mean-variance with constraints.

The key insight: weights should lie in the span of the Jacobian dA/dF,
ensuring the portfolio responds to driver movements rather than noise.

Reference: Section 4.5 of arXiv:2509.09585v2
"""

import logging

import numpy as np
from sklearn.covariance import LedoitWolf

from causal_portfolio.solvers.base import CPCMSolver

logger = logging.getLogger("cpcm.optimizer")


class ManifoldOptimizer:
    """Manifold-constrained mean-variance portfolio optimizer."""

    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_weight: float = 0.25,
        long_only: bool = False,
    ):
        """
        Args:
            risk_aversion: Lambda for mean-variance trade-off.
            max_weight: Maximum absolute weight per asset.
            long_only: If True, constrain weights >= 0.
        """
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.long_only = long_only

    def optimize(
        self,
        solver: CPCMSolver,
        F_current: np.ndarray,
        cov: np.ndarray,
        mu_override: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute optimal weights at current driver state.

        Steps:
        1. Get expected returns from solver
        2. Get Jacobian and compute tangent space projection
        3. Solve mean-variance
        4. Project onto tangent space
        5. Clip and normalize

        Args:
            solver: Fitted CPCM solver.
            F_current: (m,) current driver state.
            cov: (n_assets, n_assets) return covariance matrix.
            mu_override: optional (n_assets,) expected-return vector to use
                instead of solver.predict — e.g. an MoE posterior blend. The
                solver still supplies the Jacobian for the tangent projection.

        Returns:
            (n_assets,) optimal portfolio weights.
        """
        # Expected returns
        mu = (mu_override if mu_override is not None
              else solver.predict(F_current)).flatten()
        n = len(mu)

        # Jacobian-based tangent space
        J = solver.jacobian(F_current)  # (n_assets, m)
        projector = self._tangent_projector(J)

        # Mean-variance solution: w = (1/λ) Σ^{-1} μ
        cov_reg = cov + np.eye(n) * 1e-6
        cov_inv = np.linalg.inv(cov_reg)
        w_mv = (1.0 / self.risk_aversion) * cov_inv @ mu

        # Project onto tangent space
        w_manifold = projector @ w_mv

        # Apply constraints
        w_final = self._apply_constraints(w_manifold)

        return w_final

    def _tangent_projector(self, J: np.ndarray) -> np.ndarray:
        """Compute projection matrix onto tangent space of J.

        P = U U^T where U = left singular vectors of J.

        Args:
            J: (n_assets, m) Jacobian matrix.

        Returns:
            (n_assets, n_assets) projection matrix.
        """
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        # Keep singular vectors with non-negligible values
        k = np.sum(S > 1e-10 * S[0])
        return U[:, :k] @ U[:, :k].T

    def _apply_constraints(self, w: np.ndarray) -> np.ndarray:
        """Clip weights and normalize."""
        if self.long_only:
            w = np.maximum(w, 0.0)

        # Clip to max weight
        w = np.clip(w, -self.max_weight, self.max_weight)

        # Normalize to sum to 1 (fully invested)
        w_sum = np.sum(np.abs(w))
        if w_sum > 1e-10:
            w = w / w_sum

        return w


def estimate_covariance(
    returns: np.ndarray,
    method: str = "ledoit_wolf",
) -> np.ndarray:
    """Estimate return covariance matrix.

    Args:
        returns: (T, n_assets) return array.
        method: "ledoit_wolf" (shrinkage) or "sample".

    Returns:
        (n_assets, n_assets) covariance matrix.
    """
    valid = ~np.isnan(returns).any(axis=1)
    R = returns[valid]

    if method == "ledoit_wolf":
        return LedoitWolf().fit(R).covariance_
    else:
        return np.cov(R, rowvar=False)
