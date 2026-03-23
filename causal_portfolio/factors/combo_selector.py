"""Combo driver selection method from the CPCM paper.

Selects m drivers from a candidate set that minimize residual commonality —
the largest eigenvalue of the residual correlation matrix after regressing
asset returns on the candidate subset. This identifies the causal drivers
that best explain cross-sectional dependence.

Reference: Section 3.2 of arXiv:2509.09585v2
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("cpcm.factors")


class ComboDriverSelector:
    """Select optimal causal drivers via residual commonality minimization."""

    def select(
        self,
        returns: pd.DataFrame,
        candidates: pd.DataFrame,
        m: int = 3,
    ) -> list[str]:
        """Select the m drivers that minimize residual commonality.

        Args:
            returns: DataFrame of asset returns (n_obs x n_assets).
            candidates: DataFrame of candidate factors (n_obs x n_candidates).
            m: Number of drivers to select.

        Returns:
            List of m selected driver column names.
        """
        ranking = self.rank_all_subsets(returns, candidates, m)
        best = ranking[0]
        logger.info(
            f"Combo selected m={m} drivers: {best[0]} "
            f"(commonality score: {best[1]:.4f})"
        )
        return list(best[0])

    def rank_all_subsets(
        self,
        returns: pd.DataFrame,
        candidates: pd.DataFrame,
        m: int = 3,
    ) -> list[tuple[tuple[str, ...], float]]:
        """Score and rank all m-subsets by commonality score (ascending).

        Returns:
            List of (subset_tuple, score) sorted ascending (best first).
        """
        # Align indices and drop NaN rows
        common = returns.index.intersection(candidates.index)
        R = returns.loc[common].dropna()
        C = candidates.loc[common].reindex(R.index)

        col_names = list(C.columns)
        n_candidates = len(col_names)

        if m > n_candidates:
            raise ValueError(
                f"m={m} exceeds number of candidates ({n_candidates})"
            )

        results: list[tuple[tuple[str, ...], float]] = []

        for subset_idx in combinations(range(n_candidates), m):
            subset_names = tuple(col_names[i] for i in subset_idx)
            score = self._commonality_score(
                R.values, C.values[:, list(subset_idx)]
            )
            results.append((subset_names, score))

        results.sort(key=lambda x: x[1])
        return results

    @staticmethod
    def _commonality_score(returns: np.ndarray, drivers: np.ndarray) -> float:
        """Compute commonality score = lambda_max of residual correlation matrix.

        Args:
            returns: (T, n_assets) array of asset returns.
            drivers: (T, m) array of candidate driver values.

        Returns:
            Largest eigenvalue of the residual correlation matrix.
        """
        T, n_assets = returns.shape

        # Drop rows where any driver is NaN
        valid = ~np.isnan(drivers).any(axis=1) & ~np.isnan(returns).any(axis=1)
        R_clean = returns[valid]
        D_clean = drivers[valid]

        if len(R_clean) < max(n_assets, drivers.shape[1]) + 5:
            return float("inf")

        # Regress each asset return on the driver subset
        reg = LinearRegression()
        reg.fit(D_clean, R_clean)
        residuals = R_clean - reg.predict(D_clean)

        # Residual correlation matrix
        residual_std = residuals.std(axis=0, keepdims=True)
        residual_std = np.where(residual_std < 1e-15, 1.0, residual_std)
        residuals_normed = residuals / residual_std
        corr = (residuals_normed.T @ residuals_normed) / len(residuals_normed)

        # Largest eigenvalue = remaining shared structure
        eigenvalues = linalg.eigvalsh(corr)
        return float(eigenvalues[-1])


def run_combo_selection(
    returns: pd.DataFrame,
    candidates: pd.DataFrame,
    m: int = 3,
    top_k: int = 5,
) -> None:
    """Run Combo selection and print results."""
    selector = ComboDriverSelector()
    ranking = selector.rank_all_subsets(returns, candidates, m)

    print(f"\nCombo Driver Selection (m={m}, {len(ranking)} subsets evaluated)")
    print("=" * 60)
    for i, (subset, score) in enumerate(ranking[:top_k]):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. {str(list(subset)):40s}  score={score:.4f}{marker}")
    print()


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    T = 500
    n_assets = 10

    # True drivers: factors 0, 1, 2 with known loadings
    true_drivers = np.random.randn(T, 3)
    loadings = np.random.randn(3, n_assets) * 0.5
    noise = np.random.randn(T, n_assets) * 0.3
    returns_data = true_drivers @ loadings + noise

    # Candidates: 3 true + 4 noise factors
    noise_factors = np.random.randn(T, 4) * 0.5
    all_candidates = np.column_stack([true_drivers, noise_factors])

    returns_df = pd.DataFrame(
        returns_data,
        columns=[f"asset_{i}" for i in range(n_assets)],
    )
    candidates_df = pd.DataFrame(
        all_candidates,
        columns=[f"true_{i}" for i in range(3)] + [f"noise_{i}" for i in range(4)],
    )

    run_combo_selection(returns_df, candidates_df, m=3)
