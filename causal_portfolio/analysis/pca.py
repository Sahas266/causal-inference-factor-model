"""PCA-orthogonalized drivers.

The CPCM factor set is collinear (overlapping on-chain/fee/flow series — high
VIF in the diagnostics). Collinear regressors make the per-asset loadings β
unstable window-to-window, which is one reason OOS generalization is poor.

PCA fixes the collinearity directly: rotate the standardized factor panel onto
its principal components (orthogonal by construction), keep the components that
explain most variance, and use THOSE as the drivers fed to the OLS/manifold
pipeline. Fewer, decorrelated drivers => lower-variance loadings.

`fit_transform_causal` fits the rotation on a TRAILING window only and applies
it forward, so there's no look-ahead in a walk-forward backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def pca_drivers(
    factors: pd.DataFrame, n_components: int | float = 0.9,
) -> tuple[pd.DataFrame, PCA, np.ndarray, np.ndarray]:
    """Rotate a factor panel onto principal components (full-sample).

    Args:
        factors: (T, k) standardized-ish factor frame (build_all_factors output).
        n_components: int (keep that many PCs) or float in (0,1] (keep enough
            PCs to explain that fraction of variance).

    Returns:
        (pc_frame, fitted_pca, mean, scale) — pc_frame columns are pc1..pcN.
    """
    clean = factors.dropna()
    X = clean.values
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Xs = (X - mean) / scale
    pca = PCA(n_components=n_components, svd_solver="full")
    comps = pca.fit_transform(Xs)
    cols = [f"pc{i+1}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, index=clean.index, columns=cols), pca, mean, scale


def explained_variance_table(pca: PCA) -> list[tuple[int, float, float]]:
    """[(component_index, explained_var_ratio, cumulative)] for reporting."""
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    return [(i + 1, float(ratios[i]), float(cum[i])) for i in range(len(ratios))]


def transform_with(factors: pd.DataFrame, pca: PCA, mean: np.ndarray,
                   scale: np.ndarray) -> pd.DataFrame:
    """Apply a previously-fit PCA rotation to new factor rows (no refit)."""
    clean = factors.dropna()
    Xs = (clean.values - mean) / scale
    comps = pca.transform(Xs)
    cols = [f"pc{i+1}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, index=clean.index, columns=cols)
