"""Correlation-based portfolio construction — Hierarchical Risk Parity (HRP).

López de Prado's HRP (2016) builds weights from the return *correlation
structure* alone — no expected-return forecast, no matrix inversion (which is
what makes classical mean-variance blow up on noisy crypto covariances). Steps:

  1. Correlation -> distance:  d_ij = sqrt((1 - rho_ij)/2).
  2. Hierarchical (single-linkage) clustering on that distance.
  3. Quasi-diagonalization: reorder assets by the dendrogram so correlated
     assets sit adjacent.
  4. Recursive bisection: split the ordered list, allocate between halves
     inverse to each half's cluster variance, recurse.

It is a pure *risk-diversification* allocator. Here it serves two roles:
  - a standalone long-only book to compare against BH BTC, and
  - a covariance-aware weighting overlay that needs no return forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Correlation matrix -> distance matrix d = sqrt((1 - rho)/2)."""
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Return leaf order from a linkage matrix (López de Prado)."""
    link = link.astype(int)
    n = link[-1, 3]  # total original observations
    order = [link[-1, 0], link[-1, 1]]
    while max(order) >= n:  # while there are still merged clusters to expand
        new = []
        for x in order:
            if x < n:
                new.append(x)
            else:
                row = link[x - n]
                new.append(row[0])
                new.append(row[1])
        order = new
    return [int(x) for x in order]


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_left = _cluster_var(cov, left)
            v_right = _cluster_var(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            for idx in left:
                w[idx] *= alpha
            for idx in right:
                w[idx] *= (1.0 - alpha)
    return w


def hrp_weights(returns: pd.DataFrame, long_only: bool = True) -> pd.Series:
    """Hierarchical Risk Parity weights from a (T, n) returns frame.

    Returns a weight Series summing to 1 over the asset columns.
    """
    clean = returns.dropna()
    if clean.shape[1] == 1:
        return pd.Series([1.0], index=clean.columns)
    cov = np.cov(clean.values, rowvar=False)
    corr = np.corrcoef(clean.values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = correlation_distance(corr)
    link = linkage(squareform(dist, checks=False), method="single")
    order = _quasi_diag(link)
    w = _recursive_bisection(cov, order)
    w = np.clip(w, 0.0, None) if long_only else w
    total = w.sum()
    if total > 1e-12:
        w = w / total
    return pd.Series(w, index=clean.columns)
