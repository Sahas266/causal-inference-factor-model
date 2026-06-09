"""Wasserstein k-means (WK-means) market-regime clustering.

Port of Horvath, Issa & Muguruza (2021), "Clustering Market Regimes using the
Wasserstein Distance" (SSRN 3947905). A non-parametric, model-free alternative
to the Gaussian HMM in `regimes/hmm.py`: instead of assuming a latent Markov
chain with Gaussian emissions, it clusters the *distributions* of short return
segments directly, using the p-Wasserstein distance.

Pipeline (paper Section 1.3–2, Algorithm 1):
  1. Stream lift (Def 1.2): slice the log-return stream into M overlapping
     segments of length `h1` with sliding offset `h2`. Each segment, sorted, is
     an empirical measure µ_i with N=h1 atoms.
  2. Distance (Prop 2.5, eq 21): for two equal-size empirical measures with
     sorted atoms α, β,  W_p(µ,ν)^p = (1/N) Σ |α_i − β_i|^p. O(N log N) (sort).
  3. Barycenter / centroid aggregator (Prop 2.6, eq 22): per-order-statistic
     median across the cluster's sorted segments.
  4. k-means (Algorithm 1): assign each segment to the nearest centroid under
     W_p, recompute barycenters, iterate until the centroid movement loss
     (eq 23) < tol. Multi-start; keep the lowest total within-cluster W_p.
  5. Validation (Def 1.9): MMD self-similarity (biased MMD² with a Gaussian
     kernel) measures within-cluster homogeneity — lower = tighter regime.

Causal labelling: like the HMM wrapper, `predict_segment_labels` assigns each
segment to its nearest *fitted* centroid, and `rolling_fit_label` produces
strictly-causal per-day labels (fit on a trailing window, label the latest
segment) for walk-forward use.

Centroids are sorted by dispersion (mean |atom|) so label 0 is always the
calmest regime and the top label the most volatile — making labels comparable
across refits, mirroring the HMM's stress-ordering convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("cpcm.regimes.wkmeans")


# ── segmentation (stream lift, Def 1.2) ──────────────────────────────


def segment_stream(returns: np.ndarray, h1: int, h2: int) -> np.ndarray:
    """Lift a 1-D return stream into M overlapping segments of length h1.

    Offset h2 is the slide between consecutive segments (h2 < h1 => overlap;
    h2 == h1 => disjoint). Returns an (M, h1) array; row i is segment i.
    """
    returns = np.asarray(returns, dtype=float).reshape(-1)
    n = returns.shape[0]
    if h1 <= 0 or h2 <= 0 or h1 > n:
        return np.empty((0, h1))
    starts = list(range(0, n - h1 + 1, h2))
    return np.stack([returns[s:s + h1] for s in starts]) if starts else np.empty((0, h1))


# ── p-Wasserstein distance + barycenter (Prop 2.5 / 2.6) ─────────────


def wasserstein_distance_sorted(a_sorted: np.ndarray, b_sorted: np.ndarray, p: int = 1) -> float:
    """W_p between two equal-size empirical measures given SORTED atoms.

    W_p^p = (1/N) Σ |α_i − β_i|^p  (eq 21). Returns W_p (the p-th root).
    """
    diff = np.abs(a_sorted - b_sorted)
    if p == 1:
        return float(diff.mean())
    return float((diff ** p).mean()) ** (1.0 / p)


def barycenter_sorted(segments_sorted: np.ndarray) -> np.ndarray:
    """1-Wasserstein barycenter of equal-size measures (Prop 2.6, eq 22).

    Per-order-statistic median across the (already sorted) member segments.
    Input: (m, N) array of sorted segments. Output: (N,) sorted centroid atoms.
    """
    return np.median(segments_sorted, axis=0)


# ── MMD self-similarity (Def 1.9) ────────────────────────────────────


def _mmd2_biased(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Biased MMD² with a Gaussian kernel between 1-D samples x, y (eq 10/53)."""
    def k(a, b):
        d = a[:, None] - b[None, :]
        return np.exp(-(d * d) / (2.0 * sigma * sigma))
    return float(k(x, x).mean() + k(y, y).mean() - 2.0 * k(x, y).mean())


def mmd_self_similarity(segments: np.ndarray, sigma: float | None = None,
                        n_pairs: int = 200, seed: int = 0) -> float:
    """Within-cluster self-similarity score (Def 1.9): median pairwise MMD².

    Lower = the segments in this cluster are distributionally more alike
    (a tighter, more homogeneous regime). `segments` is (m, N) (unsorted ok).
    """
    m = segments.shape[0]
    if m < 2:
        return 0.0
    if sigma is None:
        # median heuristic on a flattened sub-sample
        flat = segments.reshape(-1)
        sample = flat if flat.size <= 2000 else np.random.default_rng(seed).choice(flat, 2000, replace=False)
        med = np.median(np.abs(sample[:, None] - sample[None, :]))
        sigma = float(med) if med > 1e-12 else 1.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(min(n_pairs, m * (m - 1) // 2)):
        i, j = rng.choice(m, 2, replace=False)
        vals.append(_mmd2_biased(segments[i], segments[j], sigma))
    return float(np.median(vals)) if vals else 0.0


# ── the classifier ───────────────────────────────────────────────────


@dataclass
class _FittedWK:
    centroids: np.ndarray   # (k, N) sorted centroid atoms, dispersion-ordered
    p: int


class WassersteinKMeans:
    """WK-means regime classifier (paper Algorithm 1)."""

    def __init__(self, n_states: int = 2, p: int = 1, h1: int = 21, h2: int = 1,
                 n_iter: int = 50, n_restarts: int = 5, tol: float = 1e-6,
                 random_state: int = 42):
        self.n_states = n_states
        self.p = p
        self.h1 = h1
        self.h2 = h2
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.tol = tol
        self.random_state = random_state
        self._fitted: _FittedWK | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted is not None

    def _dist_to_centroids(self, seg_sorted: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        return np.array([wasserstein_distance_sorted(seg_sorted, c, self.p) for c in centroids])

    def _run_once(self, segs_sorted: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        M = segs_sorted.shape[0]
        k = self.n_states
        centroids = segs_sorted[rng.choice(M, k, replace=False)].copy()
        labels = np.zeros(M, dtype=int)
        for _ in range(self.n_iter):
            # assignment
            new_labels = np.array([
                int(np.argmin(self._dist_to_centroids(s, centroids))) for s in segs_sorted
            ])
            # update (Wasserstein barycenter per cluster)
            new_centroids = centroids.copy()
            for l in range(k):
                members = segs_sorted[new_labels == l]
                if members.shape[0] > 0:
                    new_centroids[l] = barycenter_sorted(members)
            # loss = total centroid movement (eq 23)
            loss = sum(wasserstein_distance_sorted(centroids[l], new_centroids[l], self.p)
                       for l in range(k))
            centroids, labels = new_centroids, new_labels
            if loss < self.tol:
                break
        # total within-cluster W_p (inertia) for restart selection
        inertia = sum(self._dist_to_centroids(segs_sorted[i], centroids)[labels[i]]
                      for i in range(M))
        return centroids, labels, inertia

    def fit(self, returns: np.ndarray) -> "WassersteinKMeans":
        segs = segment_stream(returns, self.h1, self.h2)
        if segs.shape[0] < self.n_states:
            raise ValueError(f"only {segs.shape[0]} segments for {self.n_states} states")
        segs_sorted = np.sort(segs, axis=1)
        best = None
        for r in range(self.n_restarts):
            centroids, labels, inertia = self._run_once(segs_sorted, self.random_state + r)
            if best is None or inertia < best[2]:
                best = (centroids, labels, inertia)
        centroids = best[0]
        # dispersion-order: label 0 = calmest (smallest mean |atom|)
        order = np.argsort(np.abs(centroids).mean(axis=1))
        self._fitted = _FittedWK(centroids=centroids[order], p=self.p)
        logger.info("WK-means fitted: k=%d, inertia=%.4f (%d restarts)",
                    self.n_states, best[2], self.n_restarts)
        return self

    def predict_segment_labels(self, returns: np.ndarray) -> np.ndarray:
        """Label every segment of `returns` by nearest fitted centroid."""
        if self._fitted is None:
            raise RuntimeError("fit() first")
        segs = segment_stream(returns, self.h1, self.h2)
        segs_sorted = np.sort(segs, axis=1)
        return np.array([
            int(np.argmin(self._dist_to_centroids(s, self._fitted.centroids)))
            for s in segs_sorted
        ])

    def predict_latest_label(self, returns_window: np.ndarray) -> int:
        """Label the most recent length-h1 segment of a window (causal use)."""
        if self._fitted is None:
            raise RuntimeError("fit() first")
        w = np.asarray(returns_window, dtype=float).reshape(-1)
        seg = np.sort(w[-self.h1:])
        return int(np.argmin(self._dist_to_centroids(seg, self._fitted.centroids)))

    def centroids(self) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("fit() first")
        return self._fitted.centroids


# ── rolling causal labels (walk-forward) ─────────────────────────────


def rolling_fit_label(
    returns: pd.Series, window_size: int, refit_every: int = 21,
    n_states: int = 2, h1: int = 21, h2: int = 1, n_restarts: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """Strictly-causal per-day WK-means labels via rolling fit + latest-segment label.

    For each day t >= window_size: (re)fit WK-means on returns[t-window:t] at
    refit boundaries, then label the most recent h1-day segment ending at t.
    First `window_size` entries are NaN. Mirrors hmm.rolling_fit_decode.
    """
    r = returns.astype(float)
    vals = r.values
    T = len(vals)
    out = np.full(T, np.nan)
    if T < window_size:
        return pd.Series(out, index=r.index, name="wk_regime")

    model: WassersteinKMeans | None = None
    last_refit = -10**9
    for t in range(window_size, T):
        if model is None or (t - last_refit) >= refit_every:
            try:
                model = WassersteinKMeans(
                    n_states=n_states, h1=h1, h2=h2, n_restarts=n_restarts,
                    random_state=random_state,
                ).fit(vals[t - window_size:t])
                last_refit = t
            except Exception as e:
                logger.warning("WK-means refit failed at t=%d: %s", t, e)
                continue
        try:
            out[t] = model.predict_latest_label(vals[t - window_size:t + 1])
        except Exception:
            pass
    return pd.Series(out, index=r.index, name="wk_regime").astype("Int64")
