"""Tests for the Wasserstein k-means regime classifier (port of SSRN 3947905)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_portfolio.regimes.wkmeans import (
    WassersteinKMeans, barycenter_sorted, mmd_self_similarity,
    rolling_fit_label, segment_stream, wasserstein_distance_sorted,
)


# ── core math (paper formulas) ───────────────────────────────────────


def test_segment_overlap_count():
    r = np.arange(100.0)
    segs = segment_stream(r, h1=20, h2=1)
    assert segs.shape == (81, 20)   # 100-20+1 starts
    # disjoint
    segs2 = segment_stream(r, h1=20, h2=20)
    assert segs2.shape == (5, 20)


def test_wasserstein_p1_equals_mean_abs_sorted_diff():
    a = np.sort(np.array([0.0, 1.0, 2.0, 3.0]))
    b = np.sort(np.array([1.0, 1.0, 1.0, 1.0]))
    # mean |a-b| = (1+0+1+2)/4 = 1.0
    assert abs(wasserstein_distance_sorted(a, b, p=1) - 1.0) < 1e-12


def test_wasserstein_identical_is_zero():
    a = np.sort(np.random.default_rng(0).normal(size=50))
    assert wasserstein_distance_sorted(a, a, p=2) == 0.0


def test_wasserstein_p2():
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 4.0])
    # ((4+16)/2)^.5 = sqrt(10)
    assert abs(wasserstein_distance_sorted(a, b, p=2) - np.sqrt(10.0)) < 1e-12


def test_barycenter_is_per_orderstat_median():
    segs = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0], [0.0, 2.0, 9.0]])
    bary = barycenter_sorted(segs)
    np.testing.assert_allclose(bary, [0.0, 2.0, 4.0])


def test_mmd_self_similarity_lower_for_homogeneous():
    rng = np.random.default_rng(1)
    homo = rng.normal(0, 1, size=(20, 40))                       # all same dist
    hetero = np.vstack([rng.normal(0, 1, (10, 40)),
                        rng.normal(0, 5, (10, 40))])             # mixed scales
    assert mmd_self_similarity(homo, seed=1) < mmd_self_similarity(hetero, seed=1)


# ── clustering behaviour ─────────────────────────────────────────────


def test_separates_calm_and_volatile():
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, 400)
    vol = rng.normal(0, 0.05, 400)
    stream = np.concatenate([calm, vol])
    wk = WassersteinKMeans(n_states=2, h1=21, h2=5, n_restarts=5).fit(stream)
    labels = wk.predict_segment_labels(stream)
    # label 0 = calmest by construction; early segments calm, late volatile
    assert labels[:10].mean() < labels[-10:].mean()
    # centroid 1 should be more dispersed than centroid 0
    c = wk.centroids()
    assert np.abs(c[1]).mean() > np.abs(c[0]).mean()


def test_predict_latest_label_volatile():
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.005, 400)
    vol = rng.normal(0, 0.05, 400)
    wk = WassersteinKMeans(n_states=2, h1=21, h2=5, n_restarts=5).fit(
        np.concatenate([calm, vol]))
    # a fresh volatile window should map to the higher (non-zero) state
    test_window = rng.normal(0, 0.05, 30)
    assert wk.predict_latest_label(test_window) == 1


def test_rolling_labels_causal_shape():
    rng = np.random.default_rng(4)
    r = pd.Series(np.concatenate([rng.normal(0, 0.005, 300),
                                  rng.normal(0, 0.05, 300)]),
                  index=pd.date_range("2022-01-01", periods=600))
    lab = rolling_fit_label(r, window_size=252, refit_every=63,
                            n_states=2, h1=21, n_restarts=3)
    assert lab.index.equals(r.index)
    assert lab.iloc[:252].isna().all()          # warmup NaN
    assert lab.iloc[252:].notna().any()
