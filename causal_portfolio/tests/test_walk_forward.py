"""Tests for the walk-forward validation harness (pure, synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.validation.walk_forward import (
    compare_variants,
    evaluate,
    make_test_folds,
)


# ── make_test_folds ─────────────────────────────────────────────────


def test_folds_partition_non_overlapping():
    folds = make_test_folds(n=1000, test_days=126, purge_days=21,
                            embargo_days=5, min_fold_obs=40)
    # Folds must be non-overlapping and ordered
    for (a0, e0), (a1, e1) in zip(folds, folds[1:]):
        assert e0 <= a1, "folds overlap"
        assert a0 < e0
    # ~1000 / (126+5) ≈ 7 folds
    assert 5 <= len(folds) <= 9


def test_folds_purge_trims_start():
    folds = make_test_folds(n=300, test_days=100, purge_days=10,
                            embargo_days=0, min_fold_obs=10)
    # First fold starts at purge offset
    assert folds[0][0] == 10


def test_folds_drop_short_trailing():
    # n=150, test=100 -> fold1 [0,100), then [100,150)=50 raw, minus purge 20 = 30 < min 40 -> dropped
    folds = make_test_folds(n=150, test_days=100, purge_days=20,
                            embargo_days=0, min_fold_obs=40)
    assert len(folds) == 1


def test_folds_empty_when_too_short():
    assert make_test_folds(n=30, test_days=126, purge_days=21,
                           embargo_days=5, min_fold_obs=40) == []


# ── evaluate ────────────────────────────────────────────────────────


def _series(vals, start="2022-01-01"):
    idx = pd.date_range(start, periods=len(vals))
    return pd.Series(vals, index=idx)


def test_evaluate_strategy_beating_benchmark():
    n = 800
    rng = np.random.default_rng(0)
    # Strategy: small positive drift; benchmark: zero drift
    strat = _series(rng.normal(0.002, 0.01, n))
    bench = _series(rng.normal(0.0, 0.01, n))
    rep = evaluate(strat, bench, test_days=126, purge_days=21,
                   embargo_days=5, min_fold_obs=40)
    assert rep.n_folds >= 4
    # Strategy with real positive drift should win most folds
    assert rep.win_rate > 0.6
    assert rep.median_sharpe > rep.bench_median_sharpe


def test_evaluate_no_edge_winrate_near_half():
    n = 1200
    rng = np.random.default_rng(1)
    # Same distribution → no edge → win-rate should be roughly balanced
    strat = _series(rng.normal(0.0, 0.02, n))
    bench = _series(rng.normal(0.0, 0.02, n))
    rep = evaluate(strat, bench)
    assert 0.0 <= rep.win_rate <= 1.0
    # not a strong systematic winner
    assert rep.win_rate < 0.9


def test_evaluate_aligns_indices():
    # Mismatched but overlapping indices → evaluated on the intersection
    strat = _series(np.full(500, 0.001), start="2022-01-01")
    bench = _series(np.full(500, 0.0), start="2022-03-01")  # offset
    rep = evaluate(strat, bench, test_days=100, min_fold_obs=20)
    assert rep.n_folds >= 1
    # All folds: strat (+) beats bench (0)
    assert rep.win_rate == 1.0


def test_fold_metrics_populated():
    strat = _series(np.full(400, 0.001))
    bench = _series(np.full(400, 0.0005))
    rep = evaluate(strat, bench, test_days=100, min_fold_obs=20)
    f = rep.folds[0]
    assert f.n_obs > 0
    assert f.strat_total > f.bench_total
    assert f.beats_bench is True


# ── compare_variants ────────────────────────────────────────────────


def test_compare_ranks_by_winrate():
    n = 900
    rng = np.random.default_rng(2)
    bench = _series(rng.normal(0.0, 0.015, n))
    variants = {
        "good": _series(rng.normal(0.003, 0.015, n)),   # strong drift
        "flat": _series(rng.normal(0.0, 0.015, n)),     # no edge
        "bad": _series(rng.normal(-0.003, 0.015, n)),   # negative
    }
    out = compare_variants(variants, bench, win_rate_bar=0.8)
    names_ranked = [r[0] for r in out["ranking"]]
    assert names_ranked[0] == "good"
    assert names_ranked[-1] == "bad"
    assert "good" in out["passers"]
    assert "bad" not in out["passers"]
    assert out["n_variants"] == 3
    assert "3 variants tested" in out["multiple_testing_note"]


def test_compare_winrate_bar_filters():
    n = 600
    rng = np.random.default_rng(3)
    bench = _series(rng.normal(0.0, 0.02, n))
    variants = {"flat": _series(rng.normal(0.0, 0.02, n))}
    out = compare_variants(variants, bench, win_rate_bar=0.99)
    # A no-edge variant should not clear a 99% bar
    assert out["passers"] == []
