"""Tests for factors/combo_selector.py — Combo driver selection."""

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.factors.combo_selector import ComboDriverSelector


@pytest.fixture
def selector():
    return ComboDriverSelector()


@pytest.fixture
def synthetic_data():
    """Synthetic data with 3 true drivers and 4 noise factors.

    Asset returns are generated as: R = F_true @ loadings + noise.
    The Combo method should identify the 3 true factors.
    """
    rng = np.random.default_rng(42)
    T = 500
    n_assets = 8

    true_drivers = rng.standard_normal((T, 3))
    loadings = rng.standard_normal((3, n_assets)) * 0.5
    noise = rng.standard_normal((T, n_assets)) * 0.2

    returns_data = true_drivers @ loadings + noise
    noise_factors = rng.standard_normal((T, 4)) * 0.5
    all_candidates = np.column_stack([true_drivers, noise_factors])

    dates = pd.date_range("2023-01-01", periods=T, freq="D")
    returns_df = pd.DataFrame(
        returns_data,
        index=dates,
        columns=[f"asset_{i}" for i in range(n_assets)],
    )
    candidates_df = pd.DataFrame(
        all_candidates,
        index=dates,
        columns=[f"true_{i}" for i in range(3)] + [f"noise_{i}" for i in range(4)],
    )
    return returns_df, candidates_df


class TestComboDriverSelector:
    def test_selects_correct_number(self, selector, synthetic_data):
        returns_df, candidates_df = synthetic_data
        selected = selector.select(returns_df, candidates_df, m=3)
        assert len(selected) == 3

    def test_selects_true_drivers(self, selector, synthetic_data):
        """With strong signal, Combo should select the 3 true factors."""
        returns_df, candidates_df = synthetic_data
        selected = selector.select(returns_df, candidates_df, m=3)
        true_names = {"true_0", "true_1", "true_2"}
        assert set(selected) == true_names, (
            f"Expected {true_names}, got {set(selected)}"
        )

    def test_ranking_best_first(self, selector, synthetic_data):
        returns_df, candidates_df = synthetic_data
        ranking = selector.rank_all_subsets(returns_df, candidates_df, m=3)
        scores = [s for _, s in ranking]
        assert scores == sorted(scores), "Ranking should be ascending by score"

    def test_total_subsets_count(self, selector, synthetic_data):
        returns_df, candidates_df = synthetic_data
        ranking = selector.rank_all_subsets(returns_df, candidates_df, m=3)
        # C(7, 3) = 35
        from math import comb
        expected = comb(candidates_df.shape[1], 3)
        assert len(ranking) == expected

    def test_m_exceeds_candidates_raises(self, selector, synthetic_data):
        returns_df, candidates_df = synthetic_data
        with pytest.raises(ValueError, match="exceeds"):
            selector.select(returns_df, candidates_df, m=10)

    def test_commonality_score_positive(self, selector):
        rng = np.random.default_rng(0)
        returns = rng.standard_normal((100, 5))
        drivers = rng.standard_normal((100, 2))
        score = selector._commonality_score(returns, drivers)
        assert score > 0

    def test_better_drivers_lower_score(self, selector):
        """True drivers should yield a lower commonality score than noise."""
        rng = np.random.default_rng(42)
        T = 300
        true_F = rng.standard_normal((T, 2))
        loadings = rng.standard_normal((2, 5))
        returns = true_F @ loadings + rng.standard_normal((T, 5)) * 0.2
        noise_F = rng.standard_normal((T, 2))

        score_true = selector._commonality_score(returns, true_F)
        score_noise = selector._commonality_score(returns, noise_F)
        assert score_true < score_noise
