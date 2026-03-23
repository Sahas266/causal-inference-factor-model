"""Tests for factors/builder.py — factor computation logic."""

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.factors.builder import (
    build_all_factors,
    z_score,
    rolling_std,
    _compute_liq_flow,
    _compute_stable_flow,
    _compute_cex_dex_flow,
)


@pytest.fixture
def sample_panel():
    """Panel with realistic column names and data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    data = {
        "aave_tvl_usd": rng.uniform(5e9, 10e9, 100),
        "uni_tvl_usd": rng.uniform(3e9, 7e9, 100),
        "crv_tvl_usd": rng.uniform(1e9, 3e9, 100),
        "usdc_SplyCur": rng.uniform(20e9, 30e9, 100),
        "usdt_SplyCur": rng.uniform(80e9, 90e9, 100),
        "eth_avg_gas_price_gwei": rng.uniform(10, 100, 100),
        "eth_staking_apr": rng.uniform(3.0, 5.0, 100),
        "eth_stddev_base_fee_gwei": rng.uniform(1, 20, 100),
        "eth_cex_netflow_usd": rng.normal(0, 1e8, 100),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_macro():
    """Macro data matching FRED series."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "dff": rng.uniform(4.0, 5.5, 100),
            "dgs10": rng.uniform(3.5, 4.5, 100),
            "vixcls": rng.uniform(12, 30, 100),
            "t10y2y": rng.uniform(-1.0, 0.5, 100),
            "cpiaucsl": rng.uniform(300, 310, 100),
            "m2sl": rng.uniform(20e3, 21e3, 100),
            "dtwexbgs": rng.uniform(100, 110, 100),
        },
        index=dates,
    )


class TestZScore:
    def test_mean_zero(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = z_score(s)
        assert abs(z.mean()) < 1e-10

    def test_unit_std(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = z_score(s)
        # std computed with ddof=1 by pandas, z_score uses ddof=1
        assert abs(z.std() - 1.0) < 0.1

    def test_handles_nan(self):
        s = pd.Series([1.0, np.nan, 3.0, 4.0])
        z = z_score(s)
        assert np.isnan(z.iloc[1])
        assert not np.isnan(z.iloc[0])

    def test_constant_series(self):
        s = pd.Series([5.0, 5.0, 5.0])
        z = z_score(s)
        assert (z == 0.0).all()


class TestRollingStd:
    def test_output_length(self):
        s = pd.Series(range(20), dtype=float)
        rs = rolling_std(s, window=7)
        assert len(rs) == len(s)

    def test_initial_nan(self):
        s = pd.Series(range(10), dtype=float)
        rs = rolling_std(s, window=5)
        # First few values should be NaN (not enough data for window)
        assert np.isnan(rs.iloc[0])


class TestLiqFlow:
    def test_produces_series(self, sample_panel):
        result = _compute_liq_flow(sample_panel)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_panel)

    def test_first_value_nan(self, sample_panel):
        """diff() produces NaN for first observation."""
        result = _compute_liq_flow(sample_panel)
        assert np.isnan(result.iloc[0])

    def test_z_scored(self, sample_panel):
        result = _compute_liq_flow(sample_panel)
        valid = result.dropna()
        assert abs(valid.mean()) < 0.2  # approximately zero


class TestStableFlow:
    def test_produces_series(self, sample_panel):
        result = _compute_stable_flow(sample_panel)
        assert isinstance(result, pd.Series)

    def test_missing_columns(self):
        panel = pd.DataFrame({"unrelated": [1, 2, 3]})
        result = _compute_stable_flow(panel)
        assert result.isna().all()


class TestBuildAllFactors:
    def test_returns_dataframe(self, sample_panel, sample_macro):
        factors = build_all_factors(sample_panel, sample_macro)
        assert isinstance(factors, pd.DataFrame)

    def test_expected_columns(self, sample_panel, sample_macro):
        factors = build_all_factors(sample_panel, sample_macro)
        # Should have global + macro factors (minus all-NaN ones)
        assert "liq_flow" in factors.columns
        assert "vixcls" in factors.columns

    def test_drops_all_nan_columns(self, sample_panel, sample_macro):
        factors = build_all_factors(sample_panel, sample_macro)
        for col in factors.columns:
            assert not factors[col].isna().all()

    def test_index_aligned(self, sample_panel, sample_macro):
        factors = build_all_factors(sample_panel, sample_macro)
        assert factors.index.equals(sample_panel.index)
