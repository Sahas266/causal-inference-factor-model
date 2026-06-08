"""Tests for factors/instruments.py — IV construction (port of instruments.rs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.factors.instruments import (
    build_instruments,
    _diff,
    _lag,
    _z_score_abs,
    _lagged_z_score_spike,
    _find_columns,
)


# ── helpers ──────────────────────────────────────────────────────────


class TestDiff:
    def test_first_is_nan(self):
        d = _diff(np.array([1.0, 3.0, 6.0]))
        assert np.isnan(d[0])
        assert d[1] == 2.0 and d[2] == 3.0

    def test_nan_propagates(self):
        d = _diff(np.array([1.0, np.nan, 3.0]))
        assert np.isnan(d[1]) and np.isnan(d[2])


class TestZScoreAbs:
    def test_flags_magnitude(self):
        # A large-magnitude outlier should get a large |z|
        x = np.array([1.0, -1.0, 1.0, -1.0, 50.0])
        z = _z_score_abs(x)
        assert z[-1] == max(z, key=abs)

    def test_constant_returns_zeros(self):
        z = _z_score_abs(np.array([2.0, 2.0, 2.0]))
        assert np.allclose(z, 0.0)

    def test_all_nan_safe(self):
        z = _z_score_abs(np.array([np.nan, np.nan]))
        assert np.isnan(z).all()


class TestLag:
    def test_shifts_forward(self):
        out = _lag(np.array([1.0, 2.0, 3.0]), 1)
        assert np.isnan(out[0])
        assert out[1] == 1.0 and out[2] == 2.0

    def test_zero_lag_identity(self):
        x = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(_lag(x, 0), x, equal_nan=True)


class TestSpike:
    def test_spike_is_lagged_indicator(self):
        # Flat then a jump: the spike should appear one day AFTER the jump (lag 1)
        x = np.array([10.0] * 20 + [10000.0] + [10.0] * 20, dtype=float)
        spike = _lagged_z_score_spike(x, threshold=2.0, lag=1)
        # values are 0/1/NaN
        vals = set(np.unique(spike[~np.isnan(spike)]))
        assert vals.issubset({0.0, 1.0})
        assert np.nansum(spike) >= 1.0


class TestFindColumns:
    def test_suffix_match_first_family(self):
        panel = pd.DataFrame(columns=["eth_fees", "btc_fees", "eth_total_fees_usd"])
        cols = _find_columns(panel, ["fees", "total_fees_usd"])
        # First family (fees) wins via the fallback-chain break
        assert set(cols) == {"eth_fees", "btc_fees"}


# ── build_instruments ────────────────────────────────────────────────


@pytest.fixture
def panel():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    rng = np.random.default_rng(3)
    gas = rng.uniform(0.3, 0.9, 120)
    gas[60] = 5.0  # gas spike
    return pd.DataFrame(
        {
            "eth_avg_gas_utilization": gas,
            "eth_liquidation_volume_usd": rng.uniform(1e6, 5e6, 120),
            "usdc_SplyCur": np.linspace(20e9, 30e9, 120) + rng.normal(0, 1e8, 120),
            "usdt_SplyCur": np.linspace(80e9, 90e9, 120),
            "eth_fees": rng.uniform(1e6, 3e6, 120),
        },
        index=dates,
    )


class TestBuildInstruments:
    def test_builds_expected_instruments(self, panel):
        instruments, iv_map = build_instruments(panel)
        assert "gas_spike" in instruments.columns
        assert "stablecoin_mint" in instruments.columns
        assert "protocol_event" in instruments.columns
        assert iv_map["liq_flow"] == "gas_spike"
        assert iv_map["stable_flow"] == "stablecoin_mint"

    def test_index_aligned(self, panel):
        instruments, _ = build_instruments(panel)
        assert instruments.index.equals(panel.index)

    def test_stablecoin_mint_is_level_diff(self, panel):
        instruments, _ = build_instruments(panel)
        # level_diff is mostly non-zero (continuous), unlike sparse spikes
        nz = (instruments["stablecoin_mint"].dropna() != 0).mean()
        assert nz > 0.5

    def test_spike_is_sparse_binary(self, panel):
        instruments, _ = build_instruments(panel)
        s = instruments["gas_spike"].dropna()
        assert set(np.unique(s)).issubset({0.0, 1.0})
        assert (s != 0).mean() < 0.2  # spikes are rare

    def test_missing_sources_skips_instrument(self):
        panel = pd.DataFrame(
            {"btc_PriceUSD": [1.0, 2.0, 3.0]},
            index=pd.date_range("2023-01-01", periods=3),
        )
        instruments, iv_map = build_instruments(panel)
        assert instruments.empty or instruments.shape[1] == 0
        assert iv_map == {}
