"""Tests for scm/shocks.py — rolling (no-look-ahead) IV spike thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.scm.shocks import create_instruments


def _panel_and_factors(n: int = 400):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(5)
    panel = pd.DataFrame(
        {
            "eth_avg_gas_price_gwei": rng.uniform(10, 30, n),
            "eth_total_liquidations_usd": rng.uniform(1e6, 5e6, n),
            "aave_tvl_usd": rng.uniform(1e9, 2e9, n),
        },
        index=idx,
    )
    factors = pd.DataFrame({"stable_flow": rng.standard_normal(n)}, index=idx)
    return panel, factors


def test_builds_all_declared_instruments():
    panel, factors = _panel_and_factors()
    iv = create_instruments(panel, factors)
    assert set(iv.columns) == {
        "gas_spike", "liquidation_level", "stablecoin_mint", "protocol_event",
    }


def test_spike_labels_have_no_lookahead():
    """Changing FUTURE data must not change spike labels at earlier t — the
    old full-sample quantile threshold violated this."""
    panel, factors = _panel_and_factors()
    iv1 = create_instruments(panel, factors)

    panel2 = panel.copy()
    panel2.iloc[300:, panel2.columns.get_loc("eth_avg_gas_price_gwei")] += 1e6
    iv2 = create_instruments(panel2, factors)

    pd.testing.assert_series_equal(
        iv1["gas_spike"].iloc[:300], iv2["gas_spike"].iloc[:300]
    )


def test_missing_sources_raise_instead_of_treatment_fallback():
    """The old code silently fell back to a transform of the treatment itself
    (which cannot satisfy exclusion). It must now raise."""
    idx = pd.date_range("2022-01-01", periods=100, freq="D")
    panel = pd.DataFrame({"btc_price": np.ones(100)}, index=idx)
    factors = pd.DataFrame({"liq_flow": np.ones(100)}, index=idx)
    with pytest.raises(ValueError, match="cannot construct instrument"):
        create_instruments(panel, factors)
