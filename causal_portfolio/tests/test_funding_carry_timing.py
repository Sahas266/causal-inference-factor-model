"""Alignment tests for the funding-carry timing experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.experiments.funding_carry_timing import (
    FIT_WINDOW, SWITCH_COST, backtest, rolling_forecast,
)


def _panel(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    funding = pd.Series(rng.normal(1e-4, 5e-5, n), index=idx)
    return pd.DataFrame({
        "accrual_fwd": funding.shift(-1),
        "funding": funding,
        "gas_util": pd.Series(rng.uniform(0.3, 0.9, n), index=idx),
        "ret": pd.Series(rng.normal(0, 0.02, n), index=idx),
    }).dropna()


def test_rolling_forecast_uses_only_past_data():
    df = _panel()
    base = rolling_forecast(df, ["funding", "gas_util", "ret"])

    changed_df = df.copy()
    changed_df.iloc[-5:, :] += 1.0  # perturb only the final days
    changed = rolling_forecast(changed_df, ["funding", "gas_util", "ret"])

    # Forecasts strictly before the perturbation window must be identical.
    safe = base.index[:-(5 + 1)]
    pd.testing.assert_series_equal(base.loc[safe], changed.loc[safe])


def test_forecast_warmup_is_nan():
    df = _panel()
    pred = rolling_forecast(df, ["funding"])
    assert pred.iloc[:FIT_WINDOW].isna().all()
    assert pred.iloc[FIT_WINDOW:].notna().all()


def test_backtest_charges_switch_costs():
    df = _panel(n=300)
    always = backtest(df, pd.Series(1.0, index=df.index), "on")
    # One entry at row 10, one exit at row 20 -> exactly 2 switches charged.
    pos = pd.Series(0.0, index=df.index)
    pos.iloc[10:20] = 1.0
    pulse = backtest(df, pos, "pulse")
    assert pulse.n_switches == 2
    held = df["accrual_fwd"].iloc[10:20].sum()
    assert pulse.net.sum() == pytest.approx(held - 2 * SWITCH_COST)
    assert always.pct_in_market == 1.0
