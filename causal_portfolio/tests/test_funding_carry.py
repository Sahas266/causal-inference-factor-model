from __future__ import annotations

import numpy as np
import pandas as pd

from causal_portfolio.experiments.funding_carry import (
    common_forecast_start,
    forecast_next_funding,
)


def test_forecast_emits_for_latest_feature_row_without_future_label():
    index = pd.date_range("2026-01-01", periods=80, freq="D")
    funding = pd.DataFrame(
        {"btc": 0.001 + np.arange(len(index)) * 0.00001}, index=index
    )

    forecast = forecast_next_funding(
        funding, None, train_window=20, refit=5
    )

    assert pd.notna(forecast.loc[index[-1], "btc"])


def test_comparative_window_starts_when_every_forecast_arm_is_live():
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    factor = pd.DataFrame({"btc": [np.nan, np.nan, 1.0, 1.0, 1.0]}, index=index)
    nofactor = pd.DataFrame({"btc": [np.nan, 1.0, 1.0, 1.0, 1.0]}, index=index)

    assert common_forecast_start(factor, nofactor) == index[2]
