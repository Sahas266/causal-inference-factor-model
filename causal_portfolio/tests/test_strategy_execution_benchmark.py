from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.market_making.strategy_execution_benchmark import (
    DailyCandle,
    candle_classified_counts,
    dual_confirm_signal,
    run_historical_execution_benchmark,
)


def _candle(day: int, open_px: float, close_px: float) -> DailyCandle:
    timestamp = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000)
    high = max(open_px, close_px) * 1.01
    low = min(open_px, close_px) * 0.99
    return DailyCandle(timestamp + day * 86_400_000, open_px, high, low, close_px, 100, 1_000)


def test_dual_confirm_is_causal_and_binary():
    index = pd.date_range("2024-01-01", periods=160, tz="UTC")
    prices = pd.Series(np.linspace(100, 200, 160), index=index)
    signal = dual_confirm_signal(prices)
    assert set(signal.unique()) <= {0.0, 1.0}
    assert signal.iloc[-1] == 1.0
    changed_future = prices.copy()
    changed_future.iloc[-1] = 1_000
    assert dual_confirm_signal(changed_future).iloc[-2] == signal.iloc[-2]


def test_candle_count_classification_preserves_total_trades():
    candles = [_candle(day, 100 + day, 101 + day) for day in range(70)]
    buys, sells = candle_classified_counts(candles)
    assert np.all(buys + sells == 1_000)


def test_historical_benchmark_compares_same_switches():
    index = pd.date_range("2024-01-01", periods=260, tz="UTC")
    wave = 100 + np.arange(260) * 0.1 + 15 * np.sin(np.arange(260) / 12)
    prices = pd.Series(wave, index=index)
    candles = [
        _candle(day, float(price), float(price * (1 + 0.01 * np.sin(day))))
        for day, price in enumerate(wave)
    ]
    result = run_historical_execution_benchmark(
        prices, candles, pin_window=20, pin_starts=2
    )
    assert result.switches
    assert result.naive.mean_cost_bps > 0
    assert result.as_pin_touch.passive_fill_rate is not None
    assert len(result.switches) == sum(1 for _ in result.switches)


def test_committed_historical_artifact_matches_reported_table():
    artifact = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "execution_artifacts"
        / "trend_rotation_historical_execution.json"
    )
    data = json.loads(artifact.read_text(encoding="utf-8"))

    assert len(data["switches"]) == 98
    assert data["start"] == "2021-04-12"
    assert data["end"] == "2025-10-11"
    assert data["naive"]["mean_cost_bps"] == pytest.approx(4.50)
    assert data["naive"]["total_cost_bps"] == pytest.approx(441.00)
    assert data["naive"]["sharpe"] == pytest.approx(1.002024939955809)
    assert data["as_pin_touch"]["mean_cost_bps"] == pytest.approx(11.119479502576041)
    assert data["as_pin_touch"]["passive_fill_rate"] == pytest.approx(0.8571428571428571)
    assert data["as_pin_close"]["mean_cost_bps"] == pytest.approx(94.45677925754416)
    assert data["as_pin_close"]["passive_fill_rate"] == pytest.approx(0.42857142857142855)
    assert data["pin_usable_fraction"] == pytest.approx(0.7551020408163265)
