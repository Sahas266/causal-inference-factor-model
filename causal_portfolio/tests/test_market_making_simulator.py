from __future__ import annotations

import math

import pytest

from causal_portfolio.market_making.simulator import EventDrivenSimulator, SimulationConfig
from causal_portfolio.market_making.types import (
    AggressorSide,
    BookLevel,
    BookSnapshot,
    Quote,
    QuotePair,
    TradePrint,
)


def _pair(bid=99.0, ask=101.0):
    return QuotePair(
        Quote(True, bid, 0.5, "0x" + "1" * 32),
        Quote(False, ask, 0.5, "0x" + "2" * 32),
        reservation_price=(bid + ask) / 2.0,
        model_spread=ask - bid,
        timestamp_ms=0,
        diagnostics={"coin": "BTC"},
    )


def test_replay_models_partial_maker_fill_and_mark_to_market():
    events = [
        BookSnapshot(0, "BTC", (BookLevel(99, 2),), (BookLevel(101, 2),)),
        TradePrint(1, "BTC", 99, 0.5, AggressorSide.SELL),
        BookSnapshot(1_001, "BTC", (BookLevel(97, 2),), (BookLevel(99, 2),)),
    ]
    simulator = EventDrivenSimulator(
        SimulationConfig(latency_ms=0, queue_ahead_fraction=0)
    )
    result = simulator.run(events, lambda context: _pair())
    assert len(result.fills) == 1
    assert result.final_inventory == 0.5
    assert result.pnl_usd == pytest.approx(-0.5)
    assert result.markout_bps[1_000] > 0


def test_queue_ahead_prevents_optimistic_fill():
    events = [
        BookSnapshot(0, "BTC", (BookLevel(99, 2),), (BookLevel(101, 2),)),
        TradePrint(1, "BTC", 99, 1.0, AggressorSide.SELL),
        BookSnapshot(2, "BTC", (BookLevel(99, 2),), (BookLevel(101, 2),)),
    ]
    result = EventDrivenSimulator(
        SimulationConfig(latency_ms=0, queue_ahead_fraction=1)
    ).run(events, lambda context: _pair())
    assert result.fills == ()
    assert math.isnan(result.markout_bps[1_000])


def test_post_only_cross_is_rejected():
    events = [
        BookSnapshot(0, "BTC", (BookLevel(99, 2),), (BookLevel(101, 2),)),
        BookSnapshot(1, "BTC", (BookLevel(99, 2),), (BookLevel(101, 2),)),
    ]
    crossed = _pair(bid=101, ask=102)
    result = EventDrivenSimulator(SimulationConfig(latency_ms=0)).run(
        events, lambda context: crossed
    )
    assert result.post_only_rejections >= 1
