from __future__ import annotations

from datetime import datetime, timezone

import pytest

from causal_portfolio.market_making.artifacts import CalibrationArtifact
from causal_portfolio.market_making.calibration import MarkoutCurve
from causal_portfolio.market_making.engine import MarketMakerConfig
from causal_portfolio.market_making.pin import PINFit, PINPosterior
from causal_portfolio.market_making.replay import compare_policies
from causal_portfolio.market_making.simulator import SimulationConfig
from causal_portfolio.market_making.types import (
    AggressorSide,
    BookLevel,
    BookSnapshot,
    QuotePolicy,
    TradePrint,
)


def _artifact(as_of):
    fit = PINFit(
        0.3,
        0.5,
        40,
        20,
        0.230769,
        -100,
        60,
        True,
        False,
        100,
        (0.01, 0.01, 1, 1),
        8,
        "ok",
    )
    return CalibrationArtifact(
        "BTC",
        as_of,
        fit,
        PINPosterior(0.5, 0.3, 0.2),
        MarkoutCurve((0, 1), (2,), (2,), (100,)),
    )


def _events():
    return [
        BookSnapshot(1_000, "BTC", (BookLevel(99.5, 2),), (BookLevel(100.5, 2),)),
        TradePrint(1_001, "BTC", 99.5, 0.1, AggressorSide.SELL),
        BookSnapshot(2_000, "BTC", (BookLevel(99.4, 2),), (BookLevel(100.4, 2),)),
    ]


def _config():
    return MarketMakerConfig(
        coin="BTC",
        order_size=0.1,
        tick_size=0.01,
        size_decimals=2,
        gamma=0.01,
        kappa=2,
        horizon_seconds=60,
        max_inventory_base=0.2,
        max_total_spread_bps=300,
    )


def test_compare_policies_uses_identical_event_window():
    artifact = _artifact(datetime.fromtimestamp(0, tz=timezone.utc))
    results = compare_policies(
        _events(),
        _config(),
        artifact,
        fallback_sigma=0.02,
        simulation_config=SimulationConfig(latency_ms=0, queue_ahead_fraction=0),
    )
    assert set(results) == set(QuotePolicy)
    assert all(result.quote_replacements >= 1 for result in results.values())


def test_compare_policies_blocks_lookahead_calibration():
    artifact = _artifact(datetime.fromtimestamp(2, tz=timezone.utc))
    with pytest.raises(ValueError, match="look-ahead"):
        compare_policies(_events(), _config(), artifact, fallback_sigma=0.02)
