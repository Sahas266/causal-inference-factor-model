from __future__ import annotations

from causal_portfolio.market_making.engine import MarketMakerConfig, MarketMakerEngine
from causal_portfolio.market_making.toxicity import ToxicityConfig
from causal_portfolio.market_making.types import BookLevel, BookSnapshot, QuotePolicy


def _book(ts=1_000):
    return BookSnapshot(
        ts,
        "BTC",
        (BookLevel(99.5, 5), BookLevel(99.0, 5)),
        (BookLevel(100.5, 5), BookLevel(101.0, 5)),
        received_ms=ts,
    )


def _config(**kwargs):
    base = dict(
        coin="BTC",
        order_size=0.1,
        tick_size=0.01,
        size_decimals=2,
        gamma=0.01,
        kappa=2.0,
        horizon_seconds=60.0,
        max_inventory_base=0.2,
        max_total_spread_bps=300,
    )
    base.update(kwargs)
    return MarketMakerConfig(**base)


def test_engine_requires_connection_and_acknowledges_state():
    engine = MarketMakerEngine(_config())
    blocked = engine.decide(
        _book(), now_ms=1_000, inventory_base=0, sigma_per_sqrt_second=0.02,
        free_margin_usd=1_000, daily_pnl_usd=0,
    )
    assert blocked.reason == "disconnected"
    engine.set_connected(True)
    decision = engine.decide(
        _book(), now_ms=1_000, inventory_base=0, sigma_per_sqrt_second=0.02,
        free_margin_usd=1_000, daily_pnl_usd=0,
    )
    assert decision.replace and decision.quotes is not None
    assert decision.quotes.bid.price < 100 < decision.quotes.ask.price
    assert decision.quotes.diagnostics["coin"] == "BTC"
    engine.acknowledge(decision, success=True)
    unchanged = engine.decide(
        _book(), now_ms=1_100, inventory_base=0, sigma_per_sqrt_second=0.02,
        free_margin_usd=1_000, daily_pnl_usd=0,
    )
    assert not unchanged.replace and unchanged.reason == "unchanged"


def test_engine_suppresses_inventory_increasing_side_at_limit():
    engine = MarketMakerEngine(_config())
    engine.set_connected(True)
    decision = engine.decide(
        _book(), now_ms=1_000, inventory_base=0.2, sigma_per_sqrt_second=0.02,
        free_margin_usd=1_000, daily_pnl_usd=0,
    )
    assert decision.quotes is not None
    assert decision.quotes.bid is None
    assert decision.quotes.ask is not None


def test_engine_cancels_on_stale_book_and_missing_toxicity():
    engine = MarketMakerEngine(_config())
    engine.set_connected(True)
    decision = engine.decide(
        _book(), now_ms=1_000, inventory_base=0, sigma_per_sqrt_second=0.02,
        free_margin_usd=1_000, daily_pnl_usd=0,
    )
    engine.acknowledge(decision, success=True)
    stale = engine.decide(
        _book(), now_ms=10_000, inventory_base=0, sigma_per_sqrt_second=0.02,
        free_margin_usd=1_000, daily_pnl_usd=0,
    )
    assert stale.cancel and stale.reason == "stale_or_future_book"

    toxic_engine = MarketMakerEngine(
        _config(toxicity=ToxicityConfig(policy=QuotePolicy.AS_PIN_SIZE))
    )
    toxic_engine.set_connected(True)
    missing = toxic_engine.decide(
        _book(), now_ms=1_000, inventory_base=0, sigma_per_sqrt_second=0.02,
        free_margin_usd=1_000, daily_pnl_usd=0,
    )
    assert missing.reason == "toxicity_estimator_unavailable"
