from __future__ import annotations

import pytest

from causal_portfolio.market_making.engine import QuoteDecision
from causal_portfolio.market_making.hyperliquid import (
    HLMarketMakerAdapter,
    HLMarketMakerConfig,
)
from causal_portfolio.market_making.types import (
    BookLevel,
    BookSnapshot,
    Quote,
    QuotePair,
)


class _Exchange:
    def __init__(self):
        self.cancel_requests = None
        self.cancel_calls = 0
        self.cancel_response = {
            "status": "ok",
            "response": {"data": {"statuses": ["success", "success"]}},
        }
        self.order_requests = None
        self.order_response = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"resting": {"oid": 1}},
                        {"resting": {"oid": 2}},
                    ]
                }
            },
        }
        self.scheduled = None

    def bulk_cancel_by_cloid(self, requests):
        self.cancel_calls += 1
        self.cancel_requests = requests
        return self.cancel_response

    def bulk_orders(self, requests):
        self.order_requests = requests
        return self.order_response

    def schedule_cancel(self, timestamp):
        self.scheduled = timestamp
        return {"status": "ok"}


def _adapter(config):
    adapter = object.__new__(HLMarketMakerAdapter)
    adapter.config = config
    adapter.address = "0xabc"
    adapter._owned = {
        "0x" + "1" * 32: "BTC",
        "0x" + "2" * 32: "ETH",
    }
    adapter._subscriptions = []
    adapter._sz_decimals_cache = {}
    exchange = _Exchange()
    adapter._ensure_exchange = lambda: exchange
    adapter._save_state = lambda: None
    return adapter, exchange


def _pair():
    return QuotePair(
        bid=Quote(True, 99.0, 0.01, "0x" + "3" * 32),
        ask=Quote(False, 101.0, 0.01, "0x" + "4" * 32),
        reservation_price=100.0,
        model_spread=2.0,
        timestamp_ms=1,
        diagnostics={"coin": "BTC"},
    )


def _prepare_submit(adapter):
    adapter._owned = {}
    adapter._size_decimals = lambda coin: 5
    adapter.fetch_l2_book = lambda coin, depth=1: BookSnapshot(
        timestamp_ms=1,
        received_ms=1,
        coin=coin,
        bids=(BookLevel(98.0, 1.0),),
        asks=(BookLevel(102.0, 1.0),),
    )


def test_cancel_uses_only_strategy_owned_cloids():
    adapter, exchange = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    adapter.cancel_strategy_quotes()
    assert {request["coin"] for request in exchange.cancel_requests} == {"BTC", "ETH"}
    assert adapter._owned == {}


def test_failed_cancel_preserves_all_owned_cloids():
    adapter, exchange = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    exchange.cancel_response = {"status": "err", "response": "temporarily unavailable"}

    with pytest.raises(RuntimeError, match="failed at the exchange"):
        adapter.cancel_strategy_quotes()

    assert set(adapter._owned) == {"0x" + "1" * 32, "0x" + "2" * 32}


def test_mixed_cancel_removes_only_confirmed_cloid():
    adapter, exchange = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    exchange.cancel_response = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    "success",
                    {"error": "Order was never placed"},
                ]
            }
        },
    }

    with pytest.raises(RuntimeError, match="partially successful"):
        adapter.cancel_strategy_quotes()

    assert adapter._owned == {"0x" + "2" * 32: "ETH"}


@pytest.mark.parametrize(
    "missing_order_error",
    [
        "Order was never placed, already canceled, or filled.",
        "Order was never placed, already canceled, or filled. asset=200",
    ],
)
def test_missing_order_is_pruned_but_still_fails_current_cancel_cycle(
    missing_order_error,
):
    adapter, exchange = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    saved = []
    adapter._save_state = lambda: saved.append(dict(adapter._owned))
    exchange.cancel_response = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    "success",
                    {"error": missing_order_error},
                ]
            }
        },
    }

    with pytest.raises(RuntimeError, match="partially successful"):
        adapter.cancel_strategy_quotes()

    assert adapter._owned == {}
    assert saved == [{}]
    assert adapter.cancel_strategy_quotes() is None
    assert exchange.cancel_calls == 1


@pytest.mark.parametrize(
    "response",
    [
        {"status": "ok"},
        {
            "status": "ok",
            "response": {"data": {"statuses": ["success"]}},
        },
    ],
)
def test_cancel_rejects_missing_or_misaligned_statuses(response):
    adapter, exchange = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    exchange.cancel_response = response

    with pytest.raises(RuntimeError, match="statuses for 2 requests"):
        adapter.cancel_strategy_quotes()

    assert len(adapter._owned) == 2


def test_apply_decision_does_not_submit_after_unsuccessful_cancel():
    adapter, exchange = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    adapter.cancel_strategy_quotes = lambda: {"status": "err"}
    decision = QuoteDecision(
        quotes=_pair(),
        replace=True,
        cancel=False,
        reason="refresh",
        generation=2,
    )

    with pytest.raises(RuntimeError, match="failed at the exchange"):
        adapter.apply_decision(decision)

    assert exchange.order_requests is None


def test_partial_submit_tracks_only_resting_cloid_and_raises():
    adapter, exchange = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    _prepare_submit(adapter)
    exchange.order_response = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"resting": {"oid": 1}},
                    {"error": "Post only order would have immediately matched"},
                ]
            }
        },
    }

    with pytest.raises(RuntimeError, match="1/2 resting"):
        adapter.submit_quote_pair(_pair())

    assert adapter._owned == {"0x" + "3" * 32: "BTC"}


def test_all_rejected_submit_tracks_no_cloids_and_raises():
    adapter, exchange = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    _prepare_submit(adapter)
    exchange.order_response = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"error": "rejected bid"},
                    {"error": "rejected ask"},
                ]
            }
        },
    }

    with pytest.raises(RuntimeError, match="0/2 resting"):
        adapter.submit_quote_pair(_pair())

    assert adapter._owned == {}


def test_dead_man_requires_dedicated_account():
    adapter, _ = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    with pytest.raises(PermissionError, match="every account order"):
        adapter.arm_dead_man()


def test_mainnet_requires_explicit_configuration():
    with pytest.raises(ValueError, match="allow_mainnet"):
        HLMarketMakerConfig(testnet=False, dry_run=False)


def test_dry_run_cancel_does_not_discard_ownership():
    adapter, _ = _adapter(HLMarketMakerConfig())
    response = adapter.cancel_strategy_quotes()
    assert response["status"] == "dry_run"
    assert len(adapter._owned) == 2
