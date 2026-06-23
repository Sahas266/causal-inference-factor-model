from __future__ import annotations

import pytest

from causal_portfolio.market_making.hyperliquid import (
    HLMarketMakerAdapter,
    HLMarketMakerConfig,
)


class _Exchange:
    def __init__(self):
        self.cancel_requests = None
        self.scheduled = None

    def bulk_cancel_by_cloid(self, requests):
        self.cancel_requests = requests
        return {"status": "ok"}

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
    exchange = _Exchange()
    adapter._ensure_exchange = lambda: exchange
    adapter._save_state = lambda: None
    return adapter, exchange


def test_cancel_uses_only_strategy_owned_cloids():
    adapter, exchange = _adapter(HLMarketMakerConfig(testnet=True, dry_run=False))
    adapter.cancel_strategy_quotes()
    assert {request["coin"] for request in exchange.cancel_requests} == {"BTC", "ETH"}
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
