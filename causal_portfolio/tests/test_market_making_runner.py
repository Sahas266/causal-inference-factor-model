from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from causal_portfolio.market_making.engine import MarketMakerConfig, MarketMakerEngine
from causal_portfolio.market_making.runner import MarketMakingService, RuntimeConfig


class _Adapter:
    def __init__(self):
        self.config = SimpleNamespace(
            dry_run=True, dedicated_account=False, dead_man_timeout_seconds=30
        )
        self.decisions = []
        self.cancelled = 0
        self.closed = False

    def reconcile_owned_orders(self):
        return {"owned_open": [], "stale_local": [], "foreign_open": []}

    def fetch_account_metrics(self, coin):
        return {
            "account_value_usd": 10_000.0,
            "free_margin_usd": 10_000.0,
            "inventory_base": 0.0,
        }

    def subscribe_market(self, coin, callback):
        return [1, 2, 3]

    def subscribe_account(self, callback):
        return [4, 5]

    def apply_decision(self, decision, acknowledge_mainnet=False):
        self.decisions.append(decision)
        return {"status": "dry_run"}

    def cancel_strategy_quotes(self):
        self.cancelled += 1

    def disarm_dead_man(self):
        return None

    def close(self):
        self.closed = True


class _MissingOrderRecoveryAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self._owned = False
        self.apply_calls = 0
        self.account_fetches = 0

    def fetch_account_metrics(self, coin):
        self.account_fetches += 1
        return super().fetch_account_metrics(coin)

    def apply_decision(self, decision, acknowledge_mainnet=False):
        self.decisions.append(decision)
        self.apply_calls += 1
        if self.apply_calls == 2:
            self._owned = False
            raise RuntimeError("quote cancellation was only partially successful (0/2)")
        self._owned = True
        return {"status": "dry_run"}

    def cancel_strategy_quotes(self):
        self.cancelled += 1
        if self._owned:
            raise RuntimeError("temporary cancel failure")
        return None


def test_service_runs_shadow_decision_and_cleans_up():
    config = MarketMakerConfig(
        coin="BTC",
        order_size=0.1,
        tick_size=0.01,
        size_decimals=2,
        gamma=0.01,
        kappa=2.0,
        horizon_seconds=60,
        max_inventory_base=0.2,
        max_total_spread_bps=300,
    )
    adapter = _Adapter()
    service = MarketMakingService(
        MarketMakerEngine(config),
        adapter,
        RuntimeConfig(fallback_sigma_per_sqrt_second=0.02),
    )
    service.start()
    now = int(time.time() * 1_000)
    service.on_message(
        {
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "time": now,
                "levels": [
                    [{"px": "99.5", "sz": "5", "n": 1}],
                    [{"px": "100.5", "sz": "5", "n": 1}],
                ],
            },
        }
    )
    assert len(adapter.decisions) == 1
    assert adapter.decisions[0].quotes is not None
    service.stop()
    assert adapter.cancelled == 1
    assert adapter.closed


@pytest.mark.parametrize(
    "invalid_levels",
    [
        [[], []],
        [
            [{"px": "101", "sz": "5", "n": 1}],
            [{"px": "100", "sz": "5", "n": 1}],
        ],
    ],
    ids=["empty", "crossed"],
)
def test_invalid_book_cancels_active_quotes(invalid_levels):
    config = MarketMakerConfig(
        coin="BTC",
        order_size=0.1,
        tick_size=0.01,
        size_decimals=2,
        gamma=0.01,
        kappa=2.0,
        horizon_seconds=60,
        max_inventory_base=0.2,
        max_total_spread_bps=300,
    )
    adapter = _Adapter()
    engine = MarketMakerEngine(config)
    service = MarketMakingService(
        engine,
        adapter,
        RuntimeConfig(fallback_sigma_per_sqrt_second=0.02),
    )
    service.start()
    now = int(time.time() * 1_000)
    service.on_message(
        {
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "time": now,
                "levels": [
                    [{"px": "99.5", "sz": "5", "n": 1}],
                    [{"px": "100.5", "sz": "5", "n": 1}],
                ],
            },
        }
    )
    assert engine.active_quotes is not None

    service.on_message(
        {
            "channel": "l2Book",
            "data": {"coin": "BTC", "time": now + 1, "levels": invalid_levels},
        }
    )

    assert len(adapter.decisions) == 2
    assert adapter.decisions[-1].cancel
    assert adapter.decisions[-1].reason == "empty_or_crossed_book"
    assert engine.active_quotes is None


def test_missing_order_cleanup_allows_next_book_to_submit_fresh_quotes():
    config = MarketMakerConfig(
        coin="BTC",
        order_size=0.1,
        tick_size=0.01,
        size_decimals=2,
        gamma=0.01,
        kappa=2.0,
        horizon_seconds=60,
        max_inventory_base=0.2,
        max_total_spread_bps=300,
    )
    adapter = _MissingOrderRecoveryAdapter()
    engine = MarketMakerEngine(config)
    service = MarketMakingService(
        engine,
        adapter,
        RuntimeConfig(
            fallback_sigma_per_sqrt_second=0.02,
            volatility_window=2,
        ),
    )
    service.start()
    now = int(time.time() * 1_000) - 100

    def send_book(timestamp_ms, bid, ask):
        service.on_message(
            {
                "channel": "l2Book",
                "data": {
                    "coin": "BTC",
                    "time": timestamp_ms,
                    "levels": [
                        [{"px": str(bid), "sz": "5", "n": 1}],
                        [{"px": str(ask), "sz": "5", "n": 1}],
                    ],
                },
            }
        )

    send_book(now, 99.5, 100.5)
    assert engine.active_quotes is not None

    send_book(now + 1, 100.5, 101.5)
    assert adapter.apply_calls == 2
    assert adapter.account_fetches == 2
    assert engine.active_quotes is None
    assert not engine.connected

    send_book(now + 2, 100.5, 101.5)
    assert adapter.apply_calls == 3
    assert adapter.decisions[-1].replace
    assert engine.active_quotes == adapter.decisions[-1].quotes
