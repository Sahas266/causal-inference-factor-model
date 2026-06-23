from __future__ import annotations

import time
from types import SimpleNamespace

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
