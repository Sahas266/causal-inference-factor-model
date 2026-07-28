from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.hyperliquid import execute_plan
from causal_portfolio.execution.orderbook import L2Book, L2Level, estimate_vwap
from causal_portfolio.execution.types import AccountState, Order, RebalancePlan, TargetSnapshot


def _plan() -> RebalancePlan:
    state = AccountState("0xAAA", 10_000.0, 0.0)
    return RebalancePlan(
        timestamp_ms=1,
        target_weights={"btc": 0.1},
        current_state=state,
        target_usd={"BTC": 1_000.0},
        deltas_usd={"BTC": 1_000.0},
        orders=[Order("BTC", True, 0.01, 100_000.0, "0x" + "1" * 32)],
        skipped=[],
        equity_used=10_000.0,
        network="testnet",
        target_snapshot=TargetSnapshot(
            {"btc": 0.1}, as_of=datetime.now(timezone.utc)
        ),
    )


def _adapter(config: ExecutionConfig, ask: float = 100_100.0, size: float = 1.0):
    plan = _plan()
    adapter = MagicMock(address="0xAAA", config=config)
    adapter.fetch_mids.return_value = {"BTC": 100_000.0}
    adapter.fetch_l2_book.return_value = L2Book(
        "BTC",
        bids=[L2Level(99_900.0, 1.0)],
        asks=[L2Level(ask, size)],
    )
    adapter.fetch_state.side_effect = [plan.current_state, plan.current_state]
    adapter.submit_orders.return_value = {"status": "ok"}
    return plan, adapter


def test_estimate_vwap_walks_multiple_levels_and_requires_full_depth():
    book = L2Book(
        "BTC",
        bids=[L2Level(99.0, 2.0)],
        asks=[L2Level(101.0, 1.0), L2Level(103.0, 2.0)],
    )

    assert estimate_vwap(book, is_buy=True, size=2.0) == pytest.approx(102.0)
    assert estimate_vwap(book, is_buy=True, size=4.0) is None


def test_cost_gate_allows_cost_at_limit():
    cfg = ExecutionConfig(
        dry_run=False,
        smart_execution=False,
        repair_attempts=0,
        max_transaction_cost_bps=15.0,
        estimated_taker_fee_bps=0.0,
    )
    plan, adapter = _adapter(cfg, ask=100_150.0)

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted is True
    assert result.cost_gate_reason is None
    assert result.cost_estimate["estimated_cost_bps"] == pytest.approx(15.0)
    adapter.submit_orders.assert_called_once()


def test_cost_gate_blocks_expensive_book_before_cancel_or_submit():
    cfg = ExecutionConfig(
        dry_run=False,
        smart_execution=False,
        max_transaction_cost_bps=15.0,
        estimated_taker_fee_bps=4.5,
    )
    plan, adapter = _adapter(cfg, ask=100_200.0)

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted is False
    assert result.error is None
    assert result.cost_gate_reason == "estimated_cost_above_limit"
    assert result.cost_estimate["estimated_cost_bps"] > 15.0
    adapter.cancel_all_open.assert_not_called()
    adapter.submit_orders.assert_not_called()


def test_cost_gate_fails_closed_when_book_depth_is_insufficient():
    cfg = ExecutionConfig(
        dry_run=False,
        smart_execution=False,
        max_transaction_cost_bps=15.0,
    )
    plan, adapter = _adapter(cfg, size=0.001)

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted is False
    assert result.error is None
    assert result.cost_gate_reason == "cost_estimate_unavailable"
    assert "insufficient" in result.cost_estimate["error"]
    adapter.cancel_all_open.assert_not_called()


def test_cost_config_rejects_negative_values():
    with pytest.raises(ValueError, match="max_transaction_cost_bps"):
        ExecutionConfig(max_transaction_cost_bps=-1)
    with pytest.raises(ValueError, match="estimated_taker_fee_bps"):
        ExecutionConfig(estimated_taker_fee_bps=-1)
    with pytest.raises(ValueError, match="max_transaction_cost_bps"):
        ExecutionConfig(max_transaction_cost_bps=float("nan"))
    with pytest.raises(ValueError, match="estimated_taker_fee_bps"):
        ExecutionConfig(estimated_taker_fee_bps=float("inf"))
    with pytest.raises(ValueError, match="network_timeout_seconds"):
        ExecutionConfig(network_timeout_seconds=0)
    with pytest.raises(ValueError, match="network_timeout_seconds"):
        ExecutionConfig(network_timeout_seconds=float("nan"))
