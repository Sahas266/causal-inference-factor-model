"""Tests for the execute_plan safety gates.

Mocks the HLAdapter so we can exercise the gating logic without network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("hyperliquid", reason="hyperliquid SDK not installed")
from causal_portfolio.execution.hyperliquid import execute_plan

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.types import (
    AccountState,
    Order,
    Position,
    RebalancePlan,
    TargetSnapshot,
)


def _make_plan(address: str, orders=None, *, with_provenance: bool = True, network=None):
    state = AccountState(address=address, account_value_usd=10_000.0,
                         margin_used_usd=0.0)
    target_snapshot = (
        TargetSnapshot({"btc": 0.1}, as_of=datetime.now(timezone.utc))
        if with_provenance
        else None
    )
    return RebalancePlan(
        timestamp_ms=1000, target_weights={"btc": 0.1},
        current_state=state, target_usd={"BTC": 1000.0},
        deltas_usd={"BTC": 1000.0}, orders=orders or [],
        skipped=[], equity_used=10_000.0,
        network=network,
        target_snapshot=target_snapshot,
    )


def _mock_adapter(address: str, config: ExecutionConfig):
    a = MagicMock()
    a.address = address
    a.config = config
    a.cancel_all_open.return_value = None
    a.submit_orders.return_value = {"status": "ok"}
    a.submit_orders_book_aware.return_value = {"status": "ok"}
    # Post-state matches plan to avoid race detection
    a.fetch_state.return_value = AccountState(
        address=address, account_value_usd=10_000.0, margin_used_usd=0.0,
        positions={},
    )
    return a


# ── Gate 1: address mismatch ────────────────────────────────────────


def test_address_mismatch_blocks_execution():
    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xBBB", config=ExecutionConfig(
        testnet=True, dry_run=False,
    ))
    result = execute_plan(adapter, plan, write_audit=False)
    assert not result.submitted
    assert "address mismatch" in (result.error or "")
    adapter.submit_orders.assert_not_called()
    adapter.cancel_all_open.assert_not_called()


def test_matching_address_proceeds():
    """Default config uses the book-aware execution path."""
    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=False,
    ))
    result = execute_plan(adapter, plan, write_audit=False)
    assert result.submitted
    adapter.submit_orders_book_aware.assert_called_once()


def test_matching_address_batch_path_when_smart_disabled():
    """With smart_execution=False, execute_plan uses the batch submit_orders."""
    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=False, smart_execution=False,
    ))
    result = execute_plan(adapter, plan, write_audit=False)
    assert result.submitted
    adapter.submit_orders.assert_called_once()
    adapter.submit_orders_book_aware.assert_not_called()


def test_execute_reconcile_uses_min_order_notional_floor_for_tiny_runs():
    state = AccountState(
        address="0xAAA",
        account_value_usd=2_500.0,
        margin_used_usd=0.0,
        positions={"BTC": Position("BTC", 0.00034, 58_299.0, 19.82)},
    )
    plan = RebalancePlan(
        timestamp_ms=1000,
        target_weights={"btc": 0.0},
        current_state=state,
        target_usd={"BTC": 0.0},
        deltas_usd={"BTC": -19.82},
        orders=[Order("BTC", False, 0.00034, 58_115.0, "0x" + "1" * 32, reduce_only=True)],
        skipped=[],
        equity_used=2_500.0,
        network="testnet",
        target_snapshot=TargetSnapshot({"btc": 0.0}, as_of=datetime.now(timezone.utc)),
    )
    adapter = _mock_adapter(
        address="0xAAA",
        config=ExecutionConfig(testnet=True, dry_run=False, smart_execution=False),
    )
    adapter.fetch_state.side_effect = [
        state,  # cancel-race guard
        state,  # post-submit residual: order failed to flatten
    ]

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    assert len(result.drifts) == 1
    assert result.drifts[0].coin == "BTC"
    assert result.drifts[0].actual_usd == pytest.approx(19.82)


# ── Gate 2: mainnet acknowledgement ──────────────────────────────────


def test_mainnet_blocked_without_acknowledgement():
    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=False, dry_run=False,
    ))
    result = execute_plan(adapter, plan, write_audit=False,
                          acknowledge_mainnet=False)
    assert not result.submitted
    assert "MAINNET" in (result.error or "")
    adapter.submit_orders.assert_not_called()


def test_mainnet_proceeds_with_acknowledgement():
    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=False, dry_run=False,
    ))
    result = execute_plan(adapter, plan, write_audit=False,
                          acknowledge_mainnet=True)
    assert result.submitted


def test_live_execution_blocks_missing_target_provenance():
    plan = _make_plan(address="0xAAA", with_provenance=False)
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=False,
    ))

    result = execute_plan(adapter, plan, write_audit=False)

    assert not result.submitted
    assert "missing provenance" in (result.error or "")
    adapter.cancel_all_open.assert_not_called()
    adapter.submit_orders.assert_not_called()


def test_live_execution_can_explicitly_override_missing_target_provenance():
    plan = _make_plan(address="0xAAA", with_provenance=False)
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=False, allow_stale_signal=True,
    ))

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    adapter.cancel_all_open.assert_called_once()


def test_testnet_does_not_need_acknowledgement():
    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=False,
    ))
    result = execute_plan(adapter, plan, write_audit=False)
    assert result.submitted


# ── Gate 3: dry-run honored ──────────────────────────────────────────


def test_dry_run_skips_network_writes():
    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=True,
    ))
    result = execute_plan(adapter, plan, write_audit=False)
    assert not result.submitted
    adapter.submit_orders.assert_not_called()
    adapter.cancel_all_open.assert_not_called()


# ── Audit write integration ──────────────────────────────────────────


def test_audit_appended_by_execute_plan(tmp_path, monkeypatch):
    """Even dry-run results should be audited."""
    from causal_portfolio.execution import audit as audit_mod
    monkeypatch.setattr(audit_mod, "LOG_DIR", tmp_path)

    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=True,
    ))
    execute_plan(adapter, plan, write_audit=True)
    # Find any log file in tmp_path
    log_files = list(tmp_path.glob("rebalance-*.jsonl"))
    assert len(log_files) == 1


def test_audit_disabled_by_flag(tmp_path, monkeypatch):
    from causal_portfolio.execution import audit as audit_mod
    monkeypatch.setattr(audit_mod, "LOG_DIR", tmp_path)

    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=True,
    ))
    execute_plan(adapter, plan, write_audit=False)
    log_files = list(tmp_path.glob("rebalance-*.jsonl"))
    assert log_files == []
