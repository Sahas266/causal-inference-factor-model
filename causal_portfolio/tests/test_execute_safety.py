"""Tests for the execute_plan safety gates.

Mocks the HLAdapter so we can exercise the gating logic without network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("hyperliquid", reason="hyperliquid SDK not installed")
from causal_portfolio.execution.hyperliquid import execute_plan

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.types import (
    AccountState,
    Position,
    RebalancePlan,
)


def _make_plan(address: str, orders=None):
    state = AccountState(address=address, account_value_usd=10_000.0,
                         margin_used_usd=0.0)
    return RebalancePlan(
        timestamp_ms=1000, target_weights={"btc": 0.1},
        current_state=state, target_usd={"BTC": 1000.0},
        deltas_usd={"BTC": 1000.0}, orders=orders or [],
        skipped=[], equity_used=10_000.0,
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
