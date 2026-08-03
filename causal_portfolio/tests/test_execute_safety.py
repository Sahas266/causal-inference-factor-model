"""Tests for the execute_plan safety gates.

Mocks the HLAdapter so we can exercise the gating logic without network.
"""

from __future__ import annotations

import logging
from dataclasses import replace
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
    SkipReason,
    TargetSnapshot,
)


def _make_plan(address: str, orders=None, *, with_provenance: bool = True, network=None):
    state = AccountState(address=address, account_value_usd=10_000.0,
                         margin_used_usd=0.0)
    if orders is None:
        orders = [Order("BTC", True, 0.01, 100_000.0, "0x" + "1" * 32)]
    target_snapshot = (
        TargetSnapshot({"btc": 0.1}, as_of=datetime.now(timezone.utc))
        if with_provenance
        else None
    )
    return RebalancePlan(
        timestamp_ms=1000, target_weights={"btc": 0.1},
        current_state=state, target_usd={"BTC": 1000.0},
        deltas_usd={"BTC": 1000.0}, orders=orders,
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


def _partial_plan(allowed_usd: float, skipped_usd: float, *, reduce_only=False):
    positions = {}
    if reduce_only:
        positions = {
            "BTC": Position("BTC", allowed_usd / 100_000.0, 100_000.0, allowed_usd),
            "ETH": Position("ETH", skipped_usd / 100_000.0, 100_000.0, skipped_usd),
        }
    state = AccountState("0xAAA", 10_000.0, 0.0, positions=positions)
    return RebalancePlan(
        timestamp_ms=1000,
        target_weights={"btc": 0.0 if reduce_only else allowed_usd / 10_000.0},
        current_state=state,
        target_usd={
            "BTC": 0.0 if reduce_only else allowed_usd,
            "ETH": 0.0 if reduce_only else skipped_usd,
        },
        deltas_usd={
            "BTC": -allowed_usd if reduce_only else allowed_usd,
            "ETH": -skipped_usd if reduce_only else skipped_usd,
        },
        orders=[Order(
            "BTC",
            not reduce_only,
            allowed_usd / 100_000.0,
            100_000.0,
            "0x" + "1" * 32,
            reduce_only=reduce_only,
        )],
        skipped=[(
            "ETH",
            SkipReason.EXCEEDS_TRADE_CAP,
            "synthetic planner skip",
        )],
        equity_used=10_000.0,
        network="testnet",
        target_snapshot=TargetSnapshot(
            {"btc": 0.0 if reduce_only else allowed_usd / 10_000.0},
            as_of=datetime.now(timezone.utc),
        ),
    )


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


def test_post_submit_reconciliation_failure_preserves_submission(monkeypatch):
    import importlib

    reconcile_mod = importlib.import_module("causal_portfolio.execution.reconcile")

    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(
        address="0xAAA",
        config=ExecutionConfig(testnet=True, dry_run=False, smart_execution=False),
    )
    response = {"status": "ok", "response": {"data": "accepted"}}
    adapter.submit_orders.return_value = response
    monkeypatch.setattr(
        reconcile_mod,
        "reconcile",
        MagicMock(side_effect=RuntimeError("reconciliation unavailable")),
    )

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    assert result.response == response
    assert result.post_state == adapter.fetch_state.return_value
    assert result.error is None
    assert result.post_submit_error == "reconciliation unavailable"
    adapter.submit_orders.assert_called_once()


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


@pytest.mark.parametrize("acknowledgement", ["yes", 1, object()])
def test_mainnet_truthy_non_boolean_acknowledgement_stays_blocked(acknowledgement):
    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=False, dry_run=False,
    ))

    result = execute_plan(
        adapter,
        plan,
        write_audit=False,
        acknowledge_mainnet=acknowledgement,
    )

    assert not result.submitted
    assert "literal boolean True" in (result.error or "")
    assert "execute_plan" not in (result.error or "")
    assert "CLI" not in (result.error or "")
    adapter.cancel_all_open.assert_not_called()


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


def test_empty_live_plan_is_audited_without_exchange_calls(monkeypatch):
    from causal_portfolio.execution import audit as audit_mod

    plan = _make_plan(address="0xAAA", orders=[])
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=False,
    ))
    audit_append = MagicMock(return_value=None)
    monkeypatch.setattr(audit_mod, "append", audit_append)

    result = execute_plan(adapter, plan)

    assert not result.submitted
    assert result.error is None
    assert result.completeness_ratio == 0.0
    adapter.cancel_all_open.assert_not_called()
    adapter.fetch_state.assert_not_called()
    adapter.submit_orders.assert_not_called()
    adapter.submit_orders_book_aware.assert_not_called()
    audit_append.assert_called_once_with(result)


def test_completeness_gate_refuses_large_missing_share():
    plan = _partial_plan(allowed_usd=100.0, skipped_usd=900.0)
    plan = replace(
        plan,
        target_weights={"btc": 0.01, "wlfi": 0.09},
        target_usd={"BTC": 100.0},
        deltas_usd={"BTC": 100.0},
        skipped=[("wlfi", SkipReason.NOT_LISTED, "no HL listing")],
    )
    adapter = _mock_adapter("0xAAA", ExecutionConfig(
        testnet=True,
        dry_run=False,
        smart_execution=False,
        repair_attempts=0,
        min_rebalance_completeness=0.80,
    ))

    result = execute_plan(adapter, plan, write_audit=False)

    assert not result.submitted
    assert result.completeness_ratio == pytest.approx(0.10)
    assert "completeness" in result.error
    adapter.cancel_all_open.assert_not_called()


def test_completeness_gate_allows_only_dust_missing():
    plan = _partial_plan(allowed_usd=1_000.0, skipped_usd=1.0)
    adapter = _mock_adapter("0xAAA", ExecutionConfig(
        testnet=True,
        dry_run=False,
        smart_execution=False,
        repair_attempts=0,
        min_rebalance_completeness=0.95,
    ))

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    assert result.completeness_ratio == pytest.approx(1_000 / 1_001)
    adapter.submit_orders.assert_called_once()


def test_completeness_gate_never_blocks_all_reduce_only_plan():
    plan = _partial_plan(allowed_usd=100.0, skipped_usd=900.0, reduce_only=True)
    adapter = _mock_adapter("0xAAA", ExecutionConfig(
        testnet=True,
        dry_run=False,
        smart_execution=False,
        repair_attempts=0,
        min_rebalance_completeness=0.99,
    ))
    adapter.fetch_state.return_value = plan.current_state

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    assert result.completeness_ratio == 1.0
    adapter.submit_orders.assert_called_once()


def test_completeness_gate_blocks_only_increases_and_submits_reduce_only():
    state = AccountState(
        "0xAAA",
        10_000.0,
        0.0,
        positions={"BTC": Position("BTC", 0.001, 100_000.0, 100.0)},
    )
    plan = RebalancePlan(
        timestamp_ms=1000,
        target_weights={"btc": 0.0, "eth": 0.01, "sol": 0.09},
        current_state=state,
        target_usd={"BTC": 0.0, "ETH": 100.0, "SOL": 900.0},
        deltas_usd={"BTC": -100.0, "ETH": 100.0, "SOL": 900.0},
        orders=[
            Order(
                "BTC", False, 0.001, 100_000.0, "0x" + "1" * 32,
                reduce_only=True,
            ),
            Order("ETH", True, 0.1, 1_000.0, "0x" + "2" * 32),
        ],
        skipped=[("SOL", SkipReason.EXCEEDS_TRADE_CAP, "synthetic planner skip")],
        equity_used=10_000.0,
        network="testnet",
        target_snapshot=TargetSnapshot(
            {"btc": 0.0, "eth": 0.01, "sol": 0.09},
            as_of=datetime.now(timezone.utc),
        ),
    )
    adapter = _mock_adapter("0xAAA", ExecutionConfig(
        testnet=True,
        dry_run=False,
        smart_execution=False,
        repair_attempts=0,
        min_rebalance_completeness=0.80,
    ))
    adapter.fetch_state.return_value = state

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    assert result.error is None
    assert result.completeness_ratio == pytest.approx(0.10)
    assert [order.coin for order in result.submitted_orders] == ["BTC"]
    assert ("ETH", SkipReason.INSUFFICIENT_PLAN_COMPLETENESS) in {
        (coin, reason) for coin, reason, _detail in result.plan.skipped
    }


def test_completeness_gate_default_preserves_partial_submit():
    plan = _partial_plan(allowed_usd=100.0, skipped_usd=900.0)
    adapter = _mock_adapter("0xAAA", ExecutionConfig(
        testnet=True,
        dry_run=False,
        smart_execution=False,
        repair_attempts=0,
    ))

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    assert result.completeness_ratio == pytest.approx(0.10)
    adapter.submit_orders.assert_called_once()


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


def test_execute_plan_logs_transaction_for_any_model(caplog):
    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(address="0xAAA", config=ExecutionConfig(
        testnet=True, dry_run=True,
    ))
    caplog.set_level(logging.INFO, logger="cpcm.execution.hl")

    execute_plan(adapter, plan, write_audit=False)

    text = caplog.text
    assert f"target_id={plan.target_snapshot.target_id}" in text
    assert f"network=testnet orders={len(plan.orders)} dry_run=True" in text
    assert "execution finished: submitted=False error=None" in text


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


def test_audit_append_failure_preserves_submission_state(tmp_path, monkeypatch):
    from causal_portfolio.execution import audit as audit_mod
    from causal_portfolio.execution.run_logging import execution_run_log

    plan = _make_plan(address="0xAAA")
    adapter = _mock_adapter(
        address="0xAAA",
        config=ExecutionConfig(testnet=True, dry_run=False, smart_execution=False),
    )
    response = {"status": "ok", "response": {"data": "accepted"}}
    adapter.submit_orders.return_value = response
    monkeypatch.setattr(
        audit_mod,
        "append",
        MagicMock(side_effect=OSError("audit disk full")),
    )

    with execution_run_log("audit-failure", log_dir=tmp_path) as run_log:
        result = execute_plan(adapter, plan)

    assert result.submitted
    assert result.response == response
    assert result.error is None
    assert result.audit_error == "audit disk full"
    text = run_log.read_text(encoding="utf-8")
    assert "audit append failed" in text
    assert "OSError: audit disk full" in text
    assert "audit_error=audit disk full" in text


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
