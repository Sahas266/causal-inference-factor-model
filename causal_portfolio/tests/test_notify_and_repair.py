"""Leg-failure repair pass + Telegram notification."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("hyperliquid", reason="hyperliquid SDK not installed")
from causal_portfolio.execution import notify
from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.hyperliquid import execute_plan
from causal_portfolio.execution.types import (
    AccountState,
    AssetMeta,
    Order,
    Position,
    RebalancePlan,
    SubmitResult,
    TargetSnapshot,
)


def _plan(address="0xAAA"):
    state = AccountState(address=address, account_value_usd=10_000.0,
                         margin_used_usd=0.0)
    return RebalancePlan(
        timestamp_ms=1000, target_weights={"btc": 0.1},
        current_state=state, target_usd={"BTC": 1000.0},
        deltas_usd={"BTC": 1000.0},
        orders=[Order("BTC", True, 0.01, 100_000.0, "0x" + "1" * 32)],
        skipped=[], equity_used=10_000.0, network="testnet",
        target_snapshot=TargetSnapshot({"btc": 0.1}, as_of=datetime.now(timezone.utc)),
    )


def test_repair_pass_replans_and_resolves_drift():
    plan = _plan()
    cfg = ExecutionConfig(testnet=True, dry_run=False, smart_execution=False,
                          repair_attempts=1)
    adapter = MagicMock()
    adapter.address = "0xAAA"
    adapter.config = cfg
    adapter.cancel_all_open.return_value = None
    adapter.submit_orders.return_value = {"status": "ok"}
    adapter.submit_orders_book_aware.return_value = {"status": "ok"}
    adapter.fetch_mids.return_value = {"BTC": 100_000.0}
    adapter.fetch_meta.return_value = {
        "BTC": AssetMeta("BTC", sz_decimals=4, max_leverage=40, min_size=1e-4),
    }
    pre = plan.current_state
    unfilled = AccountState(address="0xAAA", account_value_usd=10_000.0,
                            margin_used_usd=0.0, positions={})
    filled = AccountState(
        address="0xAAA", account_value_usd=10_000.0, margin_used_usd=25.0,
        positions={"BTC": Position("BTC", 0.01, 100_000.0, 1000.0)},
    )
    # cancel-race guard, post-submit (leg failed), post-repair (filled)
    adapter.fetch_state.side_effect = [pre, unfilled, filled]

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    adapter.submit_orders_book_aware.assert_called_once()  # the repair retry
    repair_order = adapter.submit_orders_book_aware.call_args[0][0][0]
    assert repair_order.coin == "BTC" and repair_order.is_buy
    assert result.repair["resolved"] is True
    assert result.drifts == []


def test_repair_disabled_leaves_drift():
    plan = _plan()
    cfg = ExecutionConfig(testnet=True, dry_run=False, smart_execution=False,
                          repair_attempts=0)
    adapter = MagicMock()
    adapter.address = "0xAAA"
    adapter.config = cfg
    adapter.submit_orders.return_value = {"status": "ok"}
    unfilled = AccountState(address="0xAAA", account_value_usd=10_000.0,
                            margin_used_usd=0.0, positions={})
    adapter.fetch_state.side_effect = [plan.current_state, unfilled]

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.repair is None
    assert len(result.drifts) == 1
    adapter.submit_orders_book_aware.assert_not_called()


def _result(plan=None, **kw):
    return SubmitResult(plan=plan or _plan(), submitted=kw.pop("submitted", True), **kw)


def test_notify_skips_clean_dry_run(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "send", lambda text: sent.append(text) or True)
    assert notify.notify_result(_result(submitted=False)) is False
    assert notify.notify_result(_result(submitted=True)) is True
    assert "target" in sent[0] and "submitted=True" in sent[0]


def test_send_unconfigured_returns_false(monkeypatch):
    monkeypatch.setenv(notify.TOKEN_ENV, "")
    monkeypatch.setenv(notify.CHAT_ENV, "")
    assert notify.send("hi") is False


def test_send_posts_to_telegram(monkeypatch):
    monkeypatch.setenv(notify.TOKEN_ENV, "123:abc")
    monkeypatch.setenv(notify.CHAT_ENV, "-100")
    calls = {}

    def fake_post(url, json=None, timeout=None):
        calls.update(url=url, json=json)
        return MagicMock(status_code=200)

    monkeypatch.setattr("requests.post", fake_post)
    assert notify.send("hello") is True
    assert "123:abc" in calls["url"]
    assert calls["json"] == {"chat_id": "-100", "text": "hello"}
