"""Leg-failure repair pass + Telegram notification."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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


def test_repair_resolves_against_fresh_equity_target():
    plan = _plan()
    cfg = ExecutionConfig(testnet=True, dry_run=False, smart_execution=False,
                          repair_attempts=1)
    adapter = MagicMock(address="0xAAA", config=cfg)
    adapter.submit_orders.return_value = {"status": "ok"}
    adapter.submit_orders_book_aware.return_value = {"status": "ok"}
    adapter.fetch_mids.return_value = {"BTC": 100_000.0}
    adapter.fetch_meta.return_value = {
        "BTC": AssetMeta("BTC", sz_decimals=4, max_leverage=40, min_size=1e-4),
    }
    unfilled = AccountState(address="0xAAA", account_value_usd=9_000.0,
                            margin_used_usd=0.0)
    filled = AccountState(
        address="0xAAA", account_value_usd=9_000.0, margin_used_usd=25.0,
        positions={"BTC": Position("BTC", 0.009, 100_000.0, 900.0)},
    )
    adapter.fetch_state.side_effect = [plan.current_state, unfilled, filled]

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.repair["resolved"] is True
    assert result.drifts == []


def test_repair_retries_pre_submit_error_after_refreshing_state():
    plan = _plan()
    cfg = ExecutionConfig(testnet=True, dry_run=False, smart_execution=False,
                          repair_attempts=2)
    adapter = MagicMock(address="0xAAA", config=cfg)
    adapter.submit_orders.return_value = {"status": "ok"}
    adapter.submit_orders_book_aware.return_value = {"status": "ok"}
    adapter.fetch_mids.side_effect = [
        RuntimeError("temporary market-data error"),
        {"BTC": 100_000.0},
    ]
    adapter.fetch_meta.return_value = {
        "BTC": AssetMeta("BTC", sz_decimals=4, max_leverage=40, min_size=1e-4),
    }
    unfilled = AccountState(address="0xAAA", account_value_usd=10_000.0,
                            margin_used_usd=0.0)
    filled = AccountState(
        address="0xAAA", account_value_usd=10_000.0, margin_used_usd=25.0,
        positions={"BTC": Position("BTC", 0.01, 100_000.0, 1000.0)},
    )
    # cancel-race, initial reconcile, recovery refresh, successful retry
    adapter.fetch_state.side_effect = [plan.current_state, unfilled, unfilled, filled]

    result = execute_plan(adapter, plan, write_audit=False)

    assert adapter.submit_orders_book_aware.call_count == 1
    assert result.repair["resolved"] is True
    assert result.drifts == []


def test_repair_stops_after_ambiguous_submit_error():
    plan = _plan()
    cfg = ExecutionConfig(testnet=True, dry_run=False, smart_execution=False,
                          repair_attempts=2)
    adapter = MagicMock(address="0xAAA", config=cfg)
    adapter.submit_orders.return_value = {"status": "ok"}
    adapter.submit_orders_book_aware.side_effect = RuntimeError("response lost")
    adapter.fetch_mids.return_value = {"BTC": 100_000.0}
    adapter.fetch_meta.return_value = {
        "BTC": AssetMeta("BTC", sz_decimals=4, max_leverage=40, min_size=1e-4),
    }
    unfilled = AccountState(address="0xAAA", account_value_usd=10_000.0,
                            margin_used_usd=0.0)
    adapter.fetch_state.side_effect = [plan.current_state, unfilled, unfilled]

    result = execute_plan(adapter, plan, write_audit=False)

    assert adapter.submit_orders_book_aware.call_count == 1
    assert result.repair["resolved"] is False
    assert result.repair["attempts"][0]["error"] == "response lost"
    assert result.drifts


def test_submit_error_compares_with_post_cancel_state():
    plan = _plan()
    cfg = ExecutionConfig(testnet=True, dry_run=False, smart_execution=False,
                          repair_attempts=0)
    adapter = MagicMock(address="0xAAA", config=cfg)
    adapter.submit_orders.side_effect = RuntimeError("pre-network failure")
    small_move = AccountState(
        address="0xAAA", account_value_usd=10_000.0, margin_used_usd=1.0,
        positions={"BTC": Position("BTC", 0.0001, 100_000.0, 10.0)},
    )
    adapter.fetch_state.side_effect = [small_move, small_move]

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted is False
    assert result.error == "pre-network failure"
    assert result.repair is None


def test_submission_error_repairs_observed_partial_fill():
    plan = _plan()
    cfg = ExecutionConfig(testnet=True, dry_run=False, smart_execution=False,
                          repair_attempts=1)
    adapter = MagicMock(address="0xAAA", config=cfg)
    adapter.submit_orders.side_effect = RuntimeError("response lost")
    adapter.submit_orders_book_aware.return_value = {"status": "ok"}
    adapter.fetch_mids.return_value = {"BTC": 100_000.0}
    adapter.fetch_meta.return_value = {
        "BTC": AssetMeta("BTC", sz_decimals=4, max_leverage=40, min_size=1e-4),
    }
    partial = AccountState(
        address="0xAAA", account_value_usd=10_000.0, margin_used_usd=12.5,
        positions={"BTC": Position("BTC", 0.005, 100_000.0, 500.0)},
    )
    filled = AccountState(
        address="0xAAA", account_value_usd=10_000.0, margin_used_usd=25.0,
        positions={"BTC": Position("BTC", 0.01, 100_000.0, 1000.0)},
    )
    adapter.fetch_state.side_effect = [plan.current_state, partial, filled]

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted is True
    assert result.error is None
    assert "response lost" in result.post_submit_error
    assert result.repair["resolved"] is True
    assert result.drifts == []
    attempt = result.repair["attempts"][0]
    assert attempt["target_usd"] == {"BTC": 1000.0}
    assert attempt["equity_used"] == 10_000.0
    assert attempt["planned_orders"][0]["coin"] == "BTC"
    assert attempt["planned_orders"][0]["cloid"]


def _result(plan=None, **kw):
    return SubmitResult(plan=plan or _plan(), submitted=kw.pop("submitted", True), **kw)


def test_notify_skips_clean_dry_run(monkeypatch):
    sent = []
    monkeypatch.setattr(
        notify, "send",
        lambda text, **kw: sent.append((text, kw)) or True,
    )
    assert notify.notify_result(_result(submitted=False)) is False
    assert notify.notify_result(_result(submitted=True)) is True
    text, kwargs = sent[0]
    assert "Target" in text and "Submitted: <b>True</b>" in text
    assert kwargs == {"parse_mode": "HTML"}


def test_format_pnl_reports_per_position_and_total():
    state = AccountState(
        address="0xAAA1234", account_value_usd=10_500.0, margin_used_usd=200.0,
        positions={
            "BTC": Position("BTC", 0.01, 100_000.0, 1010.0),   # long, up
            "ETH": Position("ETH", -1.0, 3_000.0, -2_950.0),   # short, up
        },
    )
    mids = {"BTC": 101_000.0, "ETH": 2_950.0}

    text = notify.format_pnl(state, mids)

    assert "BTC" in text and "long" in text
    assert "ETH" in text and "short" in text
    # BTC: (101000-100000)*0.01 = +10.00; ETH: (2950-3000)*-1 = +50.00
    assert "$+10.00" in text
    assert "$+50.00" in text
    assert "Total unrealized PnL: $+60.00" in text


def test_format_pnl_flags_missing_mid():
    state = AccountState(
        address="0xAAA1234", account_value_usd=1_000.0, margin_used_usd=0.0,
        positions={"BTC": Position("BTC", 0.01, 100_000.0, 1000.0)},
    )
    text = notify.format_pnl(state, mids={})
    assert "mid price unavailable" in text


def test_send_pnl_update_uses_live_state_and_mids(monkeypatch):
    state = AccountState(address="0xAAA", account_value_usd=1.0, margin_used_usd=0.0)
    mids = {"BTC": 100_000.0}
    monkeypatch.setattr(notify, "_fetch_pnl_snapshot", lambda mainnet: (state, mids))
    captured = {}
    monkeypatch.setattr(
        notify, "send",
        lambda text, **kw: captured.update(text=text, kw=kw) or True,
    )

    assert notify.send_pnl_update() is True
    assert captured["kw"] == {"parse_mode": "HTML"}
    assert "CPCM PnL Update" in captured["text"]


def test_run_pnl_loop_sleeps_between_sends(monkeypatch):
    calls = []
    monkeypatch.setattr(notify, "send_pnl_update", lambda **kw: calls.append(kw) or True)
    sleeps = []
    monkeypatch.setattr(notify.time, "sleep", lambda s: (sleeps.append(s), (_ for _ in ()).throw(KeyboardInterrupt()))[-1])

    with pytest.raises(KeyboardInterrupt):
        notify.run_pnl_loop(interval_minutes=30.0)

    assert calls == [{"mainnet": False}]
    assert sleeps == [1800.0]


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


def test_send_does_not_log_bot_token_on_request_error(monkeypatch, caplog):
    token = "123:secret"
    monkeypatch.setenv(notify.TOKEN_ENV, token)
    monkeypatch.setenv(notify.CHAT_ENV, "-100")

    def fail(url, **_kwargs):
        raise RuntimeError(url)

    monkeypatch.setattr("requests.post", fail)
    with caplog.at_level(logging.WARNING):
        assert notify.send("hello") is False

    assert token not in caplog.text


def test_dotenv_search_starts_from_working_directory(monkeypatch, tmp_path):
    seen = {}
    env_path = tmp_path / ".env"

    def find_dotenv(*, usecwd):
        seen["usecwd"] = usecwd
        return str(env_path)

    fake_dotenv = SimpleNamespace(
        find_dotenv=find_dotenv,
        load_dotenv=lambda path: seen.setdefault("path", path),
    )
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    from causal_portfolio.execution.hyperliquid import _load_dotenv_once
    _load_dotenv_once()

    assert seen == {"usecwd": True, "path": str(env_path)}
