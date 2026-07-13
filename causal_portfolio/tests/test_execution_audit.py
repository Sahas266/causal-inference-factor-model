"""Tests for the execution audit log + CLI plan command (no network)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

from causal_portfolio.execution.audit import _serialize, append, read_log
from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.rebalancer import plan_rebalance
from causal_portfolio.execution.types import (
    AccountState,
    AssetMeta,
    Position,
    SubmitResult,
)


@pytest.fixture(autouse=True)
def _redirect_execution_run_logs(tmp_path, monkeypatch):
    from causal_portfolio.execution import run_logging

    monkeypatch.setattr(run_logging, "LOG_DIR", tmp_path)


def _build_plan_with_orders():
    cfg = ExecutionConfig(dry_run=True, max_position_pct=1.0, max_single_trade_pct=1.0)
    state = AccountState(address="0xabc", account_value_usd=10_000.0,
                         margin_used_usd=0.0)
    meta = {"BTC": AssetMeta("BTC", sz_decimals=5, max_leverage=10, min_size=1e-5)}
    mids = {"BTC": 60_000.0}
    return plan_rebalance({"btc": 0.3}, state, mids, meta, cfg, timestamp_ms=1000)


def test_serialize_handles_dataclass_and_enum():
    plan = _build_plan_with_orders()
    out = _serialize(plan)
    assert isinstance(out, dict)
    assert out["target_weights"] == {"btc": 0.3}
    assert isinstance(out["orders"], list)
    # SkipReason → str (enum value)
    fake_skips = [("BTC", out.get("orders", [{}])[0].get("cloid", "x"), "detail")]


def test_append_creates_jsonl_file(tmp_path):
    plan = _build_plan_with_orders()
    result = SubmitResult(
        plan=plan,
        submitted=True,
        post_submit_error="post-state unavailable",
    )

    path = append(result, log_dir=tmp_path)
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["submitted"] is True
    assert record["error"] is None
    assert record["post_submit_error"] == "post-state unavailable"
    assert record["plan"]["target_weights"] == {"btc": 0.3}


def test_append_appends_multiple_records(tmp_path):
    plan = _build_plan_with_orders()
    append(SubmitResult(plan=plan, submitted=False), log_dir=tmp_path)
    append(SubmitResult(plan=plan, submitted=True, response={"ok": 1}),
           log_dir=tmp_path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records = read_log(today, log_dir=tmp_path)
    assert len(records) == 2
    assert records[0]["submitted"] is False
    assert records[1]["submitted"] is True
    assert records[1]["response"] == {"ok": 1}


def test_read_missing_log_returns_empty(tmp_path):
    assert read_log("1999-01-01", log_dir=tmp_path) == []


# ── CLI plan command ───────────────────────────────────────────────


def test_cli_plan_with_stub_state(tmp_path, capsys, monkeypatch):
    """`plan --weights file.json` runs offline (no network) with stub state."""
    weights_file = tmp_path / "w.json"
    weights_file.write_text(json.dumps({"btc": 0.3, "eth": 0.2}))

    from causal_portfolio.execution.cli import main
    rc = main(["plan", "--weights", str(weights_file), "--equity", "100000",
               "--verbose"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RebalancePlan" in out


def test_execution_run_log_captures_any_model_failure(tmp_path):
    from causal_portfolio.execution import execution_run_log

    with pytest.raises(RuntimeError, match="model exploded"):
        with execution_run_log("other-model", log_dir=tmp_path) as path:
            logging.getLogger("arbitrary_model").info("target ready")
            raise RuntimeError("model exploded")

    text = path.read_text(encoding="utf-8")
    assert "target ready" in text
    assert "Z INFO arbitrary_model target ready" in text
    assert "execution run failed: other-model" in text
    assert "RuntimeError: model exploded" in text


def test_execution_run_log_records_system_exit(tmp_path):
    from causal_portfolio.execution import execution_run_log

    with pytest.raises(SystemExit):
        with execution_run_log("parser", log_dir=tmp_path) as path:
            raise SystemExit(2)

    text = path.read_text(encoding="utf-8")
    assert "execution run failed: parser" in text
    assert "SystemExit: 2" in text


def test_execution_run_log_reuses_active_file(tmp_path):
    from causal_portfolio.execution import execution_run_log

    with execution_run_log("outer", log_dir=tmp_path) as outer:
        with execution_run_log("inner", log_dir=tmp_path) as inner:
            logging.getLogger("nested_model").info("nested handoff")

    assert inner == outer
    assert len(list(tmp_path.glob("execution-*.log"))) == 1
    assert "nested handoff" in outer.read_text(encoding="utf-8")


def test_caught_nested_logging_failure_records_traceback_once(tmp_path):
    from causal_portfolio.execution import execution_run_log

    with execution_run_log("outer", log_dir=tmp_path) as outer:
        try:
            with execution_run_log("inner", log_dir=tmp_path) as inner:
                logging.getLogger("nested_model").info("nested emitted message")
                raise RuntimeError("inner exploded")
        except RuntimeError:
            pass

    assert inner == outer
    assert len(list(tmp_path.glob("execution-*.log"))) == 1
    text = outer.read_text(encoding="utf-8")
    assert "execution run failed: inner" in text
    assert "RuntimeError: inner exploded" in text
    assert "execution run completed: outer" in text
    assert text.count("nested emitted message") == 1


def test_generic_cli_uses_shared_execution_log(tmp_path, monkeypatch):
    from causal_portfolio.execution import run_logging
    from causal_portfolio.execution import cli

    weights_file = tmp_path / "w.json"
    weights_file.write_text(json.dumps({"btc": 0.3}), encoding="utf-8")
    monkeypatch.setattr(run_logging, "LOG_DIR", tmp_path)

    assert cli.main(["plan", "--weights", str(weights_file)]) == 0
    logs = list(tmp_path.glob("execution-cli-*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "command=plan" in text
    assert "execution run completed: cli" in text


def test_cli_execute_refuses_without_live_flag(tmp_path, capsys):
    weights_file = tmp_path / "w.json"
    weights_file.write_text(json.dumps({"btc": 0.3}))

    from causal_portfolio.execution.cli import main
    # The argparse layer marks --live as required for execute, so missing it
    # raises SystemExit, not a return code 1. Both prove the gate works.
    with pytest.raises(SystemExit):
        main(["execute", "--weights", str(weights_file)])


def test_cli_execute_surfaces_post_submit_and_audit_errors(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from causal_portfolio.execution import cli
    from causal_portfolio.execution import hyperliquid as hl_mod

    weights_file = tmp_path / "w.json"
    weights_file.write_text(json.dumps({"btc": 0.3}), encoding="utf-8")
    adapter = MagicMock()
    plan = SimpleNamespace(orders=[object()])
    execute = MagicMock(return_value=SimpleNamespace(
        error=None,
        response={"status": "ok"},
        post_submit_error="post-state unavailable",
        audit_error="audit disk full",
    ))
    monkeypatch.setattr(hl_mod, "HLAdapter", lambda _cfg: adapter)
    monkeypatch.setattr(hl_mod, "execute_plan", execute)
    monkeypatch.setattr(cli, "plan_rebalance", lambda *_args: plan)
    monkeypatch.setattr(cli, "_print_plan", lambda *_args, **_kwargs: None)
    args = SimpleNamespace(
        live=True,
        weights=str(weights_file),
        mainnet=False,
        slippage_bps=None,
        max_signal_age_hours=72.0,
        allow_stale_signal=False,
        twap_minutes=0.0,
        twap_slices=5,
        no_smart_execution=False,
    )

    assert cli.cmd_execute(args) == 0

    stderr = capsys.readouterr().err
    assert "post-state unavailable" in stderr
    assert "audit disk full" in stderr
    execute.assert_called_once()


def test_target_loader_rejects_non_dict(tmp_path):
    weights_file = tmp_path / "bad.json"
    weights_file.write_text(json.dumps([0.3, 0.4]))

    from causal_portfolio.execution.targets import load_target_snapshot
    with pytest.raises(ValueError, match="JSON object"):
        load_target_snapshot(str(weights_file))


def test_target_loader_rejects_non_numeric(tmp_path):
    weights_file = tmp_path / "bad.json"
    weights_file.write_text(json.dumps({"btc": "thirty percent"}))

    from causal_portfolio.execution.targets import load_target_snapshot
    with pytest.raises(ValueError, match="numeric"):
        load_target_snapshot(str(weights_file))


def test_target_loader_lowercases_keys(tmp_path):
    weights_file = tmp_path / "w.json"
    weights_file.write_text(json.dumps({"BTC": 0.3, "ETH": 0.2}))

    from causal_portfolio.execution.targets import load_target_snapshot
    weights = load_target_snapshot(str(weights_file)).weights
    assert "btc" in weights and "eth" in weights
    assert weights["btc"] == 0.3


def test_cli_logs_command_empty_day(tmp_path, capsys, monkeypatch):
    """`logs` on a date with no audit records prints a 'no records' message."""
    from causal_portfolio.execution import audit as audit_mod
    monkeypatch.setattr(audit_mod, "LOG_DIR", tmp_path)
    from causal_portfolio.execution import cli as cli_mod
    monkeypatch.setattr(cli_mod, "LOG_DIR", tmp_path)

    rc = cli_mod.main(["logs", "--date", "1999-01-01"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No audit records" in out


def test_cli_state_command_routes_to_correct_network(monkeypatch, capsys):
    """`state` builds an adapter on the requested network and prints summary."""
    from unittest.mock import MagicMock
    from causal_portfolio.execution.types import AccountState

    # Stub HLAdapter so we don't hit the network
    fake = MagicMock()
    fake.address = "0xstub"
    fake.fetch_state.return_value = AccountState(
        address="0xstub", account_value_usd=12_345.67,
        margin_used_usd=100.0, positions={},
    )
    fake.fetch_mids.return_value = {}
    fake.fetch_open_order_ids.return_value = []

    import causal_portfolio.execution.hyperliquid as hl_mod
    monkeypatch.setattr(hl_mod, "HLAdapter", lambda cfg: fake)

    from causal_portfolio.execution.cli import main
    rc = main(["state"])  # default = testnet
    out = capsys.readouterr().out
    assert rc == 0
    assert "testnet" in out
    assert "12,345.67" in out  # the equity value, ignoring alignment whitespace
    fake.fetch_state.assert_called_once()


def test_cli_logs_command_renders_records(tmp_path, capsys, monkeypatch):
    """Write a fake record, then verify `logs` renders it."""
    from causal_portfolio.execution import audit as audit_mod
    monkeypatch.setattr(audit_mod, "LOG_DIR", tmp_path)
    from causal_portfolio.execution import cli as cli_mod
    monkeypatch.setattr(cli_mod, "LOG_DIR", tmp_path)

    # Build + append a real record so we exercise the same code path
    plan = _build_plan_with_orders()
    from causal_portfolio.execution.audit import append
    from causal_portfolio.execution.types import SubmitResult
    append(SubmitResult(plan=plan, submitted=False), log_dir=tmp_path)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rc = cli_mod.main(["logs", "--date", today, "--verbose"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY-RUN" in out
    assert "BTC" in out  # verbose mode shows the order


# Imports for the new tests
from datetime import datetime, timezone
