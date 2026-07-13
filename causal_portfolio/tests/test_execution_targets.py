"""Tests for provenance-preserving model targets and execution safety gates."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from causal_portfolio.execution.audit import append, read_log
from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.hyperliquid import execute_plan
from causal_portfolio.execution.rebalancer import plan_rebalance
from causal_portfolio.execution.targets import load_target_snapshot
from causal_portfolio.execution.types import (
    AccountState,
    AssetMeta,
    Order,
    SubmitResult,
    TargetSnapshot,
)


def _market():
    state = AccountState(
        address="0xAAA",
        account_value_usd=10_000.0,
        margin_used_usd=0.0,
    )
    mids = {"BTC": 50_000.0, "ETH": 2_500.0}
    meta = {
        "BTC": AssetMeta("BTC", sz_decimals=5, max_leverage=40, min_size=1e-5),
        "ETH": AssetMeta("ETH", sz_decimals=4, max_leverage=25, min_size=1e-4),
    }
    return state, mids, meta


def _adapter(config: ExecutionConfig):
    state, _, _ = _market()
    adapter = MagicMock()
    adapter.address = state.address
    adapter.config = config
    adapter.cancel_all_open.return_value = None
    adapter.fetch_state.return_value = state
    adapter.submit_orders_book_aware.return_value = {"status": "ok"}
    adapter.submit_orders.return_value = {"status": "ok"}
    return adapter


def test_load_legacy_json_retains_provenance(tmp_path):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({
        "BTC": 0.3,
        "eth": -0.2,
        "_meta": {
            "rebalance_date": "2026-06-22",
            "generated_utc": "2026-06-22T01:02:03Z",
            "strategy": "CPCM-v1",
        },
    }))

    target = load_target_snapshot(path)

    assert target.weights == {"btc": 0.3, "eth": -0.2}
    assert target.as_of == datetime(2026, 6, 22, tzinfo=timezone.utc)
    assert target.generated_at == datetime(2026, 6, 22, 1, 2, 3, tzinfo=timezone.utc)
    assert target.strategy == "CPCM-v1"
    assert len(target.target_id) == 16


def test_load_reference_rebalance_csv_selects_latest_row(tmp_path):
    path = tmp_path / "signals.csv"
    path.write_text(
        "rebalance_date,BTC,ETH,_strategy\n"
        "2026-06-20,0.10,-0.05,RP-PCA Tangency\n"
        "2026-06-22,0.30,0.20,RP-PCA Tangency\n",
        encoding="utf-8",
    )

    target = load_target_snapshot(path)

    assert target.weights == {"btc": 0.3, "eth": 0.2}
    assert target.as_of == datetime(2026, 6, 22, tzinfo=timezone.utc)
    assert target.strategy == "RP-PCA Tangency"
    assert target.metadata["rows"] == 2


def test_load_csv_accepts_weight_prefix_and_rejects_unknown_metadata(tmp_path):
    prefixed = tmp_path / "prefixed.csv"
    prefixed.write_text(
        "rebalance_date,weight_btc,_turnover\n"
        "2026-06-22,0.25,0.80\n",
        encoding="utf-8",
    )
    target = load_target_snapshot(prefixed)
    assert target.weights == {"btc": 0.25}

    bad = tmp_path / "bad.csv"
    bad.write_text(
        "rebalance_date,BTC,turnover\n"
        "2026-06-22,0.25,0.80\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown target CSV column"):
        load_target_snapshot(bad)


def test_target_id_is_stable_across_source_paths():
    as_of = datetime(2026, 6, 22, tzinfo=timezone.utc)
    left = TargetSnapshot({"btc": 0.3, "eth": 0.2}, as_of=as_of, source="a.json")
    right = TargetSnapshot({"ETH": 0.2, "BTC": 0.3}, as_of=as_of, source="b.csv")
    assert left.target_id == right.target_id


def test_target_rejects_non_finite_weight():
    with pytest.raises(ValueError, match="finite"):
        TargetSnapshot({"btc": float("nan")})


def test_freshness_reports_missing_stale_and_future_targets():
    now = datetime(2026, 6, 23, 12, tzinfo=timezone.utc)
    assert "missing" in TargetSnapshot({"btc": 1.0}).freshness_error(72, now=now)
    stale = TargetSnapshot({"btc": 1.0}, as_of=now - timedelta(hours=73))
    assert "stale" in stale.freshness_error(72, now=now)
    future = TargetSnapshot({"btc": 1.0}, as_of=now + timedelta(hours=1))
    assert "future" in future.freshness_error(72, now=now)
    fresh = TargetSnapshot({"btc": 1.0}, as_of=now - timedelta(hours=12))
    assert fresh.freshness_error(72, now=now) is None


def test_plan_carries_target_and_network_provenance():
    state, mids, meta = _market()
    target = TargetSnapshot(
        {"btc": 0.1},
        as_of=datetime.now(timezone.utc),
        strategy="CPCM-v1",
    )

    plan = plan_rebalance(
        target, state, mids, meta,
        ExecutionConfig(testnet=True, max_single_trade_pct=1.0),
        timestamp_ms=1000,
    )

    assert plan.network == "testnet"
    assert plan.target_snapshot is target
    assert plan.target_weights == {"btc": 0.1}
    assert target.target_id in plan.notes[0]


def test_execute_target_owns_model_handoff(tmp_path, monkeypatch):
    from causal_portfolio.execution import audit, execute_target, run_logging
    import causal_portfolio.execution.hyperliquid as hl

    cfg = ExecutionConfig(testnet=True, dry_run=True)
    state, mids, meta = _market()
    adapter = _adapter(cfg)
    adapter.fetch_mids.return_value = mids
    adapter.fetch_meta.return_value = meta
    monkeypatch.setattr(hl, "HLAdapter", lambda config: adapter)
    monkeypatch.setattr(audit, "LOG_DIR", tmp_path)
    monkeypatch.setattr(run_logging, "LOG_DIR", tmp_path)
    target = TargetSnapshot(
        {"btc": 0.1},
        as_of=datetime.now(timezone.utc),
        strategy="unit model",
    )

    result = execute_target(target, cfg)

    assert result.plan.target_snapshot is target
    assert result.submitted is False
    adapter.fetch_state.assert_called_once_with()
    adapter.fetch_mids.assert_called_once_with()
    adapter.fetch_meta.assert_called_once_with()
    assert len(list(tmp_path.glob("execution-unit-model-*.log"))) == 1
    assert len(list(tmp_path.glob("rebalance-*.jsonl"))) == 1


def test_execute_target_rejects_unversioned_dict():
    from causal_portfolio.execution import execute_target

    with pytest.raises(TypeError, match="TargetSnapshot"):
        execute_target({"btc": 0.1}, ExecutionConfig())


def test_network_mismatch_blocks_before_any_write():
    state, mids, meta = _market()
    plan = plan_rebalance(
        {"btc": 0.1}, state, mids, meta,
        ExecutionConfig(testnet=True, max_single_trade_pct=1.0),
    )
    adapter = _adapter(ExecutionConfig(testnet=False, dry_run=False))

    result = execute_plan(
        adapter, plan, acknowledge_mainnet=True, write_audit=False,
    )

    assert not result.submitted
    assert "network mismatch" in (result.error or "")
    adapter.cancel_all_open.assert_not_called()


def test_reverse_network_mismatch_blocks_before_any_write():
    state, mids, meta = _market()
    target = TargetSnapshot({"btc": 0.1}, as_of=datetime.now(timezone.utc))
    plan = plan_rebalance(
        target, state, mids, meta,
        ExecutionConfig(testnet=False, max_single_trade_pct=1.0),
    )
    adapter = _adapter(ExecutionConfig(testnet=True, dry_run=False))

    result = execute_plan(adapter, plan, write_audit=False)

    assert not result.submitted
    assert "network mismatch" in (result.error or "")
    adapter.cancel_all_open.assert_not_called()


def test_stale_snapshot_blocks_live_execution_unless_overridden():
    state, mids, meta = _market()
    stale = TargetSnapshot(
        {"btc": 0.1},
        as_of=datetime.now(timezone.utc) - timedelta(days=10),
    )
    plan = plan_rebalance(
        stale, state, mids, meta,
        ExecutionConfig(testnet=True, max_single_trade_pct=1.0),
    )

    blocked_adapter = _adapter(ExecutionConfig(testnet=True, dry_run=False))
    blocked = execute_plan(blocked_adapter, plan, write_audit=False)
    assert not blocked.submitted
    assert "freshness check failed" in (blocked.error or "")
    blocked_adapter.cancel_all_open.assert_not_called()

    override_adapter = _adapter(ExecutionConfig(
        testnet=True, dry_run=False, allow_stale_signal=True,
    ))
    overridden = execute_plan(override_adapter, plan, write_audit=False)
    assert overridden.submitted
    override_adapter.cancel_all_open.assert_called_once()


def test_twap_execution_submits_deterministic_child_orders(monkeypatch):
    state, mids, meta = _market()
    cfg = ExecutionConfig(
        testnet=True,
        dry_run=False,
        smart_execution=False,
        twap_minutes=0.01,
        twap_slices=3,
        max_single_trade_pct=1.0,
    )
    plan = plan_rebalance(
        TargetSnapshot({"btc": 0.3}, as_of=datetime.now(timezone.utc)),
        state,
        mids,
        meta,
        cfg,
        timestamp_ms=1000,
    )
    parent_order = plan.orders[0]
    adapter = _adapter(cfg)
    adapter.fetch_meta.return_value = meta
    sleeps: list[float] = []
    monkeypatch.setattr(
        "causal_portfolio.execution.hyperliquid.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    assert result.response["response"]["type"] == "twap"
    assert adapter.submit_orders.call_count == 3
    submitted_children = [
        call.args[0][0] for call in adapter.submit_orders.call_args_list
    ]
    assert sum(child.size for child in submitted_children) == pytest.approx(parent_order.size)
    assert all(child.cloid != parent_order.cloid for child in submitted_children)
    assert len({child.cloid for child in submitted_children}) == 3
    assert len(sleeps) == 2
    adapter.submit_orders_book_aware.assert_not_called()


def test_twap_slicer_spreads_tiny_orders_across_window():
    from causal_portfolio.execution.hyperliquid import _slice_order

    parent = Order(
        coin="BTC",
        is_buy=True,
        size=0.00002,  # two 1e-5 exchange units
        limit_px=50_000.0,
        cloid="0x" + "1" * 32,
    )

    children = _slice_order(parent, slices=5, sz_decimals=5)

    assert [idx for idx, _ in children] == [2, 4]
    assert sum(child.size for _, child in children) == pytest.approx(parent.size)
    assert len({child.cloid for _, child in children}) == 2


def test_twap_slicer_caps_effective_slices_to_child_min_notional():
    from causal_portfolio.execution.hyperliquid import _slice_order

    parent = Order(
        coin="BTC",
        is_buy=True,
        size=0.0004,  # $20 at $50k; three slices would be sub-$10
        limit_px=50_000.0,
        cloid="0x" + "2" * 32,
    )

    children = _slice_order(
        parent,
        slices=3,
        sz_decimals=5,
        min_child_notional_usd=10.0,
    )

    assert [idx for idx, _ in children] == [0, 2]
    assert len(children) == 2
    assert sum(child.size for _, child in children) == pytest.approx(parent.size)
    assert all(child.size * child.limit_px >= 10.0 for _, child in children)


def test_twap_slicer_uses_buffer_for_barely_over_minimum_children():
    from causal_portfolio.execution.hyperliquid import _slice_order

    parent = Order(
        coin="BTC",
        is_buy=True,
        size=0.00034,
        limit_px=58_861.0,  # ~$20.01 parent; two children are barely over $10
        cloid="0x" + "3" * 32,
    )

    children = _slice_order(
        parent,
        slices=3,
        sz_decimals=5,
        min_child_notional_usd=10.50,
    )

    assert len(children) == 1
    assert children[0][1].size == pytest.approx(parent.size)


def test_twap_response_logs_effective_child_plan_for_postmortem(monkeypatch):
    state, mids, meta = _market()
    cfg = ExecutionConfig(
        testnet=True,
        dry_run=False,
        smart_execution=False,
        twap_minutes=0.01,
        twap_slices=3,
        max_single_trade_pct=1.0,
    )
    plan = plan_rebalance(
        TargetSnapshot({"btc": 0.0015}, as_of=datetime.now(timezone.utc)),
        state,
        mids,
        meta,
        cfg,
        timestamp_ms=1000,
    )
    adapter = _adapter(cfg)
    adapter.fetch_meta.return_value = meta
    monkeypatch.setattr("causal_portfolio.execution.hyperliquid.time.sleep", lambda _: None)

    result = execute_plan(adapter, plan, write_audit=False)

    child_plan = result.response["response"]["data"]["child_plan"]
    assert child_plan[0]["requested_slices"] == 3
    assert child_plan[0]["effective_slices"] == 1
    assert 10.0 <= child_plan[0]["parent_notional_usd"] < 30.0
    assert child_plan[0]["children"][0]["notional_usd"] >= 10.0
    assert adapter.submit_orders.call_count == 1


def test_twap_execution_can_use_book_aware_child_slices(monkeypatch):
    state, mids, meta = _market()
    cfg = ExecutionConfig(
        testnet=True,
        dry_run=False,
        smart_execution=True,
        twap_minutes=0.01,
        twap_slices=2,
        max_single_trade_pct=1.0,
    )
    plan = plan_rebalance(
        TargetSnapshot({"btc": 0.3}, as_of=datetime.now(timezone.utc)),
        state,
        mids,
        meta,
        cfg,
        timestamp_ms=1000,
    )
    adapter = _adapter(cfg)
    adapter.fetch_meta.return_value = meta
    monkeypatch.setattr("causal_portfolio.execution.hyperliquid.time.sleep", lambda _: None)

    result = execute_plan(adapter, plan, write_audit=False)

    assert result.submitted
    assert adapter.submit_orders_book_aware.call_count == 2
    adapter.submit_orders.assert_not_called()


def test_twap_execution_writes_audit_record(tmp_path, monkeypatch):
    state, mids, meta = _market()
    cfg = ExecutionConfig(
        testnet=True,
        dry_run=False,
        smart_execution=False,
        twap_minutes=0.01,
        twap_slices=2,
        max_single_trade_pct=1.0,
    )
    target = TargetSnapshot({"btc": 0.3}, as_of=datetime.now(timezone.utc))
    plan = plan_rebalance(target, state, mids, meta, cfg, timestamp_ms=1000)
    adapter = _adapter(cfg)
    adapter.fetch_meta.return_value = meta
    monkeypatch.setattr("causal_portfolio.execution.hyperliquid.time.sleep", lambda _: None)
    from causal_portfolio.execution import audit as audit_mod
    monkeypatch.setattr(audit_mod, "LOG_DIR", tmp_path)

    result = execute_plan(adapter, plan, write_audit=True)

    assert result.submitted
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    record = read_log(today, log_dir=tmp_path)[0]
    assert record["target_id"] == target.target_id
    assert record["response"]["response"]["type"] == "twap"
    assert "drifts" in record


def test_audit_indexes_target_id_and_network(tmp_path):
    state, mids, meta = _market()
    target = TargetSnapshot(
        {"btc": 0.1},
        as_of=datetime.now(timezone.utc),
    )
    plan = plan_rebalance(
        target, state, mids, meta,
        ExecutionConfig(testnet=True, max_single_trade_pct=1.0),
    )

    append(SubmitResult(plan=plan, submitted=False), log_dir=tmp_path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    record = read_log(today, log_dir=tmp_path)[0]

    assert record["target_id"] == target.target_id
    assert record["network"] == "testnet"


def test_config_rejects_non_positive_signal_age():
    with pytest.raises(ValueError, match="max_signal_age_hours"):
        ExecutionConfig(max_signal_age_hours=0)


def test_cli_accepts_explicit_testnet_flag(monkeypatch):
    from causal_portfolio.execution import cli

    captured = {}

    def fake_execute(args):
        captured["mainnet"] = args.mainnet
        return 0

    monkeypatch.setattr(cli, "cmd_execute", fake_execute)
    assert cli.main([
        "execute", "--weights", "unused.json", "--live", "--testnet",
    ]) == 0
    assert captured["mainnet"] is False
