from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

from causal_portfolio.execution import trace
from causal_portfolio.execution.control_panel import write_control_panel
from causal_portfolio.execution.types import (
    AccountState,
    Order,
    Position,
    RebalancePlan,
    ReconcileDrift,
    SubmitResult,
    TargetSnapshot,
)


def _target(weight: float, hour: int = 12) -> TargetSnapshot:
    generated = datetime(2026, 7, 22, hour, tzinfo=timezone.utc)
    return TargetSnapshot(
        {"btc": weight},
        as_of=generated,
        generated_at=generated,
        strategy="unit-model",
    )


def _events(path, kind: str):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(
            "SELECT coin, payload_json FROM events WHERE kind=? ORDER BY id", (kind,)
        ).fetchall()


def test_cycle_rotates_once_and_never_overwrites_archive(tmp_path):
    first = _target(0.1)
    second = _target(0.2, hour=13)
    next_time = first.generated_at + timedelta(days=1)

    active = trace.start_cycle(
        first,
        expected_next_rebalance=next_time,
        model_prices={"btc": 100_000.0},
        log_dir=tmp_path,
    )
    assert active == tmp_path / "rebalance-current.sqlite3"
    trace.start_cycle(
        first,
        expected_next_rebalance=next_time,
        model_prices={"btc": 100_000.0},
        log_dir=tmp_path,
    )
    assert len(_events(active, "model_target")) == 1

    assert trace.start_cycle(second, log_dir=tmp_path) == active
    archives = list(tmp_path.glob("rebalance-*-*.sqlite3"))
    assert len(archives) == 1
    with closing(sqlite3.connect(archives[0])) as conn:
        assert conn.execute("SELECT value FROM meta WHERE key='target_id'").fetchone()[0] == first.target_id

    # Re-create the first cycle then reserve its archive name. Rotation must
    # leave the active file untouched instead of overwriting history.
    started = trace.read_meta(active)["started_utc"]
    reserved = trace.archive_path(tmp_path, started, second.target_id)
    reserved.write_text("reserved", encoding="utf-8")
    third = _target(0.3, hour=14)
    assert trace.start_cycle(third, log_dir=tmp_path) is None
    assert trace.read_meta(active)["target_id"] == second.target_id
    assert reserved.read_text(encoding="utf-8") == "reserved"


def test_portfolio_snapshot_and_control_panel_are_persisted(tmp_path):
    target = TargetSnapshot(
        {"btc": 0.1, "evil</script>": 0.0},
        as_of=datetime(2026, 7, 22, 12, tzinfo=timezone.utc),
        generated_at=datetime(2026, 7, 22, 12, tzinfo=timezone.utc),
        strategy="unit-model",
    )
    active = trace.start_cycle(target, model_prices={"btc": 100_000.0}, log_dir=tmp_path)
    state = AccountState(
        "0xAAA",
        10_100.0,
        500.0,
        {"BTC": Position("BTC", 0.01, 100_000.0, 1_010.0)},
    )
    mids = {"BTC": 101_000.0}

    assert trace.record_portfolio_snapshot(state, mids, log_dir=tmp_path)
    rows = _events(active, "portfolio_snapshot")
    assert {row[0] for row in rows} == {None, "BTC", "EVIL</SCRIPT>"}
    aggregate = json.loads(next(payload for coin, payload in rows if coin is None))
    assert aggregate["total_unrealized_pnl_usd"] == 10.0

    panel = write_control_panel(state, mids, log_dir=tmp_path, output_dir=tmp_path / "panel")
    html = panel.read_text(encoding="utf-8")
    assert "unit-model" in html
    assert "Target weight" in html
    assert "https://" not in html
    assert "evil</script>" not in html
    assert "evil\\u003c/script>" in html


def test_execution_result_records_pretrade_and_realized_cost(tmp_path):
    target = _target(0.1)
    active = trace.start_cycle(target, log_dir=tmp_path)
    state = AccountState("0xAAA", 10_000.0, 0.0)
    plan = RebalancePlan(
        1,
        target.weights,
        state,
        {"BTC": 1_000.0},
        {"BTC": 1_000.0},
        [Order("BTC", True, 0.01, 100_200.0, "0x" + "1" * 32)],
        [],
        10_000.0,
        target_snapshot=target,
        mids={"BTC": 100_000.0},
    )
    result = SubmitResult(
        plan,
        True,
        response={
            "response": {
                "data": {
                    "summaries": [
                        {"coin": "BTC", "fills": [{"sz": 0.01, "px": 100_100.0}]}
                    ]
                }
            }
        },
        drifts=[ReconcileDrift("BTC", 0.0, 1_000.0, 1_000.0, float("inf"))],
        cost_estimate={"estimated_taker_fee_bps": 4.5, "estimated_cost_bps": 14.5},
    )

    assert trace.record_execution_result(result, log_dir=tmp_path)
    payload = json.loads(_events(active, "execution_result")[0][1])
    assert payload["cost_estimate"]["estimated_cost_bps"] == 14.5
    assert payload["realized_fill_cost"]["all_in_cost_bps"] > 14.0
    assert payload["drifts"][0]["drift_pct"] is None
