from __future__ import annotations

from datetime import datetime, timezone

import pytest

import causal_portfolio.models.rppca_daily as rppca_daily
from causal_portfolio.models.rppca_daily import (
    build_parser,
    forward_fill_prices,
    refresh_local_prices,
    run_once,
)
from causal_portfolio.execution.targets import load_target_snapshot


@pytest.fixture(autouse=True)
def _redirect_trace(tmp_path, monkeypatch):
    from causal_portfolio.execution import audit

    monkeypatch.setattr(audit, "LOG_DIR", tmp_path / "execution-logs")


def test_rppca_daily_generates_loadable_forward_filled_target(tmp_path):
    prices = tmp_path / "prices.csv"
    target = tmp_path / "target.json"
    rows = ["date,btc_price,eth_price,sol_price"]
    for i in range(90):
        rows.append(f"2026-01-{(i % 30) + 1:02d},{100+i},{80+i*0.5},{30+i*0.2}")
    prices.write_text("\n".join(rows), encoding="utf-8")

    args = build_parser().parse_args([
        "--assets", "btc,eth,sol",
        "--prices-csv", str(prices),
        "--start", "2026-01-01",
        "--end", "2026-02-02",
        "--max-ffill-days", "31",
        "--cov-window", "30",
        "--mean-window", "14",
        "--n-components", "2",
        "--target-out", str(target),
    ])

    result = run_once(args)
    loaded = load_target_snapshot(result.target_path)

    assert target.exists()
    assert loaded.strategy == "RP-PCA daily tangency"
    assert loaded.as_of == datetime(2026, 1, 30, tzinfo=timezone.utc)
    assert loaded.metadata["last_data_date"] == "2026-01-30"
    assert loaded.freshness_error(
        72,
        now=datetime(2026, 2, 2, 12, tzinfo=timezone.utc),
    ) is not None
    assert set(loaded.weights) <= {"btc", "eth", "sol"}


def test_refresh_local_prices_upserts_one_batched_snapshot(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "bitcoin": {"usd": 62_751.0},
                "ethereum": {"usd": 1_776.77},
            }

    def fake_get(url, **kwargs):
        assert url.endswith("/simple/price")
        assert kwargs["params"]["ids"] == "bitcoin,ethereum"
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    db = tmp_path / "prices.duckdb"

    count = refresh_local_prices(
        ["btc", "eth"],
        db_path=db,
        now=datetime(2026, 7, 13, 16, 0, tzinfo=timezone.utc),
    )

    import duckdb

    rows = duckdb.connect(str(db), read_only=True).execute(
        "SELECT asset, metric, value FROM asset_metrics ORDER BY asset"
    ).fetchall()
    assert count == 2
    assert rows == [("btc", "price", 62_751.0), ("eth", "price", 1_776.77)]


def test_forward_fill_prices_enforces_limit(tmp_path):
    import pandas as pd

    prices = pd.DataFrame(
        {"btc": [100.0], "eth": [50.0]},
        index=pd.DatetimeIndex(["2026-01-01"]),
    )

    with pytest.raises(ValueError, match="increase --max-ffill-days"):
        forward_fill_prices(prices, end="2026-01-10", max_ffill_days=2)


def test_forward_fill_freshness_uses_stalest_asset_observation():
    import pandas as pd

    prices = pd.DataFrame(
        {
            "btc": [100.0, 101.0, 102.0],
            "eth": [50.0, None, None],
        },
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    filled, last_data_date = forward_fill_prices(
        prices,
        end="2026-01-03",
        max_ffill_days=7,
    )

    assert filled.iloc[-1].notna().all()
    assert last_data_date == pd.Timestamp("2026-01-01")


def test_default_end_resolves_at_run_time():
    args = build_parser().parse_args([])

    assert args.end is None


def test_rppca_submission_uses_standard_interface(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    target_path = tmp_path / "target.json"
    target_path.write_text(
        '{"btc": 0.1, "_meta": {"rebalance_date": "2026-07-13"}}',
        encoding="utf-8",
    )
    args = build_parser().parse_args([
        "--min-position-change-pct", "0.10",
        "--min-rebalance-completeness", "0.90",
    ])
    captured = {}

    def fake_execute(target, config, *, acknowledge_mainnet=False):
        captured.update(
            target=target,
            config=config,
            acknowledge=acknowledge_mainnet,
            calls=captured.get("calls", 0) + 1,
        )
        return SimpleNamespace(
            submitted=True,
            error=None,
            response={"status": "ok"},
            post_submit_error="post-state unavailable",
            audit_error="audit disk full",
        )

    monkeypatch.setattr(rppca_daily, "execute_model_target", fake_execute)
    rppca_daily._submit_target(args, target_path)

    assert captured["target"].weights == {"btc": 0.1}
    assert captured["config"].dry_run is False
    assert captured["config"].testnet is True
    assert captured["config"].max_transaction_cost_bps == 15.0
    assert captured["config"].estimated_taker_fee_bps == 4.5
    assert captured["config"].min_position_change_pct == 0.10
    assert captured["config"].min_rebalance_completeness == 0.90
    assert captured["acknowledge"] is False
    assert captured["calls"] == 1
    assert "post-state unavailable" in caplog.text
    assert "audit disk full" in caplog.text


def test_legacy_rppca_wrapper_delegates_to_scheduled_causal_runner():
    execution_dir = (
        rppca_daily.Path(rppca_daily.__file__).parents[1]
        / "execution"
    )
    legacy = (execution_dir / "run_rppca_daily.cmd").read_text(encoding="utf-8")
    script = (execution_dir / "run_cpcm_causal_daily.cmd").read_text(
        encoding="utf-8"
    )

    assert "run_cpcm_causal_daily.cmd" in legacy
    assert "models.rppca_daily" not in legacy
    assert "--min-position-change-pct 0.10" in script
    assert "--min-rebalance-completeness 0.90" in script


def test_rppca_submission_logs_empty_plan_as_no_op(tmp_path, monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    target_path = tmp_path / "target.json"
    target_path.write_text(
        '{"btc": 0.1, "_meta": {"rebalance_date": "2026-07-13"}}',
        encoding="utf-8",
    )
    args = build_parser().parse_args([])
    monkeypatch.setattr(
        rppca_daily,
        "execute_model_target",
        lambda *_args, **_kwargs: SimpleNamespace(
            submitted=False,
            error=None,
            response=None,
            post_submit_error=None,
            audit_error=None,
        ),
    )
    caplog.set_level(logging.INFO, logger=rppca_daily.logger.name)

    rppca_daily._submit_target(args, target_path)

    assert "RP-PCA rebalance no-op" in caplog.text
    assert "submitted RP-PCA rebalance" not in caplog.text


def test_main_logs_failures_before_execution_audit(tmp_path, monkeypatch):
    from causal_portfolio.execution import run_logging

    def fail(_args):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(run_logging, "LOG_DIR", tmp_path)
    monkeypatch.setattr(rppca_daily, "run_once", fail)

    assert rppca_daily.main([]) == 1
    logs = list(tmp_path.glob("execution-rppca-*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "execution run failed: rppca" in text
    assert "RuntimeError: model exploded" in text
