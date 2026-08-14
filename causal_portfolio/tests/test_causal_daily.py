from __future__ import annotations

from datetime import timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import causal_portfolio.models.causal_daily as causal_daily
from causal_portfolio.execution.targets import load_target_snapshot
from causal_portfolio.models.causal_daily import (
    CausalModelNotDeployable,
    MODEL_ASSET,
    MODEL_HISTORY_START,
    MODEL_SPEC_SHA256,
    VALIDATION_DECISION_START,
    VALIDATION_START,
    VALIDATION_OBSERVATIONS,
    _required_snapshot_start,
    build_causal_target,
    build_fixed_factor,
    build_parser,
    evaluate_forward_validation,
    evaluate_signal_ledger,
    exposure_from_signal,
    load_signal_ledger,
    online_innovation_signal,
    record_signal_observation,
    retire_rppca_positions,
    validate_model_signature,
    write_target,
)


def _source_panel(n: int = 2400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range("2021-01-01", periods=n, freq="D")
    common = np.cumsum(rng.normal(scale=0.03, size=n))
    return pd.DataFrame(
        {
            "btc_FeeTotNtv": 1.5 + common + rng.normal(scale=0.02, size=n),
            "doge_FeeTotNtv": 2.0 + 0.7 * common + rng.normal(scale=0.03, size=n),
            "eth_FeeTotNtv": 2.5 + 1.2 * common + rng.normal(scale=0.02, size=n),
        },
        index=index,
    )


def _write_rppca_reference(path) -> None:
    path.write_text(
        causal_daily.json.dumps(
            {
                "btc": -0.02,
                "eth": 0.01,
                "_meta": {
                    "strategy": causal_daily.RP_PCA_STRATEGY_NAME,
                    "rebalance_date": "2026-07-30",
                },
            }
        ),
        encoding="utf-8",
    )


def _model_frames(*, helpful_signal: bool = True):
    panel = _source_panel()
    factors = build_fixed_factor(panel.loc[MODEL_HISTORY_START:])
    signal = online_innovation_signal(factors)
    rng = np.random.default_rng(7)
    simple_returns = pd.Series(
        rng.normal(scale=0.0005, size=len(panel)),
        index=panel.index,
        name="btc_return",
    )
    direction = 1.0 if helpful_signal else -1.0
    for date, value in signal.items():
        next_date = date + pd.Timedelta(days=1)
        if next_date in simple_returns.index:
            simple_returns.loc[next_date] += direction * 0.004 * value
    simple_returns = simple_returns.clip(-0.05, 0.05)
    returns = np.log1p(simple_returns).to_frame()
    prices = pd.DataFrame(
        {MODEL_ASSET: 100.0 * (1.0 + simple_returns).cumprod()},
        index=returns.index,
    )
    return panel, returns, prices, signal


def test_model_signature_is_the_data_supported_lagged_dag_v2_edge():
    signature = validate_model_signature()

    assert signature["version"] == "dag_v2_causal_daily_2026_08_12"
    assert signature["spec_sha256"] == MODEL_SPEC_SHA256
    assert signature["validation_start"] == "2026-08-14"
    assert signature["treatment"] == "chain_congestion"
    assert signature["outcome"] == "btc_return"
    assert signature["lag_days"] == 1
    assert signature["transform"] == "ar1_innovation"
    assert signature["stability"] == "candidate"


def test_fixed_factor_rejects_missing_source_and_ignores_new_fallbacks():
    panel = _source_panel()
    expected = build_fixed_factor(panel)
    with_extra = panel.assign(eth_avg_gas_price_gwei=np.arange(len(panel)))

    pd.testing.assert_frame_equal(build_fixed_factor(with_extra), expected)
    with pytest.raises(ValueError, match="missing frozen DAG v2 source"):
        build_fixed_factor(panel.drop(columns="doge_FeeTotNtv"))

    missing = panel.copy()
    missing.loc[missing.index[100], "eth_FeeTotNtv"] = np.nan
    assert pd.isna(build_fixed_factor(missing).iloc[100, 0])


def test_online_signal_does_not_change_when_future_data_is_perturbed():
    factors = build_fixed_factor(_source_panel())
    cutoff = pd.Timestamp("2027-03-01")
    baseline = online_innovation_signal(factors.loc[:cutoff])
    changed = factors.copy()
    changed.loc[changed.index > cutoff, "chain_congestion"] += 1000.0
    full = online_innovation_signal(changed)

    pd.testing.assert_series_equal(full.loc[baseline.index], baseline)


def test_frozen_forward_validation_uses_first_180_outcomes_and_passes_signal():
    _, returns, _, signal = _model_frames(helpful_signal=True)
    validation = evaluate_forward_validation(signal, returns)

    assert validation.passed is True
    assert validation.status == "passed"
    assert validation.n_available >= VALIDATION_OBSERVATIONS
    assert validation.n_evaluated == VALIDATION_OBSERVATIONS
    assert validation.sharpe_lift >= 0.10
    assert validation.start == VALIDATION_START
    assert validation.end == VALIDATION_START + pd.Timedelta(
        days=VALIDATION_OBSERVATIONS - 1
    )


def test_forward_validation_rejects_a_missing_calendar_outcome():
    _, returns, _, signal = _model_frames(helpful_signal=True)
    returns = returns.drop(index=VALIDATION_START + pd.Timedelta(days=10))

    validation = evaluate_forward_validation(signal, returns)

    assert validation.status == "invalid_data"
    assert validation.passed is False
    assert "missing return date" in validation.reason


def test_forward_validation_converts_loader_log_returns_to_trading_returns():
    outcomes = pd.date_range(
        VALIDATION_START,
        periods=VALIDATION_OBSERVATIONS,
        freq="D",
    )
    decisions = outcomes - pd.Timedelta(days=1)
    signal = pd.Series(0.0, index=decisions)
    returns = pd.DataFrame(
        {"btc_return": np.log1p(np.full(VALIDATION_OBSERVATIONS, 0.01))},
        index=outcomes,
    )

    validation = evaluate_forward_validation(signal, returns)

    assert validation.benchmark_total_return == pytest.approx(
        1.01 ** VALIDATION_OBSERVATIONS - 1.0
    )


def test_signal_ledger_is_hash_chained_and_immutable(tmp_path):
    ledger = tmp_path / "signals.jsonl"
    first_now = VALIDATION_DECISION_START.to_pydatetime().replace(tzinfo=timezone.utc)
    first = record_signal_observation(
        ledger,
        now=first_now,
        source_date=VALIDATION_DECISION_START - pd.Timedelta(days=1),
        signal_z=0.25,
        btc_reference_price=100.0,
    )
    second = record_signal_observation(
        ledger,
        now=first_now + timedelta(days=1),
        source_date=VALIDATION_DECISION_START,
        signal_z=-0.50,
        btc_reference_price=101.0,
    )

    loaded = load_signal_ledger(ledger)

    assert loaded == [first, second]
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert evaluate_signal_ledger(loaded).n_available == 1
    with pytest.raises(ValueError, match="immutable causal decision conflicts"):
        record_signal_observation(
            ledger,
            now=first_now + timedelta(days=1),
            source_date=VALIDATION_DECISION_START,
            signal_z=0.75,
            btc_reference_price=102.0,
        )

    tampered = ledger.read_text(encoding="utf-8").replace('"signal_z": 0.25', '"signal_z": 0.5')
    ledger.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_signal_ledger(ledger)


def test_signal_ledger_validation_uses_first_180_mark_to_mark_outcomes():
    records = []
    price = 100.0
    for index in range(VALIDATION_OBSERVATIONS + 1):
        signal_z = 2.0 if index % 2 == 0 else -2.0
        if index:
            previous_signal = 2.0 if (index - 1) % 2 == 0 else -2.0
            price *= 1.0 + 0.01 * previous_signal
        decision_date = VALIDATION_DECISION_START + pd.Timedelta(days=index)
        records.append(
            {
                "decision_date": decision_date.date().isoformat(),
                "signal_z": signal_z,
                "exposure_multiplier": float(exposure_from_signal(signal_z)),
                "btc_reference_price": price,
            }
        )

    validation = evaluate_signal_ledger(records)

    assert validation.passed is True
    assert validation.n_evaluated == VALIDATION_OBSERVATIONS
    assert validation.start == VALIDATION_START
    assert validation.end == VALIDATION_START + pd.Timedelta(
        days=VALIDATION_OBSERVATIONS - 1
    )


def test_failed_forward_validation_blocks_target_generation():
    panel, returns, prices, _ = _model_frames(helpful_signal=False)
    now = prices.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)

    with pytest.raises(CausalModelNotDeployable, match="not deployable"):
        build_causal_target(panel, returns, prices, now=now)


def test_causal_target_is_lagged_scaled_fresh_and_loadable(tmp_path):
    panel, returns, prices, signal = _model_frames(helpful_signal=True)
    # Production never has a same-day row: Coin Metrics publishes completed
    # days and the CoinGecko filler drops the partial current day. Run as of
    # the following day so the newest row is the model's latest date.
    now = prices.index[-1].to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(days=1, hours=12)
    result = build_causal_target(
        panel,
        returns,
        prices,
        now=now,
        base_weight=0.05,
    )

    expected_multiplier = np.clip(1.0 + 0.5 * signal.iloc[-1], 0.0, 2.0)
    assert result.signal_z == pytest.approx(signal.iloc[-1])
    assert result.exposure_multiplier == pytest.approx(expected_multiplier)
    assert result.weights == {MODEL_ASSET: pytest.approx(0.05 * expected_multiplier)}
    assert 0.0 <= result.weights[MODEL_ASSET] <= 0.10
    assert result.validation.passed is True

    target_path = tmp_path / "causal.json"
    written = write_target(
        result,
        target_path=target_path,
        expected_next_rebalance=now + timedelta(days=1),
        metadata={"model_signature": validate_model_signature()},
    )
    loaded = load_target_snapshot(written.target_path)
    assert loaded.strategy == causal_daily.STRATEGY_NAME
    assert loaded.weights == pytest.approx(result.weights)
    assert loaded.metadata["forward_validation"]["passed"] is True
    assert (
        loaded.metadata["model_signature"]["version"]
        == "dag_v2_causal_daily_2026_08_12"
    )


def test_causal_target_rejects_stale_complete_row():
    panel, returns, prices, _ = _model_frames(helpful_signal=True)
    now = prices.index[-1].to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(days=5)

    with pytest.raises(ValueError, match="inputs are stale"):
        build_causal_target(panel, returns, prices, now=now)


def test_causal_target_rejects_internal_calendar_gap():
    panel, returns, prices, _ = _model_frames(helpful_signal=True)
    missing_day = panel.index[-20]
    panel = panel.drop(index=missing_day)
    now = prices.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="not calendar-contiguous"):
        build_causal_target(panel, returns, prices, now=now)


def test_causal_target_trims_to_the_day_every_source_covers():
    """A lagging source shortens the window instead of stalling the model.

    Coin Metrics publishes per asset at different times, so one series
    routinely sits a day behind the others. Refusing on disagreement stalled
    the model for hours a day; the common floor is the newest date the
    aggregate factor can be computed honestly.
    """
    panel, returns, prices, _ = _model_frames(helpful_signal=True)
    lagging_day = panel.index[-1]
    panel.loc[lagging_day, "eth_FeeTotNtv"] = np.nan
    now = lagging_day.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(days=1)

    result = build_causal_target(panel, returns, prices, now=now)

    assert result.last_data_date == panel.index[-2]


def test_a_hole_inside_the_window_is_still_rejected():
    """Trimming the tail must not become tolerance for a missing middle."""
    panel, returns, prices, _ = _model_frames(helpful_signal=True)
    panel.loc[panel.index[-5], "eth_FeeTotNtv"] = np.nan
    now = panel.index[-1].to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(days=1)

    with pytest.raises(ValueError, match="contiguous"):
        build_causal_target(panel, returns, prices, now=now)


def test_production_history_start_is_not_runtime_tunable():
    parser = build_parser()

    assert parser.parse_args([]).end is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--start", "2023-01-01"])
    assert MODEL_HISTORY_START == "2022-01-01"


def test_snapshot_overlap_uses_oldest_required_series(tmp_path):
    import duckdb

    db = tmp_path / "snapshot.duckdb"
    connection = duckdb.connect(str(db))
    connection.execute(
        "CREATE TABLE asset_metrics(provider TEXT, asset TEXT, metric TEXT, time TIMESTAMP)"
    )
    rows = [
        ("coinmetrics", "btc", "FeeTotNtv", "2026-08-10"),
        ("coinmetrics", "doge", "FeeTotNtv", "2026-08-08"),
        ("coinmetrics", "eth", "FeeTotNtv", "2026-08-09"),
        ("artemis", "btc", "price", "2026-08-11"),
    ]
    connection.executemany("INSERT INTO asset_metrics VALUES (?, ?, ?, ?)", rows)
    connection.close()

    assert _required_snapshot_start(db) == "2026-08-01"


def test_causal_submission_is_long_only_and_uses_standard_policy(tmp_path, monkeypatch):
    target_path = tmp_path / "target.json"
    target_path.write_text(
        '{"btc": 0.05, "_meta": {"rebalance_date": "2026-08-11"}}',
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--min-position-change-pct",
            "0.10",
            "--min-rebalance-completeness",
            "0.90",
        ]
    )
    captured = {}

    def fake_execute(target, config, *, acknowledge_mainnet=False):
        captured.update(target=target, config=config, acknowledge=acknowledge_mainnet)
        return SimpleNamespace(
            submitted=True,
            error=None,
            response={"status": "ok"},
            post_submit_error=None,
            audit_error=None,
        )

    monkeypatch.setattr(causal_daily, "execute_model_target", fake_execute)
    causal_daily._submit_target(args, target_path)

    assert captured["target"].weights == {"btc": 0.05}
    assert captured["config"].testnet is True
    assert captured["config"].dry_run is False
    assert captured["config"].long_only is True
    assert captured["config"].max_transaction_cost_bps == 15.0
    assert captured["config"].min_position_change_pct == 0.10
    assert captured["config"].min_rebalance_completeness == 0.90
    assert captured["acknowledge"] is False


def test_rppca_retirement_is_audited_and_marked_once(tmp_path, monkeypatch):
    marker = tmp_path / "retired.json"
    reference = tmp_path / "rppca.json"
    _write_rppca_reference(reference)
    args = build_parser().parse_args(
        [
            "--retire-rppca",
            "--retirement-marker",
            str(marker),
            "--rppca-reference-target",
            str(reference),
        ]
    )
    now = pd.Timestamp("2026-08-11T20:00:00Z").to_pydatetime()
    calls = []
    state = SimpleNamespace(address="0xtest", positions={})
    monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", state.address)

    def fake_execute(target, config, *, acknowledge_mainnet=False):
        calls.append((target, config, acknowledge_mainnet))
        return SimpleNamespace(
            submitted=False,
            error=None,
            post_submit_error=None,
            audit_error=None,
            post_state=None,
            plan=SimpleNamespace(current_state=state),
        )

    monkeypatch.setattr(causal_daily, "execute_model_target", fake_execute)

    assert retire_rppca_positions(args, now=now) == marker
    assert retire_rppca_positions(args, now=now) == marker
    payload = causal_daily.json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "retired"
    assert payload["network"] == "testnet"
    assert payload["address"] == state.address
    assert len(calls) == 1
    assert calls[0][0].weights == {"btc": 0.0, "eth": 0.0}
    assert calls[0][1].max_single_trade_pct == 1.0
    assert calls[0][1].account_decommission is True
    assert calls[0][1].account_decommission_expected_address == state.address
    assert calls[0][1].account_decommission_expected_sides == (
        ("BTC", -1),
        ("ETH", 1),
    )


def test_rppca_retirement_does_not_mark_unverified_positions(tmp_path, monkeypatch):
    marker = tmp_path / "retired.json"
    reference = tmp_path / "rppca.json"
    _write_rppca_reference(reference)
    args = build_parser().parse_args(
        [
            "--retire-rppca",
            "--retirement-marker",
            str(marker),
            "--rppca-reference-target",
            str(reference),
        ]
    )
    now = pd.Timestamp("2026-08-11T20:00:00Z").to_pydatetime()
    state = SimpleNamespace(
        address="0xtest",
        positions={"ETH": SimpleNamespace(notional_usd=25.0)},
    )
    monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", state.address)
    monkeypatch.setattr(
        causal_daily,
        "execute_model_target",
        lambda *_args, **_kwargs: SimpleNamespace(
            submitted=False,
            error=None,
            post_submit_error=None,
            audit_error=None,
            post_state=None,
            plan=SimpleNamespace(current_state=state),
        ),
    )

    with pytest.raises(RuntimeError, match="left open positions"):
        retire_rppca_positions(args, now=now)
    assert not marker.exists()


def test_rppca_retirement_marker_is_bound_to_wallet(tmp_path, monkeypatch):
    marker = tmp_path / "retired.json"
    reference = tmp_path / "rppca.json"
    _write_rppca_reference(reference)
    reference_target = load_target_snapshot(reference)
    marker.write_text(
        causal_daily.json.dumps(
            {
                "status": "retired",
                "network": "testnet",
                "address": "0xold",
                "reference_target_id": reference_target.target_id,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", "0xnew")
    args = build_parser().parse_args(
        [
            "--retirement-marker",
            str(marker),
            "--rppca-reference-target",
            str(reference),
        ]
    )

    with pytest.raises(RuntimeError, match="invalid RP-PCA retirement marker"):
        retire_rppca_positions(
            args,
            now=pd.Timestamp("2026-08-11T20:00:00Z").to_pydatetime(),
        )


def test_registered_task_compatibility_wrapper_routes_to_causal_model():
    execution_dir = causal_daily.Path(causal_daily.__file__).parents[1] / "execution"
    legacy = (execution_dir / "run_rppca_daily.cmd").read_text(encoding="utf-8")
    causal = (execution_dir / "run_cpcm_causal_daily.cmd").read_text(encoding="utf-8")

    assert "run_cpcm_causal_daily.cmd" in legacy
    assert "models.rppca_daily" not in legacy
    assert "causal_portfolio.models.causal_daily" in causal
    assert "--retire-rppca" in causal
    assert "--refresh-data" in causal
    assert "--base-weight 0.05" in causal
    assert "--min-position-change-pct 0.10" in causal
    assert "--min-rebalance-completeness 0.90" in causal
