from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_portfolio.market_making.testnet_execution_benchmark import (
    FatalBenchmarkError,
    TestnetLeg as _TestnetLeg,
    _execute_as_pin_leg,
    _execute_naive_leg,
    _leg_from_fills,
    _post_only_price,
    _restore_position,
    _roundtrip_pnl,
    _wait_for_ioc_fills,
    bootstrap_pin_signal,
    run_testnet_execution_campaign,
    run_testnet_execution_benchmark,
)
from causal_portfolio.market_making.calibration import MarkoutCurve
from causal_portfolio.market_making.pin import PINFit
from causal_portfolio.market_making.types import (
    AggressorSide,
    BookLevel,
    BookSnapshot,
    TradePrint,
)


def test_testnet_benchmark_requires_explicit_write_acknowledgement():
    with pytest.raises(PermissionError, match="acknowledge"):
        run_testnet_execution_benchmark(
            size=0.0002,
            capture_seconds=1,
            passive_timeout_seconds=1,
            acknowledge_testnet_writes=False,
        )


def test_testnet_campaign_requires_explicit_write_acknowledgement(tmp_path):
    with pytest.raises(PermissionError, match="acknowledge"):
        run_testnet_execution_campaign(
            size=0.0002,
            max_seconds=1,
            max_attempts=1,
            acknowledge_testnet_writes=False,
            output_dir=tmp_path,
        )


def test_testnet_campaign_records_failed_attempts_and_continues(monkeypatch, tmp_path):
    calls = {"n": 0}

    class FakeResult:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return dict(self._payload)

    def fake_benchmark(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("unusable PIN")
        maker_buy = 0.0002 if calls["n"] == 2 else 0.0
        maker_sell = 0.0002 if calls["n"] == 3 else 0.0
        return FakeResult({
            "restored_initial_position": True,
            "pin_fit_usable": True,
            "as_pin_buy": {"maker_filled_size": maker_buy},
            "as_pin_sell": {"maker_filled_size": maker_sell},
        })

    monkeypatch.setattr(
        "causal_portfolio.market_making.testnet_execution_benchmark.run_testnet_execution_benchmark",
        fake_benchmark,
    )

    summary = run_testnet_execution_campaign(
        size=0.0002,
        max_seconds=60,
        capture_seconds=1,
        passive_timeout_seconds=1,
        min_successful_attempts=2,
        max_attempts=5,
        acknowledge_testnet_writes=True,
        output_dir=tmp_path,
    )

    assert summary["attempts"] == 3
    assert summary["successful_restores"] == 2
    assert summary["pin_usable_attempts"] == 2
    assert summary["maker_buy_attempts"] == 1
    assert summary["maker_sell_attempts"] == 1
    assert summary["results"][0]["ok"] is False
    assert "unusable PIN" in summary["results"][0]["error"]
    assert (tmp_path / "full_execution_testnet_result_attempt_01.json").exists()
    assert (tmp_path / "full_execution_testnet_summary.json").exists()


def test_testnet_campaign_stops_on_fatal_cleanup_error(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_benchmark(**_kwargs):
        calls["n"] += 1
        raise FatalBenchmarkError("cleanup failed: final BTC position did not restore")

    monkeypatch.setattr(
        "causal_portfolio.market_making.testnet_execution_benchmark.run_testnet_execution_benchmark",
        fake_benchmark,
    )

    summary = run_testnet_execution_campaign(
        size=0.0002,
        max_seconds=60,
        capture_seconds=1,
        passive_timeout_seconds=1,
        min_successful_attempts=2,
        max_attempts=5,
        acknowledge_testnet_writes=True,
        output_dir=tmp_path,
    )

    assert calls["n"] == 1
    assert summary["attempts"] == 1
    assert summary["fatal_errors"] == 1
    assert summary["results"][0]["fatal"] is True


def test_restore_position_uses_reduce_only_and_retries(monkeypatch):
    calls = []

    class FakeAdapter:
        def __init__(self):
            self.inventory = 0.00034

        def _size_decimals(self, _coin):
            return 5

        def fetch_account_metrics(self, _coin):
            return {"inventory_base": self.inventory}

    adapter = FakeAdapter()

    def fake_execute_ioc(_adapter, **kwargs):
        calls.append(kwargs)
        assert kwargs["reduce_only"] is True
        assert kwargs["attempts"] == 4
        assert kwargs["max_band_bps"] == 50.0
        assert kwargs["require_fee_fills"] is False
        adapter.inventory = 0.0
        return 100.0, set(), []

    monkeypatch.setattr(
        "causal_portfolio.market_making.testnet_execution_benchmark._execute_ioc",
        fake_execute_ioc,
    )
    monkeypatch.setattr(
        "causal_portfolio.market_making.testnet_execution_benchmark.time.sleep",
        lambda _seconds: None,
    )

    final = _restore_position(adapter, 0.0, attempts=3, poll_seconds=0.0)

    assert final["inventory_base"] == 0.0
    assert len(calls) == 1
    assert calls[0]["is_buy"] is False
    assert calls[0]["size"] == pytest.approx(0.00034)


def test_post_only_price_survives_precision_rounding_at_touch():
    book = BookSnapshot(
        timestamp_ms=1_700_000_000_000,
        coin="BTC",
        bids=(BookLevel(58580.0, 1.0),),
        asks=(BookLevel(58590.0, 1.0),),
    )

    bid = _post_only_price(
        58589.9,
        book,
        is_buy=True,
        sz_decimals=5,
        gap_usd=1.0,
    )
    ask = _post_only_price(
        58580.1,
        book,
        is_buy=False,
        sz_decimals=5,
        gap_usd=1.0,
    )

    assert bid < book.best_ask
    assert ask > book.best_bid


def test_naive_leg_can_submit_reduce_only_close(monkeypatch):
    calls = []

    def fake_execute_ioc(_adapter, **kwargs):
        calls.append(kwargs)
        return 100.0, {123}, [{"oid": 123, "sz": "0.00034", "px": "100", "fee": "0.001"}]

    monkeypatch.setattr(
        "causal_portfolio.market_making.testnet_execution_benchmark._execute_ioc",
        fake_execute_ioc,
    )

    leg = _execute_naive_leg(
        object(),
        is_buy=False,
        size=0.00034,
        label="naive-sell",
        reduce_only=True,
    )

    assert leg.status == "filled"
    assert calls[0]["reduce_only"] is True
    assert calls[0]["is_buy"] is False


def test_leg_from_fills_computes_shortfall_and_fill_split():
    leg = _leg_from_fills(
        arm="as_pin_post_only_then_ioc",
        is_buy=True,
        reference_mid=100.0,
        requested_size=2.0,
        maker_fills=[{"sz": "1", "px": "101", "fee": "0.10"}],
        fallback_fills=[{"sz": "1", "px": "102", "fee": "0.20"}],
        elapsed_seconds=3.0,
    )

    assert leg.status == "filled"
    assert leg.filled_size == pytest.approx(2.0)
    assert leg.maker_filled_size == pytest.approx(1.0)
    assert leg.fallback_filled_size == pytest.approx(1.0)
    assert leg.average_price == pytest.approx(101.5)
    assert leg.fee_usd == pytest.approx(0.30)
    assert leg.implementation_shortfall_bps == pytest.approx(165.0)


def test_roundtrip_pnl_uses_min_filled_size_and_fees():
    buy = _TestnetLeg(
        arm="naive_ioc",
        side="buy",
        reference_mid=100.0,
        requested_size=2.0,
        filled_size=2.0,
        maker_filled_size=0.0,
        fallback_filled_size=2.0,
        average_price=100.0,
        fee_usd=0.10,
        implementation_shortfall_bps=5.0,
        elapsed_seconds=1.0,
        status="filled",
    )
    sell = _TestnetLeg(
        arm="naive_ioc",
        side="sell",
        reference_mid=102.0,
        requested_size=3.0,
        filled_size=3.0,
        maker_filled_size=0.0,
        fallback_filled_size=3.0,
        average_price=102.0,
        fee_usd=0.10,
        implementation_shortfall_bps=5.0,
        elapsed_seconds=1.0,
        status="filled",
    )

    assert _roundtrip_pnl(buy, sell) == pytest.approx(3.80)


def test_wait_for_ioc_fills_refuses_synthetic_fee_zero(monkeypatch):
    monkeypatch.setattr(
        "causal_portfolio.market_making.testnet_execution_benchmark._fills_for_oids",
        lambda *_: [],
    )
    monkeypatch.setattr(
        "causal_portfolio.market_making.testnet_execution_benchmark.time.sleep",
        lambda _: None,
    )

    with pytest.raises(RuntimeError, match="refusing to synthesize fee=0"):
        _wait_for_ioc_fills(
            adapter=object(),
            oids={123},
            start_ms=1_700_000_000_000,
            parsed_filled_size=0.0002,
            attempts=2,
            poll_seconds=0.0,
        )


def test_wait_for_ioc_fills_retries_until_real_fill(monkeypatch):
    calls = {"n": 0}

    def fake_fills(*_):
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        return [{"oid": 123, "sz": "0.0002", "px": "100", "fee": "0.001"}]

    monkeypatch.setattr(
        "causal_portfolio.market_making.testnet_execution_benchmark._fills_for_oids",
        fake_fills,
    )
    monkeypatch.setattr(
        "causal_portfolio.market_making.testnet_execution_benchmark.time.sleep",
        lambda _: None,
    )

    fills = _wait_for_ioc_fills(
        adapter=object(),
        oids={123},
        start_ms=1_700_000_000_000,
        parsed_filled_size=0.0002,
        attempts=2,
        poll_seconds=0.0,
    )

    assert fills[0]["fee"] == "0.001"


def test_as_pin_leg_refuses_unusable_pin_estimator():
    class FakeAdapter:
        def fetch_l2_book(self, *_args, **_kwargs):
            return BookSnapshot(
                timestamp_ms=1_700_000_000_000,
                coin="BTC",
                bids=(BookLevel(99.0, 1.0),),
                asks=(BookLevel(101.0, 1.0),),
            )

    unusable_fit = PINFit(
        alpha=0.5,
        delta=0.5,
        mu=1.0,
        epsilon=1.0,
        pin=0.2,
        log_likelihood=-1.0,
        sample_days=10,
        converged=False,
        boundary_solution=False,
        hessian_condition=None,
        standard_errors=None,
        starts=1,
        message="synthetic unusable fit",
    )

    with pytest.raises(ValueError, match="toxicity estimator is not usable"):
        _execute_as_pin_leg(
            FakeAdapter(),
            is_buy=True,
            size=0.0002,
            fit=unusable_fit,
            informed_buy=0.1,
            informed_sell=0.1,
            curve=MarkoutCurve((0.0, 1.0), (1.0,), (1.0,), (1,)),
            sigma=1.0,
            kappa=1.0,
            timeout_seconds=1.0,
            label="unusable-pin",
        )


def test_bootstrap_pin_signal_builds_buckets_and_markout_curve():
    events = []
    base_ts = 1_700_000_000_000
    for i in range(36):
        mid = 100.0 + 0.05 * i
        events.append(
            BookSnapshot(
                timestamp_ms=base_ts + i * 1_000,
                coin="BTC",
                bids=(BookLevel(mid - 0.5, 1.0),),
                asks=(BookLevel(mid + 0.5, 1.0),),
            )
        )
    for i in range(25):
        side = AggressorSide.BUY if i % 2 == 0 else AggressorSide.SELL
        events.append(
            TradePrint(
                timestamp_ms=base_ts + i * 1_000,
                coin="BTC",
                price=100.0 + 0.05 * i,
                size=0.01,
                aggressor=side,
                trade_id=str(i),
            )
        )

    fit, informed_buy, informed_sell, curve, sigma, buckets = bootstrap_pin_signal(
        sorted(events, key=lambda event: event.timestamp_ms),
        bucket_seconds=1,
    )

    assert buckets == 25
    assert 0.0 <= fit.pin <= 1.0
    assert 0.0 <= informed_buy <= 1.0
    assert 0.0 <= informed_sell <= 1.0
    assert sigma > 0
    assert len(curve.counts) == 1


def test_committed_testnet_artifact_matches_reported_roundtrips():
    artifact = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "execution_artifacts"
        / "trend_rotation_testnet_execution.json"
    )
    data = json.loads(artifact.read_text(encoding="utf-8"))

    assert data["coin"] == "BTC"
    assert data["size"] == pytest.approx(0.0002)
    assert data["pin_sample_buckets"] == 35
    assert data["naive_roundtrip_pnl_usd"] == pytest.approx(-0.021451)
    assert data["as_pin_roundtrip_pnl_usd"] == pytest.approx(-0.008158)
    assert data["as_pin_sell"]["maker_filled_size"] == pytest.approx(0.0002)
    assert data["as_pin_buy"]["fallback_filled_size"] == pytest.approx(0.0002)
    assert data["restored_initial_position"] is True
