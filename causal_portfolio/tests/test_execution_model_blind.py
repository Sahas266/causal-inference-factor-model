"""The execution layer must behave identically for any model.

The import boundary (test_execution_is_model_agnostic.py) proves execution
does not *import* model code. These tests prove the stronger runtime claim:
nothing in the layer assumes a particular asset universe, ticker convention,
rebalance cadence, or strategy name.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from causal_portfolio.execution import trace
from causal_portfolio.execution.types import AccountState, Position, TargetSnapshot


def _snapshot(weights, **kw):
    return TargetSnapshot(weights, as_of=datetime.now(timezone.utc), **kw)


def test_custom_asset_map_keeps_target_and_position_on_one_row(tmp_path):
    """A ticker outside DEFAULT_ASSET_MAP must still line up with its position.

    Regression: the trace stamped hl_coin from the module-level default map
    instead of the config map the planner used, so an operator-supplied
    mapping produced hl_coin=None. record_portfolio_snapshot then fell back to
    ticker.upper() ("KBONK"), which never matches the venue coin ("kBONK"),
    and the panel split one holding into a phantom target row plus an
    untargeted position row.
    """
    target = _snapshot({"bonk": 0.5}, strategy="any-model")
    trace.start_cycle(
        target,
        asset_map={"bonk": "kBONK"},          # venue casing the default lacks
        model_prices={"bonk": 0.00002},
        log_dir=tmp_path,
    )
    state = AccountState(
        address="0xABC", account_value_usd=1_000.0, margin_used_usd=10.0,
        positions={"kBONK": Position("kBONK", 1_000_000.0, 0.00002, 500.0)},
    )
    assert trace.record_portfolio_snapshot(
        state, {"kBONK": 0.000021}, log_dir=tmp_path
    )

    assets = trace.panel_data(tmp_path)["assets"]

    assert len(assets) == 1, f"expected one merged row, got {assets}"
    row = assets[0]
    assert row["coin"] == "kBONK"
    assert row["ticker"] == "bonk"
    assert row["target_weight"] == 0.5          # target present...
    assert row["position_usd"] == 500.0         # ...on the same row as the fill


def test_layer_accepts_any_ticker_universe_and_strategy_name(tmp_path):
    """No hardcoded universe: unknown tickers and any strategy string work."""
    target = _snapshot(
        {"zzz": 0.7, "qqq": -0.3}, strategy="some other model v9"
    )
    path = trace.start_cycle(
        target, asset_map={"zzz": "ZZZ", "qqq": "QQQ"}, log_dir=tmp_path
    )
    assert path is not None

    meta = trace.read_meta(path)
    assert meta["strategy"] == "some other model v9"
    assert trace.panel_data(tmp_path)["meta"]["target_id"] == target.target_id


def test_cadence_is_declared_by_the_model_not_assumed(tmp_path):
    """Execution stores whatever cadence the model declares, at any horizon."""
    for label, delta in [
        ("intraday", timedelta(minutes=5)),
        ("daily", timedelta(days=1)),
        ("monthly", timedelta(days=30)),
    ]:
        expected = datetime.now(timezone.utc) + delta
        # A distinct target id per case so each opens its own cycle.
        target = _snapshot({"btc": 0.1}, strategy=label)
        trace.start_cycle(
            target, expected_next_rebalance=expected, log_dir=tmp_path
        )
        stored = trace.expected_next_rebalance(tmp_path)
        assert stored is not None, f"{label}: cadence not stored"
        assert abs((stored - expected).total_seconds()) < 1, label


def test_cadence_may_also_arrive_as_target_metadata(tmp_path):
    """A model that only emits JSON can declare cadence without a code path."""
    expected = datetime.now(timezone.utc) + timedelta(hours=6)
    target = _snapshot(
        {"btc": 0.1},
        strategy="json-only model",
        metadata={"expected_next_rebalance": expected.isoformat()},
    )
    trace.start_cycle(target, log_dir=tmp_path)

    stored = trace.expected_next_rebalance(tmp_path)
    assert stored is not None
    assert abs((stored - expected).total_seconds()) < 1


def test_missing_cadence_degrades_instead_of_guessing(tmp_path):
    """No cadence declared must read as unknown, never a fabricated default."""
    trace.start_cycle(_snapshot({"btc": 0.1}, strategy="no-cadence"), log_dir=tmp_path)

    assert trace.expected_next_rebalance(tmp_path) is None

    from causal_portfolio.execution.notify import format_next_rebalance

    assert "unknown" in format_next_rebalance(None)
