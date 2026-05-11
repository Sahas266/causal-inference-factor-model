"""Tests for the post-execution reconciliation function."""

from __future__ import annotations

import pytest

from causal_portfolio.execution.reconcile import format_drift_summary, reconcile
from causal_portfolio.execution.types import (
    AccountState,
    Position,
    RebalancePlan,
)


def _plan(target_usd: dict[str, float], equity: float = 10_000.0) -> RebalancePlan:
    """Minimal RebalancePlan with the only field reconcile() inspects."""
    state = AccountState(address="0xtest", account_value_usd=equity,
                         margin_used_usd=0.0)
    return RebalancePlan(
        timestamp_ms=1000, target_weights={}, current_state=state,
        target_usd=target_usd, deltas_usd={}, orders=[], skipped=[],
        equity_used=equity * 1.0,
    )


def _state(positions: dict[str, tuple[float, float]], equity: float = 10_000.0) -> AccountState:
    """Build AccountState from {coin: (size, notional_usd)}."""
    pos = {c: Position(coin=c, size=sz, entry_px=abs(n) / max(abs(sz), 1e-9),
                       notional_usd=n) for c, (sz, n) in positions.items()}
    return AccountState(address="0xtest", account_value_usd=equity,
                        margin_used_usd=0.0, positions=pos)


def test_no_drift_when_on_target():
    plan = _plan({"BTC": 3000.0, "ETH": 2000.0})
    state = _state({"BTC": (0.05, 3000.0), "ETH": (0.667, 2000.0)})
    assert reconcile(plan, state) == []


def test_drift_within_tolerance_ignored():
    """$50 drift on a $3000 target = 1.67% — under both 5% and $50 caps."""
    plan = _plan({"BTC": 3000.0})
    state = _state({"BTC": (0.05, 3049.0)})  # $49 drift, under $50 abs cap
    assert reconcile(plan, state) == []


def test_drift_breaches_both_thresholds():
    """$200 drift on $3000 target = 6.7% — over BOTH 5% and $50."""
    plan = _plan({"BTC": 3000.0})
    state = _state({"BTC": (0.045, 2800.0)})
    drifts = reconcile(plan, state)
    assert len(drifts) == 1
    assert drifts[0].coin == "BTC"
    assert drifts[0].drift_usd == pytest.approx(-200.0)
    assert drifts[0].drift_pct == pytest.approx(-200.0 / 3000.0, rel=1e-9)


def test_drift_breaches_pct_not_abs_ignored():
    """10% drift on $300 target = $30 drift — under $50 abs floor."""
    plan = _plan({"BTC": 300.0})
    state = _state({"BTC": (0.005, 270.0)})  # $30 drift
    assert reconcile(plan, state) == []


def test_drift_breaches_abs_not_pct_ignored():
    """$200 drift on $100k target = 0.2% — under 5% pct floor."""
    plan = _plan({"BTC": 100_000.0})
    state = _state({"BTC": (1.667, 99_800.0)})  # $200 drift
    assert reconcile(plan, state) == []


def test_target_zero_with_residual_position_flagged():
    """Plan wanted SOL closed (target=0) but $100 still in the position."""
    plan = _plan({"SOL": 0.0})
    state = _state({"SOL": (1.0, 100.0)})  # left-over long
    drifts = reconcile(plan, state)
    assert len(drifts) == 1
    assert drifts[0].coin == "SOL"
    assert drifts[0].drift_pct == float("inf")


def test_target_zero_with_residual_under_tolerance_ignored():
    plan = _plan({"SOL": 0.0})
    state = _state({"SOL": (0.1, 15.0)})  # $15 dust — under $50 abs floor
    assert reconcile(plan, state) == []


def test_coin_in_state_but_not_in_plan_ignored():
    """We only judge what we planned. Unrelated holdings don't trigger drift."""
    plan = _plan({"BTC": 3000.0})
    state = _state({"BTC": (0.05, 3000.0), "ETH": (1.0, 3000.0)})  # ETH unplanned
    assert reconcile(plan, state) == []


def test_long_short_flip_drift():
    """Plan said go short $5000; actual ended up flat (cancel mid-flight)."""
    plan = _plan({"BTC": -5000.0})
    state = _state({})  # no position
    drifts = reconcile(plan, state)
    assert len(drifts) == 1
    # actual=0, target=-5000, drift = 0 - (-5000) = +5000
    assert drifts[0].drift_usd == pytest.approx(5000.0)


def test_custom_tolerance():
    plan = _plan({"BTC": 3000.0})
    state = _state({"BTC": (0.0495, 2970.0)})  # 1% drift, $30
    # Tight thresholds should flag this
    drifts = reconcile(plan, state, tolerance_pct=0.005, tolerance_usd=10.0)
    assert len(drifts) == 1


def test_format_drift_summary_empty():
    assert format_drift_summary([]).startswith("Reconciliation OK")


def test_format_drift_summary_renders():
    plan = _plan({"BTC": 3000.0})
    state = _state({"BTC": (0.045, 2700.0)})
    msg = format_drift_summary(reconcile(plan, state))
    assert "BTC" in msg
    assert "drift" in msg.lower()
