"""Tests for the cancel-vs-submit race detection.

Pure test — no SDK required since we test the detection function directly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hyperliquid", reason="hyperliquid SDK not installed")
from causal_portfolio.execution.hyperliquid import _detect_cancel_race

from causal_portfolio.execution.types import (
    AccountState,
    Position,
    RebalancePlan,
)


def _state(positions: dict[str, tuple[float, float]]) -> AccountState:
    pos = {c: Position(coin=c, size=sz, entry_px=abs(n) / max(abs(sz), 1e-9),
                       notional_usd=n) for c, (sz, n) in positions.items()}
    return AccountState(address="0xtest", account_value_usd=10_000.0,
                        margin_used_usd=0.0, positions=pos)


def _plan(pre_positions, target_usd):
    return RebalancePlan(
        timestamp_ms=1000, target_weights={}, current_state=_state(pre_positions),
        target_usd=target_usd, deltas_usd={}, orders=[], skipped=[],
        equity_used=10_000.0,
    )


def test_no_race_when_state_unchanged():
    plan = _plan({"BTC": (0.05, 3000.0)}, target_usd={"BTC": 5000.0})
    mid = _state({"BTC": (0.05, 3000.0)})
    assert _detect_cancel_race(plan, mid) is False


def test_race_detected_when_position_grew():
    """Stale buy order filled between cancel and submit → position grew."""
    plan = _plan({"BTC": (0.05, 3000.0)}, target_usd={"BTC": 5000.0})
    mid = _state({"BTC": (0.06, 3600.0)})  # grew by $600 — material
    assert _detect_cancel_race(plan, mid) is True


def test_race_detected_when_position_appeared():
    """No position pre-cancel; suddenly one exists → race."""
    plan = _plan({}, target_usd={"ETH": 2000.0})
    mid = _state({"ETH": (1.0, 3000.0)})
    assert _detect_cancel_race(plan, mid) is True


def test_race_detected_when_position_closed():
    """Stop-loss fired during the race window → position now zero."""
    plan = _plan({"SOL": (10.0, 1500.0)}, target_usd={"SOL": 1500.0})
    mid = _state({})
    assert _detect_cancel_race(plan, mid) is True


def test_drift_under_tolerance_not_a_race():
    """Tiny mark-to-market wiggle shouldn't count as a race."""
    plan = _plan({"BTC": (0.05, 3000.0)}, target_usd={"BTC": 3000.0})
    mid = _state({"BTC": (0.05, 3010.0)})  # $10 wiggle, under $25 tolerance
    assert _detect_cancel_race(plan, mid) is False


def test_race_only_checks_planned_coins():
    """Movement on a coin we didn't plan to touch shouldn't trip the race."""
    plan = _plan({"BTC": (0.05, 3000.0)}, target_usd={"BTC": 3000.0})
    # SOL moved a lot but isn't in target_usd — shouldn't matter
    mid = _state({"BTC": (0.05, 3000.0), "SOL": (100.0, 10_000.0)})
    assert _detect_cancel_race(plan, mid) is False


def test_race_on_one_of_many_coins():
    plan = _plan(
        {"BTC": (0.05, 3000.0), "ETH": (1.0, 3000.0)},
        target_usd={"BTC": 5000.0, "ETH": 5000.0},
    )
    mid = _state({"BTC": (0.05, 3000.0), "ETH": (1.5, 4500.0)})  # ETH drifted
    assert _detect_cancel_race(plan, mid) is True


def test_custom_tolerance():
    plan = _plan({"BTC": (0.05, 3000.0)}, target_usd={"BTC": 3000.0})
    mid = _state({"BTC": (0.05, 3010.0)})  # $10 drift
    assert _detect_cancel_race(plan, mid, tolerance_usd=5.0) is True
    assert _detect_cancel_race(plan, mid, tolerance_usd=100.0) is False
