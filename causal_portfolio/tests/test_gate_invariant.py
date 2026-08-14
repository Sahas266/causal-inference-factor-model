"""No execution gate may block a reduce-only order.

Refusing an exit keeps risk the model decided to shed. That hazard is why the
cost gate exempts reduce-only orders — and the completeness gate reintroduced
it once by re-deriving the rule for itself instead of inheriting it.

The per-gate tests elsewhere cover the two gates that exist today. These cover
`_apply_gate`, the choke point every gate routes through, so the invariant has
coverage for a gate nobody has written yet — which is precisely the one that
would otherwise get it wrong.
"""

from __future__ import annotations

import inspect

from causal_portfolio.execution import hyperliquid
from causal_portfolio.execution.hyperliquid import _apply_gate
from causal_portfolio.execution.types import Order, SkipReason

BLOCK = (SkipReason.EXCEEDS_TRANSACTION_COST, "blocked by a hostile gate")


def _order(coin: str, *, reduce_only: bool) -> Order:
    return Order(coin, True, 1.0, 100.0, "0x" + "1" * 32, reduce_only=reduce_only)


def test_a_gate_that_blocks_everything_still_lets_exits_through():
    """The worst-case gate: refuses every order it is allowed to judge."""
    orders = [
        _order("BTC", reduce_only=False),
        _order("ETH", reduce_only=True),
        _order("SOL", reduce_only=False),
        _order("DOGE", reduce_only=True),
    ]

    allowed, skipped = _apply_gate(orders, lambda _order: BLOCK)

    assert [o.coin for o in allowed] == ["ETH", "DOGE"]
    assert [c for c, _, _ in skipped] == ["BTC", "SOL"]


def test_verdict_is_never_called_for_a_reduce_only_order():
    """Structural, not behavioral: a gate cannot even inspect an exit.

    This is what makes the invariant inheritable — a future gate author has no
    reduce-only order to forget to filter.
    """
    seen: list[str] = []

    def verdict(order):
        seen.append(order.coin)
        return BLOCK

    _apply_gate(
        [_order("BTC", reduce_only=False), _order("ETH", reduce_only=True)],
        verdict,
    )

    assert seen == ["BTC"], "gate was handed a reduce-only order"


def test_an_all_reduce_only_plan_is_never_gated():
    orders = [_order("ETH", reduce_only=True), _order("SOL", reduce_only=True)]

    allowed, skipped = _apply_gate(orders, lambda _order: BLOCK)

    assert allowed == orders
    assert skipped == []


def test_a_permissive_gate_passes_everything_through_unchanged():
    orders = [_order("BTC", reduce_only=False), _order("ETH", reduce_only=True)]

    allowed, skipped = _apply_gate(orders, lambda _order: None)

    assert allowed == orders
    assert skipped == []


def test_skip_entries_carry_coin_reason_and_detail():
    """plan.skipped shape, so trace/audit/Telegram render dropped legs."""
    allowed, skipped = _apply_gate([_order("BTC", reduce_only=False)], lambda _o: BLOCK)

    assert allowed == []
    assert skipped == [("BTC", SkipReason.EXCEEDS_TRANSACTION_COST,
                        "blocked by a hostile gate")]


def test_order_is_preserved():
    """Submission order must match the plan's, for deterministic cloids."""
    orders = [_order(c, reduce_only=False) for c in ("SOL", "BTC", "ETH")]

    allowed, _ = _apply_gate(orders, lambda _order: None)

    assert [o.coin for o in allowed] == ["SOL", "BTC", "ETH"]


def test_every_gate_routes_through_the_applicator():
    """Guard against a future gate hand-rolling its own partition.

    A gate that filters `reduce_only` itself is the failure mode this whole
    file exists to prevent, so assert the choke point is actually used rather
    than merely available.
    """
    for gate in ("_check_cost_gate", "_check_completeness_gate"):
        source = inspect.getsource(getattr(hyperliquid, gate))
        assert "_apply_gate(" in source, f"{gate} does not route through _apply_gate"
