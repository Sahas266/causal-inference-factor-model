from __future__ import annotations

import pytest

from causal_portfolio.market_making.avellaneda_stoikov import (
    ASModelParams,
    finite_horizon_quotes,
    solve_finite_horizon_hjb,
    stationary_reservation_prices,
)


def test_finite_horizon_matches_paper_formulas():
    params = ASModelParams(gamma=0.1, sigma=2.0, kappa=1.5, horizon=1.0)
    quote = finite_horizon_quotes(100.0, 2.0, params)
    assert quote.reservation_price == pytest.approx(99.2)
    assert quote.total_spread == pytest.approx(
        0.4 + 20.0 * __import__("math").log1p(0.1 / 1.5)
    )
    assert quote.bid < quote.reservation_price < quote.ask


def test_inventory_moves_quotes_toward_liquidation():
    params = ASModelParams(gamma=0.01, sigma=0.5, kappa=2.0, horizon=10.0)
    flat = finite_horizon_quotes(100.0, 0.0, params)
    long = finite_horizon_quotes(100.0, 3.0, params)
    short = finite_horizon_quotes(100.0, -3.0, params)
    assert long.reservation_price < flat.reservation_price < short.reservation_price
    assert long.total_spread == pytest.approx(short.total_spread)


def test_stationary_reservations_are_distinct_from_rolling_formula():
    bid, ask = stationary_reservation_prices(100.0, 0, 0.1, 1.0, 2.0)
    assert bid < 100.0 < ask
    with pytest.raises(ValueError, match="inventory bound"):
        stationary_reservation_prices(100.0, 10, 1.0, 1.0, 1.0)


def test_hjb_is_symmetric_and_inventory_aware():
    params = ASModelParams(
        gamma=0.1, sigma=0.5, kappa=1.5, horizon=1.0, arrival_rate=5.0
    )
    result = solve_finite_horizon_hjb(params, max_inventory=4, steps=500)
    flat = result.quotes(100.0, 0)
    long = result.quotes(100.0, 2)
    short = result.quotes(100.0, -2)
    assert flat.bid == pytest.approx(200.0 - flat.ask, abs=1e-8)
    assert long.reservation_price < 100.0
    assert short.reservation_price > 100.0
    assert long.total_spread == pytest.approx(short.total_spread)
    assert result.bid_distances[-1] is None
    assert result.ask_distances[0] is None


def test_as_rejects_non_positive_bid_domain():
    params = ASModelParams(gamma=1.0, sigma=10.0, kappa=1.0, horizon=10.0)
    with pytest.raises(ValueError, match="non-positive bid"):
        finite_horizon_quotes(10.0, 10.0, params)
