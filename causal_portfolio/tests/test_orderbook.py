"""Tests for the pure order-book-aware pricing helpers."""

from __future__ import annotations

import pytest

from causal_portfolio.execution.orderbook import (
    FillResult,
    L2Book,
    L2Level,
    marketable_price,
    parse_fill_response,
)


def _book(bids, asks, coin="BTC") -> L2Book:
    return L2Book(
        coin=coin,
        bids=[L2Level(px, sz) for px, sz in bids],
        asks=[L2Level(px, sz) for px, sz in asks],
    )


# ── L2Book accessors ────────────────────────────────────────────────


def test_book_best_and_mid():
    b = _book([(100.0, 1.0), (99.0, 2.0)], [(101.0, 1.0), (102.0, 2.0)])
    assert b.best_bid == 100.0
    assert b.best_ask == 101.0
    assert b.mid == 100.5


def test_book_empty_side_mid_none():
    assert _book([], [(101.0, 1.0)]).mid is None
    assert _book([(100.0, 1.0)], []).mid is None


# ── marketable_price: BUY ───────────────────────────────────────────


def test_buy_prices_at_best_ask_when_size_fits_top():
    b = _book([(100.0, 5.0)], [(101.0, 5.0), (102.0, 5.0)])
    px = marketable_price(b, is_buy=True, size=1.0)
    # 1.0 fits in the top ask level → sweep price is best ask
    assert px == 101.0


def test_buy_walks_deeper_for_large_size():
    b = _book([(100.0, 5.0)], [(101.0, 1.0), (102.0, 1.0), (103.0, 5.0)])
    # Need 2.5 units → walk to the 103 level. Wide band so the clamp
    # doesn't interfere (tested separately).
    px = marketable_price(b, is_buy=True, size=2.5, max_band_bps=1000.0)
    assert px == 103.0


def test_buy_clamped_to_band():
    b = _book([(100.0, 5.0)], [(101.0, 0.1), (200.0, 5.0)])
    # Need 2 units → sweep would be 200, but band caps it
    px = marketable_price(b, is_buy=True, size=2.0, max_band_bps=200.0)
    # mid = 100.5, band cap = 100.5 * 1.02 = 102.51
    assert px == pytest.approx(102.51)


def test_buy_returns_none_when_best_ask_outside_band():
    # best ask far above band → cannot cross in-band
    b = _book([(100.0, 5.0)], [(150.0, 5.0)])
    # mid = 125, band cap (200bps) = 127.5 < best ask 150 → None
    px = marketable_price(b, is_buy=True, size=1.0, max_band_bps=200.0)
    assert px is None


# ── marketable_price: SELL ──────────────────────────────────────────


def test_sell_prices_at_best_bid_when_size_fits():
    b = _book([(100.0, 5.0), (99.0, 5.0)], [(101.0, 5.0)])
    px = marketable_price(b, is_buy=False, size=1.0)
    assert px == 100.0


def test_sell_walks_deeper_for_large_size():
    b = _book([(100.0, 1.0), (99.0, 1.0), (98.0, 5.0)], [(101.0, 5.0)])
    px = marketable_price(b, is_buy=False, size=2.5, max_band_bps=1000.0)
    assert px == 98.0


def test_sell_clamped_to_band_floor():
    b = _book([(100.0, 0.1), (50.0, 5.0)], [(101.0, 5.0)])
    # Need 2 units → sweep would be 50, but band floor clamps
    px = marketable_price(b, is_buy=False, size=2.0, max_band_bps=200.0)
    # mid = 100.5, band floor = 100.5 * 0.98 = 98.49
    assert px == pytest.approx(98.49)


def test_sell_returns_none_when_best_bid_outside_band():
    # This is the testnet pathology we hit: best bid below the oracle band.
    b = _book([(90.0, 5.0)], [(101.0, 5.0)])
    # mid = 95.5, band floor (200bps) = 93.59 > best bid 90 → None
    px = marketable_price(b, is_buy=False, size=1.0, max_band_bps=200.0)
    assert px is None


# ── edge cases ──────────────────────────────────────────────────────


def test_zero_size_returns_none():
    b = _book([(100.0, 5.0)], [(101.0, 5.0)])
    assert marketable_price(b, is_buy=True, size=0.0) is None


def test_empty_book_returns_none():
    assert marketable_price(_book([], []), is_buy=True, size=1.0) is None


def test_reference_mid_override():
    b = _book([(100.0, 5.0)], [(101.0, 5.0)])
    # Force a reference mid far from book mid; band computed off reference
    px = marketable_price(b, is_buy=True, size=1.0, reference_mid=101.0,
                          max_band_bps=50.0)
    # best ask 101 within band of reference 101 → price at best ask
    assert px == 101.0


# ── parse_fill_response ─────────────────────────────────────────────


def test_parse_filled():
    resp = {"status": "ok", "response": {"type": "order", "data": {
        "statuses": [{"filled": {"totalSz": "0.00278", "avgPx": "75400.0", "oid": 1}}]}}}
    fr = parse_fill_response(resp)
    assert fr.filled_size == pytest.approx(0.00278)
    assert fr.avg_px == pytest.approx(75400.0)
    assert fr.error is None


def test_parse_oracle_error():
    resp = {"status": "ok", "response": {"data": {
        "statuses": [{"error": "Price too far from oracle asset=3"}]}}}
    fr = parse_fill_response(resp)
    assert fr.filled_size == 0.0
    assert fr.is_oracle_reject
    assert not fr.is_no_match


def test_parse_no_match_error():
    resp = {"status": "ok", "response": {"data": {
        "statuses": [{"error": "Order could not immediately match against any resting orders. asset=3"}]}}}
    fr = parse_fill_response(resp)
    assert fr.is_no_match
    assert not fr.is_oracle_reject


def test_parse_resting_zero_fill():
    resp = {"status": "ok", "response": {"data": {
        "statuses": [{"resting": {"oid": 123}}]}}}
    fr = parse_fill_response(resp)
    assert fr.filled_size == 0.0
    assert fr.error is None


def test_parse_unparseable():
    fr = parse_fill_response({"garbage": True})
    assert fr.filled_size == 0.0
    assert fr.error is not None
