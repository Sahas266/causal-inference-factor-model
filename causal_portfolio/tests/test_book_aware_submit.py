"""Tests for HLAdapter.submit_orders_book_aware using a mocked SDK.

We patch the network-touching methods (fetch_l2_book, fetch_meta,
_submit_single_ioc) so we can drive fill / partial-fill / give-up scenarios
deterministically without hitting Hyperliquid.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("hyperliquid", reason="hyperliquid SDK not installed")

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.hyperliquid import HLAdapter
from causal_portfolio.execution.orderbook import L2Book, L2Level
from causal_portfolio.execution.types import Order


def _make_adapter(config: ExecutionConfig) -> HLAdapter:
    """Build an HLAdapter without running __init__ (no network/creds)."""
    a = HLAdapter.__new__(HLAdapter)
    a.config = config
    a.address = "0xtest"
    return a


def _order(coin="BTC", is_buy=True, size=0.01, px=100.0, reduce_only=False):
    return Order(coin=coin, is_buy=is_buy, size=size, limit_px=px,
                 cloid="0x" + "a" * 32, reduce_only=reduce_only)


def _filled(sz, px):
    return {"status": "ok", "response": {"data": {
        "statuses": [{"filled": {"totalSz": str(sz), "avgPx": str(px)}}]}}}


def _err(msg):
    return {"status": "ok", "response": {"data": {"statuses": [{"error": msg}]}}}


def _cfg(**kw):
    base = dict(dry_run=True, smart_poll_seconds=0.0, smart_max_attempts=5)
    base.update(kw)
    return ExecutionConfig(**base)


def test_book_aware_single_full_fill():
    cfg = _cfg()
    a = _make_adapter(cfg)
    a.fetch_meta = MagicMock(return_value={
        "BTC": __import__("causal_portfolio.execution.types", fromlist=["AssetMeta"]).AssetMeta(
            coin="BTC", sz_decimals=5, max_leverage=10, min_size=1e-5)})
    a.fetch_l2_book = MagicMock(return_value=L2Book(
        "BTC", [L2Level(99.0, 10.0)], [L2Level(101.0, 10.0)]))
    a._submit_single_ioc = MagicMock(return_value=_filled(0.01, 101.0))

    resp = a.submit_orders_book_aware([_order(size=0.01)])
    summary = resp["response"]["data"]["summaries"][0]
    assert summary["filled"] == pytest.approx(0.01)
    assert summary["error"] is None
    # One IOC was enough
    assert a._submit_single_ioc.call_count == 1


def test_book_aware_accumulates_partial_fills():
    """Two partial fills should sum to the requested size."""
    cfg = _cfg()
    a = _make_adapter(cfg)
    a.fetch_meta = MagicMock(return_value={
        "BTC": __import__("causal_portfolio.execution.types", fromlist=["AssetMeta"]).AssetMeta(
            coin="BTC", sz_decimals=5, max_leverage=10, min_size=1e-5)})
    a.fetch_l2_book = MagicMock(return_value=L2Book(
        "BTC", [L2Level(99.0, 10.0)], [L2Level(101.0, 10.0)]))
    # First IOC fills half, second fills the rest
    a._submit_single_ioc = MagicMock(side_effect=[
        _filled(0.005, 101.0),
        _filled(0.005, 101.2),
    ])

    resp = a.submit_orders_book_aware([_order(size=0.01)])
    summary = resp["response"]["data"]["summaries"][0]
    assert summary["filled"] == pytest.approx(0.01, abs=1e-6)
    assert len(summary["fills"]) == 2
    assert summary["error"] is None


def test_book_aware_gives_up_when_no_inband_price():
    """If best bid is always outside the band, we exhaust attempts and report."""
    cfg = _cfg(smart_max_attempts=3)
    a = _make_adapter(cfg)
    a.fetch_meta = MagicMock(return_value={
        "BTC": __import__("causal_portfolio.execution.types", fromlist=["AssetMeta"]).AssetMeta(
            coin="BTC", sz_decimals=5, max_leverage=10, min_size=1e-5)})
    # SELL with best bid far below band → marketable_price returns None
    a.fetch_l2_book = MagicMock(return_value=L2Book(
        "BTC", [L2Level(50.0, 10.0)], [L2Level(101.0, 10.0)]))
    a._submit_single_ioc = MagicMock()

    resp = a.submit_orders_book_aware([_order(is_buy=False, size=0.01)])
    summary = resp["response"]["data"]["summaries"][0]
    assert summary["filled"] == 0.0
    assert summary["remaining"] == pytest.approx(0.01)
    assert summary["error"] is not None
    # Never submitted because no valid price existed
    a._submit_single_ioc.assert_not_called()


def test_book_aware_retries_after_oracle_reject_then_fills():
    """First attempt rejected by oracle, book moves, second attempt fills."""
    cfg = _cfg(smart_max_attempts=4)
    a = _make_adapter(cfg)
    a.fetch_meta = MagicMock(return_value={
        "BTC": __import__("causal_portfolio.execution.types", fromlist=["AssetMeta"]).AssetMeta(
            coin="BTC", sz_decimals=5, max_leverage=10, min_size=1e-5)})
    a.fetch_l2_book = MagicMock(return_value=L2Book(
        "BTC", [L2Level(99.0, 10.0)], [L2Level(101.0, 10.0)]))
    a._submit_single_ioc = MagicMock(side_effect=[
        _err("Price too far from oracle asset=3"),
        _filled(0.01, 100.9),
    ])

    resp = a.submit_orders_book_aware([_order(size=0.01)])
    summary = resp["response"]["data"]["summaries"][0]
    assert summary["filled"] == pytest.approx(0.01)
    assert summary["error"] is None
    assert a._submit_single_ioc.call_count == 2


def test_smart_config_validation():
    with pytest.raises(ValueError, match="smart_max_attempts"):
        ExecutionConfig(smart_max_attempts=0)
    with pytest.raises(ValueError, match="smart_max_band_bps"):
        ExecutionConfig(smart_max_band_bps=0)
    with pytest.raises(ValueError, match="smart_poll_seconds"):
        ExecutionConfig(smart_poll_seconds=-1)
