from __future__ import annotations

from causal_portfolio.market_making.recorder import (
    MicrostructureStore,
    aggregate_daily_trade_counts,
    capture_day_quality,
    normalize_hl_message,
)
from causal_portfolio.market_making.types import AggressorSide, BookSnapshot, TradePrint


def _book_message():
    return {
        "channel": "l2Book",
        "data": {
            "coin": "BTC",
            "time": 1_000,
            "levels": [
                [{"px": "99", "sz": "2", "n": 1}],
                [{"px": "101", "sz": "3", "n": 1}],
            ],
        },
    }


def _trade_message():
    return {
        "channel": "trades",
        "data": [
            {"coin": "BTC", "side": "B", "px": "101", "sz": "0.2", "time": 2_000, "tid": 7},
            {"coin": "BTC", "side": "A", "px": "99", "sz": "0.1", "time": 2_001, "tid": 8},
        ],
    }


def test_normalizes_hyperliquid_messages_and_side_semantics():
    book = normalize_hl_message(_book_message(), 1_005)[0]
    assert isinstance(book, BookSnapshot)
    assert book.mid == 100
    trades = normalize_hl_message(_trade_message(), 2_005)
    assert [trade.aggressor for trade in trades] == [AggressorSide.BUY, AggressorSide.SELL]


def test_duckdb_store_deduplicates_and_replays(tmp_path):
    store = MicrostructureStore(tmp_path / "micro.duckdb")
    assert store.append(_book_message(), received_ms=1_005)
    assert not store.append(_book_message(), received_ms=1_006)
    assert store.append(_trade_message(), received_ms=2_005)
    replay = store.replay_events(coin="BTC")
    assert len(replay) == 3
    assert isinstance(replay[0], BookSnapshot)
    assert isinstance(replay[1], TradePrint)
    assert sum(row["events"] for row in store.coverage()) == 3
    counts = aggregate_daily_trade_counts(x for x in replay if isinstance(x, TradePrint))
    assert list(counts.values()) == [(1, 1)]
    store.close()


def test_capture_quality_requires_boundary_and_gap_coverage():
    day_ms = 86_400_000
    events = [
        BookSnapshot(1_000, "BTC", (), ()),
        BookSnapshot(day_ms - 1_000, "BTC", (), ()),
        BookSnapshot(day_ms + 12 * 3_600_000, "BTC", (), ()),
    ]
    quality = capture_day_quality(
        events, boundary_tolerance_ms=2_000, max_gap_ms=day_ms
    )
    assert quality["1970-01-01"]["complete"]
    assert not quality["1970-01-02"]["complete"]
