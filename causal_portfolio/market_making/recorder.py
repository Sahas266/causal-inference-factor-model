"""Hyperliquid WebSocket capture, validation, and deterministic replay parsing."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from causal_portfolio.execution.hyperliquid import MAINNET_URL, TESTNET_URL
from causal_portfolio.market_making.types import (
    AggressorSide,
    BookLevel,
    BookSnapshot,
    MarketEvent,
    TradePrint,
)


def _canonical_hash(channel: str, payload: Any) -> str:
    raw = json.dumps([channel, payload], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def normalize_hl_message(
    message: dict[str, Any], received_ms: int | None = None
) -> list[MarketEvent]:
    """Convert public HL messages into deterministic replay events."""
    channel = message.get("channel")
    data = message.get("data")
    received = int(time.time() * 1_000) if received_ms is None else int(received_ms)
    if channel == "l2Book" and isinstance(data, dict):
        levels = data.get("levels", [[], []])
        return [
            BookSnapshot(
                timestamp_ms=int(data.get("time", received)),
                received_ms=received,
                coin=str(data["coin"]),
                bids=tuple(BookLevel(float(x["px"]), float(x["sz"])) for x in levels[0]),
                asks=tuple(BookLevel(float(x["px"]), float(x["sz"])) for x in levels[1]),
            )
        ]
    if channel == "trades" and isinstance(data, list):
        return [
            TradePrint(
                timestamp_ms=int(trade.get("time", received)),
                coin=str(trade["coin"]),
                price=float(trade["px"]),
                size=float(trade["sz"]),
                aggressor=AggressorSide.from_hyperliquid(str(trade["side"])),
                trade_id=str(trade.get("tid") or trade.get("hash") or "") or None,
            )
            for trade in data
        ]
    return []


class MicrostructureStore:
    """Thread-safe append-only DuckDB raw-event store with deduplication."""

    def __init__(self, path: str | Path):
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError("MicrostructureStore requires duckdb") from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(self.path))
        self._lock = threading.Lock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_events (
                received_ms BIGINT NOT NULL,
                exchange_ms BIGINT,
                channel VARCHAR NOT NULL,
                coin VARCHAR,
                event_hash VARCHAR PRIMARY KEY,
                payload_json VARCHAR NOT NULL
            )
            """
        )

    @staticmethod
    def _identity(message: dict[str, Any], received_ms: int) -> tuple[int | None, str | None]:
        channel = message.get("channel")
        data = message.get("data")
        if channel in {"l2Book", "bbo"} and isinstance(data, dict):
            return int(data.get("time", received_ms)), str(data.get("coin") or "") or None
        if channel == "trades" and isinstance(data, list) and data:
            first_ms = min(int(x.get("time", received_ms)) for x in data)
            coin = str(data[0].get("coin") or "") or None
            return first_ms, coin
        if isinstance(data, dict):
            return int(data.get("time", received_ms)), str(data.get("coin") or "") or None
        return None, None

    def append(self, message: dict[str, Any], *, received_ms: int | None = None) -> bool:
        received = int(time.time() * 1_000) if received_ms is None else int(received_ms)
        trade_data = message.get("data")
        if message.get("channel") == "trades" and isinstance(trade_data, list):
            if not trade_data:
                return False
        if (
            message.get("channel") == "trades"
            and isinstance(trade_data, list)
            and len(trade_data) > 1
        ):
            inserted = False
            for trade in trade_data:
                inserted = self.append(
                    {"channel": "trades", "data": [trade]}, received_ms=received
                ) or inserted
            return inserted
        channel = str(message.get("channel", "unknown"))
        exchange_ms, coin = self._identity(message, received)
        payload = json.dumps(message, sort_keys=True, separators=(",", ":"))
        event_hash = _canonical_hash(channel, message.get("data"))
        with self._lock:
            before = int(
                self._connection.execute("SELECT count(*) FROM market_events").fetchone()[0]
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO market_events VALUES (?, ?, ?, ?, ?, ?)",
                [received, exchange_ms, channel, coin, event_hash, payload],
            )
            after = int(
                self._connection.execute("SELECT count(*) FROM market_events").fetchone()[0]
            )
        return after > before

    def coverage(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT coin, channel, count(*) AS events,
                   min(exchange_ms) AS first_ms, max(exchange_ms) AS last_ms,
                   max(received_ms - exchange_ms) AS max_receive_lag_ms
            FROM market_events
            GROUP BY coin, channel ORDER BY coin, channel
            """
        ).fetchall()
        return [
            {
                "coin": row[0],
                "channel": row[1],
                "events": row[2],
                "first_ms": row[3],
                "last_ms": row[4],
                "max_receive_lag_ms": row[5],
            }
            for row in rows
        ]

    def replay_events(
        self,
        *,
        coin: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[MarketEvent]:
        conditions = ["coin = ?", "channel IN ('l2Book', 'trades')"]
        parameters: list[Any] = [coin]
        if start_ms is not None:
            conditions.append("exchange_ms >= ?")
            parameters.append(start_ms)
        if end_ms is not None:
            conditions.append("exchange_ms <= ?")
            parameters.append(end_ms)
        rows = self._connection.execute(
            f"""
            SELECT received_ms, payload_json FROM market_events
            WHERE {' AND '.join(conditions)}
            ORDER BY exchange_ms, received_ms, event_hash
            """,
            parameters,
        ).fetchall()
        output: list[MarketEvent] = []
        for received_ms, payload in rows:
            output.extend(normalize_hl_message(json.loads(payload), int(received_ms)))
        return sorted(output, key=lambda event: event.timestamp_ms)

    def close(self) -> None:
        self._connection.close()


@dataclass
class HLStreamRecorder:
    coins: tuple[str, ...]
    store: MicrostructureStore
    testnet: bool = False

    def __post_init__(self) -> None:
        if not self.coins:
            raise ValueError("at least one coin is required")
        from hyperliquid.info import Info

        self.info = Info(TESTNET_URL if self.testnet else MAINNET_URL, skip_ws=False)
        self._subscriptions: list[tuple[dict[str, str], int]] = []

    def _callback(self, message: dict[str, Any]) -> None:
        self.store.append(message)

    def start(self) -> None:
        for coin in self.coins:
            for channel in ("trades", "l2Book", "bbo"):
                subscription = {"type": channel, "coin": coin}
                subscription_id = self.info.subscribe(subscription, self._callback)
                self._subscriptions.append((subscription, subscription_id))

    def stop(self) -> None:
        for subscription, subscription_id in reversed(self._subscriptions):
            try:
                self.info.unsubscribe(subscription, subscription_id)
            except Exception:
                pass
        self._subscriptions.clear()
        self.info.disconnect_websocket()


def aggregate_daily_trade_counts(events: Iterable[TradePrint]) -> dict[str, tuple[int, int]]:
    """Aggregate aggressor counts into UTC dates for PIN estimation."""
    from datetime import datetime, timezone

    counts: dict[str, list[int]] = {}
    for event in events:
        day = datetime.fromtimestamp(event.timestamp_ms / 1_000, tz=timezone.utc).date().isoformat()
        bucket = counts.setdefault(day, [0, 0])
        bucket[0 if event.aggressor == AggressorSide.BUY else 1] += 1
    return {day: (value[0], value[1]) for day, value in sorted(counts.items())}


def capture_day_quality(
    events: Iterable[MarketEvent],
    *,
    boundary_tolerance_ms: int = 5 * 60 * 1_000,
    max_gap_ms: int = 60 * 1_000,
) -> dict[str, dict[str, int | bool | None]]:
    """Validate full UTC-day L2 coverage before daily PIN calibration."""
    from datetime import datetime, timedelta, timezone

    by_day: dict[str, list[int]] = {}
    for event in events:
        if not isinstance(event, BookSnapshot):
            continue
        day = datetime.fromtimestamp(
            event.timestamp_ms / 1_000, tz=timezone.utc
        ).date().isoformat()
        by_day.setdefault(day, []).append(event.timestamp_ms)
    result: dict[str, dict[str, int | bool | None]] = {}
    for day, timestamps in sorted(by_day.items()):
        ordered = sorted(set(timestamps))
        start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        start_ms = int(start.timestamp() * 1_000)
        end_ms = int((start + timedelta(days=1)).timestamp() * 1_000)
        gaps = [right - left for left, right in zip(ordered, ordered[1:])]
        largest_gap = max(gaps) if gaps else None
        complete = (
            ordered[0] <= start_ms + boundary_tolerance_ms
            and ordered[-1] >= end_ms - boundary_tolerance_ms
            and largest_gap is not None
            and largest_gap <= max_gap_ms
        )
        result[day] = {
            "complete": complete,
            "snapshots": len(ordered),
            "first_ms": ordered[0],
            "last_ms": ordered[-1],
            "max_gap_ms": largest_gap,
        }
    return result
