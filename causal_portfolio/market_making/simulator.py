"""Deterministic event-driven maker replay with conservative queue fills."""

from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass
from typing import Callable, Iterable

from causal_portfolio.market_making.calibration import adverse_markout_bps
from causal_portfolio.market_making.types import (
    AggressorSide,
    BookSnapshot,
    FundingEvent,
    MakerFill,
    MarketEvent,
    MMInventory,
    Quote,
    QuotePair,
    TradePrint,
)


@dataclass(frozen=True)
class SimulationConfig:
    latency_ms: int = 50
    maker_fee_bps: float = 0.0
    initial_cash_usd: float = 0.0
    initial_inventory_base: float = 0.0
    queue_ahead_fraction: float = 1.0
    markout_horizons_ms: tuple[int, ...] = (1_000, 5_000, 30_000)

    def __post_init__(self) -> None:
        if self.latency_ms < 0 or not math.isfinite(self.maker_fee_bps):
            raise ValueError("latency/fee settings are invalid")
        if not 0 <= self.queue_ahead_fraction <= 1:
            raise ValueError("queue_ahead_fraction must be in [0,1]")
        if any(h <= 0 for h in self.markout_horizons_ms):
            raise ValueError("markout horizons must be positive")


@dataclass(frozen=True)
class SimulationContext:
    timestamp_ms: int
    book: BookSnapshot
    inventory: MMInventory
    active_quotes: QuotePair | None


@dataclass
class _RestingQuote:
    quote: Quote
    queue_ahead: float


@dataclass(frozen=True)
class SimulationResult:
    fills: tuple[MakerFill, ...]
    final_inventory: float
    final_cash_usd: float
    final_equity_usd: float
    pnl_usd: float
    fees_usd: float
    funding_usd: float
    spread_capture_usd: float
    inventory_pnl_usd: float
    fill_rate: float
    quote_uptime: float
    cancel_rate_per_minute: float
    max_drawdown_usd: float
    max_abs_inventory: float
    quote_replacements: int
    post_only_rejections: int
    markout_bps: dict[int, float]


QuoteFunction = Callable[[SimulationContext], QuotePair | None]


class EventDrivenSimulator:
    def __init__(self, config: SimulationConfig = SimulationConfig()):
        self.config = config

    @staticmethod
    def _same_quotes(left: QuotePair | None, right: QuotePair | None) -> bool:
        if left is None or right is None:
            return left is right
        for a, b in ((left.bid, right.bid), (left.ask, right.ask)):
            if a is None or b is None:
                if a is not b:
                    return False
            elif (a.price, a.size) != (b.price, b.size):
                return False
        return True

    def _queue_ahead(self, quote: Quote, book: BookSnapshot) -> float:
        levels = book.bids if quote.is_buy else book.asks
        for level in levels:
            if math.isclose(level.px, quote.price, rel_tol=0.0, abs_tol=1e-12):
                return level.size * self.config.queue_ahead_fraction
        return 0.0

    def run(self, events: Iterable[MarketEvent], quote_function: QuoteFunction) -> SimulationResult:
        ordered = sorted(events, key=lambda event: event.timestamp_ms)
        if not ordered:
            raise ValueError("events cannot be empty")
        inventory = MMInventory(
            base_units=self.config.initial_inventory_base,
            cash_usd=self.config.initial_cash_usd,
        )
        initial_equity: float | None = None
        book: BookSnapshot | None = None
        active_pair: QuotePair | None = None
        active_bid: _RestingQuote | None = None
        active_ask: _RestingQuote | None = None
        desired: QuotePair | None = None
        pending: list[tuple[int, int, QuotePair | None]] = []
        sequence = 0
        fills: list[MakerFill] = []
        mids_ts: list[int] = []
        mids: list[float] = []
        replacements = 0
        rejections = 0
        posted_size = 0.0
        spread_capture = 0.0
        quoted_ms = 0
        last_event_ms = ordered[0].timestamp_ms
        peak_equity: float | None = None
        max_drawdown = 0.0
        max_abs_inventory = abs(inventory.base_units)

        def apply_pending(now_ms: int) -> None:
            nonlocal active_pair, active_bid, active_ask
            nonlocal replacements, rejections, posted_size, quoted_ms, last_event_ms
            cursor = last_event_ms
            while pending and pending[0][0] <= now_ms:
                apply_ms, _, pair = heapq.heappop(pending)
                apply_ms = max(apply_ms, cursor)
                if active_bid is not None or active_ask is not None:
                    quoted_ms += apply_ms - cursor
                cursor = apply_ms
                if book is None:
                    continue
                valid_bid = pair.bid if pair is not None else None
                valid_ask = pair.ask if pair is not None else None
                if (
                    valid_bid is not None
                    and book.best_ask is not None
                    and valid_bid.price >= book.best_ask
                ):
                    valid_bid = None
                    rejections += 1
                if (
                    valid_ask is not None
                    and book.best_bid is not None
                    and valid_ask.price <= book.best_bid
                ):
                    valid_ask = None
                    rejections += 1
                if pair is None or (valid_bid is None and valid_ask is None):
                    active_pair = None
                    active_bid = active_ask = None
                else:
                    active_pair = QuotePair(
                        bid=valid_bid,
                        ask=valid_ask,
                        reservation_price=pair.reservation_price,
                        model_spread=pair.model_spread,
                        timestamp_ms=pair.timestamp_ms,
                        policy=pair.policy,
                        diagnostics=pair.diagnostics,
                    )
                    active_bid = (
                        _RestingQuote(valid_bid, self._queue_ahead(valid_bid, book))
                        if valid_bid is not None
                        else None
                    )
                    active_ask = (
                        _RestingQuote(valid_ask, self._queue_ahead(valid_ask, book))
                        if valid_ask is not None
                        else None
                    )
                    posted_size += sum(
                        quote.size for quote in (valid_bid, valid_ask) if quote is not None
                    )
                replacements += 1
            if active_bid is not None or active_ask is not None:
                quoted_ms += max(0, now_ms - cursor)
            last_event_ms = now_ms

        def execute(
            resting: _RestingQuote, trade: TradePrint
        ) -> tuple[_RestingQuote | None, MakerFill | None]:
            available = trade.size
            if resting.queue_ahead > 0:
                consumed = min(resting.queue_ahead, available)
                resting.queue_ahead -= consumed
                available -= consumed
            size = min(resting.quote.size, available)
            if size <= 0:
                return resting, None
            fee = size * resting.quote.price * self.config.maker_fee_bps / 10_000.0
            fill = MakerFill(
                timestamp_ms=trade.timestamp_ms,
                coin=trade.coin,
                is_buy=resting.quote.is_buy,
                price=resting.quote.price,
                size=size,
                fee_usd=fee,
                cloid=resting.quote.cloid,
            )
            remaining = resting.quote.size - size
            if remaining <= 1e-12:
                return None, fill
            return _RestingQuote(
                Quote(
                    resting.quote.is_buy,
                    resting.quote.price,
                    remaining,
                    resting.quote.cloid,
                ),
                0.0,
            ), fill

        for event in ordered:
            apply_pending(event.timestamp_ms)
            if isinstance(event, BookSnapshot):
                if book is not None and event.coin != book.coin:
                    raise ValueError("simulator supports one coin per replay")
                book = event
                if book.mid is not None and not book.is_crossed:
                    mids_ts.append(book.timestamp_ms)
                    mids.append(book.mid)
                    if initial_equity is None:
                        initial_equity = inventory.equity(book.mid)
                    equity = inventory.equity(book.mid)
                    peak_equity = equity if peak_equity is None else max(peak_equity, equity)
                    max_drawdown = max(max_drawdown, peak_equity - equity)
                context = SimulationContext(event.timestamp_ms, book, inventory, active_pair)
                new_desired = quote_function(context)
                if not self._same_quotes(new_desired, desired):
                    desired = new_desired
                    sequence += 1
                    heapq.heappush(
                        pending,
                        (event.timestamp_ms + self.config.latency_ms, sequence, new_desired),
                    )
                    apply_pending(event.timestamp_ms)
            elif isinstance(event, TradePrint):
                if book is None or event.coin != book.coin:
                    continue
                fill: MakerFill | None = None
                if (
                    event.aggressor == AggressorSide.SELL
                    and active_bid is not None
                    and event.price <= active_bid.quote.price
                ):
                    active_bid, fill = execute(active_bid, event)
                elif (
                    event.aggressor == AggressorSide.BUY
                    and active_ask is not None
                    and event.price >= active_ask.quote.price
                ):
                    active_ask, fill = execute(active_ask, event)
                if fill is not None:
                    inventory.apply_fill(fill)
                    fills.append(fill)
                    if book.mid is not None:
                        direction = 1.0 if fill.is_buy else -1.0
                        spread_capture += direction * (book.mid - fill.price) * fill.size
                    max_abs_inventory = max(max_abs_inventory, abs(inventory.base_units))
                    # A fill changes both inventory and desired displayed size.
                    # Force the policy to reconsider/replenish on the next book.
                    desired = None
                    if active_pair is not None:
                        active_pair = QuotePair(
                            bid=active_bid.quote if active_bid is not None else None,
                            ask=active_ask.quote if active_ask is not None else None,
                            reservation_price=active_pair.reservation_price,
                            model_spread=active_pair.model_spread,
                            timestamp_ms=active_pair.timestamp_ms,
                            policy=active_pair.policy,
                            diagnostics=active_pair.diagnostics,
                        )
            elif isinstance(event, FundingEvent):
                if book is not None and event.coin == book.coin:
                    inventory.apply_funding(event)

        if not mids:
            raise ValueError("replay contains no valid two-sided book")
        if initial_equity is None:
            initial_equity = inventory.equity(mids[0])
        final_equity = inventory.equity(mids[-1])
        pnl = final_equity - initial_equity
        inventory_pnl = pnl - spread_capture + inventory.fees_usd - inventory.funding_usd
        duration_ms = max(ordered[-1].timestamp_ms - ordered[0].timestamp_ms, 1)
        markouts: dict[int, float] = {}
        for horizon in self.config.markout_horizons_ms:
            values: list[float] = []
            for fill in fills:
                index = bisect.bisect_left(mids_ts, fill.timestamp_ms + horizon)
                if index < len(mids):
                    values.append(
                        adverse_markout_bps(
                            is_maker_buy=fill.is_buy,
                            fill_price=fill.price,
                            future_mid=mids[index],
                        )
                    )
            markouts[horizon] = sum(values) / len(values) if values else math.nan
        return SimulationResult(
            fills=tuple(fills),
            final_inventory=inventory.base_units,
            final_cash_usd=inventory.cash_usd,
            final_equity_usd=final_equity,
            pnl_usd=pnl,
            fees_usd=inventory.fees_usd,
            funding_usd=inventory.funding_usd,
            spread_capture_usd=spread_capture,
            inventory_pnl_usd=inventory_pnl,
            fill_rate=sum(fill.size for fill in fills) / posted_size if posted_size else 0.0,
            quote_uptime=min(1.0, quoted_ms / duration_ms),
            cancel_rate_per_minute=max(replacements - 1, 0) / (duration_ms / 60_000.0),
            max_drawdown_usd=max_drawdown,
            max_abs_inventory=max_abs_inventory,
            quote_replacements=replacements,
            post_only_rejections=rejections,
            markout_bps=markouts,
        )
