"""Dependency-free event and quote types used by live and replay paths."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        op = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {op}, got {value!r}")


class AggressorSide(str, Enum):
    """Direction of the liquidity taker."""

    BUY = "buy"
    SELL = "sell"

    @classmethod
    def from_hyperliquid(cls, side: str) -> "AggressorSide":
        """Map Hyperliquid trade side (B/A) to aggressor direction."""
        normalized = side.strip().upper()
        if normalized == "B":
            return cls.BUY
        if normalized == "A":
            return cls.SELL
        raise ValueError(f"Unknown Hyperliquid trade side: {side!r}")


class QuotePolicy(str, Enum):
    AS_BASELINE = "as_baseline"
    AS_PIN_SPREAD = "as_pin_spread"
    AS_PIN_SIZE = "as_pin_size"
    AS_PIN_POSTERIOR = "as_pin_posterior"


@dataclass(frozen=True)
class BookLevel:
    px: float
    size: float

    def __post_init__(self) -> None:
        _positive("px", self.px)
        _positive("size", self.size, allow_zero=True)


@dataclass(frozen=True)
class BookSnapshot:
    """Best-first L2 snapshot at exchange time."""

    timestamp_ms: int
    coin: str
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    received_ms: int | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be >= 0")
        if not self.coin:
            raise ValueError("coin cannot be empty")
        if any(a.px <= b.px for a, b in zip(self.bids, self.bids[1:])):
            raise ValueError("bids must be strictly descending")
        if any(a.px >= b.px for a, b in zip(self.asks, self.asks[1:])):
            raise ValueError("asks must be strictly ascending")

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].px if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].px if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def is_crossed(self) -> bool:
        return (
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid >= self.best_ask
        )


@dataclass(frozen=True)
class TradePrint:
    timestamp_ms: int
    coin: str
    price: float
    size: float
    aggressor: AggressorSide
    trade_id: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be >= 0")
        _positive("price", self.price)
        _positive("size", self.size)


@dataclass(frozen=True)
class FundingEvent:
    timestamp_ms: int
    coin: str
    rate: float
    mark_price: float

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or not math.isfinite(self.rate):
            raise ValueError("Funding timestamp/rate must be finite and valid")
        _positive("mark_price", self.mark_price)


@dataclass(frozen=True)
class Quote:
    is_buy: bool
    price: float
    size: float
    cloid: str

    def __post_init__(self) -> None:
        _positive("price", self.price)
        _positive("size", self.size)
        if not self.cloid.startswith("0x") or len(self.cloid) != 34:
            raise ValueError("cloid must be a 16-byte 0x-prefixed hex string")
        try:
            int(self.cloid[2:], 16)
        except ValueError as exc:
            raise ValueError("cloid must contain only hexadecimal characters") from exc


@dataclass(frozen=True)
class QuotePair:
    bid: Quote | None
    ask: Quote | None
    reservation_price: float
    model_spread: float
    timestamp_ms: int
    policy: QuotePolicy = QuotePolicy.AS_BASELINE
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _positive("reservation_price", self.reservation_price)
        _positive("model_spread", self.model_spread, allow_zero=True)
        if self.bid is not None and not self.bid.is_buy:
            raise ValueError("bid quote must have is_buy=True")
        if self.ask is not None and self.ask.is_buy:
            raise ValueError("ask quote must have is_buy=False")
        if self.bid is not None and self.ask is not None:
            if self.bid.price >= self.ask.price:
                raise ValueError("bid must be below ask")
            if not (self.bid.price <= self.reservation_price <= self.ask.price):
                raise ValueError("reservation price must lie between quotes")


@dataclass(frozen=True)
class MakerFill:
    timestamp_ms: int
    coin: str
    is_buy: bool
    price: float
    size: float
    fee_usd: float = 0.0
    cloid: str | None = None

    def __post_init__(self) -> None:
        _positive("price", self.price)
        _positive("size", self.size)
        if not math.isfinite(self.fee_usd):
            raise ValueError("fee_usd must be finite")


@dataclass
class MMInventory:
    """Cash/inventory ledger with signed base-asset inventory."""

    base_units: float = 0.0
    cash_usd: float = 0.0
    fees_usd: float = 0.0
    funding_usd: float = 0.0

    def apply_fill(self, fill: MakerFill) -> None:
        signed_size = fill.size if fill.is_buy else -fill.size
        self.base_units += signed_size
        self.cash_usd -= signed_size * fill.price
        self.cash_usd -= fill.fee_usd
        self.fees_usd += fill.fee_usd

    def apply_funding(self, event: FundingEvent) -> float:
        # Positive funding means longs pay shorts.
        payment = -self.base_units * event.mark_price * event.rate
        self.cash_usd += payment
        self.funding_usd += payment
        return payment

    def equity(self, mark_price: float) -> float:
        _positive("mark_price", mark_price)
        return self.cash_usd + self.base_units * mark_price


MarketEvent = BookSnapshot | TradePrint | FundingEvent
