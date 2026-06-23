"""Quote state machine with conservative safety and refresh gates."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from causal_portfolio.market_making.avellaneda_stoikov import (
    ASModelParams,
    finite_horizon_quotes,
)
from causal_portfolio.market_making.calibration import MarkoutCurve
from causal_portfolio.market_making.toxicity import (
    ToxicityConfig,
    ToxicitySignal,
    toxicity_adjustment,
)
from causal_portfolio.market_making.types import (
    BookSnapshot,
    Quote,
    QuotePair,
    QuotePolicy,
)


@dataclass(frozen=True)
class MarketMakerConfig:
    coin: str
    order_size: float
    tick_size: float
    size_decimals: int
    gamma: float
    kappa: float
    horizon_seconds: float
    arrival_rate: float = 1.0
    max_inventory_base: float = 0.01
    max_book_age_ms: int = 2_000
    max_daily_loss_usd: float = 100.0
    min_free_margin_usd: float = 100.0
    refresh_ticks: int = 1
    max_quote_age_ms: int = 30_000
    max_total_spread_bps: float = 100.0
    strategy_id: str = "cpcm-as-pin-v1"
    toxicity: ToxicityConfig = field(default_factory=ToxicityConfig)

    def __post_init__(self) -> None:
        positive = (
            "order_size",
            "tick_size",
            "gamma",
            "kappa",
            "horizon_seconds",
            "arrival_rate",
            "max_inventory_base",
            "max_daily_loss_usd",
            "max_total_spread_bps",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if not self.coin or not self.strategy_id:
            raise ValueError("coin and strategy_id cannot be empty")
        if self.size_decimals < 0 or self.max_book_age_ms <= 0:
            raise ValueError("size_decimals/max_book_age_ms are invalid")
        if self.min_free_margin_usd < 0 or self.refresh_ticks < 1 or self.max_quote_age_ms <= 0:
            raise ValueError("margin and refresh settings are invalid")


@dataclass(frozen=True)
class QuoteDecision:
    quotes: QuotePair | None
    replace: bool
    cancel: bool
    reason: str
    generation: int


class MarketMakerEngine:
    """Generate desired post-only quote state; never performs network I/O."""

    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.connected = False
        self._active: QuotePair | None = None
        self._generation = 0

    @property
    def active_quotes(self) -> QuotePair | None:
        return self._active

    def set_connected(self, connected: bool) -> None:
        self.connected = bool(connected)

    def _blocked(self, reason: str) -> QuoteDecision:
        return QuoteDecision(
            quotes=None,
            replace=False,
            cancel=self._active is not None,
            reason=reason,
            generation=self._generation + 1,
        )

    @staticmethod
    def _floor_tick(value: float, tick: float) -> float:
        return math.floor((value + 1e-12) / tick) * tick

    @staticmethod
    def _ceil_tick(value: float, tick: float) -> float:
        return math.ceil((value - 1e-12) / tick) * tick

    def _size(self, scale: float) -> float:
        factor = 10**self.config.size_decimals
        return math.floor(self.config.order_size * scale * factor + 1e-12) / factor

    def _cloid(self, generation: int, side: str, price: float, size: float) -> str:
        raw = (
            f"{self.config.strategy_id}|{self.config.coin}|{generation}|"
            f"{side}|{price:.12g}|{size:.12g}"
        ).encode()
        return "0x" + hashlib.sha256(raw).hexdigest()[:32]

    def _needs_refresh(self, desired: QuotePair, now_ms: int) -> bool:
        active = self._active
        if active is None:
            return True
        if now_ms - active.timestamp_ms >= self.config.max_quote_age_ms:
            return True
        threshold = self.config.refresh_ticks * self.config.tick_size
        for old, new in ((active.bid, desired.bid), (active.ask, desired.ask)):
            if (old is None) != (new is None):
                return True
            if old is not None and new is not None:
                if abs(old.price - new.price) + 1e-12 >= threshold:
                    return True
                if old.size != new.size:
                    return True
        return active.policy != desired.policy

    def decide(
        self,
        book: BookSnapshot,
        *,
        now_ms: int,
        inventory_base: float,
        sigma_per_sqrt_second: float,
        free_margin_usd: float,
        daily_pnl_usd: float,
        toxicity_signal: ToxicitySignal | None = None,
        markouts: MarkoutCurve | None = None,
    ) -> QuoteDecision:
        cfg = self.config
        if not self.connected:
            return self._blocked("disconnected")
        if book.coin != cfg.coin:
            return self._blocked(f"book coin mismatch: {book.coin}")
        observed_ms = book.received_ms if book.received_ms is not None else book.timestamp_ms
        if now_ms < observed_ms or now_ms - observed_ms > cfg.max_book_age_ms:
            return self._blocked("stale_or_future_book")
        if book.mid is None or book.is_crossed:
            return self._blocked("empty_or_crossed_book")
        if not math.isfinite(sigma_per_sqrt_second) or sigma_per_sqrt_second <= 0:
            return self._blocked("invalid_volatility")
        if (
            not math.isfinite(inventory_base)
            or abs(inventory_base) > cfg.max_inventory_base + 1e-12
        ):
            return self._blocked("inventory_limit_exceeded")
        if free_margin_usd < cfg.min_free_margin_usd:
            return self._blocked("insufficient_free_margin")
        if daily_pnl_usd <= -cfg.max_daily_loss_usd:
            return self._blocked("daily_loss_limit")
        if cfg.toxicity.policy != QuotePolicy.AS_BASELINE:
            if toxicity_signal is None or not toxicity_signal.estimator_usable:
                return self._blocked("toxicity_estimator_unavailable")

        params = ASModelParams(
            gamma=cfg.gamma,
            sigma=sigma_per_sqrt_second,
            kappa=cfg.kappa,
            horizon=cfg.horizon_seconds,
            arrival_rate=cfg.arrival_rate,
        )
        inventory_units = inventory_base / cfg.order_size
        try:
            levels = finite_horizon_quotes(book.mid, inventory_units, params)
            adjustment = toxicity_adjustment(
                toxicity_signal, cfg.toxicity, markouts=markouts
            )
        except ValueError as exc:
            return self._blocked(f"model_rejected: {exc}")

        bid_raw = levels.bid - book.mid * adjustment.bid_extra_bps / 10_000.0
        ask_raw = levels.ask + book.mid * adjustment.ask_extra_bps / 10_000.0
        # Alo quotes must not cross the observed opposite best price.
        bid_px = self._floor_tick(min(bid_raw, book.best_ask - cfg.tick_size), cfg.tick_size)
        ask_px = self._ceil_tick(max(ask_raw, book.best_bid + cfg.tick_size), cfg.tick_size)
        if bid_px <= 0 or bid_px >= ask_px:
            return self._blocked("invalid_post_only_prices")
        spread_bps = (ask_px - bid_px) / book.mid * 10_000.0
        if spread_bps > cfg.max_total_spread_bps:
            return self._blocked("spread_limit_exceeded")

        bid_size = self._size(adjustment.bid_size_scale)
        ask_size = self._size(adjustment.ask_size_scale)
        generation = self._generation + 1
        bid: Quote | None = None
        ask: Quote | None = None
        if bid_size > 0 and inventory_base + bid_size <= cfg.max_inventory_base + 1e-12:
            bid = Quote(True, bid_px, bid_size, self._cloid(generation, "bid", bid_px, bid_size))
        if ask_size > 0 and inventory_base - ask_size >= -cfg.max_inventory_base - 1e-12:
            ask = Quote(False, ask_px, ask_size, self._cloid(generation, "ask", ask_px, ask_size))
        if bid is None and ask is None:
            return self._blocked("inventory_limit_blocks_both_sides")
        desired = QuotePair(
            bid=bid,
            ask=ask,
            reservation_price=levels.reservation_price,
            model_spread=ask_px - bid_px,
            timestamp_ms=now_ms,
            policy=cfg.toxicity.policy,
            diagnostics={
                "coin": cfg.coin,
                "inventory_units": inventory_units,
                "sigma": sigma_per_sqrt_second,
                "pin": toxicity_signal.pin if toxicity_signal else None,
                "spread_bps": spread_bps,
            },
        )
        refresh = self._needs_refresh(desired, now_ms)
        return QuoteDecision(
            quotes=desired if refresh else self._active,
            replace=refresh,
            cancel=False,
            reason="refresh" if refresh else "unchanged",
            generation=generation if refresh else self._generation,
        )

    def acknowledge(self, decision: QuoteDecision, *, success: bool) -> None:
        """Advance local state only after the adapter confirms an action."""
        if not success:
            return
        if decision.cancel:
            self._active = None
            self._generation = max(self._generation, decision.generation)
        elif decision.replace and decision.quotes is not None:
            self._active = decision.quotes
            self._generation = decision.generation

    def clear(self) -> None:
        self._active = None
