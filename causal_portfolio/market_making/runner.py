"""Thread-safe shadow/testnet/live orchestration for one market-making coin."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from causal_portfolio.market_making.calibration import (
    MarkoutCurve,
    ewma_absolute_volatility,
)
from causal_portfolio.market_making.engine import MarketMakerEngine
from causal_portfolio.market_making.hyperliquid import HLMarketMakerAdapter
from causal_portfolio.market_making.pin import PINFit, pin_posteriors
from causal_portfolio.market_making.recorder import normalize_hl_message
from causal_portfolio.market_making.toxicity import ToxicitySignal
from causal_portfolio.market_making.types import AggressorSide, BookSnapshot, TradePrint

logger = logging.getLogger("cpcm.market_making.runner")


@dataclass(frozen=True)
class RuntimeConfig:
    fallback_sigma_per_sqrt_second: float
    volatility_half_life_seconds: float = 300.0
    volatility_window: int = 2_000
    account_refresh_seconds: float = 5.0
    acknowledge_mainnet: bool = False

    def __post_init__(self) -> None:
        if self.fallback_sigma_per_sqrt_second <= 0:
            raise ValueError("fallback sigma must be > 0")
        if self.volatility_half_life_seconds <= 0 or self.volatility_window < 2:
            raise ValueError("volatility configuration is invalid")
        if self.account_refresh_seconds <= 0:
            raise ValueError("account_refresh_seconds must be > 0")


class MarketMakingService:
    """Own subscriptions and serialize quote decisions under one lock."""

    def __init__(
        self,
        engine: MarketMakerEngine,
        adapter: HLMarketMakerAdapter,
        runtime: RuntimeConfig,
        *,
        toxicity_signal: ToxicitySignal | None = None,
        markouts: MarkoutCurve | None = None,
        pin_fit: PINFit | None = None,
    ):
        self.engine = engine
        self.adapter = adapter
        self.runtime = runtime
        self.toxicity_signal = toxicity_signal
        self.markouts = markouts
        self.pin_fit = pin_fit
        self._lock = threading.RLock()
        self._prices: deque[float] = deque(maxlen=runtime.volatility_window)
        self._timestamps: deque[int] = deque(maxlen=runtime.volatility_window)
        self._metrics = {
            "account_value_usd": 0.0,
            "free_margin_usd": 0.0,
            "inventory_base": 0.0,
        }
        self._start_equity: float | None = None
        self._equity_day: str | None = None
        self._last_account_refresh = 0.0
        self._last_dead_man_arm = 0.0
        self._running = False
        self._flow_day: str | None = None
        self._daily_buys = 0
        self._daily_sells = 0
        self.last_error: str | None = None

    def update_toxicity(
        self, signal: ToxicitySignal | None, markouts: MarkoutCurve | None = None
    ) -> None:
        with self._lock:
            self.toxicity_signal = signal
            if markouts is not None:
                self.markouts = markouts

    def _refresh_account(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_account_refresh < self.runtime.account_refresh_seconds:
            return
        self._metrics = self.adapter.fetch_account_metrics(self.engine.config.coin)
        today = datetime.now(timezone.utc).date().isoformat()
        if self._start_equity is None or self._equity_day != today:
            self._start_equity = self._metrics["account_value_usd"]
            self._equity_day = today
        self._last_account_refresh = now

    def _sigma(self) -> float:
        if len(self._prices) < 3:
            return self.runtime.fallback_sigma_per_sqrt_second
        try:
            return ewma_absolute_volatility(
                list(self._prices),
                list(self._timestamps),
                half_life_seconds=self.runtime.volatility_half_life_seconds,
            )
        except ValueError:
            return self.runtime.fallback_sigma_per_sqrt_second

    def _renew_dead_man(self) -> None:
        cfg = self.adapter.config
        if cfg.dry_run or not cfg.dedicated_account:
            return
        now = time.monotonic()
        if now - self._last_dead_man_arm >= cfg.dead_man_timeout_seconds / 2:
            self.adapter.arm_dead_man()
            self._last_dead_man_arm = now

    def _handle_book(self, book: BookSnapshot) -> None:
        if book.mid is None or book.is_crossed:
            return
        if not self._timestamps or book.timestamp_ms > self._timestamps[-1]:
            self._timestamps.append(book.timestamp_ms)
            self._prices.append(book.mid)
        self._refresh_account()
        self.engine.set_connected(True)
        start_equity = self._start_equity or self._metrics["account_value_usd"]
        daily_pnl = self._metrics["account_value_usd"] - start_equity
        decision = self.engine.decide(
            book,
            now_ms=int(time.time() * 1_000),
            inventory_base=self._metrics["inventory_base"],
            sigma_per_sqrt_second=self._sigma(),
            free_margin_usd=self._metrics["free_margin_usd"],
            daily_pnl_usd=daily_pnl,
            toxicity_signal=self.toxicity_signal,
            markouts=self.markouts,
        )
        if decision.replace or decision.cancel:
            try:
                self.adapter.apply_decision(
                    decision,
                    acknowledge_mainnet=self.runtime.acknowledge_mainnet,
                )
                self.engine.acknowledge(decision, success=True)
                self._renew_dead_man()
            except Exception as exc:
                self.engine.acknowledge(decision, success=False)
                self.last_error = str(exc)
                logger.exception("maker decision failed; cancelling owned quotes")
                try:
                    self.adapter.cancel_strategy_quotes()
                finally:
                    self.engine.set_connected(False)

    def _handle_trade(self, trade: TradePrint) -> None:
        if self.pin_fit is None:
            return
        day = datetime.fromtimestamp(
            trade.timestamp_ms / 1_000, tz=timezone.utc
        ).date().isoformat()
        if day != self._flow_day:
            self._flow_day = day
            self._daily_buys = 0
            self._daily_sells = 0
        if trade.aggressor == AggressorSide.BUY:
            self._daily_buys += 1
        else:
            self._daily_sells += 1
        posterior = pin_posteriors(
            [self._daily_buys], [self._daily_sells], self.pin_fit
        )[0]
        self.toxicity_signal = ToxicitySignal(
            pin=self.pin_fit.pin,
            informed_buy_probability=posterior.informed_buy,
            informed_sell_probability=posterior.informed_sell,
            estimator_usable=self.pin_fit.usable,
        )

    def on_message(self, message: dict[str, Any]) -> None:
        with self._lock:
            try:
                if message.get("channel") in {"userFills", "orderUpdates"}:
                    self._refresh_account(force=True)
                    return
                for event in normalize_hl_message(message):
                    if isinstance(event, BookSnapshot):
                        self._handle_book(event)
                    elif isinstance(event, TradePrint):
                        self._handle_trade(event)
            except Exception as exc:
                self.last_error = str(exc)
                logger.exception("market-making message handler failed")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self.adapter.reconcile_owned_orders()
            self._refresh_account(force=True)
            self.adapter.subscribe_market(self.engine.config.coin, self.on_message)
            self.adapter.subscribe_account(self.on_message)
            self.engine.set_connected(True)
            self._running = True

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self.engine.set_connected(False)
            try:
                self.adapter.cancel_strategy_quotes()
                self.adapter.disarm_dead_man()
            finally:
                self.adapter.close()
                self.engine.clear()
                self._running = False
