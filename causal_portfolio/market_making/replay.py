"""Walk-forward quote policy adapter and identical-window policy comparison."""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone

from causal_portfolio.market_making.artifacts import CalibrationArtifact
from causal_portfolio.market_making.calibration import ewma_absolute_volatility
from causal_portfolio.market_making.engine import MarketMakerConfig, MarketMakerEngine
from causal_portfolio.market_making.pin import pin_posteriors
from causal_portfolio.market_making.simulator import (
    EventDrivenSimulator,
    SimulationConfig,
    SimulationContext,
    SimulationResult,
)
from causal_portfolio.market_making.toxicity import ToxicityConfig, ToxicitySignal
from causal_portfolio.market_making.types import (
    AggressorSide,
    MarketEvent,
    QuotePair,
    QuotePolicy,
    TradePrint,
)


def _toxicity_timeline(
    events: list[MarketEvent], artifact: CalibrationArtifact
) -> tuple[list[int], list[ToxicitySignal]]:
    fit = artifact.pin_fit
    if fit is None:
        return [], []
    timestamps: list[int] = []
    signals: list[ToxicitySignal] = []
    day: str | None = None
    buys = sells = 0
    for event in sorted(events, key=lambda item: item.timestamp_ms):
        if not isinstance(event, TradePrint):
            continue
        event_day = datetime.fromtimestamp(
            event.timestamp_ms / 1_000, tz=timezone.utc
        ).date().isoformat()
        if event_day != day:
            day = event_day
            buys = sells = 0
        if event.aggressor == AggressorSide.BUY:
            buys += 1
        else:
            sells += 1
        posterior = pin_posteriors([buys], [sells], fit)[0]
        timestamps.append(event.timestamp_ms)
        signals.append(
            ToxicitySignal(
                pin=fit.pin,
                informed_buy_probability=posterior.informed_buy,
                informed_sell_probability=posterior.informed_sell,
                estimator_usable=fit.usable,
            )
        )
    return timestamps, signals


class ReplayQuotePolicy:
    """Adapt the live engine to simulator callbacks using only prior events."""

    def __init__(
        self,
        config: MarketMakerConfig,
        events: list[MarketEvent],
        artifact: CalibrationArtifact | None,
        *,
        fallback_sigma: float,
        volatility_half_life_seconds: float = 300.0,
        volatility_window: int = 2_000,
    ):
        self.engine = MarketMakerEngine(config)
        self.engine.set_connected(True)
        self.artifact = artifact
        self.fallback_sigma = fallback_sigma
        self.half_life = volatility_half_life_seconds
        self.prices: deque[float] = deque(maxlen=volatility_window)
        self.timestamps: deque[int] = deque(maxlen=volatility_window)
        self.signal_times, self.signals = (
            _toxicity_timeline(events, artifact) if artifact else ([], [])
        )
        self.initial_equity: float | None = None

    def _signal(self, timestamp_ms: int) -> ToxicitySignal | None:
        index = bisect.bisect_right(self.signal_times, timestamp_ms) - 1
        if index >= 0:
            return self.signals[index]
        if self.artifact and self.artifact.pin_fit and self.artifact.posterior:
            return ToxicitySignal(
                pin=self.artifact.pin_fit.pin,
                informed_buy_probability=self.artifact.posterior.informed_buy,
                informed_sell_probability=self.artifact.posterior.informed_sell,
                estimator_usable=self.artifact.pin_fit.usable,
            )
        return None

    def __call__(self, context: SimulationContext) -> QuotePair | None:
        if not self.timestamps or context.timestamp_ms > self.timestamps[-1]:
            self.timestamps.append(context.timestamp_ms)
            self.prices.append(context.book.mid)
        sigma = self.fallback_sigma
        if len(self.prices) >= 3:
            sigma = ewma_absolute_volatility(
                list(self.prices),
                list(self.timestamps),
                half_life_seconds=self.half_life,
            )
        equity = context.inventory.equity(context.book.mid)
        if self.initial_equity is None:
            self.initial_equity = equity
        decision = self.engine.decide(
            context.book,
            now_ms=context.timestamp_ms,
            inventory_base=context.inventory.base_units,
            sigma_per_sqrt_second=sigma,
            free_margin_usd=1e18,
            daily_pnl_usd=equity - self.initial_equity,
            toxicity_signal=self._signal(context.timestamp_ms),
            markouts=self.artifact.markouts if self.artifact else None,
        )
        if decision.cancel:
            self.engine.acknowledge(decision, success=True)
            return None
        if decision.replace:
            self.engine.acknowledge(decision, success=True)
        return decision.quotes


def compare_policies(
    events: list[MarketEvent],
    base_config: MarketMakerConfig,
    artifact: CalibrationArtifact,
    *,
    fallback_sigma: float,
    simulation_config: SimulationConfig = SimulationConfig(),
) -> dict[QuotePolicy, SimulationResult]:
    """Replay all feasible policies on exactly the same ordered events."""
    if not events:
        raise ValueError("events cannot be empty")
    first_ms = min(event.timestamp_ms for event in events)
    if artifact.as_of.timestamp() * 1_000 > first_ms:
        raise ValueError("calibration as_of must not be after replay start (look-ahead)")
    policies = [QuotePolicy.AS_BASELINE, QuotePolicy.AS_PIN_SIZE]
    if artifact.markouts is not None:
        policies.extend([QuotePolicy.AS_PIN_SPREAD, QuotePolicy.AS_PIN_POSTERIOR])
    output: dict[QuotePolicy, SimulationResult] = {}
    for policy in policies:
        config = replace(base_config, toxicity=ToxicityConfig(policy=policy))
        quote_policy = ReplayQuotePolicy(
            config, events, artifact, fallback_sigma=fallback_sigma
        )
        output[policy] = EventDrivenSimulator(simulation_config).run(
            events, quote_policy
        )
    return output
