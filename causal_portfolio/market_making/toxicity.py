"""PIN and posterior-informed quote overlays."""

from __future__ import annotations

import math
from dataclasses import dataclass

from causal_portfolio.market_making.calibration import MarkoutCurve
from causal_portfolio.market_making.types import AggressorSide, QuotePolicy


@dataclass(frozen=True)
class ToxicitySignal:
    pin: float
    informed_buy_probability: float
    informed_sell_probability: float
    estimator_usable: bool = True

    def __post_init__(self) -> None:
        for name in ("pin", "informed_buy_probability", "informed_sell_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.informed_buy_probability + self.informed_sell_probability > 1 + 1e-12:
            raise ValueError("posterior probabilities cannot sum above one")


@dataclass(frozen=True)
class ToxicityConfig:
    policy: QuotePolicy = QuotePolicy.AS_BASELINE
    min_size_scale: float = 0.25
    size_toxicity_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.min_size_scale <= 1:
            raise ValueError("min_size_scale must be in (0,1]")
        if self.size_toxicity_scale < 0 or not math.isfinite(self.size_toxicity_scale):
            raise ValueError("size_toxicity_scale must be finite and >= 0")


@dataclass(frozen=True)
class ToxicityAdjustment:
    bid_extra_bps: float = 0.0
    ask_extra_bps: float = 0.0
    bid_size_scale: float = 1.0
    ask_size_scale: float = 1.0


def toxicity_adjustment(
    signal: ToxicitySignal | None,
    config: ToxicityConfig,
    *,
    markouts: MarkoutCurve | None = None,
) -> ToxicityAdjustment:
    """Convert a validated signal into price/size controls.

    Spread premiums require a markout curve calibrated from prior fills; this
    intentionally rejects arbitrary production multipliers.
    """
    if config.policy == QuotePolicy.AS_BASELINE or signal is None:
        return ToxicityAdjustment()
    if not signal.estimator_usable:
        raise ValueError("toxicity estimator is not usable")
    if config.policy == QuotePolicy.AS_PIN_SIZE:
        scale = max(
            config.min_size_scale,
            min(1.0, 1.0 - config.size_toxicity_scale * signal.pin),
        )
        return ToxicityAdjustment(bid_size_scale=scale, ask_size_scale=scale)
    if markouts is None:
        raise ValueError("PIN spread/posterior policy requires calibrated markouts")
    buy_bps = markouts.expected_bps(AggressorSide.BUY, signal.pin)
    sell_bps = markouts.expected_bps(AggressorSide.SELL, signal.pin)
    if config.policy == QuotePolicy.AS_PIN_SPREAD:
        symmetric = (buy_bps + sell_bps) / 2.0
        return ToxicityAdjustment(symmetric, symmetric)
    if config.policy == QuotePolicy.AS_PIN_POSTERIOR:
        return ToxicityAdjustment(
            bid_extra_bps=sell_bps * signal.informed_sell_probability,
            ask_extra_bps=buy_bps * signal.informed_buy_probability,
        )
    raise ValueError(f"Unsupported quote policy: {config.policy}")
