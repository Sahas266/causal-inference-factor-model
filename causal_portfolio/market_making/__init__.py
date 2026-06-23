"""Inventory-aware market making and informed-flow controls.

The package is deliberately separate from :mod:`causal_portfolio.execution`:
portfolio rebalancing uses marketable IOC orders, while market making owns
persistent post-only quotes and must never call the rebalancer's global cancel
path.
"""

from causal_portfolio.market_making.avellaneda_stoikov import (
    ASModelParams,
    HJBResult,
    finite_horizon_quotes,
    solve_finite_horizon_hjb,
    stationary_reservation_prices,
)
from causal_portfolio.market_making.artifacts import (
    CalibrationArtifact,
    load_calibration_artifact,
)
from causal_portfolio.market_making.engine import (
    MarketMakerConfig,
    MarketMakerEngine,
    QuoteDecision,
)
from causal_portfolio.market_making.pin import PINFit, fit_pin, pin_posteriors
from causal_portfolio.market_making.simulator import (
    EventDrivenSimulator,
    SimulationConfig,
    SimulationResult,
)
from causal_portfolio.market_making.types import (
    AggressorSide,
    BookLevel,
    BookSnapshot,
    FundingEvent,
    MakerFill,
    MMInventory,
    Quote,
    QuotePair,
    QuotePolicy,
    TradePrint,
)

__all__ = [
    "ASModelParams",
    "AggressorSide",
    "BookLevel",
    "BookSnapshot",
    "CalibrationArtifact",
    "EventDrivenSimulator",
    "FundingEvent",
    "HJBResult",
    "MMInventory",
    "MakerFill",
    "MarketMakerConfig",
    "MarketMakerEngine",
    "PINFit",
    "Quote",
    "QuoteDecision",
    "QuotePair",
    "QuotePolicy",
    "SimulationConfig",
    "SimulationResult",
    "TradePrint",
    "finite_horizon_quotes",
    "fit_pin",
    "load_calibration_artifact",
    "pin_posteriors",
    "solve_finite_horizon_hjb",
    "stationary_reservation_prices",
]
