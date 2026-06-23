"""Walk-forward volatility, fill-intensity, and markout calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from causal_portfolio.market_making.types import AggressorSide


def ewma_absolute_volatility(
    prices: Sequence[float],
    timestamps_ms: Sequence[int],
    *,
    half_life_seconds: float = 300.0,
) -> float:
    """Estimate absolute price volatility per square-root second.

    Irregularly spaced price changes contribute instantaneous variance
    ``(delta_price ** 2) / delta_seconds`` with time-aware EWMA decay.
    """
    px = np.asarray(prices, dtype=float)
    ts = np.asarray(timestamps_ms, dtype=np.int64)
    if px.ndim != 1 or ts.ndim != 1 or len(px) != len(ts) or len(px) < 2:
        raise ValueError("prices/timestamps must be equal one-dimensional arrays of length >= 2")
    if np.any(~np.isfinite(px)) or np.any(px <= 0) or np.any(np.diff(ts) <= 0):
        raise ValueError("prices must be positive and timestamps strictly increasing")
    if not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be > 0")
    variance = 0.0
    initialized = False
    for change, delta_ms in zip(np.diff(px), np.diff(ts)):
        dt = float(delta_ms) / 1_000.0
        instant = float(change * change / dt)
        decay = math.exp(-math.log(2.0) * dt / half_life_seconds)
        variance = instant if not initialized else decay * variance + (1 - decay) * instant
        initialized = True
    return math.sqrt(max(variance, 0.0))


@dataclass(frozen=True)
class FillObservation:
    distance: float
    exposure_seconds: float
    fills: int
    side: AggressorSide | None = None

    def __post_init__(self) -> None:
        if self.distance < 0 or not math.isfinite(self.distance):
            raise ValueError("distance must be finite and >= 0")
        if self.exposure_seconds <= 0 or not math.isfinite(self.exposure_seconds):
            raise ValueError("exposure_seconds must be finite and > 0")
        if self.fills < 0 or int(self.fills) != self.fills:
            raise ValueError("fills must be a non-negative integer")


@dataclass(frozen=True)
class IntensityEstimate:
    arrival_rate: float
    kappa: float
    observations: int
    exposure_seconds: float
    fills: int
    converged: bool
    log_likelihood: float

    def rate(self, distance: float) -> float:
        return self.arrival_rate * math.exp(-self.kappa * distance)


def fit_exponential_intensity(
    observations: Iterable[FillObservation],
    *,
    side: AggressorSide | None = None,
) -> IntensityEstimate:
    """Fit ``lambda(distance)=A*exp(-kappa*distance)`` by Poisson MLE."""
    selected = [o for o in observations if side is None or o.side == side]
    if len(selected) < 3:
        raise ValueError("at least three quote-exposure observations are required")
    distance = np.array([o.distance for o in selected], dtype=float)
    exposure = np.array([o.exposure_seconds for o in selected], dtype=float)
    fills = np.array([o.fills for o in selected], dtype=float)
    if float(fills.sum()) <= 0:
        raise ValueError("at least one observed fill is required")

    def nll(theta: np.ndarray) -> float:
        log_a, log_k = theta
        kappa = math.exp(float(log_k))
        log_rate = log_a - kappa * distance
        expected = exposure * np.exp(np.clip(log_rate, -700, 700))
        return float(np.sum(expected - fills * (np.log(exposure) + log_rate)))

    naive_rate = max(float(fills.sum() / exposure.sum()), 1e-9)
    result = minimize(
        nll,
        np.array([math.log(naive_rate), math.log(1.0 / max(np.mean(distance), 1e-6))]),
        method="L-BFGS-B",
        bounds=[(-30.0, 30.0), (-20.0, 20.0)],
    )
    arrival_rate = math.exp(float(result.x[0]))
    kappa = math.exp(float(result.x[1]))
    return IntensityEstimate(
        arrival_rate=arrival_rate,
        kappa=kappa,
        observations=len(selected),
        exposure_seconds=float(exposure.sum()),
        fills=int(fills.sum()),
        converged=bool(result.success),
        log_likelihood=-float(result.fun),
    )


@dataclass(frozen=True)
class MarkoutObservation:
    pin: float
    aggressor: AggressorSide
    adverse_bps: float

    def __post_init__(self) -> None:
        if not 0 <= self.pin <= 1 or not math.isfinite(self.adverse_bps):
            raise ValueError("pin must be in [0,1] and adverse_bps finite")


@dataclass(frozen=True)
class MarkoutCurve:
    """Piecewise-constant expected adverse markout by PIN bucket and side."""

    edges: tuple[float, ...]
    buy_bps: tuple[float, ...]
    sell_bps: tuple[float, ...]
    counts: tuple[int, ...]

    def expected_bps(self, aggressor: AggressorSide, pin: float) -> float:
        if not 0 <= pin <= 1:
            raise ValueError("pin must be in [0,1]")
        idx = int(np.searchsorted(self.edges[1:], pin, side="right"))
        idx = min(idx, len(self.buy_bps) - 1)
        values = self.buy_bps if aggressor == AggressorSide.BUY else self.sell_bps
        return values[idx]


def fit_markout_curve(
    observations: Iterable[MarkoutObservation],
    *,
    edges: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0),
    prior_bps: float = 0.0,
    prior_weight: float = 5.0,
) -> MarkoutCurve:
    """Estimate non-negative adverse markout with light prior shrinkage."""
    obs = list(observations)
    bins = np.asarray(edges, dtype=float)
    if len(bins) < 2 or bins[0] != 0 or bins[-1] != 1 or np.any(np.diff(bins) <= 0):
        raise ValueError("edges must increase from 0 to 1")
    if prior_weight < 0:
        raise ValueError("prior_weight must be >= 0")
    buy_values: list[float] = []
    sell_values: list[float] = []
    counts: list[int] = []
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        include = lambda x: left <= x <= right if i == len(bins) - 2 else left <= x < right
        bucket = [o for o in obs if include(o.pin)]
        counts.append(len(bucket))
        for side, target in (
            (AggressorSide.BUY, buy_values),
            (AggressorSide.SELL, sell_values),
        ):
            values = [max(0.0, o.adverse_bps) for o in bucket if o.aggressor == side]
            denominator = len(values) + prior_weight
            estimate = (
                (sum(values) + prior_weight * prior_bps) / denominator
                if denominator > 0
                else prior_bps
            )
            target.append(float(max(0.0, estimate)))
    return MarkoutCurve(
        edges=tuple(map(float, bins)),
        buy_bps=tuple(buy_values),
        sell_bps=tuple(sell_values),
        counts=tuple(counts),
    )


def adverse_markout_bps(*, is_maker_buy: bool, fill_price: float, future_mid: float) -> float:
    """Signed adverse selection: positive means the market moved against us."""
    if fill_price <= 0 or future_mid <= 0:
        raise ValueError("prices must be > 0")
    direction = -1.0 if is_maker_buy else 1.0
    return direction * (future_mid - fill_price) / fill_price * 10_000.0
