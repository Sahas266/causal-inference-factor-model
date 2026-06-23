"""Avellaneda-Stoikov quote controls and a bounded-inventory HJB benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class ASModelParams:
    """Parameters in one consistent time/price unit system.

    ``sigma`` is absolute price volatility per square-root time unit,
    ``horizon`` uses that same time unit, ``inventory`` is measured in quote
    fill units, and ``gamma`` is inverse dollars of risk tolerance.
    """

    gamma: float
    sigma: float
    kappa: float
    horizon: float
    arrival_rate: float = 1.0

    def __post_init__(self) -> None:
        for name in ("gamma", "sigma", "kappa", "horizon", "arrival_rate"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")


@dataclass(frozen=True)
class ASQuoteLevels:
    reservation_price: float
    bid: float
    ask: float
    total_spread: float
    inventory_skew: float


def finite_horizon_quotes(
    mid: float,
    inventory: float,
    params: ASModelParams,
    *,
    remaining_horizon: float | None = None,
) -> ASQuoteLevels:
    """Paper's finite-horizon closed-form approximation.

    The result assumes symmetric exponential fill intensity and is not the
    exact HJB control. Prices are left unrounded so exchange-specific tick
    handling can conservatively round bid down and ask up.
    """
    _finite("mid", mid)
    _finite("inventory", inventory)
    if mid <= 0:
        raise ValueError("mid must be > 0")
    tau = params.horizon if remaining_horizon is None else remaining_horizon
    if not math.isfinite(tau) or tau < 0 or tau > params.horizon:
        raise ValueError("remaining_horizon must be in [0, params.horizon]")
    variance = params.sigma * params.sigma * tau
    skew = inventory * params.gamma * variance
    reservation = mid - skew
    liquidity_spread = (2.0 / params.gamma) * math.log1p(
        params.gamma / params.kappa
    )
    total_spread = params.gamma * variance + liquidity_spread
    bid = reservation - total_spread / 2.0
    ask = reservation + total_spread / 2.0
    if bid <= 0:
        raise ValueError(
            "Model produced a non-positive bid; parameters/inventory are outside "
            "the usable approximation domain"
        )
    return ASQuoteLevels(reservation, bid, ask, total_spread, skew)


def stationary_reservation_prices(
    mid: float,
    inventory: int,
    gamma: float,
    sigma: float,
    omega: float,
) -> tuple[float, float]:
    """Paper's discounted infinite-horizon *reservation* bid and ask.

    This is not presented as a complete stationary maker quote control. It is
    kept separate to prevent a rolling finite horizon from being mislabeled as
    the paper's infinite-horizon formulation.
    """
    for name, value in (("mid", mid), ("gamma", gamma), ("sigma", sigma), ("omega", omega)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")
    q = int(inventory)
    denominator = 2.0 * omega - gamma * gamma * q * q * sigma * sigma
    if denominator <= 0:
        raise ValueError("omega violates the stationary inventory bound")
    ask_arg = 1.0 + ((1 - 2 * q) * gamma * gamma * sigma * sigma) / denominator
    bid_arg = 1.0 + ((-1 - 2 * q) * gamma * gamma * sigma * sigma) / denominator
    if ask_arg <= 0 or bid_arg <= 0:
        raise ValueError("stationary reservation price is undefined for this inventory")
    ask = mid + math.log(ask_arg) / gamma
    bid = mid + math.log(bid_arg) / gamma
    return bid, ask


@dataclass(frozen=True)
class HJBResult:
    inventories: tuple[int, ...]
    phi: tuple[float, ...]
    bid_distances: tuple[float | None, ...]
    ask_distances: tuple[float | None, ...]
    horizon: float
    steps: int

    def index(self, inventory: int) -> int:
        try:
            return self.inventories.index(inventory)
        except ValueError as exc:
            raise ValueError(f"inventory {inventory} outside HJB grid") from exc

    def quotes(self, mid: float, inventory: int) -> ASQuoteLevels:
        i = self.index(inventory)
        db = self.bid_distances[i]
        da = self.ask_distances[i]
        if db is None or da is None:
            raise ValueError("two-sided quotes are unavailable at the inventory boundary")
        bid = mid - db
        ask = mid + da
        reservation = (bid + ask) / 2.0
        return ASQuoteLevels(
            reservation_price=reservation,
            bid=bid,
            ask=ask,
            total_spread=db + da,
            inventory_skew=mid - reservation,
        )


def solve_finite_horizon_hjb(
    params: ASModelParams,
    *,
    max_inventory: int,
    steps: int = 2_000,
) -> HJBResult:
    """Numerically solve the exact bounded-inventory AS control ODE.

    The exponential-utility Brownian-price ansatz removes the price dimension.
    We integrate the resulting inventory-state ODE forward in time-to-maturity
    with RK4. At ``+/-max_inventory`` the order that would increase absolute
    inventory is disabled.
    """
    if max_inventory < 1:
        raise ValueError("max_inventory must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    inventories = tuple(range(-max_inventory, max_inventory + 1))
    n = len(inventories)
    dt = params.horizon / steps
    phi = [0.0] * n
    c = math.log1p(params.gamma / params.kappa) / params.gamma
    h_scale = (
        params.arrival_rate
        * params.gamma
        / (params.kappa + params.gamma)
        * math.exp(-params.kappa * c)
    )

    def derivative(values: list[float]) -> list[float]:
        out: list[float] = []
        for i, q in enumerate(inventories):
            rewards = 0.0
            if i < n - 1:  # bid fill moves q -> q + 1
                rewards += h_scale * math.exp(
                    params.kappa * (values[i + 1] - values[i])
                )
            if i > 0:  # ask fill moves q -> q - 1
                rewards += h_scale * math.exp(
                    params.kappa * (values[i - 1] - values[i])
                )
            out.append(
                -0.5 * params.gamma * params.sigma * params.sigma * q * q
                + rewards / params.gamma
            )
        return out

    for _ in range(steps):
        k1 = derivative(phi)
        p2 = [x + 0.5 * dt * k for x, k in zip(phi, k1)]
        k2 = derivative(p2)
        p3 = [x + 0.5 * dt * k for x, k in zip(phi, k2)]
        k3 = derivative(p3)
        p4 = [x + dt * k for x, k in zip(phi, k3)]
        k4 = derivative(p4)
        phi = [
            x + dt * (a + 2 * b + 2 * c4 + d) / 6.0
            for x, a, b, c4, d in zip(phi, k1, k2, k3, k4)
        ]

    bid_distances: list[float | None] = []
    ask_distances: list[float | None] = []
    for i in range(n):
        bid_distances.append(c + phi[i] - phi[i + 1] if i < n - 1 else None)
        ask_distances.append(c + phi[i] - phi[i - 1] if i > 0 else None)
    return HJBResult(
        inventories=inventories,
        phi=tuple(phi),
        bid_distances=tuple(bid_distances),
        ask_distances=tuple(ask_distances),
        horizon=params.horizon,
        steps=steps,
    )
