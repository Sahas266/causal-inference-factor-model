"""Stable estimation of the Easley-Kiefer-O'Hara-Paperman PIN model.

This module implements the common symmetric-uninformed-intensity variant:
uninformed buy and sell arrivals share one epsilon parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp


@dataclass(frozen=True)
class PINFit:
    alpha: float
    delta: float
    mu: float
    epsilon: float
    pin: float
    log_likelihood: float
    sample_days: int
    converged: bool
    boundary_solution: bool
    hessian_condition: float | None
    standard_errors: tuple[float, float, float, float] | None
    starts: int
    message: str

    @property
    def usable(self) -> bool:
        return (
            self.converged
            and not self.boundary_solution
            and self.sample_days >= 20
            and self.hessian_condition is not None
            and self.hessian_condition < 1e12
            and self.standard_errors is not None
            and math.isfinite(self.pin)
            and 0.0 <= self.pin <= 1.0
        )


@dataclass(frozen=True)
class PINPosterior:
    no_news: float
    informed_buy: float
    informed_sell: float

    @property
    def informed(self) -> float:
        return self.informed_buy + self.informed_sell


def _counts(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        raise ValueError(f"{name} must contain finite non-negative integer counts")
    return arr


def _validated_pair(
    buys: Sequence[int] | np.ndarray, sells: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    b = _counts(buys, "buys")
    s = _counts(sells, "sells")
    if b.shape != s.shape:
        raise ValueError("buys and sells must have the same length")
    return b, s


def _unpack(theta: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(expit(theta[0])),
        float(expit(theta[1])),
        float(np.exp(theta[2])),
        float(np.exp(theta[3])),
    )


def _poisson_logpmf(count: np.ndarray, rate: float) -> np.ndarray:
    return count * math.log(rate) - rate - gammaln(count + 1.0)


def _component_logs(
    buys: np.ndarray,
    sells: np.ndarray,
    alpha: float,
    delta: float,
    mu: float,
    epsilon: float,
) -> np.ndarray:
    tiny = np.finfo(float).tiny
    no_news = (
        math.log(max(1.0 - alpha, tiny))
        + _poisson_logpmf(buys, epsilon)
        + _poisson_logpmf(sells, epsilon)
    )
    good_news = (
        math.log(max(alpha * (1.0 - delta), tiny))
        + _poisson_logpmf(buys, epsilon + mu)
        + _poisson_logpmf(sells, epsilon)
    )
    bad_news = (
        math.log(max(alpha * delta, tiny))
        + _poisson_logpmf(buys, epsilon)
        + _poisson_logpmf(sells, epsilon + mu)
    )
    return np.column_stack((no_news, good_news, bad_news))


def pin_log_likelihood(
    buys: Sequence[int] | np.ndarray,
    sells: Sequence[int] | np.ndarray,
    *,
    alpha: float,
    delta: float,
    mu: float,
    epsilon: float,
) -> float:
    """Evaluate the log-sum-exp three-state Poisson-mixture likelihood."""
    b, s = _validated_pair(buys, sells)
    if not (0 < alpha < 1 and 0 < delta < 1 and mu > 0 and epsilon > 0):
        raise ValueError("alpha/delta must be in (0,1); rates must be > 0")
    return float(logsumexp(_component_logs(b, s, alpha, delta, mu, epsilon), axis=1).sum())


def _numerical_hessian(fun, x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    n = x.size
    hessian = np.empty((n, n), dtype=float)
    f0 = float(fun(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step
        hessian[i, i] = (fun(x + ei) - 2.0 * f0 + fun(x - ei)) / (step * step)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step
            value = (
                fun(x + ei + ej)
                - fun(x + ei - ej)
                - fun(x - ei + ej)
                + fun(x - ei - ej)
            ) / (4.0 * step * step)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def fit_pin(
    buys: Sequence[int] | np.ndarray,
    sells: Sequence[int] | np.ndarray,
    *,
    min_days: int = 20,
    n_starts: int = 16,
    seed: int = 0,
) -> PINFit:
    """Fit PIN by deterministic multi-start L-BFGS-B in transformed space."""
    b, s = _validated_pair(buys, sells)
    if b.size < min_days:
        raise ValueError(f"PIN requires at least {min_days} days, got {b.size}")
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")

    def objective(theta: np.ndarray) -> float:
        alpha, delta, mu, epsilon = _unpack(theta)
        logs = _component_logs(b, s, alpha, delta, mu, epsilon)
        value = -float(logsumexp(logs, axis=1).sum())
        return value if math.isfinite(value) else 1e300

    mean_flow = max(float(np.mean(np.concatenate((b, s)))), 1e-3)
    imbalance = max(float(np.mean(np.abs(b - s))), 1e-3)
    starts = [
        np.array([0.0, 0.0, math.log(imbalance), math.log(mean_flow * 0.75)]),
        np.array([-1.4, 0.0, math.log(mean_flow), math.log(mean_flow * 0.5)]),
        np.array([1.4, 0.0, math.log(imbalance * 2), math.log(mean_flow * 0.8)]),
    ]
    rng = np.random.default_rng(seed)
    while len(starts) < n_starts:
        starts.append(
            np.array(
                [
                    rng.uniform(-2.5, 1.5),
                    rng.uniform(-1.5, 1.5),
                    math.log(mean_flow) + rng.normal(0, 1.0),
                    math.log(mean_flow) + rng.normal(-0.3, 0.7),
                ]
            )
        )
    bounds = [(-9.0, 9.0), (-9.0, 9.0), (-12.0, 20.0), (-12.0, 20.0)]
    results = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        for start in starts[:n_starts]
    ]
    finite_results = [result for result in results if math.isfinite(float(result.fun))]
    if not finite_results:
        raise RuntimeError("PIN optimization produced no finite solution")
    best = min(finite_results, key=lambda result: float(result.fun))
    alpha, delta, mu, epsilon = _unpack(np.asarray(best.x, dtype=float))
    pin = alpha * mu / (alpha * mu + 2.0 * epsilon)
    boundary = any(
        abs(float(x) - lo) < 1e-3 or abs(float(x) - hi) < 1e-3
        for x, (lo, hi) in zip(best.x, bounds)
    ) or min(alpha, 1 - alpha, delta, 1 - delta) < 2e-4

    condition: float | None = None
    standard_errors: tuple[float, float, float, float] | None = None
    try:
        hessian = _numerical_hessian(objective, np.asarray(best.x, dtype=float))
        condition_value = float(np.linalg.cond(hessian))
        condition = condition_value if math.isfinite(condition_value) else None
        covariance = np.linalg.inv(hessian)
        transformed_se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        jacobian = np.array(
            [alpha * (1 - alpha), delta * (1 - delta), mu, epsilon]
        )
        standard_errors = tuple(float(x) for x in transformed_se * jacobian)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass
    return PINFit(
        alpha=alpha,
        delta=delta,
        mu=mu,
        epsilon=epsilon,
        pin=pin,
        log_likelihood=-float(best.fun),
        sample_days=int(b.size),
        converged=bool(best.success),
        boundary_solution=boundary,
        hessian_condition=condition,
        standard_errors=standard_errors,
        starts=n_starts,
        message=str(best.message),
    )


def pin_posteriors(
    buys: Sequence[int] | np.ndarray,
    sells: Sequence[int] | np.ndarray,
    fit: PINFit,
) -> list[PINPosterior]:
    """Posterior probabilities of no-news, informed-buy, informed-sell days."""
    b, s = _validated_pair(buys, sells)
    logs = _component_logs(b, s, fit.alpha, fit.delta, fit.mu, fit.epsilon)
    probabilities = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
    return [PINPosterior(*map(float, row)) for row in probabilities]


def rolling_pin(
    buys: Sequence[int] | np.ndarray,
    sells: Sequence[int] | np.ndarray,
    *,
    window: int = 60,
    n_starts: int = 8,
    seed: int = 0,
) -> list[PINFit | None]:
    """Walk-forward PIN estimates; each output uses data through that day."""
    b, s = _validated_pair(buys, sells)
    if window < 20:
        raise ValueError("window must be >= 20")
    output: list[PINFit | None] = [None] * len(b)
    for end in range(window, len(b) + 1):
        output[end - 1] = fit_pin(
            b[end - window : end],
            s[end - window : end],
            min_days=window,
            n_starts=n_starts,
            seed=seed + end,
        )
    return output


def simulate_pin_counts(
    days: int,
    *,
    alpha: float,
    delta: float,
    mu: float,
    epsilon: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic daily buy/sell counts for estimator recovery tests."""
    if days < 1 or not (0 < alpha < 1 and 0 < delta < 1 and mu > 0 and epsilon > 0):
        raise ValueError("invalid PIN simulation parameters")
    rng = np.random.default_rng(seed)
    event = rng.random(days) < alpha
    bad = rng.random(days) < delta
    buy_rates = np.full(days, epsilon, dtype=float)
    sell_rates = np.full(days, epsilon, dtype=float)
    buy_rates[event & ~bad] += mu
    sell_rates[event & bad] += mu
    return rng.poisson(buy_rates), rng.poisson(sell_rates)
