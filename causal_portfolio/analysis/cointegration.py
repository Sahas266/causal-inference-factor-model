"""Cointegration analysis + market-neutral spread signals (stat-arb).

Why this matters for the project: every directional strategy here loses to
buy-and-hold BTC out of sample. A cointegration-based spread trade is *market
neutral* — it bets on a stationary linear combination of two assets reverting,
not on the market going up — so it has a genuine shot at an OOS edge that does
NOT require beating BTC's drift. This module supplies the primitives; the
walk-forward backtest lives in `experiments/statarb.py`.

Methods:
  - Engle-Granger (pairwise): regress log-price A on log-price B, ADF-test the
    residual for stationarity. If stationary, the residual is the tradeable
    spread; the regression slope is the hedge ratio.
  - Johansen (basket): rank test for the number of cointegrating vectors among
    >=2 log-price series (for multi-asset baskets).

Trading a spread: standardize the in-sample residual to a z-score, then fade
deviations — short the spread when z is high, long when z is low, flatten near
zero (a band with separate entry/exit thresholds to limit churn).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger("cpcm.analysis.cointegration")


@dataclass
class CointPair:
    a: str            # dependent asset (long leg, 1 unit)
    b: str            # independent asset (short leg, beta units)
    beta: float       # hedge ratio (slope of log_a ~ log_b)
    const: float      # regression intercept
    mu: float         # in-sample spread mean
    sigma: float      # in-sample spread std
    adf_p: float      # ADF p-value on the residual (lower = more stationary)


def _ols_hedge(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float]:
    """Slope + intercept of log_a = const + beta * log_b (least squares)."""
    X = np.column_stack([np.ones_like(log_b), log_b])
    coef, *_ = np.linalg.lstsq(X, log_a, rcond=None)
    return float(coef[1]), float(coef[0])


def spread_series(log_a: np.ndarray, log_b: np.ndarray, beta: float, const: float) -> np.ndarray:
    """Residual spread = log_a - (const + beta*log_b)."""
    return log_a - (const + beta * log_b)


def find_cointegrated_pairs(
    log_prices: pd.DataFrame, *, adf_pvalue: float = 0.05,
    min_obs: int = 120,
) -> list[CointPair]:
    """Engle-Granger over all asset pairs; keep stationary-residual pairs.

    `log_prices` is a (T, n_assets) frame of LOG prices on the (training)
    window. For each ordered pair we fit a hedge ratio and ADF-test the
    residual; pairs with ADF p < `adf_pvalue` are returned, best (lowest ADF p)
    first.
    """
    cols = list(log_prices.columns)
    clean = log_prices.dropna()
    if len(clean) < min_obs:
        return []
    pairs: list[CointPair] = []
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                continue
            a, b = cols[i], cols[j]
            la, lb = clean[a].values, clean[b].values
            beta, const = _ols_hedge(la, lb)
            if not np.isfinite(beta) or abs(beta) < 1e-6:
                continue
            resid = spread_series(la, lb, beta, const)
            sigma = float(resid.std())
            if sigma < 1e-9:
                continue
            try:
                adf_p = float(adfuller(resid, autolag="AIC")[1])
            except Exception:
                continue
            if adf_p < adf_pvalue:
                pairs.append(CointPair(
                    a=a, b=b, beta=beta, const=const,
                    mu=float(resid.mean()), sigma=sigma, adf_p=adf_p,
                ))
    # Deduplicate symmetric pairs: keep the orientation with the lower ADF p.
    best: dict[frozenset, CointPair] = {}
    for p in pairs:
        key = frozenset((p.a, p.b))
        if key not in best or p.adf_p < best[key].adf_p:
            best[key] = p
    return sorted(best.values(), key=lambda p: p.adf_p)


def spread_positions(
    z: np.ndarray, *, entry: float = 1.5, exit: float = 0.5,
) -> np.ndarray:
    """Banded mean-reversion positions from a spread z-score series.

    Position in {-1, 0, +1}: enter SHORT spread (-1) when z >= +entry, enter
    LONG spread (+1) when z <= -entry, flatten when |z| <= exit, else hold the
    previous position. The hysteresis (entry > exit) limits churn.
    """
    pos = np.zeros(len(z))
    cur = 0.0
    for t in range(len(z)):
        zt = z[t]
        if np.isnan(zt):
            pos[t] = cur
            continue
        if cur == 0.0:
            if zt >= entry:
                cur = -1.0
            elif zt <= -entry:
                cur = 1.0
        else:
            if abs(zt) <= exit:
                cur = 0.0
        pos[t] = cur
    return pos


def johansen_rank(log_prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Johansen trace-test cointegration rank for a basket of log-price series.

    Returns the estimated number of cointegrating vectors at the 95% level and
    the leading cointegrating vector (eigenvector). For multi-asset baskets.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    clean = log_prices.dropna()
    res = coint_johansen(clean.values, det_order, k_ar_diff)
    # trace stat vs 95% critical value (column index 1)
    rank = int(np.sum(res.lr1 > res.cvt[:, 1]))
    return {
        "rank": rank,
        "trace_stats": res.lr1.tolist(),
        "crit_95": res.cvt[:, 1].tolist(),
        "cointegrating_vector": res.evec[:, 0].tolist(),
        "assets": list(log_prices.columns),
    }
