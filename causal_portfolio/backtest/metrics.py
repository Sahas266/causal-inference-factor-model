"""Standard portfolio risk metrics."""

import numpy as np
import pandas as pd

# Crypto trades 24/7, so a year is 365 daily observations (not 252 trading
# days). Single source of truth for every Sharpe/Sortino/Calmar in the repo.
ANNUALIZATION = 365


def sharpe_ratio(
    returns: np.ndarray | pd.Series,
    rf: float = 0.0,
    annualization: int = ANNUALIZATION,
) -> float:
    """Annualized Sharpe ratio."""
    excess = np.asarray(returns) - rf / annualization
    if len(excess) < 2 or np.std(excess) < 1e-15:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(annualization))


def sortino_ratio(
    returns: np.ndarray | pd.Series,
    rf: float = 0.0,
    annualization: int = ANNUALIZATION,
) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    excess = np.asarray(returns) - rf / annualization
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-15:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(annualization))


def max_drawdown(returns: np.ndarray | pd.Series) -> float:
    """Maximum drawdown (negative value)."""
    cumulative = np.cumprod(1 + np.asarray(returns))
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(np.min(drawdown))


def calmar_ratio(
    returns: np.ndarray | pd.Series,
    annualization: int = ANNUALIZATION,
) -> float:
    """Calmar ratio = annualized return / |max drawdown|."""
    mdd = max_drawdown(returns)
    if abs(mdd) < 1e-15:
        return 0.0
    ann_return = np.mean(returns) * annualization
    return float(ann_return / abs(mdd))


def average_turnover(weights_history: np.ndarray) -> float:
    """Average daily turnover from weight changes.

    Args:
        weights_history: (T, n_assets) weight matrix.

    Returns:
        Mean absolute weight change per period.
    """
    if len(weights_history) < 2:
        return 0.0
    diffs = np.abs(np.diff(weights_history, axis=0))
    return float(np.mean(np.sum(diffs, axis=1)))
