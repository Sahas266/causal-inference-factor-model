"""Simple baseline strategies for the search.

Goal: find something that beats buy-and-hold BTC over the full sample
WITH realistic transaction costs. The CPCM driver pipeline lost money in
the full-period A/B/C test; the hypothesis here is that a much simpler
regime-gated long-only allocator might capture the bull-market upside
while sitting in cash during the 2022 drawdown.

All strategies implement the same minimal interface:

    def run(self, returns: pd.DataFrame, macro: pd.DataFrame, dates) -> Result

Where Result is the same BacktestResult dataclass used elsewhere, plus
optional regime telemetry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from causal_portfolio.backtest.costs import cost_drag_return
from causal_portfolio.backtest.engine import BacktestResult
from causal_portfolio.backtest.metrics import (
    average_turnover,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from causal_portfolio.backtest.threshold import should_rebalance
from causal_portfolio.regimes.hmm import (
    RegimeClassifier,
    build_regime_features,
)

logger = logging.getLogger("cpcm.backtest.strategies")


# ── helpers ──────────────────────────────────────────────────────────


def _walk(
    returns: pd.DataFrame,
    weight_fn,
    *,
    initial_weights: np.ndarray | None = None,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    threshold_l1: float = 0.0,
    start_idx: int = 0,
) -> BacktestResult:
    """Generic walk-forward loop with costs and no-trade band.

    weight_fn(t) -> target_weights at time t. Returns NaN-prefixed where
    the strategy has no opinion yet.
    """
    T, n_assets = returns.shape
    R = returns.values
    portfolio_returns = np.zeros(T - start_idx)
    weights_history = np.zeros((T - start_idx, n_assets))
    rebalance_dates: list = []
    current_w = (initial_weights if initial_weights is not None
                 else np.zeros(n_assets))

    for t in range(start_idx, T):
        idx = t - start_idx
        # Day's mark-to-market on existing weights
        day_r = R[t]
        valid = ~np.isnan(day_r)
        gross_r = float(np.nansum(current_w[valid] * day_r[valid])) if valid.any() else 0.0

        # Get target weights for next period (decision uses info up to t)
        target = weight_fn(t)
        if target is None or np.isnan(target).any():
            # Strategy abstains — hold previous weights
            weights_history[idx] = current_w
            portfolio_returns[idx] = gross_r
            continue

        # Decide whether to rebalance
        if should_rebalance(current_w, target, threshold_l1):
            equity_proxy = 1.0  # we track returns; equity is normalized
            drag = cost_drag_return(
                current_w, target, equity_proxy,
                fee_bps=fee_bps, slippage_bps=slippage_bps,
            )
            net_r = gross_r - drag
            current_w = target.copy()
            rebalance_dates.append(returns.index[t])
        else:
            net_r = gross_r

        weights_history[idx] = current_w
        portfolio_returns[idx] = net_r

    port_values = np.cumprod(1 + portfolio_returns)
    return BacktestResult(
        total_return=float(port_values[-1] / port_values[0] - 1),
        sharpe=sharpe_ratio(portfolio_returns),
        sortino=sortino_ratio(portfolio_returns),
        max_dd=max_drawdown(portfolio_returns),
        calmar=calmar_ratio(portfolio_returns),
        avg_turnover=average_turnover(weights_history),
        coherence_score=0.0,
        returns_series=portfolio_returns,
        weights_history=weights_history,
        rebalance_dates=rebalance_dates,
    )


# ── baselines ────────────────────────────────────────────────────────


def buy_and_hold(
    returns: pd.DataFrame, asset: str = "btc_return",
    fee_bps: float = 5.0, slippage_bps: float = 5.0,
) -> BacktestResult:
    """100% in one asset, never rebalance. Pays cost on the initial buy."""
    if asset not in returns.columns:
        raise KeyError(f"{asset!r} not in returns columns: {list(returns.columns)}")
    target = np.zeros(returns.shape[1])
    target[returns.columns.get_loc(asset)] = 1.0

    def weight_fn(t):
        return target if t == 0 else target  # constant — no_trade band auto-skips

    return _walk(
        returns, weight_fn, threshold_l1=0.0, fee_bps=fee_bps, slippage_bps=slippage_bps,
        initial_weights=np.zeros(returns.shape[1]),
    )


def equal_weight_basket(
    returns: pd.DataFrame, fee_bps: float = 5.0, slippage_bps: float = 5.0,
    rebalance_freq: int = 21,
) -> BacktestResult:
    """Equal-weighted across all return columns, monthly rebalance."""
    n = returns.shape[1]
    target = np.ones(n) / n

    def weight_fn(t):
        if t % rebalance_freq == 0:
            return target
        return None  # hold

    return _walk(
        returns, weight_fn, fee_bps=fee_bps, slippage_bps=slippage_bps,
        initial_weights=np.zeros(n), threshold_l1=0.0,
    )


# ── regime-gated long-only ───────────────────────────────────────────


@dataclass
class RegimeGatedResult(BacktestResult):
    regime_labels: np.ndarray = field(default_factory=lambda: np.array([]))


def regime_gated_long_only(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    universe_weights: dict[str, float] | None = None,
    hmm_window: int = 504,
    hmm_refit_every: int = 63,
    n_states: int = 2,
    n_restarts: int = 5,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    threshold_l1: float = 0.0,
    stress_state: int = 1,
    stress_allocation: float = 0.0,
) -> RegimeGatedResult:
    """Use HMM to detect regime; allocate long during calm, cash during stress.

    Args:
        returns: (T, n_assets) DataFrame; columns end with "_return".
        macro: (T, n_macro) DataFrame; must include "vixcls".
        universe_weights: dict of `ticker -> target weight` for the calm
            regime, e.g. {"btc": 0.7, "eth": 0.3}. Weights are renormalized
            to sum to 1. Defaults to 100% BTC if BTC is in the universe,
            else equal-weight all assets.
        hmm_window: rolling-window size for HMM refit.
        hmm_refit_every: refit cadence in days.
        n_states: number of regimes for the HMM.
        n_restarts: number of HMM seeds to try at each fit.
        fee_bps, slippage_bps: transaction-cost parameters.
        threshold_l1: skip rebalance if |Δw|_L1 < this. Default 0 = always
            execute; with regime switching, a threshold of 0.5+ effectively
            only trades at regime changes.
        stress_state: which HMM state (after sort-by-VIX-mean relabeling)
            should trigger cash-out. State 1 is high-stress by convention.
        stress_allocation: target weight in the long basket during stress.
            0.0 = full cash. 0.3 = partial de-risk to 30% long.

    Returns:
        RegimeGatedResult with regime_labels populated.
    """
    # Resolve calm-state target weights
    n_assets = returns.shape[1]
    asset_names = [c.replace("_return", "") for c in returns.columns]
    if universe_weights is None:
        if "btc" in asset_names:
            universe_weights = {"btc": 1.0}
        else:
            universe_weights = {a: 1.0 / n_assets for a in asset_names}

    calm_target = np.zeros(n_assets)
    total = float(sum(universe_weights.values()))
    if total <= 0:
        raise ValueError("universe_weights must sum to > 0")
    for ticker, w in universe_weights.items():
        col = f"{ticker}_return"
        if col not in returns.columns:
            logger.warning("Skipping %s — not in returns columns", ticker)
            continue
        calm_target[returns.columns.get_loc(col)] = w / total

    stress_target = calm_target * stress_allocation

    # Build regime features (uses VIX + BTC realized vol — 21d warmup)
    feats_full = build_regime_features(macro, returns)
    common = returns.index.intersection(feats_full.index)
    R = returns.loc[common]
    feats = feats_full.loc[common]

    T = len(R)
    start_idx = max(hmm_window, 0)
    if T <= start_idx + 30:
        raise ValueError(f"Need T > {start_idx + 30}, got T={T}")

    regime_labels = np.full(T, -1, dtype=int)
    classifier: RegimeClassifier | None = None
    last_refit_t = -np.inf

    def weight_fn(t: int):
        nonlocal classifier, last_refit_t
        if t < start_idx:
            return None
        if classifier is None or (t - last_refit_t) >= hmm_refit_every:
            window = feats.iloc[t - hmm_window : t]
            classifier = RegimeClassifier(
                n_states=n_states, n_restarts=n_restarts,
            ).fit(window)
            last_refit_t = t

        window = feats.iloc[t - hmm_window + 1 : t + 1]
        post = classifier.forward_filter(window)[-1]
        label = int(post.argmax())
        regime_labels[t] = label
        return stress_target if label == stress_state else calm_target

    base = _walk(
        R, weight_fn,
        initial_weights=np.zeros(n_assets),
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        threshold_l1=threshold_l1, start_idx=start_idx,
    )
    return RegimeGatedResult(
        total_return=base.total_return, sharpe=base.sharpe,
        sortino=base.sortino, max_dd=base.max_dd, calmar=base.calmar,
        avg_turnover=base.avg_turnover, coherence_score=base.coherence_score,
        returns_series=base.returns_series,
        weights_history=base.weights_history,
        rebalance_dates=base.rebalance_dates,
        regime_labels=regime_labels[start_idx:],
    )
