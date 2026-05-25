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


def _walk_with_stop_loss(
    returns: pd.DataFrame, weight_fn, *, initial_weights, fee_bps, slippage_bps,
    threshold_l1, start_idx, stop_loss_pct: float, stop_reset_pct: float, n_assets: int,
) -> BacktestResult:
    """Like _walk, but with a trailing-stop overlay.

    Tracks running equity = cumulative product of net returns. Whenever
    drawdown from the running peak exceeds `stop_loss_pct`, forces the
    target weights to zero (full cash) until the drawdown recovers below
    `stop_reset_pct`.
    """
    T = len(returns)
    R = returns.values
    portfolio_returns = np.zeros(T - start_idx)
    weights_history = np.zeros((T - start_idx, n_assets))
    rebalance_dates: list = []
    current_w = initial_weights.copy()

    equity = 1.0
    peak = 1.0
    stop_active = False

    for t in range(start_idx, T):
        idx = t - start_idx
        day_r = R[t]
        valid = ~np.isnan(day_r)
        gross_r = float(np.nansum(current_w[valid] * day_r[valid])) if valid.any() else 0.0

        # Compute target from regime
        regime_target = weight_fn(t)
        if regime_target is None or np.isnan(regime_target).any():
            target = current_w
        else:
            target = regime_target

        # Apply stop-loss overlay BEFORE the rebalance decision
        drawdown = 1.0 - (equity / peak) if peak > 0 else 0.0
        if not stop_active and drawdown >= stop_loss_pct:
            stop_active = True
        elif stop_active and drawdown <= stop_reset_pct:
            stop_active = False
        if stop_active:
            target = np.zeros(n_assets)

        # Rebalance decision
        if should_rebalance(current_w, target, threshold_l1):
            drag = cost_drag_return(
                current_w, target, equity_usd=1.0,
                fee_bps=fee_bps, slippage_bps=slippage_bps,
            )
            net_r = gross_r - drag
            current_w = target.copy()
            rebalance_dates.append(returns.index[t])
        else:
            net_r = gross_r

        weights_history[idx] = current_w
        portfolio_returns[idx] = net_r

        # Update equity tracker
        equity *= (1.0 + net_r)
        if equity > peak:
            peak = equity

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
    min_posterior_to_switch: float = 0.0,
    stop_loss_pct: float | None = None,
    stop_reset_pct: float | None = None,
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
        min_posterior_to_switch: confidence gate. If proposing to switch
            regime, require the new regime's posterior probability ≥ this.
            Otherwise stay in current regime. 0.0 = always switch on argmax
            (default). Typical values: 0.55–0.85. Reduces flapping at
            boundary days where the posterior is near 50/50.
        stop_loss_pct: optional trailing-stop overlay. If equity drops more
            than this fraction below its running peak, force cash regardless
            of regime label. e.g. 0.25 = stop out at -25% drawdown from peak.
            None disables.
        stop_reset_pct: when stop is active, resume regime-based allocation
            once equity recovers within this fraction of peak. e.g. 0.10
            means resume when current_dd ≤ 10%. Defaults to half of
            stop_loss_pct (more conservative re-entry than exit). Ignored
            if stop_loss_pct is None.

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
    current_regime: int | None = None  # for posterior-gating hysteresis

    # Stop-loss defaults: half the trigger for re-entry (more conservative)
    stop_reset = stop_reset_pct if stop_reset_pct is not None else (
        (stop_loss_pct or 0.0) * 0.5
    )

    def regime_target_fn(t: int):
        """Return regime-target weights at time t, ignoring stop-loss.

        Updates current_regime via posterior gating. Stop-loss is applied
        inside the walk loop where running equity is known.
        """
        nonlocal classifier, last_refit_t, current_regime
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
        proposed_label = int(post.argmax())

        # Posterior-confidence gate: only switch if confident
        if current_regime is None:
            current_regime = proposed_label
        elif proposed_label != current_regime:
            if min_posterior_to_switch > 0 and post[proposed_label] < min_posterior_to_switch:
                pass  # not confident enough, hold previous regime
            else:
                current_regime = proposed_label
        regime_labels[t] = current_regime
        return stress_target if current_regime == stress_state else calm_target

    if stop_loss_pct is None:
        base = _walk(
            R, regime_target_fn,
            initial_weights=np.zeros(n_assets),
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            threshold_l1=threshold_l1, start_idx=start_idx,
        )
    else:
        base = _walk_with_stop_loss(
            R, regime_target_fn,
            initial_weights=np.zeros(n_assets),
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            threshold_l1=threshold_l1, start_idx=start_idx,
            stop_loss_pct=stop_loss_pct, stop_reset_pct=stop_reset,
            n_assets=n_assets,
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
