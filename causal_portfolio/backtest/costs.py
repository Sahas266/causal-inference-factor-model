"""Transaction-cost model for the backtester.

Pure functions — no state. Used inside walk-forward loops to deduct realistic
costs from daily portfolio returns.

Cost components:

1. **Fixed fee** (`fee_bps`): linear in traded notional. Default 5 bps which
   matches Hyperliquid perp taker fees as of late 2025.

2. **Slippage** (`slippage_bps`): an additional linear-in-notional cost that
   models the gap between mid and the realized fill price. Default 5 bps for
   the size buckets we typically trade.

3. **No-trade band** is handled separately by `should_rebalance()` in
   `threshold.py` — when the proposed weight change is below threshold we
   skip the trade entirely (zero cost).

Cost is applied to **traded notional**, which is `equity * |Δw|_L1 / 2`
because every reallocation involves a sell *and* a buy that sum to |Δw|
(half of which is the sell side, half the buy side — but you pay fees on
both, so the factor is 1, not 1/2). Standard practice in equity backtest
literature.
"""

from __future__ import annotations

import numpy as np


def trade_cost_usd(
    prev_weights: np.ndarray,
    new_weights: np.ndarray,
    equity_usd: float,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> float:
    """Cost in USD of moving from prev_weights to new_weights.

    Args:
        prev_weights: shape (n_assets,)
        new_weights:  shape (n_assets,)
        equity_usd:   account equity in dollars at trade time
        fee_bps:      exchange fee in basis points (HL taker = 5 bps)
        slippage_bps: average execution gap from mid in basis points

    Returns:
        Total USD cost of the rebalance. Zero if weights unchanged.
    """
    if prev_weights.shape != new_weights.shape:
        raise ValueError(
            f"shape mismatch: prev={prev_weights.shape} new={new_weights.shape}"
        )
    delta_l1 = float(np.abs(new_weights - prev_weights).sum())
    if delta_l1 < 1e-12:
        return 0.0
    bps_total = fee_bps + slippage_bps
    # |Δw|_L1 is the total amount of weight that needs to move. Each unit
    # of weight movement corresponds to `equity` USD of notional traded
    # (you sell `x` somewhere and buy `x` elsewhere; you pay fees on each
    # leg, but they're already a round-trip in |Δw| accounting).
    notional_traded = delta_l1 * equity_usd
    return notional_traded * (bps_total / 10_000.0)


def cost_drag_return(
    prev_weights: np.ndarray,
    new_weights: np.ndarray,
    equity_usd: float,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> float:
    """Cost expressed as a return drag (cost_usd / equity_usd).

    Subtract from the day's portfolio return to apply.
    """
    if equity_usd <= 0:
        return 0.0
    return trade_cost_usd(
        prev_weights, new_weights, equity_usd, fee_bps, slippage_bps,
    ) / equity_usd
