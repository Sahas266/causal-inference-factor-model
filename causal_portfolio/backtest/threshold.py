"""No-trade-band logic — skip rebalances when the signal change is small.

The model produces a new target weight vector each rebalance period. In
practice, two consecutive targets often differ by a small amount that's
mostly noise. Trading on that noise burns fees without improving Sharpe.

Solution: a threshold on the L1 distance between current and target weights.
Below it, hold the existing weights. Above it, execute the full rebalance.

Two related approaches considered but **not** implemented here:

- **Partial rebalance** (move part of the way toward target): smoother but
  introduces a hyperparameter (the partial fraction) and makes the strategy
  state-dependent in ways that hurt interpretability.
- **Hysteresis** (require larger ΔL1 to enter than to maintain): cleaner
  in regime-switching contexts but adds two thresholds to tune.

For now we stick with a single L1 threshold. The strategy search code can
sweep it.
"""

from __future__ import annotations

import numpy as np


def should_rebalance(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    threshold_l1: float = 0.0,
) -> bool:
    """Return True if the trade should fire, False if we hold.

    Args:
        current_weights: shape (n_assets,)
        target_weights:  shape (n_assets,)
        threshold_l1:    minimum L1 distance to justify a trade.
                          0.0 means always rebalance (no skipping).
                          0.10 means require ≥10% total weight reallocation.

    Returns:
        bool — True to execute the rebalance, False to skip.
    """
    if threshold_l1 <= 0:
        return True
    if current_weights.shape != target_weights.shape:
        raise ValueError(
            f"shape mismatch: current={current_weights.shape} "
            f"target={target_weights.shape}"
        )
    delta_l1 = float(np.abs(target_weights - current_weights).sum())
    return delta_l1 >= threshold_l1
