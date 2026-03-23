"""Martingale defect diagnostic for CPCM structural coherence.

If the model is well-specified, the discounted portfolio value process
should be a martingale under the risk-neutral measure. The defect
measures deviation from this condition.

Reference: Section 5.6 of arXiv:2509.09585v2
"""

import numpy as np

from causal_portfolio.solvers.base import CPCMSolver


def martingale_defect(
    portfolio_values: np.ndarray,
    filtered_states: np.ndarray,
    solver: CPCMSolver,
) -> np.ndarray:
    """Compute martingale defect time series.

    defect_t = E[V_{t+1} | F_t] - V_t

    where E[V_{t+1} | F_t] is approximated using the solver's predicted
    returns applied to the current portfolio value.

    Args:
        portfolio_values: (T,) cumulative portfolio values.
        filtered_states: (T, m) filtered driver states.
        solver: Fitted CPCM solver.

    Returns:
        (T-1,) defect series (should be near zero if well-specified).
    """
    T = len(portfolio_values)
    defects = np.zeros(T - 1)

    for t in range(T - 1):
        predicted_return = solver.predict(filtered_states[t]).mean()
        expected_value = portfolio_values[t] * (1 + predicted_return)
        defects[t] = expected_value - portfolio_values[t + 1]

    return defects


def structural_coherence_score(defects: np.ndarray) -> float:
    """Summary statistic for martingale defect.

    Returns mean absolute defect — should be < 0.01 for good models.
    """
    return float(np.mean(np.abs(defects)))
