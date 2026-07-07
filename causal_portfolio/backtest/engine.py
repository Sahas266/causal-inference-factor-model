"""Walk-forward backtest engine for CPCM strategies.

Implements a rolling-window backtest:
1. At each rebalance date: fit solver on training window, run EKF filter,
   compute optimal weights via ManifoldOptimizer.
2. Between rebalances: hold constant weights, compute portfolio returns.
3. Compute aggregate metrics over the full test period.

Reference: Section 5 of arXiv:2509.09585v2
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from causal_portfolio.backtest.costs import cost_drag_return
from causal_portfolio.backtest.metrics import (
    average_turnover,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from causal_portfolio.filters.ekf import CPCMKalmanFilter
from causal_portfolio.optimizer.diagnostics import (
    martingale_defect,
    structural_coherence_score,
)
from causal_portfolio.optimizer.manifold import ManifoldOptimizer, estimate_covariance
from causal_portfolio.solvers.base import CPCMSolver

logger = logging.getLogger("cpcm.backtest")


@dataclass
class BacktestResult:
    """Results from a CPCM backtest run."""

    total_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd: float = 0.0
    calmar: float = 0.0
    avg_turnover: float = 0.0
    coherence_score: float = 0.0
    returns_series: np.ndarray = field(default_factory=lambda: np.array([]))
    weights_history: np.ndarray = field(default_factory=lambda: np.array([]))
    rebalance_dates: list = field(default_factory=list)
    n_failed_rebalances: int = 0

    def summary(self) -> str:
        # Coherence is a full-test-period EKF diagnostic (fit in-sample on the
        # test window), NOT an out-of-sample metric — labeled accordingly.
        return (
            f"Sharpe: {self.sharpe:.3f} | Sortino: {self.sortino:.3f} | "
            f"MaxDD: {self.max_dd:.1%} | Calmar: {self.calmar:.3f} | "
            f"Turnover: {self.avg_turnover:.3f} | "
            f"Coherence(in-sample diag): {self.coherence_score:.4f}"
        )

    @classmethod
    def from_returns(
        cls,
        portfolio_returns: np.ndarray,
        weights_history: np.ndarray,
        rebalance_dates: list,
        coherence_score: float = 0.0,
        **extra,
    ):
        """Build a result (metrics included) from a realized return path.

        `extra` forwards subclass-only fields (e.g. regime_labels) so
        subclasses can call `RegimeBacktestResult.from_returns(...)` directly.
        """
        port_values = np.cumprod(1 + portfolio_returns)
        return cls(
            # port_values[0] is already 1 + r[0]; dividing by it would drop
            # the first day's return. Start value is 1.0 by construction.
            total_return=float(port_values[-1] - 1.0),
            sharpe=sharpe_ratio(portfolio_returns),
            sortino=sortino_ratio(portfolio_returns),
            max_dd=max_drawdown(portfolio_returns),
            calmar=calmar_ratio(portfolio_returns),
            avg_turnover=average_turnover(weights_history),
            coherence_score=coherence_score,
            returns_series=portfolio_returns,
            weights_history=weights_history,
            rebalance_dates=rebalance_dates,
            **extra,
        )


class CPCMBacktester:
    """Walk-forward backtest engine."""

    def __init__(
        self,
        solver: CPCMSolver,
        optimizer: ManifoldOptimizer,
        use_ekf: bool = True,
        rebalance_freq: int = 5,
        train_window: int = 252,
        fee_bps: float = 5.0,
        slippage_bps: float = 5.0,
        fit_kwargs: dict | None = None,
    ):
        """
        Args:
            solver: CPCM solver instance (will be re-fitted at each rebalance).
            optimizer: ManifoldOptimizer for weight computation.
            use_ekf: Whether to apply EKF filtering to drivers.
            rebalance_freq: Rebalance every N trading days.
            train_window: Rolling training window size.
            fee_bps / slippage_bps: transaction costs charged on L1 weight
                turnover at every rebalance (same model as costs.py, so CPCM
                results are cost-comparable to the strategies.py baselines).
            fit_kwargs: extra kwargs forwarded to every solver.fit() call
                (e.g. V4 epochs / lr / jacobian lambda).
        """
        self.solver = solver
        self.optimizer = optimizer
        self.use_ekf = use_ekf
        self.rebalance_freq = rebalance_freq
        self.train_window = train_window
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}

    def run(
        self,
        returns: np.ndarray,
        drivers: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            returns: (T, n_assets) daily asset returns.
            drivers: (T, m) daily driver values.
            dates: (T,) date array for logging.

        Returns:
            BacktestResult with all metrics.
        """
        T, n_assets = returns.shape
        m = drivers.shape[1]

        if T <= self.train_window:
            raise ValueError(
                f"Need T > train_window: {T} <= {self.train_window}"
            )

        # Initialize
        portfolio_returns = np.zeros(T - self.train_window)
        weights_history = np.zeros((T - self.train_window, n_assets))
        current_weights = np.ones(n_assets) / n_assets  # equal weight start
        rebalance_dates = []

        test_start = self.train_window

        n_failed = 0
        for t in range(test_start, T):
            idx = t - test_start

            # ── Rebalance check ──
            cost_drag = 0.0
            if idx % self.rebalance_freq == 0:
                try:
                    new_weights = self._rebalance(returns, drivers, t, m)
                    cost_drag = cost_drag_return(
                        current_weights, new_weights, 1.0,
                        fee_bps=self.fee_bps, slippage_bps=self.slippage_bps,
                    )
                    current_weights = new_weights
                    if dates is not None:
                        rebalance_dates.append(dates[t])
                except Exception as e:
                    if idx == 0:
                        # No valid weights were ever computed — trading the
                        # initialized 1/n portfolio would be an undisclosed
                        # equal-weight backtest. Fail loudly instead.
                        raise RuntimeError(
                            f"First rebalance failed at t={t}: {e}"
                        ) from e
                    n_failed += 1
                    logger.warning(f"Rebalance failed at t={t}: {e}")
                    # Keep previous weights

            # ── Compute portfolio return ──
            day_return = returns[t]
            valid = ~np.isnan(day_return)
            if valid.any():
                portfolio_returns[idx] = np.nansum(
                    current_weights[valid] * day_return[valid]
                )
            portfolio_returns[idx] -= cost_drag
            weights_history[idx] = current_weights

        # ── Compute metrics ──
        # Martingale defect (if EKF was used)
        coherence = 0.0
        if self.use_ekf and len(portfolio_returns) > 10:
            try:
                port_values = np.cumprod(1 + portfolio_returns)
                ekf = CPCMKalmanFilter(m=m)
                ekf.fit_dynamics(drivers[test_start:])
                filtered, _ = ekf.filter(drivers[test_start:])
                defects = martingale_defect(port_values, filtered, self.solver)
                coherence = structural_coherence_score(defects)
            except Exception:
                pass

        result = BacktestResult.from_returns(
            portfolio_returns, weights_history, rebalance_dates,
            coherence_score=coherence,
            n_failed_rebalances=n_failed,
        )
        if n_failed:
            logger.warning(
                "%d rebalance(s) failed; stale weights were held through "
                "those periods", n_failed,
            )

        logger.info(f"Backtest complete: {result.summary()}")
        return result

    def _rebalance(
        self,
        returns: np.ndarray,
        drivers: np.ndarray,
        t: int,
        m: int,
    ) -> np.ndarray:
        """Fit solver and compute new weights at time t."""
        # Training window
        train_start = max(0, t - self.train_window)
        D_train = drivers[train_start:t]
        R_train = returns[train_start:t]

        # Fit solver
        self.solver.fit(D_train, R_train, **self.fit_kwargs)

        # Apply EKF filtering
        if self.use_ekf:
            ekf = CPCMKalmanFilter(m=m)
            ekf.fit_dynamics(D_train)
            filtered, _ = ekf.filter(D_train)
            F_current = filtered[-1]
        else:
            F_current = D_train[-1]

        # Estimate covariance
        cov = estimate_covariance(R_train)

        # Optimize weights
        return self.optimizer.optimize(self.solver, F_current, cov)
