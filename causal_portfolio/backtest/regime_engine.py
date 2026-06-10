"""Regime-conditional walk-forward backtester.

Each rebalance:
  1. Fit HMM (multi-start) on the rolling window of regime features.
  2. Decode regime labels for the training window (in-window Viterbi).
  3. For each regime k with enough obs: run ComboDriverSelector on the
     regime-k subset of training data; fit a per-regime V1 solver on the
     regime-k subset using those drivers; estimate per-regime covariance.
  4. Determine current regime via forward filter (causal — uses only past).
  5. HARD mode: use current regime's solver/drivers/cov directly.
     MOE mode: blend mu across all regimes by posterior weights; use the
     argmax regime's Jacobian and covariance for the manifold projection.

This is research code — the comparable global backtester is CPCMBacktester
which takes already-selected drivers and uses one global solver.

Note on look-ahead: the in-window decode (step 2) uses Viterbi over the
training window only, so it doesn't see future. The current regime label
(step 4) is from forward filter — strictly causal. There's still a subtle
relabeling drift between refit boundaries that we don't address yet (state 0
may not point to the same true regime across refits — see the diagnostic
results in driver_selection_regimes.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from causal_portfolio.backtest.engine import BacktestResult
from causal_portfolio.factors.combo_selector import ComboDriverSelector
from causal_portfolio.filters.ekf import CPCMKalmanFilter
from causal_portfolio.optimizer.manifold import ManifoldOptimizer, estimate_covariance
from causal_portfolio.regimes.hmm import (
    RegimeClassifier,
    RollingHMM,
    build_regime_features,
)
from causal_portfolio.solvers.v1_linear import V1LinearSolver

logger = logging.getLogger("cpcm.backtest.regime")


@dataclass
class _RegimeModel:
    """Per-regime training output."""
    drivers: list[str]
    solver: V1LinearSolver
    covariance: np.ndarray
    n_train: int


@dataclass
class RegimeBacktestResult(BacktestResult):
    """Extends BacktestResult with regime telemetry."""
    regime_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    posterior_history: np.ndarray = field(default_factory=lambda: np.array([]))
    per_regime_n_rebalances: dict = field(default_factory=dict)


class RegimeConditionalBacktester:
    """Walk-forward backtester with HMM-conditioned driver + solver selection."""

    def __init__(
        self,
        optimizer: ManifoldOptimizer,
        *,
        mode: str = "hard",  # 'hard' or 'moe'
        n_states: int = 2,
        m: int = 3,
        hmm_window: int = 504,
        hmm_refit_every: int = 63,
        hmm_n_restarts: int = 5,
        rebalance_freq: int = 5,
        train_window: int = 252,
        min_obs_per_regime: int = 60,
    ):
        if mode not in ("hard", "moe"):
            raise ValueError(f"mode must be 'hard' or 'moe', got {mode!r}")
        self.optimizer = optimizer
        self.mode = mode
        self.n_states = n_states
        self.m = m
        self.hmm_window = hmm_window
        self.hmm_refit_every = hmm_refit_every
        self.hmm_n_restarts = hmm_n_restarts
        self.rebalance_freq = rebalance_freq
        self.train_window = train_window
        self.min_obs_per_regime = min_obs_per_regime

        # Filled during run()
        self._global_drivers: Optional[list[str]] = None
        self._global_solver: Optional[V1LinearSolver] = None

    # ── public API ──────────────────────────────────────────────────

    def run(
        self,
        returns: pd.DataFrame,
        factor_panel: pd.DataFrame,
        macro: pd.DataFrame,
        dates: Optional[np.ndarray] = None,
    ) -> RegimeBacktestResult:
        """Run the regime-conditional walk-forward backtest.

        Args:
            returns: (T, n_assets) DataFrame of simple returns.
            factor_panel: (T, n_factors) DataFrame of ALL candidate factors.
            macro: (T, n_macro) DataFrame — used for HMM features (needs 'vixcls').
            dates: optional date array; defaults to returns.index.
        """
        # Align all inputs on common dates
        common = (
            returns.dropna().index
            .intersection(factor_panel.dropna(how="all").index)
            .intersection(macro.index)
        )
        R_all = returns.loc[common]
        M_all = macro.loc[common]

        # Build regime features (uses VIX + BTC realized vol with 21d warmup).
        # The first ~21 rows will have NaN btc_vol — exclude those from `common`
        # so the HMM never sees NaN inputs.
        feats_raw = build_regime_features(M_all, R_all)
        common = common.intersection(feats_raw.index)
        if dates is None:
            dates = np.asarray(common)
        R = returns.loc[common]
        F_panel = factor_panel.loc[common].dropna(axis=1, how="all")
        feats = feats_raw.loc[common]

        T = len(common)
        n_assets = R.shape[1]

        if T <= max(self.train_window, self.hmm_window):
            raise ValueError(
                f"Need T > max(train_window, hmm_window) = "
                f"{max(self.train_window, self.hmm_window)}, got T={T}"
            )

        test_start = max(self.train_window, self.hmm_window)
        n_test = T - test_start

        portfolio_returns = np.zeros(n_test)
        weights_history = np.zeros((n_test, n_assets))
        regime_labels = np.full(n_test, -1, dtype=int)
        posterior_history = np.full((n_test, self.n_states), np.nan)
        current_weights = np.ones(n_assets) / n_assets
        rebalance_dates: list = []
        per_regime_rebal: dict[int, int] = {k: 0 for k in range(self.n_states)}

        # HMM refit state (amortized daily forward filter — see RollingHMM)
        roller = RollingHMM(
            feats, self.hmm_window, self.hmm_refit_every,
            n_states=self.n_states, n_restarts=self.hmm_n_restarts,
        )
        regime_models: dict[int, _RegimeModel] = {}

        # Global fallback (used before HMM is initialized and when a regime is empty)
        self._fit_global_fallback(R, F_panel, test_start)

        for t in range(test_start, T):
            idx = t - test_start
            should_rebalance = (idx % self.rebalance_freq == 0)

            # ── HMM refit ────────────────────────────────────────────
            if roller.needs_refit(t):
                classifier = roller.refit(t)
                # Per-regime models fit on training window (last `train_window` days)
                regime_models = self._fit_regime_models(
                    classifier, R.iloc[t - self.train_window : t],
                    F_panel.iloc[t - self.train_window : t],
                    feats.iloc[t - self.train_window : t],
                )

            # ── Current regime via forward filter (causal) ──────────
            posterior = roller.posterior(t)
            current_regime = int(posterior.argmax())
            regime_labels[idx] = current_regime
            posterior_history[idx] = posterior

            # ── Rebalance ────────────────────────────────────────────
            if should_rebalance:
                try:
                    current_weights = self._compute_weights(
                        regime_models, current_regime, posterior,
                        F_panel.iloc[t - self.train_window : t],
                    )
                    rebalance_dates.append(dates[t])
                    per_regime_rebal[current_regime] = (
                        per_regime_rebal.get(current_regime, 0) + 1
                    )
                except Exception as e:
                    logger.warning("Rebalance failed at t=%d: %s", t, e)

            # ── P&L ──────────────────────────────────────────────────
            day_return = R.iloc[t].values
            valid = ~np.isnan(day_return)
            if valid.any():
                portfolio_returns[idx] = float(
                    np.nansum(current_weights[valid] * day_return[valid])
                )
            weights_history[idx] = current_weights

        # ── Metrics ──────────────────────────────────────────────────
        # coherence_score not computed here; could add via martingale_defect
        result = RegimeBacktestResult.from_returns(
            portfolio_returns, weights_history, rebalance_dates,
            regime_labels=regime_labels,
            posterior_history=posterior_history,
            per_regime_n_rebalances=per_regime_rebal,
        )
        logger.info(
            "Regime-conditional (%s) backtest complete: %s | per-regime rebal: %s",
            self.mode, result.summary(), per_regime_rebal,
        )
        return result

    # ── helpers ─────────────────────────────────────────────────────

    def _fit_global_fallback(
        self,
        returns: pd.DataFrame,
        factor_panel: pd.DataFrame,
        test_start: int,
    ) -> None:
        """Fit a global driver subset + solver from the initial training portion.

        Used as fallback when a regime has too few training observations.
        Only fit ONCE on the pre-test window — this is just a safety net,
        not the strategy. Look-ahead-safe.
        """
        slot = factor_panel.iloc[:test_start].dropna(how="all")
        slot_R = returns.iloc[:test_start].loc[slot.index]
        slot_F = slot.dropna(axis=1, how="all")
        if slot_F.shape[1] < self.m or slot_R.empty:
            self._global_drivers = list(slot_F.columns[: self.m])
            self._global_solver = None
            return
        selector = ComboDriverSelector()
        ranking = selector.rank_all_subsets(slot_R, slot_F, m=min(self.m, slot_F.shape[1]))
        drivers = list(ranking[0][0])
        D = slot_F[drivers].values
        # Drop NaN aligned rows
        mask = ~(np.isnan(D).any(axis=1) | np.isnan(slot_R.values).any(axis=1))
        if mask.sum() < 30:
            self._global_drivers = drivers
            self._global_solver = None
            return
        solver = V1LinearSolver(alpha=1.0)
        solver.fit(D[mask], slot_R.values[mask])
        self._global_drivers = drivers
        self._global_solver = solver

    def _fit_regime_models(
        self,
        classifier: RegimeClassifier,
        returns_tr: pd.DataFrame,
        factor_panel_tr: pd.DataFrame,
        feats_tr: pd.DataFrame,
    ) -> dict[int, _RegimeModel]:
        """Per-regime driver selection + solver fit on the training window."""
        # Label the training window (in-window Viterbi is fine — no future leak)
        labels = classifier.decode(feats_tr)
        F = factor_panel_tr.dropna(axis=1, how="all")
        models: dict[int, _RegimeModel] = {}
        for k in range(self.n_states):
            k_mask = labels == k
            n_k = int(k_mask.sum())
            if n_k < self.min_obs_per_regime:
                continue
            R_k = returns_tr.iloc[k_mask].dropna()
            F_k = F.iloc[k_mask].loc[R_k.index].dropna(axis=1, how="all")
            if F_k.shape[1] < self.m or len(R_k) < 30:
                continue
            # Per-regime driver selection
            selector = ComboDriverSelector()
            ranking = selector.rank_all_subsets(R_k, F_k, m=min(self.m, F_k.shape[1]))
            drivers_k = list(ranking[0][0])
            # Per-regime solver fit
            D_k = F_k[drivers_k].values
            R_k_vals = R_k.values
            mask = ~(np.isnan(D_k).any(axis=1) | np.isnan(R_k_vals).any(axis=1))
            if mask.sum() < 30:
                continue
            solver_k = V1LinearSolver(alpha=1.0)
            solver_k.fit(D_k[mask], R_k_vals[mask])
            cov_k = estimate_covariance(R_k_vals[mask])
            models[k] = _RegimeModel(
                drivers=drivers_k, solver=solver_k,
                covariance=cov_k, n_train=int(mask.sum()),
            )
        return models

    def _compute_weights(
        self,
        regime_models: dict[int, _RegimeModel],
        current_regime: int,
        posterior: np.ndarray,
        factor_panel_tr: pd.DataFrame,
    ) -> np.ndarray:
        """Pick or blend per-regime predictions and run the optimizer."""
        # Pick model for current regime; fall back to any available or to global
        primary = regime_models.get(current_regime)
        if primary is None:
            # Fall back to any regime with a model
            primary = next(iter(regime_models.values()), None)
        if primary is None:
            # Last resort: use global fallback
            if self._global_solver is None:
                raise RuntimeError("No regime models and no global fallback")
            F_global, _ = self._current_filtered_state(
                self._global_drivers, factor_panel_tr,
            )
            return self.optimizer.optimize(
                self._global_solver, F_global,
                estimate_covariance(factor_panel_tr.values),
            )

        # Compute current driver vector + filtered state for primary regime
        F_primary, _ = self._current_filtered_state(
            primary.drivers, factor_panel_tr,
        )

        if self.mode == "hard":
            return self.optimizer.optimize(
                primary.solver, F_primary, primary.covariance,
            )

        # ── MoE mode: blend mu across regimes ──────────────────────
        n_assets = primary.covariance.shape[0]
        mu_blend = np.zeros(n_assets)
        total_weight = 0.0
        for k, model in regime_models.items():
            p_k = float(posterior[k])
            if p_k < 1e-3:
                continue
            F_k, _ = self._current_filtered_state(model.drivers, factor_panel_tr)
            mu_k = model.solver.predict(F_k).flatten()
            mu_blend += p_k * mu_k
            total_weight += p_k
        if total_weight > 0:
            mu_blend /= total_weight  # renormalize if some posterior fell below tolerance

        # Use primary regime's J + cov for the projection
        return self.optimizer.optimize(
            primary.solver, F_primary, primary.covariance, mu_override=mu_blend,
        )

    def _current_filtered_state(
        self, drivers: list[str], factor_panel_tr: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get filtered last-day driver values for a specific driver subset."""
        D_tr = factor_panel_tr[drivers].dropna().values
        if len(D_tr) < 10:
            # Not enough data — use last observed values unfiltered
            return factor_panel_tr[drivers].iloc[-1].values, D_tr

        ekf = CPCMKalmanFilter(m=len(drivers))
        ekf.fit_dynamics(D_tr)
        filtered, _ = ekf.filter(D_tr)
        return filtered[-1], D_tr
