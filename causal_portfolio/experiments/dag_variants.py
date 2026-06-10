"""Test different DAG arrangements through the full CPCM backtest.

Context: in the Python pipeline the DAG (causal_portfolio/scm/graph.py) is
purely decorative — the backtest regresses returns on the Combo-selected
factors and never applies the DAG's edge structure or its stated macro
lag-1. So the DAG's *return-equation structure* (which factors are direct
causes of returns, and at what lag) is an untested lever.

A "DAG variant" here is a concrete specification of that structure:
  - which factors have edges into the return node (the regressor pool)
  - the lag on those edges (contemporaneous vs predictive)
  - whether to Combo-select a subset or use the whole pool

Each variant is run through the SAME downstream pipeline (V1 solver + EKF +
manifold optimizer + walk-forward backtest), so differences in results are
attributable to the DAG structure alone.

Overfitting discipline: we test several structures and would naturally pick
the best — exactly the trap that produced false positives earlier in this
project. So every variant is evaluated on BOTH a full window and a held-out
OOS window, and judged against buy-and-hold BTC. A variant only "wins" if it
beats baseline AND BH BTC out of sample, not just in-sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from causal_portfolio.backtest.engine import CPCMBacktester
from causal_portfolio.backtest.metrics import (
    max_drawdown, sharpe_ratio, sortino_ratio,
)
from causal_portfolio.factors.builder import (  # noqa: F401 — re-exported
    FACTOR_SOURCE_ASSETS, GLOBAL_FACTORS, MACRO_FACTORS, MACRO_SERIES,
    PANEL_METRICS, build_all_factors,
)
from causal_portfolio.factors.combo_selector import ComboDriverSelector
from causal_portfolio.optimizer.manifold import ManifoldOptimizer
from causal_portfolio.solvers.v1_linear import V1LinearSolver

logger = logging.getLogger("cpcm.experiments.dag")


@dataclass(frozen=True)
class DagVariant:
    """A return-equation structure to test.

    factors: explicit factor pool (edges into return). None = all available.
    lag_global / lag_macro: lag applied to global / macro factor edges.
    combo_m: if set, Combo-select this many from the pool; else use all.
    """
    name: str
    factors: tuple[str, ...] | None
    lag_global: int
    lag_macro: int
    combo_m: int | None
    description: str = ""


# ── The variants to test ────────────────────────────────────────────
#
# Systematic grid: every "base structure" (factor pool + optional Combo
# selection) is run BOTH contemporaneous (lag 0 everywhere) and fully lagged
# (lag 1 everywhere). The lag-everywhere versions are the predictive forms —
# they use only past factor values, removing the contemporaneous look-ahead
# edge. This covers "lag everywhere, all the configs".

_BASE_STRUCTURES = [
    # (label, factor_pool, combo_m, description)
    ("combo3", None, 3,
     "Combo-select m=3 from all factors (the pipeline's current default pool)."),
    ("all", None, None,
     "Every available factor → returns (dense DAG)."),
    ("global", tuple(GLOBAL_FACTORS), None,
     "Only on-chain global factors → returns."),
    ("macro", tuple(MACRO_FACTORS), None,
     "Only macro factors → returns."),
    ("onchain_core", ("chain_congestion", "mev_pressure", "liq_flow"), None,
     "The drivers Option-2 found dominate: chain_congestion + mev_pressure + liq_flow."),
]


def _make_variants() -> list[DagVariant]:
    variants: list[DagVariant] = []
    for label, pool, combo_m, desc in _BASE_STRUCTURES:
        for lag in (0, 1):
            lag_tag = "lag0 (contemporaneous)" if lag == 0 else "lag1 (everywhere, predictive)"
            variants.append(DagVariant(
                name=f"{label}_lag{lag}",
                factors=pool, lag_global=lag, lag_macro=lag, combo_m=combo_m,
                description=f"{desc} {lag_tag}.",
            ))
    return variants


VARIANTS: list[DagVariant] = _make_variants()


@dataclass
class VariantResult:
    name: str
    window: str
    drivers_used: list[str]
    total_return: float
    sharpe: float
    sortino: float
    max_dd: float
    n_obs: int
    error: str | None = None


def load_inputs(assets: list[str], start: str, end: str):
    from causal_portfolio.data import get_loader
    loader = get_loader()
    # Panel spans trading assets + stablecoin factor sources, so stable_flow and
    # the stablecoin_mint instrument have data; returns cover trading assets only.
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, start, end)
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    factors = build_all_factors(panel, macro)
    return returns, factors


def _build_drivers(
    variant: DagVariant, returns: pd.DataFrame, factors: pd.DataFrame,
    *, train_window: int = 252,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Construct the (returns, drivers, dates, driver_names) for a variant.

    Applies per-factor-type lags, optionally Combo-selects, and aligns.
    Combo selection sees only the first `train_window` rows — strictly before
    the first evaluated return (the backtester's evaluation starts after its
    initial train window), so the selected drivers cannot leak OOS data.
    """
    available = factors.dropna(axis=1, how="all")
    pool = list(available.columns) if variant.factors is None else [
        f for f in variant.factors if f in available.columns
    ]
    if not pool:
        raise ValueError(f"variant {variant.name}: no factors available from pool")

    # Apply lags by factor type
    lagged = pd.DataFrame(index=available.index)
    for f in pool:
        lag = variant.lag_macro if f in MACRO_FACTORS else variant.lag_global
        lagged[f] = available[f].shift(lag) if lag else available[f]

    # Align with returns, drop NaN rows from lagging
    common = lagged.dropna().index.intersection(returns.dropna().index)
    lagged = lagged.loc[common]
    R = returns.loc[common]

    # Optional Combo selection within the pool — on the train window ONLY
    # (selecting on the full sample would pick drivers using the very days
    # the walk-forward backtest then scores as out-of-sample).
    if variant.combo_m is not None:
        m = min(variant.combo_m, lagged.shape[1])
        ranking = ComboDriverSelector().rank_all_subsets(
            R.iloc[:train_window], lagged.iloc[:train_window], m=m)
        selected = list(ranking[0][0])
    else:
        selected = list(lagged.columns)

    D = lagged[selected].values
    Rv = R.values
    dates = np.asarray(common)
    return Rv, D, dates, selected


def run_variant(
    variant: DagVariant, returns: pd.DataFrame, factors: pd.DataFrame,
    window_label: str, *,
    train_window: int = 252, rebalance_freq: int = 5,
    risk_aversion: float = 1.0, max_weight: float = 0.25,
) -> VariantResult:
    try:
        Rv, D, dates, selected = _build_drivers(
            variant, returns, factors, train_window=train_window)
        if len(Rv) <= train_window + 10:
            return VariantResult(variant.name, window_label, selected,
                                 float("nan"), float("nan"), float("nan"),
                                 float("nan"), len(Rv),
                                 error=f"too few rows ({len(Rv)})")
        solver = V1LinearSolver(alpha=1.0)
        optimizer = ManifoldOptimizer(risk_aversion=risk_aversion, max_weight=max_weight)
        bt = CPCMBacktester(solver=solver, optimizer=optimizer, use_ekf=True,
                            rebalance_freq=rebalance_freq, train_window=train_window)
        res = bt.run(Rv, D, dates)
        return VariantResult(
            variant.name, window_label, selected,
            res.total_return, res.sharpe, res.sortino, res.max_dd,
            len(res.returns_series),
        )
    except Exception as e:
        logger.warning("variant %s failed: %s", variant.name, e)
        return VariantResult(variant.name, window_label, [], float("nan"),
                             float("nan"), float("nan"), float("nan"), 0, error=str(e))


def buy_and_hold_btc(returns: pd.DataFrame) -> dict:
    """BH BTC over the given returns window (no costs, for relative comparison)."""
    if "btc_return" not in returns.columns:
        return {}
    r = returns["btc_return"].dropna().values
    pv = np.cumprod(1 + r)
    return {
        # cumprod starts at 1+r[0]; dividing by pv[0] would drop day 1's return
        "total_return": float(pv[-1] - 1.0),
        "sharpe": float(sharpe_ratio(r)),
        "max_dd": float(max_drawdown(r)),
        "n_obs": len(r),
    }
