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
from causal_portfolio.factors.builder import (
    GLOBAL_FACTORS, MACRO_FACTORS, build_all_factors,
)
from causal_portfolio.factors.combo_selector import ComboDriverSelector
from causal_portfolio.optimizer.manifold import ManifoldOptimizer
from causal_portfolio.solvers.v1_linear import V1LinearSolver

logger = logging.getLogger("cpcm.experiments.dag")

PANEL_METRICS = [
    "PriceUSD", "price", "tvl_usd", "SplyCur",
    "stablecoin_circulating_usd", "FeeTotNtv",
    "FlowInExNtv", "FlowOutExNtv",
    "avg_gas_price_gwei", "avg_base_fee_gwei",
    "stddev_base_fee_gwei", "staking_apr",
    "cex_netflow_usd", "lp_net_flow_usd", "mev_revenue_eth",
]
MACRO_SERIES = ["DFF", "DGS10", "VIXCLS", "T10Y2Y", "CPIAUCSL", "M2SL", "DTWEXBGS"]


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

VARIANTS: list[DagVariant] = [
    DagVariant("baseline_combo", None, 0, 0, 3,
               "Current pipeline: Combo-select m=3 from all factors, contemporaneous."),
    DagVariant("baseline_macro_lag1", None, 0, 1, 3,
               "Combo m=3 but macro edges lagged 1 day (honors the stated DAG lag)."),
    DagVariant("global_only", tuple(GLOBAL_FACTORS), 0, 0, None,
               "Only on-chain global factors → returns, contemporaneous."),
    DagVariant("macro_only", tuple(MACRO_FACTORS), 0, 1, None,
               "Only macro factors → returns, lagged 1 day."),
    DagVariant("onchain_core", ("chain_congestion", "mev_pressure", "liq_flow"), 0, 0, None,
               "The drivers Option-2 found dominate: chain_congestion + mev_pressure + liq_flow."),
    DagVariant("all_lag0", None, 0, 0, None,
               "Every factor → returns, contemporaneous (dense DAG)."),
    DagVariant("all_lag1", None, 1, 1, None,
               "Every factor → returns, lagged 1 day (fully predictive, no contemporaneous edge)."),
    DagVariant("combo_all_lag1", None, 1, 1, 3,
               "Combo m=3 with all edges lagged 1 day."),
]


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
    panel = loader.load_panel(assets, PANEL_METRICS, start, end)
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    factors = build_all_factors(panel, macro)
    return returns, factors


def _build_drivers(
    variant: DagVariant, returns: pd.DataFrame, factors: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Construct the (returns, drivers, dates, driver_names) for a variant.

    Applies per-factor-type lags, optionally Combo-selects, and aligns.
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

    # Optional Combo selection within the pool
    if variant.combo_m is not None:
        m = min(variant.combo_m, lagged.shape[1])
        ranking = ComboDriverSelector().rank_all_subsets(R, lagged, m=m)
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
        Rv, D, dates, selected = _build_drivers(variant, returns, factors)
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
        "total_return": float(pv[-1] / pv[0] - 1),
        "sharpe": float(sharpe_ratio(r)),
        "max_dd": float(max_drawdown(r)),
        "n_obs": len(r),
    }
