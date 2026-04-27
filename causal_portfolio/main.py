"""CPCM Pipeline — main entry point.

Orchestrates: data loading → factor computation → Combo driver selection →
Star-DAG construction → V1 solver fitting → weight output.

Usage:
    python -m causal_portfolio.main [--assets btc,eth,sol] [--m 3]
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from causal_portfolio.data import get_loader
from causal_portfolio.factors.builder import build_all_factors
from causal_portfolio.factors.combo_selector import ComboDriverSelector
from causal_portfolio.scm.graph import build_cpcm_dag, summarize_dag
from causal_portfolio.solvers.v1_linear import V1LinearSolver

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("cpcm.main")

DEFAULT_ASSETS = [
    "btc", "eth", "sol", "bnb", "avax", "xrp", "doge",
    "uni", "aave", "link",
]


def run_pipeline(
    assets: list[str] | None = None,
    m: int = 3,
    start: str = "2022-01-01",
    end: str = "2025-12-31",
    alpha: float = 1.0,
) -> dict:
    """Run the full CPCM Phase 1+2 pipeline.

    Returns:
        Dictionary with selected drivers, solver diagnostics, and current weights.
    """
    assets = assets or DEFAULT_ASSETS

    # ── 1. Load data ────────────────────────────────────────────
    logger.info(f"Loading data for {len(assets)} assets: {assets}")
    loader = get_loader()
    panel = loader.load_panel(assets, _all_metrics(), start, end)
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(
        ["DFF", "DGS10", "VIXCLS", "T10Y2Y", "CPIAUCSL", "M2SL", "DTWEXBGS"],
        start, end,
    )

    if panel.empty or returns.empty:
        logger.error("No data loaded. Check Supabase connection.")
        return {}

    logger.info(f"Panel: {panel.shape}, Returns: {returns.shape}, Macro: {macro.shape}")

    # ── 2. Compute factors ──────────────────────────────────────
    factors = build_all_factors(panel, macro)
    logger.info(f"Computed {len(factors.columns)} factors: {list(factors.columns)}")

    # ── 3. Combo driver selection ───────────────────────────────
    selector = ComboDriverSelector()
    available_factors = factors.dropna(axis=1, how="all")
    m_actual = min(m, len(available_factors.columns))

    if m_actual < 2:
        logger.error(f"Only {m_actual} non-NaN factors available. Need at least 2.")
        return {}

    selected = selector.select(returns, available_factors, m=m_actual)
    logger.info(f"Selected drivers (m={m_actual}): {selected}")

    # ── 4. Build Star-DAG ───────────────────────────────────────
    dag = build_cpcm_dag(assets, selected)
    summary = summarize_dag(dag)
    logger.info(f"DAG: {summary['total_nodes']} nodes, {summary['total_edges']} edges")

    # ── 5. Fit V1 solver ────────────────────────────────────────
    # Align drivers and returns
    common_idx = factors[selected].dropna().index.intersection(returns.dropna().index)
    D = factors.loc[common_idx, selected].values
    R = returns.loc[common_idx].values

    solver = V1LinearSolver(alpha=alpha)
    solver.fit(D, R)
    diag = solver.diagnostics
    logger.info(f"V1 fitted: mean R²={diag['mean_r2']:.4f}")

    # ── 6. Current weights (mean-variance, no manifold yet) ────
    mu = solver.predict(D[-1:]).flatten()
    cov = solver.residual_cov_
    # Simple mean-variance: w = Sigma^{-1} mu / sum(Sigma^{-1} mu)
    cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    raw_w = cov_inv @ mu
    weights = raw_w / np.sum(np.abs(raw_w))  # normalize

    return_cols = list(returns.columns)
    weight_dict = {col.replace("_return", ""): float(w)
                   for col, w in zip(return_cols, weights)}

    return {
        "selected_drivers": selected,
        "dag_summary": summary,
        "solver_diagnostics": diag,
        "current_weights": weight_dict,
        "r2_per_asset": {
            col.replace("_return", ""): float(r2)
            for col, r2 in zip(return_cols, diag["r2_per_asset"])
        },
    }


def _all_metrics() -> list[str]:
    """All metrics needed for factor computation."""
    return [
        "PriceUSD", "price",
        "tvl_usd", "SplyCur", "stablecoin_circulating_usd",
        "FeeTotNtv", "FlowInExNtv", "FlowOutExNtv",
        "avg_gas_price_gwei", "avg_base_fee_gwei", "stddev_base_fee_gwei",
        "staking_apr", "cex_netflow_usd", "lp_net_flow_usd",
        "mev_revenue_eth",
        "TxCnt", "AdrActCnt", "CapMrktCurUSD",
    ]


def main():
    parser = argparse.ArgumentParser(description="CPCM Pipeline")
    parser.add_argument("--assets", type=str, default=None,
                        help="Comma-separated asset tickers")
    parser.add_argument("--m", type=int, default=3,
                        help="Number of drivers to select")
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization strength")
    args = parser.parse_args()

    assets = args.assets.split(",") if args.assets else None
    result = run_pipeline(assets, args.m, args.start, args.end, args.alpha)

    if not result:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("CPCM Pipeline Results")
    print("=" * 60)
    print(f"\nSelected drivers (m={len(result['selected_drivers'])}): "
          f"{result['selected_drivers']}")
    print(f"\nDAG: {result['dag_summary']['total_nodes']} nodes, "
          f"{result['dag_summary']['total_edges']} edges")
    print(f"\nV1 Solver — mean R²: {result['solver_diagnostics']['mean_r2']:.4f}")

    print("\nR² per asset:")
    for asset, r2 in result["r2_per_asset"].items():
        print(f"  {asset:12s} {r2:.4f}")

    print("\nCurrent portfolio weights:")
    for asset, w in sorted(result["current_weights"].items(),
                           key=lambda x: -abs(x[1])):
        bar = "#" * int(abs(w) * 50)
        sign = "+" if w >= 0 else "-"
        print(f"  {asset:12s} {sign}{abs(w):.4f}  {bar}")


if __name__ == "__main__":
    main()
