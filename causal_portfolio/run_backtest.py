"""CPCM Backtest Runner — CLI entry point.

Usage:
    python -m causal_portfolio.run_backtest --solver v1 --m 3 --assets btc,eth,sol
    python -m causal_portfolio.run_backtest --solver v4 --use-ekf --rebalance-freq 5

    # Dump the most-recent rebalance weights to JSON for the executor:
    python -m causal_portfolio.run_backtest --dump-weights weights.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from causal_portfolio.backtest.engine import CPCMBacktester
from causal_portfolio.data import get_loader
from causal_portfolio.factors.builder import build_all_factors
from causal_portfolio.factors.combo_selector import ComboDriverSelector
from causal_portfolio.optimizer.manifold import ManifoldOptimizer
from causal_portfolio.solvers.v1_linear import V1LinearSolver

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("cpcm.backtest")

DEFAULT_ASSETS = ["btc", "eth", "sol", "bnb", "avax", "uni", "aave", "link", "doge"]


def run(
    assets: list[str],
    solver_name: str = "v1",
    m: int = 3,
    start: str = "2022-01-01",
    end: str = "2025-12-31",
    use_ekf: bool = True,
    rebalance_freq: int = 5,
    train_window: int = 252,
    risk_aversion: float = 1.0,
    max_weight: float = 0.25,
    dump_weights_path: str | None = None,
):
    """Run full CPCM backtest pipeline."""

    # ── 1. Load data ────────────────────────────────────────────
    logger.info(f"Loading data for {len(assets)} assets...")
    loader = get_loader()

    all_metrics = [
        "PriceUSD", "price", "tvl_usd", "SplyCur",
        "stablecoin_circulating_usd", "FeeTotNtv",
        "FlowInExNtv", "FlowOutExNtv",
        "avg_gas_price_gwei", "avg_base_fee_gwei",
        "stddev_base_fee_gwei", "staking_apr",
        "cex_netflow_usd", "lp_net_flow_usd", "mev_revenue_eth",
    ]

    panel = loader.load_panel(assets, all_metrics, start, end)
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(
        ["DFF", "DGS10", "VIXCLS", "T10Y2Y", "CPIAUCSL", "M2SL", "DTWEXBGS"],
        start, end,
    )

    if panel.empty or returns.empty:
        logger.error("No data loaded. Check Supabase connection.")
        sys.exit(1)

    # ── 2. Compute factors + Combo selection ────────────────────
    factors = build_all_factors(panel, macro)
    available = factors.dropna(axis=1, how="all")
    m_actual = min(m, len(available.columns))

    selector = ComboDriverSelector()
    selected = selector.select(returns, available, m=m_actual)
    logger.info(f"Selected drivers: {selected}")

    # ── 3. Align data ──────────────────────────────────────────
    common = (
        factors[selected].dropna().index
        .intersection(returns.dropna().index)
    )
    D = factors.loc[common, selected].values
    R = returns.loc[common].values
    dates = common.values

    # ── 4. Create solver ───────────────────────────────────────
    if solver_name == "v4":
        from causal_portfolio.solvers.v4_pinn import V4PINNSolver
        solver = V4PINNSolver(
            m_drivers=m_actual, n_assets=R.shape[1],
            hidden_dim=64, n_layers=3,
        )
    else:
        solver = V1LinearSolver(alpha=1.0)

    # ── 5. Run backtest ────────────────────────────────────────
    optimizer = ManifoldOptimizer(
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )
    backtester = CPCMBacktester(
        solver=solver,
        optimizer=optimizer,
        use_ekf=use_ekf,
        rebalance_freq=rebalance_freq,
        train_window=train_window,
    )

    result = backtester.run(R, D, dates)

    # ── 6. Print results ───────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"CPCM Backtest Results ({start} to {end})")
    print(f"Solver: {'Combo-EKF-' if use_ekf else 'Combo-'}"
          f"{'V4' if solver_name == 'v4' else 'V1'}, "
          f"m={m_actual} drivers: {selected}")
    print("=" * 60)
    print(f"  Total Return:      {result.total_return:+.1%}")
    print(f"  Sharpe Ratio:      {result.sharpe:.3f}")
    print(f"  Sortino Ratio:     {result.sortino:.3f}")
    print(f"  Max Drawdown:      {result.max_dd:.1%}")
    print(f"  Calmar Ratio:      {result.calmar:.3f}")
    print(f"  Avg Turnover:      {result.avg_turnover:.3f}")
    print(f"  Coherence Score:   {result.coherence_score:.4f}")
    print(f"  Rebalances:        {len(result.rebalance_dates)}")
    print()

    # ── 7. Optionally dump the most-recent weight vector ─────────────
    if dump_weights_path:
        _dump_weights(
            result=result, assets=assets, returns_cols=list(returns.columns),
            out_path=dump_weights_path,
            metadata={
                "solver": solver_name, "m": m_actual,
                "use_ekf": use_ekf, "start": start, "end": end,
                "selected_drivers": list(selected),
                "rebalance_freq": rebalance_freq,
            },
        )

    return result


def _dump_weights(result, assets, returns_cols, out_path: str, metadata: dict) -> None:
    """Write the most-recent rebalance weights to a JSON file.

    The output is consumable by `python -m causal_portfolio.execution.cli`:
    a flat object mapping lowercase ticker → signed weight, plus a `_meta`
    field with provenance info that the executor ignores.

    The "most recent" weights come from `result.weights_history[-1]` — that's
    the vector that was in force on the last day of the backtest test window.
    Asset order is the order of `returns.columns` (stripped of `_return`),
    which matches the column order of weights_history.
    """
    weights_vec = result.weights_history[-1] if len(result.weights_history) else None
    if weights_vec is None or len(weights_vec) == 0:
        logger.warning("No weights to dump (empty backtest result)")
        return

    # Column names from `returns` look like "btc_return" — strip the suffix.
    asset_order = [c.rsplit("_", 1)[0] for c in returns_cols]
    if len(asset_order) != len(weights_vec):
        logger.warning(
            "asset/weight length mismatch: %d assets vs %d weights — skipping dump",
            len(asset_order), len(weights_vec),
        )
        return

    payload = {a: float(w) for a, w in zip(asset_order, weights_vec) if abs(w) > 0}
    payload["_meta"] = {
        **metadata,
        "rebalance_date": str(result.rebalance_dates[-1]) if result.rebalance_dates else None,
        "n_rebalances": len(result.rebalance_dates),
        "gross_exposure": float(np.sum(np.abs(weights_vec))),
        "n_nonzero": int(np.sum(np.abs(weights_vec) > 0)),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Dumped %d non-zero weights to %s (gross exposure %.4f)",
        payload["_meta"]["n_nonzero"], out_path, payload["_meta"]["gross_exposure"],
    )


def main():
    parser = argparse.ArgumentParser(description="CPCM Backtest Runner")
    parser.add_argument("--assets", type=str, default=None)
    parser.add_argument("--solver", choices=["v1", "v4"], default="v1")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--use-ekf", action="store_true", default=True)
    parser.add_argument("--no-ekf", dest="use_ekf", action="store_false")
    parser.add_argument("--rebalance-freq", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument(
        "--dump-weights", type=str, default=None, metavar="PATH",
        help="After the backtest, write the most-recent weight vector to PATH as JSON",
    )
    args = parser.parse_args()

    assets = args.assets.split(",") if args.assets else DEFAULT_ASSETS
    run(
        assets=assets,
        solver_name=args.solver,
        m=args.m,
        start=args.start,
        end=args.end,
        use_ekf=args.use_ekf,
        rebalance_freq=args.rebalance_freq,
        train_window=args.train_window,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        dump_weights_path=args.dump_weights,
    )


if __name__ == "__main__":
    main()
