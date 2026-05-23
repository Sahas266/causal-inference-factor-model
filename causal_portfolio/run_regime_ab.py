"""Run the global vs hard-switch vs MoE backtest comparison.

This is the Phase B empirical gate. Three strategies are run on the same
dates/assets:

  - GLOBAL: existing CPCMBacktester with ComboDriverSelector picking once.
  - HARD-SWITCH: RegimeConditionalBacktester(mode='hard'). HMM + per-regime
                 driver selection + per-regime solver, route by current regime.
  - MOE: RegimeConditionalBacktester(mode='moe'). Same as hard-switch but
         blends mu across regimes by posterior weights.

Metrics reported side by side: total return, Sharpe, Sortino, max DD,
Calmar, turnover, per-regime rebalance counts.

Usage:
    python -m causal_portfolio.run_regime_ab \\
        --assets btc,eth,sol,bnb,avax,uni,aave,link,doge \\
        --start 2022-01-01 --end 2025-12-31 \\
        --m 3 --n-states 2 \\
        --rebalance-freq 5 \\
        --out causal_portfolio/docs/driver_selection_regimes.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.backtest.engine import CPCMBacktester
from causal_portfolio.backtest.regime_engine import (
    RegimeBacktestResult,
    RegimeConditionalBacktester,
)
from causal_portfolio.data import get_loader
from causal_portfolio.factors.builder import build_all_factors
from causal_portfolio.factors.combo_selector import ComboDriverSelector
from causal_portfolio.optimizer.manifold import ManifoldOptimizer
from causal_portfolio.solvers.v1_linear import V1LinearSolver

logger = logging.getLogger("cpcm.run_regime_ab")


def _load_data(assets: list[str], start: str, end: str):
    loader = get_loader()
    panel_metrics = [
        "PriceUSD", "price", "tvl_usd", "SplyCur",
        "stablecoin_circulating_usd", "FeeTotNtv",
        "FlowInExNtv", "FlowOutExNtv",
        "avg_gas_price_gwei", "avg_base_fee_gwei",
        "stddev_base_fee_gwei", "staking_apr",
        "cex_netflow_usd", "lp_net_flow_usd", "mev_revenue_eth",
    ]
    macro_series = ["DFF", "DGS10", "VIXCLS", "T10Y2Y",
                    "CPIAUCSL", "M2SL", "DTWEXBGS"]
    panel = loader.load_panel(assets, panel_metrics, start, end)
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(macro_series, start, end)
    factors = build_all_factors(panel, macro)
    return returns, factors, macro


def _run_global(
    returns: pd.DataFrame, factors: pd.DataFrame, m: int,
    optimizer: ManifoldOptimizer, train_window: int, rebalance_freq: int,
):
    """Run the existing CPCMBacktester with global driver selection."""
    available = factors.dropna(axis=1, how="all")
    selector = ComboDriverSelector()
    m_actual = min(m, available.shape[1])
    ranking = selector.rank_all_subsets(returns, available, m=m_actual)
    selected = list(ranking[0][0])
    logger.info("Global selector picked: %s", selected)

    common = (
        available[selected].dropna().index
        .intersection(returns.dropna().index)
    )
    D = available.loc[common, selected].values
    R = returns.loc[common].values
    dates = np.asarray(common)

    solver = V1LinearSolver(alpha=1.0)
    backtester = CPCMBacktester(
        solver=solver, optimizer=optimizer,
        use_ekf=True, rebalance_freq=rebalance_freq,
        train_window=train_window,
    )
    return backtester.run(R, D, dates), selected


def _run_regime(
    returns: pd.DataFrame, factors: pd.DataFrame, macro: pd.DataFrame,
    mode: str, m: int, n_states: int, optimizer: ManifoldOptimizer,
    train_window: int, rebalance_freq: int, hmm_window: int, hmm_refit_every: int,
) -> RegimeBacktestResult:
    backtester = RegimeConditionalBacktester(
        optimizer=optimizer, mode=mode,
        n_states=n_states, m=m,
        hmm_window=hmm_window, hmm_refit_every=hmm_refit_every,
        rebalance_freq=rebalance_freq, train_window=train_window,
    )
    return backtester.run(returns, factors, macro)


# ── reporting ──────────────────────────────────────────────────────


def _metrics_table(results: dict) -> str:
    headers = ["Metric", *results.keys()]
    rows: list[list[str]] = [
        ["Total return"] + [f"{r.total_return:+.1%}" for r in results.values()],
        ["Sharpe"] + [f"{r.sharpe:.3f}" for r in results.values()],
        ["Sortino"] + [f"{r.sortino:.3f}" for r in results.values()],
        ["Max drawdown"] + [f"{r.max_dd:.1%}" for r in results.values()],
        ["Calmar"] + [f"{r.calmar:.3f}" for r in results.values()],
        ["Avg turnover"] + [f"{r.avg_turnover:.3f}" for r in results.values()],
        ["Rebalances"] + [f"{len(r.rebalance_dates)}" for r in results.values()],
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _verdict(results: dict) -> str:
    global_r = results.get("Global")
    hard_r = results.get("Hard-switch")
    moe_r = results.get("MoE")
    if global_r is None:
        return "No global baseline."

    def cmp_sharpe(a, b):
        return a.sharpe - b.sharpe

    msgs = []
    if hard_r is not None:
        delta = cmp_sharpe(hard_r, global_r)
        if delta > 0.05:
            msgs.append(f"- **Hard-switch beats global by Sharpe {delta:+.3f}.**")
        elif delta < -0.05:
            msgs.append(f"- Hard-switch loses to global by Sharpe {-delta:.3f}.")
        else:
            msgs.append(f"- Hard-switch ≈ global (Sharpe delta {delta:+.3f}, within noise).")

    if moe_r is not None:
        delta = cmp_sharpe(moe_r, global_r)
        if delta > 0.05:
            msgs.append(f"- **MoE beats global by Sharpe {delta:+.3f}.**")
        elif delta < -0.05:
            msgs.append(f"- MoE loses to global by Sharpe {-delta:.3f}.")
        else:
            msgs.append(f"- MoE ≈ global (Sharpe delta {delta:+.3f}, within noise).")

    return "\n".join(msgs)


def render_markdown(results: dict, run_args: dict, selected_global: list[str]) -> str:
    lines: list[str] = []
    lines.append("### Run parameters\n")
    lines.append(f"- Date range: `{run_args['start']}` → `{run_args['end']}`")
    lines.append(f"- Assets: `{', '.join(run_args['assets'])}`")
    lines.append(f"- m: `{run_args['m']}`, n_states: `{run_args['n_states']}`")
    lines.append(f"- Rebalance every `{run_args['rebalance_freq']}` days, "
                 f"train window `{run_args['train_window']}`")
    lines.append(f"- HMM window `{run_args['hmm_window']}` days, "
                 f"refit every `{run_args['hmm_refit_every']}`")
    lines.append(f"- Global selector picked: `{selected_global}`")
    lines.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    lines.append("### Backtest metrics\n")
    lines.append(_metrics_table(results))
    lines.append("")

    # Per-regime rebalance counts if available
    for name, r in results.items():
        if hasattr(r, "per_regime_n_rebalances") and r.per_regime_n_rebalances:
            counts = ", ".join(f"R{k}: {v}" for k, v in sorted(r.per_regime_n_rebalances.items()))
            lines.append(f"- {name} per-regime rebalance counts: {counts}")
    lines.append("")

    lines.append("### Verdict\n")
    lines.append(_verdict(results))
    lines.append("")
    return "\n".join(lines)


def append_to_doc(md: str, out_path: str) -> None:
    path = Path(out_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    sentinel = "## Phase B step 2: regime-conditional A/B/C backtest\n"
    if sentinel in existing:
        head, _ = existing.split(sentinel, 1)
        new = head + sentinel + "\n" + md
    else:
        new = existing.rstrip() + "\n\n" + sentinel + "\n" + md
    path.write_text(new, encoding="utf-8")
    logger.info("Wrote A/B/C results to %s", out_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=str,
                   default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", type=str, default="2022-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--rebalance-freq", type=int, default=5)
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--hmm-window", type=int, default=504)
    p.add_argument("--hmm-refit-every", type=int, default=63)
    p.add_argument("--max-weight", type=float, default=0.25)
    p.add_argument("--out", type=str,
                   default="causal_portfolio/docs/driver_selection_regimes.md")
    p.add_argument("--skip-moe", action="store_true",
                   help="Run only global + hard-switch (faster)")
    args = p.parse_args()

    assets = args.assets.split(",")
    returns, factors, macro = _load_data(assets, args.start, args.end)
    optimizer = ManifoldOptimizer(risk_aversion=1.0, max_weight=args.max_weight)

    results: dict = {}

    logger.info("=== Running GLOBAL ===")
    global_result, selected_global = _run_global(
        returns, factors, args.m, optimizer,
        args.train_window, args.rebalance_freq,
    )
    results["Global"] = global_result

    logger.info("=== Running HARD-SWITCH ===")
    hard_result = _run_regime(
        returns, factors, macro, "hard",
        args.m, args.n_states, optimizer,
        args.train_window, args.rebalance_freq,
        args.hmm_window, args.hmm_refit_every,
    )
    results["Hard-switch"] = hard_result

    if not args.skip_moe:
        logger.info("=== Running MoE ===")
        moe_result = _run_regime(
            returns, factors, macro, "moe",
            args.m, args.n_states, optimizer,
            args.train_window, args.rebalance_freq,
            args.hmm_window, args.hmm_refit_every,
        )
        results["MoE"] = moe_result

    # ── console summary ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<15} {'Sharpe':>8} {'Total':>8} {'MaxDD':>8} {'Turnover':>10}")
    for name, r in results.items():
        print(f"{name:<15} {r.sharpe:>8.3f} {r.total_return:>+8.1%} "
              f"{r.max_dd:>+8.1%} {r.avg_turnover:>10.3f}")
    print()
    print(_verdict(results))

    md = render_markdown(results, run_args={
        "assets": assets, "start": args.start, "end": args.end,
        "m": args.m, "n_states": args.n_states,
        "rebalance_freq": args.rebalance_freq,
        "train_window": args.train_window,
        "hmm_window": args.hmm_window,
        "hmm_refit_every": args.hmm_refit_every,
    }, selected_global=selected_global)
    append_to_doc(md, args.out)
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
