"""Strategy search: find a CPCM-related configuration that beats buy-and-hold BTC.

Runs a small set of strategies on the same date/asset window with realistic
transaction costs, and prints a comparison table. The goal is a strategy
with **positive Sharpe AND total return ≥ buy-and-hold BTC** over the full
2022-2025 period.

Strategies compared:
  - BH_BTC:          buy-and-hold BTC (the bar to beat)
  - EW_BASKET:       equal-weighted basket of provided assets, monthly rebal
  - REGIME_BTC:      HMM-gated 100% BTC during calm, cash during stress
  - REGIME_EW:       HMM-gated equal-weight basket during calm, cash during stress
  - REGIME_BTC_T:    same as REGIME_BTC with no-trade band threshold
  - REGIME_BTC_50:   HMM-gated BTC with 50% allocation during stress (partial de-risk)

Each strategy is run with the same fee + slippage assumptions. Threshold-
sensitivity sweep optionally runs for the best regime-gated config.

Usage:
    python -m causal_portfolio.run_strategy_search \\
        --assets btc,eth,sol --start 2022-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.backtest.engine import BacktestResult
from causal_portfolio.backtest.strategies import (
    buy_and_hold,
    equal_weight_basket,
    regime_gated_long_only,
)
from causal_portfolio.data import get_loader
from causal_portfolio.factors.builder import MACRO_SERIES

logger = logging.getLogger("cpcm.strategy_search")


@dataclass
class StrategyMetrics:
    name: str
    total_return: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    avg_turnover: float
    n_rebalances: int

    @classmethod
    def from_result(cls, name: str, r: BacktestResult) -> "StrategyMetrics":
        return cls(
            name=name,
            total_return=r.total_return, sharpe=r.sharpe, sortino=r.sortino,
            max_dd=r.max_dd, calmar=r.calmar, avg_turnover=r.avg_turnover,
            n_rebalances=len(r.rebalance_dates),
        )


def _load_data(assets: list[str], start: str, end: str):
    loader = get_loader()
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    return returns, macro


def run_search(
    assets: list[str], start: str, end: str,
    fee_bps: float = 5.0, slippage_bps: float = 5.0,
    hmm_window: int = 504, hmm_refit_every: int = 63,
    threshold_l1: float = 0.5,
) -> list[StrategyMetrics]:
    """Run all baseline strategies and return a list of metrics."""
    returns, macro = _load_data(assets, start, end)
    # Align both to the same date range; macro often has weekend gaps already ffilled
    returns = returns.dropna(how="all")
    macro = macro.loc[returns.index.intersection(macro.index)].ffill()
    returns = returns.loc[macro.index]

    cost_kwargs = dict(fee_bps=fee_bps, slippage_bps=slippage_bps)

    metrics: list[StrategyMetrics] = []

    # 1. Buy and hold BTC — the bar to beat
    if "btc_return" in returns.columns:
        r = buy_and_hold(returns, asset="btc_return", **cost_kwargs)
        metrics.append(StrategyMetrics.from_result("BH_BTC", r))

    # 2. Equal-weighted basket (monthly rebal)
    r = equal_weight_basket(returns, rebalance_freq=21, **cost_kwargs)
    metrics.append(StrategyMetrics.from_result("EW_BASKET", r))

    # 3. Regime-gated 100% BTC
    if "btc_return" in returns.columns:
        r = regime_gated_long_only(
            returns, macro, universe_weights={"btc": 1.0},
            hmm_window=hmm_window, hmm_refit_every=hmm_refit_every,
            stress_allocation=0.0,
            **cost_kwargs,
        )
        metrics.append(StrategyMetrics.from_result("REGIME_BTC", r))

    # 4. Regime-gated equal-weight basket
    universe = {a: 1.0 for a in assets if f"{a}_return" in returns.columns}
    if universe:
        r = regime_gated_long_only(
            returns, macro, universe_weights=universe,
            hmm_window=hmm_window, hmm_refit_every=hmm_refit_every,
            stress_allocation=0.0,
            **cost_kwargs,
        )
        metrics.append(StrategyMetrics.from_result("REGIME_EW", r))

    # 5. Regime-gated BTC with no-trade threshold
    if "btc_return" in returns.columns:
        r = regime_gated_long_only(
            returns, macro, universe_weights={"btc": 1.0},
            hmm_window=hmm_window, hmm_refit_every=hmm_refit_every,
            stress_allocation=0.0,
            threshold_l1=threshold_l1,
            **cost_kwargs,
        )
        metrics.append(StrategyMetrics.from_result(
            f"REGIME_BTC_T{int(threshold_l1*100)}", r,
        ))

    # 6. Regime-gated BTC with 50% partial de-risk (not full cash)
    if "btc_return" in returns.columns:
        r = regime_gated_long_only(
            returns, macro, universe_weights={"btc": 1.0},
            hmm_window=hmm_window, hmm_refit_every=hmm_refit_every,
            stress_allocation=0.5,
            **cost_kwargs,
        )
        metrics.append(StrategyMetrics.from_result("REGIME_BTC_50", r))

    return metrics


# ── reporting ──────────────────────────────────────────────────────


def format_table(metrics: list[StrategyMetrics]) -> str:
    if not metrics:
        return "(no results)"
    header = (f"{'Strategy':<18} {'Total':>9} {'Sharpe':>8} {'Sortino':>9} "
              f"{'MaxDD':>9} {'Calmar':>8} {'Turnover':>9} {'Rebals':>7}")
    sep = "-" * len(header)
    lines = [header, sep]
    for m in metrics:
        lines.append(
            f"{m.name:<18} {m.total_return:>+9.1%} {m.sharpe:>8.3f} "
            f"{m.sortino:>9.3f} {m.max_dd:>+9.1%} {m.calmar:>8.3f} "
            f"{m.avg_turnover:>9.3f} {m.n_rebalances:>7d}"
        )
    return "\n".join(lines)


def find_winner(metrics: list[StrategyMetrics]) -> tuple[StrategyMetrics, list[str]]:
    """Return the strategy that beats buy-and-hold BTC with the highest Sharpe."""
    bh = next((m for m in metrics if m.name == "BH_BTC"), None)
    if bh is None:
        return max(metrics, key=lambda m: m.sharpe), ["No BH_BTC baseline"]

    beaters = [m for m in metrics
               if m.name != "BH_BTC" and m.total_return > bh.total_return]
    notes = []
    if not beaters:
        notes.append(f"No strategy beat BH_BTC total return ({bh.total_return:+.1%}).")
        return max(metrics, key=lambda m: m.sharpe), notes

    # Among beaters, pick highest Sharpe
    best = max(beaters, key=lambda m: m.sharpe)
    notes.append(
        f"Winner '{best.name}' beat BH_BTC by "
        f"{(best.total_return - bh.total_return)*100:.1f}pp total return "
        f"and {best.sharpe - bh.sharpe:+.3f} Sharpe."
    )
    return best, notes


def render_markdown(
    metrics: list[StrategyMetrics], run_args: dict,
    winner: StrategyMetrics, notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append("### Run parameters\n")
    lines.append(f"- Date range: `{run_args['start']}` to `{run_args['end']}`")
    lines.append(f"- Assets: `{', '.join(run_args['assets'])}`")
    lines.append(f"- Fee: `{run_args['fee_bps']}` bps, "
                 f"slippage: `{run_args['slippage_bps']}` bps "
                 f"(round-trip on |Δw|_L1 * equity)")
    lines.append(f"- HMM window: `{run_args['hmm_window']}` days, "
                 f"refit every `{run_args['hmm_refit_every']}` days")
    lines.append(f"- No-trade threshold (for regime configs that use one): "
                 f"`{run_args['threshold_l1']}` L1 distance")
    lines.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    lines.append("### Results\n")
    lines.append("```")
    lines.append(format_table(metrics))
    lines.append("```\n")

    lines.append("### Verdict\n")
    for n in notes:
        lines.append(f"- {n}")
    lines.append(f"- Highest-Sharpe winner: **{winner.name}** "
                 f"(Sharpe `{winner.sharpe:.3f}`, "
                 f"total `{winner.total_return:+.1%}`, "
                 f"max DD `{winner.max_dd:+.1%}`).")
    return "\n".join(lines)


def append_to_doc(md: str, out_path: str) -> None:
    path = Path(out_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    sentinel = "## Strategy search: beating buy-and-hold BTC with costs\n"
    if sentinel in existing:
        head, _ = existing.split(sentinel, 1)
        new = head + sentinel + "\n" + md
    else:
        new = existing.rstrip() + "\n\n" + sentinel + "\n" + md
    path.write_text(new, encoding="utf-8")
    logger.info("Wrote strategy-search results to %s", out_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=str, default="btc,eth,sol")
    p.add_argument("--start", type=str, default="2022-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--hmm-window", type=int, default=504)
    p.add_argument("--hmm-refit-every", type=int, default=63)
    p.add_argument("--threshold-l1", type=float, default=0.5)
    p.add_argument("--out", type=str,
                   default="causal_portfolio/docs/driver_selection_regimes.md")
    args = p.parse_args()

    assets = args.assets.split(",")
    metrics = run_search(
        assets, args.start, args.end,
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        hmm_window=args.hmm_window, hmm_refit_every=args.hmm_refit_every,
        threshold_l1=args.threshold_l1,
    )
    winner, notes = find_winner(metrics)

    print()
    print(format_table(metrics))
    print()
    for n in notes:
        print(f"  {n}")
    print(f"  Highest-Sharpe winner: {winner.name}")

    md = render_markdown(metrics, run_args={
        "assets": assets, "start": args.start, "end": args.end,
        "fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps,
        "hmm_window": args.hmm_window, "hmm_refit_every": args.hmm_refit_every,
        "threshold_l1": args.threshold_l1,
    }, winner=winner, notes=notes)
    append_to_doc(md, args.out)
    print(f"\nResults appended to {args.out}")


if __name__ == "__main__":
    main()
