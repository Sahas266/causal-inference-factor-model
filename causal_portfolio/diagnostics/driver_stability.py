"""Driver-stability analysis (Option 2 from driver_selection_regimes.md).

Splits a date range into rolling sub-windows, runs the Combo driver selector
inside each window, and reports how stable the "top m-subset" is across
windows. Pure diagnostic — does not modify the strategy.

Usage:
    python -m causal_portfolio.diagnostics.driver_stability \\
        --assets btc,eth,sol,bnb,avax,uni,aave,link,doge \\
        --start 2022-01-01 --end 2025-12-31 \\
        --m 3 --window-days 365 --step-days 60 \\
        --out causal_portfolio/docs/driver_selection_regimes.md
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.data import get_loader
from causal_portfolio.factors.builder import (
    FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS, build_all_factors,
)
from causal_portfolio.factors.combo_selector import ComboDriverSelector

logger = logging.getLogger("cpcm.diagnostics.driver_stability")


@dataclass
class WindowResult:
    """Result of running rank_all_subsets on a single sub-window."""
    start: pd.Timestamp
    end: pd.Timestamp
    n_obs: int
    n_candidates: int
    winning_subset: tuple[str, ...]
    winning_score: float
    runner_up: tuple[str, ...]
    runner_up_score: float
    global_subset_rank: int   # 1-indexed rank of the *global* winner here
    global_subset_score: float


@dataclass
class StabilityReport:
    """Aggregate stability metrics across all sub-windows."""
    n_windows: int
    global_winner: tuple[str, ...]
    global_score: float
    window_results: list[WindowResult]
    subset_win_counts: Counter         # subset_tuple → count
    driver_appearance_counts: Counter  # driver_name → count of winning subsets it appeared in
    global_winner_avg_rank: float
    global_winner_pct_optimal: float   # fraction of windows where it was rank-1


def _load_inputs(
    assets: list[str], start: str, end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (factors_df, returns_df) aligned on date."""
    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, start, end)
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    factors = build_all_factors(panel, macro)
    return factors, returns


def analyze(
    assets: list[str],
    start: str,
    end: str,
    m: int = 3,
    window_days: int = 365,
    step_days: int = 60,
    min_obs: int = 60,
) -> StabilityReport:
    """Run the stability analysis. See module docstring."""
    factors, returns = _load_inputs(assets, start, end)
    available = factors.dropna(axis=1, how="all")
    logger.info("Loaded %d factors, %d return columns", available.shape[1], returns.shape[1])

    # Global selection over the full range — our baseline
    selector = ComboDriverSelector()
    m_actual = min(m, available.shape[1])
    global_ranking = selector.rank_all_subsets(returns, available, m=m_actual)
    global_winner, global_score = global_ranking[0]
    logger.info("Global winner: %s (score=%.4f)", global_winner, global_score)

    # Sub-windows
    common = returns.index.intersection(available.index)
    if not isinstance(common, pd.DatetimeIndex):
        common = pd.DatetimeIndex(common)
    full_start = common.min()
    full_end = common.max()
    win_size = timedelta(days=window_days)
    step = timedelta(days=step_days)

    window_results: list[WindowResult] = []
    subset_wins: Counter = Counter()
    driver_appearances: Counter = Counter()

    cursor = full_start
    while cursor + win_size <= full_end:
        w_start = cursor
        w_end = cursor + win_size
        slice_mask = (common >= w_start) & (common < w_end)
        sub_dates = common[slice_mask]
        sub_returns = returns.loc[sub_dates]
        sub_factors = available.loc[sub_dates]

        # Skip windows with too few aligned observations
        aligned = sub_returns.dropna().index.intersection(sub_factors.dropna(how="any").index)
        if len(aligned) < min_obs:
            logger.debug("Skipping window %s → %s: only %d aligned obs",
                         w_start.date(), w_end.date(), len(aligned))
            cursor += step
            continue

        local_ranking = selector.rank_all_subsets(sub_returns, sub_factors, m=m_actual)
        winner, w_score = local_ranking[0]
        runner_up, ru_score = local_ranking[1] if len(local_ranking) > 1 else (winner, w_score)

        # Where does the *global* winner rank in this sub-window?
        global_rank = next(
            (i + 1 for i, (s, _) in enumerate(local_ranking) if s == global_winner),
            -1,
        )
        global_local_score = next(
            (score for s, score in local_ranking if s == global_winner),
            float("nan"),
        )

        window_results.append(WindowResult(
            start=w_start, end=w_end, n_obs=len(aligned),
            n_candidates=available.shape[1],
            winning_subset=winner, winning_score=w_score,
            runner_up=runner_up, runner_up_score=ru_score,
            global_subset_rank=global_rank,
            global_subset_score=global_local_score,
        ))

        subset_wins[winner] += 1
        for d in winner:
            driver_appearances[d] += 1

        cursor += step

    if not window_results:
        raise ValueError(
            f"No sub-windows had >= {min_obs} aligned observations. "
            f"Try a wider --window-days or shorter --step-days."
        )

    avg_rank = float(np.mean([w.global_subset_rank for w in window_results
                              if w.global_subset_rank > 0]))
    pct_optimal = float(np.mean([1.0 if w.global_subset_rank == 1 else 0.0
                                 for w in window_results]))

    return StabilityReport(
        n_windows=len(window_results),
        global_winner=global_winner, global_score=global_score,
        window_results=window_results,
        subset_win_counts=subset_wins,
        driver_appearance_counts=driver_appearances,
        global_winner_avg_rank=avg_rank,
        global_winner_pct_optimal=pct_optimal,
    )


# ── markdown rendering ──────────────────────────────────────────────


def _interpretation(report: StabilityReport) -> str:
    """Apply the decision tree from the options doc to the report."""
    pct = report.global_winner_pct_optimal
    if pct >= 0.70:
        verdict = (
            "**Skip Option 3.** The global driver pick is robust across "
            "sub-windows. Time is better spent on transaction costs, "
            "slippage modeling, and other backtest-fidelity improvements."
        )
    elif pct >= 0.30:
        verdict = (
            "**Maybe Option 3, lightweight.** Subsets shift across sub-"
            "windows but the same drivers tend to dominate. Try per-regime "
            "β fitting first; keep global driver selection unchanged."
        )
    else:
        verdict = (
            "**Option 3 justified.** The global pick is rarely optimal in "
            "any specific sub-window. Full HMM + per-regime driver selection "
            "has the highest expected payoff."
        )
    return verdict


def render_markdown(report: StabilityReport, run_args: dict) -> str:
    """Render the report as markdown ready to append to the doc."""
    lines: list[str] = []
    lines.append("### Run parameters\n")
    lines.append(f"- Date range: `{run_args['start']}` to `{run_args['end']}`")
    lines.append(f"- Assets: `{', '.join(run_args['assets'])}`")
    lines.append(f"- m (subset size): `{run_args['m']}`")
    lines.append(f"- Window: `{run_args['window_days']}` days, step `{run_args['step_days']}` days")
    lines.append(f"- Sub-windows analyzed: **{report.n_windows}**")
    lines.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    lines.append("### Global winner (current strategy pick)\n")
    lines.append(f"- Drivers: `{list(report.global_winner)}`")
    lines.append(f"- Commonality score: `{report.global_score:.4f}`")
    lines.append(f"- Rank-1 in sub-windows: **{report.global_winner_pct_optimal:.1%}**")
    lines.append(f"- Average rank across sub-windows: **{report.global_winner_avg_rank:.2f}**\n")

    lines.append("### Subset win frequency (top 10)\n")
    lines.append("| Rank | Subset | Wins | % of windows |")
    lines.append("|---:|---|---:|---:|")
    total = report.n_windows
    for i, (subset, count) in enumerate(report.subset_win_counts.most_common(10), 1):
        marker = " ← global" if subset == report.global_winner else ""
        lines.append(f"| {i} | `{list(subset)}`{marker} | {count} | {count/total:.1%} |")
    lines.append("")

    lines.append("### Individual driver win frequency\n")
    lines.append("How often each driver appeared in the *winning* m-subset.\n")
    lines.append("| Driver | Appearances | % of windows |")
    lines.append("|---|---:|---:|")
    for driver, count in report.driver_appearance_counts.most_common():
        marker = " ← in global pick" if driver in report.global_winner else ""
        lines.append(f"| `{driver}`{marker} | {count} | {count/total:.1%} |")
    lines.append("")

    lines.append("### Per-window winners (chronological)\n")
    lines.append("| Window | n_obs | Winner | Score | Global pick rank |")
    lines.append("|---|---:|---|---:|---:|")
    for w in report.window_results:
        gr = "?" if w.global_subset_rank < 0 else str(w.global_subset_rank)
        lines.append(
            f"| {w.start.date()} → {w.end.date()} | {w.n_obs} | "
            f"`{list(w.winning_subset)}` | {w.winning_score:.4f} | {gr} |"
        )
    lines.append("")

    lines.append("### Interpretation\n")
    lines.append(_interpretation(report))
    lines.append("")
    return "\n".join(lines)


def append_to_doc(md: str, out_path: str) -> None:
    """Append the rendered markdown to the doc's results section, replacing
    any previous results block."""
    path = Path(out_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    sentinel_open = "## Option 2 results\n"
    sentinel_close = None  # we just replace everything after the sentinel

    if sentinel_open in existing:
        head, _ = existing.split(sentinel_open, 1)
        new = head + sentinel_open + "\n" + md
    else:
        new = existing.rstrip() + "\n\n" + sentinel_open + "\n" + md
    path.write_text(new, encoding="utf-8")
    logger.info("Wrote results block to %s", out_path)


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=str, default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", type=str, default="2022-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--window-days", type=int, default=365)
    p.add_argument("--step-days", type=int, default=60)
    p.add_argument("--min-obs", type=int, default=60)
    p.add_argument("--out", type=str,
                   default="causal_portfolio/docs/driver_selection_regimes.md")
    args = p.parse_args()

    assets = args.assets.split(",")
    report = analyze(
        assets=assets, start=args.start, end=args.end, m=args.m,
        window_days=args.window_days, step_days=args.step_days,
        min_obs=args.min_obs,
    )
    md = render_markdown(report, run_args={
        "assets": assets, "start": args.start, "end": args.end,
        "m": args.m, "window_days": args.window_days, "step_days": args.step_days,
    })

    # Print a short summary to stdout
    print("=" * 60)
    print(f"Sub-windows: {report.n_windows}")
    print(f"Global pick: {list(report.global_winner)}")
    print(f"  rank-1 frequency: {report.global_winner_pct_optimal:.1%}")
    print(f"  avg rank:         {report.global_winner_avg_rank:.2f}")
    print()
    print("Top driver appearance:")
    for d, c in report.driver_appearance_counts.most_common(10):
        print(f"  {d:<25}  {c:>3}  ({c/report.n_windows:.1%})")
    print("=" * 60)

    append_to_doc(md, args.out)
    print(f"\nResults appended to {args.out}")


if __name__ == "__main__":
    main()
