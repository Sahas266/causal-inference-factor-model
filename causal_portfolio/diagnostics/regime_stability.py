"""Phase A: per-regime driver stability diagnostic.

Fits a Gaussian HMM on (VIX, BTC realized vol), labels every day with a
regime state, and runs `ComboDriverSelector.rank_all_subsets` separately
within each regime. Compares to the global pick from Option 2.

Output goes to the same markdown doc that Option 2 wrote to.

Usage:
    python -m causal_portfolio.diagnostics.regime_stability \\
        --assets btc,eth,sol,bnb,avax,uni,aave,link,doge \\
        --start 2022-01-01 --end 2025-12-31 \\
        --m 3 --n-states 2 \\
        --out causal_portfolio/docs/driver_selection_regimes.md
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.data import get_loader
from causal_portfolio.factors.builder import build_all_factors
from causal_portfolio.factors.combo_selector import ComboDriverSelector
from causal_portfolio.regimes.hmm import (
    RegimeClassifier,
    build_regime_features,
    dwell_stats,
)

logger = logging.getLogger("cpcm.diagnostics.regime_stability")


@dataclass
class RegimeWinner:
    """Top driver subset within one regime."""
    regime: int
    n_obs: int
    pct_of_sample: float
    winner: tuple[str, ...]
    score: float
    runner_up: tuple[str, ...]
    runner_up_score: float
    global_pick_rank: int   # 1-indexed rank of global subset within this regime
    global_pick_score: float


@dataclass
class RegimeReport:
    n_states: int
    feature_columns: list[str]
    transition_matrix: np.ndarray
    state_means: np.ndarray
    global_winner: tuple[str, ...]
    global_score: float
    dwell: dict[int, dict]
    regime_winners: list[RegimeWinner]
    n_total_obs: int


def _load_inputs(
    assets: list[str], start: str, end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loader = get_loader()
    all_metrics = [
        "PriceUSD", "price", "tvl_usd", "SplyCur",
        "stablecoin_circulating_usd", "FeeTotNtv",
        "FlowInExNtv", "FlowOutExNtv",
        "avg_gas_price_gwei", "avg_base_fee_gwei",
        "stddev_base_fee_gwei", "staking_apr",
        "cex_netflow_usd", "lp_net_flow_usd", "mev_revenue_eth",
    ]
    macro_series = ["DFF", "DGS10", "VIXCLS", "T10Y2Y",
                    "CPIAUCSL", "M2SL", "DTWEXBGS"]

    panel = loader.load_panel(assets, all_metrics, start, end)
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(macro_series, start, end)
    factors = build_all_factors(panel, macro)
    return factors, returns, macro


def analyze(
    assets: list[str],
    start: str,
    end: str,
    m: int = 3,
    n_states: int = 2,
    min_obs_per_regime: int = 60,
) -> RegimeReport:
    factors, returns, macro = _load_inputs(assets, start, end)
    available = factors.dropna(axis=1, how="all")

    # Global pick (same as Option 2)
    selector = ComboDriverSelector()
    m_actual = min(m, available.shape[1])
    global_ranking = selector.rank_all_subsets(returns, available, m=m_actual)
    global_winner, global_score = global_ranking[0]
    logger.info("Global winner: %s (score=%.4f)", global_winner, global_score)

    # Build regime features + fit HMM
    feats = build_regime_features(macro, returns)
    classifier = RegimeClassifier(n_states=n_states).fit(feats)
    # In-sample decode for the diagnostic (look-ahead OK at this stage)
    labels = classifier.decode(feats)
    label_series = pd.Series(labels, index=feats.index, name="regime")
    logger.info("Regime label distribution: %s",
                {int(k): int(v) for k, v in label_series.value_counts().items()})

    # Per-regime subset selection
    regime_winners: list[RegimeWinner] = []
    common = returns.index.intersection(available.index)
    for state in range(n_states):
        state_dates = label_series[label_series == state].index.intersection(common)
        if len(state_dates) < min_obs_per_regime:
            logger.warning("Regime %d has only %d obs — skipping selection",
                           state, len(state_dates))
            continue
        sub_returns = returns.loc[state_dates]
        sub_factors = available.loc[state_dates]
        # Skip cols entirely NaN within this regime
        sub_factors = sub_factors.dropna(axis=1, how="all")
        if sub_factors.shape[1] < m_actual:
            logger.warning("Regime %d has %d non-NaN candidate factors, need %d",
                           state, sub_factors.shape[1], m_actual)
            continue

        ranking = selector.rank_all_subsets(sub_returns, sub_factors, m=m_actual)
        winner, w_score = ranking[0]
        runner_up, ru_score = ranking[1] if len(ranking) > 1 else (winner, w_score)

        # Where does the global pick rank in this regime? Only valid if all
        # global drivers are still in the candidate pool.
        if all(d in sub_factors.columns for d in global_winner):
            global_rank = next(
                (i + 1 for i, (s, _) in enumerate(ranking) if s == global_winner),
                -1,
            )
            global_local_score = next(
                (score for s, score in ranking if s == global_winner),
                float("nan"),
            )
        else:
            global_rank = -1
            global_local_score = float("nan")

        regime_winners.append(RegimeWinner(
            regime=state,
            n_obs=len(state_dates),
            pct_of_sample=len(state_dates) / len(label_series),
            winner=winner, score=w_score,
            runner_up=runner_up, runner_up_score=ru_score,
            global_pick_rank=global_rank,
            global_pick_score=global_local_score,
        ))

    return RegimeReport(
        n_states=n_states,
        feature_columns=list(feats.columns),
        transition_matrix=classifier.transition_matrix(),
        state_means=classifier.state_means(),
        global_winner=global_winner,
        global_score=global_score,
        dwell=dwell_stats(labels),
        regime_winners=regime_winners,
        n_total_obs=len(label_series),
    )


# ── markdown rendering ──────────────────────────────────────────────


def _verdict(report: RegimeReport) -> str:
    """Apply the Phase A decision logic."""
    winners = [tuple(w.winner) for w in report.regime_winners]
    if len(winners) < 2:
        return ("**Phase A inconclusive.** Only one regime produced a valid "
                "selection. HMM may need more data, fewer states, or different "
                "features. Suggest re-running with `--n-states 2` and a wider "
                "date range, or trying a different feature set.")

    # Are the regime winners distinct from each other?
    distinct_winners = len(set(winners))
    # Are they also distinct from the global pick?
    distinct_from_global = sum(1 for w in winners if w != report.global_winner)

    if distinct_winners == 1 and winners[0] == report.global_winner:
        return ("**Phase B NOT justified.** All regimes pick the same drivers, "
                "and they match the global pick. Regime conditioning would add "
                "complexity without changing the strategy. Use the time for "
                "transaction costs or other backtest-fidelity improvements.")

    if distinct_winners == 1:
        return ("**Phase B partially justified.** All regimes pick the same "
                "drivers, but those differ from the global pick. Try replacing "
                "the global selector with this single regime-agnostic pick; "
                "full regime conditioning probably won't help further.")

    if distinct_from_global >= len(winners) - 1:
        return ("**Phase B JUSTIFIED.** Different regimes select different "
                "drivers, and the regime-specific picks differ from the global "
                "pick. This is the expected pattern when regime-conditional "
                "selection unlocks Sharpe. Proceed with full implementation.")

    return ("**Phase B partially justified.** Regimes show different winners "
            "but at least one matches the global pick. Worth implementing — "
            "the marginal regimes will benefit even if the dominant one doesn't.")


def render_markdown(report: RegimeReport, run_args: dict) -> str:
    lines: list[str] = []
    lines.append("### Run parameters\n")
    lines.append(f"- Date range: `{run_args['start']}` to `{run_args['end']}`")
    lines.append(f"- Assets: `{', '.join(run_args['assets'])}`")
    lines.append(f"- m (subset size): `{run_args['m']}`")
    lines.append(f"- HMM states: `{run_args['n_states']}`")
    lines.append(f"- Regime features: `{', '.join(report.feature_columns)}`")
    lines.append(f"- Total observations: **{report.n_total_obs}**")
    lines.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    lines.append("### HMM characterization\n")
    lines.append("State means (z-scored features, sorted by feature-0 mean):\n")
    lines.append(f"| State | {' | '.join(report.feature_columns)} | Interpretation |")
    lines.append("|---:|" + "---:|" * len(report.feature_columns) + "---|")
    interpretations = ["low-stress (risk-on)", "high-stress (risk-off)"]
    for i, mean_row in enumerate(report.state_means):
        cells = " | ".join(f"{x:+.3f}" for x in mean_row)
        interp = interpretations[i] if i < len(interpretations) else f"state {i}"
        lines.append(f"| {i} | {cells} | {interp} |")
    lines.append("")

    lines.append("Transition matrix `P[i,j] = P(next=j | current=i)`:\n")
    lines.append("| From \\ To | " + " | ".join(f"State {j}" for j in range(report.n_states)) + " |")
    lines.append("|---:|" + "---:|" * report.n_states)
    for i, row in enumerate(report.transition_matrix):
        cells = " | ".join(f"{x:.3f}" for x in row)
        lines.append(f"| State {i} | {cells} |")
    lines.append("")

    lines.append("### Regime dwell statistics\n")
    lines.append("| State | Days | % of sample | # of runs | Mean run length | Max run length |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for state, s in report.dwell.items():
        lines.append(
            f"| {state} | {s['days']} | {s['pct']:.1%} | {s['n_runs']} | "
            f"{s['mean_run_length']:.1f} | {s['max_run_length']} |"
        )
    lines.append("")

    lines.append("### Per-regime winning subsets\n")
    lines.append("| Regime | n_obs | % | Winner | Score | Runner-up | Global pick rank |")
    lines.append("|---:|---:|---:|---|---:|---|---:|")
    for w in report.regime_winners:
        gr = "?" if w.global_pick_rank < 0 else str(w.global_pick_rank)
        lines.append(
            f"| {w.regime} | {w.n_obs} | {w.pct_of_sample:.1%} | "
            f"`{list(w.winner)}` | {w.score:.4f} | `{list(w.runner_up)}` | {gr} |"
        )
    lines.append("")
    lines.append(f"**Global pick (Option 2 baseline):** `{list(report.global_winner)}` "
                 f"(score `{report.global_score:.4f}`)\n")

    lines.append("### Driver overlap across regimes\n")
    if len(report.regime_winners) >= 2:
        all_drivers = sorted(set().union(*(set(w.winner) for w in report.regime_winners)))
        lines.append("| Driver | " +
                     " | ".join(f"R{w.regime}" for w in report.regime_winners) +
                     " | In global pick? |")
        lines.append("|---|" + "---|" * (len(report.regime_winners) + 1))
        for d in all_drivers:
            cells = []
            for w in report.regime_winners:
                cells.append("✓" if d in w.winner else "—")
            in_global = "✓" if d in report.global_winner else "—"
            lines.append(f"| `{d}` | " + " | ".join(cells) + f" | {in_global} |")
        lines.append("")

    lines.append("### Verdict\n")
    lines.append(_verdict(report))
    lines.append("")
    return "\n".join(lines)


def append_to_doc(md: str, out_path: str) -> None:
    """Append the Phase A markdown to the doc, replacing any prior block."""
    path = Path(out_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    sentinel = "## Phase A: HMM regime-conditional selection results\n"

    if sentinel in existing:
        head, _ = existing.split(sentinel, 1)
        new = head + sentinel + "\n" + md
    else:
        new = existing.rstrip() + "\n\n" + sentinel + "\n" + md
    path.write_text(new, encoding="utf-8")
    logger.info("Wrote Phase A results to %s", out_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=str, default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", type=str, default="2022-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--min-obs-per-regime", type=int, default=60)
    p.add_argument("--out", type=str,
                   default="causal_portfolio/docs/driver_selection_regimes.md")
    args = p.parse_args()

    assets = args.assets.split(",")
    report = analyze(
        assets=assets, start=args.start, end=args.end,
        m=args.m, n_states=args.n_states,
        min_obs_per_regime=args.min_obs_per_regime,
    )
    md = render_markdown(report, run_args={
        "assets": assets, "start": args.start, "end": args.end,
        "m": args.m, "n_states": args.n_states,
    })

    # Console summary
    print("=" * 60)
    print(f"HMM ({args.n_states} states) regime characterization:")
    for state in range(args.n_states):
        s = report.dwell.get(state, {})
        print(f"  State {state}: {s.get('days', 0)} days "
              f"({s.get('pct', 0):.1%} of sample), {s.get('n_runs', 0)} runs")
    print()
    for w in report.regime_winners:
        print(f"  Regime {w.regime}: {list(w.winner)}  score={w.score:.4f}")
    print(f"  Global:    {list(report.global_winner)}  score={report.global_score:.4f}")
    print("=" * 60)

    append_to_doc(md, args.out)
    print(f"\nResults appended to {args.out}")


if __name__ == "__main__":
    main()
