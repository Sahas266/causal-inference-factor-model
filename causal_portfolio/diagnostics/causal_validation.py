"""Phase B step 1: validate that causal regime labeling matches Phase A.

Compares two pipelines on the same data:
  - Non-causal (Phase A): fit HMM on full sample, Viterbi decode.
  - Causal (production-grade): rolling-window fit + forward filter.

Reports:
  - Label agreement between the two label sequences
  - Per-state agreement (state 0 vs state 1 separately — one regime may be
    easier to identify causally than the other)
  - Whether per-regime driver winners survive the switch to causal labels

If agreement is high (>85%) and winners survive, Phase A's results carry
over to a real-time pipeline. If not, we either change features or accept
a smaller realized lift.

Usage:
    python -m causal_portfolio.diagnostics.causal_validation \\
        --start 2022-01-01 --end 2025-12-31 \\
        --n-states 2 --window 252 --refit-every 21
"""

from __future__ import annotations

import argparse
import logging
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
    rolling_fit_decode,
)

logger = logging.getLogger("cpcm.diagnostics.causal_validation")


@dataclass
class CausalCompare:
    n_states: int
    n_total: int
    n_compared: int                 # excludes warmup NaN rows in causal labels
    overall_agreement: float
    per_state_agreement: dict[int, float]
    non_causal_winners: dict[int, tuple[str, ...]]
    causal_winners: dict[int, tuple[str, ...]]
    causal_dwell: dict[int, dict]
    non_causal_dwell: dict[int, dict]
    window: int
    refit_every: int


def analyze(
    assets: list[str],
    start: str,
    end: str,
    m: int = 3,
    n_states: int = 2,
    window: int = 252,
    refit_every: int = 21,
    n_restarts: int = 5,
    min_obs_per_regime: int = 60,
) -> CausalCompare:
    # ── Load data (mirror regime_stability.py) ───────────────────────
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
    available = factors.dropna(axis=1, how="all")
    feats = build_regime_features(macro, returns)

    # ── Non-causal labels (Phase A baseline) ──────────────────────────
    nc_classifier = RegimeClassifier(n_states=n_states, n_restarts=n_restarts).fit(feats)
    nc_labels = pd.Series(nc_classifier.decode(feats), index=feats.index, name="nc")
    logger.info("Non-causal labels: %s",
                {int(k): int(v) for k, v in nc_labels.value_counts().items()})

    # ── Causal labels (rolling fit + forward filter) ─────────────────
    causal_labels = rolling_fit_decode(
        feats, window_size=window, refit_every=refit_every,
        n_states=n_states, n_restarts=n_restarts,
    )
    logger.info("Causal labels: %s",
                {int(k): int(v) for k, v in causal_labels.dropna().value_counts().items()})

    # ── Compare ─────────────────────────────────────────────────────
    compared = causal_labels.dropna()
    nc_aligned = nc_labels.loc[compared.index]
    overall = float((compared.values == nc_aligned.values).mean())

    per_state = {}
    for state in range(n_states):
        mask = nc_aligned == state
        if mask.sum() > 0:
            per_state[state] = float((compared.loc[mask].values == state).mean())
        else:
            per_state[state] = float("nan")

    # ── Per-regime winners under each label set ──────────────────────
    selector = ComboDriverSelector()
    m_actual = min(m, available.shape[1])

    nc_winners: dict[int, tuple[str, ...]] = {}
    causal_winners: dict[int, tuple[str, ...]] = {}

    for state in range(n_states):
        # Non-causal subset
        nc_dates = nc_labels[nc_labels == state].index.intersection(returns.index).intersection(available.index)
        if len(nc_dates) >= min_obs_per_regime:
            nc_F = available.loc[nc_dates].dropna(axis=1, how="all")
            if nc_F.shape[1] >= m_actual:
                ranking = selector.rank_all_subsets(returns.loc[nc_dates], nc_F, m=m_actual)
                nc_winners[state] = ranking[0][0]

        # Causal subset
        c_dates = causal_labels[causal_labels == state].dropna().index.intersection(returns.index).intersection(available.index)
        if len(c_dates) >= min_obs_per_regime:
            c_F = available.loc[c_dates].dropna(axis=1, how="all")
            if c_F.shape[1] >= m_actual:
                ranking = selector.rank_all_subsets(returns.loc[c_dates], c_F, m=m_actual)
                causal_winners[state] = ranking[0][0]

    return CausalCompare(
        n_states=n_states,
        n_total=len(feats),
        n_compared=len(compared),
        overall_agreement=overall,
        per_state_agreement=per_state,
        non_causal_winners=nc_winners,
        causal_winners=causal_winners,
        causal_dwell=dwell_stats(compared.values.astype(int)),
        non_causal_dwell=dwell_stats(nc_labels.values),
        window=window,
        refit_every=refit_every,
    )


def _verdict(report: CausalCompare) -> str:
    """Decision for whether to proceed with backtester wiring."""
    if report.overall_agreement >= 0.85:
        # Winners should match too
        same_winners = all(
            report.causal_winners.get(s) == report.non_causal_winners.get(s)
            for s in range(report.n_states)
            if s in report.causal_winners and s in report.non_causal_winners
        )
        if same_winners:
            return ("**Causal labels reproduce Phase A.** Overall agreement "
                    f"is {report.overall_agreement:.1%} and per-regime "
                    "winners match. **Proceed with backtester wiring** — the "
                    "in-sample lift should largely survive in real-time use.")
        return ("**Partial validation.** Label agreement is high "
                f"({report.overall_agreement:.1%}) but at least one per-"
                "regime winner differs between causal and non-causal labels. "
                "Backtester wiring is still worth trying; expect a haircut "
                "on the in-sample lift.")

    if report.overall_agreement >= 0.70:
        return ("**Causal labels noisier than expected.** Overall agreement "
                f"is {report.overall_agreement:.1%}. Rolling-window fit "
                "produces somewhat different regime structure than full-"
                "sample. Try widening the window, increasing refit cadence, "
                "or switching features before backtester wiring.")

    return ("**Causal validation FAILED.** Overall agreement is only "
            f"{report.overall_agreement:.1%}. The regime structure detected "
            "in Phase A does not survive causal identification. Either the "
            "structure is genuinely non-real-time (it required future info "
            "to spot), or the features need to be reconsidered. **Do not "
            "proceed with full Phase B until this is resolved.**")


def render_markdown(report: CausalCompare, run_args: dict) -> str:
    lines: list[str] = []
    lines.append("### Run parameters\n")
    lines.append(f"- Date range: `{run_args['start']}` → `{run_args['end']}`")
    lines.append(f"- HMM states: `{report.n_states}`")
    lines.append(f"- Rolling window: `{report.window}` days")
    lines.append(f"- Refit cadence: every `{report.refit_every}` days")
    lines.append(f"- Total observations: **{report.n_total}**")
    lines.append(f"- Causal-labeled observations (post warmup): **{report.n_compared}**")
    lines.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    lines.append("### Label agreement\n")
    lines.append(f"- **Overall agreement: {report.overall_agreement:.1%}**\n")
    lines.append("Per-state agreement (how often causal label matches non-causal label, conditional on non-causal state):\n")
    lines.append("| Non-causal state | Causal agreement |")
    lines.append("|---:|---:|")
    for state, pct in report.per_state_agreement.items():
        lines.append(f"| {state} | {pct:.1%} |")
    lines.append("")

    lines.append("### Dwell statistics comparison\n")
    lines.append("| State | Non-causal days / runs | Causal days / runs |")
    lines.append("|---:|---|---|")
    for state in range(report.n_states):
        nc = report.non_causal_dwell.get(state, {})
        c = report.causal_dwell.get(state, {})
        nc_str = f"{nc.get('days', 0)} / {nc.get('n_runs', 0)} runs (mean {nc.get('mean_run_length', 0):.0f}d)"
        c_str = f"{c.get('days', 0)} / {c.get('n_runs', 0)} runs (mean {c.get('mean_run_length', 0):.0f}d)"
        lines.append(f"| {state} | {nc_str} | {c_str} |")
    lines.append("")

    lines.append("### Per-regime winners comparison\n")
    lines.append("| State | Non-causal winner | Causal winner | Match |")
    lines.append("|---:|---|---|---|")
    for state in range(report.n_states):
        nc = report.non_causal_winners.get(state)
        c = report.causal_winners.get(state)
        nc_str = f"`{list(nc)}`" if nc else "—"
        c_str = f"`{list(c)}`" if c else "—"
        match = "✓" if nc == c and nc is not None else "—"
        lines.append(f"| {state} | {nc_str} | {c_str} | {match} |")
    lines.append("")

    lines.append("### Verdict\n")
    lines.append(_verdict(report))
    lines.append("")
    return "\n".join(lines)


def append_to_doc(md: str, out_path: str) -> None:
    path = Path(out_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    sentinel = "## Phase B step 1: causal label validation\n"

    if sentinel in existing:
        head, _ = existing.split(sentinel, 1)
        new = head + sentinel + "\n" + md
    else:
        new = existing.rstrip() + "\n\n" + sentinel + "\n" + md
    path.write_text(new, encoding="utf-8")
    logger.info("Wrote causal validation results to %s", out_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=str, default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", type=str, default="2022-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--refit-every", type=int, default=21)
    p.add_argument("--n-restarts", type=int, default=5)
    p.add_argument("--out", type=str,
                   default="causal_portfolio/docs/driver_selection_regimes.md")
    args = p.parse_args()

    assets = args.assets.split(",")
    report = analyze(
        assets=assets, start=args.start, end=args.end,
        m=args.m, n_states=args.n_states,
        window=args.window, refit_every=args.refit_every,
        n_restarts=args.n_restarts,
    )
    md = render_markdown(report, run_args={
        "start": args.start, "end": args.end,
    })

    print("=" * 60)
    print(f"Overall agreement: {report.overall_agreement:.1%}")
    print(f"Per-state agreement: "
          f"{ {s: f'{p:.1%}' for s, p in report.per_state_agreement.items()} }")
    for state in range(report.n_states):
        nc = report.non_causal_winners.get(state)
        c = report.causal_winners.get(state)
        match = "MATCH" if nc == c else "DIFFER"
        print(f"  state {state}: NC={list(nc) if nc else '?'}  causal={list(c) if c else '?'}  [{match}]")
    print("=" * 60)

    append_to_doc(md, args.out)
    print(f"\nResults appended to {args.out}")


if __name__ == "__main__":
    main()
