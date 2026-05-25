"""Sweep posterior-gating, stop-loss, and cost variations on REGIME_EW.

Three sweeps, all run on the OOS-clean window (2024-01-01 → 2026-01-01)
where REGIME_EW underperformed BH_BTC by -42pp. Question: can any of
these production-readiness features recover the lift?

Baselines for reference:
  - BH_BTC over 2024-01-01 → 2026-01-01: +58.4% / Sharpe 0.638 / -33.1% DD
  - REGIME_EW (vanilla) over same:        +16.6% / Sharpe 0.375 / -43.2% DD

Sweeps:
  A. Posterior gating: min_posterior_to_switch ∈ {0.0, 0.55, 0.65, 0.75, 0.85}
  B. Stop-loss:        stop_loss_pct ∈ {None, 0.15, 0.20, 0.25, 0.30}
  C. Cost sensitivity: (fee_bps, slippage_bps) ∈ several combos

Output appended to regime_ew_production_readiness.md.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import (
    average_turnover, max_drawdown, sharpe_ratio, sortino_ratio,
)
from causal_portfolio.backtest.strategies import (
    buy_and_hold, regime_gated_long_only,
)
from causal_portfolio.data import get_loader

logger = logging.getLogger("cpcm.hardening")


def _load(assets, start, end):
    loader = get_loader()
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(["VIXCLS"], start, end)
    common = returns.index.intersection(macro.index)
    return returns.loc[common], macro.loc[common].ffill()


def _slice_metrics(result, index, s_ts, e_ts) -> dict:
    n = len(result.returns_series)
    aligned = index[-n:]
    mask = (aligned >= s_ts) & (aligned < e_ts)
    if mask.sum() == 0:
        return {"n_obs": 0}
    r = result.returns_series[mask]
    w = result.weights_history[mask]
    pv = np.cumprod(1 + r)
    return {
        "n_obs": int(mask.sum()),
        "total": float(pv[-1] / pv[0] - 1),
        "sharpe": float(sharpe_ratio(r)),
        "sortino": float(sortino_ratio(r)),
        "max_dd": float(max_drawdown(r)),
        "turnover": float(average_turnover(w)),
    }


@dataclass
class SweepRow:
    label: str
    metric: dict


def run_sweep_posterior_gate(returns, macro, oos_start, oos_end):
    universe = {"btc": 1.0, "eth": 1.0, "sol": 1.0}
    base_kwargs = dict(
        hmm_window=504, hmm_refit_every=63, n_states=2, n_restarts=5,
        fee_bps=5.0, slippage_bps=5.0, threshold_l1=0.10,
        universe_weights=universe,
    )
    rows: list[SweepRow] = []
    for gate in [0.0, 0.55, 0.65, 0.75, 0.85]:
        logger.info("posterior gate = %.2f", gate)
        r = regime_gated_long_only(
            returns, macro, **base_kwargs, min_posterior_to_switch=gate,
        )
        m = _slice_metrics(r, returns.index, oos_start, oos_end)
        rows.append(SweepRow(f"gate={gate:.2f}", m))
    return rows


def run_sweep_stop_loss(returns, macro, oos_start, oos_end):
    universe = {"btc": 1.0, "eth": 1.0, "sol": 1.0}
    base_kwargs = dict(
        hmm_window=504, hmm_refit_every=63, n_states=2, n_restarts=5,
        fee_bps=5.0, slippage_bps=5.0, threshold_l1=0.10,
        universe_weights=universe,
    )
    rows: list[SweepRow] = []
    for stop in [None, 0.15, 0.20, 0.25, 0.30]:
        label = "stop=off" if stop is None else f"stop={stop:.0%}"
        logger.info("stop-loss = %s", label)
        r = regime_gated_long_only(
            returns, macro, **base_kwargs, stop_loss_pct=stop,
        )
        m = _slice_metrics(r, returns.index, oos_start, oos_end)
        rows.append(SweepRow(label, m))
    return rows


def run_sweep_cost(returns, macro, oos_start, oos_end):
    universe = {"btc": 1.0, "eth": 1.0, "sol": 1.0}
    rows: list[SweepRow] = []
    for fee, slip in [(0, 0), (5, 5), (10, 10), (20, 20), (5, 30), (30, 30)]:
        logger.info("cost = (%d, %d) bps", fee, slip)
        r = regime_gated_long_only(
            returns, macro,
            universe_weights=universe,
            hmm_window=504, hmm_refit_every=63, n_states=2, n_restarts=5,
            threshold_l1=0.10,
            fee_bps=float(fee), slippage_bps=float(slip),
        )
        m = _slice_metrics(r, returns.index, oos_start, oos_end)
        rows.append(SweepRow(f"fee={fee} slip={slip}", m))
    return rows


# ── reporting ──────────────────────────────────────────────────────


def render_table(title: str, rows: list[SweepRow], baseline: dict) -> str:
    lines = [f"### {title}\n"]
    lines.append(f"Baseline (BH_BTC over OOS): total {baseline['total']:+.1%}, "
                 f"Sharpe {baseline['sharpe']:.3f}, MaxDD {baseline['max_dd']:+.1%}\n")
    lines.append("| Config | n_obs | Total | Sharpe | Sortino | MaxDD | Turnover | Beats BH? |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        m = row.metric
        if m.get("n_obs", 0) == 0:
            lines.append(f"| {row.label} | 0 | — | — | — | — | — | — |")
            continue
        beats = "✓" if m["total"] > baseline["total"] else "—"
        lines.append(
            f"| {row.label} | {m['n_obs']} | {m['total']:+.1%} | "
            f"{m['sharpe']:.3f} | {m['sortino']:.3f} | {m['max_dd']:+.1%} | "
            f"{m['turnover']:.3f} | {beats} |"
        )
    lines.append("")
    return "\n".join(lines)


def append_to_doc(md_chunks: list[str], out_path: str, section: str) -> None:
    path = Path(out_path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    sentinel = f"## {section}"
    body = "## " + section + "\n\n" + "\n".join(md_chunks) + "\n"
    if sentinel in text:
        before, after = text.split(sentinel, 1)
        # remove old block until next H2
        rest = after.split("\n## ", 1)
        new_after = ("\n## " + rest[1]) if len(rest) > 1 else ""
        new = before + body + new_after
    else:
        new = text.rstrip() + "\n\n" + body
    path.write_text(new, encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--oos-start", default="2024-01-01")
    p.add_argument("--oos-end", default="2026-01-01")
    p.add_argument("--out", default="causal_portfolio/docs/regime_ew_production_readiness.md")
    args = p.parse_args()

    assets = args.assets.split(",")
    returns, macro = _load(assets, args.start, args.end)

    # BH_BTC baseline on OOS window
    bh = buy_and_hold(returns, "btc_return", fee_bps=5.0, slippage_bps=5.0)
    oos_s = pd.Timestamp(args.oos_start)
    oos_e = pd.Timestamp(args.oos_end)
    bh_oos = _slice_metrics(bh, returns.index, oos_s, oos_e)

    print(f"\nBH_BTC OOS baseline ({args.oos_start} to {args.oos_end}):")
    print(f"  total {bh_oos['total']:+.1%}  Sharpe {bh_oos['sharpe']:.3f}  MaxDD {bh_oos['max_dd']:+.1%}")

    print("\n--- Sweep A: posterior gating ---")
    rows_a = run_sweep_posterior_gate(returns, macro, oos_s, oos_e)
    for r in rows_a:
        print(f"  {r.label:<20} total {r.metric.get('total',0):+.1%}  Sharpe {r.metric.get('sharpe',0):.3f}  MaxDD {r.metric.get('max_dd',0):+.1%}")

    print("\n--- Sweep B: stop-loss ---")
    rows_b = run_sweep_stop_loss(returns, macro, oos_s, oos_e)
    for r in rows_b:
        print(f"  {r.label:<20} total {r.metric.get('total',0):+.1%}  Sharpe {r.metric.get('sharpe',0):.3f}  MaxDD {r.metric.get('max_dd',0):+.1%}")

    print("\n--- Sweep C: cost sensitivity ---")
    rows_c = run_sweep_cost(returns, macro, oos_s, oos_e)
    for r in rows_c:
        print(f"  {r.label:<20} total {r.metric.get('total',0):+.1%}  Sharpe {r.metric.get('sharpe',0):.3f}  MaxDD {r.metric.get('max_dd',0):+.1%}")

    # Render markdown
    chunks = [
        f"OOS window: `{args.oos_start}` → `{args.oos_end}`.  "
        f"BH_BTC baseline: total {bh_oos['total']:+.1%}, Sharpe {bh_oos['sharpe']:.3f}, MaxDD {bh_oos['max_dd']:+.1%}.  "
        f"Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n",
        render_table("Sweep A: posterior-confidence gating", rows_a, bh_oos),
        render_table("Sweep B: stop-loss overlay", rows_b, bh_oos),
        render_table("Sweep C: cost sensitivity", rows_c, bh_oos),
    ]
    append_to_doc(chunks, args.out, "Sweeps on OOS window (Tasks 2, 3, 5)")
    print(f"\nResults appended to {args.out}")


if __name__ == "__main__":
    main()
