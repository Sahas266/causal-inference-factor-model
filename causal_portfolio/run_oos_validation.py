"""Out-of-sample validation for REGIME_EW.

Runs the same strategy (no hyperparameter retuning) on two held-out periods:

  - First-half:  2022-01-01 to 2023-12-31 (includes 2022 selloff)
  - Second-half: 2024-01-01 to 2025-12-31 (recovery + bull)

If the lift over BH_BTC survives in both, the strategy is robust. If it
survives only in the half that includes the selloff, the regime gate is
just "avoid 2022" rather than a generally-useful signal. If it survives
in neither, the headline result was likely overfit.

Important nuance: the HMM still needs trailing data for its rolling
windows. We load a wider date range and let the walk-forward backtest
warm up; metrics are reported only on the held-out window.

Usage:
    python -m causal_portfolio.run_oos_validation
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.backtest.engine import BacktestResult
from causal_portfolio.backtest.metrics import (
    average_turnover, calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio,
)
from causal_portfolio.backtest.strategies import (
    buy_and_hold, regime_gated_long_only,
)
from causal_portfolio.data import get_loader

logger = logging.getLogger("cpcm.oos")


def _load(assets, start, end):
    loader = get_loader()
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(["VIXCLS"], start, end)
    common = returns.index.intersection(macro.index)
    return returns.loc[common], macro.loc[common].ffill()


def _slice_metrics(result: BacktestResult, returns_index, slice_start, slice_end) -> dict:
    """Recompute metrics over the slice [slice_start, slice_end) of result."""
    # result.returns_series is aligned with the LAST len(returns_series) rows
    # of returns_index (the warmup eats the head).
    n_returns = len(result.returns_series)
    aligned_index = returns_index[-n_returns:]
    mask = (aligned_index >= slice_start) & (aligned_index < slice_end)
    if mask.sum() == 0:
        return {"slice_start": str(slice_start), "slice_end": str(slice_end),
                "n_obs": 0}
    r = result.returns_series[mask]
    w = result.weights_history[mask]
    port_values = np.cumprod(1 + r)
    return {
        "slice_start": str(slice_start),
        "slice_end": str(slice_end),
        "n_obs": int(mask.sum()),
        "total_return": float(port_values[-1] / port_values[0] - 1),
        "sharpe": float(sharpe_ratio(r)),
        "sortino": float(sortino_ratio(r)),
        "max_dd": float(max_drawdown(r)),
        "calmar": float(calmar_ratio(r)),
        "avg_turnover": float(average_turnover(w)),
    }


def run_oos(
    assets: list[str] = None,
    full_start: str = "2022-01-01",
    full_end: str = "2025-12-31",
    splits: list[tuple[str, str, str]] = None,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    threshold_l1: float = 0.10,
) -> dict:
    """Returns a structured dict of metrics per slice per strategy."""
    assets = assets or ["btc", "eth", "sol"]
    splits = splits or [
        ("first_half", "2022-01-01", "2024-01-01"),
        ("second_half", "2024-01-01", "2026-01-01"),
        ("full", full_start, "2026-01-01"),  # sanity
    ]

    returns, macro = _load(assets, full_start, full_end)

    # Run REGIME_EW once over the full sample with locked hyperparameters
    universe = {a: 1.0 for a in assets if f"{a}_return" in returns.columns}
    regime_full = regime_gated_long_only(
        returns, macro, universe_weights=universe,
        hmm_window=504, hmm_refit_every=63, n_states=2, n_restarts=5,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        threshold_l1=threshold_l1,
    )
    bh_full = buy_and_hold(returns, asset="btc_return",
                           fee_bps=fee_bps, slippage_bps=slippage_bps)

    output: dict = {"splits": {}}
    for name, s, e in splits:
        s_ts = pd.Timestamp(s)
        e_ts = pd.Timestamp(e)
        output["splits"][name] = {
            "REGIME_EW": _slice_metrics(regime_full, returns.index, s_ts, e_ts),
            "BH_BTC": _slice_metrics(bh_full, returns.index, s_ts, e_ts),
        }
    return output


# ── reporting ──────────────────────────────────────────────────────


def _fmt_pct(x: float) -> str:
    return f"{x:+.1%}" if x is not None and not np.isnan(x) else "—"


def render_markdown(result: dict) -> str:
    lines: list[str] = []
    lines.append("### Run timestamp\n")
    lines.append(f"`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")
    lines.append("### Per-slice metrics\n")
    lines.append("Same strategy (same hyperparameters), evaluated on each slice.\n")
    lines.append("| Slice | Strategy | n_obs | Total | Sharpe | MaxDD | Lift vs BH (Total) | Lift vs BH (Sharpe) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for slice_name, strategies in result["splits"].items():
        bh = strategies["BH_BTC"]
        ew = strategies["REGIME_EW"]
        if ew.get("n_obs", 0) == 0:
            lines.append(f"| {slice_name} | (no data) | 0 | — | — | — | — | — |")
            continue
        total_lift = ew["total_return"] - bh["total_return"]
        sharpe_lift = ew["sharpe"] - bh["sharpe"]
        lines.append(
            f"| {slice_name} | BH_BTC | {bh['n_obs']} | "
            f"{_fmt_pct(bh['total_return'])} | {bh['sharpe']:.3f} | "
            f"{_fmt_pct(bh['max_dd'])} | — | — |"
        )
        lines.append(
            f"| {slice_name} | REGIME_EW | {ew['n_obs']} | "
            f"{_fmt_pct(ew['total_return'])} | {ew['sharpe']:.3f} | "
            f"{_fmt_pct(ew['max_dd'])} | {_fmt_pct(total_lift)} | {sharpe_lift:+.3f} |"
        )
    lines.append("")

    # Verdict
    first = result["splits"].get("first_half", {})
    second = result["splits"].get("second_half", {})
    lines.append("### Verdict\n")
    if first and second:
        f_lift = first["REGIME_EW"]["total_return"] - first["BH_BTC"]["total_return"]
        s_lift = second["REGIME_EW"]["total_return"] - second["BH_BTC"]["total_return"]
        if f_lift > 0 and s_lift > 0:
            lines.append("- **REGIME_EW beats BH_BTC in BOTH halves.** Strategy generalizes; not overfit to one period.")
        elif f_lift > 0 and s_lift <= 0:
            lines.append(f"- REGIME_EW beats BH_BTC ONLY in the first half "
                         f"(by {_fmt_pct(f_lift)}). The lift is concentrated "
                         f"in the 2022 selloff window. Strategy is real but "
                         f"its edge is 'avoid the next crypto winter,' not "
                         f"general alpha. Second half lift: {_fmt_pct(s_lift)}.")
        elif s_lift > 0 and f_lift <= 0:
            lines.append("- Unexpected: lift in second half only. Worth investigating.")
        else:
            lines.append("- REGIME_EW underperforms BH_BTC in both halves. Headline result was likely overfit.")
    return "\n".join(lines)


def append_to_doc(md: str, out_path: str) -> None:
    path = Path(out_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    sentinel = "### Results\n\n*(populated by `run_oos_validation.py`)*"
    if sentinel in existing:
        new = existing.replace(sentinel, "### Results\n\n" + md)
    else:
        # Append as a fresh OOS-results section
        new = existing.rstrip() + "\n\n### OOS validation results (appended)\n\n" + md
    path.write_text(new, encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="causal_portfolio/docs/regime_ew_production_readiness.md")
    args = p.parse_args()

    result = run_oos()
    md = render_markdown(result)

    # Console summary
    print("\n" + "=" * 70)
    print("OOS VALIDATION")
    print("=" * 70)
    for slice_name, strategies in result["splits"].items():
        bh = strategies["BH_BTC"]
        ew = strategies["REGIME_EW"]
        if ew.get("n_obs", 0) == 0:
            continue
        print(f"\n{slice_name} ({ew['n_obs']} obs):")
        print(f"  BH_BTC:    total {_fmt_pct(bh['total_return']):>8}  Sharpe {bh['sharpe']:.3f}  MaxDD {_fmt_pct(bh['max_dd']):>8}")
        print(f"  REGIME_EW: total {_fmt_pct(ew['total_return']):>8}  Sharpe {ew['sharpe']:.3f}  MaxDD {_fmt_pct(ew['max_dd']):>8}")
        lift = ew['total_return'] - bh['total_return']
        print(f"  LIFT:      total {_fmt_pct(lift):>8}  Sharpe {ew['sharpe']-bh['sharpe']:+.3f}")

    append_to_doc(md, args.out)
    print(f"\nResults appended to {args.out}")


if __name__ == "__main__":
    main()
