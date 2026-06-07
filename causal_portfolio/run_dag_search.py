"""Run all DAG variants through the full backtest and compare.

Evaluates each return-equation structure on a full window and a held-out OOS
window, against buy-and-hold BTC. Writes a results table + honest verdict to
causal_portfolio/docs/dag_variant_search.md.

Usage:
    python -m causal_portfolio.run_dag_search
    python -m causal_portfolio.run_dag_search --assets btc,eth,sol,bnb,avax
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from causal_portfolio.experiments.dag_variants import (
    VARIANTS, buy_and_hold_btc, load_inputs, run_variant,
)

logger = logging.getLogger("cpcm.run_dag_search")


def _slice(returns, factors, start, end):
    idx = returns.index
    mask = (idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end))
    return returns.loc[mask], factors.loc[factors.index.isin(returns.index[mask])]


def run(assets, full_start, full_end, oos_start):
    returns, factors = load_inputs(assets, full_start, full_end)

    windows = {
        "full": (full_start, full_end),
        "oos": (oos_start, full_end),
    }
    results: dict[str, list] = {}
    bh: dict[str, dict] = {}
    for wlabel, (s, e) in windows.items():
        R, F = _slice(returns, factors, s, e)
        bh[wlabel] = buy_and_hold_btc(R)
        rows = []
        for v in VARIANTS:
            logger.info("[%s] running variant %s", wlabel, v.name)
            rows.append(run_variant(v, R, F, wlabel))
        results[wlabel] = rows
    return results, bh


def _fmt(x, pct=True):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:+.1%}" if pct else f"{x:.3f}"


def render_markdown(results, bh, assets, run_args) -> str:
    L = []
    L.append("# DAG Variant Search\n")
    L.append("Each variant is a different return-equation structure (which factors "
             "are direct causes of returns, at what lag, with/without Combo "
             "selection), run through the identical downstream pipeline (V1 solver "
             "+ EKF + manifold optimizer + walk-forward backtest). Differences are "
             "attributable to DAG structure alone.\n")
    L.append("**Lag grid:** every factor-pool config is run both contemporaneous "
             "(`_lag0`, edges use same-day factor values) and fully lagged "
             "(`_lag1`, edges use prior-day values everywhere — predictive, no "
             "contemporaneous look-ahead). This covers lagging across all configs.\n")
    L.append(f"- Assets: `{', '.join(assets)}`")
    L.append(f"- Full window: `{run_args['full_start']}` → `{run_args['full_end']}`")
    L.append(f"- OOS window: `{run_args['oos_start']}` → `{run_args['full_end']}`")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    # ── Config descriptions ──
    from causal_portfolio.experiments.dag_variants import VARIANTS
    L.append("## Configurations\n")
    L.append("| Config | Factor pool | Selection | Lag | What it tests |")
    L.append("|---|---|---|---|---|")
    for v in VARIANTS:
        if v.factors is None:
            pool = "all available factors"
        elif len(v.factors) > 4:
            pool = f"{len(v.factors)} factors ({', '.join(v.factors[:3])}, etc.)"
        else:
            pool = ", ".join(v.factors)
        sel = f"Combo m={v.combo_m}" if v.combo_m else "use whole pool"
        lag = "contemporaneous" if v.lag_global == 0 and v.lag_macro == 0 else "lagged 1 day (everywhere)"
        L.append(f"| `{v.name}` | {pool} | {sel} | {lag} | {v.description} |")
    L.append("")

    for wlabel in ("full", "oos"):
        b = bh.get(wlabel, {})
        L.append(f"## {wlabel.upper()} window\n")
        L.append(f"**Buy & Hold BTC:** total {_fmt(b.get('total_return'))}, "
                 f"Sharpe {_fmt(b.get('sharpe'), pct=False)}, "
                 f"MaxDD {_fmt(b.get('max_dd'))}\n")
        L.append("| Variant | Drivers | Total | Sharpe | Sortino | MaxDD | Beats BH? |")
        L.append("|---|---|---:|---:|---:|---:|:--:|")
        for r in sorted(results[wlabel], key=lambda x: (x.sharpe if x.sharpe == x.sharpe else -9)):
            beats = ""
            if r.sharpe == r.sharpe and b.get("total_return") is not None:
                beats = "✓" if r.total_return > b["total_return"] else "—"
            drv = ", ".join(r.drivers_used) if r.drivers_used else (r.error or "—")
            L.append(f"| {r.name} | {drv} | {_fmt(r.total_return)} | "
                     f"{_fmt(r.sharpe, pct=False)} | {_fmt(r.sortino, pct=False)} | "
                     f"{_fmt(r.max_dd)} | {beats} |")
        L.append("")

    # Verdict: a variant wins only if it beats BH BTC in BOTH windows
    full_by = {r.name: r for r in results["full"]}
    oos_by = {r.name: r for r in results["oos"]}
    bh_full = bh["full"].get("total_return", 0)
    bh_oos = bh["oos"].get("total_return", 0)
    winners = []
    for name in full_by:
        rf, ro = full_by[name], oos_by.get(name)
        if rf.sharpe == rf.sharpe and ro and ro.sharpe == ro.sharpe:
            if rf.total_return > bh_full and ro.total_return > bh_oos:
                winners.append((name, rf, ro))

    L.append("## Verdict\n")
    if winners:
        L.append("Variants that beat Buy & Hold BTC in **both** the full and OOS "
                 "windows (the only honest bar):\n")
        for name, rf, ro in sorted(winners, key=lambda t: -t[2].sharpe):
            L.append(f"- **{name}** — full {_fmt(rf.total_return)} / OOS "
                     f"{_fmt(ro.total_return)} (OOS Sharpe {ro.sharpe:.3f})")
        L.append("")
    else:
        L.append("**No variant beats Buy & Hold BTC in both windows.** "
                 "Restructuring the DAG's return equation does not, on this data, "
                 "produce a configuration that robustly beats simply holding BTC. "
                 "Consistent with every other strategy result in this project.\n")
    return "\n".join(L)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--full-start", default="2022-01-01")
    p.add_argument("--full-end", default="2025-12-31")
    p.add_argument("--oos-start", default="2024-01-01")
    p.add_argument("--out", default="causal_portfolio/docs/dag_variant_search.md")
    args = p.parse_args()

    assets = args.assets.split(",")
    results, bh = run(assets, args.full_start, args.full_end, args.oos_start)

    # Console summary
    print("\n" + "=" * 78)
    for wlabel in ("full", "oos"):
        b = bh[wlabel]
        print(f"\n{wlabel.upper()} — BH BTC: total {b.get('total_return',0):+.1%} "
              f"Sharpe {b.get('sharpe',0):.3f}")
        for r in sorted(results[wlabel], key=lambda x: -(x.sharpe if x.sharpe==x.sharpe else -9)):
            print(f"  {r.name:<22} total {r.total_return:+8.1%}  "
                  f"Sharpe {r.sharpe:6.3f}  MaxDD {r.max_dd:+7.1%}")

    md = render_markdown(results, bh, assets, {
        "full_start": args.full_start, "full_end": args.full_end,
        "oos_start": args.oos_start,
    })
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
