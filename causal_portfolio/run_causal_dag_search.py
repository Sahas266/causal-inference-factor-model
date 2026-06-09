"""Step 6 — re-run the DAG variants CAUSALLY.

The earlier DAG variant search (run_dag_search.py) scored each return-equation
structure with a purely correlational fit (Ridge/OLS via the V1 solver). This
re-runs the SAME variants but through the Step-5 causal machinery: for each
variant's factor pool + lag config, estimate the per-asset loadings two ways —
OLS vs gated 2SLS — and score BOTH arms out-of-sample (walk-forward folds)
against buy-and-hold BTC.

We do this regardless of the Step-5 finding that the declared instruments are
weak: the point is a complete, on-the-record causal sweep over DAG structures,
with the instrument-strength gate reported per variant so any "2SLS == OLS"
result is explained by the gate rather than left ambiguous.

Run:  python -m causal_portfolio.run_causal_dag_search
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.experiments.dag_variants import (
    FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS, VARIANTS,
)
from causal_portfolio.factors.builder import MACRO_FACTORS, build_all_factors
from causal_portfolio.factors.combo_selector import ComboDriverSelector
from causal_portfolio.factors.instruments import build_instruments
from causal_portfolio.experiments.ols_vs_2sls import run_ab
from causal_portfolio.validation.walk_forward import compare_variants

logger = logging.getLogger("cpcm.run_causal_dag_search")


def _prepare_variant_factors(variant, returns, factors):
    """Apply the variant's factor pool + per-type lags (+ optional Combo).

    Returns a factor DataFrame (lagged, subset) ready for run_ab, or None if
    the variant has no usable factors.
    """
    available = factors.dropna(axis=1, how="all")
    pool = (list(available.columns) if variant.factors is None
            else [f for f in variant.factors if f in available.columns])
    if not pool:
        return None

    lagged = pd.DataFrame(index=available.index)
    for f in pool:
        lag = variant.lag_macro if f in MACRO_FACTORS else variant.lag_global
        lagged[f] = available[f].shift(lag) if lag else available[f]

    if variant.combo_m is not None:
        common = lagged.dropna().index.intersection(returns.dropna().index)
        if len(common) < 50:
            return None
        m = min(variant.combo_m, lagged.shape[1])
        ranking = ComboDriverSelector().rank_all_subsets(
            returns.loc[common], lagged.loc[common], m=m)
        selected = list(ranking[0][0])
        lagged = lagged[selected]
    return lagged


def run(assets, start, end, *, train_window, rebalance_freq, f_threshold):
    from causal_portfolio.data import get_loader
    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    returns = loader.load_returns(assets, start, end)
    factors = build_all_factors(panel, macro)
    instruments, iv_map = build_instruments(panel)

    # Benchmark from the same simple-return basis run_ab uses.
    rows = []
    bh_ref = None
    for v in VARIANTS:
        vf = _prepare_variant_factors(v, returns, factors)
        if vf is None or vf.shape[1] == 0:
            rows.append((v, None, None, None))
            continue
        try:
            ab = run_ab(returns, vf, instruments, iv_map,
                        train_window=train_window, rebalance_freq=rebalance_freq,
                        f_threshold=f_threshold)
        except Exception as e:
            logger.warning("variant %s failed: %s", v.name, e)
            rows.append((v, None, None, None))
            continue
        bh_ref = ab.bh_btc
        cmp = compare_variants(
            {"OLS": ab.ols_returns, "2SLS": ab.tsls_returns}, ab.bh_btc,
            win_rate_bar=0.8)
        rows.append((v, ab, cmp, _gate_summary(ab.gate)))
        logger.info("[%s] OLS wr=%.0f%% 2SLS wr=%.0f%% gate=%s", v.name,
                    cmp["reports"]["OLS"].win_rate * 100,
                    cmp["reports"]["2SLS"].win_rate * 100, rows[-1][3])
    return rows, bh_ref


def _gate_summary(gate) -> str:
    if not gate.seen:
        return "no instrumented factors in pool"
    parts = []
    for nm in sorted(gate.seen):
        passed = gate.passed.get(nm, 0)
        parts.append(f"{nm} {passed}/{gate.seen[nm]}")
    return "; ".join(parts)


def render_markdown(rows, bh_ref, args) -> str:
    L = ["# Causal DAG Variant Search (OLS vs gated 2SLS)\n"]
    L.append("Each variant's return-equation structure (factor pool + lags + "
             "optional Combo selection) is run through the Step-5 causal A/B: "
             "per-asset loadings estimated by OLS and by gated 2SLS, scored "
             "out-of-sample (walk-forward folds) against buy-and-hold BTC. The "
             "instrument-strength gate (first-stage partial F ≥ "
             f"{args['f_threshold']}) is reported per variant.\n")
    L.append(f"- Assets: `{', '.join(args['assets'])}`")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | train "
             f"{args['train_window']}d, rebalance {args['rebalance_freq']}d")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    if bh_ref is not None and len(bh_ref):
        r = bh_ref.values
        pv = np.cumprod(1 + r)
        L.append(f"**Buy & Hold BTC (OOS):** total {pv[-1]/pv[0]-1:+.1%}, "
                 f"Sharpe {np.mean(r)/(np.std(r)+1e-12)*np.sqrt(365):.3f}\n")

    L.append("| Variant | OLS wr | 2SLS wr | OLS medSharpe | 2SLS medSharpe | "
             "Instrument gate (passed/seen) |")
    L.append("|---|---:|---:|---:|---:|---|")
    for v, ab, cmp, gate in rows:
        if cmp is None:
            L.append(f"| {v.name} | — | — | — | — | (skipped) |")
            continue
        o = cmp["reports"]["OLS"]; t = cmp["reports"]["2SLS"]
        L.append(f"| {v.name} | {o.win_rate:.0%} | {t.win_rate:.0%} | "
                 f"{o.median_sharpe:.3f} | {t.median_sharpe:.3f} | {gate} |")
    L.append("")

    # Instrument validity caveat (data-level exclusion check)
    L.append("## Instrument validity caveat\n")
    L.append("`stablecoin_mint` clears the strength gate only under the **lag1** "
             "configs — but that is an artifact, not a real instrument. Under "
             "lag1 the treatment `stable_flow` is `z_score(diff(supply))` lagged "
             "one day, while `stablecoin_mint` is `diff(supply)` lagged one day: "
             "the same underlying series up to scaling. So the first-stage F is "
             "near-infinite by construction (the instrument *is* the treatment), "
             "which violates the exclusion restriction. Tellingly, where this "
             "fake-strong instrument was used, 2SLS did **worse** OOS than OLS "
             "(e.g. `global_lag1`: 2SLS 25% vs OLS 50%) — instrumenting a factor "
             "with itself adds variance without identification. Every other "
             "instrument fails the strength gate outright. Net: there is no "
             "valid, strong instrument in this DAG on this data.\n")

    # Verdict
    winners = []
    for v, ab, cmp, gate in rows:
        if cmp is None:
            continue
        t = cmp["reports"]["2SLS"]
        if t.win_rate >= 0.8:   # robust bar: beats BH in >=80% of folds
            winners.append((v.name, t.win_rate))
    L.append("## Verdict\n")
    if winners:
        L.append("Variants whose **causal (2SLS)** arm beat BH BTC in ≥80% of "
                 "OOS folds:\n")
        for nm, wr in sorted(winners, key=lambda x: -x[1]):
            L.append(f"- **{nm}** — 2SLS fold win-rate {wr:.0%}")
        L.append("\n(Discount by the gate column + the multiple-testing across "
                 f"{len([r for r in rows if r[2]])} variants.)")
    else:
        L.append("**No variant's causal arm beats BH BTC in ≥80% of OOS folds.** "
                 "Where the gate column shows 0 passes, 2SLS reduced to OLS by "
                 "design (weak instruments), so the causal and correlational arms "
                 "coincide — and neither robustly beats simply holding BTC. This "
                 "is the same conclusion reached by every other experiment in the "
                 "project, now confirmed under explicit causal estimation across "
                 "all DAG structures.")
    return "\n".join(L)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--rebalance-freq", type=int, default=5)
    p.add_argument("--f-threshold", type=float, default=10.0)
    p.add_argument("--out", default="causal_portfolio/docs/causal_dag_variant_search.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    rows, bh_ref = run(assets, args.start, args.end,
                       train_window=args.train_window,
                       rebalance_freq=args.rebalance_freq,
                       f_threshold=args.f_threshold)

    print("\n" + "=" * 78)
    for v, ab, cmp, gate in rows:
        if cmp is None:
            print(f"  {v.name:<22} skipped")
            continue
        o = cmp["reports"]["OLS"]; t = cmp["reports"]["2SLS"]
        print(f"  {v.name:<22} OLS wr {o.win_rate:.0%}  2SLS wr {t.win_rate:.0%}  "
              f"| {gate}")

    md = render_markdown(rows, bh_ref, {
        "assets": assets, "start": args.start, "end": args.end,
        "train_window": args.train_window, "rebalance_freq": args.rebalance_freq,
        "f_threshold": args.f_threshold,
    })
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
