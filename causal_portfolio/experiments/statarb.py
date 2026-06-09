"""Market-neutral cointegration stat-arb — walk-forward, scored vs BH BTC.

At each refit, find cointegrated pairs on a trailing window (Engle-Granger,
ADF gate), then trade each pair's spread mean-reversion out-of-sample until the
next refit. Pair PnL is dollar-neutral (long 1 unit A, short beta units B,
gross exposure normalized); the book equal-weights active pairs. The combined
OOS return series is scored against buy-and-hold BTC with the walk-forward
harness.

This is the one strategy class in the project that is NOT a bet on BTC going
up, so it is the most plausible route to an honest OOS edge.

Run:  python -m causal_portfolio.experiments.statarb
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.analysis.cointegration import (
    find_cointegrated_pairs, spread_positions, spread_series,
)
from causal_portfolio.validation.walk_forward import metric_row

logger = logging.getLogger("cpcm.experiments.statarb")
ANNUALIZATION = 365


@dataclass
class StatArbResult:
    returns: pd.Series
    bh_btc: pd.Series
    n_pairs_per_refit: list[int] = field(default_factory=list)
    pair_usage: dict[str, int] = field(default_factory=dict)


def _load_prices(assets, start, end):
    from causal_portfolio.data import get_loader
    loader = get_loader()
    # Price levels (not returns): try PriceUSD then price.
    panel = loader.load_panel(assets, ["PriceUSD"], start, end)
    if panel.empty:
        panel = loader.load_panel(assets, ["price"], start, end)
    # columns like btc_PriceUSD -> btc
    panel.columns = [c.rsplit("_", 1)[0] for c in panel.columns]
    return panel.sort_index()


def run_statarb(
    prices: pd.DataFrame, *, train_window: int = 252, refit_every: int = 21,
    entry: float = 1.5, exit: float = 0.5, adf_pvalue: float = 0.05,
    max_pairs: int = 10,
) -> StatArbResult:
    prices = prices.dropna(how="all").ffill()
    log_p = np.log(prices)
    simple_ret = prices.pct_change()
    idx = prices.index
    T = len(idx)
    if T <= train_window + refit_every:
        raise ValueError(f"too few rows ({T})")

    daily = pd.Series(0.0, index=idx)
    n_pairs_hist: list[int] = []
    usage: dict[str, int] = {}

    t = train_window
    while t < T:
        train = log_p.iloc[t - train_window:t]
        pairs = find_cointegrated_pairs(train, adf_pvalue=adf_pvalue,
                                        min_obs=train_window // 2)[:max_pairs]
        n_pairs_hist.append(len(pairs))
        seg_end = min(t + refit_every, T)

        if pairs:
            # Precompute each pair's OOS z-score path + position, then PnL.
            seg_idx = idx[t:seg_end]
            pair_rets = []
            for p in pairs:
                key = f"{p.a}~{p.b}"
                usage[key] = usage.get(key, 0) + 1
                la = log_p[p.a].iloc[t - train_window:seg_end].values
                lb = log_p[p.b].iloc[t - train_window:seg_end].values
                spr = spread_series(la, lb, p.beta, p.const)
                z = (spr - p.mu) / p.sigma
                pos_full = spread_positions(z, entry=entry, exit=exit)
                # OOS slice (drop the training prefix); use yesterday's position
                pos_oos = pos_full[train_window:]
                ra = simple_ret[p.a].iloc[t:seg_end].values
                rb = simple_ret[p.b].iloc[t:seg_end].values
                gross = 1.0 + abs(p.beta)
                # long spread = +1 unit A, -beta units B
                spread_ret = (ra - p.beta * rb) / gross
                pnl = np.concatenate([[0.0], pos_oos[:-1]])[:len(spread_ret)] * spread_ret
                pair_rets.append(pd.Series(np.nan_to_num(pnl), index=seg_idx[:len(pnl)]))
            if pair_rets:
                book = pd.concat(pair_rets, axis=1).mean(axis=1)  # equal-weight pairs
                daily.loc[book.index] = book.values
        t = seg_end

    test_idx = idx[train_window:]
    ret = daily.loc[test_idx]
    btc = prices["btc"].pct_change().reindex(test_idx).fillna(0.0) if "btc" in prices else pd.Series(0.0, index=test_idx)
    return StatArbResult(returns=ret, bh_btc=btc,
                         n_pairs_per_refit=n_pairs_hist, pair_usage=usage)


def render_markdown(res: StatArbResult, cmp: dict, args: dict) -> str:
    L = ["# Cointegration Stat-Arb (market-neutral) vs BH BTC\n"]
    L.append("Walk-forward Engle-Granger pair selection (ADF-gated) + banded "
             "spread mean-reversion, dollar-neutral, equal-weighted across active "
             "pairs. Scored OOS against buy-and-hold BTC.\n")
    L.append(f"- Assets: `{', '.join(args['assets'])}`")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | train "
             f"{args['train_window']}d, refit {args['refit_every']}d | "
             f"entry z={args['entry']}, exit z={args['exit']}, ADF p<{args['adf_pvalue']}\n")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    avg_pairs = np.mean(res.n_pairs_per_refit) if res.n_pairs_per_refit else 0
    L.append(f"Cointegrated pairs found per refit: mean {avg_pairs:.1f}, "
             f"max {max(res.n_pairs_per_refit) if res.n_pairs_per_refit else 0}.\n")
    if res.pair_usage:
        top = sorted(res.pair_usage.items(), key=lambda kv: -kv[1])[:8]
        L.append("Most-selected pairs (refits active): " +
                 ", ".join(f"`{k}` ({v})" for k, v in top) + "\n")

    def _line(name, s):
        return metric_row(name, s, with_dd=True)

    L.append("## Full OOS window\n")
    L.append("| Strategy | Total | Sharpe | MaxDD |")
    L.append("|---|---:|---:|---:|")
    L.append(_line("Stat-Arb", res.returns))
    L.append(_line("BH BTC", res.bh_btc))
    L.append("")

    L.append("## Walk-forward folds vs BH BTC\n")
    L.append(cmp["multiple_testing_note"] + "\n")
    L.append("| Strategy | Win-rate vs BH | Median Sharpe | Folds |")
    L.append("|---|---:|---:|---:|")
    for name, wr, ms in cmp["ranking"]:
        rep = cmp["reports"][name]
        L.append(f"| {name} | {wr:.0%} | {ms:.3f} | {rep.n_folds} |")
    L.append("")

    sa = cmp["reports"]["StatArb"]
    r = res.returns.values
    full_sharpe = float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(ANNUALIZATION))
    corr_btc = float(np.corrcoef(res.returns.values, res.bh_btc.values)[0, 1])
    L.append("## Verdict\n")
    L.append(f"*A market-neutral book should NOT be judged on beating BH BTC's "
             f"total return — that is a directional benchmark and not its job. "
             f"The honest questions are: is its standalone risk-adjusted return "
             f"positive and stable, and does it diversify BTC?*\n")
    L.append(f"- BTC correlation of daily returns: **{corr_btc:+.2f}** "
             f"(near zero confirms market-neutral by construction).")
    L.append(f"- Full-window Sharpe (the honest aggregate, no costs): "
             f"**{full_sharpe:.2f}**.")
    L.append(f"- Per-fold median Sharpe {sa.median_sharpe:.2f} vs full-window "
             f"{full_sharpe:.2f}: a large gap means good folds are offset by bad "
             f"stretches / autocorrelated drawdowns — inconsistent, not a stable edge.\n")
    if full_sharpe >= 1.0:
        L.append(f"Standalone Sharpe ≥ 1 gross and ~zero BTC correlation: a "
                 f"**promising market-neutral sleeve** worth costed validation "
                 f"(borrow, slippage, capacity) and blending with a BTC core.")
    elif full_sharpe >= 0.4:
        L.append(f"Standalone gross Sharpe {full_sharpe:.2f} is **mediocre** and "
                 f"this charges **no transaction costs** — pairwise spread trading "
                 f"churns, so net Sharpe would be materially lower, likely ~0. The "
                 f"near-zero BTC correlation is the only real positive; not "
                 f"tradeable as-is, but the diversification angle is worth refining "
                 f"(cost model, tighter pair stability filter, vol-targeting).")
    else:
        L.append(f"Standalone gross Sharpe {full_sharpe:.2f} — **no edge.** "
                 f"Cointegration relationships in this universe were too unstable "
                 f"OOS to trade profitably at these thresholds, before any costs. "
                 f"Consistent with the rest of the project.")
    return "\n".join(L)


def run(assets, start, end, **kw):
    from causal_portfolio.validation.walk_forward import compare_variants
    prices = _load_prices(assets, start, end)
    res = run_statarb(prices, **kw)
    cmp = compare_variants({"StatArb": res.returns}, res.bh_btc, win_rate_bar=0.8)
    return res, cmp


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge,xrp,crv,ltc")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--refit-every", type=int, default=21)
    p.add_argument("--entry", type=float, default=1.5)
    p.add_argument("--exit", type=float, default=0.5)
    p.add_argument("--adf-pvalue", type=float, default=0.05)
    p.add_argument("--max-pairs", type=int, default=10)
    p.add_argument("--out", default="causal_portfolio/docs/cointegration_statarb.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    res, cmp = run(assets, args.start, args.end,
                   train_window=args.train_window, refit_every=args.refit_every,
                   entry=args.entry, exit=args.exit, adf_pvalue=args.adf_pvalue,
                   max_pairs=args.max_pairs)

    sa = cmp["reports"]["StatArb"]
    print("\n" + "=" * 70)
    print(f"Stat-Arb: win-rate vs BH {sa.win_rate:.0%} | median Sharpe "
          f"{sa.median_sharpe:.3f} | folds {sa.n_folds}")
    print(f"avg pairs/refit: {np.mean(res.n_pairs_per_refit):.1f}")

    md = render_markdown(res, cmp, {
        "assets": assets, "start": args.start, "end": args.end,
        "train_window": args.train_window, "refit_every": args.refit_every,
        "entry": args.entry, "exit": args.exit, "adf_pvalue": args.adf_pvalue,
    })
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
