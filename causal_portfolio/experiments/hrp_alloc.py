"""Hierarchical Risk Parity allocation vs equal-weight vs BH BTC.

HRP builds long-only weights from the return correlation structure alone (no
return forecast, no covariance inversion). Walk-forward: re-estimate HRP weights
on a trailing window each rebalance, hold OOS. Compared against an equal-weight
book and buy-and-hold BTC.

This isolates the *portfolio-construction* lever (correlation-aware
diversification) from any alpha signal — if HRP can't beat BH BTC, it confirms
that smarter weighting of the same long crypto universe doesn't escape BTC's
dominance.

Run:  python -m causal_portfolio.experiments.hrp_alloc
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.analysis.correlation import hrp_weights
from causal_portfolio.validation.walk_forward import metric_row

logger = logging.getLogger("cpcm.experiments.hrp")


@dataclass
class HRPResult:
    hrp_returns: pd.Series
    ew_returns: pd.Series
    bh_btc: pd.Series


def run_alloc(returns_simple: pd.DataFrame, *, train_window: int = 252,
              rebalance_freq: int = 21) -> HRPResult:
    R = returns_simple.dropna(how="all").fillna(0.0)
    idx = R.index
    cols = list(R.columns)
    T = len(idx)
    n = len(cols)
    if T <= train_window + rebalance_freq:
        raise ValueError(f"too few rows ({T})")

    hrp = np.zeros(T - train_window)
    ew = np.zeros(T - train_window)
    w_hrp = pd.Series(np.ones(n) / n, index=cols)
    w_ew = pd.Series(np.ones(n) / n, index=cols)

    for t in range(train_window, T):
        i = t - train_window
        if i % rebalance_freq == 0:
            train = R.iloc[t - train_window:t]
            try:
                w_hrp = hrp_weights(train, long_only=True).reindex(cols).fillna(0.0)
            except Exception as e:
                logger.warning("HRP failed t=%d: %s", t, e)
        day = R.iloc[t]
        hrp[i] = float((w_hrp * day).sum())
        ew[i] = float((w_ew * day).sum())

    test_idx = idx[train_window:]
    btc = R["btc"].reindex(test_idx).fillna(0.0) if "btc" in R else pd.Series(0.0, index=test_idx)
    return HRPResult(pd.Series(hrp, index=test_idx),
                     pd.Series(ew, index=test_idx), btc)


def render_markdown(res: HRPResult, cmp: dict, args: dict) -> str:
    L = ["# Hierarchical Risk Parity (HRP) vs Equal-Weight vs BH BTC\n"]
    L.append("Correlation-distance hierarchical clustering -> quasi-diagonal "
             "-> recursive bisection (López de Prado HRP), long-only, refit on a "
             "trailing window. Pure correlation-based diversification, no alpha "
             "signal. Scored OOS vs an equal-weight book and BH BTC.\n")
    L.append(f"- Assets: `{', '.join(args['assets'])}`")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | train "
             f"{args['train_window']}d, rebalance {args['rebalance_freq']}d")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    def _line(name, s):
        return metric_row(name, s, with_dd=True)

    L.append("## Full OOS window\n| Strategy | Total | Sharpe | MaxDD |\n|---|---:|---:|---:|")
    L.append(_line("HRP", res.hrp_returns))
    L.append(_line("Equal-Weight", res.ew_returns))
    L.append(_line("BH BTC", res.bh_btc))
    L.append("")

    L.append("## Walk-forward folds vs BH BTC\n")
    L.append(cmp["multiple_testing_note"] + "\n")
    L.append("| Strategy | Win-rate vs BH | Median Sharpe | Folds |\n|---|---:|---:|---:|")
    for name, wr, ms in cmp["ranking"]:
        rep = cmp["reports"][name]
        L.append(f"| {name} | {wr:.0%} | {ms:.3f} | {rep.n_folds} |")
    L.append("")

    hrp_wr = cmp["reports"]["HRP"].win_rate
    ew_wr = cmp["reports"]["EqualWeight"].win_rate
    L.append("## Verdict\n")
    if hrp_wr >= 0.8:
        L.append(f"HRP beat BH BTC in {hrp_wr:.0%} of folds — correlation-aware "
                 f"weighting added real value. Validate vs costs/turnover.")
    else:
        L.append(f"HRP did not beat BH BTC ({hrp_wr:.0%} of folds; equal-weight "
                 f"{ew_wr:.0%}). Smarter correlation-based weighting of the same "
                 f"long crypto universe still doesn't escape BTC's dominance — the "
                 f"problem isn't weighting, it's that the universe is a "
                 f"high-correlation, BTC-led basket. Diversification reduces "
                 f"drawdown but caps upside below BH BTC.")
    return "\n".join(L)


def run(assets, start, end, **kw):
    from causal_portfolio.data import get_loader
    from causal_portfolio.validation.walk_forward import compare_variants
    rets = get_loader().load_prices(assets, start, end).sort_index().pct_change()
    res = run_alloc(rets, **kw)
    cmp = compare_variants({"HRP": res.hrp_returns, "EqualWeight": res.ew_returns},
                           res.bh_btc, win_rate_bar=0.8)
    return res, cmp


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge,xrp,crv,ltc")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--rebalance-freq", type=int, default=21)
    p.add_argument("--out", default="causal_portfolio/docs/hrp_allocation.md")
    args = p.parse_args()
    assets = args.assets.split(",")
    res, cmp = run(assets, args.start, args.end, train_window=args.train_window,
                   rebalance_freq=args.rebalance_freq)
    for name in ("HRP", "EqualWeight"):
        rep = cmp["reports"][name]
        print(f"{name:>12}: win-rate vs BH {rep.win_rate:.0%} | median Sharpe "
              f"{rep.median_sharpe:.3f}")
    md = render_markdown(res, cmp, {
        "assets": assets, "start": args.start, "end": args.end,
        "train_window": args.train_window, "rebalance_freq": args.rebalance_freq,
    })
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
