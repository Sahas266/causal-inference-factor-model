"""Cross-sectional momentum/reversal — the first per-asset-varying drivers.

Every CPCM factor is GLOBAL (one value per day, shared by all assets), so the
cross-section was only ever differentiated through estimated loadings. Lagged
per-asset returns are the missing driver class: they vary in the cross
section by construction. This tests the classic signals in market-neutral
(beta-hedged residual) space:

  rev1   — yesterday's residual return (short-term reversal: long losers)
  mom7   — trailing 7d residual return
  mom30  — trailing 30d residual return
  mom30_7 — trailing 30d skipping the most recent 7d (standard momentum
            construction, avoids reversal contamination)
  combo  — mean z of (-rev1, +mom30_7): pre-registered sign convention from
           the equity literature (reversal short-term, momentum medium-term),
           NOT fitted to this sample.

Portfolios: daily-rebalanced rank weights (demeaned, gross 1) over whichever
assets have data that day (NaN-tolerant — late listings enter when ready).
No costs: a daily-churn rank L/S is cost-heavy, so any marginal Sharpe here
is an UPPER bound; the verdict discounts accordingly.

Honesty: Newey-West t on the mean, fold Sharpes, and a circular time-shift
placebo of the signal matrix (structure-preserving null), as in
cross_sectional.py.

Run:  python -m causal_portfolio.experiments.momentum_xs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import ANNUALIZATION
from causal_portfolio.experiments.cross_sectional import (
    _newey_west_t, beta_hedged_residuals,
)

logger = logging.getLogger("cpcm.experiments.momentum_xs")

WIDE_UNIVERSE = ("btc,eth,sol,bnb,avax,uni,aave,link,doge,crv,pendle,xrp,"
                 "shib,zec,tao,pepe,aero,jup,ena,hype")


# ── signals ─────────────────────────────────────────────────────────

def build_signals(residuals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-asset signal frames, all known at day t (no look-ahead)."""
    r = residuals
    return {
        "rev1": r,                                   # yesterday's return
        "mom7": r.rolling(7, min_periods=5).sum(),
        "mom30": r.rolling(30, min_periods=20).sum(),
        "mom30_7": (r.rolling(30, min_periods=20).sum()
                    - r.rolling(7, min_periods=5).sum()),
    }


def combo_signal(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pre-registered: short-term reversal + medium momentum (z-averaged)."""
    def _xz(df):  # cross-sectional z per day
        mu = df.mean(axis=1)
        sd = df.std(axis=1)
        return df.sub(mu, axis=0).div(sd + 1e-15, axis=0)
    return (-_xz(signals["rev1"]) + _xz(signals["mom30_7"])) / 2.0


# ── rank L/S backtest ───────────────────────────────────────────────

@dataclass
class MomResult:
    name: str
    sharpe: float
    nw_tstat: float
    total: float
    fold_sharpes: list[float]
    avg_breadth: float


def rank_ls(signal: pd.DataFrame, residuals: pd.DataFrame,
            n_folds: int = 4) -> MomResult | None:
    """Daily L/S: weights from the signal at close of day t-1, applied to
    residual returns over day t (the loop below indexes S at i-1, R at i)."""
    common = signal.index.intersection(residuals.index)
    S = signal.loc[common]
    R = residuals.loc[common]

    pnl = []
    breadth = []
    dates = []
    for i in range(1, len(common)):
        s_row = S.iloc[i - 1]         # signal at t-1 close
        r_row = R.iloc[i]             # residual return at t
        ok = s_row.notna() & r_row.notna()
        if ok.sum() < 5:
            continue
        ranks = s_row[ok].rank()
        w = ranks - ranks.mean()
        gross = w.abs().sum()
        if gross < 1e-12:
            continue
        w = w / gross
        pnl.append(float((w * r_row[ok]).sum()))
        breadth.append(int(ok.sum()))
        dates.append(common[i])
    if len(pnl) < 200:
        return None
    p = pd.Series(pnl, index=dates)
    sharpe = float(p.mean() / (p.std() + 1e-15) * np.sqrt(ANNUALIZATION))
    folds = np.array_split(p.values, n_folds)
    fold_sh = [float(np.mean(f) / (np.std(f) + 1e-15) * np.sqrt(ANNUALIZATION))
               for f in folds if len(f) > 20]
    pv = np.cumprod(1 + p.values)
    return MomResult("", sharpe, _newey_west_t(p.values),
                     float(pv[-1] - 1.0), fold_sh, float(np.mean(breadth)))


def placebo_p(signal: pd.DataFrame, residuals: pd.DataFrame,
              real_sharpe: float, n: int = 100, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    runs = 0
    for _ in range(n):
        k = int(rng.integers(60, len(signal) - 60))
        shifted = pd.DataFrame(np.roll(signal.values, k, axis=0),
                               index=signal.index, columns=signal.columns)
        res = rank_ls(shifted, residuals)
        if res is None:
            continue
        runs += 1
        hits += abs(res.sharpe) >= abs(real_sharpe)
    return hits / max(runs, 1)


# ── report ──────────────────────────────────────────────────────────

def render_markdown(rows: list[tuple[MomResult, float]], args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# Cross-sectional momentum / reversal (per-asset drivers)\n"]
    L.append("Daily rank L/S on beta-hedged residual returns over a "
             f"{args['n_assets']}-asset universe (NaN-tolerant; late "
             "listings enter when listed). No transaction costs — Sharpes "
             "are upper bounds for a daily-churn strategy.\n")
    L.append(f"- Universe: `{args['assets']}`")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | β window "
             f"{args['beta_window']}d")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("| Signal | Sharpe | NW t | total | placebo p | avg breadth | "
             "fold Sharpes |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for r, p in rows:
        L.append(f"| {r.name} | {r.sharpe:.3f} | {r.nw_tstat:.2f} | "
                 f"{r.total:+.1%} | {p:.2f} | {r.avg_breadth:.1f} | "
                 f"{', '.join(f'{s:.2f}' for s in r.fold_sharpes)} |")
    L.append("")

    sig = [(r, p) for r, p in rows if abs(r.nw_tstat) > 2 and p < 0.05]
    L.append("## Verdict\n")
    if sig:
        L.append("Signals clearing both NW |t|>2 and placebo p<0.05: "
                 + ", ".join(f"**{r.name}** (Sharpe {r.sharpe:.2f}, "
                             f"p={p:.2f})" for r, p in sig)
                 + ". Before believing: subtract realistic costs (daily "
                 "rank churn ≈ 50–150% turnover/day) and re-test net.")
    else:
        L.append("**No momentum/reversal signal survives both the "
                 "Newey-West and placebo bars** in beta-hedged residual "
                 "space on this universe/window. The per-asset driver class "
                 "does not rescue cross-sectional predictability either.")
    return "\n".join(L)


def main() -> None:
    import argparse
    from pathlib import Path
    from causal_portfolio.data import get_loader

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default=WIDE_UNIVERSE)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--beta-window", type=int, default=90)
    p.add_argument("--n-placebo", type=int, default=100)
    p.add_argument("--out", default="causal_portfolio/docs/momentum_xs.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    loader = get_loader()
    returns = loader.load_returns(assets, args.start, args.end)
    returns.index = pd.to_datetime(returns.index)
    residuals = beta_hedged_residuals(returns, beta_window=args.beta_window)

    signals = build_signals(residuals)
    signals["combo"] = combo_signal(signals)

    rows = []
    for name, sig in signals.items():
        res = rank_ls(sig, residuals)
        if res is None:
            logger.info("%s: too few rows", name)
            continue
        res.name = name
        pp = placebo_p(sig, residuals, res.sharpe, n=args.n_placebo)
        logger.info("%s: Sharpe=%.3f NWt=%.2f placebo_p=%.2f",
                    name, res.sharpe, res.nw_tstat, pp)
        rows.append((res, pp))

    md = render_markdown(rows, {**vars(args), "n_assets": len(assets)})
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
