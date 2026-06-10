"""Direction 2 — event studies around discrete on-chain shocks.

Continuous-treatment 2SLS wasted the genuinely exogenous-ish variation the
warehouse contains. This treats the shocks as natural experiments: for each
event type, compare forward returns and forward realized volatility after
event days against a block-bootstrap null of random non-event days.

Event types (built from the warehouse mirror, all defined at day t from data
available at day t):
  - blob_spike          |z(Δ eth blob usage)| > 2          (post-Dencun)
  - queue_jump          z(Δ validator entry queue) > 2
  - stable_mint_shock   z(Δ total stablecoin supply) > 2
  - stable_burn_shock   z(Δ total stablecoin supply) < -2
  - liq_spike           |z(Δ liquidation volume)| > 2
  - etf_inflow/outflow  z(Δ ETH ETF AUM in ETH units) ≷ ±2 (native units —
                        flow shocks, not price marks; 2024+)
  - funding_extreme     cross-asset mean 8h funding in top/bottom 5%
  - gas_spike           |z(Δ gas utilization)| > 2 (the original instrument)

Outcomes: cumulative BTC and equal-weight basket return over the next 1/3/5/10
days; realized vol over the next 5 days vs unconditional. Null: moving-block
bootstrap (block=10) of pseudo-event sets of the same size drawn from the
event-eligible sample, 1000 reps -> two-sided empirical p per horizon.

Caveat documented per event: exogeneity is NOT equal across events. Liquidation
spikes and funding extremes are reactions to price moves (conditioning, not
causation); blob/queue/mint shocks have better claims to exogenous timing.

Run:  python -m causal_portfolio.experiments.event_studies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("cpcm.experiments.event_studies")

HORIZONS = (1, 3, 5, 10)
VOL_H = 5


# ── event construction ──────────────────────────────────────────────

def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-15)


def _spike(s: pd.Series, thresh: float = 2.0, side: str = "abs") -> pd.Series:
    z = _z(s.diff())
    if side == "abs":
        ind = z.abs() > thresh
    elif side == "pos":
        ind = z > thresh
    else:
        ind = z < -thresh
    return ind.where(z.notna())


def build_events(wide: pd.DataFrame) -> dict[str, tuple[pd.Series, str]]:
    """event_name -> (bool series, exogeneity note)."""
    ev: dict[str, tuple[pd.Series, str]] = {}

    def col(name):
        return wide[name].astype(float) if name in wide.columns else None

    blob = col("eth_blob_size_mib")
    if blob is not None:
        ev["blob_spike"] = (_spike(blob), "good: blob demand is L2-driven")
    queue = col("eth_queue_active_amount")
    if queue is not None:
        ev["queue_jump"] = (_spike(queue, side="pos"),
                            "good: staking entry decisions, slow-moving")
    supply_cols = [c for c in wide.columns
                   if c.endswith("_SplyCur")
                   or c.endswith("_stablecoin_circulating_usd")]
    if supply_cols:
        total = wide[supply_cols].sum(axis=1)
        ev["stable_mint_shock"] = (_spike(total, side="pos"),
                                   "moderate: issuance responds to demand")
        ev["stable_burn_shock"] = (_spike(total, side="neg"),
                                   "moderate: redemptions respond to stress")
    liq_cols = [c for c in wide.columns if c.endswith("_liquidation_volume_usd")]
    if liq_cols:
        ev["liq_spike"] = (_spike(wide[liq_cols].sum(axis=1)),
                           "poor: liquidations REACT to price moves")
    etf = col("eth_etf_aum_native")
    if etf is not None:
        ev["etf_inflow"] = (_spike(etf, side="pos"),
                            "moderate: TradFi flow shocks (2024+, short)")
        ev["etf_outflow"] = (_spike(etf, side="neg"),
                             "moderate: TradFi flow shocks (2024+, short)")
    fund_cols = [c for c in wide.columns if c.endswith("_funding_rate_8h")]
    if fund_cols:
        f = wide[fund_cols].mean(axis=1)
        ev["funding_extreme_pos"] = ((f > f.quantile(0.95)).where(f.notna()),
                                     "poor: funding reacts to price/positioning")
        ev["funding_extreme_neg"] = ((f < f.quantile(0.05)).where(f.notna()),
                                     "poor: funding reacts to price/positioning")
    gas = col("eth_avg_gas_utilization")
    if gas is not None:
        ev["gas_spike"] = (_spike(gas), "moderate: usage bursts")
    return ev


# ── event-study statistics ──────────────────────────────────────────

@dataclass
class EventStat:
    name: str
    note: str
    n_events: int
    rows: list[tuple]      # (outcome, horizon, event_mean, null_mean, p)
    vol_ratio: float
    vol_p: float


def _block_null(values: np.ndarray, eligible: np.ndarray, n_ev: int,
                n_boot: int, rng, block: int = 10) -> np.ndarray:
    """Bootstrap distribution of mean(outcome) over pseudo-event sets."""
    idx_pool = np.flatnonzero(eligible)
    out = np.empty(n_boot)
    n_blocks = max(1, int(np.ceil(n_ev / block)))
    for b in range(n_boot):
        starts = rng.choice(idx_pool, size=n_blocks, replace=True)
        take = np.concatenate([np.arange(s, s + block) for s in starts])
        take = take[take < len(values)][:n_ev]
        out[b] = np.nanmean(values[take])
    return out


def study_event(
    name: str, ind: pd.Series, note: str,
    btc: pd.Series, ew: pd.Series, *,
    n_boot: int = 1000, seed: int = 0,
) -> EventStat | None:
    rng = np.random.default_rng(seed)
    aligned = pd.concat([ind.rename("ev"), btc.rename("btc"),
                         ew.rename("ew")], axis=1)
    aligned = aligned[aligned["ev"].notna()]
    if len(aligned) < 200:
        return None
    ev_mask = aligned["ev"].astype(bool).values
    n_ev = int(ev_mask.sum())
    if n_ev < 8:
        return None

    rows = []
    for outcome, series in (("btc", aligned["btc"]), ("ew", aligned["ew"])):
        for h in HORIZONS:
            fwd = (series.shift(-1).rolling(h).sum().shift(-(h - 1))).values
            valid = ~np.isnan(fwd)
            ev_mean = float(np.nanmean(fwd[ev_mask & valid]))
            null = _block_null(fwd, valid, n_ev, n_boot, rng)
            p = float(np.mean(np.abs(null - np.nanmean(null))
                              >= abs(ev_mean - np.nanmean(null))))
            rows.append((outcome, h, ev_mean, float(np.nanmean(null)), p))

    # Forward realized vol (next VOL_H days) on the EW basket
    fwd_vol = (aligned["ew"].shift(-1).rolling(VOL_H).std()
               .shift(-(VOL_H - 1))).values
    valid = ~np.isnan(fwd_vol)
    ev_vol = float(np.nanmean(fwd_vol[ev_mask & valid]))
    null_v = _block_null(fwd_vol, valid, n_ev, n_boot, rng)
    base = float(np.nanmean(null_v))
    vol_ratio = ev_vol / (base + 1e-15)
    vol_p = float(np.mean(np.abs(null_v - base) >= abs(ev_vol - base)))

    return EventStat(name, note, n_ev, rows, vol_ratio, vol_p)


# ── report ──────────────────────────────────────────────────────────

def render_markdown(stats: list[EventStat], args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# Event studies on on-chain shocks (Direction 2)\n"]
    L.append("Forward returns and forward 5d realized vol after event days vs "
             "a moving-block bootstrap null (block=10, 1000 reps, two-sided "
             "empirical p). Events defined at day t from data available at "
             "day t; outcomes start at t+1 — no overlap.\n")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | basket: "
             f"`{', '.join(args['assets'])}`")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("| Event | n | Exogeneity | BTC +5d (p) | EW +5d (p) | "
             "fwd vol ratio (p) |")
    L.append("|---|---:|---|---:|---:|---:|")
    for s in stats:
        btc5 = next(r for r in s.rows if r[0] == "btc" and r[1] == 5)
        ew5 = next(r for r in s.rows if r[0] == "ew" and r[1] == 5)
        L.append(f"| {s.name} | {s.n_events} | {s.note.split(':')[0]} | "
                 f"{btc5[2]:+.2%} ({btc5[4]:.2f}) | "
                 f"{ew5[2]:+.2%} ({ew5[4]:.2f}) | "
                 f"{s.vol_ratio:.2f} ({s.vol_p:.3f}) |")
    L.append("")

    L.append("## Full horizon detail\n")
    for s in stats:
        L.append(f"### {s.name} — n={s.n_events} ({s.note})\n")
        L.append("| Outcome | h | event mean | null mean | p |")
        L.append("|---|---:|---:|---:|---:|")
        for outcome, h, em, nm, p in s.rows:
            mark = " **" if p < 0.05 else ""
            L.append(f"| {outcome} | {h}d | {em:+.2%}{mark} | {nm:+.2%} | {p:.3f} |")
        L.append(f"\nForward 5d vol ratio (event/uncond): "
                 f"**{s.vol_ratio:.2f}** (p={s.vol_p:.3f})\n")

    sig_ret = [(s.name, r) for s in stats for r in s.rows if r[4] < 0.05]
    sig_vol = [s for s in stats if s.vol_p < 0.05]
    L.append("## Verdict\n")
    L.append(f"- Return effects with p<0.05: {len(sig_ret)} of "
             f"{sum(len(s.rows) for s in stats)} (outcome,horizon) cells "
             f"across {len(stats)} events — at α=0.05, "
             f"~{0.05 * sum(len(s.rows) for s in stats):.0f} expected by "
             f"chance. Treat isolated hits accordingly.")
    if sig_vol:
        L.append(f"- **Volatility effects are the real story:** "
                 + "; ".join(f"`{s.name}` vol ratio {s.vol_ratio:.2f} "
                             f"(p={s.vol_p:.3f})" for s in sig_vol)
                 + ". Events that move vol but not mean returns are usable "
                   "through risk targeting (Direction 5), not direction bets.")
    else:
        L.append("- No event moves forward volatility beyond the block-"
                 "bootstrap null either.")
    return "\n".join(L)


def main() -> None:
    import argparse
    from pathlib import Path
    from causal_portfolio.data import get_loader

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--long-parquet",
                   default="causal_portfolio/data/cache/iv_search_long.parquet")
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default="causal_portfolio/docs/event_studies.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    long = pd.read_parquet(args.long_parquet)
    long["time"] = pd.to_datetime(long["time"], utc=True).dt.tz_localize(None)
    long["col"] = long["asset"] + "_" + long["metric"]
    wide = (long.pivot_table(index="time", columns="col", values="value",
                             aggfunc="last").resample("D").last())
    wide = wide.loc[(wide.index >= args.start) & (wide.index <= args.end)]

    loader = get_loader()
    returns = loader.load_returns(assets, args.start, args.end)
    returns.index = pd.to_datetime(returns.index)
    btc = returns["btc_return"]
    ew = returns.mean(axis=1)

    events = build_events(wide)
    stats = []
    for name, (ind, note) in events.items():
        st = study_event(name, ind, note, btc, ew, n_boot=args.n_boot)
        if st is None:
            logger.info("skipping %s (too few events/rows)", name)
            continue
        logger.info("%s: n=%d vol_ratio=%.2f (p=%.3f)",
                    name, st.n_events, st.vol_ratio, st.vol_p)
        stats.append(st)

    md = render_markdown(stats, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"{len(stats)} event types -> {args.out}")


if __name__ == "__main__":
    main()
