"""On-chain demand/flow 'DAGs': one on-chain state -> risk-on/off -> return.

Each rule is the smallest causal DAG:  onchain_state_t -> exposure_t ->
return_{t+1}.  The state is a *demand or flow* proxy read from the warehouse
(exchange net flow, active addresses, fees/congestion, DeFi TVL, stablecoin
supply).  Risk-on holds the asset (BTC or ETH); risk-off rotates to 0% (cash /
stables — conservative, a real stable yield would only help).  Decision uses
data through day t and is applied to day t+1's return (one-day lag), with a
switching cost on every exposure change.

We REUSE the regime-rotation harness verbatim — ``backtest_rule`` (1-day lag,
fee, metrics) and ``placebo_p`` (circular time-shift null) — and only add the
on-chain signal builders.  Every signal is causal: trailing moving-average
crosses and trailing z-scores, never a full-sample threshold.  Winners are
placebo-tested so we don't mistake "average exposure level" for "timing skill".

Strategies (built only where the data exists; see the markdown caveats for the
exact coverage window of each series):

  1. exch_flow   Exchange-flow gate.  Risk-ON when exchange *net flow* is
                 negative (coins leaving exchanges = accumulation), risk-OFF
                 when inflows spike.  Trailing z-score of the flow series.
                 ETH: ``cex_netflow_usd`` (best coverage).  BTC: native
                 ``FlowInExNtv - FlowOutExNtv`` (negative = net outflow).
  2. addr_growth Network-growth gate.  Risk-ON when active addresses
                 (``AdrActCnt``) are above their trailing 30d MA.
  3. fee_demand  Fee/congestion demand gate.  Risk-ON when fees (``fees`` USD,
                 a demand proxy; the repo flagged a candidate
                 chain-congestion->BTC edge) are above their trailing 30d MA.
  4. tvl_trend   TVL-trend gate.  Risk-ON when DeFi ``tvl_usd`` is above its
                 trailing 30d MA (capital flowing into DeFi).
  5. stbl_supply Stablecoin-supply gate.  Risk-ON when aggregate stablecoin
                 supply (USDT+USDC+USDe ``SplyCur``/``mc``) is above its
                 trailing 30d MA (dry-powder inflow).  Applied to both BTC and
                 ETH (it is a market-wide liquidity signal).

Scoring vs buy-and-hold of the SAME asset: annualized return, Sharpe, max
drawdown, Calmar, % time in market, # switches, plus the placebo p-value on
Sharpe.  A rule is only interesting if it (a) beats BH on Sharpe and (b) its
TIMING survives the placebo (p < 0.10).

Run:  python -m causal_portfolio.experiments.onchain_flow_dags
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

# REUSE the regime-rotation harness (do not edit it).
from causal_portfolio.experiments.regime_rotation import backtest_rule, placebo_p

logger = logging.getLogger("cpcm.experiments.onchain_flow_dags")

DB_PATH = "causal_portfolio/data/cpcm_local.duckdb"

# Most warehouse on-chain series carry roughly a 1-day publication lag
# (end-of-day chain aggregates land the following day).  The harness already
# applies a one-day decision->return lag; we add ONE extra business-day shift
# on the raw series so the signal at decision time only uses data that would
# actually be published by then.
PUBLICATION_LAG_DAYS = 1


# ── warehouse access ────────────────────────────────────────────────

def _load_metric(con, asset: str, metric: str) -> pd.Series:
    """Daily series for (asset, metric) from asset_metrics_best, indexed by date."""
    df = con.sql(
        f"""
        select cast(time at time zone 'UTC' as date) as day, avg(value) as v
        from asset_metrics_best
        where asset = '{asset}' and metric = '{metric}'
        group by 1 order by 1
        """
    ).df()
    if not len(df):
        return pd.Series(dtype=float)
    return pd.Series(df["v"].values, index=pd.to_datetime(df["day"])).sort_index()


def _align(s: pd.Series, idx: pd.DatetimeIndex, *, ffill_limit: int = 5) -> pd.Series:
    """Align an on-chain series to the daily return index.

    Applies the publication lag, reindexes to the return calendar, and
    forward-fills small gaps only (``ffill_limit`` days) so a stale value never
    silently carries a quarter.
    """
    s = s.shift(PUBLICATION_LAG_DAYS)
    return s.reindex(idx.union(s.index)).ffill(limit=ffill_limit).reindex(idx)


# ── causal signal primitives ────────────────────────────────────────

def _ma_cross_up(s: pd.Series, window: int = 30) -> pd.Series:
    """1 when s is above its trailing MA (risk-on), else 0.  Causal: the MA at
    t uses s[..t] only."""
    ma = s.rolling(window, min_periods=window // 2).mean()
    return (s > ma).astype(float)


def _zscore(s: pd.Series, window: int = 30) -> pd.Series:
    """Trailing z-score (causal): (s - rolling_mean) / rolling_std."""
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / (sd + 1e-12)


# ── strategy builders: each returns {name: exposure Series in {0,1}} ─

def build_exch_flow(con, idx, asset: str) -> dict[str, pd.Series]:
    """Risk-ON when exchange net flow is negative (net outflow = accumulation).

    Uses a trailing z-score: ON when z < +threshold (i.e. inflows are not in
    their upper tail).  Negative/low net flow -> ON; an inflow spike -> OFF.
    """
    rules: dict[str, pd.Series] = {}
    if asset == "eth":
        netflow = _load_metric(con, "eth", "cex_netflow_usd")
        src = "cex_netflow_usd"
    else:
        fin = _load_metric(con, asset, "FlowInExNtv")
        fout = _load_metric(con, asset, "FlowOutExNtv")
        if not len(fin) or not len(fout):
            return rules
        netflow = (fin - fout).dropna()          # negative = net outflow = bullish
        src = "FlowInExNtv-FlowOutExNtv"
    if not len(netflow):
        return rules
    f = _align(netflow, idx)
    z = _zscore(f, 30)
    # Risk-ON unless inflows spike into the upper tail.
    rules[f"exch_flow[{asset}|{src}]"] = (z < 1.0).astype(float)
    # Stricter variant: ON only on genuine net outflow (z below zero-ish).
    rules[f"exch_flow_strict[{asset}|{src}]"] = (z < 0.0).astype(float)
    return rules


def build_addr_growth(con, idx, asset: str) -> dict[str, pd.Series]:
    """Risk-ON when active addresses are above their trailing 30d MA."""
    s = _load_metric(con, asset, "AdrActCnt")
    if not len(s):
        return {}
    a = _align(s, idx)
    return {f"addr_growth[{asset}|AdrActCnt]": _ma_cross_up(a, 30)}


def build_fee_demand(con, idx, asset: str) -> dict[str, pd.Series]:
    """Risk-ON when fees (USD demand proxy) are above their trailing 30d MA."""
    s = _load_metric(con, asset, "fees")
    if not len(s):
        return {}
    a = _align(s, idx)
    return {f"fee_demand[{asset}|fees]": _ma_cross_up(a, 30)}


def build_tvl_trend(con, idx, asset: str) -> dict[str, pd.Series]:
    """Risk-ON when DeFi TVL is above its trailing 30d MA."""
    s = _load_metric(con, asset, "tvl_usd")
    if not len(s):
        return {}
    a = _align(s, idx)
    return {f"tvl_trend[{asset}|tvl_usd]": _ma_cross_up(a, 30)}


def _agg_stablecoin_supply(con) -> pd.Series:
    """USDT+USDC+USDe aggregate circulating supply (USD).

    USDT/USDC use ``SplyCur`` (full history, ~$1 each so USD≈units); USDe is
    young (joins late 2023) and only reliably exposed via ``mc``.  We sum
    whatever is available each day, so the series simply gains the USDe leg once
    it begins — a real-world liquidity aggregate.
    """
    legs = []
    for a, m in [("usdt", "SplyCur"), ("usdc", "SplyCur"), ("usde", "mc")]:
        s = _load_metric(con, a, m)
        if len(s):
            legs.append(s.rename(f"{a}_{m}"))
    if not legs:
        return pd.Series(dtype=float)
    wide = pd.concat(legs, axis=1).sort_index()
    return wide.sum(axis=1, min_count=1)


def build_stbl_supply(con, idx, asset: str, agg: pd.Series) -> dict[str, pd.Series]:
    """Risk-ON when aggregate stablecoin supply is above its trailing 30d MA."""
    if not len(agg):
        return {}
    a = _align(agg, idx)
    return {f"stbl_supply[{asset}|USDT+USDC+USDe]": _ma_cross_up(a, 30)}


def build_all_rules(con, idx, asset: str, stbl_agg: pd.Series) -> dict[str, pd.Series]:
    rules: dict[str, pd.Series] = {"always_in": pd.Series(1.0, index=idx)}
    rules.update(build_exch_flow(con, idx, asset))
    rules.update(build_addr_growth(con, idx, asset))
    rules.update(build_fee_demand(con, idx, asset))
    rules.update(build_tvl_trend(con, idx, asset))
    rules.update(build_stbl_supply(con, idx, asset, stbl_agg))
    return rules


# ── per-asset run ───────────────────────────────────────────────────

@dataclass
class Row:
    asset: str
    name: str
    ann_return: float
    sharpe: float
    max_dd: float
    calmar: float
    pct_in_market: float
    n_switches: int
    placebo_p: float | None
    coverage: str


def _coverage_str(con, asset: str, rule_name: str, idx) -> str:
    """Human-readable coverage window for the series behind a rule."""
    spec = {
        "exch_flow": ("eth", "cex_netflow_usd") if asset == "eth"
        else (asset, "FlowInExNtv"),
        "addr_growth": (asset, "AdrActCnt"),
        "fee_demand": (asset, "fees"),
        "tvl_trend": (asset, "tvl_usd"),
        "stbl_supply": ("usdt", "SplyCur"),
    }
    key = rule_name.split("[")[0].replace("_strict", "")
    if key not in spec:
        return ""
    a, m = spec[key]
    df = con.sql(
        f"""select min(cast(time as date)) mn, max(cast(time as date)) mx,
                   count(*) n
            from asset_metrics_best where asset='{a}' and metric='{m}'"""
    ).df()
    if not len(df) or df["n"].iloc[0] == 0:
        return "no data"
    mn = pd.Timestamp(df["mn"].iloc[0]).date()
    mx = pd.Timestamp(df["mx"].iloc[0]).date()
    return f"{mn}->{mx}"


def run_asset(con, returns: pd.DataFrame, asset: str, stbl_agg: pd.Series,
              *, fee_bps_oneway: float, n_placebo: int) -> tuple[list[Row], object]:
    ret = returns[f"{asset}_return"].copy()
    ret.index = pd.to_datetime(ret.index)
    idx = ret.index
    rules = build_all_rules(con, idx, asset, stbl_agg)

    results = {}
    for name, exp in rules.items():
        r = backtest_rule(ret, exp, fee_bps_oneway=fee_bps_oneway)
        r.name = name
        results[name] = r
        logger.info("%-4s %-34s ann=%+.0f%% sharpe=%.2f maxDD=%.0f%% "
                    "calmar=%.2f in=%.0f%%", asset, name, r.ann_return * 100,
                    r.sharpe, r.max_dd * 100, r.calmar, r.pct_in_market * 100)

    bh = results["always_in"]
    rows: list[Row] = []
    for name, r in results.items():
        pp = None
        if name != "always_in" and (r.sharpe > bh.sharpe or r.calmar > bh.calmar):
            pp = placebo_p(ret, rules[name], r, metric="sharpe",
                           n=n_placebo, fee_bps_oneway=fee_bps_oneway)
            logger.info("%-4s placebo %-34s p(Sharpe)=%.2f", asset, name, pp)
        cov = "—" if name == "always_in" else _coverage_str(con, asset, name, idx)
        rows.append(Row(asset, name, r.ann_return, r.sharpe, r.max_dd, r.calmar,
                        r.pct_in_market, r.n_switches, pp, cov))
    return rows, bh


# ── report ──────────────────────────────────────────────────────────

def render_markdown(rows_by_asset: dict, bh_by_asset: dict, args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# On-chain demand/flow DAGs (state -> risk-on/off -> return)\n"]
    L.append("Each rule rotates one asset between full exposure and 0% (cash) "
             "on a single **on-chain demand/flow** state variable. Decision at "
             "day t (data through t, plus a 1-day publication lag on the raw "
             "series), applied to day t+1's return, "
             f"{args['fee_bps_oneway']:.0f}bp one-way switching cost. Every "
             "signal is causal (trailing 30d MA cross or trailing z-score; no "
             "full-sample thresholds). Benchmark is buy-and-hold of the SAME "
             "asset.\n")
    L.append(f"- Window `{args['start']}` -> `{args['end']}`")
    L.append(f"- Placebo: {args['n_placebo']} circular time-shifts of the "
             "exposure path; p = share matching/beating the real rule's "
             "Sharpe. High p => the timing carries no information.")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    for asset in ("btc", "eth"):
        rows = rows_by_asset[asset]
        bh = bh_by_asset[asset]
        L.append(f"## {asset.upper()}\n")
        L.append("| Rule | Ann.ret | Sharpe | MaxDD | Calmar | %in mkt | "
                 "switches | placebo p | coverage |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in sorted(rows, key=lambda r: -r.calmar):
            pp = f"{r.placebo_p:.2f}" if r.placebo_p is not None else "—"
            tag = " <- BH" if r.name == "always_in" else ""
            L.append(
                f"| {r.name}{tag} | {r.ann_return:+.0%} | {r.sharpe:.2f} | "
                f"{r.max_dd:.0%} | {r.calmar:.2f} | {r.pct_in_market:.0%} | "
                f"{r.n_switches} | {pp} | {r.coverage} |")
        L.append("")
        L.append(f"Buy-and-hold {asset.upper()}: ann {bh.ann_return:+.0%}, "
                 f"Sharpe {bh.sharpe:.2f}, maxDD {bh.max_dd:.0%}, "
                 f"Calmar {bh.calmar:.2f}.\n")

    # Verdict across both assets.
    L.append("## Verdict\n")
    survivors = []
    for asset in ("btc", "eth"):
        bh = bh_by_asset[asset]
        for r in rows_by_asset[asset]:
            if (r.name != "always_in" and r.placebo_p is not None
                    and r.placebo_p < 0.10 and r.sharpe > bh.sharpe):
                survivors.append(r)
    if survivors:
        L.append("Rules that beat BH on Sharpe AND whose timing survives the "
                 "placebo (p < 0.10):")
        for r in sorted(survivors, key=lambda r: -r.sharpe):
            bh = bh_by_asset[r.asset]
            L.append(f"- **{r.name}** ({r.asset.upper()}) — Sharpe {r.sharpe:.2f} "
                     f"vs {bh.sharpe:.2f}, maxDD {r.max_dd:.0%} vs "
                     f"{bh.max_dd:.0%}, {r.n_switches} switches "
                     f"(placebo p={r.placebo_p:.2f})")
        L.append("\nRead this conservatively: a placebo p near the 0.10 cutoff "
                 "is a weak pass, not a strong edge. Survivors with heavy "
                 "turnover (hundreds of switches) are also the most cost- and "
                 "lag-sensitive, and none clears BH out-of-sample on an "
                 "independent cycle (we only have one).")
    else:
        # Did anything at least beat BH Sharpe (before placebo)?
        beat = [r for asset in ("btc", "eth") for r in rows_by_asset[asset]
                if r.name != "always_in" and r.sharpe > bh_by_asset[asset].sharpe]
        L.append("**No on-chain demand/flow state both beats BH on Sharpe and "
                 "survives the placebo.** " +
                 (f"{len(beat)} rule(s) edge out BH on Sharpe, but their TIMING "
                  "does not beat a random-timed overlay with the same average "
                  "exposure — the risk-adjusted improvement comes from sitting "
                  "in cash part of the time, not from skillful flow reads."
                  if beat else
                  "Nothing even out-Sharpes BH; the gates mostly cut return "
                  "without a commensurate drawdown benefit."))
    L.append("\n### Caveats\n")
    L.append("- **Uneven, late-starting coverage.** On-chain series differ in "
             "history (see the coverage column). BTC `tvl_usd` starts "
             "2021-03; USDe joins the stablecoin aggregate only in late 2023; "
             "`cex_netflow_usd` is ETH-only (BTC uses a native FlowIn-FlowOut "
             "proxy in different units).")
    L.append("- **Publication lag.** A 1-day lag is applied to every raw "
             "series before the harness's own 1-day decision->return lag; real "
             "feeds can be later or revised. Gaps are forward-filled at most 5 "
             "days, so a stale read never carries indefinitely.")
    L.append("- **One cycle.** 2021-2025 is a single bull/bear/bull regime. "
             "These rules are not validated out-of-sample on an independent "
             "cycle; a placebo pass is necessary, not sufficient.")
    L.append("- **Cash, not stable yield.** Risk-off earns 0%. A real stable "
             "yield would only improve every gated rule, never hurt it.")
    L.append("- **Consistent with the repo's standing finding:** buy-and-hold "
             "is the benchmark nothing here beats out-of-sample.")
    return "\n".join(L)


# ── main ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from pathlib import Path
    from causal_portfolio.data import get_loader

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--fee-bps-oneway", type=float, default=5.0)
    p.add_argument("--n-placebo", type=int, default=200)
    p.add_argument("--out", default="causal_portfolio/docs/onchain_flow_dags.md")
    args = p.parse_args()

    loader = get_loader()
    returns = loader.load_returns(["btc", "eth"], args.start, args.end)
    returns.index = pd.to_datetime(returns.index)

    con = duckdb.connect(DB_PATH, read_only=True)
    stbl_agg = _agg_stablecoin_supply(con)
    logger.info("aggregate stablecoin supply: %d days, %s -> %s",
                len(stbl_agg),
                stbl_agg.index.min().date() if len(stbl_agg) else "na",
                stbl_agg.index.max().date() if len(stbl_agg) else "na")

    rows_by_asset, bh_by_asset = {}, {}
    for asset in ("btc", "eth"):
        rows, bh = run_asset(con, returns, asset, stbl_agg,
                             fee_bps_oneway=args.fee_bps_oneway,
                             n_placebo=args.n_placebo)
        rows_by_asset[asset] = rows
        bh_by_asset[asset] = bh
    con.close()

    md = render_markdown(rows_by_asset, bh_by_asset, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
