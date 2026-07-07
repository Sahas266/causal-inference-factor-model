"""8h funding -> next-8h returns: the intraday-alignment experiment.

The daily-horizon program ended in a structural null: on-chain factors are
downstream of returns at daily bars (`causal_directions_summary.md`), and the
daily funding gates were exact ties with buy-and-hold
(`dag_strategies_findings.md` section D). The one alignment it left open is
the funding epoch itself: the rate paid at epoch t is fixed by the premium
*before* t, and the next-8h return is measured strictly after, so the
reverse-causality objection is weakest here.

Question: does the funding state known at t predict the (t, t+8h] return?

Pre-registered design (kept deliberately small — see the multiplicity caveat
in `dag_strategies_findings.md`):

1. Predictive regression: next-8h return on funding z-score, Newey-West HAC
   t-stats, with and without trailing-return/vol controls. BTC and ETH.
2. Three long/flat overlays at 5bp one-way costs vs buy-and-hold on the same
   grid, each placebo-tested by circular shift when it beats BH on Sharpe:
   - extreme_off: flat when funding is above its rolling q90 (crowded longs);
   - carry_on:    long only when funding <= 0 (paid to hold);
   - band:        long only when funding is inside its rolling [q10, q90].
3. The same machinery on the 24h grid as the daily control (the known tie).

Data:
- Prices: hourly WBTC/WETH from Dune spellbook `prices.hour`
  (query https://dune.com/queries/7868219), cached at
  `causal_portfolio/data/cache/btc_eth_hourly_dune.csv`. Wrapped-coin spot is
  a proxy for the perp: the basis is a few bps, far below 8h vol. Re-pull the
  query to refresh.
- Funding: warehouse `funding_rate_8h` (provider=hyperliquid), hourly
  observations, 2023-11 -> 2026-01, from the local DuckDB mirror.

Alignment contract: signals at grid time t use only funding observations in
(t-8h, t] and prices <= t; the return earned is p(t+8h)/p(t) - 1. Fees are
charged on every exposure change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_portfolio.backtest.metrics import max_drawdown, sharpe_ratio

PRICES_CSV = Path("causal_portfolio/data/cache/btc_eth_hourly_dune.csv")
DUCKDB_PATH = Path("causal_portfolio/data/cpcm_local.duckdb")

HOURS_PER_BAR = {"8h": 8, "24h": 24}
BARS_PER_YEAR = {"8h": 3 * 365, "24h": 365}
ROLL_DAYS = 90          # rolling window for z-scores / quantiles, in days
MIN_ROLL_DAYS = 30
FEE_BPS_ONEWAY = 5.0
PLACEBO_N = 200


# ── data ─────────────────────────────────────────────────────────────

def load_hourly_prices(path: Path = PRICES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["ts"].str.replace(" UTC", "", regex=False), utc=True)
    out = pd.DataFrame(
        {"btc": df["btc_price"].values, "eth": df["eth_price"].values},
        index=ts,
    ).sort_index()
    return out


def load_hourly_funding(asset: str, db: Path = DUCKDB_PATH) -> pd.Series:
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT time, value FROM asset_metrics_best
            WHERE provider = 'hyperliquid' AND metric = 'funding_rate_8h'
              AND lower(asset) = ?
            ORDER BY time
            """,
            [asset.lower()],
        ).df()
    finally:
        con.close()
    ts = pd.to_datetime(rows["time"], utc=True).dt.floor("h")
    s = pd.Series(rows["value"].values, index=ts, name=f"{asset}_funding")
    return s[~s.index.duplicated(keep="last")].sort_index()


# ── alignment ────────────────────────────────────────────────────────

def build_panel(
    prices: pd.Series, funding: pd.Series, bar: str
) -> pd.DataFrame:
    """One row per bar-start t: forward return over (t, t+bar], and funding
    state using only observations in (t-bar, t]."""
    hours = HOURS_PER_BAR[bar]
    grid = prices.resample(f"{hours}h").last()  # stamped at bar start, value = last price <= end
    # `resample().last()` labels the bar by its START but uses prices up to
    # its END — so grid[t] is the price AT t+bar. Re-index so px[t] is the
    # price at t exactly: take the hourly price at the grid timestamps.
    px = prices.reindex(grid.index)
    fwd_ret = px.shift(-1) / px - 1.0  # (t, t+bar] return, indexed at t

    trailing = funding.rolling(f"{hours}h", closed="right").mean()
    fund = trailing.reindex(grid.index)
    obs_count = funding.rolling(f"{hours}h", closed="right").count().reindex(grid.index)
    fund = fund.where(obs_count >= max(1, int(hours * 0.75)))

    roll = ROLL_DAYS * 24 // hours
    min_roll = MIN_ROLL_DAYS * 24 // hours
    z = (fund - fund.rolling(roll, min_periods=min_roll).mean()) / fund.rolling(
        roll, min_periods=min_roll
    ).std()
    q10 = fund.rolling(roll, min_periods=min_roll).quantile(0.10)
    q90 = fund.rolling(roll, min_periods=min_roll).quantile(0.90)

    trail_ret = px / px.shift(1) - 1.0
    trail_vol = trail_ret.rolling(roll // 3, min_periods=min_roll // 3).std()

    return pd.DataFrame({
        "px": px, "fwd_ret": fwd_ret, "fund": fund, "z": z,
        "q10": q10, "q90": q90, "trail_ret": trail_ret, "trail_vol": trail_vol,
    })


# ── tests ────────────────────────────────────────────────────────────

def hac_regression(panel: pd.DataFrame, controls: bool) -> dict:
    cols = ["z"] + (["trail_ret", "trail_vol"] if controls else [])
    data = panel[["fwd_ret"] + cols].dropna()
    X = sm.add_constant(data[cols])
    fit = sm.OLS(data["fwd_ret"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return {
        "n": int(fit.nobs),
        "beta_bps": float(fit.params["z"] * 1e4),
        "t": float(fit.tvalues["z"]),
        "p": float(fit.pvalues["z"]),
    }


@dataclass
class RuleResult:
    name: str
    ann_return: float
    sharpe: float
    max_dd: float
    pct_in_market: float
    n_switches: int
    net: pd.Series


def backtest(panel: pd.DataFrame, exposure: pd.Series, bar: str, name: str) -> RuleResult:
    ppy = BARS_PER_YEAR[bar]
    e = exposure.astype(float).clip(0, 1).fillna(0.0)
    net = e * panel["fwd_ret"] - e.diff().abs().fillna(e.abs()) * FEE_BPS_ONEWAY / 1e4
    net = net.dropna()
    return RuleResult(
        name=name,
        ann_return=float(net.mean() * ppy),
        sharpe=float(sharpe_ratio(net.values, annualization=ppy)),
        max_dd=float(max_drawdown(net.values)),
        pct_in_market=float((e > 0.01).mean()),
        n_switches=int((e.diff().abs() > 0.01).sum()),
        net=net,
    )


def placebo_p(panel: pd.DataFrame, exposure: pd.Series, real: RuleResult,
              bar: str, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    e = exposure.astype(float).clip(0, 1).fillna(0.0).values
    hits = 0
    for _ in range(PLACEBO_N):
        k = int(rng.integers(30, len(e) - 30))
        shifted = pd.Series(np.roll(e, k), index=panel.index)
        hits += backtest(panel, shifted, bar, "shift").sharpe >= real.sharpe
    return hits / PLACEBO_N


def run_asset(asset: str, prices: pd.DataFrame, bar: str) -> dict:
    funding = load_hourly_funding(asset)
    panel = build_panel(prices[asset], funding, bar)
    # evaluate only where every signal is defined, one common window
    valid = panel.dropna(subset=["fwd_ret", "z", "q10", "q90"]).index
    panel = panel.loc[valid]

    reg_plain = hac_regression(panel, controls=False)
    reg_ctrl = hac_regression(panel, controls=True)

    exposures = {
        "bh": pd.Series(1.0, index=panel.index),
        "extreme_off": (panel["fund"] <= panel["q90"]).astype(float),
        "carry_on": (panel["fund"] <= 0).astype(float),
        "band": ((panel["fund"] >= panel["q10"]) & (panel["fund"] <= panel["q90"])).astype(float),
    }
    results, placebos = {}, {}
    bh = backtest(panel, exposures["bh"], bar, "bh")
    results["bh"] = bh
    for name, e in exposures.items():
        if name == "bh":
            continue
        r = backtest(panel, e, bar, name)
        results[name] = r
        if r.sharpe > bh.sharpe:
            placebos[name] = placebo_p(panel, e, r, bar)
    return {
        "asset": asset, "bar": bar,
        "start": str(panel.index[0].date()), "end": str(panel.index[-1].date()),
        "n_bars": len(panel),
        "reg_plain": reg_plain, "reg_ctrl": reg_ctrl,
        "results": results, "placebos": placebos,
    }


def render(out: dict) -> str:
    lines = [
        f"## {out['asset'].upper()} @ {out['bar']}  "
        f"({out['start']} -> {out['end']}, {out['n_bars']} bars)",
        f"- regression fwd_ret ~ funding_z:        "
        f"beta={out['reg_plain']['beta_bps']:+.2f} bps/sigma  "
        f"t={out['reg_plain']['t']:+.2f}  p={out['reg_plain']['p']:.3f}  (n={out['reg_plain']['n']})",
        f"- with trailing ret/vol controls:        "
        f"beta={out['reg_ctrl']['beta_bps']:+.2f} bps/sigma  "
        f"t={out['reg_ctrl']['t']:+.2f}  p={out['reg_ctrl']['p']:.3f}",
        "",
        "| rule | ann | sharpe | maxDD | in-mkt | switches | placebo p |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for name, r in out["results"].items():
        p = out["placebos"].get(name)
        lines.append(
            f"| {name} | {r.ann_return:+.1%} | {r.sharpe:.2f} | {r.max_dd:.1%} "
            f"| {r.pct_in_market:.0%} | {r.n_switches} "
            f"| {'-' if p is None else f'{p:.2f}'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", default="btc,eth")
    args = parser.parse_args()
    prices = load_hourly_prices()
    for asset in args.assets.split(","):
        for bar in ("8h", "24h"):
            print(render(run_asset(asset.strip(), prices, bar)))
            print()


if __name__ == "__main__":
    main()
