"""Can the causal graph time a delta-neutral funding carry?

The one direction the causal program validated is INTO funding: the
chain_congestion -> funding edge survived DML placebos, and causal discovery
found returns -> on-chain state (docs/causal_directions_summary.md,
docs/iv_findings.md). Funding itself is directly monetizable: short perp +
long spot earns the funding rate when positive. So the honest question is
not "do factors predict returns" (dead) but:

    Does forecasting funding with its causal parents (congestion, returns)
    gate the carry better than naive persistence?

Arms (pre-registered, decided at end of day t, earning day t+1's accrual):
  0. always_on   — hold the carry every day.
  1. sign_gate   — hold when today's funding accrual > 0 (naive persistence).
  2. ar_gate     — hold when trailing-252d AR(1) forecast of tomorrow > 0.
  3. causal_gate — hold when trailing-252d forecast from
                   [funding_t, gas_utilization_{t-1}, btc_return_t] > 0.

The causal graph's incremental value is arm 3 vs arm 2 (same machinery, only
the parents added), plus the raw one-step forecast MSE comparison.

Accounting: daily accrual = mean(hourly funding_rate_8h obs in the day) * 3
(three 8h funding periods/day) on unit notional. Position changes cost 20bp
(two legs, ~10bp taker+slippage each). Basis PnL of the hedged position is
ignored (small, zero-mean) — noted in the doc. Sharpe annualization 365.

Placebo: circular shift of each gated arm's position series (n=200); p =
share of shifts with Sharpe >= real. High p = the gate's timing is noise.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import max_drawdown, sharpe_ratio

DUCKDB_PATH = Path("causal_portfolio/data/cpcm_local.duckdb")
FIT_WINDOW = 252
SWITCH_COST = 20 / 1e4  # per position change, both legs
PLACEBO_N = 200


# ── data ─────────────────────────────────────────────────────────────

def _query(sql: str, params: list) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()


def load_daily_funding(asset: str) -> pd.Series:
    """Daily carry accrual: mean of the day's hourly funding_rate_8h obs x 3."""
    rows = _query(
        """SELECT time, value FROM asset_metrics_best
           WHERE provider='hyperliquid' AND metric='funding_rate_8h'
             AND lower(asset)=? ORDER BY time""",
        [asset.lower()],
    )
    ts = pd.to_datetime(rows["time"], utc=True).dt.tz_convert(None)
    hourly = pd.Series(rows["value"].values, index=ts).sort_index()
    daily = hourly.resample("D").mean() * 3.0
    count = hourly.resample("D").count()
    return daily.where(count >= 18)  # require most of the day observed


def load_daily_metric(asset: str, metric: str) -> pd.Series:
    rows = _query(
        """SELECT time, value FROM asset_metrics_best
           WHERE lower(asset)=? AND metric=? ORDER BY time""",
        [asset.lower(), metric],
    )
    ts = pd.to_datetime(rows["time"], utc=True).dt.tz_convert(None)
    return (
        pd.Series(rows["value"].values, index=ts).sort_index()
        .resample("D").last()
    )


def build_panel(asset: str) -> pd.DataFrame:
    """One row per day t with tomorrow's accrual and today's known regressors."""
    funding = load_daily_funding(asset)
    gas = load_daily_metric("eth", "avg_gas_utilization")
    px = load_daily_metric(asset, "price")
    ret = px.pct_change()

    df = pd.DataFrame({
        "accrual_fwd": funding.shift(-1),  # earned over day t+1
        "funding": funding,                # known at end of day t
        # Daily on-chain aggregates realistically publish next day — lag 1.
        "gas_util": gas.shift(1),
        "ret": ret,                        # from prices through t
    })
    return df.dropna()


# ── forecasting gates ────────────────────────────────────────────────

def rolling_forecast(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """One-step forecast of accrual_fwd from `cols`, trailing-window OLS.

    The regression at day t is fit on rows [t-FIT_WINDOW, t) — strictly past
    (each row r pairs regressors at r with the NEXT day's accrual, so the
    last training pair uses accrual through day t). Forecast applies the fit
    to day t's regressors.
    """
    y = df["accrual_fwd"].values
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    out = np.full(len(df), np.nan)
    for t in range(FIT_WINDOW, len(df)):
        Xw, yw = X[t - FIT_WINDOW:t], y[t - FIT_WINDOW:t]
        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[t] = X[t] @ beta
    return pd.Series(out, index=df.index)


@dataclass
class ArmResult:
    name: str
    ann_return: float
    sharpe: float
    max_dd: float
    pct_in_market: float
    n_switches: int
    net: pd.Series


def backtest(df: pd.DataFrame, position: pd.Series, name: str) -> ArmResult:
    pos = position.astype(float).clip(0, 1).fillna(0.0)
    net = pos * df["accrual_fwd"] - pos.diff().abs().fillna(pos.abs()) * SWITCH_COST
    net = net.dropna()
    return ArmResult(
        name=name,
        ann_return=float(net.mean() * 365),
        sharpe=float(sharpe_ratio(net.values)),
        max_dd=float(max_drawdown(net.values)),
        pct_in_market=float((pos > 0).mean()),
        n_switches=int((pos.diff().abs() > 0).sum()),
        net=net,
    )


def placebo_p(df: pd.DataFrame, position: pd.Series, real: ArmResult,
              seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    pos = position.astype(float).clip(0, 1).fillna(0.0).values
    hits = 0
    for _ in range(PLACEBO_N):
        k = int(rng.integers(30, len(pos) - 30))
        shifted = pd.Series(np.roll(pos, k), index=df.index)
        hits += backtest(df, shifted, "shift").sharpe >= real.sharpe
    return hits / PLACEBO_N


def run_asset(asset: str) -> dict:
    df = build_panel(asset)

    pred_ar = rolling_forecast(df, ["funding"])
    pred_causal = rolling_forecast(df, ["funding", "gas_util", "ret"])

    # Common evaluation window: both forecasts defined.
    valid = pred_ar.notna() & pred_causal.notna()
    df, pred_ar, pred_causal = df[valid], pred_ar[valid], pred_causal[valid]

    # Forecast quality (the direct causal-increment readout).
    err_ar = float(((pred_ar - df["accrual_fwd"]) ** 2).mean())
    err_causal = float(((pred_causal - df["accrual_fwd"]) ** 2).mean())

    positions = {
        "always_on": pd.Series(1.0, index=df.index),
        "sign_gate": (df["funding"] > 0).astype(float),
        "ar_gate": (pred_ar > 0).astype(float),
        "causal_gate": (pred_causal > 0).astype(float),
    }
    results, placebos = {}, {}
    for name, pos in positions.items():
        r = backtest(df, pos, name)
        results[name] = r
        if name != "always_on":
            placebos[name] = placebo_p(df, pos, r)
    return {
        "asset": asset,
        "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
        "n_days": len(df),
        "mse_ar": err_ar, "mse_causal": err_causal,
        "results": results, "placebos": placebos,
    }


def render(out: dict) -> str:
    lines = [
        f"## {out['asset'].upper()} carry ({out['start']} -> {out['end']}, "
        f"{out['n_days']} days)",
        f"- forecast MSE: AR(1)={out['mse_ar']:.3e}  "
        f"AR+parents={out['mse_causal']:.3e}  "
        f"(causal delta {100 * (out['mse_causal'] / out['mse_ar'] - 1):+.2f}%)",
        "",
        "| arm | ann | sharpe | maxDD | in-mkt | switches | placebo p |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for name, r in out["results"].items():
        p = out["placebos"].get(name)
        lines.append(
            f"| {name} | {r.ann_return:+.2%} | {r.sharpe:.2f} | {r.max_dd:.2%} "
            f"| {r.pct_in_market:.0%} | {r.n_switches} "
            f"| {'-' if p is None else f'{p:.2f}'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", default="btc,eth")
    args = parser.parse_args()
    for asset in args.assets.split(","):
        print(render(run_asset(asset.strip())))
        print()


if __name__ == "__main__":
    main()
