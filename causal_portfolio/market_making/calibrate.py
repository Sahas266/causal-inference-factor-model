"""Fit walk-forward PIN, fill intensity, and markout artifacts from captured data."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone

from causal_portfolio.market_making.artifacts import CalibrationArtifact
from causal_portfolio.market_making.calibration import (
    FillObservation,
    MarkoutObservation,
    fit_exponential_intensity,
    fit_markout_curve,
)
from causal_portfolio.market_making.pin import fit_pin, pin_posteriors
from causal_portfolio.market_making.recorder import (
    MicrostructureStore,
    aggregate_daily_trade_counts,
    capture_day_quality,
)
from causal_portfolio.market_making.types import AggressorSide, TradePrint


def _markouts(path: str | None, horizon_ms: int):
    if path is None:
        return None
    observations = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("horizon_ms") and int(row["horizon_ms"]) != horizon_ms:
                continue
            observations.append(
                MarkoutObservation(
                    pin=float(row["pin"]),
                    aggressor=AggressorSide(row["aggressor"].lower()),
                    adverse_bps=float(row["adverse_bps"]),
                )
            )
    return fit_markout_curve(observations) if observations else None


def _intensities(path: str | None):
    if path is None:
        return None, None
    observations = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            observations.append(
                FillObservation(
                    distance=float(row["distance"]),
                    exposure_seconds=float(row["exposure_seconds"]),
                    fills=int(row["fills"]),
                    side=AggressorSide(row["aggressor"].lower()),
                )
            )
    buy = fit_exponential_intensity(observations, side=AggressorSide.BUY)
    sell = fit_exponential_intensity(observations, side=AggressorSide.SELL)
    return buy, sell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--markouts-csv")
    parser.add_argument("--markout-horizon-ms", type=int, default=5_000)
    parser.add_argument("--fill-observations-csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.window < 20:
        parser.error("--window must be >= 20")
    coin = args.coin.upper()
    store = MicrostructureStore(args.database)
    try:
        events = store.replay_events(coin=coin)
    finally:
        store.close()
    trades = [event for event in events if isinstance(event, TradePrint)]
    daily = aggregate_daily_trade_counts(trades)
    quality = capture_day_quality(events)
    complete_days = [day for day, details in quality.items() if details["complete"]]
    usable_days = [day for day in complete_days if day in daily]
    if len(usable_days) < args.window:
        parser.error(
            f"need {args.window} complete UTC days with trades; found {len(usable_days)}"
        )
    selected_days = usable_days[-args.window :]
    sample = [daily[day] for day in selected_days]
    buys = [value[0] for value in sample]
    sells = [value[1] for value in sample]
    pin_fit = fit_pin(
        buys, sells, min_days=args.window, n_starts=args.starts, seed=0
    )
    posterior = pin_posteriors([buys[-1]], [sells[-1]], pin_fit)[0]
    markouts = _markouts(args.markouts_csv, args.markout_horizon_ms)
    buy_intensity, sell_intensity = _intensities(args.fill_observations_csv)
    data_as_of = datetime.fromisoformat(selected_days[-1]).replace(
        tzinfo=timezone.utc
    ) + timedelta(days=1)
    artifact = CalibrationArtifact(
        coin=coin,
        as_of=data_as_of,
        pin_fit=pin_fit,
        posterior=posterior,
        markouts=markouts,
        buy_intensity=buy_intensity,
        sell_intensity=sell_intensity,
    )
    artifact.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
