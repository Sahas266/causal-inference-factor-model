"""Compare AS/PIN policies on one deterministic Hyperliquid event window."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime

from causal_portfolio.market_making.artifacts import load_calibration_artifact
from causal_portfolio.market_making.engine import MarketMakerConfig
from causal_portfolio.market_making.recorder import MicrostructureStore
from causal_portfolio.market_making.replay import compare_policies
from causal_portfolio.market_making.simulator import SimulationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--order-size", type=float, required=True)
    parser.add_argument("--tick-size", type=float, required=True)
    parser.add_argument("--size-decimals", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=300)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--max-inventory", type=float, required=True)
    parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=int, default=50)
    parser.add_argument("--start", help="ISO-8601 replay start; defaults to calibration as_of")
    parser.add_argument("--end", help="ISO-8601 replay end")
    parser.add_argument("--output")
    args = parser.parse_args()
    coin = args.coin.upper()
    artifact = load_calibration_artifact(args.calibration)
    if artifact.coin != coin:
        parser.error(f"calibration is for {artifact.coin}, not {coin}")
    start_ms = int(artifact.as_of.timestamp() * 1_000)
    if args.start:
        start_ms = int(
            datetime.fromisoformat(args.start.replace("Z", "+00:00")).timestamp()
            * 1_000
        )
    end_ms = None
    if args.end:
        end_ms = int(
            datetime.fromisoformat(args.end.replace("Z", "+00:00")).timestamp()
            * 1_000
        )
    store = MicrostructureStore(args.database)
    try:
        events = store.replay_events(coin=coin, start_ms=start_ms, end_ms=end_ms)
    finally:
        store.close()
    config = MarketMakerConfig(
        coin=coin,
        order_size=args.order_size,
        tick_size=args.tick_size,
        size_decimals=args.size_decimals,
        gamma=args.gamma,
        kappa=args.kappa,
        horizon_seconds=args.horizon_seconds,
        max_inventory_base=args.max_inventory,
    )
    results = compare_policies(
        events,
        config,
        artifact,
        fallback_sigma=args.sigma,
        simulation_config=SimulationConfig(
            latency_ms=args.latency_ms,
            maker_fee_bps=args.maker_fee_bps,
        ),
    )
    payload = {}
    for policy, result in results.items():
        summary = asdict(result)
        summary.pop("fills", None)
        payload[policy.value] = summary
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    rendered = json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
