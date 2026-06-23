"""CLI for read-only Hyperliquid microstructure capture."""

from __future__ import annotations

import argparse
import json
import time

from causal_portfolio.market_making.recorder import HLStreamRecorder, MicrostructureStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coins", default="BTC,ETH", help="Comma-separated HL coin names")
    parser.add_argument("--output", default="data/market_making/hyperliquid.duckdb")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="0 runs until Ctrl-C")
    args = parser.parse_args()
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be >= 0")
    coins = tuple(x.strip().upper() for x in args.coins.split(",") if x.strip())
    store = MicrostructureStore(args.output)
    recorder = HLStreamRecorder(coins, store, testnet=args.testnet)
    recorder.start()
    started = time.monotonic()
    try:
        while args.duration_seconds == 0 or time.monotonic() - started < args.duration_seconds:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        print(json.dumps(store.coverage(), indent=2))
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
