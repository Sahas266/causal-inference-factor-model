"""Run one safe-by-default Avellaneda-Stoikov market-making service."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from causal_portfolio.market_making.artifacts import load_calibration_artifact
from causal_portfolio.market_making.engine import MarketMakerConfig, MarketMakerEngine
from causal_portfolio.market_making.hyperliquid import (
    HLMarketMakerAdapter,
    HLMarketMakerConfig,
)
from causal_portfolio.market_making.runner import MarketMakingService, RuntimeConfig
from causal_portfolio.market_making.toxicity import ToxicityConfig, ToxicitySignal
from causal_portfolio.market_making.types import QuotePolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--order-size", type=float, required=True)
    parser.add_argument("--tick-size", type=float, required=True)
    parser.add_argument("--size-decimals", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--horizon-seconds", type=float, default=300.0)
    parser.add_argument(
        "--sigma", type=float, required=True, help="Absolute price vol / sqrt(second)"
    )
    parser.add_argument("--max-inventory", type=float, required=True)
    parser.add_argument("--max-daily-loss-usd", type=float, default=100.0)
    parser.add_argument("--min-free-margin-usd", type=float, default=100.0)
    parser.add_argument("--max-spread-bps", type=float, default=100.0)
    parser.add_argument(
        "--policy",
        choices=[policy.value for policy in QuotePolicy],
        default=QuotePolicy.AS_BASELINE.value,
    )
    parser.add_argument("--calibration")
    parser.add_argument("--max-calibration-age-hours", type=float, default=48.0)
    parser.add_argument("--allow-stale-calibration", action="store_true")
    parser.add_argument("--use-calibrated-intensity", action="store_true")
    parser.add_argument("--mainnet", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--allow-mainnet", action="store_true")
    parser.add_argument("--acknowledge-mainnet", action="store_true")
    parser.add_argument("--dedicated-account", action="store_true")
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.mainnet and args.live and not (
        args.allow_mainnet and args.acknowledge_mainnet and args.dedicated_account
    ):
        parser.error(
            "live mainnet requires --allow-mainnet --acknowledge-mainnet "
            "--dedicated-account"
        )
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    coin = args.coin.upper()
    policy = QuotePolicy(args.policy)
    artifact = load_calibration_artifact(args.calibration) if args.calibration else None
    if artifact is not None and artifact.coin != coin:
        parser.error(f"calibration is for {artifact.coin}, not {coin}")
    if artifact is not None and not args.allow_stale_calibration:
        freshness_error = artifact.freshness_error(args.max_calibration_age_hours)
        if freshness_error:
            parser.error(freshness_error)
    if policy != QuotePolicy.AS_BASELINE:
        if artifact is None or artifact.pin_fit is None or artifact.posterior is None:
            parser.error("PIN policies require --calibration with PIN fit/posterior")
        if policy in {QuotePolicy.AS_PIN_SPREAD, QuotePolicy.AS_PIN_POSTERIOR}:
            if artifact.markouts is None:
                parser.error("spread/posterior policies require calibrated markouts")
    kappa = args.kappa
    arrival_rate = args.arrival_rate
    if args.use_calibrated_intensity:
        if artifact is None or artifact.buy_intensity is None or artifact.sell_intensity is None:
            parser.error("--use-calibrated-intensity requires both side estimates")
        kappa = (artifact.buy_intensity.kappa + artifact.sell_intensity.kappa) / 2
        arrival_rate = (
            artifact.buy_intensity.arrival_rate + artifact.sell_intensity.arrival_rate
        ) / 2
    mm_config = MarketMakerConfig(
        coin=coin,
        order_size=args.order_size,
        tick_size=args.tick_size,
        size_decimals=args.size_decimals,
        gamma=args.gamma,
        kappa=kappa,
        horizon_seconds=args.horizon_seconds,
        arrival_rate=arrival_rate,
        max_inventory_base=args.max_inventory,
        max_daily_loss_usd=args.max_daily_loss_usd,
        min_free_margin_usd=args.min_free_margin_usd,
        max_total_spread_bps=args.max_spread_bps,
        toxicity=ToxicityConfig(policy=policy),
    )
    network = "mainnet" if args.mainnet else "testnet"
    adapter_config = HLMarketMakerConfig(
        testnet=not args.mainnet,
        dry_run=not args.live,
        allow_mainnet=args.allow_mainnet,
        dedicated_account=args.dedicated_account,
        state_path=Path(f"data/market_making/{network}-{coin.lower()}-cloids.json"),
    )
    adapter = HLMarketMakerAdapter(adapter_config)
    signal = None
    if artifact is not None and artifact.pin_fit is not None and artifact.posterior is not None:
        signal = ToxicitySignal(
            pin=artifact.pin_fit.pin,
            informed_buy_probability=artifact.posterior.informed_buy,
            informed_sell_probability=artifact.posterior.informed_sell,
            estimator_usable=artifact.pin_fit.usable,
        )
    service = MarketMakingService(
        MarketMakerEngine(mm_config),
        adapter,
        RuntimeConfig(
            fallback_sigma_per_sqrt_second=args.sigma,
            acknowledge_mainnet=args.acknowledge_mainnet,
        ),
        toxicity_signal=signal,
        markouts=artifact.markouts if artifact else None,
        pin_fit=artifact.pin_fit if artifact else None,
    )
    service.start()
    started = time.monotonic()
    try:
        while args.duration_seconds == 0 or time.monotonic() - started < args.duration_seconds:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
