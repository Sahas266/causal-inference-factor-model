"""Minimal-size, round-trip testnet A/B for trend-rotation execution."""

from __future__ import annotations

import bisect
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from causal_portfolio.execution.orderbook import parse_fill_response
from causal_portfolio.execution.orderbook import (
    L2Book as ExecutionL2Book,
    L2Level as ExecutionL2Level,
    marketable_price,
)
from causal_portfolio.execution.precision import floor_size, round_price as _round_price
from causal_portfolio.execution.types import make_cloid
from causal_portfolio.market_making.avellaneda_stoikov import (
    ASModelParams,
    finite_horizon_quotes,
)
from causal_portfolio.market_making.calibration import (
    MarkoutObservation,
    adverse_markout_bps,
    ewma_absolute_volatility,
    fit_markout_curve,
)
from causal_portfolio.market_making.hyperliquid import (
    HLMarketMakerAdapter,
    HLMarketMakerConfig,
    PostOnlyWouldCrossError,
)
from causal_portfolio.market_making.pin import PINFit, fit_pin, pin_posteriors
from causal_portfolio.market_making.recorder import normalize_hl_message
from causal_portfolio.market_making.toxicity import (
    ToxicityConfig,
    ToxicitySignal,
    toxicity_adjustment,
)
from causal_portfolio.market_making.types import (
    AggressorSide,
    BookSnapshot,
    Quote,
    QuotePair,
    QuotePolicy,
    TradePrint,
)


@dataclass(frozen=True)
class TestnetLeg:
    arm: str
    side: str
    reference_mid: float
    requested_size: float
    filled_size: float
    maker_filled_size: float
    fallback_filled_size: float
    average_price: float | None
    fee_usd: float
    implementation_shortfall_bps: float | None
    elapsed_seconds: float
    status: str


@dataclass(frozen=True)
class TestnetExecutionBenchmark:
    coin: str
    size: float
    capture_seconds: float
    passive_timeout_seconds: float
    pin_proxy: float
    pin_fit_usable: bool
    pin_sample_buckets: int
    informed_buy_probability: float
    informed_sell_probability: float
    sigma_per_sqrt_second: float
    kappa: float
    buy_adverse_markout_bps: float
    sell_adverse_markout_bps: float
    naive_buy: TestnetLeg
    naive_sell: TestnetLeg
    as_pin_buy: TestnetLeg
    as_pin_sell: TestnetLeg
    naive_roundtrip_pnl_usd: float
    as_pin_roundtrip_pnl_usd: float
    initial_btc_position: float
    final_btc_position: float
    restored_initial_position: bool
    methodology_quality: str = "testnet_bucket_pin_bootstrap"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_testnet_events(duration_seconds: float) -> list[BookSnapshot | TradePrint]:
    from hyperliquid.info import Info
    from causal_portfolio.execution.hyperliquid import TESTNET_URL

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    info = Info(TESTNET_URL, skip_ws=False)
    messages: list[tuple[int, dict[str, Any]]] = []
    lock = threading.Lock()

    def callback(message: dict[str, Any]) -> None:
        with lock:
            messages.append((int(time.time() * 1_000), message))

    subscriptions = []
    try:
        for channel in ("trades", "l2Book"):
            subscription = {"type": channel, "coin": "BTC"}
            subscriptions.append((subscription, info.subscribe(subscription, callback)))
        time.sleep(duration_seconds)
    finally:
        for subscription, subscription_id in reversed(subscriptions):
            try:
                info.unsubscribe(subscription, subscription_id)
            except Exception:
                pass
        info.disconnect_websocket()
    events: list[BookSnapshot | TradePrint] = []
    for received_ms, message in messages:
        events.extend(normalize_hl_message(message, received_ms))
    return sorted(events, key=lambda event: event.timestamp_ms)


def bootstrap_pin_signal(
    events: list[BookSnapshot | TradePrint], *, bucket_seconds: int = 5
) -> tuple[PINFit, float, float, Any, float, int]:
    trades = [event for event in events if isinstance(event, TradePrint)]
    books = [event for event in events if isinstance(event, BookSnapshot) and event.mid]
    if not trades or len(books) < 3:
        raise ValueError("capture has insufficient trades or L2 snapshots")
    bucket_ms = bucket_seconds * 1_000
    first = min(trade.timestamp_ms for trade in trades)
    last = max(trade.timestamp_ms for trade in trades)
    bucket_count = int((last - first) // bucket_ms) + 1
    if bucket_count < 20:
        raise ValueError(
            f"capture spans only {bucket_count} PIN buckets; at least 20 are required"
        )
    buys = np.zeros(bucket_count, dtype=int)
    sells = np.zeros(bucket_count, dtype=int)
    for trade in trades:
        index = min((trade.timestamp_ms - first) // bucket_ms, bucket_count - 1)
        if trade.aggressor == AggressorSide.BUY:
            buys[index] += 1
        else:
            sells[index] += 1
    fit = fit_pin(buys, sells, min_days=20, n_starts=8, seed=29)
    posterior = pin_posteriors([int(buys[-1])], [int(sells[-1])], fit)[0]
    mid_times = [book.timestamp_ms for book in books]
    markouts: list[MarkoutObservation] = []
    for trade in trades:
        index = bisect.bisect_left(mid_times, trade.timestamp_ms + 5_000)
        if index >= len(books):
            continue
        is_maker_buy = trade.aggressor == AggressorSide.SELL
        markouts.append(
            MarkoutObservation(
                pin=fit.pin,
                aggressor=trade.aggressor,
                adverse_bps=adverse_markout_bps(
                    is_maker_buy=is_maker_buy,
                    fill_price=trade.price,
                    future_mid=books[index].mid,
                ),
            )
        )
    curve = fit_markout_curve(
        markouts,
        edges=(0.0, 1.0),
        prior_bps=0.0,
        prior_weight=5.0,
    )
    sigma = ewma_absolute_volatility(
        [book.mid for book in books],
        [book.timestamp_ms for book in books],
        half_life_seconds=60,
    )
    return (
        fit,
        posterior.informed_buy,
        posterior.informed_sell,
        curve,
        sigma,
        bucket_count,
    )


class FatalBenchmarkError(RuntimeError):
    """Unsafe-to-continue condition: dirty account state or failed cleanup.

    The campaign runner stops on this type instead of retrying; everything
    else (thin capture, unusable PIN, partial fill) is recorded and skipped.
    """


def _cloid(label: str) -> str:
    return make_cloid(f"{label}|{time.time_ns()}")


def _post_only_price(
    raw_price: float,
    book: BookSnapshot,
    *,
    is_buy: bool,
    sz_decimals: int,
    gap_usd: float = 5.0,
) -> float:
    """Round a quote price while preserving strict post-only placement.

    AS/PIN computes the economically desired quote from a captured book, but
    Hyperliquid validates against a fresh live book at submission time. This
    helper clamps the quote away from the live touch and then nudges it again
    if five-significant-figure rounding would otherwise cross the spread.
    """
    if book.best_bid is None or book.best_ask is None:
        raise ValueError("post-only quote requires a live two-sided book")
    if raw_price <= 0:
        raise ValueError("post-only quote price must be positive")

    gap = max(gap_usd, 1.0)
    tick_nudge = max(1.0, (book.mid or raw_price) * 1e-5)

    if is_buy:
        candidate = min(raw_price, book.best_ask - gap)
        for _ in range(16):
            price = _round_price(candidate, sz_decimals)
            if price > 0 and price < book.best_ask:
                return price
            candidate -= tick_nudge
        raise ValueError("could not derive a non-crossing post-only bid")

    candidate = max(raw_price, book.best_bid + gap)
    for _ in range(16):
        price = _round_price(candidate, sz_decimals)
        if price > book.best_bid:
            return price
        candidate += tick_nudge
    raise ValueError("could not derive a non-crossing post-only ask")


def _fills_for_oids(
    adapter: HLMarketMakerAdapter, oids: set[int], start_ms: int
) -> list[dict[str, Any]]:
    if not oids:
        return []
    fills = adapter.info.user_fills_by_time(adapter.address, start_ms)
    return [fill for fill in fills if int(fill.get("oid", -1)) in oids]


def _wait_for_ioc_fills(
    adapter: HLMarketMakerAdapter,
    oids: set[int],
    start_ms: int,
    parsed_filled_size: float,
    *,
    attempts: int = 8,
    poll_seconds: float = 0.75,
) -> list[dict[str, Any]]:
    """Return real fee-bearing fill records for an IOC response.

    Hyperliquid's order response can report a fill before
    `user_fills_by_time` indexes the corresponding fee-bearing fill record.
    For benchmark accounting we refuse to synthesize a `fee=0` fill: if the
    response says filled and fill details never appear, the run should be
    retried rather than reporting substituted shortfall/PnL.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(attempts):
        fills = _fills_for_oids(adapter, oids, start_ms)
        if fills:
            return fills
        if attempt < attempts - 1:
            time.sleep(poll_seconds)
    if parsed_filled_size > 0:
        raise RuntimeError(
            "IOC response reported a fill, but user_fills_by_time did not "
            "return fee-bearing fill details; refusing to synthesize fee=0 "
            "benchmark metrics. Rerun or increase fill polling."
        )
    return []


def _leg_from_fills(
    *,
    arm: str,
    is_buy: bool,
    reference_mid: float,
    requested_size: float,
    maker_fills: list[dict[str, Any]],
    fallback_fills: list[dict[str, Any]],
    elapsed_seconds: float,
) -> TestnetLeg:
    all_fills = maker_fills + fallback_fills
    filled_size = sum(float(fill["sz"]) for fill in all_fills)
    notional = sum(float(fill["sz"]) * float(fill["px"]) for fill in all_fills)
    fee = sum(float(fill.get("fee", 0.0)) for fill in all_fills)
    average = notional / filled_size if filled_size > 0 else None
    shortfall = None
    if average is not None:
        effective = (notional + fee) / filled_size if is_buy else (notional - fee) / filled_size
        direction = 1.0 if is_buy else -1.0
        shortfall = direction * (effective - reference_mid) / reference_mid * 10_000
    maker_size = sum(float(fill["sz"]) for fill in maker_fills)
    fallback_size = sum(float(fill["sz"]) for fill in fallback_fills)
    status = "filled" if filled_size >= requested_size - 1e-10 else "partial_or_unfilled"
    return TestnetLeg(
        arm=arm,
        side="buy" if is_buy else "sell",
        reference_mid=reference_mid,
        requested_size=requested_size,
        filled_size=filled_size,
        maker_filled_size=maker_size,
        fallback_filled_size=fallback_size,
        average_price=average,
        fee_usd=fee,
        implementation_shortfall_bps=shortfall,
        elapsed_seconds=elapsed_seconds,
        status=status,
    )


def _execute_ioc(
    adapter: HLMarketMakerAdapter,
    *,
    is_buy: bool,
    size: float,
    label: str,
    start_ms: int,
    reduce_only: bool = False,
    attempts: int = 8,
    poll_seconds: float = 1.5,
    max_band_bps: float = 100.0,
    require_fee_fills: bool = True,
) -> tuple[float, set[int], list[dict[str, Any]]]:
    sz_decimals = adapter._size_decimals("BTC")
    from hyperliquid.utils.types import Cloid

    last_reference = math.nan
    last_oids: set[int] = set()
    for attempt in range(attempts):
        book = adapter.fetch_l2_book("BTC", depth=10)
        reference = book.mid
        last_reference = reference
        execution_book = ExecutionL2Book(
            coin="BTC",
            bids=[ExecutionL2Level(level.px, level.size) for level in book.bids],
            asks=[ExecutionL2Level(level.px, level.size) for level in book.asks],
        )
        raw_price = marketable_price(
            execution_book,
            is_buy,
            size,
            max_band_bps=max_band_bps,
        )
        if raw_price is None:
            if attempt < attempts - 1:
                time.sleep(poll_seconds)
            continue

        price = _round_price(raw_price, sz_decimals)
        response = adapter._ensure_exchange().order(
            "BTC",
            is_buy,
            size,
            price,
            order_type={"limit": {"tif": "Ioc"}},
            reduce_only=reduce_only,
            cloid=Cloid.from_str(_cloid(f"{label}-{attempt}")),
        )
        parsed = parse_fill_response(response)
        oids: set[int] = set()
        statuses = response.get("response", {}).get("data", {}).get("statuses", [])
        for status in statuses:
            if status.get("filled", {}).get("oid") is not None:
                oids.add(int(status["filled"]["oid"]))
        last_oids = oids
        if parsed.filled_size > 0:
            if require_fee_fills:
                fills = _wait_for_ioc_fills(adapter, oids, start_ms, parsed.filled_size)
            else:
                fills = _fills_for_oids(adapter, oids, start_ms)
            return reference, oids, fills
        if attempt < attempts - 1:
            time.sleep(poll_seconds)
    return last_reference, last_oids, []


def _execute_naive_leg(
    adapter: HLMarketMakerAdapter,
    *,
    is_buy: bool,
    size: float,
    label: str,
    reduce_only: bool = False,
) -> TestnetLeg:
    started = time.monotonic()
    start_ms = int(time.time() * 1_000) - 1_000
    reference, _, fills = _execute_ioc(
        adapter,
        is_buy=is_buy,
        size=size,
        label=label,
        start_ms=start_ms,
        reduce_only=reduce_only,
    )
    return _leg_from_fills(
        arm="naive_ioc",
        is_buy=is_buy,
        reference_mid=reference,
        requested_size=size,
        maker_fills=[],
        fallback_fills=fills,
        elapsed_seconds=time.monotonic() - started,
    )


def _execute_as_pin_leg(
    adapter: HLMarketMakerAdapter,
    *,
    is_buy: bool,
    size: float,
    fit: PINFit,
    informed_buy: float,
    informed_sell: float,
    curve: Any,
    sigma: float,
    kappa: float,
    timeout_seconds: float,
    label: str,
    reduce_only_fallback: bool = False,
) -> TestnetLeg:
    started = time.monotonic()
    start_ms = int(time.time() * 1_000) - 1_000
    book = adapter.fetch_l2_book("BTC", depth=5)
    reference = book.mid
    params = ASModelParams(1e-4, max(sigma, 1e-6), kappa, timeout_seconds)
    levels = finite_horizon_quotes(reference, -1.0 if is_buy else 1.0, params)
    signal = ToxicitySignal(fit.pin, informed_buy, informed_sell, fit.usable)
    adjustment = toxicity_adjustment(
        signal,
        ToxicityConfig(policy=QuotePolicy.AS_PIN_POSTERIOR),
        markouts=curve,
    )
    sz_decimals = adapter._size_decimals("BTC")
    if is_buy:
        raw_price = min(
            levels.bid - reference * adjustment.bid_extra_bps / 10_000,
            book.best_ask - 1,
        )
    else:
        raw_price = max(
            levels.ask + reference * adjustment.ask_extra_bps / 10_000,
            book.best_bid + 1,
        )
    response: dict[str, Any] | None = None
    last_crossing_error: ValueError | None = None
    for submit_attempt in range(5):
        live_book = adapter.fetch_l2_book("BTC", depth=5)
        price = _post_only_price(
            raw_price,
            live_book,
            is_buy=is_buy,
            sz_decimals=sz_decimals,
            gap_usd=1.0 + submit_attempt * 2.0,
        )
        quote = Quote(is_buy, price, size, _cloid(f"{label}-{submit_attempt}"))
        pair = QuotePair(
            bid=quote if is_buy else None,
            ask=None if is_buy else quote,
            reservation_price=levels.reservation_price,
            model_spread=levels.total_spread,
            timestamp_ms=int(time.time() * 1_000),
            policy=QuotePolicy.AS_PIN_POSTERIOR,
            diagnostics={"coin": "BTC", "pin_proxy": fit.pin},
        )
        try:
            response = adapter.submit_quote_pair(pair)
            break
        except PostOnlyWouldCrossError as exc:
            if submit_attempt == 4:
                raise
            last_crossing_error = exc
            time.sleep(0.3)
    if response is None:
        if last_crossing_error is not None:
            raise last_crossing_error
        raise RuntimeError("post-only quote submission did not return a response")
    statuses = response.get("response", {}).get("data", {}).get("statuses", [])
    oids = {
        int(status["resting"]["oid"])
        for status in statuses
        if status.get("resting", {}).get("oid") is not None
    }
    deadline = time.monotonic() + timeout_seconds
    maker_fills: list[dict[str, Any]] = []
    if oids:
        while time.monotonic() < deadline:
            time.sleep(0.5)
            maker_fills = _fills_for_oids(adapter, oids, start_ms)
            if sum(float(fill["sz"]) for fill in maker_fills) >= size - 1e-10:
                break
    adapter.cancel_strategy_quotes()
    time.sleep(0.75)
    maker_fills = _fills_for_oids(adapter, oids, start_ms)
    maker_size = sum(float(fill["sz"]) for fill in maker_fills)
    remaining = floor_size(max(size - maker_size, 0.0), sz_decimals)
    fallback_fills: list[dict[str, Any]] = []
    if remaining >= 10**-sz_decimals:
        _, _, fallback_fills = _execute_ioc(
            adapter,
            is_buy=is_buy,
            size=remaining,
            label=f"{label}-fallback",
            start_ms=start_ms,
            reduce_only=reduce_only_fallback,
            attempts=24,
            poll_seconds=1.0,
        )
    return _leg_from_fills(
        arm="as_pin_post_only_then_ioc",
        is_buy=is_buy,
        reference_mid=reference,
        requested_size=size,
        maker_fills=maker_fills,
        fallback_fills=fallback_fills,
        elapsed_seconds=time.monotonic() - started,
    )


def _roundtrip_pnl(buy: TestnetLeg, sell: TestnetLeg) -> float:
    if buy.average_price is None or sell.average_price is None:
        return math.nan
    size = min(buy.filled_size, sell.filled_size)
    return size * (sell.average_price - buy.average_price) - buy.fee_usd - sell.fee_usd


def _require_filled(leg: TestnetLeg) -> None:
    if leg.status != "filled":
        raise RuntimeError(
            f"{leg.arm} {leg.side} did not fill: "
            f"{leg.filled_size}/{leg.requested_size}"
        )


def _restore_position(
    adapter: HLMarketMakerAdapter,
    initial_position: float,
    *,
    attempts: int = 24,
    poll_seconds: float = 5.0,
) -> dict[str, float]:
    decimals = adapter._size_decimals("BTC")
    min_size = 10**-decimals
    final = adapter.fetch_account_metrics("BTC")
    for attempt in range(attempts):
        delta = initial_position - final["inventory_base"]
        size = floor_size(abs(delta), decimals)
        if size < min_size:
            return final
        _execute_ioc(
            adapter,
            is_buy=delta > 0,
            size=size,
            label=f"benchmark-cleanup-{attempt}",
            start_ms=int(time.time() * 1_000) - 1_000,
            reduce_only=True,
            attempts=4,
            poll_seconds=1.0,
            max_band_bps=50.0,
            require_fee_fills=False,
        )
        time.sleep(poll_seconds if attempt < attempts - 1 else 0.75)
        final = adapter.fetch_account_metrics("BTC")
    return final


def run_testnet_execution_benchmark(
    *,
    size: float = 0.0002,
    capture_seconds: float = 120.0,
    passive_timeout_seconds: float = 15.0,
    acknowledge_testnet_writes: bool = False,
    state_path: str | Path = "tmp/trend-benchmark-testnet-cloids.json",
) -> TestnetExecutionBenchmark:
    if not acknowledge_testnet_writes:
        raise PermissionError("testnet benchmark requires acknowledge_testnet_writes=True")
    adapter = HLMarketMakerAdapter(
        HLMarketMakerConfig(testnet=True, dry_run=False, state_path=Path(state_path))
    )
    try:
        initial = adapter.fetch_account_metrics("BTC")
        if abs(initial["inventory_base"]) > 1e-10:
            raise FatalBenchmarkError(
                "testnet benchmark requires a flat initial BTC position"
            )
        if adapter.info.frontend_open_orders(adapter.address):
            raise FatalBenchmarkError(
                "testnet benchmark requires no pre-existing open orders"
            )

        events = capture_testnet_events(capture_seconds)
        fit, informed_buy, informed_sell, curve, sigma, buckets = bootstrap_pin_signal(events)
        book = adapter.fetch_l2_book("BTC", depth=2)
        kappa = 2.0 / max(book.best_ask - book.best_bid, 1.0)

        try:
            naive_buy = _execute_naive_leg(adapter, is_buy=True, size=size, label="naive-buy")
            _require_filled(naive_buy)
            naive_sell = _execute_naive_leg(
                adapter,
                is_buy=False,
                size=size,
                label="naive-sell",
                reduce_only=True,
            )
            _require_filled(naive_sell)
            as_buy = _execute_as_pin_leg(
                adapter,
                is_buy=True,
                size=size,
                fit=fit,
                informed_buy=informed_buy,
                informed_sell=informed_sell,
                curve=curve,
                sigma=sigma,
                kappa=kappa,
                timeout_seconds=passive_timeout_seconds,
                label="as-pin-buy",
            )
            _require_filled(as_buy)
            as_sell = _execute_as_pin_leg(
                adapter,
                is_buy=False,
                size=size,
                fit=fit,
                informed_buy=informed_buy,
                informed_sell=informed_sell,
                curve=curve,
                sigma=sigma,
                kappa=kappa,
                timeout_seconds=passive_timeout_seconds,
                label="as-pin-sell",
                reduce_only_fallback=True,
            )
            _require_filled(as_sell)
        finally:
            cleanup_error: Exception | None = None
            try:
                adapter.cancel_strategy_quotes()
                final = _restore_position(adapter, initial["inventory_base"])
                if abs(final["inventory_base"] - initial["inventory_base"]) > 1e-10:
                    cleanup_error = FatalBenchmarkError(
                        "cleanup failed: final BTC position "
                        f"{final['inventory_base']} != initial {initial['inventory_base']}"
                    )
            except Exception as exc:
                cleanup_error = FatalBenchmarkError(f"cleanup failed: {exc}")
            if cleanup_error is not None:
                raise cleanup_error
    finally:
        adapter.close()
    restored = abs(final["inventory_base"] - initial["inventory_base"]) <= 1e-10
    return TestnetExecutionBenchmark(
        coin="BTC",
        size=size,
        capture_seconds=capture_seconds,
        passive_timeout_seconds=passive_timeout_seconds,
        pin_proxy=fit.pin,
        pin_fit_usable=fit.usable,
        pin_sample_buckets=buckets,
        informed_buy_probability=informed_buy,
        informed_sell_probability=informed_sell,
        sigma_per_sqrt_second=sigma,
        kappa=kappa,
        buy_adverse_markout_bps=curve.buy_bps[0],
        sell_adverse_markout_bps=curve.sell_bps[0],
        naive_buy=naive_buy,
        naive_sell=naive_sell,
        as_pin_buy=as_buy,
        as_pin_sell=as_sell,
        naive_roundtrip_pnl_usd=_roundtrip_pnl(naive_buy, naive_sell),
        as_pin_roundtrip_pnl_usd=_roundtrip_pnl(as_buy, as_sell),
        initial_btc_position=initial["inventory_base"],
        final_btc_position=final["inventory_base"],
        restored_initial_position=restored,
    )


def _campaign_stats(attempts: list[dict[str, Any]]) -> dict[str, int]:
    """Success/maker-coverage counts used by both the stop rule and the summary."""
    successful = [
        item for item in attempts
        if item.get("ok") and item.get("restored_initial_position")
    ]

    def maker_attempts(leg: str) -> int:
        return sum(
            1 for item in successful
            if item.get(leg, {}).get("maker_filled_size", 0.0) > 0
        )

    return {
        "successful_restores": len(successful),
        "pin_usable_attempts": sum(1 for item in successful if item.get("pin_fit_usable")),
        "maker_buy_attempts": maker_attempts("as_pin_buy"),
        "maker_sell_attempts": maker_attempts("as_pin_sell"),
    }


def run_testnet_execution_campaign(
    *,
    size: float,
    max_seconds: float = 3 * 60 * 60,
    capture_seconds: float = 900.0,
    passive_timeout_seconds: float = 20.0,
    min_successful_attempts: int = 3,
    max_attempts: int | None = None,
    acknowledge_testnet_writes: bool = False,
    output_dir: str | Path = "tmp",
    state_path_prefix: str = "full-execution-testnet-cloids",
) -> dict[str, Any]:
    """Run repeated AS+PIN testnet attempts, recording failures instead of aborting.

    A single capture may legitimately fail diagnostics, or a tiny leg may
    partially fill. Those are useful observations during a multi-hour testnet
    campaign, not reasons to discard the whole run. Each attempt still relies
    on `run_testnet_execution_benchmark` for cleanup.
    """
    if not acknowledge_testnet_writes:
        raise PermissionError("testnet campaign requires acknowledge_testnet_writes=True")
    if size <= 0:
        raise ValueError("size must be > 0")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be > 0")
    if capture_seconds <= 0:
        raise ValueError("capture_seconds must be > 0")
    if passive_timeout_seconds <= 0:
        raise ValueError("passive_timeout_seconds must be > 0")
    if min_successful_attempts < 1:
        raise ValueError("min_successful_attempts must be >= 1")
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("max_attempts must be >= 1 when set")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max_seconds
    attempts: list[dict[str, Any]] = []
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        started = datetime.now(timezone.utc)
        try:
            result = run_testnet_execution_benchmark(
                size=size,
                capture_seconds=capture_seconds,
                passive_timeout_seconds=passive_timeout_seconds,
                acknowledge_testnet_writes=True,
                state_path=output_path / f"{state_path_prefix}-{attempt}.json",
            )
            record = result.to_dict()
            record.update({
                "attempt": attempt,
                "ok": True,
                "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            fatal = isinstance(exc, FatalBenchmarkError)
            record = {
                "attempt": attempt,
                "ok": False,
                "fatal": fatal,
                "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        attempts.append(record)
        (output_path / f"full_execution_testnet_result_attempt_{attempt:02d}.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

        stats = _campaign_stats(attempts)
        if (
            stats["successful_restores"] >= min_successful_attempts
            and stats["pin_usable_attempts"] >= min_successful_attempts
            and stats["maker_buy_attempts"] > 0
            and stats["maker_sell_attempts"] > 0
        ):
            break
        if record.get("fatal"):
            break
        if max_attempts is not None and attempt >= max_attempts:
            break

    summary = {
        "attempts": len(attempts),
        **_campaign_stats(attempts),
        "fatal_errors": sum(1 for item in attempts if item.get("fatal")),
        "results": attempts,
    }
    (output_path / "full_execution_testnet_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
