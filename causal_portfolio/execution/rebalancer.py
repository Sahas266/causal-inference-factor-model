"""Pure functions: (target_weights, account state, mids, meta, config) → RebalancePlan.

No network calls, no SDK imports. Fully unit-testable. The HL adapter feeds
this function the inputs it pulls from the API and submits the resulting orders.

Algorithm:
  1. Map CPCM tickers → HL coins; drop unlisted with SkipReason.NOT_LISTED
  2. Apply max_position_pct cap to each weight (preserving sign)
  3. Re-normalize so |w| sums to ≤ 1.0 (or less if caps clipped)
  4. target_usd[c] = equity * leverage * w[c]
  5. delta_usd[c] = target_usd[c] - current_usd[c]
  6. Filter: no-trade band and single-trade cap (full closes remain exempt)
  7. Convert USD → signed size via mids; round to sz_decimals
  8. Drop coins where rounded size = 0
  9. Build IOC limit orders with deterministic cloid
"""

from __future__ import annotations

import logging
import time

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.precision import (
    round_price as _round_price,
    round_size as _round_size,
)
from causal_portfolio.execution.types import (
    AccountState,
    AssetMeta,
    Order,
    RebalancePlan,
    SkipReason,
    TargetSnapshot,
    make_cloid,
)

logger = logging.getLogger("cpcm.execution.rebalancer")


def _make_cloid(timestamp_ms: int, coin: str, is_buy: bool, size: float) -> str:
    """Deterministic cloid from the immutable parts of a plan order."""
    return make_cloid(f"{timestamp_ms}:{coin}:{is_buy}:{size:.10f}")


def _apply_position_caps(
    weights: dict[str, float], cap_pct: float
) -> tuple[dict[str, float], list[str]]:
    """Clip each |weight| to cap_pct. Returns (clipped_weights, capped_coins)."""
    capped = {}
    hit_cap = []
    for coin, w in weights.items():
        magnitude = min(abs(w), cap_pct)
        if magnitude < abs(w):
            hit_cap.append(coin)
        capped[coin] = magnitude * (1.0 if w >= 0 else -1.0)
    return capped, hit_cap


def _is_same_side_reduce(current_notional: float, target_notional: float) -> bool:
    """True when moving from current to target only lowers existing risk."""
    return (
        current_notional != 0
        and abs(target_notional) < abs(current_notional)
        and (
            target_notional == 0
            or (current_notional > 0) == (target_notional > 0)
        )
    )


def plan_rebalance(
    target_weights: dict[str, float] | TargetSnapshot,
    state: AccountState,
    mids: dict[str, float],
    meta: dict[str, AssetMeta],
    config: ExecutionConfig,
    timestamp_ms: int | None = None,
) -> RebalancePlan:
    """Compute the orders needed to move from `state` to `target_weights`.

    Args:
        target_weights: CPCM ticker (lowercase) → signed weight in [-1, 1],
            or a TargetSnapshot retaining model provenance. Weights need not
            sum to 1; their absolute sum is the target gross exposure.
        state: current account snapshot.
        mids: HL coin name → mid price (USD).
        meta: HL coin name → AssetMeta (sz_decimals, max_leverage, min_size).
        config: execution config (caps, slippage, leverage, etc.).
        timestamp_ms: override for deterministic cloid generation in tests.

    Returns:
        RebalancePlan — orders + skipped reasons + audit-friendly intermediate
        state. Submit snapshots through the public `execute_target()`; use
        `hyperliquid.execute_plan()` only for an already prepared plan.
    """
    ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    notes: list[str] = []
    skipped: list[tuple[str, SkipReason, str]] = []
    target_snapshot = target_weights if isinstance(target_weights, TargetSnapshot) else None
    weights = target_snapshot.weights if target_snapshot is not None else target_weights
    if target_snapshot is not None:
        as_of = target_snapshot.as_of.isoformat() if target_snapshot.as_of else "missing"
        notes.append(f"target {target_snapshot.target_id} as_of={as_of}")

    # ── 1. Map tickers, drop unlisted ────────────────────────────────
    mapped: dict[str, float] = {}
    for ticker, w in weights.items():
        hl_coin = config.asset_map.get(ticker.lower())
        if hl_coin is None or hl_coin not in meta:
            skipped.append((
                ticker, SkipReason.NOT_LISTED,
                f"no HL listing for ticker '{ticker}'",
            ))
            continue
        mapped[hl_coin] = w

    # ── 2. Long-only enforcement ─────────────────────────────────────
    if config.long_only:
        zeroed = []
        for coin, w in list(mapped.items()):
            if w < 0:
                zeroed.append(coin)
                skipped.append((
                    coin, SkipReason.LONG_ONLY_VIOLATION,
                    f"weight {w:+.4f} dropped under long_only=True",
                ))
                mapped[coin] = 0.0
        if zeroed:
            notes.append(f"long_only zeroed: {zeroed}")

    # ── 3. Position-size caps ────────────────────────────────────────
    mapped, hit_cap = _apply_position_caps(mapped, config.max_position_pct)
    if hit_cap:
        notes.append(f"position cap ({config.max_position_pct}) clipped: {hit_cap}")

    # ── 3b. Gross-exposure cap: scale so Σ|w| ≤ 1.0 ─────────────────
    gross = sum(abs(w) for w in mapped.values())
    if gross > 1.0:
        scale = 1.0 / gross
        mapped = {coin: w * scale for coin, w in mapped.items()}
        notes.append(f"gross exposure {gross:.4f} > 1.0; scaled all weights by {scale:.4f}")

    # ── 4. Compute target USD per coin ──────────────────────────────
    equity_used = state.account_value_usd * config.leverage
    target_usd = {coin: equity_used * w for coin, w in mapped.items()}

    # Coins we currently hold but aren't in target weights → close them
    for coin, pos in state.positions.items():
        if coin not in target_usd and abs(pos.notional_usd) > 0:
            target_usd[coin] = 0.0
            notes.append(f"closing {coin}: held but not in target weights")

    # ── 5. Deltas ────────────────────────────────────────────────────
    current_usd = {c: p.notional_usd for c, p in state.positions.items()}

    # Hyperliquid exposes a max leverage per market. Pre-filter orders that
    # would increase risk beyond that limit; allow same-side reductions even
    # when the current position is already above the limit, because those are
    # risk-reducing reduce-only orders.
    if state.account_value_usd > 0:
        for coin, target_notional in list(target_usd.items()):
            asset_meta = meta.get(coin)
            if asset_meta is None:
                continue
            max_abs_notional = state.account_value_usd * asset_meta.max_leverage
            if abs(target_notional) <= max_abs_notional:
                continue
            current_notional = current_usd.get(coin, 0.0)
            same_side_reduce = _is_same_side_reduce(current_notional, target_notional)
            if same_side_reduce:
                notes.append(
                    f"{coin} target remains above {asset_meta.max_leverage}x "
                    "asset leverage limit but order reduces existing exposure"
                )
                continue
            skipped.append((
                coin,
                SkipReason.EXCEEDS_LEVERAGE_LIMIT,
                (
                    f"|target|=${abs(target_notional):,.2f} exceeds "
                    f"{asset_meta.max_leverage}x asset cap "
                    f"${max_abs_notional:,.2f}"
                ),
            ))
            target_usd[coin] = current_notional
    deltas_usd = {
        coin: target_usd.get(coin, 0.0) - current_usd.get(coin, 0.0)
        for coin in set(target_usd) | set(current_usd)
    }

    # ── 6. Build orders ──────────────────────────────────────────────
    orders: list[Order] = []
    single_trade_cap_usd = config.max_single_trade_pct * state.account_value_usd
    if config.max_single_trade_usd is not None:
        single_trade_cap_usd = min(single_trade_cap_usd, config.max_single_trade_usd)

    for coin, delta in sorted(deltas_usd.items()):
        if abs(delta) == 0:
            continue

        current_notional = current_usd.get(coin, 0.0)
        target_notional = target_usd.get(coin, 0.0)
        # A fraction of the existing position is scale-aware and directly
        # targets resize churn; opens, flips, and closes are directional calls.
        if (
            current_notional * target_notional > 0
            and abs(delta) / abs(current_notional) < config.min_position_change_pct
        ):
            skipped.append((
                coin,
                SkipReason.BELOW_NO_TRADE_BAND,
                (
                    f"|delta|/|current|={abs(delta) / abs(current_notional):.2%} "
                    f"below no-trade band {config.min_position_change_pct:.2%}"
                ),
            ))
            continue

        if abs(delta) > single_trade_cap_usd:
            # Don't silently shrink — surface to operator. Skip this coin.
            skipped.append((
                coin, SkipReason.EXCEEDS_TRADE_CAP,
                f"|delta|=${abs(delta):,.2f} > single-trade cap ${single_trade_cap_usd:,.2f}",
            ))
            continue

        mid = mids.get(coin)
        if mid is None or mid <= 0:
            skipped.append((coin, SkipReason.NOT_LISTED, f"no mid price for {coin}"))
            continue
        if coin not in meta:
            skipped.append((coin, SkipReason.NOT_LISTED, f"no asset metadata for {coin}"))
            continue

        is_buy = delta > 0
        raw_size = abs(delta) / mid
        m = meta[coin]
        size = _round_size(raw_size, m.sz_decimals)

        if size == 0 or size < m.min_size:
            skipped.append((
                coin, SkipReason.BELOW_MIN_SIZE,
                f"raw_size={raw_size:.6f} rounded to {size} at {m.sz_decimals} decimals",
            ))
            continue

        rounded_notional = size * mid
        # reduce_only=True when the order makes the absolute position SMALLER
        # without flipping sign (target == 0, a full close, always qualifies).
        # HL allows wider price bands for reduce-only orders — they can't
        # accidentally open new exposure.
        is_reduce_only = _is_same_side_reduce(current_notional, target_notional)
        is_full_reduce_close = is_reduce_only and target_notional == 0

        if (
            rounded_notional < config.min_order_notional_usd
            and not is_full_reduce_close
        ):
            skipped.append((
                coin,
                SkipReason.BELOW_MIN_NOTIONAL,
                (
                    f"rounded notional=${rounded_notional:.2f} below minimum "
                    f"${config.min_order_notional_usd:.2f}"
                ),
            ))
            continue

        # Precision-loss warning: if rounding shrank the trade noticeably
        # (>10% off the intended notional), surface it. Trade still goes out,
        # but the operator should know precision is tight on this asset.
        intended_notional = abs(delta)
        shrink_pct = (intended_notional - rounded_notional) / intended_notional
        if shrink_pct > 0.10:
            notes.append(
                f"precision warning {coin}: rounded ${rounded_notional:.2f} "
                f"is {shrink_pct:.1%} below target ${intended_notional:.2f} "
                f"(sz_decimals={m.sz_decimals})"
            )

        raw_limit_px = mid * config.slippage_factor(is_buy)
        limit_px = _round_price(raw_limit_px, m.sz_decimals)
        cloid = _make_cloid(ts, coin, is_buy, size)

        orders.append(Order(
            coin=coin, is_buy=is_buy, size=size,
            limit_px=limit_px, cloid=cloid, reduce_only=is_reduce_only,
        ))

    plan = RebalancePlan(
        timestamp_ms=ts,
        target_weights=dict(weights),
        current_state=state,
        target_usd=target_usd,
        deltas_usd=deltas_usd,
        orders=orders,
        skipped=skipped,
        equity_used=equity_used,
        notes=notes,
        network="testnet" if config.testnet else "mainnet",
        target_snapshot=target_snapshot,
        mids=dict(mids),
    )
    logger.info(plan.summary())
    return plan
