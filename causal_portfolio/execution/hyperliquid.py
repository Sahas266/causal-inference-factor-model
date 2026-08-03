"""Thin Hyperliquid SDK adapter.

Wraps the official `hyperliquid-python-sdk` to:
  - Read account state, mids, and asset metadata into the plain dataclasses
    that rebalancer.py expects (no SDK types leak past this boundary).
  - Submit a list of Order objects via bulk_orders.
  - Cancel open orders before placing a new batch (clean-slate rebalance).

Lazy-imports the SDK so the rest of the execution layer is testable without
it. Install with: `pip install hyperliquid-python-sdk~=0.24.0`.

Default base URL is testnet. Mainnet only when ExecutionConfig.testnet=False
AND ExecutionConfig.dry_run=False (caller's responsibility to gate).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, replace
from typing import Any

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.orderbook import (
    L2Book,
    L2Level,
    estimate_vwap,
    marketable_price,
    parse_fill_response,
)
from causal_portfolio.execution.precision import round_price as _round_price, round_size
from causal_portfolio.execution.rebalancer import _is_same_side_reduce, plan_rebalance
from causal_portfolio.execution.run_logging import execution_run_log
from causal_portfolio.execution.types import (
    AccountState,
    AssetMeta,
    Order,
    Position,
    RebalancePlan,
    SkipReason,
    SubmitResult,
    TargetSnapshot,
    make_cloid,
)

logger = logging.getLogger("cpcm.execution.hl")

MAINNET_URL = "https://api.hyperliquid.xyz"
TESTNET_URL = "https://api.hyperliquid-testnet.xyz"

# Accept either short HL_* names or the more descriptive HYPERLIQUID_* names
ADDRESS_ENV_VARS = ("HL_ADDRESS", "HYPERLIQUID_WALLET_ADDRESS")
KEY_ENV_VARS = ("HL_PRIVATE_KEY", "HYPERLIQUID_PRIVATE_KEY")


def _read_env_chain(names: tuple[str, ...]) -> str | None:
    """Return the first non-empty env var from `names`, or None."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    return None


def _load_dotenv_once() -> None:
    """Load the nearest .env above the working directory, when available."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    load_dotenv(find_dotenv(usecwd=True))


class HLAdapter:
    """Stateful network client. One instance per (address, network) pair."""

    def __init__(self, config: ExecutionConfig, address: str | None = None,
                 secret_key: str | None = None):
        # Lazy import — the rest of the execution layer doesn't need the SDK.
        try:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
        except ImportError as e:
            raise ImportError(
                "hyperliquid-python-sdk not installed. Run: "
                "pip install hyperliquid-python-sdk~=0.24.0 eth-account~=0.13.7"
            ) from e

        _load_dotenv_once()
        self.config = config
        self.address = address or _read_env_chain(ADDRESS_ENV_VARS)
        self.secret_key = secret_key or _read_env_chain(KEY_ENV_VARS)
        if not self.address:
            raise ValueError(
                "Wallet address required: set HL_ADDRESS or "
                "HYPERLIQUID_WALLET_ADDRESS env var, or pass address= directly"
            )

        self.base_url = TESTNET_URL if config.testnet else MAINNET_URL
        logger.info("HLAdapter on %s for %s", self.base_url, self.address[:8] + "...")

        self.info = Info(
            self.base_url,
            skip_ws=True,
            timeout=config.network_timeout_seconds,
        )

        # Exchange client only needs a wallet for write operations. Reads work
        # without one. We construct it lazily so dry-run / read-only flows
        # don't require a private key.
        self._exchange: Any = None
        self._Account = Account
        self._Exchange = Exchange
        self._meta_cache: dict[str, AssetMeta] | None = None

    def _ensure_exchange(self) -> Any:
        if self._exchange is None:
            if not self.secret_key:
                raise ValueError(
                    "Private key required for write operations: set "
                    "HL_PRIVATE_KEY or HYPERLIQUID_PRIVATE_KEY env var"
                )
            wallet = self._Account.from_key(self.secret_key)
            self._exchange = self._Exchange(
                wallet,
                self.base_url,
                account_address=self.address,
                timeout=self.config.network_timeout_seconds,
            )
        return self._exchange

    # ── reads ───────────────────────────────────────────────────────

    def fetch_state(self) -> AccountState:
        """Snapshot positions + equity from clearinghouseState."""
        raw = self.info.user_state(self.address)
        positions: dict[str, Position] = {}
        for ap in raw.get("assetPositions", []):
            p = ap.get("position", {})
            coin = p.get("coin")
            szi = float(p.get("szi", 0))
            if szi == 0 or coin is None:
                continue
            entry = float(p.get("entryPx", 0) or 0)
            notional = float(p.get("positionValue", 0))
            # `positionValue` from HL is unsigned magnitude. Sign it by szi.
            signed_notional = notional if szi > 0 else -notional
            positions[coin] = Position(
                coin=coin, size=szi, entry_px=entry,
                notional_usd=signed_notional,
            )
        margin_summary = raw.get("marginSummary", {})
        return AccountState(
            address=self.address,
            account_value_usd=float(margin_summary.get("accountValue", 0)),
            margin_used_usd=float(margin_summary.get("totalMarginUsed", 0)),
            positions=positions,
        )

    def fetch_mids(self) -> dict[str, float]:
        raw = self.info.all_mids()
        return {coin: float(px) for coin, px in raw.items()}

    def fetch_meta(self, refresh: bool = False) -> dict[str, AssetMeta]:
        """Pull per-coin metadata (size decimals, max leverage).

        Memoized — asset metadata is effectively static for the life of an
        adapter, and book-aware execution would otherwise re-fetch it on
        every IOC slice. Pass refresh=True to force a re-fetch.
        """
        if self._meta_cache is not None and not refresh:
            return self._meta_cache
        raw = self.info.meta()
        meta: dict[str, AssetMeta] = {}
        for asset_info in raw.get("universe", []):
            coin = asset_info.get("name")
            sz_dec = int(asset_info.get("szDecimals", 4))
            max_lev = int(asset_info.get("maxLeverage", 10))
            if coin is None:
                continue
            meta[coin] = AssetMeta(
                coin=coin, sz_decimals=sz_dec,
                max_leverage=max_lev, min_size=10 ** -sz_dec,
            )
        self._meta_cache = meta
        return meta

    def fetch_open_order_ids(self) -> list[int]:
        raw = self.info.open_orders(self.address)
        return [int(o["oid"]) for o in raw if "oid" in o]

    def fetch_l2_book(self, coin: str, depth: int = 10) -> L2Book:
        """Fetch the L2 order book for a coin into our plain L2Book type.

        HL `l2_snapshot` returns {'levels': [[bids], [asks]]} where each level
        is {'px','sz','n'}; bids are best-first (descending), asks best-first
        (ascending).
        """
        raw = self.info.l2_snapshot(coin)
        levels = raw.get("levels", [[], []])
        bids = [L2Level(float(l["px"]), float(l["sz"])) for l in levels[0][:depth]]
        asks = [L2Level(float(l["px"]), float(l["sz"])) for l in levels[1][:depth]]
        return L2Book(coin=coin, bids=bids, asks=asks)

    # ── writes ──────────────────────────────────────────────────────

    def cancel_all_open(self) -> dict[str, Any] | None:
        """Cancel every open order on this account. Best-effort."""
        # SDK signature: bulk_cancel(list of {coin, oid}) — needs coin per oid,
        # so query frontend_open_orders (which includes coin) in one call.
        raw = self.info.frontend_open_orders(self.address)
        requests = [{"coin": o["coin"], "oid": int(o["oid"])} for o in raw if "oid" in o]
        if not requests:
            return None
        ex = self._ensure_exchange()
        return ex.bulk_cancel(requests)

    def submit_orders(self, orders: list[Order]) -> dict[str, Any]:
        """Submit a batch of orders atomically via bulk_orders.

        The HL SDK expects `cloid` to be a `Cloid` instance, not a plain
        hex string. We wrap our deterministic-cloid strings here so the
        rest of the codebase can stay dependency-free of the SDK.
        """
        if not orders:
            return {"status": "ok", "response": {"data": {"statuses": []}}}
        from hyperliquid.utils.types import Cloid
        ex = self._ensure_exchange()
        order_args = []
        for o in orders:
            order_args.append({
                "coin": o.coin,
                "is_buy": o.is_buy,
                "sz": o.size,
                "limit_px": o.limit_px,
                "order_type": {"limit": {"tif": o.tif}},
                "reduce_only": o.reduce_only,
                "cloid": Cloid.from_str(o.cloid),
            })
        return ex.bulk_orders(order_args)

    def _submit_single_ioc(
        self, coin: str, is_buy: bool, size: float, limit_px: float,
        reduce_only: bool, cloid: str | None = None,
    ) -> dict[str, Any]:
        """Submit one IOC limit order. Returns the raw HL response."""
        ex = self._ensure_exchange()
        kwargs: dict[str, Any] = {}
        if cloid is not None:
            from hyperliquid.utils.types import Cloid
            kwargs["cloid"] = Cloid.from_str(cloid)
        return ex.order(
            coin, is_buy, size, limit_px,
            order_type={"limit": {"tif": "Ioc"}},
            reduce_only=reduce_only,
            **kwargs,
        )

    def submit_orders_book_aware(self, orders: list[Order]) -> dict[str, Any]:
        """Fill each order by chasing the live L2 book with IOC slices.

        For every order, repeatedly: fetch the current book, compute a
        marketable in-band price for the remaining size, submit an IOC, and
        accumulate fills. Retries up to `config.smart_max_attempts`, sleeping
        `config.smart_poll_seconds` between attempts so transient oracle/book
        divergence (where no crossable in-band price exists) can resolve.

        Returns a synthetic response dict summarizing per-coin fills, shaped
        loosely like the HL batch response so the audit log stays uniform.
        """
        cfg = self.config
        summaries: list[dict[str, Any]] = []
        for o in orders:
            remaining = o.size
            filled_total = 0.0
            fills: list[dict[str, Any]] = []
            last_error: str | None = None
            # sz_decimals for size/price rounding
            sz_dec = self._sz_decimals(o.coin)
            min_size = 10 ** -sz_dec

            for attempt in range(cfg.smart_max_attempts):
                if remaining < min_size:
                    break
                book = self.fetch_l2_book(o.coin)
                px = marketable_price(
                    book, o.is_buy, remaining,
                    max_band_bps=cfg.smart_max_band_bps,
                )
                if px is None:
                    last_error = "no crossable in-band price (book/oracle divergence)"
                    logger.warning(
                        "%s: no valid price attempt %d/%d; waiting %.1fs",
                        o.coin, attempt + 1, cfg.smart_max_attempts,
                        cfg.smart_poll_seconds,
                    )
                    time.sleep(cfg.smart_poll_seconds)
                    continue

                px = _round_price(px, sz_dec)
                size_r = self._round_size_down(remaining, sz_dec)
                if size_r < min_size:
                    break
                # Per-slice deterministic cloid derived from the plan order's
                # cloid + attempt index — a network-level retry of the same
                # slice gets the same cloid and HL rejects the duplicate.
                slice_cloid = make_cloid(f"{o.cloid}:{attempt}")
                resp = self._submit_single_ioc(
                    o.coin, o.is_buy, size_r, px, o.reduce_only,
                    cloid=slice_cloid,
                )
                fr = parse_fill_response(resp)
                if fr.filled_size > 0:
                    filled_total += fr.filled_size
                    remaining = max(0.0, remaining - fr.filled_size)
                    fills.append({"sz": fr.filled_size, "px": fr.avg_px,
                                  "attempt": attempt + 1})
                    last_error = None
                else:
                    last_error = fr.error
                    if fr.is_ambiguous:
                        # We couldn't tell whether the slice filled. Retrying
                        # blind could double-fill — abort this coin and let
                        # reconcile/post-state surface the drift.
                        logger.error(
                            "%s: ambiguous order response (%s) — aborting "
                            "retries to avoid a possible double-fill",
                            o.coin, fr.error,
                        )
                        break
                    # Oracle reject → too aggressive for current band; no-match
                    # → not aggressive enough. Either way a fresh book on the
                    # next attempt re-clamps. Brief pause to let it move.
                    time.sleep(cfg.smart_poll_seconds)

                if remaining < min_size:
                    break

            summaries.append({
                "coin": o.coin, "is_buy": o.is_buy,
                "requested": o.size, "filled": filled_total,
                "remaining": remaining, "fills": fills,
                "error": last_error if filled_total < o.size - min_size else None,
            })
        return {"status": "ok", "response": {"type": "book_aware",
                                             "data": {"summaries": summaries}}}

    def _sz_decimals(self, coin: str) -> int:
        meta = self.fetch_meta()
        return meta[coin].sz_decimals if coin in meta else 4

    @staticmethod
    def _round_size_down(size: float, sz_decimals: int) -> float:
        return round_size(size, sz_decimals) if size > 0 else 0.0


def _aggregate_transaction_cost(
    legs: list[dict[str, Any]], fee_bps: float
) -> dict[str, Any]:
    """Aggregate already-sampled per-leg costs without another book fetch."""
    planned_notional = sum(leg["planned_notional_usd"] for leg in legs)
    if planned_notional <= 0:
        raise ValueError("planned notional is zero")
    slippage_usd = sum(leg["slippage_usd"] for leg in legs)
    fee_usd = sum(leg["fee_usd"] for leg in legs)
    total_cost = slippage_usd + fee_usd
    return {
        "planned_notional_usd": planned_notional,
        "slippage_usd": slippage_usd,
        "fee_usd": fee_usd,
        "estimated_cost_usd": total_cost,
        "estimated_cost_bps": total_cost / planned_notional * 10_000.0,
        "estimated_taker_fee_bps": fee_bps,
        "legs": legs,
    }


def _estimate_transaction_cost(adapter: HLAdapter, orders: list[Order]) -> dict[str, Any]:
    """Estimate all-in taker fee and full-size L2 impact before submission."""
    mids = adapter.fetch_mids()
    fee_bps = adapter.config.estimated_taker_fee_bps
    legs: list[dict[str, Any]] = []
    for order in orders:
        leg = {
            "coin": order.coin,
            "side": "buy" if order.is_buy else "sell",
            "reduce_only": order.reduce_only,
            "size": order.size,
        }
        try:
            mid = mids.get(order.coin)
            if mid is None or mid <= 0:
                raise ValueError("mid price unavailable")
            book = adapter.fetch_l2_book(order.coin, depth=100)
            vwap = estimate_vwap(book, order.is_buy, order.size)
            if vwap is None:
                raise ValueError("insufficient or invalid L2 depth")
            notional = order.size * mid
            leg_slippage = order.size * abs(vwap - mid)
            leg_fee = order.size * vwap * fee_bps / 10_000.0
            leg.update({
                "mid": mid,
                "estimated_vwap": vwap,
                "planned_notional_usd": notional,
                "slippage_usd": leg_slippage,
                "fee_usd": leg_fee,
                "estimated_cost_bps": (
                    (leg_slippage + leg_fee) / notional * 10_000.0
                ),
            })
        except Exception as exc:
            leg["error"] = str(exc)
        legs.append(leg)

    estimated_legs = [leg for leg in legs if "error" not in leg]
    estimate = (
        _aggregate_transaction_cost(estimated_legs, fee_bps)
        if estimated_legs
        else {"estimated_taker_fee_bps": fee_bps, "legs": []}
    )
    estimate["legs"] = legs
    errors = [f"{leg['coin']}: {leg['error']}" for leg in legs if "error" in leg]
    if errors:
        estimate["error"] = "; ".join(errors)
    return estimate


def _check_cost_gate(
    adapter: HLAdapter, orders: list[Order]
) -> tuple[
    dict[str, Any] | None,
    str | None,
    list[Order],
    list[tuple[str, SkipReason, str]],
]:
    limit = adapter.config.max_transaction_cost_bps
    if limit is None:
        return None, None, orders, []

    # Reduce-only orders are fully exempt: failing closed on an exit preserves
    # the risk the order was meant to remove. Only exposure increases are gated.
    try:
        estimate = _estimate_transaction_cost(adapter, orders)
    except Exception as exc:
        logger.warning("transaction-cost estimate unavailable: %s", exc)
        failed_estimate = {"max_transaction_cost_bps": limit, "error": str(exc)}
        allowed = [order for order in orders if order.reduce_only]
        # If the shared mids snapshot fails, no increasing leg has a defensible
        # estimate, so each fails closed; reduce-only exits remain exempt.
        skipped = [
            (
                order.coin,
                SkipReason.COST_ESTIMATE_UNAVAILABLE,
                f"transaction-cost estimate unavailable: {exc}",
            )
            for order in orders
            if not order.reduce_only
        ]
        if skipped:
            failed_estimate.update({
                "reduce_only_exempt": bool(allowed),
                "submitted_scope": "reduce_only" if allowed else "blocked_orders",
                "dropped_legs": [
                    {"coin": coin, "skip_reason": reason.value, "error": detail}
                    for coin, reason, detail in skipped
                ],
            })
            return failed_estimate, "cost_estimate_unavailable", allowed, skipped
        failed_estimate.update({
            "reduce_only_exempt": True,
            "submitted_scope": "reduce_only",
        })
        return failed_estimate, None, orders, []

    allowed: list[Order] = []
    submitted_legs: list[dict[str, Any]] = []
    dropped_legs: list[dict[str, Any]] = []
    skipped: list[tuple[str, SkipReason, str]] = []
    unavailable = expensive = False
    for order, leg in zip(orders, estimate["legs"], strict=True):
        if order.reduce_only:
            leg["gate_status"] = "reduce_only_exempt"
            allowed.append(order)
            submitted_legs.append(leg)
        elif "error" in leg:
            unavailable = True
            leg["gate_status"] = "blocked_estimate_unavailable"
            leg["skip_reason"] = SkipReason.COST_ESTIMATE_UNAVAILABLE.value
            dropped_legs.append(leg)
            skipped.append((
                order.coin,
                SkipReason.COST_ESTIMATE_UNAVAILABLE,
                f"transaction-cost estimate unavailable: {leg['error']}",
            ))
        elif leg["estimated_cost_bps"] > limit:
            expensive = True
            leg["gate_status"] = "blocked_above_limit"
            leg["skip_reason"] = SkipReason.EXCEEDS_TRANSACTION_COST.value
            dropped_legs.append(leg)
            skipped.append((
                order.coin,
                SkipReason.EXCEEDS_TRANSACTION_COST,
                (
                    f"estimated cost {leg['estimated_cost_bps']:.2f} bps "
                    f"exceeds {limit:.2f} bps limit"
                ),
            ))
        else:
            leg["gate_status"] = "allowed"
            allowed.append(order)
            submitted_legs.append(leg)

    if not skipped:
        estimate.update({
            "max_transaction_cost_bps": limit,
            "reduce_only_exempt": any(order.reduce_only for order in orders),
            "aggregate_scope": "submitted_orders",
        })
        return estimate, None, orders, []

    report_legs = submitted_legs if allowed else dropped_legs
    estimated_report_legs = [leg for leg in report_legs if "error" not in leg]
    report = (
        _aggregate_transaction_cost(
            estimated_report_legs, estimate["estimated_taker_fee_bps"]
        )
        if estimated_report_legs
        else {"estimated_taker_fee_bps": estimate["estimated_taker_fee_bps"]}
    )
    report.update({
        "legs": report_legs,
        "evaluated_legs": estimate["legs"],
        "dropped_legs": dropped_legs,
        "max_transaction_cost_bps": limit,
        "reduce_only_exempt": any(order.reduce_only for order in allowed),
        "aggregate_scope": "submitted_orders" if allowed else "blocked_orders",
        "submitted_scope": (
            "reduce_only"
            if allowed and all(order.reduce_only for order in allowed)
            else "partial_plan" if allowed
            else "blocked_orders"
        ),
    })
    if unavailable:
        report["error"] = estimate["error"]
    if expensive:
        report["blocked_exposure_increasing_estimate"] = (
            _aggregate_transaction_cost(
                [leg for leg in dropped_legs if "error" not in leg],
                estimate["estimated_taker_fee_bps"],
            )
        )
    reason = (
        "per_leg_cost_gate"
        if unavailable and expensive
        else "cost_estimate_unavailable" if unavailable
        else "estimated_cost_above_limit"
    )
    return report, reason, allowed, skipped


def _plan_completeness(plan: RebalancePlan, orders: list[Order]) -> float:
    """Submitted share of intended exposure-increasing gross trade notional."""
    # Gross notional weights omissions honestly: one large dropped leg matters
    # more than several dust legs. Reduce-only intent is excluded because no
    # execution gate may block de-risking. Unmapped target skips are rebuilt
    # from raw weights; this conservatively understates completeness because
    # those weights were dropped before position caps and gross normalization.
    current = {
        coin.casefold(): position.notional_usd
        for coin, position in plan.current_state.positions.items()
    }
    targets = {coin.casefold(): value for coin, value in plan.target_usd.items()}
    intended = {}
    for coin, delta in plan.deltas_usd.items():
        key = coin.casefold()
        if delta and not _is_same_side_reduce(
            current.get(key, 0.0), targets.get(key, 0.0)
        ):
            intended[key] = abs(delta)
    target_weights = {
        ticker.casefold(): weight for ticker, weight in plan.target_weights.items()
    }
    for coin, _reason, _detail in plan.skipped:
        key = coin.casefold()
        if key not in intended and key in target_weights:
            target = target_weights[key] * plan.equity_used
            if not _is_same_side_reduce(current.get(key, 0.0), target):
                intended[key] = abs(target)
    total = sum(intended.values())
    if total <= 0:
        return 1.0
    submitted_coins = {
        order.coin.casefold() for order in orders if not order.reduce_only
    }
    submitted = sum(
        notional for coin, notional in intended.items() if coin in submitted_coins
    )
    return min(submitted / total, 1.0)


def _check_completeness_gate(
    plan: RebalancePlan,
    orders: list[Order],
    minimum: float,
) -> tuple[float, list[Order], str | None, list[tuple[str, SkipReason, str]]]:
    ratio = _plan_completeness(plan, orders)
    gateable = [order for order in orders if not order.reduce_only]
    if minimum <= 0 or ratio >= minimum or not gateable:
        return ratio, orders, None, []

    reason = f"plan completeness {ratio:.1%} below minimum {minimum:.1%}"
    # Execution safety invariant: gates may withhold exposure increases only;
    # reduce-only exits always survive to submission.
    allowed = [order for order in orders if order.reduce_only]
    skipped = [
        (order.coin, SkipReason.INSUFFICIENT_PLAN_COMPLETENESS, reason)
        for order in gateable
    ]
    return ratio, allowed, reason, skipped


def execute_target(
    target: TargetSnapshot,
    config: ExecutionConfig,
    *,
    acknowledge_mainnet: bool = False,
) -> SubmitResult:
    """Execute one provenance-carrying model target."""
    if not isinstance(target, TargetSnapshot):
        raise TypeError("target must be a TargetSnapshot")
    with execution_run_log(target.strategy or "model"):
        try:
            from causal_portfolio.execution import trace

            trace.start_cycle(target, asset_map=config.asset_map)
        except Exception:
            logger.exception("trace cycle initialization failed (continuing)")
        adapter = HLAdapter(config)
        state = adapter.fetch_state()
        mids = adapter.fetch_mids()
        plan = plan_rebalance(target, state, mids, adapter.fetch_meta(), config)
        return execute_plan(
            adapter,
            plan,
            acknowledge_mainnet=acknowledge_mainnet,
        )


def execute_plan(
    adapter: HLAdapter,
    plan: RebalancePlan,
    *,
    acknowledge_mainnet: bool = False,
    write_audit: bool = True,
) -> SubmitResult:
    """Execute a RebalancePlan. Honors dry_run flag in adapter.config.

    Safety gates (in addition to ExecutionConfig.dry_run):

    1. Address match: plan.current_state.address MUST equal adapter.address.
       Refuses to execute otherwise. Prevents cross-account misfires when
       multiple adapters/plans coexist in one process.

    2. Network binding: plans produced for testnet cannot be submitted through
       a mainnet adapter, or vice versa.

    3. Signal freshness: dated model targets must be recent unless the config
       explicitly opts into stale-signal execution.

    4. Mainnet acknowledgement: live mainnet writes require an explicit
       `acknowledge_mainnet=True` kwarg. The CLI sets it after a confirmation
       prompt; programmatic callers must opt in deliberately.

    5. Audit write: appending to the rotating JSONL log is attempted for every
       result unless `write_audit=False`; failures are returned in `audit_error`.
       Test code may disable this; production should leave the default enabled.

    6. Leg-failure fallback: post-trade reconciliation drift triggers up to
       `config.repair_attempts` re-plan + book-aware retry passes; unresolved
       drift and any error are pushed to Telegram (see execution/notify.py)
       when TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are configured.
    """
    target_id = (
        plan.target_snapshot.target_id
        if plan.target_snapshot is not None
        else "unversioned"
    )
    network = plan.network or ("testnet" if adapter.config.testnet else "mainnet")
    logger.info(
        "execution started: target_id=%s network=%s orders=%d dry_run=%s",
        target_id,
        network,
        len(plan.orders),
        adapter.config.dry_run,
    )
    if write_audit and plan.target_snapshot is not None:
        try:
            from causal_portfolio.execution import trace

            trace.start_cycle(
                plan.target_snapshot, asset_map=adapter.config.asset_map
            )
        except Exception:
            logger.exception("trace cycle initialization failed (continuing)")
    result = _execute_plan_inner(adapter, plan, acknowledge_mainnet=acknowledge_mainnet)
    audit_path = None
    if write_audit:
        try:
            from causal_portfolio.execution.audit import append as audit_append
            audit_path = audit_append(result)
        except Exception as e:
            logger.exception("audit append failed (continuing)")
            result = replace(result, audit_error=str(e))
    logger.info(
        "execution finished: submitted=%s error=%s post_submit_error=%s "
        "audit_error=%s target_id=%s audit=%s",
        result.submitted,
        result.error,
        result.post_submit_error,
        result.audit_error,
        target_id,
        audit_path,
    )
    try:
        from causal_portfolio.execution import trace

        trace.record_execution_result(result)
    except Exception:
        logger.exception("trace execution write failed (continuing)")
    try:
        from causal_portfolio.execution import notify
        notify.notify_result(result)
    except Exception:
        logger.exception("telegram notification failed (continuing)")
    return result


def _execute_plan_inner(
    adapter: HLAdapter, plan: RebalancePlan, *, acknowledge_mainnet: bool,
) -> SubmitResult:
    # Gate 1: address-of-plan must match adapter address
    if plan.current_state.address != adapter.address:
        return SubmitResult(
            plan=plan, submitted=False,
            error=(f"address mismatch: plan built for {plan.current_state.address!r} "
                   f"but adapter writes from {adapter.address!r}"),
            submitted_orders=[],
        )

    adapter_network = "testnet" if adapter.config.testnet else "mainnet"
    if plan.network is not None and plan.network != adapter_network:
        return SubmitResult(
            plan=plan,
            submitted=False,
            error=(f"network mismatch: plan built for {plan.network!r} "
                   f"but adapter is on {adapter_network!r}"),
            submitted_orders=[],
        )

    if adapter.config.dry_run:
        logger.info("DRY RUN — not submitting %d orders", len(plan.orders))
        return SubmitResult(
            plan=plan,
            submitted=False,
            submitted_orders=[],
            completeness_ratio=_plan_completeness(plan, plan.orders),
        )

    if not plan.orders:
        logger.info("no orders to submit")
        return SubmitResult(
            plan=plan,
            submitted=False,
            submitted_orders=[],
            completeness_ratio=_plan_completeness(plan, []),
        )

    if not adapter.config.allow_stale_signal:
        if plan.target_snapshot is None:
            freshness_error = (
                "target is missing provenance; live execution requires a "
                "TargetSnapshot loaded from JSON/CSV"
            )
        else:
            freshness_error = plan.target_snapshot.freshness_error(
                adapter.config.max_signal_age_hours
            )
        if freshness_error is not None:
            target_id = (
                plan.target_snapshot.target_id
                if plan.target_snapshot is not None
                else "unversioned"
            )
            return SubmitResult(
                plan=plan,
                submitted=False,
                error=(f"signal freshness check failed for target "
                       f"{target_id}: {freshness_error}. "
                       f"Set allow_stale_signal=True only for an intentional override."),
                submitted_orders=[],
            )

    # Gate 4: mainnet writes need explicit acknowledgement
    if adapter.config.is_live_mainnet() and acknowledge_mainnet is not True:
        return SubmitResult(
            plan=plan, submitted=False,
            error=("LIVE MAINNET write blocked: acknowledge_mainnet must be the "
                   "literal boolean True."),
            submitted_orders=[],
        )

    if adapter.config.is_live_mainnet():
        logger.warning("LIVE MAINNET write: %d orders, $%.0f gross",
                       len(plan.orders),
                       sum(abs(d) for d in plan.deltas_usd.values()))

    cost_estimate, cost_gate_reason, orders_to_submit, cost_skips = _check_cost_gate(
        adapter, plan.orders
    )
    if cost_skips:
        plan = replace(plan, skipped=[*plan.skipped, *cost_skips])
    (
        completeness_ratio,
        orders_to_submit,
        completeness_gate_reason,
        completeness_skips,
    ) = _check_completeness_gate(
        plan,
        orders_to_submit,
        adapter.config.min_rebalance_completeness,
    )
    if completeness_skips:
        plan = replace(plan, skipped=[*plan.skipped, *completeness_skips])
    if not orders_to_submit:
        return SubmitResult(
            plan=plan,
            submitted=False,
            error=completeness_gate_reason,
            cost_estimate=cost_estimate,
            cost_gate_reason=cost_gate_reason,
            submitted_orders=[],
            completeness_ratio=completeness_ratio,
        )

    submission_error = None
    submission_started = False
    mid_state = plan.current_state
    post = None
    try:
        adapter.cancel_all_open()

        # Cancel-vs-submit race guard: a stale order could fill between the
        # cancel call and our submit. Re-fetch state and compare to the state
        # the plan was built from. If positions moved materially, bail out
        # so the caller can re-plan against fresh state.
        mid_state = adapter.fetch_state()
        if _detect_cancel_race(plan, mid_state):
            return SubmitResult(
                plan=plan, submitted=False, post_state=mid_state,
                error="cancel-race detected: positions moved between cancel and submit",
                cost_estimate=cost_estimate,
                cost_gate_reason=cost_gate_reason,
                submitted_orders=[],
                completeness_ratio=completeness_ratio,
            )

        submission_started = True
        if adapter.config.twap_minutes > 0:
            response = _submit_orders_twap(adapter, orders_to_submit)
        elif adapter.config.smart_execution:
            response = adapter.submit_orders_book_aware(orders_to_submit)
        else:
            response = adapter.submit_orders(orders_to_submit)
    except Exception as e:
        logger.exception("submission failed before completion")
        # Best-effort post-state snapshot: orders may have partially gone out
        # before the failure, and the audit log should capture where we landed.
        try:
            post = adapter.fetch_state()
        except Exception:
            logger.warning("post-failure state fetch also failed")
        if (
            not submission_started
            or post is None
            or not _position_sizes_changed(plan, mid_state, post)
        ):
            return SubmitResult(
                plan=plan,
                submitted=False,
                post_state=post,
                error=str(e),
                cost_estimate=cost_estimate,
                cost_gate_reason=cost_gate_reason,
                submitted_orders=[],
                completeness_ratio=completeness_ratio,
            )
        logger.warning("submission response failed after positions changed; reconciling")
        response = None
        submission_error = str(e)

    try:
        if post is None:
            post = adapter.fetch_state()
        from causal_portfolio.execution.reconcile import format_drift_summary, reconcile
        drifts = reconcile(
            plan,
            post,
            tolerance_usd=adapter.config.min_order_notional_usd,
        )
        logger.info(format_drift_summary(drifts))
        repair = None
        if drifts and adapter.config.repair_attempts > 0:
            repair, post, drifts = _repair_failed_legs(adapter, plan, post, drifts)
            logger.info("post-repair: " + format_drift_summary(drifts))
        return SubmitResult(plan=plan, submitted=True,
                            response=response, post_state=post, drifts=drifts,
                            post_submit_error=submission_error, repair=repair,
                            cost_estimate=cost_estimate,
                            cost_gate_reason=cost_gate_reason,
                            submitted_orders=list(orders_to_submit),
                            completeness_ratio=completeness_ratio)
    except Exception as e:
        logger.exception("post-submit state/reconciliation failed")
        post_submit_error = str(e)
        if submission_error:
            post_submit_error = f"{submission_error}; post-submit checks failed: {e}"
        return SubmitResult(
            plan=plan,
            submitted=True,
            response=response,
            post_state=post,
            post_submit_error=post_submit_error,
            cost_estimate=cost_estimate,
            cost_gate_reason=cost_gate_reason,
            submitted_orders=list(orders_to_submit),
            completeness_ratio=completeness_ratio,
        )


def _repair_failed_legs(
    adapter: HLAdapter,
    plan: RebalancePlan,
    post: AccountState,
    drifts: list,
) -> tuple[dict, AccountState, list]:
    """Fallback for failed/partial legs: re-plan residuals, retry book-aware.

    Reconciliation drift is the universal leg-failure signal — it catches
    unfilled IOC slices, oracle-band rejects, and ambiguous responses across
    the batch, book-aware, and TWAP paths alike, because it compares actual
    post-trade positions to the plan target. Re-planning against fresh state
    (rather than resubmitting the original orders) makes the retry safe after
    ambiguous fills: whatever actually filled is already in the state.

    Returns (repair_summary, final_post_state, final_drifts).
    """
    from causal_portfolio.execution.reconcile import reconcile

    cfg = adapter.config
    target = (
        plan.target_snapshot if plan.target_snapshot is not None
        else plan.target_weights
    )
    attempts: list[dict[str, Any]] = []
    for i in range(cfg.repair_attempts):
        attempt: dict[str, Any] = {"attempt": i + 1}
        repair_plan = None
        submission_started = False
        try:
            repair_plan = plan_rebalance(
                target, post, adapter.fetch_mids(), adapter.fetch_meta(), cfg,
            )
            attempt.update({
                "orders": len(repair_plan.orders),
                "target_usd": dict(repair_plan.target_usd),
                "equity_used": repair_plan.equity_used,
                "planned_orders": [asdict(order) for order in repair_plan.orders],
            })
            if not repair_plan.orders:
                attempt["note"] = "residual not tradeable (dust/caps)"
                attempts.append(attempt)
                break
            cost_estimate, cost_gate_reason, repair_orders, cost_skips = _check_cost_gate(
                adapter, repair_plan.orders
            )
            if cost_skips:
                repair_plan = replace(
                    repair_plan,
                    skipped=[*repair_plan.skipped, *cost_skips],
                )
            if cost_estimate is not None:
                attempt["cost_estimate"] = cost_estimate
            if cost_gate_reason is not None:
                attempt["cost_gate_reason"] = cost_gate_reason
            attempt["skipped"] = list(repair_plan.skipped)
            (
                completeness_ratio,
                repair_orders,
                completeness_gate_reason,
                completeness_skips,
            ) = _check_completeness_gate(
                repair_plan,
                repair_orders,
                cfg.min_rebalance_completeness,
            )
            if completeness_skips:
                repair_plan = replace(
                    repair_plan,
                    skipped=[*repair_plan.skipped, *completeness_skips],
                )
                attempt["skipped"] = list(repair_plan.skipped)
            attempt["completeness_ratio"] = completeness_ratio
            if not repair_orders:
                if completeness_gate_reason is not None:
                    attempt["completeness_gate_reason"] = completeness_gate_reason
                attempts.append(attempt)
                break
            if completeness_gate_reason is not None:
                attempt["completeness_gate_reason"] = completeness_gate_reason
            logger.warning(
                "leg repair attempt %d/%d: %d residual order(s) for %s",
                i + 1, cfg.repair_attempts, len(repair_orders),
                [o.coin for o in repair_orders],
            )
            submission_started = True
            response = adapter.submit_orders_book_aware(repair_orders)
            attempt["response"] = response
            post = adapter.fetch_state()
            drifts = reconcile(
                repair_plan, post, tolerance_usd=cfg.min_order_notional_usd,
            )
            attempt["remaining_drifts"] = len(drifts)
            attempts.append(attempt)
            if not drifts:
                break
        except Exception as e:
            logger.exception("leg repair attempt %d failed", i + 1)
            attempt["error"] = str(e)
            attempts.append(attempt)
            try:
                post = adapter.fetch_state()
                drifts = reconcile(
                    repair_plan or plan,
                    post,
                    tolerance_usd=cfg.min_order_notional_usd,
                )
            except Exception as state_error:
                attempt["state_error"] = str(state_error)
                break
            if not drifts or submission_started:
                break
    return {"attempts": attempts, "resolved": not drifts}, post, drifts


def _detect_cancel_race(plan, mid_state, tolerance_usd: float = 25.0) -> bool:
    """Return True if positions moved materially between plan and mid_state.

    Compares plan.current_state.positions to mid_state.positions for every coin
    the plan touches. Trips on any coin whose notional changed by more than
    tolerance_usd — that's our signal a stale order filled.
    """
    pre = plan.current_state.positions
    # Check planned coins plus anything held pre-trade — a position we hold
    # but don't retarget can still close (e.g. stop-loss) during the cancel
    # window, invalidating the equity snapshot. Coins appearing in mid_state
    # only (never planned, never held) are deliberately ignored: that's
    # outside activity, not our race.
    coins_touched = set(plan.target_usd.keys()) | set(pre.keys())
    for coin in coins_touched:
        pre_notional = pre[coin].notional_usd if coin in pre else 0.0
        mid_notional = mid_state.positions[coin].notional_usd if coin in mid_state.positions else 0.0
        if abs(mid_notional - pre_notional) > tolerance_usd:
            logger.warning(
                "cancel-race on %s: pre=$%.2f mid=$%.2f (drift $%.2f)",
                coin, pre_notional, mid_notional, mid_notional - pre_notional,
            )
            return True
    return False


def _position_sizes_changed(
    plan: RebalancePlan,
    before: AccountState,
    post: AccountState,
) -> bool:
    """Return whether a planned or pre-existing position changed size."""
    before_positions = before.positions
    coins = set(plan.target_usd) | set(before_positions)
    return any(
        (before_positions[coin].size if coin in before_positions else 0.0)
        != (post.positions[coin].size if coin in post.positions else 0.0)
        for coin in coins
    )


def _child_cloid(parent_cloid: str, slice_index: int) -> str:
    return make_cloid(f"{parent_cloid}:twap:{slice_index}")


def _slice_order_units(
    order: Order,
    *,
    requested_slices: int,
    effective_slices: int,
    sz_decimals: int,
) -> list[tuple[int, Order]]:
    """Split `order` into `effective_slices` spread over `requested_slices` slots."""
    factor = 10 ** sz_decimals
    total_units = int(round(order.size * factor))
    if total_units <= 0:
        return []

    children: list[tuple[int, Order]] = []
    previous_cumulative_units = 0
    for idx in range(effective_slices):
        cumulative_units = (total_units * (idx + 1)) // effective_slices
        units = cumulative_units - previous_cumulative_units
        previous_cumulative_units = cumulative_units
        if units <= 0:
            continue
        child_size = units / factor
        requested_idx = (
            (requested_slices * (idx + 1)) // effective_slices
        ) - 1
        children.append((
            max(0, requested_idx),
            replace(
                order,
                size=child_size,
                cloid=_child_cloid(order.cloid, max(0, requested_idx)),
            ),
        ))
    return children


def _slice_order(
    order: Order,
    *,
    slices: int,
    sz_decimals: int,
    min_child_notional_usd: float = 0.0,
) -> list[tuple[int, Order]]:
    """Split an order into deterministic TWAP child orders.

    Sizes are split in integer exchange size-units so child orders add exactly
    to the parent size at the market's precision. Tiny orders may not appear
    in every slice, but total child size never exceeds the parent order size.

    When `min_child_notional_usd` is set, the effective slice count is capped
    so no child is sent below the exchange's minimum order value. If the parent
    itself is below the floor (for example a reduce-only dust close), it is sent
    as one child rather than being subdivided into guaranteed rejects.
    """
    factor = 10 ** sz_decimals
    total_units = int(round(order.size * factor))
    if total_units <= 0:
        return []
    requested_slices = max(1, slices)
    min_child_notional_usd = max(0.0, min_child_notional_usd)
    if min_child_notional_usd == 0:
        return _slice_order_units(
            order,
            requested_slices=requested_slices,
            effective_slices=requested_slices,
            sz_decimals=sz_decimals,
        )

    max_effective_slices = min(requested_slices, total_units)
    for effective_slices in range(max_effective_slices, 0, -1):
        children = _slice_order_units(
            order,
            requested_slices=requested_slices,
            effective_slices=effective_slices,
            sz_decimals=sz_decimals,
        )
        if len(children) <= 1 or all(
            child.size * child.limit_px >= min_child_notional_usd
            for _, child in children
        ):
            return children
    return []


def _submit_orders_twap(adapter: HLAdapter, orders: list[Order]) -> dict[str, Any]:
    """Submit a plan as client-side TWAP child orders.

    The adapter still owns pricing. With smart_execution=True each child slice
    is repriced against the live L2 book; otherwise each child inherits the
    parent IOC limit price.
    """
    cfg = adapter.config
    if not orders:
        return {"status": "ok", "response": {"type": "twap", "data": {"slices": []}}}

    meta = adapter.fetch_meta()
    child_orders_by_slice: list[list[Order]] = [[] for _ in range(cfg.twap_slices)]
    child_plan: list[dict[str, Any]] = []
    # HL enforces the minimum at the actual execution price, not our parent
    # limit. Keep a small buffer so a child that is barely $10 at plan time
    # does not become a live reject after book-aware repricing/slippage.
    min_child_notional_usd = cfg.min_order_notional_usd * 1.05
    for order in orders:
        sz_decimals = meta[order.coin].sz_decimals if order.coin in meta else 4
        children = _slice_order(
            order,
            slices=cfg.twap_slices,
            sz_decimals=sz_decimals,
            min_child_notional_usd=min_child_notional_usd,
        )
        parent_notional = order.size * order.limit_px
        child_plan.append({
            "parent_cloid": order.cloid,
            "coin": order.coin,
            "side": "buy" if order.is_buy else "sell",
            "parent_size": order.size,
            "parent_limit_px": order.limit_px,
            "parent_notional_usd": parent_notional,
            "requested_slices": cfg.twap_slices,
            "effective_slices": len(children),
            "min_child_notional_usd": min_child_notional_usd,
            "children": [
                {
                    "slice": idx + 1,
                    "cloid": child.cloid,
                    "size": child.size,
                    "limit_px": child.limit_px,
                    "notional_usd": child.size * child.limit_px,
                    "reduce_only": child.reduce_only,
                    "tif": child.tif,
                }
                for idx, child in children
            ],
        })
        for idx, child in children:
            child_orders_by_slice[idx].append(child)

    interval_seconds = (
        (cfg.twap_minutes * 60.0) / max(cfg.twap_slices - 1, 1)
        if cfg.twap_slices > 1
        else 0.0
    )
    slice_responses: list[dict[str, Any]] = []
    for idx, child_orders in enumerate(child_orders_by_slice):
        if child_orders:
            if cfg.smart_execution:
                response = adapter.submit_orders_book_aware(child_orders)
            else:
                response = adapter.submit_orders(child_orders)
        else:
            response = {"status": "ok", "response": {"data": {"statuses": []}}}
        slice_responses.append({
            "slice": idx + 1,
            "orders": len(child_orders),
            "response": response,
        })
        if idx < len(child_orders_by_slice) - 1 and interval_seconds > 0:
            time.sleep(interval_seconds)

    return {
        "status": "ok",
        "response": {
            "type": "twap",
            "data": {
                "slices": slice_responses,
                "twap_minutes": cfg.twap_minutes,
                "twap_slices": cfg.twap_slices,
                "child_plan": child_plan,
            },
        },
    }
