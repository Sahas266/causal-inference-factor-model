"""Plain-Python data classes for the execution layer.

Intentionally has zero dependencies on the Hyperliquid SDK so the pure
rebalancer can be tested without importing it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def make_cloid(seed: str) -> str:
    """Deterministic Hyperliquid client order id from a seed string.

    HL accepts 16-byte (32-char) hex cloids. Hashing the seed keeps retries
    idempotent: the same seed always yields the same cloid, so HL rejects an
    accidental duplicate submission. Every order path (plan orders, IOC
    slices, TWAP children, maker quotes) derives its cloid here; idempotency
    is only as strong as the seed the caller constructs.
    """
    return "0x" + hashlib.sha256(seed.encode()).hexdigest()[:32]


class SkipReason(str, Enum):
    """Why a coin was excluded from the order list."""
    NOT_LISTED = "not_listed_on_hl"
    BELOW_MIN_SIZE = "size_rounded_to_zero"
    BELOW_MIN_NOTIONAL = "below_min_order_notional"
    BELOW_NO_TRADE_BAND = "below_no_trade_band"
    COST_ESTIMATE_UNAVAILABLE = "transaction_cost_estimate_unavailable"
    EXCEEDS_TRANSACTION_COST = "transaction_cost_limit_exceeded"
    INSUFFICIENT_PLAN_COMPLETENESS = "insufficient_plan_completeness"
    EXCEEDS_TRADE_CAP = "single_trade_cap_exceeded"
    EXCEEDS_LEVERAGE_LIMIT = "asset_leverage_limit_exceeded"
    LONG_ONLY_VIOLATION = "negative_weight_with_long_only"


@dataclass(frozen=True)
class AssetMeta:
    """Per-coin metadata pulled from Hyperliquid's `info.meta()`."""
    coin: str           # canonical HL name, e.g. "BTC"
    sz_decimals: int    # digits after the decimal in valid size
    max_leverage: int
    min_size: float     # smallest valid order size = 10**-sz_decimals


@dataclass(frozen=True)
class Position:
    """Current holding for one coin. Sign convention: + long, - short."""
    coin: str
    size: float          # signed
    entry_px: float
    notional_usd: float  # signed: positive = long exposure, negative = short

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0


@dataclass(frozen=True)
class AccountState:
    """Snapshot of an HL account at a point in time."""
    address: str
    account_value_usd: float           # total equity (cash + unrealized PnL)
    margin_used_usd: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def free_margin_usd(self) -> float:
        return max(self.account_value_usd - self.margin_used_usd, 0.0)


@dataclass(frozen=True)
class TargetSnapshot:
    """Model target weights plus the provenance needed for safe execution."""

    weights: dict[str, float]
    as_of: datetime | None = None
    generated_at: datetime | None = None
    strategy: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for raw_ticker, raw_weight in self.weights.items():
            ticker = raw_ticker.strip().lower()
            if not ticker:
                raise ValueError("Target ticker names cannot be empty")
            if ticker in normalized:
                raise ValueError(f"Duplicate target ticker after normalization: {ticker!r}")
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise ValueError(f"Weight for {raw_ticker!r} must be numeric")
            weight = float(raw_weight)
            if not math.isfinite(weight):
                raise ValueError(f"Weight for {raw_ticker!r} must be finite")
            normalized[ticker] = weight
        object.__setattr__(self, "weights", normalized)
        object.__setattr__(self, "metadata", dict(self.metadata))
        for field_name in ("as_of", "generated_at"):
            value = getattr(self, field_name)
            if value is not None:
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                object.__setattr__(self, field_name, value.astimezone(timezone.utc))

    @property
    def target_id(self) -> str:
        """Stable identifier for the economic target, independent of file path."""
        payload = {
            "weights": sorted(self.weights.items()),
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "strategy": self.strategy,
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def freshness_error(
        self,
        max_age_hours: float,
        *,
        now: datetime | None = None,
        future_tolerance_minutes: float = 5.0,
    ) -> str | None:
        """Return a blocking freshness error, or None when the target is current."""
        if self.as_of is None:
            return "target is missing as_of/rebalance_date provenance"
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        age_hours = (current.astimezone(timezone.utc) - self.as_of).total_seconds() / 3600
        if age_hours < -(future_tolerance_minutes / 60):
            return f"target as_of is {abs(age_hours):.1f}h in the future"
        if age_hours > max_age_hours:
            return f"target is stale: {age_hours:.1f}h old (limit {max_age_hours:.1f}h)"
        return None


@dataclass(frozen=True)
class Order:
    """A single order to submit. Maps directly to HL's order action."""
    coin: str           # HL coin name
    is_buy: bool        # True = buy, False = sell
    size: float         # always positive; direction encoded by is_buy
    limit_px: float     # IOC limit (we never use pure market)
    cloid: str          # client order id for idempotency
    reduce_only: bool = False
    tif: str = "Ioc"    # time-in-force; "Ioc" = immediate-or-cancel


@dataclass(frozen=True)
class RebalancePlan:
    """Output of plan_rebalance: everything needed to audit/execute a rebalance.

    A plan with empty `orders` is a valid no-op (already on-target or all dust).
    """
    timestamp_ms: int
    target_weights: dict[str, float]   # CPCM ticker → weight (signed)
    current_state: AccountState
    target_usd: dict[str, float]       # HL coin → target signed USD exposure
    deltas_usd: dict[str, float]       # HL coin → signed USD trade size
    orders: list[Order]
    skipped: list[tuple[str, SkipReason, str]]  # (coin, reason, detail)
    equity_used: float                  # equity * leverage (the scale factor)
    notes: list[str] = field(default_factory=list)
    network: str | None = None
    target_snapshot: TargetSnapshot | None = None
    mids: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        n_orders = len(self.orders)
        gross_usd = sum(abs(d) for d in self.deltas_usd.values())
        return (
            f"RebalancePlan: {n_orders} orders, ${gross_usd:,.0f} gross trade size, "
            f"{len(self.skipped)} skipped, equity ${self.current_state.account_value_usd:,.0f}"
        )


@dataclass(frozen=True)
class ReconcileDrift:
    """A coin whose post-execution notional differs materially from target."""
    coin: str
    target_usd: float       # what plan.target_usd[coin] said
    actual_usd: float       # signed actual notional from post_state
    drift_usd: float        # actual - target (signed)
    drift_pct: float        # drift_usd / abs(target_usd) if target nonzero, else inf


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of submitting a RebalancePlan to Hyperliquid."""
    plan: RebalancePlan
    submitted: bool                     # False if dry_run
    response: dict | None = None        # raw HL response, if submitted
    post_state: AccountState | None = None
    error: str | None = None             # failure before known completed submission
    drifts: list[ReconcileDrift] = field(default_factory=list)  # post-trade reconciliation
    post_submit_error: str | None = None  # post-state or reconciliation failure
    audit_error: str | None = None        # structured audit append failure
    # Leg-failure fallback summary: {"attempts": [...], "resolved": bool}
    # when a repair pass ran, else None. `drifts` reflects post-repair state.
    repair: dict | None = None
    # Pre-submit all-in fee + L2 impact estimate. Dict mirrors `repair` so the
    # audit format can evolve without breaking positional result compatibility.
    cost_estimate: dict | None = None
    cost_gate_reason: str | None = None
    # Initial orders actually handed to the exchange. None keeps old manually
    # constructed results distinguishable from a known empty submission.
    # Read it through `effective_submitted_orders`, not directly.
    submitted_orders: list[Order] | None = None
    # Submitted share of intended exposure-increasing gross trade notional.
    completeness_ratio: float | None = None

    @property
    def effective_submitted_orders(self) -> list[Order]:
        """Orders that actually reached the exchange, resolving the None case.

        `submitted_orders` is None on results built before the field existed
        (and by hand in tests), where the best available answer is "all planned
        orders if we submitted, otherwise none". Since gates can now submit a
        strict subset of `plan.orders`, every consumer needs the same three-way
        rule — audit counts, trace fill attribution, and the Telegram summary
        must not disagree about what went out.
        """
        if self.submitted_orders is not None:
            return self.submitted_orders
        return list(self.plan.orders) if self.submitted else []
