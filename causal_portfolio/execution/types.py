"""Plain-Python data classes for the execution layer.

Intentionally has zero dependencies on the Hyperliquid SDK so the pure
rebalancer can be tested without importing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SkipReason(str, Enum):
    """Why a coin was excluded from the order list."""
    NOT_LISTED = "not_listed_on_hl"
    BELOW_MIN_SIZE = "size_rounded_to_zero"
    EXCEEDS_TRADE_CAP = "single_trade_cap_exceeded"
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
    error: str | None = None
    drifts: list[ReconcileDrift] = field(default_factory=list)  # post-trade reconciliation
