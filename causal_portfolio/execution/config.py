"""Execution-layer configuration with safe defaults.

Default policy:
  - dry_run = True (no orders submitted)
  - testnet = True
  - leverage = 1.0 (no leverage)
  - long-short permitted
  - Conservative position and trade caps

To go live: explicitly set dry_run=False and testnet=False, or use the CLI
flags --live and --mainnet (the CLI requires both for any mainnet write).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Default mapping from CPCM tickers (lowercase) to Hyperliquid coin names
# (uppercase). Verify dynamically against `info.meta()` at runtime — coins
# that aren't listed will be skipped with SkipReason.NOT_LISTED.
DEFAULT_ASSET_MAP: dict[str, str] = {
    "btc": "BTC",
    "eth": "ETH",
    "sol": "SOL",
    "bnb": "BNB",
    "avax": "AVAX",
    "xrp": "XRP",
    "doge": "DOGE",
    "uni": "UNI",
    "aave": "AAVE",
    "link": "LINK",
    "crv": "CRV",
    "pendle": "PENDLE",
    "pepe": "kPEPE",   # HL uses 1k-share units for very low-price coins
    "shib": "kSHIB",
    "ena": "ENA",
    "jup": "JUP",
    "tao": "TAO",
    "hype": "HYPE",
    "aero": "AERO",
    "pol": "POL",
    "morpho": "MORPHO",
    # Stablecoins (usdc, usdt, usde) intentionally absent — they're not perps.
    # Asset coverage gaps (wlfi, zec, lido, etc.) caught at runtime.
}


@dataclass(frozen=True)
class ExecutionConfig:
    """All knobs in one immutable struct so plans are reproducible."""

    # ── Capital sizing ──────────────────────────────────────────────────
    leverage: float = 1.0          # gross_exposure_usd = equity * leverage
    long_only: bool = False        # if True, negative weights → 0 with warning

    # ── Position and trade caps ─────────────────────────────────────────
    max_position_pct: float = 0.30        # cap per-asset weight magnitude
    max_single_trade_pct: float = 0.10    # cap any single trade as % of equity
    max_single_trade_usd: float | None = None  # absolute cap, None = use pct only
    min_order_notional_usd: float = 10.0  # HL perp minimum; full reduce-only close exempt

    # ── Slippage ────────────────────────────────────────────────────────
    slippage_bps: int = 30         # IOC limit at mid ± slippage_bps/10000

    # ── Asset universe ──────────────────────────────────────────────────
    asset_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ASSET_MAP))

    # ── Order-book-aware execution ──────────────────────────────────────
    # When True, execute_plan chases the live L2 book with IOC slices instead
    # of a single batch order at a fixed mid±slippage price. Handles thin
    # books, partial fills, and transient oracle-band divergence by retrying
    # on fresh book data.
    smart_execution: bool = True
    smart_max_attempts: int = 8        # IOC slices per order before giving up
    smart_poll_seconds: float = 1.5    # wait between attempts
    smart_max_band_bps: float = 200.0  # don't price further than this from mid

    # ── Network ─────────────────────────────────────────────────────────
    testnet: bool = True

    # ── Safety ──────────────────────────────────────────────────────────
    dry_run: bool = True           # default OFF — must opt-in to live trading
    max_signal_age_hours: float = 72.0
    allow_stale_signal: bool = False

    # ── TWAP ────────────────────────────────────────────────────────────
    # Optional client-side time slicing. The plan still computes one economic
    # order per coin; execute_plan splits those orders into deterministic
    # child CLOIDs and submits them over the requested window.
    twap_minutes: float = 0.0
    twap_slices: int = 5
    max_twap_minutes: float = 30.0

    def __post_init__(self):
        # Light validation. Don't catch every bad combination — just the obvious
        # foot-guns that would silently corrupt a rebalance plan.
        if self.leverage <= 0:
            raise ValueError(f"leverage must be > 0, got {self.leverage}")
        if not (0 < self.max_position_pct <= 1.0):
            raise ValueError(f"max_position_pct must be in (0, 1], got {self.max_position_pct}")
        if not (0 < self.max_single_trade_pct <= 1.0):
            raise ValueError(f"max_single_trade_pct must be in (0, 1], got {self.max_single_trade_pct}")
        if self.max_single_trade_usd is not None and self.max_single_trade_usd <= 0:
            raise ValueError(
                f"max_single_trade_usd must be > 0 when set, got {self.max_single_trade_usd}"
            )
        if self.min_order_notional_usd < 0:
            raise ValueError(
                f"min_order_notional_usd must be >= 0, got {self.min_order_notional_usd}"
            )
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")
        if self.smart_max_attempts < 1:
            raise ValueError(f"smart_max_attempts must be >= 1, got {self.smart_max_attempts}")
        if self.smart_poll_seconds < 0:
            raise ValueError(f"smart_poll_seconds must be >= 0, got {self.smart_poll_seconds}")
        if self.smart_max_band_bps <= 0:
            raise ValueError(f"smart_max_band_bps must be > 0, got {self.smart_max_band_bps}")
        if self.max_signal_age_hours <= 0:
            raise ValueError(
                f"max_signal_age_hours must be > 0, got {self.max_signal_age_hours}"
            )
        if self.twap_minutes < 0:
            raise ValueError(f"twap_minutes must be >= 0, got {self.twap_minutes}")
        if self.twap_slices < 1:
            raise ValueError(f"twap_slices must be >= 1, got {self.twap_slices}")
        if self.max_twap_minutes <= 0:
            raise ValueError(
                f"max_twap_minutes must be > 0, got {self.max_twap_minutes}"
            )
        if self.twap_minutes > self.max_twap_minutes:
            raise ValueError(
                f"twap_minutes={self.twap_minutes} exceeds max_twap_minutes="
                f"{self.max_twap_minutes}"
            )

    def slippage_factor(self, is_buy: bool) -> float:
        """Multiplier on mid for the IOC limit price."""
        delta = self.slippage_bps / 10_000.0
        return 1.0 + delta if is_buy else 1.0 - delta

    def is_live_mainnet(self) -> bool:
        """The dangerous combination — used by CLI to gate destructive paths."""
        return (not self.dry_run) and (not self.testnet)
