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
    "pepe": "PEPE",
    "shib": "kSHIB",   # HL lists shiba as kSHIB (1k shares)
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

    # ── Filters ─────────────────────────────────────────────────────────
    min_trade_usd: float = 25.0    # below this → skipped (dust)

    # ── Slippage ────────────────────────────────────────────────────────
    slippage_bps: int = 30         # IOC limit at mid ± slippage_bps/10000

    # ── Asset universe ──────────────────────────────────────────────────
    asset_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ASSET_MAP))

    # ── Network ─────────────────────────────────────────────────────────
    testnet: bool = True

    # ── Safety ──────────────────────────────────────────────────────────
    dry_run: bool = True           # default OFF — must opt-in to live trading

    def __post_init__(self):
        # Light validation. Don't catch every bad combination — just the obvious
        # foot-guns that would silently corrupt a rebalance plan.
        if self.leverage <= 0:
            raise ValueError(f"leverage must be > 0, got {self.leverage}")
        if not (0 < self.max_position_pct <= 1.0):
            raise ValueError(f"max_position_pct must be in (0, 1], got {self.max_position_pct}")
        if not (0 < self.max_single_trade_pct <= 1.0):
            raise ValueError(f"max_single_trade_pct must be in (0, 1], got {self.max_single_trade_pct}")
        if self.min_trade_usd < 0:
            raise ValueError(f"min_trade_usd must be >= 0, got {self.min_trade_usd}")
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")

    def slippage_factor(self, is_buy: bool) -> float:
        """Multiplier on mid for the IOC limit price."""
        delta = self.slippage_bps / 10_000.0
        return 1.0 + delta if is_buy else 1.0 - delta

    def is_live_mainnet(self) -> bool:
        """The dangerous combination — used by CLI to gate destructive paths."""
        return (not self.dry_run) and (not self.testnet)
