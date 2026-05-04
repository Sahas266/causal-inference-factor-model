"""Thin Hyperliquid SDK adapter.

Wraps the official `hyperliquid-python-sdk` to:
  - Read account state, mids, and asset metadata into the plain dataclasses
    that rebalancer.py expects (no SDK types leak past this boundary).
  - Submit a list of Order objects via bulk_orders.
  - Cancel open orders before placing a new batch (clean-slate rebalance).

Lazy-imports the SDK so the rest of the execution layer is testable without
it. Install with: `pip install hyperliquid-python-sdk`.

Default base URL is testnet. Mainnet only when ExecutionConfig.testnet=False
AND ExecutionConfig.dry_run=False (caller's responsibility to gate).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.types import (
    AccountState,
    AssetMeta,
    Order,
    Position,
    RebalancePlan,
    SubmitResult,
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
    """Load .env from repo root if dotenv is available. Idempotent enough."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


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
                "pip install hyperliquid-python-sdk eth-account"
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

        self.info = Info(self.base_url, skip_ws=True)

        # Exchange client only needs a wallet for write operations. Reads work
        # without one. We construct it lazily so dry-run / read-only flows
        # don't require a private key.
        self._exchange: Any = None
        self._Account = Account
        self._Exchange = Exchange

    def _ensure_exchange(self) -> Any:
        if self._exchange is None:
            if not self.secret_key:
                raise ValueError(
                    "Private key required for write operations: set "
                    "HL_PRIVATE_KEY or HYPERLIQUID_PRIVATE_KEY env var"
                )
            wallet = self._Account.from_key(self.secret_key)
            self._exchange = self._Exchange(wallet, self.base_url, account_address=self.address)
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
            open_order_ids=[],  # filled by fetch_open_orders if needed
        )

    def fetch_mids(self) -> dict[str, float]:
        raw = self.info.all_mids()
        return {coin: float(px) for coin, px in raw.items()}

    def fetch_meta(self) -> dict[str, AssetMeta]:
        """Pull per-coin metadata (size decimals, max leverage)."""
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
        return meta

    def fetch_open_order_ids(self) -> list[int]:
        raw = self.info.open_orders(self.address)
        return [int(o["oid"]) for o in raw if "oid" in o]

    # ── writes ──────────────────────────────────────────────────────

    def cancel_all_open(self) -> dict[str, Any] | None:
        """Cancel every open order on this account. Best-effort."""
        oids = self.fetch_open_order_ids()
        if not oids:
            return None
        ex = self._ensure_exchange()
        # SDK signature: bulk_cancel(list of {coin, oid}) — needs coin per oid.
        # Easier path: query frontend_open_orders which includes coin, then cancel.
        raw = self.info.frontend_open_orders(self.address)
        requests = [{"coin": o["coin"], "oid": int(o["oid"])} for o in raw if "oid" in o]
        if not requests:
            return None
        return ex.bulk_cancel(requests)

    def submit_orders(self, orders: list[Order]) -> dict[str, Any]:
        """Submit a batch of orders atomically via bulk_orders."""
        if not orders:
            return {"status": "ok", "response": {"data": {"statuses": []}}}
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
                "cloid": o.cloid,
            })
        return ex.bulk_orders(order_args)


def execute_plan(adapter: HLAdapter, plan: RebalancePlan) -> SubmitResult:
    """Execute a RebalancePlan. Honors dry_run flag in adapter.config."""
    if adapter.config.dry_run:
        logger.info("DRY RUN — not submitting %d orders", len(plan.orders))
        return SubmitResult(plan=plan, submitted=False)

    if adapter.config.is_live_mainnet():
        logger.warning("LIVE MAINNET write: %d orders, $%.0f gross",
                       len(plan.orders),
                       sum(abs(d) for d in plan.deltas_usd.values()))

    try:
        adapter.cancel_all_open()
        response = adapter.submit_orders(plan.orders)
        post = adapter.fetch_state()
        return SubmitResult(plan=plan, submitted=True,
                            response=response, post_state=post)
    except Exception as e:
        logger.exception("submit_orders failed")
        return SubmitResult(plan=plan, submitted=False, error=str(e))
