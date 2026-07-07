"""Dedicated Hyperliquid maker adapter using post-only strategy-owned orders."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from causal_portfolio.execution.hyperliquid import (
    ADDRESS_ENV_VARS,
    KEY_ENV_VARS,
    MAINNET_URL,
    TESTNET_URL,
    _load_dotenv_once,
    _read_env_chain,
)
from causal_portfolio.execution.precision import round_price as _round_price
from causal_portfolio.market_making.engine import QuoteDecision
from causal_portfolio.market_making.types import BookLevel, BookSnapshot, Quote, QuotePair

logger = logging.getLogger("cpcm.market_making.hl")


class PostOnlyWouldCrossError(ValueError):
    """A post-only quote would cross the live opposite touch.

    Raised instead of submitting, because HL would either reject the Alo
    order or (worse) a retry at the same price would keep failing. Callers
    that reprice-and-retry catch this type rather than matching message text.
    """


@dataclass(frozen=True)
class HLMarketMakerConfig:
    testnet: bool = True
    dry_run: bool = True
    allow_mainnet: bool = False
    dedicated_account: bool = False
    dead_man_timeout_seconds: int = 30
    state_path: Path | None = None

    def is_live_mainnet(self) -> bool:
        """The dangerous combination — mirrors ExecutionConfig.is_live_mainnet."""
        return (not self.dry_run) and (not self.testnet)

    def __post_init__(self) -> None:
        if self.dead_man_timeout_seconds < 5:
            raise ValueError("dead_man_timeout_seconds must be >= 5")
        if self.is_live_mainnet() and not self.allow_mainnet:
            raise ValueError("live mainnet requires allow_mainnet=True")


class HLMarketMakerAdapter:
    """WebSocket/read/write boundary for one dedicated market-making account.

    Unlike ``HLAdapter``, this class has no account-wide cancel method. Normal
    refreshes cancel only CLOIDs previously submitted by this strategy.
    """

    def __init__(
        self,
        config: HLMarketMakerConfig = HLMarketMakerConfig(),
        *,
        address: str | None = None,
        secret_key: str | None = None,
    ):
        try:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
        except ImportError as exc:
            raise ImportError(
                "Install hyperliquid-python-sdk==0.24.0 and eth-account==0.13.7"
            ) from exc
        _load_dotenv_once()
        self.config = config
        self.address = address or _read_env_chain(ADDRESS_ENV_VARS)
        self.secret_key = secret_key or _read_env_chain(KEY_ENV_VARS)
        if not self.address:
            raise ValueError("market-maker adapter requires a wallet address")
        self.base_url = TESTNET_URL if config.testnet else MAINNET_URL
        self.info = Info(self.base_url, skip_ws=False)
        self._Account = Account
        self._Exchange = Exchange
        self._exchange: Any = None
        self._owned: dict[str, str] = {}
        self._subscriptions: list[tuple[dict[str, str], int]] = []
        self._sz_decimals_cache: dict[str, int] = {}
        self._load_state()

    def _ensure_exchange(self) -> Any:
        if self._exchange is None:
            if not self.secret_key:
                raise ValueError("write operations require HL_PRIVATE_KEY")
            wallet = self._Account.from_key(self.secret_key)
            self._exchange = self._Exchange(
                wallet, self.base_url, account_address=self.address
            )
        return self._exchange

    def _load_state(self) -> None:
        path = self.config.state_path
        if path is None or not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("network") != ("testnet" if self.config.testnet else "mainnet"):
            raise ValueError("maker state file belongs to a different network")
        if payload.get("address") not in (None, self.address):
            raise ValueError("maker state file belongs to a different address")
        self._owned = {str(k): str(v) for k, v in payload.get("owned_cloids", {}).items()}

    def _save_state(self) -> None:
        path = self.config.state_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "network": "testnet" if self.config.testnet else "mainnet",
            "address": self.address,
            "owned_cloids": self._owned,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def fetch_l2_book(self, coin: str, *, depth: int = 20) -> BookSnapshot:
        raw = self.info.l2_snapshot(coin)
        levels = raw.get("levels", [[], []])
        timestamp = int(raw.get("time", int(time.time() * 1_000)))
        return BookSnapshot(
            timestamp_ms=timestamp,
            received_ms=int(time.time() * 1_000),
            coin=coin,
            bids=tuple(BookLevel(float(x["px"]), float(x["sz"])) for x in levels[0][:depth]),
            asks=tuple(BookLevel(float(x["px"]), float(x["sz"])) for x in levels[1][:depth]),
        )

    def subscribe_market(
        self, coin: str, callback: Callable[[dict[str, Any]], None]
    ) -> list[int]:
        ids: list[int] = []
        for channel in ("trades", "l2Book", "bbo"):
            subscription = {"type": channel, "coin": coin}
            subscription_id = self.info.subscribe(subscription, callback)
            self._subscriptions.append((subscription, subscription_id))
            ids.append(subscription_id)
        return ids

    def subscribe_account(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> list[int]:
        ids: list[int] = []
        for channel in ("userFills", "orderUpdates"):
            subscription = {"type": channel, "user": self.address}
            subscription_id = self.info.subscribe(subscription, callback)
            self._subscriptions.append((subscription, subscription_id))
            ids.append(subscription_id)
        return ids

    def unsubscribe_all(self) -> None:
        for subscription, subscription_id in reversed(self._subscriptions):
            try:
                self.info.unsubscribe(subscription, subscription_id)
            except Exception:
                logger.exception("failed to unsubscribe %s", subscription)
        self._subscriptions.clear()

    def reconcile_owned_orders(self) -> dict[str, list[str]]:
        """Compare persisted ownership with currently open orders."""
        raw = self.info.frontend_open_orders(self.address)
        open_cloids = {
            str(order.get("cloid")): str(order.get("coin"))
            for order in raw
            if order.get("cloid")
        }
        stale = sorted(set(self._owned) - set(open_cloids))
        for cloid in stale:
            self._owned.pop(cloid, None)
        self._save_state()
        return {
            "owned_open": sorted(set(self._owned) & set(open_cloids)),
            "stale_local": stale,
            "foreign_open": sorted(set(open_cloids) - set(self._owned)),
        }

    def fetch_account_metrics(self, coin: str) -> dict[str, float]:
        """Return only the risk state needed by the maker engine."""
        raw = self.info.user_state(self.address)
        summary = raw.get("marginSummary", {})
        account_value = float(summary.get("accountValue", 0.0))
        margin_used = float(summary.get("totalMarginUsed", 0.0))
        inventory = 0.0
        for item in raw.get("assetPositions", []):
            position = item.get("position", {})
            if position.get("coin") == coin:
                inventory = float(position.get("szi", 0.0))
                break
        return {
            "account_value_usd": account_value,
            "free_margin_usd": max(account_value - margin_used, 0.0),
            "inventory_base": inventory,
        }

    def cancel_strategy_quotes(self) -> dict[str, Any] | None:
        """Cancel only persisted strategy CLOIDs; foreign orders are untouched."""
        if not self._owned:
            return None
        if self.config.dry_run:
            return {"status": "dry_run", "cancel_cloids": sorted(self._owned)}
        from hyperliquid.utils.types import Cloid

        requests = [
            {"coin": coin, "cloid": Cloid.from_str(cloid)}
            for cloid, coin in sorted(self._owned.items())
        ]
        response = self._ensure_exchange().bulk_cancel_by_cloid(requests)
        if response.get("status") == "ok":
            self._owned.clear()
            self._save_state()
        return response

    @staticmethod
    def _quotes(pair: QuotePair) -> list[Quote]:
        return [quote for quote in (pair.bid, pair.ask) if quote is not None]

    def submit_quote_pair(
        self,
        pair: QuotePair,
        *,
        acknowledge_mainnet: bool = False,
    ) -> dict[str, Any]:
        if self.config.is_live_mainnet() and not acknowledge_mainnet:
            raise PermissionError("live mainnet maker write requires acknowledge_mainnet=True")
        quotes = self._quotes(pair)
        if self.config.dry_run:
            return {
                "status": "dry_run",
                "orders": [
                    {
                        "coin": pair.diagnostics.get("coin"),
                        "is_buy": q.is_buy,
                        "px": q.price,
                        "sz": q.size,
                        "cloid": q.cloid,
                    }
                    for q in quotes
                ],
            }
        from hyperliquid.utils.types import Cloid

        coin = str(pair.diagnostics.get("coin", ""))
        if not coin:
            raise ValueError("QuotePair diagnostics must include coin")
        live_book = self.fetch_l2_book(coin, depth=1)
        for quote in quotes:
            if _round_price(quote.price, self._size_decimals(coin)) != quote.price:
                raise ValueError("quote price violates Hyperliquid precision rules")
            if (
                quote.is_buy
                and live_book.best_ask is not None
                and quote.price >= live_book.best_ask
            ):
                raise PostOnlyWouldCrossError(
                    "bid would cross live ask; refusing post-only submission"
                )
            if (
                not quote.is_buy
                and live_book.best_bid is not None
                and quote.price <= live_book.best_bid
            ):
                raise PostOnlyWouldCrossError(
                    "ask would cross live bid; refusing post-only submission"
                )
        requests = [
            {
                "coin": coin,
                "is_buy": quote.is_buy,
                "sz": quote.size,
                "limit_px": quote.price,
                "order_type": {"limit": {"tif": "Alo"}},
                "reduce_only": False,
                "cloid": Cloid.from_str(quote.cloid),
            }
            for quote in quotes
        ]
        response = self._ensure_exchange().bulk_orders(requests)
        statuses = response.get("response", {}).get("data", {}).get("statuses", [])
        for quote, status in zip(quotes, statuses):
            if "resting" in status:
                self._owned[quote.cloid] = coin
        self._save_state()
        return response

    def _size_decimals(self, coin: str) -> int:
        # Asset metadata is static for the life of an adapter; without the
        # cache every quote replacement pays a full-universe meta() fetch.
        cached = self._sz_decimals_cache.get(coin)
        if cached is not None:
            return cached
        for item in self.info.meta().get("universe", []):
            if item.get("name") == coin:
                value = int(item.get("szDecimals", 4))
                self._sz_decimals_cache[coin] = value
                return value
        raise ValueError(f"coin is not listed on Hyperliquid: {coin}")

    def apply_decision(
        self,
        decision: QuoteDecision,
        *,
        acknowledge_mainnet: bool = False,
    ) -> dict[str, Any]:
        if decision.cancel:
            return {"cancel": self.cancel_strategy_quotes()}
        if not decision.replace or decision.quotes is None:
            return {"status": "unchanged"}
        cancelled = self.cancel_strategy_quotes()
        submitted = self.submit_quote_pair(
            decision.quotes, acknowledge_mainnet=acknowledge_mainnet
        )
        return {"cancel": cancelled, "submit": submitted}

    def arm_dead_man(self) -> dict[str, Any]:
        """Arm HL's account-wide scheduled cancel on a dedicated account only."""
        if not self.config.dedicated_account:
            raise PermissionError(
                "Hyperliquid scheduleCancel affects every account order; set "
                "dedicated_account=True only on a maker-only account/subaccount"
            )
        if self.config.dry_run:
            return {"status": "dry_run"}
        cancel_at = int(time.time() * 1_000) + self.config.dead_man_timeout_seconds * 1_000
        return self._ensure_exchange().schedule_cancel(cancel_at)

    def disarm_dead_man(self) -> dict[str, Any] | None:
        if self.config.dry_run:
            return {"status": "dry_run"}
        if not self.config.dedicated_account:
            return None
        return self._ensure_exchange().schedule_cancel(None)

    def close(self) -> None:
        self.unsubscribe_all()
        try:
            self.info.disconnect_websocket()
        except Exception:
            logger.debug("websocket already closed", exc_info=True)
