"""One-active-rebalance SQLite trace with permanent cycle archives."""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from causal_portfolio.execution.config import DEFAULT_ASSET_MAP
from causal_portfolio.execution.types import AccountState, RebalancePlan, SubmitResult, TargetSnapshot

logger = logging.getLogger("cpcm.execution.trace")

ACTIVE_NAME = "rebalance-current.sqlite3"
_ARCHIVE_RETRIES = 20
_ARCHIVE_RETRY_SECONDS = 0.1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    target_id TEXT,
    kind TEXT NOT NULL,
    coin TEXT,
    price_source TEXT,
    mid_price REAL,
    target_weight REAL,
    target_usd REAL,
    delta_usd REAL,
    position_size REAL,
    position_usd REAL,
    unrealized_pnl_usd REAL,
    equity_usd REAL,
    margin_used_usd REAL,
    free_margin_usd REAL,
    gross_exposure_usd REAL,
    net_exposure_usd REAL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind_ts ON events(kind, ts_utc);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json(value: Any) -> str:
    def convert(item: Any) -> Any:
        if is_dataclass(item) and not isinstance(item, type):
            return {key: convert(val) for key, val in asdict(item).items()}
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        return item

    return json.dumps(
        convert(value),
        allow_nan=False,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )


def active_path(log_dir: Path | None = None) -> Path:
    if log_dir is None:
        from causal_portfolio.execution import audit

        log_dir = audit.LOG_DIR
    return Path(log_dir) / ACTIVE_NAME


def archive_path(log_dir: Path, started_utc: str, target_id: str) -> Path:
    try:
        stamp = datetime.fromisoformat(started_utc).astimezone(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    except ValueError:
        stamp = re.sub(r"[^0-9A-Za-z]+", "", started_utc)[:32] or "unknown"
    safe_target = re.sub(r"[^0-9A-Za-z_-]+", "", target_id) or "unversioned"
    return Path(log_dir) / f"rebalance-{stamp}-{safe_target}.sqlite3"


def _archive_active(path: Path, archive: Path) -> None:
    """Retry brief Windows sharing violations from a concurrent PnL tick."""
    for attempt in range(_ARCHIVE_RETRIES):
        try:
            path.replace(archive)
            return
        except OSError:
            if attempt == _ARCHIVE_RETRIES - 1:
                raise
            time.sleep(_ARCHIVE_RETRY_SECONDS)


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)


def read_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with _connect(path) as conn:
        return dict(conn.execute("SELECT key, value FROM meta"))


def _write_meta(conn: sqlite3.Connection, values: dict[str, Any]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in values.items() if value is not None],
    )


def _insert_event(
    conn: sqlite3.Connection,
    *,
    target_id: str | None,
    kind: str,
    ts_utc: str | None = None,
    coin: str | None = None,
    price_source: str | None = None,
    mid_price: float | None = None,
    target_weight: float | None = None,
    target_usd: float | None = None,
    delta_usd: float | None = None,
    position_size: float | None = None,
    position_usd: float | None = None,
    unrealized_pnl_usd: float | None = None,
    equity_usd: float | None = None,
    margin_used_usd: float | None = None,
    free_margin_usd: float | None = None,
    gross_exposure_usd: float | None = None,
    net_exposure_usd: float | None = None,
    payload: Any = None,
) -> None:
    conn.execute(
        """
        INSERT INTO events(
            ts_utc, target_id, kind, coin, price_source, mid_price,
            target_weight, target_usd, delta_usd, position_size, position_usd,
            unrealized_pnl_usd, equity_usd, margin_used_usd, free_margin_usd,
            gross_exposure_usd, net_exposure_usd, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts_utc or _utc_now().isoformat(),
            target_id,
            kind,
            coin,
            price_source,
            mid_price,
            target_weight,
            target_usd,
            delta_usd,
            position_size,
            position_usd,
            unrealized_pnl_usd,
            equity_usd,
            margin_used_usd,
            free_margin_usd,
            gross_exposure_usd,
            net_exposure_usd,
            _json(payload or {}),
        ),
    )


def _record_model_target(
    path: Path,
    target: TargetSnapshot,
    model_prices: dict[str, float] | None,
    price_source: str | None,
    asset_map: dict[str, str] | None = None,
) -> None:
    with _connect(path) as conn:
        existing = conn.execute(
            "SELECT id, coin, price_source, mid_price, payload_json FROM events "
            "WHERE kind='model_target' AND target_id=? ORDER BY id",
            (target.target_id,),
        ).fetchall()
        if existing:
            # A model may open the trace before an ExecutionConfig exists, then
            # execute_target() reopens the same cycle with the authoritative
            # venue map. Enrich those original rows instead of keeping the
            # guessed/default map forever.
            for row_id, ticker, old_source, old_mid, payload_json in existing:
                payload = json.loads(payload_json)
                payload["metadata"] = target.metadata
                if asset_map is not None:
                    payload["hl_coin"] = asset_map.get(ticker)
                conn.execute(
                    "UPDATE events SET price_source=?, mid_price=?, payload_json=? "
                    "WHERE id=?",
                    (
                        price_source if price_source is not None else old_source,
                        (model_prices or {}).get(ticker, old_mid),
                        _json(payload),
                        row_id,
                    ),
                )
            return
        ts = _utc_now().isoformat()
        for ticker, weight in sorted(target.weights.items()):
            _insert_event(
                conn,
                target_id=target.target_id,
                kind="model_target",
                ts_utc=ts,
                coin=ticker,
                price_source=price_source,
                mid_price=(model_prices or {}).get(ticker),
                target_weight=weight,
                payload={
                    # Must be the SAME map the planner used, not the module
                    # default: a config with a custom asset_map would otherwise
                    # store hl_coin=None here, and record_portfolio_snapshot's
                    # ticker.upper() fallback would key the target row
                    # differently from the position row, splitting the panel's
                    # target-vs-actual comparison into phantom rows.
                    "hl_coin": (asset_map or DEFAULT_ASSET_MAP).get(ticker),
                    "metadata": target.metadata,
                },
            )


def start_cycle(
    target: TargetSnapshot,
    *,
    expected_next_rebalance: datetime | None = None,
    model_prices: dict[str, float] | None = None,
    price_source: str | None = None,
    asset_map: dict[str, str] | None = None,
    log_dir: Path | None = None,
) -> Path | None:
    """Start/reuse a target cycle; archive an older cycle without overwrite.

    `asset_map` should be the executing ExecutionConfig.asset_map so the trace
    resolves model tickers to venue coins exactly as the planner did. It
    defaults to DEFAULT_ASSET_MAP only so a model can open a cycle before an
    execution config exists.
    """
    if expected_next_rebalance is not None and expected_next_rebalance.tzinfo is None:
        expected_next_rebalance = expected_next_rebalance.replace(tzinfo=timezone.utc)
    path = active_path(log_dir)
    directory = path.parent
    try:
        if path.exists():
            old = read_meta(path)
            if old.get("target_id") != target.target_id:
                archive = archive_path(
                    directory,
                    old.get("started_utc", "unknown"),
                    old.get("target_id", "unversioned"),
                )
                if archive.exists():
                    logger.error("trace archive exists; preserving active cycle: %s", archive)
                    return None
                try:
                    _archive_active(path, archive)
                except OSError:
                    logger.exception("trace archive failed; preserving active cycle")
                    return None
        _initialize(path)
        with _connect(path) as conn:
            meta = dict(conn.execute("SELECT key, value FROM meta"))
            if not meta:
                _write_meta(conn, {
                    "target_id": target.target_id,
                    "strategy": target.strategy or "unknown",
                    "signal_time": target.as_of.isoformat() if target.as_of else None,
                    "generated_at": (
                        target.generated_at.isoformat() if target.generated_at else None
                    ),
                    "expected_next_rebalance": (
                        expected_next_rebalance.astimezone(timezone.utc).isoformat()
                        if expected_next_rebalance
                        else target.metadata.get("expected_next_rebalance")
                    ),
                    "started_utc": _utc_now().isoformat(),
                })
            else:
                # Same target, richer provenance: preserve the cycle start but
                # accept cadence supplied after the first best-effort open.
                next_rebalance = None
                if expected_next_rebalance is not None:
                    next_rebalance = expected_next_rebalance.astimezone(
                        timezone.utc
                    ).isoformat()
                elif not meta.get("expected_next_rebalance"):
                    next_rebalance = target.metadata.get("expected_next_rebalance")
                if next_rebalance:
                    _write_meta(conn, {"expected_next_rebalance": next_rebalance})
        _record_model_target(path, target, model_prices, price_source, asset_map)
        return path
    except Exception:
        logger.exception("trace cycle initialization failed (continuing without trace)")
        return None


def _matching_active(target_id: str | None, log_dir: Path | None) -> Path | None:
    path = active_path(log_dir)
    if not path.exists():
        return None
    meta = read_meta(path)
    if target_id is not None and meta.get("target_id") != target_id:
        return None
    return path


def _realized_fill_cost(result: SubmitResult) -> dict[str, float] | None:
    """Compute observed fill slippage plus configured fee estimate when available."""
    response = result.response or {}
    data = response.get("response", {}).get("data", {})
    fills: list[tuple[str, float, float]] = []
    for summary in data.get("summaries", []):
        coin = summary.get("coin")
        for fill in summary.get("fills", []):
            if coin and fill.get("px") is not None:
                fills.append((coin, float(fill["sz"]), float(fill["px"])))
    if not fills:
        # HL's statuses array aligns with the orders actually submitted, so a
        # gated-out leg would otherwise shift attribution onto the wrong coin.
        for order, status in zip(
            result.effective_submitted_orders, data.get("statuses", [])
        ):
            fill = status.get("filled") if isinstance(status, dict) else None
            if fill and fill.get("avgPx") is not None:
                fills.append((
                    order.coin,
                    float(fill.get("totalSz", 0.0)),
                    float(fill["avgPx"]),
                ))
    if not fills:
        return None
    fee_bps = float((result.cost_estimate or {}).get("estimated_taker_fee_bps", 0.0))
    notional = slippage = fee = 0.0
    for coin, size, fill_px in fills:
        mid = result.plan.mids.get(coin)
        if mid is None or mid <= 0 or size <= 0:
            continue
        notional += size * mid
        slippage += size * abs(fill_px - mid)
        fee += size * fill_px * fee_bps / 10_000.0
    if notional <= 0:
        return None
    total = slippage + fee
    return {
        "filled_notional_usd": notional,
        "slippage_usd": slippage,
        "estimated_fee_usd": fee,
        "all_in_cost_usd": total,
        "all_in_cost_bps": total / notional * 10_000.0,
    }


def record_execution_result(
    result: SubmitResult, *, log_dir: Path | None = None
) -> bool:
    plan = result.plan
    target_id = plan.target_snapshot.target_id if plan.target_snapshot else None
    path = _matching_active(target_id, log_dir)
    if path is None:
        return False
    try:
        ts = _utc_now().isoformat()
        orders = {order.coin: order for order in plan.orders}
        submitted_order_list = result.effective_submitted_orders
        submitted_orders = {order.coin: order for order in submitted_order_list}
        with _connect(path) as conn:
            for coin in sorted(set(plan.target_usd) | set(plan.deltas_usd)):
                position = plan.current_state.positions.get(coin)
                order = orders.get(coin)
                submitted_order = submitted_orders.get(coin)
                _insert_event(
                    conn,
                    target_id=target_id,
                    kind="execution_plan",
                    ts_utc=ts,
                    coin=coin,
                    price_source="hyperliquid_mid",
                    mid_price=plan.mids.get(coin),
                    target_weight=(
                        plan.target_usd.get(coin, 0.0) / plan.equity_used
                        if plan.equity_used
                        else None
                    ),
                    target_usd=plan.target_usd.get(coin),
                    delta_usd=plan.deltas_usd.get(coin),
                    position_size=position.size if position else 0.0,
                    position_usd=position.notional_usd if position else 0.0,
                    equity_usd=plan.current_state.account_value_usd,
                    payload={
                        "order": asdict(order) if order else None,
                        "submitted_order": (
                            asdict(submitted_order) if submitted_order else None
                        ),
                        "notes": plan.notes,
                        "skipped": [row for row in plan.skipped if row[0] == coin],
                    },
                )
            post = result.post_state
            payload = {
                "submitted": result.submitted,
                "error": result.error,
                "post_submit_error": result.post_submit_error,
                "audit_error": result.audit_error,
                "response": result.response,
                "drifts": result.drifts,
                "repair": result.repair,
                "cost_estimate": result.cost_estimate,
                "cost_gate_reason": result.cost_gate_reason,
                "completeness_ratio": result.completeness_ratio,
                "skipped": plan.skipped,
                "planned_order_count": len(plan.orders),
                "submitted_order_count": len(submitted_order_list),
                "submitted_orders": submitted_order_list,
                "realized_fill_cost": _realized_fill_cost(result),
            }
            _insert_event(
                conn,
                target_id=target_id,
                kind="execution_result",
                ts_utc=ts,
                equity_usd=(post or plan.current_state).account_value_usd,
                margin_used_usd=(post or plan.current_state).margin_used_usd,
                free_margin_usd=(post or plan.current_state).free_margin_usd,
                payload=payload,
            )
            if result.cost_gate_reason:
                _insert_event(
                    conn,
                    target_id=target_id,
                    kind="cost_gate_partial" if result.submitted else "cost_gate_noop",
                    ts_utc=ts,
                    equity_usd=plan.current_state.account_value_usd,
                    payload={
                        **(result.cost_estimate or {}),
                        "reason": result.cost_gate_reason,
                    },
                )
        return True
    except Exception:
        logger.exception("trace execution write failed (continuing)")
        return False


def record_portfolio_snapshot(
    state: AccountState,
    mids: dict[str, float],
    *,
    log_dir: Path | None = None,
) -> bool:
    path = _matching_active(None, log_dir)
    if path is None:
        return False
    try:
        meta = read_meta(path)
        target_id = meta.get("target_id")
        with _connect(path) as conn:
            target_rows = conn.execute(
                "SELECT coin, target_weight, payload_json FROM events "
                "WHERE kind='model_target' ORDER BY id"
            ).fetchall()
            targets: dict[str, float] = {}
            display: dict[str, str] = {}
            for ticker, weight, payload_json in target_rows:
                payload = json.loads(payload_json)
                hl_coin = payload.get("hl_coin") or str(ticker).upper()
                targets[hl_coin] = float(weight)
                display[hl_coin] = ticker
            coins = sorted(set(targets) | set(state.positions))
            ts = _utc_now().isoformat()
            total_pnl = 0.0
            missing_mid_coins: list[str] = []
            for coin in coins:
                position = state.positions.get(coin)
                mid = mids.get(coin)
                if position is not None and mid is None:
                    missing_mid_coins.append(coin)
                pnl = (
                    (mid - position.entry_px) * position.size
                    if position is not None and mid is not None
                    else None
                )
                if pnl is not None:
                    total_pnl += pnl
                _insert_event(
                    conn,
                    target_id=target_id,
                    kind="portfolio_snapshot",
                    ts_utc=ts,
                    coin=coin,
                    price_source="hyperliquid_mid",
                    mid_price=mid,
                    target_weight=targets.get(coin),
                    position_size=position.size if position else 0.0,
                    position_usd=position.notional_usd if position else 0.0,
                    unrealized_pnl_usd=pnl,
                    equity_usd=state.account_value_usd,
                    margin_used_usd=state.margin_used_usd,
                    free_margin_usd=state.free_margin_usd,
                    payload={
                        "ticker": display.get(coin, coin.lower()),
                        "actual_weight": (
                            (position.notional_usd if position else 0.0)
                            / state.account_value_usd
                            if state.account_value_usd
                            else None
                        ),
                        "entry_px": position.entry_px if position else None,
                    },
                )
            notionals = [position.notional_usd for position in state.positions.values()]
            gross = sum(abs(value) for value in notionals)
            net = sum(notionals)
            complete_total_pnl = None if missing_mid_coins else total_pnl
            _insert_event(
                conn,
                target_id=target_id,
                kind="portfolio_snapshot",
                ts_utc=ts,
                equity_usd=state.account_value_usd,
                margin_used_usd=state.margin_used_usd,
                free_margin_usd=state.free_margin_usd,
                gross_exposure_usd=gross,
                net_exposure_usd=net,
                unrealized_pnl_usd=complete_total_pnl,
                payload={
                    "total_unrealized_pnl_usd": complete_total_pnl,
                    "priced_unrealized_pnl_usd": total_pnl,
                    "missing_mid_coins": missing_mid_coins,
                    "position_count": len(state.positions),
                },
            )
        return True
    except Exception:
        logger.exception("trace portfolio write failed (continuing)")
        return False


def record_portfolio_error(
    error: BaseException,
    *,
    log_dir: Path | None = None,
) -> bool:
    """Record a failed scheduled portfolio tick without inventing metrics."""
    path = _matching_active(None, log_dir)
    if path is None:
        return False
    try:
        meta = read_meta(path)
        with _connect(path) as conn:
            _insert_event(
                conn,
                target_id=meta.get("target_id"),
                kind="portfolio_snapshot_error",
                payload={
                    "error_type": type(error).__name__,
                    "message": str(error)[:2_000],
                },
            )
        return True
    except Exception:
        logger.exception("trace portfolio error write failed (continuing)")
        return False


def expected_next_rebalance(log_dir: Path | None = None) -> datetime | None:
    value = read_meta(active_path(log_dir)).get("expected_next_rebalance")
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def panel_data(log_dir: Path | None = None) -> dict[str, Any]:
    path = active_path(log_dir)
    meta = read_meta(path)
    if not meta:
        return {
            "meta": {},
            "metrics": {},
            "assets": [],
            "status": {},
            "health": {},
        }
    with _connect(path) as conn:
        latest = conn.execute(
            "SELECT ts_utc FROM events WHERE kind='portfolio_snapshot' AND coin IS NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        metrics: dict[str, Any] = {}
        assets: list[dict[str, Any]] = []
        if latest:
            ts = latest[0]
            aggregate = conn.execute(
                "SELECT equity_usd, margin_used_usd, free_margin_usd, "
                "gross_exposure_usd, net_exposure_usd, unrealized_pnl_usd, payload_json "
                "FROM events WHERE kind='portfolio_snapshot' AND coin IS NULL "
                "AND ts_utc=? ORDER BY id DESC LIMIT 1",
                (ts,),
            ).fetchone()
            metrics = {
                "updated_at": ts,
                "equity_usd": aggregate[0],
                "margin_used_usd": aggregate[1],
                "free_margin_usd": aggregate[2],
                "gross_exposure_usd": aggregate[3],
                "net_exposure_usd": aggregate[4],
                "unrealized_pnl_usd": aggregate[5],
                **json.loads(aggregate[6]),
            }
            for row in conn.execute(
                "SELECT coin, mid_price, target_weight, position_size, position_usd, "
                "unrealized_pnl_usd, payload_json FROM events "
                "WHERE kind='portfolio_snapshot' AND coin IS NOT NULL AND ts_utc=? "
                "ORDER BY coin",
                (ts,),
            ):
                payload = json.loads(row[6])
                assets.append({
                    "coin": row[0],
                    "ticker": payload.get("ticker"),
                    "mid_price": row[1],
                    "target_weight": row[2],
                    "actual_weight": payload.get("actual_weight"),
                    "position_size": row[3],
                    "position_usd": row[4],
                    "unrealized_pnl_usd": row[5],
                })
        status_row = conn.execute(
            "SELECT kind, ts_utc, payload_json FROM events "
            "WHERE kind IN ('cost_gate_noop','execution_result') "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        status = (
            {"kind": status_row[0], "ts_utc": status_row[1], **json.loads(status_row[2])}
            if status_row
            else {}
        )
        health_row = conn.execute(
            "SELECT kind, ts_utc, payload_json FROM events "
            "WHERE kind='portfolio_snapshot_error' "
            "OR (kind='portfolio_snapshot' AND coin IS NULL) "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        health = (
            {"kind": health_row[0], "ts_utc": health_row[1], **json.loads(health_row[2])}
            if health_row
            else {}
        )
    return {
        "meta": meta,
        "metrics": metrics,
        "assets": assets,
        "status": status,
        "health": health,
    }
