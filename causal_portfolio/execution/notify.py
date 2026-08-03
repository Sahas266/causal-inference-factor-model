"""Telegram notifications for model and execution updates.

Configuration (env vars, loadable from a cwd/parent .env):
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHAT_ID    — target chat/channel id (e.g. "-1001234567890")

Unconfigured or failing notifications are logged and swallowed, so Telegram
failures never change the execution result. HTTP delay is capped at 15 seconds.

Formatted messages (execution results, portfolio state, PnL) use Telegram's
HTML parse mode for bold/monospace/emoji; ad hoc --test/--message text is
sent plain so arbitrary operator text is never mangled by markup parsing.

Setup guide: causal_portfolio/docs/telegram_notifications.md
CLI:
  python -m causal_portfolio.execution.notify --test
  python -m causal_portfolio.execution.notify --message "hello"
  python -m causal_portfolio.execution.notify --state              # live account summary
  python -m causal_portfolio.execution.notify --pnl                # live unrealized PnL, once
  python -m causal_portfolio.execution.notify --pnl-loop --interval-minutes 30
"""

from __future__ import annotations

import html
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("cpcm.execution.notify")

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"
_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_LEN = 4096  # Telegram hard limit per message
_HTML_TAGS = re.compile(r"</?(?:b|code)>")


def _credentials() -> tuple[str, str] | None:
    # Reuse the adapter's dotenv bootstrap so .env works for notify-only runs.
    from causal_portfolio.execution.hyperliquid import _load_dotenv_once

    _load_dotenv_once()
    token = os.environ.get(TOKEN_ENV, "").strip()
    chat_id = os.environ.get(CHAT_ENV, "").strip()
    if token and chat_id:
        return token, chat_id
    return None


def is_configured() -> bool:
    return _credentials() is not None


def _esc(value: Any) -> str:
    """Escape a dynamic value for interpolation into an HTML-mode message."""
    return html.escape(str(value))


def _has_balanced_html(text: str) -> bool:
    """Return whether supported Telegram tags are balanced and correctly nested."""
    stack: list[str] = []
    for match in _HTML_TAGS.finditer(text):
        token = match.group(0)
        if token.startswith("</"):
            tag = token[2:-1]
            if not stack or stack.pop() != tag:
                return False
        else:
            stack.append(token[1:-1])
    return not stack


def _message_chunks(text: str, parse_mode: str | None) -> tuple[list[str], str | None]:
    """Split at safe boundaries instead of truncating Telegram HTML."""
    if len(text) <= _MAX_LEN:
        return [text], parse_mode
    if parse_mode == "HTML":
        lines = text.splitlines(keepends=True)
        if any(len(line) > _MAX_LEN for line in lines):
            # Internal formatted messages use only these two tags. A single
            # pathological line is safer as plain text than broken HTML.
            text = html.unescape(_HTML_TAGS.sub("", text))
            parse_mode = None
        else:
            chunks: list[str] = []
            current = ""
            for line in lines:
                if current and len(current) + len(line) > _MAX_LEN:
                    chunks.append(current)
                    current = ""
                current += line
            if current:
                chunks.append(current)
            if all(_has_balanced_html(chunk) for chunk in chunks):
                return chunks, parse_mode
            # A tag spans a chunk boundary. Send the complete content as plain
            # text so Telegram cannot reject a malformed HTML fragment.
            text = html.unescape(_HTML_TAGS.sub("", text))
            parse_mode = None
    return [text[i:i + _MAX_LEN] for i in range(0, len(text), _MAX_LEN)], parse_mode


def send(text: str, *, parse_mode: str | None = None) -> bool:
    """Send one Telegram message. Returns True on success. Never raises."""
    creds = _credentials()
    if creds is None:
        logger.debug("telegram not configured; dropping notification")
        return False
    token, chat_id = creds
    try:
        import requests

        chunks, effective_mode = _message_chunks(text, parse_mode)
        for chunk in chunks:
            payload: dict[str, Any] = {"chat_id": chat_id, "text": chunk}
            if effective_mode:
                payload["parse_mode"] = effective_mode
            resp = requests.post(
                _API_URL.format(token=token),
                json=payload,
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning("telegram send failed: HTTP %s %s",
                               resp.status_code, resp.text[:200])
                return False
        return True
    except Exception as e:
        # Request errors can include the token-bearing URL in their message.
        logger.warning("telegram send failed: %s", type(e).__name__)
        return False


def _status_emoji(result: Any) -> str:
    if result.error:
        return "🛑"
    if result.cost_gate_reason:
        return "⚠️" if result.submitted else "⏸️"
    if result.drifts or (result.repair and not result.repair.get("resolved", True)):
        return "⚠️"
    if result.submitted:
        return "✅"
    return "💤"  # dry-run / no-op


def format_next_rebalance(
    next_rebalance: datetime | None,
    *,
    now: datetime | None = None,
) -> str:
    """HTML-safe countdown line for Telegram summaries."""
    if next_rebalance is None:
        return "Next rebalance: <b>unknown</b>"
    if next_rebalance.tzinfo is None:
        next_rebalance = next_rebalance.replace(tzinfo=timezone.utc)
    next_rebalance = next_rebalance.astimezone(timezone.utc)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    label = next_rebalance.strftime("%Y-%m-%d %H:%M UTC")
    seconds = (next_rebalance - current).total_seconds()
    if seconds <= 0:
        return f"Next rebalance: <b>OVERDUE</b> ({label})"
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    duration = f"{hours}h {minutes}m" if hours else f"{minutes}m"
    return f"Next rebalance: <b>{duration}</b> ({label})"


def format_result(result: Any) -> str:
    """HTML-formatted summary of a SubmitResult for the channel."""
    plan = result.plan
    target_id = (
        plan.target_snapshot.target_id
        if plan.target_snapshot is not None
        else "unversioned"
    )
    gross = sum(abs(d) for d in plan.deltas_usd.values())
    submitted_orders = getattr(result, "submitted_orders", None)
    submitted_count = (
        len(submitted_orders)
        if submitted_orders is not None
        else len(plan.orders) if result.submitted else 0
    )
    lines = [
        f"{_status_emoji(result)} <b>CPCM Execution</b> — {_esc(plan.network or '?')}",
        f"Target <code>{_esc(target_id)}</code>",
        f"Submitted: <b>{result.submitted}</b> · Orders: "
        f"<b>{submitted_count} of {len(plan.orders)} submitted</b>",
        f"Gross: <b>${gross:,.0f}</b> · Equity: <b>${plan.current_state.account_value_usd:,.0f}</b>",
    ]
    if result.cost_estimate:
        estimate = result.cost_estimate
        measured = estimate.get("estimated_cost_bps")
        limit = estimate.get("max_transaction_cost_bps")
        if measured is not None:
            lines.append(
                f"Estimated cost: <b>{measured:.2f} bps</b>"
                + (f" / {_esc(limit)} bps max" if limit is not None else "")
            )
    if result.cost_gate_reason:
        lines.append(f"⏸ <b>Cost gate:</b> {_esc(result.cost_gate_reason)}")
    repair = getattr(result, "repair", None)
    if repair:
        r_emoji = "✅" if repair.get("resolved") else "⚠️"
        lines.append(
            f"{r_emoji} Leg repair: {len(repair.get('attempts', []))} attempt(s), "
            f"resolved=<b>{repair.get('resolved')}</b>"
        )
    if result.drifts:
        lines.append("")
        lines.append("🚨 <b>Unresolved leg failures</b>")
        for d in result.drifts:
            lines.append(
                f"  • {_esc(d.coin)}: target ${d.target_usd:,.2f} → actual "
                f"${d.actual_usd:,.2f} (Δ ${d.drift_usd:+,.2f})"
            )
    for label, err in (
        ("Error", result.error),
        ("Post-submit error", result.post_submit_error),
        ("Audit error", result.audit_error),
    ):
        if err:
            lines.append(f"🛑 <b>{label}:</b> {_esc(err)}")
    return "\n".join(lines)


def notify_result(result: Any) -> bool:
    """Notify the channel about an execution result.

    Sends only for real submissions or failures — silent for clean dry-runs.
    """
    noteworthy = (
        result.submitted
        or result.error
        or result.post_submit_error
        or result.audit_error
        or result.cost_gate_reason
    )
    if not noteworthy:
        return False
    return send(format_result(result), parse_mode="HTML")


def format_state(state: Any) -> str:
    """HTML-formatted account snapshot: equity, margin, open positions."""
    lines = [
        f"📊 <b>CPCM Portfolio</b> — <code>{_esc(state.address[:8])}...</code>",
        f"Equity: <b>${state.account_value_usd:,.2f}</b> · "
        f"Margin used: <b>${state.margin_used_usd:,.2f}</b>",
    ]
    if not state.positions:
        lines.append("")
        lines.append("No open positions")
        return "\n".join(lines)
    lines.append("")
    for coin, pos in sorted(state.positions.items()):
        dir_emoji = "🟢" if pos.size > 0 else "🔴"
        lines.append(
            f"{dir_emoji} <b>{_esc(coin)}</b>: {pos.size:+g} "
            f"(${pos.notional_usd:+,.2f})"
        )
    return "\n".join(lines)


def format_pnl(
    state: Any,
    mids: dict[str, float],
    *,
    next_rebalance: datetime | None = None,
    now: datetime | None = None,
) -> str:
    """HTML-formatted unrealized PnL per position, marked to `mids`.

    PnL for coin c = (mid - entry_px) * size — valid for both longs and
    shorts since `size` carries the sign. `mids` typically comes from
    HLAdapter.fetch_mids(); a coin missing from it is reported, not guessed.
    """
    lines = [
        f"💰 <b>CPCM PnL Update</b> — <code>{_esc(state.address[:8])}...</code>",
        f"Equity: <b>${state.account_value_usd:,.2f}</b>",
        format_next_rebalance(next_rebalance, now=now),
    ]
    if not state.positions:
        lines.append("")
        lines.append("No open positions")
        return "\n".join(lines)

    lines.append("")
    total_pnl = 0.0
    missing_mid_coins: list[str] = []
    for coin, pos in sorted(state.positions.items()):
        mid = mids.get(coin)
        if mid is None:
            missing_mid_coins.append(coin)
            lines.append(f"⚪ <b>{_esc(coin)}</b>: mid price unavailable")
            continue
        pnl = (mid - pos.entry_px) * pos.size
        total_pnl += pnl
        basis = abs(pos.size) * pos.entry_px
        pnl_pct = (pnl / basis * 100.0) if basis > 0 else 0.0
        emoji = "🟢" if pnl >= 0 else "🔴"
        direction = "long" if pos.size > 0 else "short"
        lines.append(
            f"{emoji} <b>{_esc(coin)}</b> ({direction}): "
            f"${pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )
    lines.append("")
    if missing_mid_coins:
        lines.append(
            "⚠️ <b>Total unrealized PnL unavailable</b> · missing mids: "
            + ", ".join(_esc(coin) for coin in missing_mid_coins)
        )
        if len(missing_mid_coins) < len(state.positions):
            lines.append(f"Priced positions subtotal: <b>${total_pnl:+,.2f}</b>")
    else:
        total_emoji = "🟢" if total_pnl >= 0 else "🔴"
        lines.append(f"{total_emoji} <b>Total unrealized PnL: ${total_pnl:+,.2f}</b>")
    return "\n".join(lines)


def _fetch_pnl_snapshot(mainnet: bool) -> tuple[Any, dict[str, float]]:
    from causal_portfolio.execution.config import ExecutionConfig
    from causal_portfolio.execution.hyperliquid import HLAdapter

    adapter = HLAdapter(ExecutionConfig(testnet=not mainnet))
    return adapter.fetch_state(), adapter.fetch_mids()


def send_pnl_update(*, mainnet: bool = False) -> bool:
    """Fetch live state + mids and send one formatted PnL message."""
    try:
        state, mids = _fetch_pnl_snapshot(mainnet)
    except Exception as error:
        try:
            from causal_portfolio.execution import trace

            trace.record_portfolio_error(error)
        except Exception:
            logger.exception("portfolio failure trace write failed (continuing)")
        try:
            from causal_portfolio.execution.control_panel import write_control_panel

            write_control_panel(None, {})
        except Exception:
            logger.exception("control-panel failure update failed (continuing)")
        raise
    next_rebalance = None
    try:
        from causal_portfolio.execution import trace

        trace.record_portfolio_snapshot(state, mids)
        next_rebalance = trace.expected_next_rebalance()
    except Exception:
        logger.exception("portfolio trace write failed (continuing)")
    try:
        from causal_portfolio.execution.control_panel import write_control_panel

        write_control_panel(state, mids)
    except Exception:
        logger.exception("control-panel write failed (continuing)")
    return send(
        format_pnl(state, mids, next_rebalance=next_rebalance),
        parse_mode="HTML",
    )


def run_pnl_loop(*, interval_minutes: float, mainnet: bool = False) -> None:
    """Send a PnL update every `interval_minutes`, forever.

    # ponytail: single-process sleep loop, not a scheduler — restart on crash
    # is the operator's job (Task Scheduler / systemd), same as rppca_daily.
    """
    interval_seconds = max(1.0, interval_minutes) * 60.0
    logger.info("starting PnL loop: every %.1f minute(s)", interval_minutes)
    while True:
        try:
            ok = send_pnl_update(mainnet=mainnet)
            logger.info("pnl update %s", "sent" if ok else "send failed")
        except Exception:
            logger.exception("pnl loop iteration failed (continuing)")
        time.sleep(interval_seconds)


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Send a Telegram notification")
    p.add_argument("--test", action="store_true", help="Send a test ping")
    p.add_argument("--message", help="Send an arbitrary message")
    p.add_argument("--state", action="store_true",
                   help="Send current HL account state (uses HL_* env creds)")
    p.add_argument("--pnl", action="store_true",
                   help="Send current unrealized PnL, once")
    p.add_argument("--pnl-loop", action="store_true",
                   help="Send unrealized PnL every --interval-minutes, forever")
    p.add_argument("--interval-minutes", type=float, default=30.0,
                   help="PnL loop interval (default 30)")
    p.add_argument("--mainnet", action="store_true",
                   help="Read state from mainnet (read-only; default testnet)")
    args = p.parse_args(argv)

    if args.pnl_loop:
        try:
            run_pnl_loop(interval_minutes=args.interval_minutes, mainnet=args.mainnet)
        except KeyboardInterrupt:
            return 0
        return 0  # unreachable; run_pnl_loop only exits via KeyboardInterrupt

    if args.pnl or args.state:
        # The exchange is unreachable often enough (DNS blocks, captive
        # portals, HL maintenance) that a scheduled 30-minute run must not
        # spew a traceback per tick. Report one concise line and exit
        # non-zero so Task Scheduler still records the failure.
        try:
            if args.pnl:
                ok = send_pnl_update(mainnet=args.mainnet)
            else:
                from causal_portfolio.execution.config import ExecutionConfig
                from causal_portfolio.execution.hyperliquid import HLAdapter

                adapter = HLAdapter(ExecutionConfig(testnet=not args.mainnet))
                ok = send(format_state(adapter.fetch_state()), parse_mode="HTML")
        except Exception as e:
            logger.warning("exchange unreachable: %s: %s", type(e).__name__, e)
            print(f"exchange unreachable: {type(e).__name__}")
            return 1
    elif args.message:
        if not is_configured():
            print(f"Not configured: set {TOKEN_ENV} and {CHAT_ENV} (see "
                  "causal_portfolio/docs/telegram_notifications.md)")
            return 1
        ok = send(args.message)
    else:
        if not is_configured():
            print(f"Not configured: set {TOKEN_ENV} and {CHAT_ENV} (see "
                  "causal_portfolio/docs/telegram_notifications.md)")
            return 1
        ok = send("CPCM notification test ping")
    print("sent" if ok else "send failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
