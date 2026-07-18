"""Telegram notifications for model and execution updates.

Configuration (env vars, loadable from repo-root .env):
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHAT_ID    — target chat/channel id (e.g. "-1001234567890")

Unconfigured or failing notifications are logged and swallowed — a missing
Telegram channel must never block or fail an execution run.

Setup guide: causal_portfolio/docs/telegram_notifications.md
CLI:
  python -m causal_portfolio.execution.notify --test
  python -m causal_portfolio.execution.notify --message "hello"
  python -m causal_portfolio.execution.notify --state   # live account summary
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("cpcm.execution.notify")

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"
_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_LEN = 4096  # Telegram hard limit per message


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


def send(text: str) -> bool:
    """Send one Telegram message. Returns True on success. Never raises."""
    creds = _credentials()
    if creds is None:
        logger.debug("telegram not configured; dropping notification")
        return False
    token, chat_id = creds
    try:
        import requests

        resp = requests.post(
            _API_URL.format(token=token),
            json={"chat_id": chat_id, "text": text[:_MAX_LEN]},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("telegram send failed: HTTP %s %s",
                           resp.status_code, resp.text[:200])
            return False
        return True
    except Exception:
        logger.exception("telegram send failed")
        return False


def format_result(result: Any) -> str:
    """Human-readable summary of a SubmitResult for the channel."""
    plan = result.plan
    target_id = (
        plan.target_snapshot.target_id
        if plan.target_snapshot is not None
        else "unversioned"
    )
    gross = sum(abs(d) for d in plan.deltas_usd.values())
    lines = [
        f"CPCM execution [{plan.network or '?'}] target {target_id}",
        f"submitted={result.submitted} orders={len(plan.orders)} "
        f"gross=${gross:,.0f} equity=${plan.current_state.account_value_usd:,.0f}",
    ]
    repair = getattr(result, "repair", None)
    if repair:
        lines.append(
            f"leg repair: {len(repair.get('attempts', []))} attempt(s), "
            f"resolved={repair.get('resolved')}"
        )
    if result.drifts:
        lines.append("UNRESOLVED LEG FAILURES:")
        for d in result.drifts:
            lines.append(
                f"  {d.coin}: target ${d.target_usd:,.2f} actual ${d.actual_usd:,.2f} "
                f"(drift ${d.drift_usd:+,.2f})"
            )
    for label, err in (
        ("error", result.error),
        ("post_submit_error", result.post_submit_error),
        ("audit_error", result.audit_error),
    ):
        if err:
            lines.append(f"{label}: {err}")
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
    )
    if not noteworthy:
        return False
    return send(format_result(result))


def format_state(state: Any) -> str:
    lines = [
        f"CPCM portfolio {state.address[:8]}...",
        f"equity=${state.account_value_usd:,.2f} "
        f"margin_used=${state.margin_used_usd:,.2f}",
    ]
    for coin, pos in sorted(state.positions.items()):
        lines.append(f"  {coin}: {pos.size:+g} (${pos.notional_usd:+,.2f})")
    if not state.positions:
        lines.append("  no open positions")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Send a Telegram notification")
    p.add_argument("--test", action="store_true", help="Send a test ping")
    p.add_argument("--message", help="Send an arbitrary message")
    p.add_argument("--state", action="store_true",
                   help="Send current HL account state (uses HL_* env creds)")
    p.add_argument("--mainnet", action="store_true",
                   help="Read state from mainnet (read-only; default testnet)")
    args = p.parse_args(argv)

    if not is_configured():
        print(f"Not configured: set {TOKEN_ENV} and {CHAT_ENV} (see "
              "causal_portfolio/docs/telegram_notifications.md)")
        return 1

    if args.state:
        from causal_portfolio.execution.config import ExecutionConfig
        from causal_portfolio.execution.hyperliquid import HLAdapter

        adapter = HLAdapter(ExecutionConfig(testnet=not args.mainnet))
        ok = send(format_state(adapter.fetch_state()))
    elif args.message:
        ok = send(args.message)
    else:
        ok = send("CPCM notification test ping")
    print("sent" if ok else "send failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
