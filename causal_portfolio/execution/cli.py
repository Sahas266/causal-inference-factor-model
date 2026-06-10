"""Command-line interface for the execution layer.

Usage:
    # Dry-run: load weights from JSON and print the plan (no network writes)
    python -m causal_portfolio.execution.cli plan --weights weights.json

    # Pull live state from testnet, dry-run
    python -m causal_portfolio.execution.cli plan --weights weights.json --live-state

    # Actually submit on testnet
    python -m causal_portfolio.execution.cli execute --weights weights.json --live --testnet

    # Mainnet (requires explicit --mainnet AND a confirmation prompt)
    python -m causal_portfolio.execution.cli execute --weights weights.json --live --mainnet

JSON file format:
    {
      "btc": 0.30,
      "eth": 0.25,
      "sol": -0.10,
      ...
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from datetime import datetime, timezone

from causal_portfolio.execution.audit import LOG_DIR, read_log
from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.rebalancer import plan_rebalance
from causal_portfolio.execution.types import AccountState, AssetMeta

logger = logging.getLogger("cpcm.execution.cli")


def _load_weights(path: str) -> dict[str, float]:
    """Load and validate a weights JSON file.

    Keys prefixed with `_` are treated as metadata and ignored. The CPCM
    pipeline writes a `_meta` field with provenance info (solver, date,
    selected drivers, etc.) that the executor doesn't need.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Weights file must be a JSON object, got {type(raw).__name__}")
    for k, v in raw.items():
        if k.startswith("_"):
            continue  # metadata, skip
        if not isinstance(v, (int, float)):
            raise ValueError(f"Weight for {k!r} must be numeric, got {type(v).__name__}")
    return {k.lower(): float(v) for k, v in raw.items() if not k.startswith("_")}


def _stub_state_and_market(equity: float = 10_000.0) -> tuple[AccountState, dict, dict]:
    """Synthetic state for offline planning. Used when --live-state is not set."""
    state = AccountState(
        address="0xstub",
        account_value_usd=equity,
        margin_used_usd=0.0,
    )
    # Minimal stub mids/meta for the default asset map. Real values come from
    # the adapter when --live-state is set.
    from causal_portfolio.execution.config import DEFAULT_ASSET_MAP
    stub_mid = 100.0
    mids = {hl: stub_mid for hl in DEFAULT_ASSET_MAP.values()}
    meta = {hl: AssetMeta(coin=hl, sz_decimals=4, max_leverage=10, min_size=1e-4)
            for hl in DEFAULT_ASSET_MAP.values()}
    return state, mids, meta


def _print_plan(plan, verbose: bool = True) -> None:
    print(plan.summary())
    if not verbose:
        return
    if plan.notes:
        print("\nNotes:")
        for n in plan.notes:
            print(f"  - {n}")
    if plan.orders:
        print("\nOrders:")
        print(f"  {'Coin':<8} {'Side':<5} {'Size':>14} {'Limit Px':>12} {'Notional':>14}")
        for o in plan.orders:
            side = "BUY" if o.is_buy else "SELL"
            notional = o.size * o.limit_px
            print(f"  {o.coin:<8} {side:<5} {o.size:>14.6f} {o.limit_px:>12.4f} ${notional:>13,.2f}")
    if plan.skipped:
        print("\nSkipped:")
        for coin, reason, detail in plan.skipped:
            print(f"  {coin:<8} {reason.value:<30} {detail}")


def _confirm_mainnet(plan) -> bool:
    print("\n" + "=" * 60)
    print("LIVE MAINNET TRADING — this will submit real orders")
    print(f"  Account: {plan.current_state.address}")
    print(f"  Equity:  ${plan.current_state.account_value_usd:,.2f}")
    print(f"  Orders:  {len(plan.orders)}")
    gross = sum(abs(d) for d in plan.deltas_usd.values())
    print(f"  Gross:   ${gross:,.2f}")
    print("=" * 60)
    response = input("Type 'EXECUTE' to confirm: ").strip()
    return response == "EXECUTE"


def cmd_plan(args) -> int:
    weights = _load_weights(args.weights)
    cfg = ExecutionConfig(testnet=not args.mainnet, dry_run=True)

    if args.live_state:
        from causal_portfolio.execution.hyperliquid import HLAdapter
        adapter = HLAdapter(cfg)
        state = adapter.fetch_state()
        mids = adapter.fetch_mids()
        meta = adapter.fetch_meta()
    else:
        state, mids, meta = _stub_state_and_market(equity=args.equity)

    plan = plan_rebalance(weights, state, mids, meta, cfg)
    _print_plan(plan, verbose=args.verbose)
    return 0


def cmd_execute(args) -> int:
    if not args.live:
        print("Refusing to submit without --live. Use 'plan' for dry-runs.", file=sys.stderr)
        return 1

    weights = _load_weights(args.weights)
    cfg_kwargs = dict(testnet=not args.mainnet, dry_run=False)
    if args.slippage_bps is not None:
        cfg_kwargs["slippage_bps"] = args.slippage_bps
    cfg = ExecutionConfig(**cfg_kwargs)

    from causal_portfolio.execution.hyperliquid import HLAdapter, execute_plan
    adapter = HLAdapter(cfg)

    state = adapter.fetch_state()
    mids = adapter.fetch_mids()
    meta = adapter.fetch_meta()
    plan = plan_rebalance(weights, state, mids, meta, cfg)
    _print_plan(plan, verbose=True)

    if not plan.orders:
        print("\nNo orders to submit.")
        return 0

    if args.mainnet and not _confirm_mainnet(plan):
        print("Mainnet execution canceled.")
        return 1

    # execute_plan handles audit appending internally now.
    result = execute_plan(adapter, plan, acknowledge_mainnet=args.mainnet)

    if result.error:
        print(f"\nERROR: {result.error}", file=sys.stderr)
        return 2
    print(f"\nSubmitted. Response: {result.response}")
    return 0


def cmd_state(args) -> int:
    """Pretty-print the current HL account state (read-only, no orders)."""
    from causal_portfolio.execution.hyperliquid import HLAdapter

    cfg = ExecutionConfig(testnet=not args.mainnet, dry_run=True)
    adapter = HLAdapter(cfg)

    state = adapter.fetch_state()
    network = "mainnet" if args.mainnet else "testnet"
    print(f"Hyperliquid {network} state — {state.address}")
    print("=" * 70)
    print(f"  equity:        ${state.account_value_usd:>12,.2f}")
    print(f"  margin used:   ${state.margin_used_usd:>12,.2f}")
    print(f"  free margin:   ${state.free_margin_usd:>12,.2f}")
    print(f"  positions:     {len(state.positions)}")

    if state.positions:
        try:
            mids = adapter.fetch_mids()
        except Exception:
            mids = {}
        print()
        print(f"  {'Coin':<10} {'Size':>14} {'Entry':>12} {'Mark':>12} {'Notional':>14}  Side")
        for coin, p in sorted(state.positions.items()):
            mark = mids.get(coin, p.entry_px)
            side = "LONG " if p.is_long else "SHORT"
            print(
                f"  {coin:<10} {p.size:>+14.6f} {p.entry_px:>12.4f} "
                f"{mark:>12.4f} ${p.notional_usd:>+13,.2f}  {side}"
            )

    if args.verbose:
        try:
            open_oids = adapter.fetch_open_order_ids()
            print()
            print(f"  open orders: {len(open_oids)}")
        except Exception as e:
            print(f"  (couldn't fetch open orders: {e})")

    return 0


def cmd_logs(args) -> int:
    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records = read_log(date)
    if not records:
        print(f"No audit records for {date} (looked in {LOG_DIR}/rebalance-{date}.jsonl).")
        return 0

    print(f"Audit log for {date} — {len(records)} record(s)")
    print("=" * 78)
    for i, rec in enumerate(records, 1):
        plan = rec.get("plan") or {}
        n_orders = len(plan.get("orders") or [])
        n_skipped = len(plan.get("skipped") or [])
        gross = sum(abs(v) for v in (plan.get("deltas_usd") or {}).values())
        equity = (plan.get("current_state") or {}).get("account_value_usd", 0)
        status = "SUBMITTED" if rec.get("submitted") else "DRY-RUN" if rec.get("error") is None else "ERROR"
        if rec.get("error"):
            status = "ERROR"
        print(f"\n[{i}] {rec.get('ts_utc', '?')}  status={status}")
        print(f"     equity=${equity:,.2f}  orders={n_orders}  skipped={n_skipped}  gross=${gross:,.2f}")
        if rec.get("error"):
            print(f"     error: {rec['error']}")
        if args.verbose:
            for o in plan.get("orders") or []:
                side = "BUY " if o.get("is_buy") else "SELL"
                notional = (o.get("size") or 0) * (o.get("limit_px") or 0)
                print(f"     - {o.get('coin'):<8} {side} sz={o.get('size'):<12} px={o.get('limit_px'):<12} ${notional:,.2f}")
            for s in plan.get("skipped") or []:
                # skipped is serialized as a list of [coin, reason, detail]
                if isinstance(s, list) and len(s) >= 3:
                    print(f"     skip {s[0]:<8} {s[1]:<28} {s[2]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="causal_portfolio.execution.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Compute and print a rebalance plan (dry-run)")
    p_plan.add_argument("--weights", required=True, help="Path to weights JSON file")
    p_plan.add_argument("--live-state", action="store_true",
                        help="Pull current state/mids/meta from Hyperliquid (read-only)")
    p_plan.add_argument("--mainnet", action="store_true", help="Use mainnet for state read")
    p_plan.add_argument("--equity", type=float, default=10_000.0,
                        help="Stub equity for offline planning (default: 10000)")
    p_plan.add_argument("--verbose", "-v", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_state = sub.add_parser("state", help="Pretty-print the current HL account state")
    p_state.add_argument("--mainnet", action="store_true", help="Read from mainnet (default: testnet)")
    p_state.add_argument("--verbose", "-v", action="store_true",
                         help="Also fetch open order count")
    p_state.set_defaults(func=cmd_state)

    p_logs = sub.add_parser("logs", help="Pretty-print the audit log for a date")
    p_logs.add_argument("--date", help="UTC date YYYY-MM-DD (default: today)")
    p_logs.add_argument("--verbose", "-v", action="store_true",
                        help="Print every order and skip reason per record")
    p_logs.set_defaults(func=cmd_logs)

    p_exec = sub.add_parser("execute", help="Submit a rebalance plan to Hyperliquid")
    p_exec.add_argument("--weights", required=True)
    p_exec.add_argument("--live", action="store_true", required=True,
                        help="Required to actually submit (safety gate)")
    p_exec.add_argument("--mainnet", action="store_true",
                        help="Hit mainnet instead of testnet (requires confirmation)")
    p_exec.add_argument("--slippage-bps", type=int, default=None,
                        help="Override ExecutionConfig.slippage_bps for this run. "
                             "HL may reject orders priced too far from oracle; "
                             "10 bps is a safe default for liquid perps.")
    p_exec.set_defaults(func=cmd_execute)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
