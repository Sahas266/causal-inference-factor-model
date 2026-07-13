# Hyperliquid Execution Layer

Translates a CPCM weight vector into orders on Hyperliquid, with safety rails.

## Architecture

```
target weights (w)
        │
        ▼
  rebalancer.py  ◄── PURE: no network, no SDK. Fully unit-tested.
        │              Inputs: weights + state + mids + meta + config
        ▼              Output: RebalancePlan (orders + skip list)
  RebalancePlan
        │
        ▼
  hyperliquid.py ◄── Thin SDK wrapper. Reads state, submits book-aware IOC orders.
        │              Lazy-imports hyperliquid-python-sdk.
        ▼
  Hyperliquid (testnet by default)
        │
        ▼
  audit.py logs every rebalance to logs/rebalance-YYYY-MM-DD.jsonl
```

## Quick start

```bash
# 1. Write a weights file (fine for planning; live execution requires a date)
echo '{"btc": 0.3, "eth": 0.2, "sol": -0.1}' > weights.json

# 2. Dry-run (no network) — print the plan
python -m causal_portfolio.execution.cli plan --weights weights.json --equity 100000

# 3. Pull live state from testnet, dry-run
export HL_ADDRESS=0x...
python -m causal_portfolio.execution.cli plan --weights weights.json --live-state

# 4. Submit on testnet
export HL_PRIVATE_KEY=0x...
python -m causal_portfolio.execution.cli execute --weights weights.json --live --testnet

# 5. Mainnet (requires confirmation prompt)
python -m causal_portfolio.execution.cli execute --weights weights.json --live --mainnet

# Optional: slice a live rebalance into deterministic child IOC batches
python -m causal_portfolio.execution.cli execute --weights weights.json --live --testnet --twap-minutes 10 --twap-slices 5
```

## Model-independent logging

The execution CLI automatically writes a unique text log to
`causal_portfolio/execution/logs/execution-cli-*.log`. It captures all logger
output emitted during the CLI run plus failure tracebacks. Every `execute_plan()`
call attempts to append a structured JSONL audit record by default, containing
the target, planned orders, exchange response, post-trade state, reconciliation
drift, and any post-submit error. An append failure is returned as
`SubmitResult.audit_error` and logged in the text run log.

`SubmitResult.submitted=True` remains authoritative when post-submit state fetch,
reconciliation, or audit logging fails. `post_submit_error` and `audit_error`
require operator attention, but neither makes the completed trade safe to retry.

New models can produce the standard dated JSON/CSV target and invoke the CLI.
Models calling the execution API directly use the public model handoff:

```python
from datetime import datetime, timezone

from causal_portfolio.execution import ExecutionConfig, TargetSnapshot, execute_target

target = TargetSnapshot(
    weights={"btc": 0.3, "eth": 0.2},
    as_of=datetime.now(timezone.utc),
    strategy="my-model",
)
result = execute_target(target, ExecutionConfig())  # dry-run testnet by default
```

JSON/CSV producers call `load_target_snapshot()` first, then pass the returned
`TargetSnapshot` to `execute_target()`.

## Daily local RP-PCA runner

Generate an RP-PCA target from the local DuckDB snapshot or a wide price CSV:

```bash
python -m causal_portfolio.execution.rppca_daily --target-out tmp/rppca_daily_target.json
```

Run locally every 24 hours and execute on Hyperliquid testnet:

```bash
python -m causal_portfolio.execution.rppca_daily --loop --every-hours 24 --execute --target-out tmp/rppca_daily_target.json
```

Mainnet is non-interactive for scheduling, so it requires both explicit flags:

```bash
python -m causal_portfolio.execution.rppca_daily --loop --execute --mainnet --ack-mainnet
```

The runner forward-fills local daily prices but stamps the target with the
latest real data date. If the warehouse is stale, execution is blocked by the
normal `max_signal_age_hours` check unless `--allow-stale-signal` is passed.

For live submission, use a model-produced JSON containing `_meta.as_of` or
`_meta.rebalance_date`. The CLI also accepts the RP-PCA-style asset-weight CSV
contract: `rebalance_date` plus one column per asset. When a CSV contains a
rebalance history, the executor selects the latest dated row. Targets older
than 72 hours, future-dated targets, and targets without a date are blocked;
`--allow-stale-signal` is an explicit operator override.
Programmatic live execution follows the same rule: pass a `TargetSnapshot`
from `load_target_snapshot()`. Bare dict targets are accepted for dry-run
planning, but live writes reject them unless `allow_stale_signal=True`.

## Required environment variables for network calls

- `HL_ADDRESS` — your Hyperliquid wallet address (read + write)
- `HL_PRIVATE_KEY` — only needed for write operations (cancel, submit)

Reads work without a private key; writes require it. Never commit either to a `.env` checked into the repo.

## Optional dependency

The network adapter requires `hyperliquid-python-sdk`:

```bash
pip install "hyperliquid-python-sdk==0.24.0" "eth-account==0.13.7"
```

Not in `requirements.txt` because the pure rebalancer is useful without it (planning, testing, paper-trading). Only install when you're ready to hit testnet.

## Safety defaults

| Knob | Default | Rationale |
|---|---|---|
| `dry_run` | True | Must explicitly opt-in to live trading |
| `testnet` | True | Mainnet only via explicit flag |
| `leverage` | 1.0 | No implicit leverage |
| `max_position_pct` | 0.30 | Cap any single asset at 30% of equity |
| `max_single_trade_pct` | 0.10 | One trade can't move more than 10% of equity |
| `min_order_notional_usd` | $10 | Surface sub-minimum sleeves before exchange reject |
| `slippage_bps` | 30 | IOC limit at mid ± 30bps |
| `max_signal_age_hours` | 72 | Block stale model targets before live writes |
| `smart_execution` | True | Reprice IOC slices against the live L2 book |
| `twap_minutes` | 0 | Disabled unless an operator requests client-side slicing |
| `max_twap_minutes` | 30 | Cap synchronous client-side TWAP windows |

To override, construct `ExecutionConfig` directly in Python or edit defaults in `config.py`.
Every plan is also tagged `testnet` or `mainnet`; execution refuses a plan
built for the other network.
Target exposures are pre-filtered against each market's Hyperliquid
`max_leverage`; same-side reduce-only orders are still allowed when they lower
an already over-limit position.
Non-reducing orders below `min_order_notional_usd` are skipped in the dry-run
plan; full reduce-only closes are allowed so cleanup can flatten tiny residuals.

## Audit log format

Each rebalance appends one JSON record to `logs/rebalance-YYYY-MM-DD.jsonl`:

```json
{
  "ts_utc": "2026-05-04T13:48:51.547000+00:00",
  "network": "testnet",
  "target_id": "9f51a4bd24d7c2b1",
  "submitted": true,
  "error": null,
  "post_submit_error": null,
  "plan": {
    "target_weights": {"btc": 0.3, "eth": 0.2},
    "orders": [...],
    "skipped": [...]
  },
  "response": {...},
  "post_state": {...}
}
```

Never auto-rotated. Debugging old fills requires the raw record.

## Known limitations / future work

- **No iceberg execution.** TWAP is client-side time slicing only; it does not
  use hidden liquidity or exchange-native algos.
- **No fill-aware intra-TWAP replanning.** Child slices use the original target
  and current L2 book. Re-run `plan` after a partial execution if the market or
  account state changed materially.
- **TWAP is not crash-atomic.** Normal errors are audited, but a hard process
  crash during the sleep/submission window may leave partial real fills before
  the final audit record is written.

## What this layer does NOT do

- Iceberg / hidden-liquidity execution
- Funding rate optimization
- Stop-loss / take-profit overlays
- Cross-exchange routing
- Backtest paper-trading sink (could reuse `rebalancer.py` later)

## Tests

```bash
# Pure logic — no SDK, no network
python -m pytest causal_portfolio/tests/test_rebalancer.py -v

# Audit + CLI offline paths
python -m pytest causal_portfolio/tests/test_execution_audit.py -v
```
