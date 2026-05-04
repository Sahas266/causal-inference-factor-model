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
  hyperliquid.py ◄── Thin SDK wrapper. Reads state, submits bulk_orders.
        │              Lazy-imports hyperliquid-python-sdk.
        ▼
  Hyperliquid (testnet by default)
        │
        ▼
  audit.py logs every rebalance to logs/rebalance-YYYY-MM-DD.jsonl
```

## Quick start

```bash
# 1. Write a weights file
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
```

## Required environment variables for network calls

- `HL_ADDRESS` — your Hyperliquid wallet address (read + write)
- `HL_PRIVATE_KEY` — only needed for write operations (cancel, submit)

Reads work without a private key; writes require it. Never commit either to a `.env` checked into the repo.

## Optional dependency

The network adapter requires `hyperliquid-python-sdk`:

```bash
pip install hyperliquid-python-sdk eth-account
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
| `min_trade_usd` | $25 | Filter dust below this |
| `slippage_bps` | 30 | IOC limit at mid ± 30bps |

To override, construct `ExecutionConfig` directly in Python or edit defaults in `config.py`.

## Audit log format

Each rebalance appends one JSON record to `logs/rebalance-YYYY-MM-DD.jsonl`:

```json
{
  "ts_utc": "2026-05-04T13:48:51.547000+00:00",
  "submitted": true,
  "error": null,
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

## What this layer does NOT do

- TWAP / iceberg execution (single batch, all-or-nothing)
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
