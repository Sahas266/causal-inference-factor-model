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
  trace.py keeps one active SQLite cycle and permanent completed archives
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

# Apply the model-agnostic all-in transaction-cost gate
python -m causal_portfolio.execution.cli execute --weights weights.json --live --testnet --max-transaction-cost-bps 15

# Also suppress tiny resizes and refuse materially incomplete increasing plans
python -m causal_portfolio.execution.cli execute --weights weights.json --live --testnet --min-position-change-pct 0.10 --min-rebalance-completeness 0.90

# 5. Mainnet (requires confirmation prompt)
python -m causal_portfolio.execution.cli execute --weights weights.json --live --mainnet

# Optional: slice a live rebalance into deterministic child IOC batches
python -m causal_portfolio.execution.cli execute --weights weights.json --live --testnet --twap-minutes 10 --twap-slices 5
```

## Model-independent logging

The execution CLI automatically writes a unique text log to
`~/.cpcm-execution/logs/execution-cli-*.log` (override with
`CPCM_EXECUTION_LOG_DIR`). It captures all logger
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

## Model-agnostic boundary

`causal_portfolio.execution` depends on stdlib, `requests`, and the Hyperliquid
SDK — and on nothing else in this repository. Models depend on execution; the
dependency never runs the other way. In particular execution must not import
`causal_portfolio.{data,factors,solvers,backtest,models}`, numpy, or pandas,
because none of those ship in the wheel and any one of them would tie the
engine to a specific model.

Practically this means every model — CPCM, research RP-PCA, or anything added later —
reaches Hyperliquid through the same call and receives the same execution
policy (caps, precision, cost gate, repair, audit, trace, notifications):

```text
model -> TargetSnapshot -> execute_target(target, ExecutionConfig) -> SubmitResult
```

The transaction-cost gate is deliberately part of that shared policy: it reads
only the rounded orders, live mids, live L2 books, and config, never strategy
metadata, forecasts, or covariance. Model-specific gating belongs upstream and
should change the submitted `TargetSnapshot` instead.

`causal_portfolio/tests/test_execution_is_model_agnostic.py` enforces this by
static import analysis; it fails with the offending file and line.

### Model-blind at runtime, not just at import

The layer also holds no runtime assumption about *which* model is driving it:

- **Asset universe.** `ExecutionConfig.asset_map` is the single source of
  ticker-to-venue-coin mapping, and everything downstream — planner, trace,
  control panel — resolves through the map that actually executed. Tickers
  outside `DEFAULT_ASSET_MAP` are fully supported; pass your own map.
- **Rebalance cadence.** Execution never assumes or derives one. A model
  declares its own next rebalance, either via
  `trace.start_cycle(..., expected_next_rebalance=...)` or by putting
  `expected_next_rebalance` in `TargetSnapshot.metadata`. Undeclared reads as
  `unknown` in the countdown; it is never guessed. Five-minute, daily, and
  monthly cadences are all first-class.
- **Strategy identity.** `TargetSnapshot.strategy` is provenance only — it
  names the run log and appears in the panel, and is never branched on.
- **Self-veto.** A model may refuse its own output by setting
  `execution_eligible: False` in `TargetSnapshot.metadata`; live submission
  then stops before any exchange write. That single key is the entire
  contract. Execution never inspects *why* — fold Sharpes, placebo p-values,
  holdout status and anything else a model records alongside it are opaque
  here, and belong in metadata purely for audit. An absent key means eligible,
  so models written before this contract are unaffected; any falsy value
  refuses. Dry runs are exempt, so a vetoed target can still be planned and
  inspected.
- **Cost and safety policy** is uniform across models by design: the cost gate
  reads only orders, mids, books, and config. Model-specific gating belongs
  upstream, in the weights you submit.

`max_signal_age_hours` (default 72) bounds how stale a *signal* may be at
submission, which is independent of rebalance frequency — a monthly model
executing promptly has an age near zero. Raise it explicitly if you
intentionally execute on older signals.

`causal_portfolio/tests/test_execution_model_blind.py` covers these at
runtime, including a custom-asset-map regression.

## Daily CPCM causal runner

The scheduled strategy entrypoint is `causal_portfolio.models.causal_daily`.
It uses the repository's data-supported **DAG v2**, not the original hand-drawn
Star-DAG: the only candidate return edge is the one-day-lagged AR(1) innovation
in `chain_congestion` to next-day BTC return. The source signature is frozen to
`btc_FeeTotNtv + doge_FeeTotNtv + eth_FeeTotNtv`; newly available fallback
columns cannot silently change the model.

Generate a target without submitting it:

```bash
python -m causal_portfolio.models.causal_daily --target-out tmp/cpcm_causal_daily_target.json
```

The checked-in scheduled command first retires the last audited placeholder
portfolio through scoped reduce-only orders, refreshes the exact three-asset
source set directly from Coin Metrics, then assesses the registered holdout.
The retirement checks the wallet, prior RP-PCA universe and position sides;
unrecognized holdings or any resting order block it. An address- and
target-bound local marker prevents repetition only after two stable flat-account
checks. If the model passes, it maps the original exposure rule
`clip(1 + 0.5z, 0, 2)` to a 5% base BTC weight (0-10% account exposure) and uses
a 15 bp cost ceiling, 10% same-direction no-trade band, and 90%
plan-completeness gate:

```bash
causal_portfolio/execution/run_cpcm_causal_daily.cmd
```

The existing Windows task is still named `CPCM_RPPCA_Daily_HL_Testnet`, but its
registered compatibility entrypoint `run_rppca_daily.cmd` now delegates to the
CPCM causal runner. This avoids silently leaving the old placeholder active
before the task itself is renamed by an operator.

Mainnet remains non-interactive for scheduling and requires both explicit
flags:

```bash
python -m causal_portfolio.models.causal_daily --execute --mainnet --ack-mainnet
```

The June rule and its original transform were not truly prospectively frozen:
most of their nominal post-2025 window was already observable, and the research
innovation helper changed later. Production therefore uses a new immutable spec
registered on 2026-08-14 (including source set, 2022 history start, trailing AR
window, lag, sizing rule, cost gate, and implementation fingerprints). Starting
with the 2026-08-15 decision, each signal and Hyperliquid mainnet BTC mid at the
09:54 America/Los_Angeles mark is written once to
`cpcm_causal_signal_ledger_2026_08_14_mainnet_marks.jsonl`. The gate scores the
first 180 consecutive scheduled mark-to-mark outcomes beginning 2026-08-16;
causal-rule net Sharpe after 5 bp turnover cost must exceed buy-and-hold BTC
Sharpe by at least 0.10. A missed decision restarts the consecutive window.
Until those outcomes exist, status is `pending` and no causal target can be
emitted. For context only, the retrospective
2026-01-01 through 2026-06-29 check lost 32.9% and trailed BTC buy-and-hold by
0.0035 gross Sharpe and 0.0064 net Sharpe, so it provides no evidence of edge.
The runner also fails closed
when any raw BTC/DOGE/ETH fee stream or BTC price is missing, disagrees on date,
or is older than 72 hours. There is no weak-IV-to-OLS execution fallback.
The model refresh no longer depends on the repository's Supabase mirror.

`causal_portfolio.models.rppca_daily` remains available for historical research
and regression tests, but no checked-in scheduled wrapper calls it.

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

## SDK compatibility

The `cpcm-execution` package installs the tested SDK line automatically. For a
repo-only execution environment, install it directly with:

```bash
pip install "hyperliquid-python-sdk~=0.24.0" "eth-account~=0.13.7"
```

`cpcm-execution` and the repository requirements install these automatically;
the direct command is useful for an execution-only development environment.

## Safety defaults

| Knob | Default | Rationale |
|---|---|---|
| `dry_run` | True | Must explicitly opt-in to live trading |
| `testnet` | True | Mainnet only via explicit flag |
| `network_timeout_seconds` | 15 | Bound each SDK HTTP request |
| `leverage` | 1.0 | No implicit leverage |
| `max_position_pct` | 0.30 | Cap any single asset at 30% of equity |
| `max_single_trade_pct` | 0.10 | One trade can't move more than 10% of equity |
| `min_order_notional_usd` | $10 | Surface sub-minimum sleeves before exchange reject |
| `slippage_bps` | 30 | IOC limit at mid ± 30bps |
| `max_transaction_cost_bps` | None | Optional all-in fee + live L2 impact ceiling |
| `estimated_taker_fee_bps` | 4.5 | Fee component for the pre-trade estimate |
| `min_position_change_pct` | 0 | Optional same-direction resize no-trade band |
| `min_rebalance_completeness` | 0 | Optional minimum submitted share of intended increasing notional |
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
The minimum notional is exchange validation, not rebalance policy. When the
transaction-cost gate is enabled, missing or insufficient L2 depth blocks the
submission before open-order cancellation. The daily causal runner explicitly enables a 15 bp
limit; review it after roughly 30 logged rebalances rather than auto-tuning it.

## Rebalance trace and control panel

The active cycle is `~/.cpcm-execution/logs/rebalance-current.sqlite3`.
Starting a different target archives it as
`rebalance-<started-utc>-<target-id>.sqlite3`; archives are never overwritten
or deleted automatically. The 30-minute PnL command appends marked positions,
PnL, equity, margin, and exposure, then atomically writes the standalone panel
to `~/.cpcm-execution/control-panel/index.html`.

## Audit log format

Each rebalance appends one JSON record to
`~/.cpcm-execution/logs/rebalance-YYYY-MM-DD.jsonl` by default:

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
