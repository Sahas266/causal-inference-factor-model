# Full Execution Testnet Run Plan

## Objective

Test execution of a hypothetical strategy, not just isolated order plumbing. The
strategy is represented by a Risk-Premium PCA-style handoff: a dated rebalance
row of asset weights, like the RP-PCA workflow where factor weights are mapped
back to asset weights.

This run covers:

- RP-PCA-style target loading, freshness checks, risk caps, IOC/TWAP execution,
  reconciliation, and audit logs;
- AS+PIN execution on the BTC leg of the same strategy-sized target using the
  real Avellaneda-Stoikov quote model, real EKOP PIN fit from captured testnet
  aggressor-side trades, toxicity adjustment, post-only `Alo`, cancel, IOC
  fallback, and cleanup.

This is a functional test of execution. It does not prove alpha or execution
superiority, and it does not run the external RP-PCA repo live; it uses that
repo's target-weight output shape as the strategy placeholder.

## Preconditions

Run from the repository root:

```powershell
venv\Scripts\python.exe -m pytest `
  causal_portfolio\tests\test_execute_safety.py `
  causal_portfolio\tests\test_execution_targets.py `
  causal_portfolio\tests\test_rebalancer.py `
  causal_portfolio\tests\test_testnet_execution_benchmark.py -q
```

Required environment:

- `HL_ADDRESS` points to the intended Hyperliquid testnet account.
- `HL_PRIVATE_KEY` is loaded locally and never committed.
- Account starts flat or with only a deliberately tiny BTC position.
- No pre-existing open orders.
- No mainnet flags are used.

Abort if any preflight check fails.

## Confirmation Gate

Use the checked-in wrapper so a dry preflight is the default and live testnet
writes require an explicit flag:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tmp\run_full_execution_testnet.ps1
```

The command above performs read-only preflight, creates a run directory, and
stops before any write. To start the actual testnet run after operator approval:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tmp\run_full_execution_testnet.ps1 -ConfirmedStart
```

If preflight finds residual material testnet positions from a prior partial
run, `-ConfirmedStart` first runs confirmed cleanup and verifies flat material
state before generating new strategy targets. To cleanup only:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tmp\run_full_execution_testnet.ps1 -CleanupOnly
```

## Transaction Logging & Post-Mortem Capture

Create a run directory before any write and keep every state snapshot, response,
fill record, and terminal line there. Do not write secrets.

```powershell
$runId = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$env:TESTNET_RUN_DIR = "tmp\full_execution_testnet_$runId"
$env:TESTNET_RUN_START_MS = [string][DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
New-Item -ItemType Directory -Force $env:TESTNET_RUN_DIR | Out-Null
Start-Transcript -Path "$env:TESTNET_RUN_DIR\terminal_transcript.txt"
```

Use this snapshot helper after preflight, after entry, after flatten, after the
AS+PIN campaign, and during cleanup. It captures account state, open orders,
all user fills since `TESTNET_RUN_START_MS`, the current audit JSONL, and the
run metadata needed to reconstruct the transaction timeline.

```powershell
$env:SNAPSHOT_LABEL = "preflight"
$snapshotScript = @'
import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from causal_portfolio.execution.audit import LOG_DIR
from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.hyperliquid import HLAdapter

label = os.environ["SNAPSHOT_LABEL"]
run_dir = Path(os.environ["TESTNET_RUN_DIR"])
start_ms = int(os.environ["TESTNET_RUN_START_MS"])
adapter = HLAdapter(ExecutionConfig(testnet=True, dry_run=True))
state = adapter.fetch_state()
open_orders = adapter.info.frontend_open_orders(adapter.address)
fills = adapter.info.user_fills_by_time(adapter.address, start_ms)
audit_path = LOG_DIR / f"rebalance-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"

payload = {
    "label": label,
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "network": "testnet",
    "address": adapter.address,
    "state": asdict(state),
    "open_orders": open_orders,
    "fills_since_run_start": fills,
}
(run_dir / f"{label}_snapshot.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
if audit_path.exists():
    shutil.copy2(audit_path, run_dir / f"{label}_{audit_path.name}")
'@
$snapshotScript | venv\Scripts\python.exe -
```

Stop the transcript at the end of the run with `Stop-Transcript`.

## Six-Hour Window Allocation

Budget the live testnet window as follows:

- 0:00-0:20: preflight, state read, RP-PCA placeholder target generation, and
  dry-run review;
- 0:20-0:45: portfolio rebalance entry, flatten, state/audit verification;
- 0:45-2:45: real testnet trade/L2 capture for PIN, volatility, and markout
  inputs;
- 2:45-5:45: repeated tiny AS+PIN paired trials on the strategy BTC leg;
- 5:45-6:00: cleanup, final state/open-order checks, and evidence capture.

If the portfolio rebalance phase takes longer than planned, preserve the final
15-minute cleanup buffer and shorten the repeated AS+PIN trial window first.

## Strategy Placeholder Under Test

Use a small synthetic RP-PCA-like long/short crypto target:

```text
strategy = rppca_placeholder_tangency
raw weights = BTC +0.45, ETH +0.30, SOL -0.25
```

The raw vector is scaled to a tiny gross notional before writing the CSV. The
executor should not care whether the weights came from the external RP-PCA
pipeline or this placeholder; it should safely execute a fresh dated target.

## Phase 1: Read-Only Account Check

Inspect state before any write:

```powershell
venv\Scripts\python.exe -m causal_portfolio.execution.cli state --verbose
```

Acceptance:

- output says `testnet`;
- BTC position is zero or intentionally tiny;
- open orders count is zero;
- equity and address match the intended testnet account.

## Phase 2: Generate RP-PCA-Style Target CSV

Create a fresh target CSV under ignored `tmp/`. It includes an older row to
prove latest-row selection and a fresh row for execution.

```powershell
$script = @'
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.hyperliquid import HLAdapter

cfg = ExecutionConfig(testnet=True, dry_run=True)
adapter = HLAdapter(cfg)
state = adapter.fetch_state()
mids = adapter.fetch_mids()

raw = {"BTC": 0.45, "ETH": 0.30, "SOL": -0.25}
target_gross_usd = 45.0
scale = target_gross_usd / (state.account_value_usd * sum(abs(v) for v in raw.values()))
weights = {asset: value * scale for asset, value in raw.items()}

now = datetime.now(timezone.utc)
target_path = Path("tmp/rppca_placeholder_targets.csv")
target_path.parent.mkdir(parents=True, exist_ok=True)
with target_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["rebalance_date", "generated_at", "strategy", "BTC", "ETH", "SOL"],
    )
    writer.writeheader()
    writer.writerow({
        "rebalance_date": (now - timedelta(days=30)).isoformat(),
        "generated_at": (now - timedelta(days=30)).isoformat(),
        "strategy": "rppca_placeholder_tangency",
        "BTC": 0.0,
        "ETH": 0.0,
        "SOL": 0.0,
    })
    writer.writerow({
        "rebalance_date": now.isoformat(),
        "generated_at": now.isoformat(),
        "strategy": "rppca_placeholder_tangency",
        **weights,
    })

btc_size = math.floor(abs(state.account_value_usd * weights["BTC"] / mids["BTC"]) * 1e5) / 1e5
metadata = {
    "target_path": str(target_path),
    "equity": state.account_value_usd,
    "target_gross_usd": target_gross_usd,
    "weights": weights,
    "estimated_btc_size": btc_size,
}
Path("tmp/rppca_placeholder_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)
print(json.dumps(metadata, indent=2))
'@
$script | venv\Scripts\python.exe -
```

Review `estimated_btc_size`. With a $45 gross target, the BTC leg should be
about `$20.25 / live BTC mid`. Do not lower gross notional below $45 unless you
also change the raw vector; the smallest sleeve must clear the execution
layer's $10 `min_order_notional_usd` check.

## Phase 3: Portfolio Rebalance Execution

Dry-run against live state. `plan` is testnet by default and has no `--testnet`
flag; only `execute` accepts `--testnet`/`--mainnet`.

```powershell
venv\Scripts\python.exe -m causal_portfolio.execution.cli plan `
  --weights tmp\rppca_placeholder_targets.csv --live-state -v
```

Continue only if:

- network is testnet;
- target ID/provenance is present;
- target is fresh;
- BTC/ETH/SOL mapping is correct;
- gross notional is near the chosen `target_gross_usd`;
- skipped assets, if any, are understood before proceeding.

Execute the strategy target with TWAP enabled:

```powershell
venv\Scripts\python.exe -m causal_portfolio.execution.cli execute `
  --weights tmp\rppca_placeholder_targets.csv --live --testnet `
  --twap-minutes 0.1 --twap-slices 3
```

The TWAP executor caps each order's effective child count when a requested
slice would fall below `min_order_notional_usd`; it applies a 5% buffer because
HL evaluates the minimum against the live execution price, not just the parent
limit. The audit response includes `response.response.data.child_plan` with
parent notional, requested/effective slices, child CLOIDs, child sizes, limit
prices, and child notionals. Verify every child notional clears the buffered
floor or is a single reduce-only dust close.

Immediately snapshot the post-entry state and verify the entry actually filled:

```powershell
$env:SNAPSHOT_LABEL = "post_entry"
$snapshotScript | venv\Scripts\python.exe -

$fillCheckScript = @'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["TESTNET_RUN_DIR"])
metadata = json.loads(Path("tmp/rppca_placeholder_metadata.json").read_text(encoding="utf-8"))
snapshot = json.loads((run_dir / "post_entry_snapshot.json").read_text(encoding="utf-8"))
positions = snapshot["state"].get("positions", {})
weights = metadata["weights"]
equity = metadata["equity"]

targets = {coin: equity * weight for coin, weight in weights.items()}
actuals = {coin: positions.get(coin, {}).get("notional_usd", 0.0) for coin in targets}
filled_gross = sum(abs(actuals[coin]) for coin in targets)
target_gross = sum(abs(value) for value in targets.values())

report = {
    "targets_usd": targets,
    "actuals_usd": actuals,
    "filled_gross_usd": filled_gross,
    "target_gross_usd": target_gross,
    "fill_ratio": filled_gross / target_gross if target_gross else 1.0,
}
(run_dir / "post_entry_fill_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if report["fill_ratio"] < 0.80:
    raise SystemExit("post-entry fill ratio below 80%; do not flatten until investigated")
'@
$fillCheckScript | venv\Scripts\python.exe -
```

Create a fresh RP-PCA-style flatten target:

```powershell
$script = @'
import csv
from datetime import datetime, timezone
from pathlib import Path

now = datetime.now(timezone.utc)
path = Path("tmp/rppca_placeholder_flatten.csv")
with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["rebalance_date", "generated_at", "strategy", "BTC", "ETH", "SOL"],
    )
    writer.writeheader()
    writer.writerow({
        "rebalance_date": now.isoformat(),
        "generated_at": now.isoformat(),
        "strategy": "rppca_placeholder_flatten",
        "BTC": 0.0,
        "ETH": 0.0,
        "SOL": 0.0,
    })
print(path)
'@
$script | venv\Scripts\python.exe -
```

Dry-run the flatten target, then execute it through the plain batch submit path
to cover `submit_orders` as well as TWAP/book-aware execution. This is a
coverage leg, not the final safety cleanup: if the exchange rejects one child
because book/oracle bands are temporarily divergent, the wrapper records the
plain-batch response and then runs confirmed material-position cleanup before
AS+PIN.

```powershell
venv\Scripts\python.exe -m causal_portfolio.execution.cli plan `
  --weights tmp\rppca_placeholder_flatten.csv --live-state -v

venv\Scripts\python.exe -m causal_portfolio.execution.cli execute `
  --weights tmp\rppca_placeholder_flatten.csv --live --testnet --no-smart-execution
```

Confirm post-plain-batch state and audit log:

```powershell
$env:SNAPSHOT_LABEL = "post_plain_batch_flatten"
$snapshotScript | venv\Scripts\python.exe -

venv\Scripts\python.exe -m causal_portfolio.execution.cli state --verbose
venv\Scripts\python.exe -m causal_portfolio.execution.cli logs --verbose
```

Acceptance:

- entry and flatten records appear in `logs/rebalance-YYYY-MM-DD.jsonl`;
- entry response is `type=twap` with child-slice summaries;
- entry `child_plan` shows no sub-$10 child slices except single reduce-only
  dust closes;
- `post_entry_fill_check.json` shows at least 80% of intended gross filled
  before flattening;
- flatten response is the plain `bulk_orders` shape;
- flatten orders are `reduce_only=true` where they shrink positions;
- records include network, target ID, response, post-state, and drift fields;
- any material residual after the coverage flatten is closed by the confirmed
  cleanup gate and captured in `confirmed_cleanup_result.json`;
- `post_rebalance_cleanup_snapshot.json` shows BTC/ETH/SOL positions flat or
  below minimum executable dust before AS+PIN starts;
- BTC specifically is flat to exchange min-size before AS+PIN starts, because
  the AS+PIN harness requires an exactly flat initial BTC position;
- no open orders remain.

## Phase 4: AS+PIN Execution on the Strategy BTC Leg

Use the actual AS+PIN benchmark harness with the BTC size derived from the
RP-PCA placeholder target. The current AS+PIN testnet harness is BTC-specific,
so this phase validates AS+PIN execution for the BTC leg of the strategy rather
than all RP-PCA assets.

Do not substitute candle PIN, fixed toxicity, or placeholder quote metrics. The
run must call:

- `capture_testnet_events` for real testnet `trades` and `l2Book`;
- `fit_pin` / `pin_posteriors` on bucketed aggressor-side counts;
- `finite_horizon_quotes` for Avellaneda-Stoikov quotes;
- `toxicity_adjustment` for PIN/posterior quote adjustment;
- `submit_quote_pair` for post-only `Alo`;
- `cancel_strategy_quotes`;
- IOC fallback for any unfilled remainder;
- `_restore_position` cleanup.

Run repeated tiny paired tests for the remainder of the six-hour window. Each
attempt captures fresh testnet trades/books, fits PIN, generates AS quotes,
submits post-only `Alo`, cancels, falls back with IOC as needed, and restores
flat inventory. Stop early only after at least three successful paired attempts
and both AS+PIN sides have maker-filled at least once.

```powershell
$script = @'
import json
import os
from pathlib import Path
from causal_portfolio.market_making.testnet_execution_benchmark import run_testnet_execution_campaign

metadata = json.loads(Path("tmp/rppca_placeholder_metadata.json").read_text(encoding="utf-8"))
run_dir = Path(os.environ["TESTNET_RUN_DIR"])
size = metadata["estimated_btc_size"]
if size <= 0:
    raise SystemExit("estimated_btc_size is zero; increase target_gross_usd and regenerate")

summary = run_testnet_execution_campaign(
    size=size,
    max_seconds=6 * 60 * 60,
    capture_seconds=900,
    passive_timeout_seconds=180,
    min_successful_attempts=3,
    acknowledge_testnet_writes=True,
    output_dir=run_dir,
    state_path_prefix="full-execution-testnet-cloids",
)
print("SUMMARY")
print(json.dumps(summary, indent=2))
'@
$script | venv\Scripts\python.exe -

$env:SNAPSHOT_LABEL = "post_as_pin"
$snapshotScript | venv\Scripts\python.exe -
```

Acceptance:

- at least three paired attempts complete inside the six-hour window;
- every accepted attempt has `pin_sample_buckets >= 20`; with 900-second
  captures, prefer `>= 180`;
- at least three attempts have `pin_fit_usable=true`; the harness passes
  `fit.usable` into `ToxicitySignal`, so unusable PIN aborts the AS+PIN quote
  path instead of silently bypassing toxicity;
- every accepted AS+PIN buy and sell leg reaches `status="filled"`;
- maker fill coverage is observed on both sides across the repeated attempts:
  at least one attempt with `as_pin_buy.maker_filled_size > 0` and at least one
  attempt with `as_pin_sell.maker_filled_size > 0`;
- any remainder is filled by IOC fallback, but fallback must not be the only
  observed path across the campaign;
- reported fees, shortfall, and PnL come from real fee-bearing
  `user_fills_by_time` records; the harness raises instead of fabricating
  `fee=0` metrics;
- every attempt restores initial inventory, or the campaign stops immediately
  for cleanup;
- final BTC position is flat and no strategy-owned orders remain.

## Optional Negative-Gate Check

Build a target whose `as_of`/`rebalance_date` is more than 72 hours old and try
to execute it without `--allow-stale-signal`. It must fail with
`signal freshness check failed` before canceling or submitting anything.

## Stop Conditions

Stop immediately if:

- any command targets mainnet;
- target provenance is missing or stale;
- planned notional is larger than the intended tiny strategy target;
- account address is unexpected;
- open orders exist before the run;
- `cancel_strategy_quotes` fails;
- final inventory is not restored;
- audit logs are missing for the rebalance phase.

## Final Evidence to Save

Keep these local artifacts for review:

- `logs/rebalance-YYYY-MM-DD.jsonl`;
- `tmp/rppca_placeholder_targets.csv`;
- `tmp/rppca_placeholder_metadata.json`;
- `tmp/rppca_placeholder_flatten.csv`;
- `tmp/full_execution_testnet_<runId>/full_execution_testnet_result_attempt_*.json`;
- `tmp/full_execution_testnet_<runId>/full_execution_testnet_summary.json`;
- terminal output from preflight, dry-run plans, final state, and AS+PIN result.
- `tmp/full_execution_testnet_<runId>/terminal_transcript.txt`;
- `tmp/full_execution_testnet_<runId>/*_snapshot.json`;
- `tmp/full_execution_testnet_<runId>/post_entry_fill_check.json`;
- `tmp/full_execution_testnet_<runId>/confirmed_cleanup_result.json` if
  cleanup was needed;
- copied `rebalance-YYYY-MM-DD.jsonl` snapshots containing TWAP `child_plan`
  and reconcile `drifts`;

Summarize:

- account/network checked;
- RP-PCA placeholder target IDs executed;
- fill status per submit path: TWAP strategy entry, plain-batch flatten,
  AS+PIN BTC legs;
- per-attempt PIN estimate, `pin_fit_usable`, and sample bucket count;
- per-attempt AS+PIN maker/fallback shares per leg;
- per-attempt fees, shortfall, and round-trip PnL from real fee-bearing fill
  records;
- aggregate maker coverage across the six-hour campaign;
- final inventory and open-order cleanup status.
