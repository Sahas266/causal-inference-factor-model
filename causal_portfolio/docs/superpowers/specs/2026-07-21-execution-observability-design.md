# Execution Threshold and Rebalance Observability Design

Date: 2026-07-21  
Baseline: `execution` at `cdd3c148`

## Objective

Complete the execution-layer work without replacing behavior that already
exists. Add a configurable portfolio-level no-trade threshold, show the next
rebalance in Telegram updates, run RP-PCA once daily, publish PnL and portfolio
metrics every 30 minutes, retain a compact price-to-model-to-execution trace
for every rebalance, and generate a framework-free control panel.

## Current State and Task Coverage

| Requested task | Current state | Design action |
|---|---|---|
| Standard execution-engine interface | Done: `execute_target(TargetSnapshot, ExecutionConfig) -> SubmitResult`; RP-PCA uses it | Keep unchanged |
| Hyperliquid failed-leg fallback, retry, and notification | Done and hardened in `cdd3c148`; stale-target false alerts and ambiguous partial submissions fixed | Keep tests; record repair/result events |
| Telegram bot/channel for model and portfolio updates | Configured and reachable; bot can post to the channel | Keep credentials external |
| Package execution algorithm and Hyperliquid SDK | Done as `cpcm-execution` 0.1.2 with compatible dependency ranges | Include the new module with no dependency and release the new public features as 0.2.0 |
| Explain suspicious `$1,000` leg failures | Done: `$0` matched leaked synthetic test data; `$1,200` matched stale-target repair accounting | Keep tests isolated; future events become traceable locally |
| Inspect Telegram history for debugging | Bot API cannot provide a general history of the bot's own old posts | Use local audit and SQLite trace going forward |
| Maintain a streamed-work todo list | Active in the task plan | Update through delivery |
| Cost-aware rebalance threshold in execution | Missing; only backtest L1 helper exists | Add execution-level L1 no-trade band |
| Beautify Telegram messages | Done in 0.1.2 using escaped Telegram HTML | Reuse; add countdown and threshold no-op text |
| PnL every 30 minutes | Formatter, `--pnl`, and loop exist; nothing is scheduled or running | Schedule the one-shot `--pnl` command every 30 minutes |
| RP-PCA rebalance once daily | Existing testnet task runs daily at 12:54 ET and is healthy | Preserve and verify the existing task; never create a duplicate |
| Time to next rebalance | Missing | Store next rebalance time and render a countdown |
| SQLite mapping price action to model action | Missing | Add one active rebalance trace database |
| Log model output, PnL, and portfolio metrics every 30-minute tick | Missing | Record structured model and portfolio observations in SQLite |
| Preserve completed rebalance data | Missing | Archive each completed database; never discard or overwrite it |
| Bare HTML + JavaScript weights control panel | Missing | Generate one standalone local HTML snapshot; no framework or server |

The existing daily testnet task is healthy: it last completed successfully on
2026-07-21 at 12:54 ET and is scheduled daily at 12:54 ET.

## 1. Execution-Level No-Trade Threshold

Add `rebalance_threshold_l1: float = 0.0` to `ExecutionConfig`. The default
preserves all existing callers. Reject negative values.

This threshold is deliberately model-agnostic. It may consume only the target
weights supplied through `TargetSnapshot`, live `AccountState`, the executor's
mapped and capped target notionals, and `ExecutionConfig`. It must not import
model or backtest code, inspect strategy metadata, or use covariance, factor
exposure, forecasts, expected returns, or model confidence. Every model routed
through `execute_target()` receives the same execution policy.

`plan_rebalance()` already computes capped target notionals and live position
deltas. Immediately after that calculation, compute:

```text
turnover_l1 = sum(abs(delta_usd)) / equity_used
```

If the configured threshold is positive and `turnover_l1` is below it, return
the normal auditable plan with no orders and a machine-readable
`below_rebalance_threshold` skip reason. Equality executes. This location
includes held assets absent from the new target and uses the same capped target
the executor would otherwise trade.

Expose the setting through the execution CLI and RP-PCA CLI. Do not silently
choose a nonzero production value: the generic library remains at zero. Testnet
operators can set a value after the new trace shows actual turnover and costs.
Threshold no-ops must notify Telegram and be stored as model decisions.

## 2. Telegram PnL and Rebalance Countdown

Keep the escaped HTML format in 0.1.2. Extend the existing PnL update with:

- current account equity and unrealized PnL;
- open positions;
- `Next rebalance: <duration> (<timestamp>)`;
- `OVERDUE` when the expected time has passed.

The RP-PCA target update receives the same countdown. The expected next time
is stored when the target is generated as `generated_at + every_hours`; the
current scheduled command therefore remains 24 hours. If trace metadata is
missing, display `unknown` rather than guessing or failing the notification.

Use Windows Task Scheduler for recurrence because it already runs the daily
model. Create `CPCM_Portfolio_30m_HL_Testnet`, invoking the existing one-shot
`python -m causal_portfolio.execution.notify --pnl` every 30 minutes from the
repository working directory. Do not use a second always-running Python
process. The existing loop stays compatible but is not the deployed path.

Keep `CPCM_RPPCA_Daily_HL_Testnet` enabled at its existing once-daily 12:54 ET
schedule. Deployment checks and updates that task by name; it does not create a
second daily rebalance task.

Telegram failures remain warnings and never change an execution result. No test
message or live order is sent during automated verification.

## 3. One-Rebalance SQLite Trace with Permanent Archives

Add a small `execution/trace.py` module using only stdlib `sqlite3`. The active
file lives beside execution logs as `rebalance-current.sqlite3`. It contains
only one target cycle and uses two tables:

```text
meta(key PRIMARY KEY, value)
events(id, ts_utc, target_id, kind, coin, price_source, mid_price,
       target_weight, target_usd, delta_usd, position_size,
       position_usd, unrealized_pnl_usd, equity_usd, margin_used_usd,
       free_margin_usd, gross_exposure_usd, net_exposure_usd, payload_json)
```

All timestamps are UTC ISO-8601. JSON uses deterministic key ordering. Short
connections and ordinary transactions are sufficient for the daily writer and
30-minute snapshot writer; no dependency, service, or ORM is added.

### Lifecycle

1. RP-PCA generates a new `TargetSnapshot`.
2. If the active database has a different target id, close it and rename it to
   `rebalance-<started-utc>-<target-id>.sqlite3`.
3. Never overwrite an archive. If archival fails, leave the old database
   untouched, warn, and skip tracing the new cycle.
4. Create the new active database and store target id, strategy, signal time,
   generation time, and expected next rebalance.
5. Archives have no automatic retention deletion.

### Events

- `model_target`: one row per asset with the model-input price and target
  weight when RP-PCA writes the target.
- `execution_plan`: one row per affected coin with live HL mid, target USD,
  delta USD, and serialized planned order details.
- `execution_result`: post-state, drift, repair, response, and error data.
- `portfolio_snapshot`: every 30-minute PnL run records HL mids, positions, and
  unrealized PnL for the union of current target and held coins. One aggregate
  row also records equity, margin used/free, gross/net exposure, total
  unrealized PnL, and position count.
- `threshold_noop`: records the measured L1 turnover and configured threshold.

Trace writes are observability, not trading gates. Ordinary write failures log
a warning and execution continues. Rotation is the exception: failure preserves
the completed database and disables new-cycle tracing rather than risking data
loss.

### Integration boundary

RP-PCA starts the rich cycle with model prices and the next rebalance time.
`execute_target()` also performs a best-effort cycle initialization so another
model using the standard interface still gets a trace. The shared
`execute_plan()` path records the plan and result, which also covers callers
that already prepared a plan. Repeated calls for the same target id append to
the active cycle and do not rotate it.

## 4. Framework-Free Control Panel

Generate `~/.cpcm-execution/control-panel/index.html` atomically after each
30-minute portfolio snapshot. It is a standalone document built from one
packaged HTML template with embedded JSON and plain JavaScript. The operator
opens the file directly; there is no HTTP server, API, framework, build step,
CDN, or external font.

The panel shows:

- target id, strategy, last update, freshness, and rebalance countdown;
- equity, unrealized PnL, margin/free margin, gross/net exposure;
- target versus actual weight for every target or held asset;
- current mid, position notional, and per-position PnL;
- latest threshold decision and execution/repair status.

Use a brutally minimal dark data-console direction: semantic HTML, high
contrast, 4/8px spacing, tabular numbers, responsive tables, visible keyboard
focus, and text/sign indicators in addition to color. Frequent data refreshes
have no animation. JavaScript parses embedded JSON and writes values with
`textContent`; serialized `<` characters are escaped to prevent script-breakout
in dynamic data.

Only the active cycle appears in the panel. Archived SQLite files remain the
historical source and are not loaded into the browser.

## 5. Data Flow

```text
RP-PCA prices -> TargetSnapshot -> rotate/start SQLite cycle -> model_target
                               -> execute_target -> execution_plan/result

30-minute scheduled --pnl -> HL state + mids -> portfolio_snapshot + metrics
                                            -> standalone control panel
                                            -> Telegram PnL + countdown
```

The existing JSONL audit remains the authoritative submission record. SQLite is
the queryable within-cycle chronology used to compare price movement, model
weights, planned trades, actual state, repairs, and PnL.

## 6. Verification and Delivery

Add focused regressions for:

- below/equal/above threshold behavior and explicit threshold no-op reason;
- archive rotation preserving the old database and refusing overwrite;
- model, plan/result, and 30-minute snapshot rows;
- aggregate PnL and portfolio metrics on every 30-minute tick;
- countdown, overdue, and missing-metadata formatting;
- Telegram HTML escaping and test credential isolation.
- standalone control-panel generation, safe embedded JSON, target/actual
  weights, keyboard semantics, and no network/framework dependency.

Then run the full Python suite, read-only Hyperliquid integration tests,
`git diff --check`, package build/install metadata checks, both installed CLI
help commands, and a read-only query of both Windows scheduled tasks. Do not
submit a live order or post a verification message.

These are backward-compatible public features, not a patch-only fix. Update the
package and changelog to 0.2.0; retain dynamic version metadata and the existing
compatible Hyperliquid SDK dependency range.

## Non-Goals

- Predicting whether expected alpha exceeds fees: targets do not carry a
  defensible expected-benefit value. Model-specific tracking-error,
  factor-capture, covariance, or alpha gates belong upstream and may influence
  the submitted `TargetSnapshot`, but cannot alter the generic execution
  threshold. The L1 band is the existing repo pattern.
- Tick or order-book history: 30-minute mids are enough for this requested
  rebalance-level map. The separate DuckDB market-making recorder owns
  microstructure data.
- Telegram as an audit database: local archives provide deterministic history.
- A dashboard server, frontend framework, package manager, charting library, or
  historical archive browser. The generated panel covers current-cycle weights
  and metrics.
- Automatic archive deletion, compaction, additional dashboards, or a new
  service.
