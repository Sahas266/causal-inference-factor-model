# Changelog — cpcm-execution

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) — MAJOR = breaking public API
(`execute_target`, `plan_rebalance`, `ExecutionConfig`, types), MINOR =
backwards-compatible features, PATCH = fixes. Exchange SDK compatibility line:
`hyperliquid-python-sdk~=0.24.0` (minor bump = at least MINOR here, re-run the
testnet gate first).

## [0.2.0] - 2026-07-22

### Added
- Model-agnostic pre-submit transaction-cost gate using configured taker fees
  plus full-size live L2 VWAP impact; RP-PCA starts at a 15 bp ceiling.
- One-active-cycle SQLite trace with permanent no-overwrite archives for model
  targets, plans, results, cost decisions, PnL, and portfolio metrics.
- Framework-free local HTML control panel generated after each PnL snapshot.
- Telegram rebalance countdown and cost-gate no-op details.
- One-shot 30-minute PnL wrapper for Windows Task Scheduler.

## [0.1.2] - 2026-07-21

### Added
- Telegram messages (execution results, portfolio state, RP-PCA targets) now
  render as HTML: bold labels, monospace ids, 🟢/🔴/⚠️/🛑 status emoji.
- `notify.format_pnl` / `--pnl`: live unrealized PnL per position, marked to
  current mids, plus account total.
- `notify.run_pnl_loop` / `--pnl-loop --interval-minutes`: standalone
  recurring PnL notifier (default 30 min); `run_pnl_notifier.cmd` wrapper.

## [0.1.1] - 2026-07-20

### Fixed
- Reconcile and repair partial fills observed after a submission response error.
- Retry pre-submit repair failures after refreshing account state; stop after
  ambiguous submission errors. Reconcile against the fresh-equity target.
- Keep logs outside installed package directories and record repair details.
- Load `.env` from the caller's working tree and redact Telegram bot tokens
  from request-error logs.
- Accept patch-level SDK and `eth-account` fixes within tested minor lines.

## [0.1.0] - 2026-07-17

First packaged release of the execution layer.

### Added
- `execute_target(TargetSnapshot, ExecutionConfig)` — standard model handoff:
  state fetch, plan, safety gates, submit, reconcile, audit, run log.
- `plan_rebalance` pure planner; book-aware IOC execution; client-side TWAP.
- Leg-failure fallback: post-trade drift triggers up to
  `ExecutionConfig.repair_attempts` re-plan + book-aware retry passes
  (`SubmitResult.repair`).
- Telegram notifications (`execution.notify`): execution results, leg
  failures, model target updates; `cpcm-notify` CLI.
- `cpcm-exec` CLI (plan / state / execute).
- Safety gates: dry-run default, testnet default, address + network binding,
  signal freshness, explicit mainnet acknowledgement, JSONL audit log.
