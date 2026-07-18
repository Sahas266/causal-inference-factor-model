# Changelog — cpcm-execution

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) — MAJOR = breaking public API
(`execute_target`, `plan_rebalance`, `ExecutionConfig`, types), MINOR =
backwards-compatible features, PATCH = fixes. Pinned exchange SDK:
`hyperliquid-python-sdk==0.24.0` (bump = at least MINOR here, re-run the
testnet gate first).

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
