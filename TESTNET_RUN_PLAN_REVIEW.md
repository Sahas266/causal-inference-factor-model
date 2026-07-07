# Full Execution Testnet Run Plan — Review & Resolution

**Date:** 2026-06-29  
**Reviewed file:** `causal_portfolio/docs/full_execution_testnet_run_plan.md`  
**Status:** Ready for a constrained Hyperliquid testnet run after explicit user confirmation.

## Verdict

The prior review findings are resolved. A partial testnet start also surfaced two operational fixes, now incorporated: reconciliation uses the execution min-notional floor for tiny runs, and the wrapper defaults to read-only preparation unless `-ConfirmedStart` is passed.

## Findings

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| R1 | Blocker | `plan --testnet` was invalid. | Runbook uses `plan --live-state`; only `execute` uses `--testnet`. |
| R2 | Metric substitution | IOC fee accounting could synthesize `fee=0`. | Harness now raises unless fee-bearing fills are indexed. |
| R3 | Gate bypass | AS+PIN hardcoded estimator usability. | Harness now passes `fit.usable` into `ToxicitySignal`. |
| R4 | Coverage gap | Maker-fill acceptance could be one-sided. | Campaign requires maker coverage on both AS+PIN sides. |
| R5 | Coverage gap | Plain batch submit path was not exercised. | Flatten uses `--no-smart-execution`. |
| R6 | Coverage gap | Freshness block path was not tested. | Runbook includes stale-target negative gate. |
| R7 | Correctness | Fixed weight contradicted tiny BTC sizing. | RP-PCA placeholder scales weights from live equity. |
| N1 | Coverage gap | Small ETH/SOL sleeves could fall below HL min notional. | Rebalancer has `min_order_notional_usd=10.0`; runbook uses $45 gross. |
| N2 | Robustness | AS+PIN campaign failures aborted the whole run. | `run_testnet_execution_campaign` records failures and stops only on fatal cleanup/preflight errors. |
| N3 | Blocker | TWAP children could fall below min notional after parent validation. | TWAP slicer caps per-order effective slices and logs the child plan. |
| N4 | Acceptance gap | Phase 3 could pass with a no-fill entry. | Runbook now snapshots post-entry state and fails if gross fill ratio is below 80%. |
| N5 | Acceptance gap | `$50` reconcile floor marked a ~$20 BTC residual as OK in a $45 test. | `execute_plan` now reconciles with `min_order_notional_usd`; audit records include drifts. |
| N6 | Safety/process | The run could start immediately from the wrapper. | Wrapper is read-only by default; writes require `-ConfirmedStart` or `-CleanupOnly`. |
| N7 | Cleanup robustness | Plain-batch flatten can be rejected by book/oracle divergence and is not a reliable safety cleanup. | Plain batch remains a coverage leg; confirmed cleanup verifies flat material state before AS+PIN. |
| N8 | AS+PIN preflight | Portfolio cleanup can leave sub-$10 BTC dust, but AS+PIN requires exactly flat BTC. | Confirmed cleanup now closes BTC down to exchange min-size and gates AS+PIN on zero BTC dust. |
| N9 | TWAP edge | Children barely above $10 can still reject when live execution price moves below the parent limit. | TWAP child sizing now uses a 5% min-notional buffer. |

## Transaction Logging Added

The runbook now starts a per-run transcript directory under `tmp/full_execution_testnet_<runId>/` and captures:

- terminal transcript;
- preflight, post-entry, post-flatten, and post-AS+PIN state snapshots;
- open orders at each checkpoint;
- all `user_fills_by_time` records since run start;
- copied audit JSONL records;
- TWAP `child_plan` with parent/child CLOIDs, sizes, prices, notionals, requested/effective slices;
- reconcile `drifts`;
- `post_entry_fill_check.json`;
- `confirmed_cleanup_result.json` when cleanup is needed;
- AS+PIN attempt and campaign summary JSON files.

## Go Criteria

Proceed only after the user explicitly confirms `-ConfirmedStart`. Offline tests must pass, `state --verbose` must confirm testnet, the address must be expected, and no open orders may exist. If material residual testnet positions exist, confirmed cleanup runs before target generation and must verify flat material state. Before AS+PIN, BTC must be flat to exchange min-size even if the notional is below $10. Stop immediately on mainnet flags, stale/missing target provenance, unexpected address, open-order preflight failure, cleanup failure, or post-entry fill ratio below 80%.
