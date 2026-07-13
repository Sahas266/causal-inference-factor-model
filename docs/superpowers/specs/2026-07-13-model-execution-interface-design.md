# Model-to-Execution Interface Design

## Goal

Give every portfolio model one stable handoff into execution. Models produce
portfolio intent; the execution package owns exchange state, order planning,
safety checks, logging, submission, audit records, and reconciliation.

## Public Contract

Add and export one high-level function:

```python
execute_target(
    target: TargetSnapshot,
    config: ExecutionConfig,
    *,
    acknowledge_mainnet: bool = False,
) -> SubmitResult
```

`TargetSnapshot` is the only model contract. It carries normalized weights,
an `as_of` timestamp, strategy name, source, and optional metadata. Models
that write JSON or CSV first use the existing `load_target_snapshot()` helper.
Raw dictionaries are excluded from this interface so live targets cannot lose
their freshness and provenance data.

## Execution Flow

`execute_target()` lives with the existing Hyperliquid orchestration code and:

1. Starts the shared execution run log, named from `target.strategy` when set.
2. Creates `HLAdapter` from the operator-owned `ExecutionConfig`.
3. Fetches account state, mids, and asset metadata.
4. Calls `plan_rebalance()` with the supplied snapshot.
5. Calls `execute_plan()` and returns its `SubmitResult` unchanged.

The shared run logger reuses an already-active execution log. This lets model
runners capture model generation and execution in one file without duplicate
handlers. `execute_plan()` remains responsible for the structured JSONL audit.

## Errors and Safety

Adapter construction and pre-plan network failures are logged with a traceback
and re-raised because no valid `SubmitResult` exists yet. Planning safety gates,
exchange responses, post-trade state, and reconciliation remain represented by
`SubmitResult`. Mainnet still requires `acknowledge_mainnet=True`; safe defaults
remain testnet and dry-run through `ExecutionConfig`.

## Integration and Tests

RP-PCA will call the shared function instead of duplicating adapter, planning,
and submission code. The CLI keeps its lower-level sequence because it must
display a completed plan before asking for interactive mainnet confirmation.

Tests will use a local fake adapter to prove the full target-to-result flow,
automatic logging, and active-log reuse without network access. Existing
execution safety, audit, RP-PCA, and benchmark tests remain the regression gate.

## Non-Goals

This contract does not cover the separate market-making quote engine, add a
model plugin hierarchy, or introduce another configuration abstraction.
