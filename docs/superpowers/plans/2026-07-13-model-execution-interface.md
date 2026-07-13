# Model-to-Execution Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every portfolio model one typed, logged call from `TargetSnapshot` to `SubmitResult`.

**Architecture:** Add `execute_target()` to the existing Hyperliquid orchestration module and export it from the execution package. It reuses the current adapter, planner, executor, audit, and run logger; RP-PCA delegates to it while the interactive CLI retains plan preview control.

**Tech Stack:** Python 3.11+, stdlib logging/contextvars, pytest, existing Hyperliquid adapter.

## Global Constraints

- `TargetSnapshot` is the only model input contract; reject raw dictionaries.
- Execution owns exchange reads, planning, logging, safety, submission, audit, and reconciliation.
- Preserve testnet/dry-run defaults and explicit mainnet acknowledgement.
- Add no dependency, runner class, plugin hierarchy, or market-making integration.
- Stage named files only; exclude `.codex/` and `tmp/`.

---

### Task 1: Make execution run logging reentrant

**Files:**
- Modify: `causal_portfolio/execution/run_logging.py`
- Test: `causal_portfolio/tests/test_execution_audit.py`

**Interfaces:**
- Consumes: existing `execution_run_log(name, log_dir=None)` context manager.
- Produces: the same signature, reusing the active log path for nested calls.

- [ ] **Step 1: Write the failing nested-log test**

Add:

```python
def test_execution_run_log_reuses_active_file(tmp_path):
    from causal_portfolio.execution import execution_run_log

    with execution_run_log("outer", log_dir=tmp_path) as outer:
        with execution_run_log("inner", log_dir=tmp_path) as inner:
            logging.getLogger("nested_model").info("nested handoff")

    assert inner == outer
    assert len(list(tmp_path.glob("execution-*.log"))) == 1
    assert "nested handoff" in outer.read_text(encoding="utf-8")
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
venv\Scripts\python.exe -m pytest causal_portfolio/tests/test_execution_audit.py::test_execution_run_log_reuses_active_file -q
```

Expected: FAIL because nested calls create different files.

- [ ] **Step 3: Implement active-log reuse**

Import `ContextVar`, define:

```python
_ACTIVE_LOG: ContextVar[Path | None] = ContextVar("cpcm_execution_log", default=None)
```

At the start of `execution_run_log()`:

```python
active_path = _ACTIVE_LOG.get()
if active_path is not None:
    yield active_path
    return
```

Set a token after installing the file handler and reset it during cleanup:

```python
token = _ACTIVE_LOG.set(path)
try:
    ...
finally:
    root.removeHandler(handler)
    root.setLevel(previous_level)
    handler.close()
    _ACTIVE_LOG.reset(token)
```

Keep the current UTC formatting and `BaseException` failure logging. Add:

```python
# ponytail: process-wide handler assumes one execution run at a time; use
# queue handlers if concurrent execution is introduced.
```

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: PASS.

---

### Task 2: Add the standard model handoff

**Files:**
- Modify: `causal_portfolio/execution/hyperliquid.py`
- Modify: `causal_portfolio/execution/__init__.py`
- Test: `causal_portfolio/tests/test_execution_targets.py`

**Interfaces:**
- Consumes: `TargetSnapshot`, `ExecutionConfig`, `HLAdapter`, `plan_rebalance()`, `execute_plan()`, `execution_run_log()`.
- Produces: `execute_target(target: TargetSnapshot, config: ExecutionConfig, *, acknowledge_mainnet: bool = False) -> SubmitResult`.

- [ ] **Step 1: Write failing contract tests**

Add imports inside the tests so collection succeeds before the export exists:

```python
def test_execute_target_owns_model_handoff(tmp_path, monkeypatch):
    from causal_portfolio.execution import audit, execute_target, run_logging
    import causal_portfolio.execution.hyperliquid as hl

    cfg = ExecutionConfig(testnet=True, dry_run=True)
    state, mids, meta = _market()
    adapter = _adapter(cfg)
    adapter.fetch_mids.return_value = mids
    adapter.fetch_meta.return_value = meta
    monkeypatch.setattr(hl, "HLAdapter", lambda config: adapter)
    monkeypatch.setattr(audit, "LOG_DIR", tmp_path)
    monkeypatch.setattr(run_logging, "LOG_DIR", tmp_path)
    target = TargetSnapshot(
        {"btc": 0.1},
        as_of=datetime.now(timezone.utc),
        strategy="unit model",
    )

    result = execute_target(target, cfg)

    assert result.plan.target_snapshot is target
    assert result.submitted is False
    adapter.fetch_state.assert_called_once_with()
    adapter.fetch_mids.assert_called_once_with()
    adapter.fetch_meta.assert_called_once_with()
    assert len(list(tmp_path.glob("execution-unit-model-*.log"))) == 1
    assert len(list(tmp_path.glob("rebalance-*.jsonl"))) == 1


def test_execute_target_rejects_unversioned_dict():
    from causal_portfolio.execution import execute_target

    with pytest.raises(TypeError, match="TargetSnapshot"):
        execute_target({"btc": 0.1}, ExecutionConfig())
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
venv\Scripts\python.exe -m pytest causal_portfolio/tests/test_execution_targets.py::test_execute_target_owns_model_handoff causal_portfolio/tests/test_execution_targets.py::test_execute_target_rejects_unversioned_dict -q
```

Expected: FAIL because `execute_target` is not exported.

- [ ] **Step 3: Implement the minimal orchestration function**

In `hyperliquid.py`, import `plan_rebalance`, `execution_run_log`, and
`TargetSnapshot`, then add before `execute_plan()`:

```python
def execute_target(
    target: TargetSnapshot,
    config: ExecutionConfig,
    *,
    acknowledge_mainnet: bool = False,
) -> SubmitResult:
    """Execute one provenance-carrying model target."""
    if not isinstance(target, TargetSnapshot):
        raise TypeError("target must be a TargetSnapshot")
    with execution_run_log(target.strategy or "model"):
        adapter = HLAdapter(config)
        plan = plan_rebalance(
            target,
            adapter.fetch_state(),
            adapter.fetch_mids(),
            adapter.fetch_meta(),
            config,
        )
        return execute_plan(
            adapter,
            plan,
            acknowledge_mainnet=acknowledge_mainnet,
        )
```

Export `execute_target` from `causal_portfolio.execution.__init__` and replace
its nonexistent `execute_rebalance` docstring entry with the exact new call.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: 2 passed.

- [ ] **Step 5: Commit the core interface and logging**

Stage only the core execution/logger/tests already in scope, then commit:

```powershell
git add -- causal_portfolio/execution/__init__.py causal_portfolio/execution/cli.py causal_portfolio/execution/hyperliquid.py causal_portfolio/execution/run_logging.py causal_portfolio/tests/test_execute_safety.py causal_portfolio/tests/test_execution_audit.py causal_portfolio/tests/test_execution_targets.py
git commit -m "feat(execution): add standard model handoff"
```

---

### Task 3: Route RP-PCA through the standard interface

**Files:**
- Modify: `causal_portfolio/execution/rppca_daily.py`
- Modify: `causal_portfolio/execution/README.md`
- Modify: `causal_portfolio/execution/rebalancer.py`
- Test: `causal_portfolio/tests/test_rppca_daily.py`
- Add: `docs/superpowers/plans/2026-07-13-model-execution-interface.md`

**Interfaces:**
- Consumes: public `execute_target()` and existing `load_target_snapshot()`.
- Produces: RP-PCA submission through the shared handoff with unchanged CLI flags and error behavior.

- [ ] **Step 1: Write the failing RP-PCA delegation test**

Add:

```python
def test_rppca_submission_uses_standard_interface(tmp_path, monkeypatch):
    from types import SimpleNamespace

    target_path = tmp_path / "target.json"
    target_path.write_text(
        '{"btc": 0.1, "_meta": {"rebalance_date": "2026-07-13"}}',
        encoding="utf-8",
    )
    args = build_parser().parse_args([])
    captured = {}

    def fake_execute(target, config, *, acknowledge_mainnet=False):
        captured.update(target=target, config=config, acknowledge=acknowledge_mainnet)
        return SimpleNamespace(error=None, response={"status": "ok"})

    monkeypatch.setattr(rppca_daily, "execute_model_target", fake_execute)
    rppca_daily._submit_target(args, target_path)

    assert captured["target"].weights == {"btc": 0.1}
    assert captured["config"].dry_run is False
    assert captured["config"].testnet is True
    assert captured["acknowledge"] is False
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
venv\Scripts\python.exe -m pytest causal_portfolio/tests/test_rppca_daily.py::test_rppca_submission_uses_standard_interface -q
```

Expected: FAIL because the alias and `_submit_target()` do not exist.

- [ ] **Step 3: Replace RP-PCA's duplicate orchestration**

Import the public helper under an unambiguous alias:

```python
from causal_portfolio.execution import execute_target as execute_model_target
```

Rename the RP-PCA helper to `_submit_target`, keep its config construction and
mainnet guard, then replace adapter/planner/executor setup with:

```python
target = load_target_snapshot(target_path)
result = execute_model_target(
    target,
    cfg,
    acknowledge_mainnet=args.mainnet,
)
```

Keep the existing `RuntimeError` on `result.error` and success log. Remove the
unused `plan_rebalance` import. Update `run_once()` to call `_submit_target()`.

- [ ] **Step 4: Document the public call and correct stale API names**

In the execution README, replace the manual wrapper example with:

```python
from causal_portfolio.execution import ExecutionConfig, TargetSnapshot, execute_target

target = TargetSnapshot(
    weights={"btc": 0.3, "eth": 0.2},
    as_of=datetime.now(timezone.utc),
    strategy="my-model",
)
result = execute_target(target, ExecutionConfig())  # dry-run testnet by default
```

State that JSON/CSV producers call `load_target_snapshot()` first. Fix the
stale `rebalancer.py` reference to point at public `execute_target()` while
retaining `execute_plan()` as the lower-level prepared-plan API.

- [ ] **Step 5: Verify the focused integration**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Run the full execution regression gate**

Run:

```powershell
venv\Scripts\python.exe -m pytest causal_portfolio/tests/test_rebalancer.py causal_portfolio/tests/test_orderbook.py causal_portfolio/tests/test_book_aware_submit.py causal_portfolio/tests/test_execute_safety.py causal_portfolio/tests/test_execution_targets.py causal_portfolio/tests/test_execution_audit.py causal_portfolio/tests/test_execution_dashboard_smoke.py causal_portfolio/tests/test_hl_integration.py causal_portfolio/tests/test_rppca_daily.py causal_portfolio/tests/test_strategy_execution_benchmark.py causal_portfolio/tests/test_testnet_execution_benchmark.py -q
git diff --check
```

Expected: all tests pass; `git diff --check` reports no errors.

- [ ] **Step 7: Commit integration and documentation**

```powershell
git add -- causal_portfolio/execution/README.md causal_portfolio/execution/rebalancer.py causal_portfolio/execution/rppca_daily.py causal_portfolio/tests/test_rppca_daily.py docs/superpowers/plans/2026-07-13-model-execution-interface.md
git commit -m "refactor(execution): use shared model handoff"
```

- [ ] **Step 8: Verify identity and push**

```powershell
git log -3 --format="%h %an <%ae> | %cn <%ce> | %s"
git status --short
git push origin execution
```

Expected: only `Sahas266 <sahas266@gmail.com>` appears on new commits; `.codex/`
and `tmp/` remain untracked; `origin/execution` advances successfully.
