# Codex Next Session Handoff

Date: 2026-04-10
Branch: `v1`

## Purpose

This file is the single-session handoff for the next agent session.

It summarizes:

- what this repo is
- what was verified locally
- what Claude Code history says happened here
- where docs and committed code do not match
- what is currently dirty in the working tree
- what remains unresolved

Read this first, then use:

- `CODEX_REPO_ANALYSIS.md`
- `CODEX_CLAUDE_SESSION_HISTORY.md`

for the deeper details.

## Repo Summary

This monorepo has four meaningful areas:

1. `backfill_data/`
   - multi-provider historical data backfill system
   - operational center of the repo
   - writes normalized records to Supabase
2. `causal_model/`
   - Rust CPCM causal graph and estimation engine
   - builds factors, identifies effects, runs OLS and 2SLS
3. `causal_portfolio/`
   - Python CPCM layer
   - Supabase loader, factor builder, solver stack, backtest engine, Streamlit dashboard
4. `defi_pipeline/`
   - FastAPI scaffold for real-time DeFi metrics
   - still mostly placeholder / partial

## Main Technical Findings

### `backfill_data/` is the most operationally mature committed module

Important files:

- `backfill_data/backfill.py`
- `backfill_data/src/core/orchestrator.py`
- `backfill_data/src/providers/registry.py`
- `backfill_data/config/coin_manifest.json`
- `backfill_data/config/endpoints/`

Important behaviors confirmed in code:

- provider auto-discovery via `ProviderRegistry.auto_discover()`
- recursive endpoint loading via `backfill.py --recursive`
- provider-agnostic orchestration
- `provider_priority` written into records
- `backfill_progress` used for checkpointing

### `causal_model/` is real and tested

Important files:

- `causal_model/crates/cpcm-cli/src/main.rs`
- `causal_model/crates/cpcm-cli/src/pipeline.rs`
- `causal_model/crates/cpcm-core/src/cpcm_dag.rs`

The Rust pipeline is not a sketch. It builds and tests cleanly.

### `causal_portfolio/` is more mature than older docs imply

Important files:

- `causal_portfolio/main.py`
- `causal_portfolio/run_backtest.py`
- `causal_portfolio/data/supabase_loader.py`
- `causal_portfolio/factors/builder.py`
- `causal_portfolio/solvers/v4_pinn.py`
- `causal_portfolio/dashboard.py`

It contains:

- Supabase-backed panel loading
- factor construction
- Combo driver selection
- V1 and V4 solver paths
- EKF integration
- walk-forward backtesting
- Streamlit dashboard

### `defi_pipeline/` is still scaffold-level

Important files:

- `defi_pipeline/app/main.py`
- `defi_pipeline/app/api/v1/router.py`

Status:

- many endpoints still return placeholder responses
- some production wiring is commented out
- should be treated as planned / partial

## Verification Results

### Passed

1. Backfill provider tests:
   - `backfill_data/tests/providers/test_allium.py`
   - `backfill_data/tests/providers/test_coinmetrics.py`
   - `backfill_data/tests/providers/test_defillama.py`
   - Result: 59 tests passed

2. Rust tests:
   - `cargo test -q` in `causal_model/`
   - Result: 61 tests passed across the crates

### Not currently runnable in this checked-in venv

`causal_portfolio` tests failed at collection because the current environment is missing packages such as:

- `pandas`
- `sklearn`
- `torch`

Interpretation:

- the code exists
- the current local `venv` is not ready for the full Python quant stack

## Biggest Mismatches Between Docs and Current Tree

### Artemis and Hyperliquid are described as active, but are not committed now

Docs currently claim active Artemis / Hyperliquid support in:

- `CLAUDE.md`
- `backfill_data/BACKFILL_STATUS.md`

But in the current tree:

- `backfill_data/src/providers/artemis/` exists only as an empty directory with `__pycache__`
- `backfill_data/src/providers/hyperliquid/` exists only as an empty directory with `__pycache__`
- there is no committed:
  - `backfill_data/config/providers/artemis.json`
  - `backfill_data/config/providers/hyperliquid.json`

This is not guesswork. Claude session history shows those files existed locally in prior sessions and were later removed from the working tree.

### `requirements.txt` conflicts with the Windows `uvloop` note

Docs say `uvloop` must be conditional on Windows.

Current `requirements.txt` still pins:

- `uvloop==0.22.1`

unconditionally.

### Docker healthcheck mismatch

`docker-compose.yml` checks:

- `http://localhost:8501/healthz`

But the Streamlit dashboard does not expose `/healthz`.

### `funding_basis` is still intentionally missing

Both Python and Rust factor logic still treat `funding_basis` as a gap due to missing free data wiring.

### `defi_pipeline` docs can overstate readiness

The code still contains placeholders and testing comments.

## Claude Code History Summary

Primary session artifacts live under:

- `C:\\Users\\sahas\\.claude\\projects\\c--Users-sahas-Github-Repos-causal-inference-factor-model\\`

Most important sessions:

1. `19f3d61e-2cb3-44c0-be75-c7e51ad70561.jsonl`
   - huge session
   - starts with: `Add All 26 Coins to Backfill System`
   - created / modified manifest, generator, recursive config loading, orchestration, docs

2. `1b324927-6ab3-4a01-b1d7-571fe20cdbd7.jsonl`
   - Allium / FM-13 session
   - aligns with committed Allium provider work

3. `98929c20-765c-424f-88a2-2cb3c9626630.jsonl`
   - continuation session using `MEMORY.md` as task context

4. `eee4714b-7ca3-4aeb-9f0b-c984be8646c2.jsonl`
   - POL / MATIC lineage session
   - starts with: `Do we have a full price history for POL?`
   - references Artemis and Hyperliquid files that are no longer present in the committed tree

The durable Claude memory file is:

- `C:\\Users\\sahas\\.claude\\projects\\c--Users-sahas-Github-Repos-causal-inference-factor-model\\memory\\MEMORY.md`

Use it carefully:

- it is very useful
- but it records operational / local-session truth, not just committed git truth

## Supabase Status

### Supabase MCP

The user said Supabase MCP was added, but direct MCP access still failed.

Latest observed error:

- `Auth required, when send initialize request`

Meaning:

- the MCP server exists
- it is not authenticated for this session yet

### Local Supabase fallback

A fallback path is available through the repo itself:

- root `.env` is missing
- `backfill_data/.env` exists

The following keys were confirmed present in `backfill_data/.env`:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `ALLIUM_API_KEY`
- `COINGECKO_API_KEY`
- `FRED_API_KEY`
- `ARTEMIS_API_KEY`
- `DUNE_API_KEY`
- Hyperliquid wallet / key values

Relevant client wrapper:

- `backfill_data/src/core/storage/supabase_manager.py`

Important note:

- I started to use the local Supabase fallback path to inspect the live database
- those shell commands were interrupted by the user before completion
- no database summary was completed yet in this session

So:

- database inspection is still pending
- either authenticated Supabase MCP or the local `.env` fallback should be used next

## Files Created During This Session

New repo-local notes created:

- `CODEX_REPO_ANALYSIS.md`
- `CODEX_CLAUDE_SESSION_HISTORY.md`

These should be preserved. They contain the main analysis and history reconstruction already completed.

## Current Worktree State

Observed `git status --short --branch`:

- branch: `v1`
- modified: `CLAUDE.md`
- modified: `backfill_data/BACKFILL_STATUS.md`
- untracked: `CODEX_CLAUDE_SESSION_HISTORY.md`
- untracked: `CODEX_REPO_ANALYSIS.md`
- untracked: `backfill_data/scripts/backfill_matic_history.py`

Interpretation:

- there are important local docs and lineage notes not committed yet
- those local files likely reflect real work but are not part of the current git baseline

## Recommended Next Actions

### Best next action

Inspect the live Supabase database and reconcile it with:

- `CLAUDE.md`
- `backfill_data/BACKFILL_STATUS.md`
- current committed provider implementations

Priority order:

1. retry authenticated Supabase MCP
2. if still broken, use `backfill_data/.env` plus `SupabaseManager`

### After that

1. Decide whether Artemis and Hyperliquid should be:
   - restored from Claude backups
   - or removed from the docs
2. Normalize `CLAUDE.md` against actual committed code
3. Normalize `backfill_data/BACKFILL_STATUS.md` against:
   - actual database state
   - actual committed provider support
4. Fix `uvloop` Windows dependency handling
5. Fix or remove the Docker Streamlit `/healthz` healthcheck

## Fast Start Checklist For Next Session

1. Read this file.
2. Read `CODEX_REPO_ANALYSIS.md`.
3. Read `CODEX_CLAUDE_SESSION_HISTORY.md`.
4. Check `git status`.
5. Retry Supabase MCP auth.
6. If MCP still fails, use `backfill_data/.env` and `SupabaseManager` to inspect:
   - public tables
   - public views
   - row counts
   - provider coverage
   - asset coverage
   - date ranges
7. Reconcile the docs with the real DB state.
