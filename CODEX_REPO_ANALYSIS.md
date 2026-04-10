# Codex Repo Analysis

Date: 2026-04-09

## Executive Summary

This repository is a monorepo for a crypto causal inference research and trading stack with three main codebases:

1. `backfill_data/` - the operational data ingestion system
2. `causal_model/` - the Rust CPCM causal estimation engine
3. `causal_portfolio/` - the Python portfolio, backtest, and dashboard layer

There is also a fourth module, `defi_pipeline/`, which is mostly scaffolded and not yet at the same maturity level as the other three.

The repo is real and substantial. The strongest committed pieces today are:

- the provider-based backfill framework in `backfill_data/`
- the Rust causal graph / estimation engine in `causal_model/`
- the Python CPCM backtest and Streamlit dashboard in `causal_portfolio/`

The biggest caveat is that the documentation and Claude project memory describe a larger operational state than what is currently committed in git. In particular, Artemis and Hyperliquid are described as active providers in the docs, but their committed provider implementations are not present in the working tree anymore.

## Repository Structure

### `backfill_data/`

Purpose:

- Pull historical crypto, macro, and on-chain data from multiple providers
- Normalize records into shared schemas
- Write everything into Supabase
- Track backfill progress and resume failed work

Key implementation points:

- CLI entrypoint: `backfill_data/backfill.py`
- Orchestration: `backfill_data/src/core/orchestrator.py`
- Auto-discovery registry: `backfill_data/src/providers/registry.py`
- Config manifest: `backfill_data/config/coin_manifest.json`
- Endpoint catalog: `backfill_data/config/endpoints/`

Committed provider implementations that are clearly present:

- `allium`
- `coingecko`
- `coinmetrics`
- `defillama`
- `dune`
- `fred`

Directories named `artemis/` and `hyperliquid/` exist under `backfill_data/src/providers/`, but in the current tree they only contain `__pycache__` and no committed source files.

Operational patterns:

- endpoint configs are JSON files
- configs can be loaded recursively via `--recursive`
- every provider writes to common tables using `(provider, asset/metric/market, time)` style keys
- rows carry `provider_priority`
- backfill progress is checkpointed in `backfill_progress`

Current manifest scope:

- 26 tracked assets from the `coins` file
- 155 endpoint JSON files under `backfill_data/config/endpoints/`

### `causal_model/`

Purpose:

- Implement the CPCM causal graph and estimation pipeline in Rust
- Pull wide data panels from Supabase
- Build factors and covariates
- Identify causal effects
- Estimate OLS and 2SLS models

Key implementation points:

- CLI entrypoint: `causal_model/crates/cpcm-cli/src/main.rs`
- full pipeline: `causal_model/crates/cpcm-cli/src/pipeline.rs`
- DAG definition: `causal_model/crates/cpcm-core/src/cpcm_dag.rs`
- data layer: `causal_model/crates/cpcm-data/`
- factor layer: `causal_model/crates/cpcm-factors/`
- estimation layer: `causal_model/crates/cpcm-estimate/`

This is not a stub. It compiles and its tests pass locally.

### `causal_portfolio/`

Purpose:

- Consume the Supabase-backed CPCM data model from Python
- Build factors and select drivers
- Run backtests
- Expose a Streamlit dashboard

Key implementation points:

- pipeline entrypoint: `causal_portfolio/main.py`
- backtest runner: `causal_portfolio/run_backtest.py`
- Supabase loader: `causal_portfolio/data/supabase_loader.py`
- factor builder: `causal_portfolio/factors/builder.py`
- PINN solver: `causal_portfolio/solvers/v4_pinn.py`
- dashboard: `causal_portfolio/dashboard.py`

This layer is more mature than the original docs imply. It now contains a real backtest engine, multiple solvers, an EKF path, and a sizeable Streamlit interface.

### `defi_pipeline/`

Purpose:

- FastAPI API for real-time DeFi metrics

Status:

- still scaffold / partial implementation
- several endpoints return placeholder responses
- logging, metrics, and cache initialization still contain testing comments

This module is useful as a direction-of-travel artifact, but not yet comparable to the maturity of `backfill_data/` or `causal_model/`.

## Data and Runtime Assumptions

The repo assumes Supabase is the central store for both the backfill system and the CPCM model layers.

Important runtime assumptions visible in code and docs:

- `causal_portfolio` reads from `asset_metrics_best`
- `causal_model` can read from `asset_metrics_best` or `asset_metrics`
- many docs assume `.env` exists at repo root and / or `backfill_data/.env`
- the current checked-in `venv` is incomplete for the Python quant stack

## What Is Verified vs What Is Claimed

### Verified Now

- Backfill provider tests pass for committed providers:
  - `test_allium.py`
  - `test_coinmetrics.py`
  - `test_defillama.py`
  - Result: 59 tests passed
- Rust tests in `causal_model/` pass
  - Result: 61 tests passed across the crates

### Not Verified in Current Venv

- `causal_portfolio` tests do not currently collect because the checked-in environment is missing packages such as:
  - `pandas`
  - `sklearn`
  - `torch`

That means the code exists, but this specific environment is not currently ready to run the full Python quant stack.

## Important Mismatches

### 1. Docs describe Artemis and Hyperliquid as active committed providers

Files affected:

- `CLAUDE.md`
- `backfill_data/BACKFILL_STATUS.md`

Problem:

- the docs describe Artemis and Hyperliquid provider configs, query coverage, and row counts
- the committed provider code and provider config JSON files for those providers are not currently in git

Most likely explanation:

- they existed in local / session state at one point
- later checkout or pull removed them from the working tree
- Claude project memory retained the larger operational picture

### 2. Windows note for `uvloop` does not match `requirements.txt`

The docs correctly warn that `uvloop` should be conditional on Windows, but `requirements.txt` currently pins it unconditionally.

### 3. Docker healthcheck mismatch

`docker-compose.yml` checks `http://localhost:8501/healthz`, but the Streamlit dashboard does not expose a `/healthz` route.

### 4. `funding_basis` remains intentionally missing

Both Python and Rust CPCM factor logic still treat `funding_basis` as a gap because a free source is not wired in.

### 5. `defi_pipeline` is still mostly a framework shell

The repo docs sometimes read as if it is a production pipeline. The code still shows placeholder responses and commented-out production wiring.

## Current Local Worktree State

At analysis time, the git worktree was dirty:

- modified: `CLAUDE.md`
- modified: `backfill_data/BACKFILL_STATUS.md`
- untracked: `backfill_data/scripts/backfill_matic_history.py`

This matters because some of the current documentation appears to be describing local operational state, not just committed code.

## Recommended Interpretation

If you want the most reliable mental model of this repo right now:

- trust `backfill_data/` core orchestration and committed provider code
- trust `causal_model/` Rust pipeline and tests
- trust `causal_portfolio/` code structure, but not the current venv readiness
- treat `defi_pipeline/` as planned / partial
- treat `CLAUDE.md` and `BACKFILL_STATUS.md` as a mix of:
  - committed truth
  - local Supabase truth
  - Claude memory of removed files

## Good Next Cleanup Tasks

1. Reconcile `CLAUDE.md` with the current committed tree.
2. Reconcile `backfill_data/BACKFILL_STATUS.md` with:
   - committed provider implementations
   - actual Supabase state
   - local untracked scripts
3. Decide whether Artemis and Hyperliquid should be:
   - restored from Claude backups
   - intentionally removed from the docs
4. Fix the Windows `uvloop` dependency declaration.
5. Fix or remove the Streamlit `/healthz` Docker healthcheck.
