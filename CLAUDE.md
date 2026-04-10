# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Repository Overview

This monorepo has four meaningful areas:

1. `backfill_data/`
   - operational multi-provider historical data backfill system
   - writes normalized records into Supabase
2. `causal_model/`
   - Rust CPCM causal graph and estimation engine
   - builds factors, identifies effects, and runs OLS / 2SLS
3. `causal_portfolio/`
   - Python CPCM data, factor, backtest, and dashboard layer
   - consumes the Supabase warehouse
4. `defi_pipeline/`
   - FastAPI real-time DeFi metrics API
   - still scaffold / partial compared with the other three modules

## Environment

- Python: `3.12.6` on Windows
- Virtual environment: `./venv/` and use `./venv/Scripts/python.exe`
- `uvloop` must stay conditional on Windows:
  - `uvloop>=0.19.0; sys_platform != "win32"`
- Supabase project: `jnulpcqpftnwvknwuqpa`
- Required env vars for warehouse-backed work:
  - `SUPABASE_URL`
  - `SUPABASE_KEY`
- Optional provider env vars still relevant to the committed provider tree:
  - `ALLIUM_API_KEY`
  - `COINMETRICS_API_KEY`
  - `DEFILLAMA_API_KEY`
  - `DUNE_API_KEY`
  - `COINGECKO_API_KEY`
  - `FRED_API_KEY`

Important:

- The live warehouse includes Artemis and Hyperliquid data at scale.
- The current committed worktree does not include Artemis or Hyperliquid provider source/config files.
- Treat Supabase as the source of truth for live data coverage.
- Treat the repo as the source of truth for what can be re-run from the current committed tree.

## Core Commands

### `backfill_data/`

Run from `backfill_data/` unless noted.

```bash
cd backfill_data

# List committed providers
python backfill.py --list-providers

# Validate a config
python backfill.py --config config/endpoints/btc/btc_coinmetrics.json --validate-only

# Run a single coin directory
python backfill.py --config config/endpoints/sol/

# Run all committed endpoint configs recursively
python backfill.py --config config/endpoints/ --recursive

# Orchestrate all coins
python scripts/backfill_all_coins.py
python scripts/backfill_all_coins.py --asset sol,btc
python scripts/backfill_all_coins.py --dry-run

# Derived metrics
python scripts/compute_derived_metrics.py
python scripts/compute_derived_metrics.py --asset btc

# Resume failed committed backfills
python backfill.py --resume-failed
```

Tests:

```bash
cd backfill_data
python -m pytest
python -m pytest tests/providers/test_coinmetrics.py -v
python -m pytest tests/providers/test_allium.py -v
python -m pytest tests/providers/test_defillama.py -v
python -m pytest -m unit
python -m pytest -m integration
```

### `causal_model/`

```bash
cd causal_model
cargo test -q
```

### `causal_portfolio/`

```bash
# From repo root
python -m causal_portfolio.main --assets btc,eth,sol --m 3 --start 2022-01-01 --end 2025-12-31
python -m causal_portfolio.run_backtest --solver v1 --m 3 --assets btc,eth,sol
python -m causal_portfolio.run_backtest --solver v4 --use-ekf --rebalance-freq 5
streamlit run causal_portfolio/dashboard.py
```

### `defi_pipeline/`

```bash
cd defi_pipeline
uvicorn app.main:app --reload
alembic upgrade head
```

## Architecture

### `backfill_data/`

Key files:

- `backfill_data/backfill.py`
- `backfill_data/src/core/orchestrator.py`
- `backfill_data/src/providers/registry.py`
- `backfill_data/src/core/storage/progress_tracker.py`
- `backfill_data/config/coin_manifest.json`
- `backfill_data/config/database_schema.sql`
- `backfill_data/config/endpoints/`

Design:

- provider auto-discovery through `ProviderRegistry.auto_discover()`
- provider-agnostic orchestration
- streaming fetch + batch upsert
- checkpointing in `backfill_progress`
- shared Supabase warehouse tables and best-provider views

Committed providers clearly present in the current tree:

- `allium`
- `coingecko`
- `coinmetrics`
- `defillama`
- `dune`
- `fred`

Warehouse-only historical providers right now:

- `artemis`
- `hyperliquid`

Current state of those warehouse-only providers in the repo:

- `backfill_data/src/providers/artemis/` contains only `__pycache__`
- `backfill_data/src/providers/hyperliquid/` contains only `__pycache__`
- no committed `config/providers/artemis.json`
- no committed `config/providers/hyperliquid.json`
- no committed Artemis / Hyperliquid endpoint JSON files under `config/endpoints/`

### `causal_model/`

Key files:

- `causal_model/crates/cpcm-cli/src/main.rs`
- `causal_model/crates/cpcm-cli/src/pipeline.rs`
- `causal_model/crates/cpcm-core/src/cpcm_dag.rs`
- `causal_model/crates/cpcm-data/src/client.rs`

This is a real, tested Rust pipeline. It can read from `asset_metrics_best` or `asset_metrics`.

### `causal_portfolio/`

Key files:

- `causal_portfolio/main.py`
- `causal_portfolio/run_backtest.py`
- `causal_portfolio/data/supabase_loader.py`
- `causal_portfolio/factors/builder.py`
- `causal_portfolio/solvers/v4_pinn.py`
- `causal_portfolio/dashboard.py`

This layer implements the CPCM pipeline described in `Causal PDE-Control Models for Portfolio Optimization.md`.

Important runtime behavior:

- `CPCMDataLoader` reads from `asset_metrics_best`
- panel data is pivoted wide as `{asset}_{metric}`
- macro data is loaded from `asset='macro'`

### `defi_pipeline/`

Treat this as planned / partial. It is useful context, but it is not yet at the maturity of `backfill_data/`, `causal_model/`, or `causal_portfolio/`.

## Live Warehouse Snapshot

Verified on `2026-04-10`:

- `asset_metrics`: `2,848,307` rows
- `asset_metrics_best`: `2,817,204` rows
- providers: `9`
- assets: `29`
- metrics: `338`
- date range: `2021-01-01` -> `2026-01-01`

Provider row counts in `asset_metrics`:

- `artemis`: `1,556,065`
- `hyperliquid`: `620,890`
- `derived`: `282,597`
- `dune`: `143,246`
- `coinmetrics`: `132,166`
- `defillama`: `52,557`
- `coingecko`: `22,064`
- `fred`: `20,452`
- `allium`: `18,270`

Asset scope:

- target 26-coin universe is present
- additional live assets are `macro`, `matic`, and legacy `lido`

Important mismatch:

- `backfill_progress`, `provider_health`, and `recent_backfill_activity` do not reflect Artemis or Hyperliquid
- those views currently only reflect the committed-provider operational metadata
- for live coverage questions, trust the fact tables first

## CPCM Roadmap Alignment

The roadmap lives in `Causal PDE-Control Models for Portfolio Optimization.md`.

Current warehouse coverage against the roadmap:

- StableFlow: strong
  - Dune stablecoin mint / burn / netflow metrics
  - Artemis stablecoin supply and transfer metrics
  - derived `stablecoin_net_flow_usd`
- Funding Basis: strong
  - Hyperliquid `funding_rate_8h`
  - Hyperliquid `funding_premium`
- Chain Congestion: partial
  - Dune gas utilization and chain activity metrics
- LiqFlow / LP Flow: partial
  - derived `tvl_net_flow_usd`
  - DefiLlama TVL / volume metrics
  - Dune UNI V3 LP-flow-style data
- Staking Yield: partial
  - Artemis staking-related metrics
  - Dune staking queries
- MEV Pressure: proxy-only right now
  - no direct confirmed `mev_*` warehouse series
- CEX / DEX Flow: partial to strong
  - Dune bridge / flow queries
  - derived `dex_cex_volume_ratio`

## Storage Notes

Current database size is about `1,088 MB` after dropping the unused `idx_asset_metrics_metric_time` index on `2026-04-10`.

Almost all of that is `public.asset_metrics`:

- total relation size: `1,190,313,984` bytes
- heap/table: `405,659,648` bytes
- indexes: `784,506,880` bytes

Storage change applied:

- dropped `public.idx_asset_metrics_metric_time`
- database size moved from about `1,147 MB` to `1,088 MB`
- `public.asset_metrics` index footprint moved from `784,506,880` bytes to `721,747,968` bytes

Remaining guidance:

- keep `idx_asset_metrics_priority` because it supports the best-provider view pattern
- do not remove additional indexes unless they are validated against real workloads

## Supabase Schema

Public tables:

- `asset_metrics`
- `exchange_metrics`
- `market_trades`
- `market_candles`
- `pair_candles`
- `market_orderbooks`
- `market_open_interest`
- `market_funding_rates`
- `market_liquidations`
- `market_implied_volatility`
- `market_greeks`
- `backfill_progress`

Public views:

- `asset_metrics_best`
- `exchange_metrics_best`
- `market_orderbooks_best_quotes`
- `data_coverage_by_provider`
- `provider_health`
- `recent_backfill_activity`

## Practical Guidance

- If you need the current truth about Artemis or Hyperliquid, query Supabase, not the local provider tree.
- If you need to restore local Artemis or Hyperliquid execution, use session history / backups rather than assuming the code is still committed.
- If you need storage relief, investigate index removal before deleting CPCM-relevant data.
- If docs and code disagree, prefer:
  1. live Supabase fact tables for coverage
  2. committed code for runnable local behavior
  3. session notes for historical context
