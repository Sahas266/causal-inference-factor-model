# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This monorepo contains three interconnected systems for a **causal inference factor model** applied to crypto markets:

1. **`backfill_data/`** — Multi-provider crypto data backfill system (primary active module)
2. **`causal_portfolio/`** — Structural Causal Model (SCM) for portfolio optimization using DoWhy/EconML
3. **`defi_pipeline/`** — FastAPI-based real-time DeFi metrics pipeline (planned/in-progress)

## Environment

- **Python**: 3.12.6 on Windows
- **Virtual environment**: `./venv/` — use `./venv/Scripts/python.exe` for commands
- **`uvloop` is Windows-incompatible** — always keep it conditional: `uvloop>=0.19.0; sys_platform != "win32"` in `requirements.txt`
- **Supabase**: used as the database backend across all modules (project ID: `jnulpcqpftnwvknwuqpa`)
- **Required env vars** (in `backfill_data/.env`): `SUPABASE_URL`, `SUPABASE_KEY`, `ALLIUM_API_KEY`
- **Optional env vars**: `COINMETRICS_API_KEY` (community tier works without it), `DEFILLAMA_API_KEY` (Pro endpoints only), `DUNE_API_KEY`
- **Supabase auth**: Use `service_role` key (not `sbp_...` management token) for data API writes

## Commands

All commands should be run from the `backfill_data/` directory unless noted.

### Running the backfill system

```bash
# From repo root
cd backfill_data

# List available providers
python backfill.py --list-providers

# Validate an endpoint config (no data fetch)
python backfill.py --config config/endpoints/btc_metrics.json --validate-only

# Run a specific endpoint backfill
python backfill.py --config config/endpoints/btc_metrics.json

# Run all ETH endpoints (primary set)
python backfill.py --config config/endpoints/eth/

# Run ETH fallback endpoints (Allium)
python backfill.py --config config/endpoints/eth_fallback/

# Run per-provider convenience scripts
python scripts/run_provider_backfill.py
python scripts/backfill_defillama.py
python scripts/backfill_coinmetrics.py
python scripts/backfill_dune.py
python scripts/backfill_coingecko.py

# Resume failed backfills
python backfill.py --resume-failed
```

### Tests

```bash
# From backfill_data/
cd backfill_data

# Run all tests
python -m pytest

# Run provider-specific tests
python -m pytest tests/providers/test_coinmetrics.py -v
python -m pytest tests/providers/test_allium.py -v
python -m pytest tests/providers/test_defillama.py -v

# Run by marker (unit/integration/slow)
python -m pytest -m unit
python -m pytest -m integration

# Run with coverage
python -m pytest --cov=src --cov-report=html
```

`pytest.ini` is in `backfill_data/` and sets `testpaths = tests`.

### Code quality

```bash
# From backfill_data/
black src/ tests/
isort src/ tests/
mypy src/
flake8 src/ tests/
```

### DeFi pipeline (defi_pipeline/)

```bash
cd defi_pipeline
uvicorn app.main:app --reload
alembic revision --autogenerate -m "migration message"
alembic upgrade head
```

## Architecture

### backfill_data — Plugin-Based Provider System

The core design principle: **the orchestrator never knows about provider-specific details**. Providers implement a standard interface and are auto-discovered. Each provider has its own separate pipeline — endpoint configs explicitly name which provider to use.

**Data flow:**
```
CLI (backfill.py)
  → BackfillOrchestrator (src/core/orchestrator.py)
    → ProviderRegistry.auto_discover() — scans src/providers/ subdirs
    → validate_endpoint() per provider
    → fetch_data_stream() — generator, memory-efficient pagination
    → DatabaseWriter.upsert_batch() → Supabase
    → ProgressTracker — checkpoint/resume in backfill_progress table
```

**Key files:**
- `src/core/interfaces.py` — `DataProviderInterface` and `RateLimiterInterface` ABCs. Every provider implements these exactly.
- `src/core/orchestrator.py` — `BackfillOrchestrator`: validates, threads, streams, writes. Supports `provider_instance_id` for multi-entry dedup.
- `src/core/utils/config_loader.py` — Config loading with relative path resolution (checks path exists before prefixing)
- `src/core/storage/progress_tracker.py` — Checkpoint/resume with JSON-safe datetime serialization
- `src/providers/registry.py` — `ProviderRegistry`: auto-discovery via `pkgutil.iter_modules`
- `src/providers/coinmetrics/` — market data: asset metrics, trades, candles, orderbooks, derivatives
- `src/providers/allium/` — token prices and DEX trades via Developer REST API (`ALLIUM_API_KEY`)
- `src/providers/defillama/` — TVL, DEX volumes, fees, stablecoin flow, coin prices (free + Pro)
- `src/schemas/` — Pydantic models for each data type
- `config/providers/{name}.json` — provider-level config (API URL, rate limits, retry policy)
- `config/endpoints/{name}.json` — endpoint configs specifying table, primary keys, provider params
- `config/endpoints/eth/` — ETH-specific primary endpoint pack (11 configs across CoinMetrics + DefiLlama)
- `config/endpoints/eth_fallback/` — Allium fallback endpoints for ETH price data
- `scripts/` — Per-provider convenience backfill scripts
- `ETH_DATA_POINTS_CATALOG.md` — Prioritized ETH metrics catalog with provider mapping

**Adding a new provider** (4 steps only):
1. `mkdir src/providers/{name}` with `provider.py`, `client.py`, `rate_limiter.py`, `transformer.py`, `__init__.py`
2. Implement `DataProviderInterface` in `provider.py`
3. In `__init__.py`: `ProviderRegistry.register(MyProvider)`
4. Add `config/providers/{name}.json`

**Database (Supabase):** Progress tracked in `backfill_progress` table. Data tables use `(provider, asset/metric/market, time)` composite primary keys for multi-provider redundancy. `asset_metrics_best` view selects one provider per data point. The `asset_metrics` table has a `provider_priority` column and uses `created_at` (not `inserted_at`).

**Implemented providers & endpoint types:**

*CoinMetrics* (`config/providers/coinmetrics.json`) — community tier at `community-api.coinmetrics.io/v4` (no key required):
- `timeseries/asset-metrics`, `timeseries/exchange-metrics`, `timeseries/market-trades`
- `timeseries/market-orderbooks`, `timeseries/pair-candles`, `timeseries/market-candles`
- `timeseries/market-open-interest`, `timeseries/market-liquidations`, `timeseries/market-funding-rates`
- `timeseries/market-implied-volatility`, `timeseries/market-greeks`
- **Community restrictions**: derivatives/market endpoints (candles, open interest, funding rates, liquidations) return 403. Asset metrics and pair candles work.
- Endpoint configs: `btc_metrics.json`, `market_candles.json`, `market_funding_rates.json`, ETH configs in `config/endpoints/eth/`

*Allium* (`config/providers/allium.json`) — requires `ALLIUM_API_KEY`:
- Uses **Developer REST API** (subscription-based), NOT Explorer SQL (which requires separate compute credits and is exhausted)
- `developer/prices/history` — POST, OHLCV token prices by contract address. Granularity: `15s | 1m | 5m | 1h | 1d`
- `developer/{chain}/dex/trades` — GET, DEX trade events (paginated)
- `developer/{chain}/raw/blocks` — GET, raw block data for EVM chains
- `developer/bitcoin/raw/blocks`, `developer/bitcoin/raw/transactions` — GET, Bitcoin-specific data
- Auth: `X-API-KEY` header against `https://api.allium.so/api/v1`
- OpenAPI spec: `https://api.allium.so/openapi.json` — useful for discovering new endpoints
- Note: Allium MCP server (mcp.allium.so) uses Explorer credits (exhausted) — use REST API directly
- Endpoint configs: `allium_eth_on_chain.json`, `allium_btc_on_chain.json`, `allium_dex_volumes.json` (being migrated to Developer API)

*DefiLlama* (`config/providers/defillama.json`) — no key needed for free endpoints:
- `protocol/tvl`, `chain/tvl`, `dex/summary`, `fees/summary`, `stablecoin/charts`, `coin/chart`
- `yields/pool-chart` (Pro only — requires `DEFILLAMA_API_KEY` in URL path)
- Multi-domain URLs: `api.llama.fi` (TVL/DEX/fees), `stablecoins.llama.fi` (stablecoin charts), `coins.llama.fi` (coin prices)
- Pro API: `pro-api.llama.fi/{api_key}/...`
- Most endpoints return full history in one call; transformer handles date filtering
- Endpoint configs: `defillama_chain_tvl.json`, `defillama_protocol_tvl.json`, `defillama_stablecoin_flow.json`, `defillama_dex_volumes.json`, ETH configs in `config/endpoints/eth/`

### causal_portfolio — SCM-Based Portfolio Model

Uses **DoWhy** for causal identification and **EconML** for estimation. The SCM graph defines causal relationships between crypto factors (macro, on-chain, market microstructure) and asset returns.

- `scm/loaders.py` — data loading (currently CSV mocks; planned Supabase integration)
- `scm/shocks.py` — instrumental variable (IV) / exogenous shock construction (one instrument per factor)
- `scm/graph.py` — causal DAG definition
- `scm/model.py` — DoWhy SCM instantiation
- `scm/estimation.py` — causal effect identification and estimation
- `main.py` — wires all components together

### defi_pipeline — Real-Time DeFi Metrics API

FastAPI + Celery + PostgreSQL/TimescaleDB pipeline collecting 7 on-chain indicators: Liq Flow, Stableflow, Funding Basis, Chain Congestion, Staking Yield, MEV Pressure, CEX/DEX Flow.

- `app/collectors/` — data collection logic (implements `base.py` interface)
- `app/providers/` — external API integrations
- `app/api/v1/` — REST endpoints
- `app/core/` — DB, cache (Redis), config, exceptions
- `docker/docker-compose.yml` — containerized deployment

## ETH Backfill Status

Current state (as of March 9, 2026): **61,395 rows** in `asset_metrics` covering 2021-01-01 → 2026-01-01. See `ETH_BACKFILL_PLAN.md`, `ETH_BACKFILL_WORKLOG.md`, and `ETH_DATA_POINTS_CATALOG.md` for details.

**Backfilled data (3 providers):**
- *CoinMetrics* (31,059 rows): PriceUSD, CapMrktCurUSD, TxCnt, AdrActCnt, SplyCur, BlkCnt, HashRate, ROI30d, FeeTotNtv, IssTotNtv, FlowInExNtv, FlowOutExNtv, TxTfrCnt
- *DefiLlama* (21,201 rows): ethereum chain TVL, protocol TVL (aave/uniswap/curve/lido/makerdao), DEX volumes (uniswap/curve), fees (uniswap/lido/aave), stablecoin circulating
- *Allium* (9,135 rows): WETH OHLCV (price_usd, open/high/low/close_usd) as fallback

**Fixed bugs (this session):**
- DefiLlama stablecoin flow string-epoch timestamps — `_unix_to_iso`/`_in_range` now cast `int(ts)`
- DefiLlama DEX volumes — transformer reads `totalDataChart` not just `dailyVolume`
- CoinMetrics 403 — now treated as `fatal: True` (no retry)
- Supabase disconnect on large upserts — database writer retries transient connection errors

**Remaining issues:**
- CoinMetrics derivatives endpoints: 403 on community tier — configs moved to `config/endpoints/eth_disabled/`
- Allium endpoint configs (`allium_*.json`) still reference old `explorer/sql` — need migration to Developer API
- Allium tests (`test_allium.py`) still test V1 SQL interface — need rewrite for V2 Developer API
- Dune requires pre-saved queries in the Dune UI — API cannot run ad-hoc SQL
- CoinGecko free API limits historical data to last 365 days; Pro key needed for full range

**Provider priority**: CoinMetrics > DefiLlama > CoinGecko > Allium (fallback only).

## Supabase Schema

Project ID: `jnulpcqpftnwvknwuqpa` (may be INACTIVE — call `restore_project` if connection times out).

**Tables:** `asset_metrics`, `exchange_metrics`, `market_trades`, `market_candles`, `pair_candles`, `market_orderbooks`, `market_open_interest`, `market_funding_rates`, `market_liquidations`, `market_implied_volatility`, `market_greeks`, `backfill_progress`

**Views:** `asset_metrics_best`, `exchange_metrics_best`, `market_orderbooks_best_quotes`, `data_coverage_by_provider`, `provider_health`, `recent_backfill_activity`

## MCP Servers

- **Atlassian**: enabled via plugin, domain `whitestarcapital.atlassian.net`
- **Supabase**: enabled via plugin
- **Allium**: configured at user scope (`mcp.allium.so`) — MCP server credits are separate from the REST API key
- **CoinGecko**: added to project `.mcp.json` (`mcp-remote` → `https://mcp.pro-api.coingecko.com/mcp`) — requires Claude Code restart to activate

## Test Markers

Tests use pytest markers defined in `backfill_data/pytest.ini`:
- `unit` — no external dependencies
- `integration` — requires live API or database
- `slow` — long-running tests
- `requires_api_key` — needs `COINMETRICS_API_KEY`
- `requires_database` — needs Supabase connection
