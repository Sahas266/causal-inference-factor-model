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
- **Optional env vars**: `COINMETRICS_API_KEY` (community tier works without it), `DEFILLAMA_API_KEY` (Pro endpoints only), `DUNE_API_KEY`, `COINGECKO_API_KEY` (Pro tier for full history), `FRED_API_KEY` (FRED economic data)
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
python backfill.py --config config/endpoints/btc/btc_coinmetrics.json --validate-only

# Run a single coin's endpoints
python backfill.py --config config/endpoints/sol/

# Run all endpoints recursively (all 26 coins)
python backfill.py --config config/endpoints/ --recursive

# Run all coins via orchestration script
python scripts/backfill_all_coins.py                        # all coins
python scripts/backfill_all_coins.py --asset sol,btc        # specific coins
python scripts/backfill_all_coins.py --dry-run              # list what would run

# Generate endpoint configs from coin manifest
python scripts/generate_coin_configs.py

# Compute derived metrics (volatility, ratios)
python scripts/compute_derived_metrics.py                   # all assets with price data
python scripts/compute_derived_metrics.py --asset btc       # single asset

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
- `src/providers/coingecko/` — market chart (price/mcap/volume timeseries) and coin data (ath/atl/supply snapshots)
- `src/providers/dune/` — pre-saved query results from Dune Analytics (`DUNE_API_KEY`)
- `src/providers/fred/` — FRED economic data series (`FRED_API_KEY`)
- `src/schemas/` — Pydantic models for each data type
- `config/providers/{name}.json` — provider-level config (API URL, rate limits, retry policy)
- `config/coin_manifest.json` — master 26-coin definitions (tickers, provider IDs, date ranges)
- `config/endpoints/{coin}/` — per-coin endpoint configs (25 directories, ~78 generated configs)
- `config/endpoints/macro/` — FRED macro endpoint configs (8 configs, 30 series)
- `config/endpoints/eth/` — ETH-specific endpoint pack (17 configs across CoinMetrics, DefiLlama, CoinGecko, Dune)
- `config/endpoints/eth_disabled/` — CoinMetrics derivatives endpoints blocked on community tier (5 configs)
- `config/endpoints/eth_fallback/` — Allium fallback endpoints for ETH price data
- `scripts/generate_coin_configs.py` — reads manifest, generates per-coin endpoint configs
- `scripts/backfill_all_coins.py` — orchestrates backfill across all coins (--asset, --dry-run)
- `scripts/compute_derived_metrics.py` — computes realized volatility, DEX/CEX ratio (--asset)
- `scripts/` — Per-provider convenience backfill scripts
- `BACKFILL_STATUS.md` — comprehensive status of all 26 coins across all providers

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
- Endpoint configs: `allium_eth_on_chain.json`, `allium_btc_on_chain.json`, `allium_dex_volumes.json` (migrated to Developer API)

*DefiLlama* (`config/providers/defillama.json`) — no key needed for free endpoints:
- `protocol/tvl`, `chain/tvl`, `dex/summary`, `fees/summary`, `stablecoin/charts`, `coin/chart`
- `yields/pool-chart` (Pro only — requires `DEFILLAMA_API_KEY` in URL path)
- Multi-domain URLs: `api.llama.fi` (TVL/DEX/fees), `stablecoins.llama.fi` (stablecoin charts), `coins.llama.fi` (coin prices)
- Pro API: `pro-api.llama.fi/{api_key}/...`
- Most endpoints return full history in one call; transformer handles date filtering
- Endpoint configs: `defillama_chain_tvl.json`, `defillama_protocol_tvl.json`, `defillama_stablecoin_flow.json`, `defillama_dex_volumes.json`, ETH configs in `config/endpoints/eth/`

*CoinGecko* (`config/providers/coingecko.json`) — Pro API via `COINGECKO_API_KEY`:
- `market_chart` — `/coins/{id}/market_chart/range`: daily price, market cap, volume timeseries
- `coin_data` — `/coins/{id}`: snapshot data (fdv, ath, atl, total_supply, max_supply)
- Free tier: `api.coingecko.com/api/v3` (30 req/min, last 365 days via `/market_chart?days=365`)
- Pro tier: `pro-api.coingecko.com/api/v3` (500 req/min, full historical range via `/market_chart/range`)
- CoinGecko MCP also available via `.mcp.json` for interactive queries
- Endpoint configs: `config/endpoints/eth/eth_coingecko.json`

*Dune Analytics* (`config/providers/dune.json`) — requires `DUNE_API_KEY`:
- `query_results` — `/query/{id}/results`: fetches pre-saved query results
- Queries must be created in the Dune UI first; the API only reads results
- Each endpoint config specifies a `query_id`
- Rate limit: 40 req/min on standard tier
- Dune CLI installed at `~/.local/bin/dune.exe` for interactive query management
- Active queries: 6811495 (staking), 6811496 (burn), 6811497 (whales), 6811498 (bridges), 6811499 (CEX flows), 6815240 (staking APR), 6815241 (net issuance), 6815242 (L2 settlement), 6815244 (stablecoin netflow), 6830995 (SOL activity), 6830996 (SOL DEX), 6830997 (SOL staking), 6830998 (BNB activity), 6830999 (BNB DEX), 6831000 (AVAX activity), 6831001 (AVAX DEX)
- Endpoint configs: `config/endpoints/eth/eth_dune_*.json` (12 configs), `config/endpoints/sol/sol_dune_*.json` (3), `config/endpoints/bnb/bnb_dune_*.json` (5), `config/endpoints/avax/avax_dune_*.json` (5)

*FRED* (`config/providers/fred.json`) — requires `FRED_API_KEY`:
- `series/observations` — `/fred/series/observations`: historical economic time series
- Supports comma-separated `series_ids` per config — each series fetched independently
- Rate limit: 120 req/min
- 30 series across 8 endpoint configs in `config/endpoints/macro/`:
  - `fred_interest_rates.json`: DFF, DGS2, DGS10, DGS30, DFEDTARU, T10Y2Y, T10Y3M
  - `fred_inflation.json`: CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, T5YIE, T10YIE, MICH
  - `fred_money_supply.json`: M2SL, WALCL, RRPONTSYD
  - `fred_risk_volatility.json`: VIXCLS, BAMLH0A0HYM2, TEDRATE
  - `fred_labor_growth.json`: UNRATE, PAYEMS, ICSA, GDPC1, INDPRO
  - `fred_commodities.json`: DCOILWTICO, PPIACO
  - `fred_financial_conditions.json`: NFCI, STLFSI2
  - `fred_dollar.json`: DTWEXBGS
- All stored with `asset='macro'`, `metric=<series_id>`, date range 2021-01-01 to 2026-01-01

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

## Backfill Status (26 Coins)

Current state (as of March 15, 2026): **~430,300 rows** in `asset_metrics` across 7 providers, covering 26 target coins + macro data. See `BACKFILL_STATUS.md` for the full per-coin breakdown.

**26 coins:** USDC, USDT, USDe, BTC, ETH, BNB, HYPE, XRP, PENDLE, UNI, JUP, TAO, LINK, ZEC, ENA, MORPHO, AERO, SOL, AVAX, POL, WLFI, CRV, AAVE, PEPE, SHIB, DOGE

**Rows by provider:**
- *CoinMetrics* (140,679 rows): btc, eth, bnb, xrp, doge, zec, aave, uni, link, usdc, usdt — 3-13 metrics each
- *Dune* (138,137 rows): ETH (42 metrics, 12 queries) + SOL/BNB/AVAX (activity, DEX volume, SOL staking, block congestion, liquidations, flashloans)
- *Derived* (49,330 rows): all 26 coins — realized_volatility_7d/30d, dex_cex_volume_ratio (ETH only)
- *DefiLlama* (40,667 rows): chain TVL (btc, bnb, sol, avax, pol, hype) + protocol TVL (aave, uni, crv, pendle, morpho, ena, jup) + ETH ecosystem
- *CoinGecko* (22,772 rows): all 26 coins — snapshots + market_chart timeseries (last 365 days via free tier)
- *FRED* (20,452 rows): 30 macro series (interest rates, inflation, money supply, risk, labor, commodities, dollar)
- *Allium* (18,270 rows): BTC + ETH OHLCV fallback

**Known gaps:**
- CoinGecko Pro key: unlocks full historical range for market_chart (currently limited to last 365 days)
- CoinMetrics Pro key: derivatives (OI, funding, liquidations), Tier 2/3 assets (SOL, AVAX, POL, SHIB)
- Funding Basis (perp funding rates): no free source — requires CoinMetrics Pro or Hyperliquid API

**Dune query IDs:** 6811495 (staking), 6811496 (burn), 6811497 (whales), 6811498 (bridges), 6811499 (CEX flows), 6815240 (staking APR), 6815241 (net issuance), 6815242 (L2 settlement), 6815244 (stablecoin netflow), 6830995 (SOL activity), 6830996 (SOL DEX), 6830997 (SOL staking), 6830998 (BNB activity), 6830999 (BNB DEX), 6831000 (AVAX activity), 6831001 (AVAX DEX), 6831658 (ETH block congestion/MEV), 6831661 (ETH flashloans/LP flow), 6831664 (ETH liquidations), 6831685 (BNB block congestion), 6831686 (AVAX block congestion), 6831687 (BNB liquidations), 6831688 (AVAX liquidations), 6831689 (BNB flashloans), 6831690 (AVAX flashloans)

**Provider priority**: CoinMetrics (1) > FRED (1) > DefiLlama (2) > CoinGecko (3) > Allium (4) > Derived (99).

## Supabase Schema

Project ID: `jnulpcqpftnwvknwuqpa` (may be INACTIVE — call `restore_project` if connection times out).

**Tables:** `asset_metrics`, `exchange_metrics`, `market_trades`, `market_candles`, `pair_candles`, `market_orderbooks`, `market_open_interest`, `market_funding_rates`, `market_liquidations`, `market_implied_volatility`, `market_greeks`, `backfill_progress`

**Views:** `asset_metrics_best`, `exchange_metrics_best`, `market_orderbooks_best_quotes`, `data_coverage_by_provider`, `provider_health`, `recent_backfill_activity`

## MCP Servers

- **Atlassian**: enabled via plugin, domain `whitestarcapital.atlassian.net`
- **Supabase**: enabled via plugin
- **Allium**: configured at user scope (`mcp.allium.so`) — MCP server credits are separate from the REST API key
- **CoinGecko**: active via project `.mcp.json` (`mcp-remote` → `https://mcp.pro-api.coingecko.com/mcp`) — Pro API with full historical range

## Test Markers

Tests use pytest markers defined in `backfill_data/pytest.ini`:
- `unit` — no external dependencies
- `integration` — requires live API or database
- `slow` — long-running tests
- `requires_api_key` — needs `COINMETRICS_API_KEY`
- `requires_database` — needs Supabase connection
