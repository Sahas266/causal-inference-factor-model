# ETH Backfill Worklog (Allium, CoinGecko, CoinMetrics, Dune, DefiLlama, Supabase)

This document captures the complete implementation and debugging history of the ETH backfill work completed so far, including:
- what was changed,
- why it was changed,
- runtime/validation outcomes,
- Supabase state after runs,
- and what remains unresolved.

All timestamps below refer to work performed on March 8-9, 2026 (America/New_York).

## 1. Objective and scope

The requested scope across messages was:
- Add provider backfill scripts for CoinGecko, CoinMetrics, Dune, and DefiLlama.
- Store Dune API key and ensure it is not committed.
- Store CoinMetrics Community API endpoint details.
- Build a comprehensive ETH data-point catalog across Allium, CoinGecko, CoinMetrics, Dune, and DefiLlama.
- Update Supabase and ingest ETH data with provider fallback.
- Prefer non-Allium providers first; use Allium only when necessary.
- Ensure each ingested data point records its source provider.
- Use Supabase MCP if available.

## 2. MCP and environment reality

### 2.1 Supabase MCP availability check
- Supabase MCP server was requested, but no Supabase MCP server was configured in workspace MCP configs for this session.
- Existing MCP configs contained other servers, not Supabase.

Decision taken:
- Proceeded with the repository’s direct Supabase client integration (`supabase-py`) rather than MCP.

### 2.2 Repo/runtime baseline
- Python backfill framework already present with provider registry architecture.
- Existing providers in use: `allium`, `coinmetrics`, `defillama`.
- Dune and CoinGecko scripts were scaffolded.
- Network calls required elevated permissions in this environment.

## 3. Secret handling and local config updates

### 3.1 API keys and .env
- Local `backfill_data/.env` was used for secrets.
- `.env` is ignored by `backfill_data/.gitignore`.

Added/updated locally:
- `DUNE_API_KEY=...`
- `SUPABASE_URL=...`
- `SUPABASE_KEY=...`

Notes:
- No secrets were written to tracked source files.
- No git push was performed.

### 3.2 CoinMetrics community endpoint storage
CoinMetrics provider config was aligned to community API:
- Base URL: `https://community-api.coinmetrics.io/v4`
- Community use without API key.

File:
- `backfill_data/config/providers/coinmetrics.json`

## 4. New scripts and configs introduced in this effort

### 4.1 Provider scripts
The `backfill_data/scripts/` folder includes:
- `backfill_coingecko.py`
- `backfill_coinmetrics.py`
- `backfill_defillama.py`
- `backfill_dune.py`
- `run_provider_backfill.py`

### 4.2 ETH endpoint packs
Added endpoint packs:
- `backfill_data/config/endpoints/eth/` (primary set)
- `backfill_data/config/endpoints/eth_fallback/` (Allium fallback set)

Primary ETH set includes endpoint configs such as:
- `eth_asset_metrics_primary.json`
- `eth_pair_candles_coinmetrics.json`
- `eth_market_candles_coinmetrics.json`
- `eth_open_interest_coinmetrics.json`
- `eth_funding_rates_coinmetrics.json`
- `eth_liquidations_coinmetrics.json`
- `eth_chain_tvl_defillama.json`
- `eth_protocol_tvl_defillama.json`
- `eth_dex_volumes_defillama.json`
- `eth_fees_defillama.json`
- `eth_stablecoin_flow_defillama.json`

Fallback set includes:
- `eth_price_allium_fallback.json`

## 5. ETH data-point catalog work

A full ETH data-point catalog was created in:
- `backfill_data/ETH_DATA_POINTS_CATALOG.md`

This catalog documents:
- prioritized ETH metrics,
- provider mapping,
- implementation status,
- and pull ordering.

## 6. Code fixes applied during execution

Below are the concrete code changes made to unblock validation/backfill runs.

### 6.1 Orchestrator date-range fallback parsing
Issue:
- Some provider validations return `valid=True` but no adjusted date bounds.
- Runtime would then pass `None` dates and fail later.

Fix:
- Added `_parse_date_range_bound(...)` in orchestrator.
- If provider-adjusted bounds are missing, fallback to endpoint `date_range.start/end`.

File:
- `backfill_data/src/core/orchestrator.py`

### 6.2 Orchestrator provider-instance ID support
Issue:
- Endpoints with multiple entries for the same provider (e.g., several DefiLlama protocol configs) collided in progress IDs.

Fix:
- Added `provider_instance_id` per provider entry during validation.
- Endpoint progress key now uses `..._{provider}_{provider_instance_id}`.

File:
- `backfill_data/src/core/orchestrator.py`

### 6.3 ConfigLoader relative path handling
Issue:
- `--config config/endpoints/eth` could be misresolved by loader into doubled paths.

Fix:
- `load_endpoint_config` now first checks whether the provided relative path already exists before prefixing `config/endpoints`.

File:
- `backfill_data/src/core/utils/config_loader.py`

### 6.4 DefiLlama client endpoint modernization
Issue:
- DefiLlama client used stale paths (`/api/...`) and wrong host assumptions for stablecoins/coins.

Live probe outcomes confirmed working paths:
- `https://api.llama.fi/v2/historicalChainTvl/{chain}`
- `https://api.llama.fi/protocol/{protocol}`
- `https://api.llama.fi/summary/dexs/{protocol}`
- `https://api.llama.fi/summary/fees/{protocol}`
- `https://stablecoins.llama.fi/stablecoincharts/{chain}`
- `https://coins.llama.fi/chart/{coins}`

Fix:
- Replaced DefiLlama client implementation to use current endpoints.
- Added explicit `stablecoins_base_url` and `coins_base_url`.
- Updated DefiLlama provider initialization and provider config to pass/use these bases.

Files:
- `backfill_data/src/providers/defillama/client.py`
- `backfill_data/src/providers/defillama/provider.py`
- `backfill_data/config/providers/defillama.json`

### 6.5 CoinMetrics catalog parsing resilience
Issue:
- Catalog parser only matched `asset`/`market`, but some endpoints use `pair` or `exchange`.

Fix:
- Expanded identifier matching in `parse_catalog_times(...)` to include:
  - `asset`, `market`, `pair`, `exchange`
- Added fallback: when catalog returns one row with min/max bounds, use it even if identifier field naming differs.

File:
- `backfill_data/src/providers/coinmetrics/client.py`

### 6.6 CoinMetrics validation behavior for community catalogs lacking min/max
Issue:
- Some community catalog responses had data but omitted `min_time/max_time`.
- Validation incorrectly treated those as invalid.

Fix:
- If catalog has `data` but no parsed min/max, validation now returns `valid=True` and lets orchestrator use endpoint-configured date range.

File:
- `backfill_data/src/providers/coinmetrics/provider.py`

### 6.7 Progress tracker JSON serialization fix
Issue:
- `backfill_progress.config` upserts failed with:
  - `Object of type datetime is not JSON serializable`
- Cause: provider config snapshot included datetime objects (`actual_start`, `actual_end`).

Fix:
- Added recursive `_to_json_safe(...)` conversion in progress tracker.
- Config snapshot now serializes datetimes to ISO strings before insert/upsert.

File:
- `backfill_data/src/core/storage/progress_tracker.py`

## 7. Validation and live probe findings

## 7.1 CoinMetrics community accessibility (as observed)

Accessible:
- asset metrics catalog/timeseries (with caveats by metric set)
- pair candles catalog/timeseries (but some ranges returned empty data)

Not accessible in community mode for configured markets (403):
- market candles
- market open interest
- funding rates
- liquidations

## 7.2 DefiLlama endpoint behavior (after client fix)

Validated and reachable:
- chain TVL
- protocol TVL
- DEX summary
- fees summary
- stablecoin charts
- coin chart

## 7.3 ETH config validation progression

Validation runs evolved from:
- initially `0 valid / 11 skipped` (before network permissions and endpoint fixes),
to
- `7 valid / 4 skipped` (after fixes).

Skipped endpoints were primarily CoinMetrics community-restricted derivative/market datasets.

## 8. Supabase authentication and key correction

Issue encountered:
- Supabase data API operations failed with:
  - `"Invalid API key"` when using the provided `sbp_...` token as `SUPABASE_KEY`.

Resolution:
- Used the `sbp_...` token with Supabase Management API to fetch project API keys.
- Retrieved project `service_role` key.
- Updated local `.env` to use service role key for data writes.

Result:
- Supabase table operations via `supabase-py` then succeeded.

## 9. Backfill execution history and status

### 9.1 One long run was user-interrupted
- A full ETH backfill command was started and later intentionally interrupted.
- Because interruption happened mid-run, state was verified directly in Supabase afterward.

### 9.2 Current Supabase table counts (post-interruption verification)
- `backfill_progress`: 11 rows
- `asset_metrics`: 2196 rows
- `pair_candles`: 0 rows

### 9.3 `asset_metrics` data currently present (provider/asset/metric)

From Supabase paginated query:
- `defillama / ethereum / tvl_usd`: 366 rows (2025-01-01 to 2026-01-01)
- `defillama / aave / tvl_usd`: 366 rows (2025-01-01 to 2026-01-01)
- `defillama / uniswap / tvl_usd`: 366 rows (2025-01-01 to 2026-01-01)
- `defillama / curve / tvl_usd`: 365 rows (2025-01-01 to 2026-01-01)
- `defillama / lido / tvl_usd`: 366 rows (2025-01-01 to 2026-01-01)
- `defillama / uniswap / fees_usd`: 366 rows (2025-01-01 to 2026-01-01)
- `defillama / eth / price_usd`: 1 row

### 9.4 `backfill_progress` endpoint states (current)

Observed rows:
- `eth_asset_metrics_primary_coinmetrics_0`: `pending`
- `eth_asset_metrics_primary_defillama_1`: `completed` (1)
- `eth_chain_tvl_defillama_defillama_0`: `completed` (366)
- `eth_dex_volumes_defillama_defillama_0`: `completed` (0)
- `eth_fees_defillama_defillama_0`: `completed` (366)
- `eth_pair_candles_coinmetrics_coinmetrics_0`: `completed` (0)
- `eth_protocol_tvl_defillama_defillama_0`: `completed` (366)
- `eth_protocol_tvl_defillama_defillama_1`: `completed` (366)
- `eth_protocol_tvl_defillama_defillama_2`: `completed` (365)
- `eth_protocol_tvl_defillama_defillama_3`: `completed` (366)
- `eth_stablecoin_flow_defillama_defillama_0`: `failed` with error:
  - `'str' object cannot be interpreted as an integer`

## 10. Why specific endpoints are incomplete

### 10.1 Stablecoin flow failure
Root cause:
- DefiLlama stablecoin payload `date` is a string epoch, e.g. `"1511913600"`.
- Transformer path expects integer timestamp.
- Conversion path raised type error and endpoint failed.

### 10.2 CoinMetrics asset metrics pending
Root cause:
- For requested ETH metric bundle in community API, live fetch returned `403` in direct probe.
- Error handler currently treats unknown statuses as retryable, which can loop/retry instead of failing fast.
- Interruption occurred while this path was still unresolved in-run.

### 10.3 Pair candles zero-record completion
Root cause:
- Endpoint response was `200` with empty `data` for configured ETH date range.
- Job completed successfully with zero upserts.

## 11. Fallback policy behavior so far

Requested policy:
- Use Allium only when no other choice.

Observed behavior in current ingestion:
- Most inserted records came from DefiLlama.
- Allium fallback endpoint exists in config (`eth_fallback`) but was not the source of current inserted rows in `asset_metrics`.
- CoinMetrics derivatives/market endpoints were skipped by validation due provider restrictions.

## 12. Files touched for this effort (key subset)

Primary code/config files modified or added:
- `backfill_data/config/providers/coinmetrics.json`
- `backfill_data/config/providers/defillama.json`
- `backfill_data/src/core/orchestrator.py`
- `backfill_data/src/core/utils/config_loader.py`
- `backfill_data/src/core/storage/progress_tracker.py`
- `backfill_data/src/providers/coinmetrics/client.py`
- `backfill_data/src/providers/coinmetrics/provider.py`
- `backfill_data/src/providers/defillama/client.py`
- `backfill_data/src/providers/defillama/provider.py`
- `backfill_data/config/endpoints/eth/*`
- `backfill_data/config/endpoints/eth_fallback/*`
- `backfill_data/scripts/backfill_*.py`
- `backfill_data/ETH_DATA_POINTS_CATALOG.md`

## 13. Security and git status notes

- Secrets remain local in `.env` and are not tracked by git.
- No push/commit has been performed by this workflow.
- Repo has unrelated pre-existing modified/untracked files outside this exact backfill flow.

## 14. Current end-state summary

Achieved:
- Multi-provider ETH backfill framework wired and exercised.
- Supabase writes working with correct key class (service role).
- 2196 ETH-related `asset_metrics` rows written.
- Provider attribution present on each record via schema fields (`provider`, `provider_priority`).
- Progress tracking rows populated for endpoint/provider instances.

Remaining blockers:
- DefiLlama stablecoin transformer needs string-epoch handling.
- CoinMetrics community restrictions require:
  - metric subset adjustments and/or
  - fallback remapping for restricted endpoint types.
- CoinMetrics 403 handling should fail fast (non-retry) for this workflow.

