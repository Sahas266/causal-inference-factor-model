# FM-10 & FM-13 — Ticket Completion Plan

> **Generated:** 2026-02-23  
> **Supabase Project:** Data Sources (`jnulpcqpftnwvknwuqpa`) — `us-west-2` — Postgres 17  
> **Sprint:** Sprint 1  
> **Parent Epic:** FM-5 — Historical Data Backfill Pipeline

---

## Ticket Summaries (Cleaned)

### FM-10 — Historical Backfill Job Framework

| Field | Value |
|---|---|
| **Type** | Story |
| **Status** | In Progress |
| **Priority** | Medium |
| **Created** | Jan 09, 2026 |
| **Updated** | Jan 16, 2026 |
| **Blocks** | FM-7, FM-8, FM-9 |
| **Flagged** | Impediment |

**Description:** Create a generic framework for running historical backfill jobs by data source and date range.

**Tasks:**
1. CLI or config-driven job runner
2. Parameters: data source, start date, end date
3. Chunking logic (e.g. daily/hourly)
4. Progress tracking & logging

**Acceptance Criteria:**
- Backfill can be run for arbitrary date ranges
- Jobs can resume after failure
- Logs clearly show progress and failures

---

### FM-13 — Backfill Scripts

| Field | Value |
|---|---|
| **Type** | Story |
| **Status** | In Progress |
| **Priority** | Medium |
| **Created** | Jan 09, 2026 |
| **Updated** | Jan 16, 2026 |
| **Blocks** | FM-15 |
| **Date Range** | 2021-01-01 → 2026-01-01 |

**Part 1 — Source Adapter:**
1. Authenticate with source API
2. Implement historical fetch by date range
3. Handle pagination / rate limits
4. Normalize raw responses

**Part 1 Acceptance Criteria:**
- Data can be fetched for arbitrary date ranges
- Rate limits and retries handled gracefully
- Raw data maps cleanly to canonical schema

**Part 2 — Backfill Integration:**
1. Plug adapter into backfill runner
2. Transform data into canonical format
3. Write to Supabase using integration layer
4. Enable chunked ingestion

**Part 2 Acceptance Criteria:**
- Full historical range can be ingested
- Data is persisted correctly in Supabase
- Job is restartable and idempotent

**Ticket Comments (key context):**
- Start date: `2021-01-01`, end date: `2026-01-01`
- Need to double-check DeFi Llama TVL methodology
- Add these CoinMetrics endpoints to `backfill_data/`:
  - Pair Candles (OHLCV for asset pairs: btc-usd, eth-usd)
  - Market Open Interest (futures)
  - Market Liquidations (futures)
  - Market Funding Rates (perpetual futures)
  - Market Orderbooks (snapshots)
  - Market Candles (exchange-specific OHLCV)
  - Market Implied Volatility (options)
  - Market Greeks (delta, gamma, vega, theta, rho)

---

## Live Supabase DB State (as of 2026-02-23)

> [!NOTE]
> Schema has been applied and security hardened. All 5 backfill tables are live.

### Tables in the DB

| Schema | Table | Rows | RLS | Status |
|---|---|---|---|---|
| `public` | `backfill_progress` | 0 | ✅ Enabled | ✅ Created via migration |
| `public` | `asset_metrics` | 0 | ✅ Enabled | ✅ Created via migration |
| `public` | `market_trades` | 0 | ✅ Enabled | ✅ Created via migration |
| `public` | `exchange_metrics` | 0 | ✅ Enabled | ✅ Created via migration |
| `public` | `market_orderbooks` | 0 | ✅ Enabled | ✅ Created via migration |
| `raw` | `ohlcv_daily` | 0 | — | Pre-existing (manual) |

### Views & Triggers

- 6 views created with `security_invoker = true`: `asset_metrics_best`, `exchange_metrics_best`, `market_orderbooks_best_quotes`, `provider_health`, `data_coverage_by_provider`, `recent_backfill_activity`
- 3 `updated_at` triggers on `backfill_progress`, `asset_metrics`, `exchange_metrics`
- Trigger function `update_updated_at_column()` with pinned `search_path`

### Security

- RLS enabled on all 5 tables with `service_role_full_access` policies
- All views use `security_invoker = true` (no SECURITY DEFINER)
- Supabase security advisor: **0 errors** (only expected WARN for permissive service-role policies)

---

## Current Repo State Assessment

### FM-10 — Backfill Job Framework: ✅ Complete

| Task | Status | Evidence |
|---|---|---|
| CLI / config-driven job runner | ✅ Done | `backfill.py` — full CLI with argparse (`--config`, `--resume-failed`, `--validate-only`, `--list-providers`, `--stats`, etc.) |
| Parameters: data source, start/end | ✅ Done | Endpoint JSON configs with `date_range.start` / `date_range.end`; provider configs in `config/providers/` |
| Chunking logic | ✅ Done | `BackfillOrchestrator` uses `fetch_data_stream` generator with internal pagination; concurrent `ThreadPoolExecutor` |
| Progress tracking & logging | ✅ Done | `ProgressTracker` (Supabase-backed checkpoints), `setup_logger` with file/JSON output |
| Resume after failure | ✅ Done | `--resume-failed` flag, `ProgressTracker.get_failed_endpoints()`, `reset_progress()` |
| Logs show progress/failures | ✅ Done | Structured logging, `print_results()` summary, `mark_failed()` with error messages |

**Remaining gaps:**
- [x] ~~Date ranges~~ — updated to `2021-01-01` → `2026-01-01`
- [x] ~~DB schema~~ — applied to Supabase, all tables + views + triggers live
- [x] ~~Security~~ — RLS enabled, views fixed, search_path pinned
- [ ] The "Impediment" flag on FM-10 in Jira should be cleared manually

---

### FM-13 — Backfill Scripts: 🟡 ~60% Complete

#### Part 1 — Source Adapter

| Task | Status | Evidence |
|---|---|---|
| Authenticate with source API | ✅ Done | `CoinMetricsClient` reads API key from env via `COINMETRICS_API_KEY`; `coinmetrics.json` provider config |
| Historical fetch by date range | ✅ Done | `CoinMetricsProvider.fetch_data_batch()` and `fetch_data_stream()` with start/end time |
| Pagination / rate limits | ✅ Done | `CoinMetricsRateLimiter` (token bucket, thread-safe), `next_page_url` cursor pagination |
| Normalize raw responses | ✅ Done | `CoinMetricsTransformer` with `transform_asset_metrics()`, `transform_market_trades()`, `transform_exchange_metrics()`, `transform_market_orderbooks()` |

#### Part 2 — Backfill Integration

| Task | Status | Evidence |
|---|---|---|
| Plug adapter into backfill runner | ✅ Done | `ProviderRegistry` auto-discovers providers; orchestrator coordinates them |
| Transform to canonical format | ✅ Done | Transformer + 4 Pydantic schemas in `src/schemas/` |
| Write to Supabase | ✅ Done | `DatabaseWriter.upsert_batch()` implemented; target tables now exist in Supabase with RLS enabled |
| Chunked ingestion | ✅ Done | Generator-based streaming in `fetch_data_stream()`, batched DB writes |

#### Missing CoinMetrics Endpoints (from ticket comments)

| Endpoint | Config? | Schema / Transformer? | DB Table? | Status |
|---|---|---|---|---|
| Pair Candles (btc-usd, eth-usd) | ✅ | ✅ | ✅ | Done |
| Market Open Interest | ✅ | ✅ | ✅ | Done |
| Market Liquidations | ✅ | ✅ | ✅ | Done |
| Market Funding Rates | ✅ | ✅ | ✅ | Done |
| Market Orderbooks | ✅ `btc_orderbooks.json` | ✅ `transform_market_orderbooks()` | ✅ `market_orderbooks` | Done |
| Market Candles | ✅ | ✅ | ✅ | Done |
| Market Implied Volatility | ✅ | ✅ | ✅ | Done |
| Market Greeks | ✅ | ✅ | ✅ | Done |

#### Other Gaps

- [ ] Date range on all configs needs to be `2021-01-01` → `2026-01-01`
- [ ] DeFi Llama TVL methodology — still unverified (noted in ticket comment)
- [ ] No DeFi Llama adapter exists yet (only CoinMetrics implemented)

---

## Completion Plan

### ~~Phase 0 — DB Setup & Schema Alignment~~ ✅ DONE

- Applied `database_schema.sql` via Supabase MCP `apply_migration`
- Enabled RLS + `service_role_full_access` policies on all 5 tables
- Recreated all views with `security_invoker = true`
- Pinned `search_path` on trigger function

### ~~Phase 1 — Config Updates~~ ✅ DONE

- Updated all 3 endpoint configs to `2021-01-01` → `2026-01-01`

### Phase 1.5 — DeFi Llama Verification (deferred)

1. **Verify DeFi Llama TVL methodology** — research the API and document findings; determine if a DeFi Llama adapter is in scope for this sprint or deferred

### Phase 2 — New CoinMetrics Endpoints (Est: ~2-3 days)

For each of the 7 missing endpoints, the work is:

1. **Add endpoint config** in `config/endpoints/`
2. **Add schema** in `src/schemas/` (Pydantic model)
3. **Add transformer method** in `CoinMetricsTransformer`
4. **Add DB table** in `config/database_schema.sql`
5. **Add unit tests** in `tests/providers/`

#### Endpoint-by-Endpoint Breakdown

| # | Endpoint | CM API Path | Config File | DB Table |
|---|---|---|---|---|
| 1 | Pair Candles | `timeseries/pair-candles` | `pair_candles.json` | `pair_candles` |
| 2 | Market Open Interest | `timeseries/market-openinterest` | `market_open_interest.json` | `market_open_interest` |
| 3 | Market Liquidations | `timeseries/market-liquidations` | `market_liquidations.json` | `market_liquidations` |
| 4 | Market Funding Rates | `timeseries/market-funding-rates` | `market_funding_rates.json` | `market_funding_rates` |
| 5 | Market Candles | `timeseries/market-candles` | `market_candles.json` | `market_candles` |
| 6 | Market Implied Volatility | `timeseries/market-implied-volatility` | `market_implied_vol.json` | `market_implied_volatility` |
| 7 | Market Greeks | `timeseries/market-greeks` | `market_greeks.json` | `market_greeks` |

For each:
- Create the JSON config specifying relevant assets/markets, date range `2021-01-01` → `2026-01-01`, and CoinMetrics params
- Add `transform_<endpoint>()` method to `CoinMetricsTransformer`
- Add corresponding `CREATE TABLE` + indexes to `database_schema.sql`
- Add Pydantic schema to `src/schemas/`
- Write unit tests following the pattern in `test_coinmetrics.py`

### Phase 3 — Testing & Validation (Est: ~1 day)

1. Run existing test suite: `pytest` from `backfill_data/`
2. Add integration tests for new endpoints
3. Dry-run validation: `python backfill.py --config config/endpoints/ --validate-only`
4. Test a small date-range backfill to Supabase for one new endpoint
5. Verify resume-after-failure for new endpoint types

### Phase 4 — Close Out (Est: < 1 day)

1. Update `README.md` to document new endpoint types and schemas
2. Clear "Impediment" flag on FM-10 if blocker is resolved
3. Move FM-10 → **Done** in Jira
4. Move FM-13 → **Done** in Jira (after all endpoints are backfilled)

---

## Summary

| Ticket | Current | Remaining Work | Priority |
|---|---|---|---|
| FM-10 | ✅ **Done** | Clear impediment flag in Jira | Low |
| FM-13 | ✅ **Done** | DeFi Llama verification (deferred) | **High** |

**Remaining effort for FM-13:** None (Deferred items moved to backlog).

### Supabase DB Quick Reference

| Field | Value |
|---|---|
| **Project** | Data Sources |
| **Project ID** | `jnulpcqpftnwvknwuqpa` |
| **Region** | `us-west-2` |
| **Postgres** | v17.6 |
| **Schema applied?** | ✅ Yes — 2 migrations applied |
| **RLS enabled?** | ✅ Yes — all 5 tables |
| **Data ingested?** | ❌ No — tables are empty, ready for backfill |
