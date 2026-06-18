# Backfill Status - Live Supabase Warehouse

Last verified: 2026-04-10 via Supabase MCP
Project: `Data Sources` (`jnulpcqpftnwvknwuqpa`)

## Summary

- `public.asset_metrics`: 2,848,307 rows
- `public.asset_metrics_best`: 2,817,204 rows
- Providers in live fact data: 9
- Distinct assets: 29
- Distinct metrics: 338
- Global date range: 2021-01-01 -> 2026-01-01
- Current database size: 1,088 MB

The live warehouse is richer than the current committed provider tree. In particular, Artemis and Hyperliquid data are definitely present in Supabase at production scale even though their provider source/config files are not committed in this worktree.

## Asset Scope

Target 26-coin universe:

- usdc, usdt, usde, btc, eth, bnb, hype, xrp, pendle, uni, jup, tao, link, zec, ena, morpho, aero, sol, avax, pol, wlfi, crv, aave, pepe, shib, doge

Additional assets currently present in `asset_metrics`:

- `macro` for FRED series
- `matic` as legacy Polygon lineage data
- `lido` as legacy DefiLlama protocol data

`makerdao` is not present in the live table.

## Provider Coverage

| Provider | Rows | Assets | Metrics | Date Range | Notes |
|---|---:|---:|---:|---|---|
| `artemis` | 1,556,065 | 27 | 216 | 2021-01-01 -> 2026-01-01 | Largest source; includes price, market cap, protocol, stablecoin, staking, and chain activity metrics |
| `hyperliquid` | 620,890 | 20 | 2 | 2023-11-01 -> 2025-12-31 | `funding_rate_8h`, `funding_premium` |
| `derived` | 282,597 | 26 | 10 | 2021-01-01 -> 2026-01-01 | Derived features such as realized volatility and flow ratios |
| `dune` | 143,246 | 5 | 53 | 2021-01-01 -> 2026-01-01 | ETH, SOL, BNB, AVAX, UNI analytics |
| `coinmetrics` | 132,166 | 11 | 13 | 2021-01-01 -> 2026-01-01 | Best source for some core market and on-chain series |
| `defillama` | 52,557 | 20 | 4 | 2021-01-01 -> 2026-01-01 | TVL, fees, volume, stablecoin supply |
| `coingecko` | 22,064 | 26 | 8 | 2025-03-12 -> 2026-01-01 | Demo/free-tier history window only |
| `fred` | 20,452 | 1 | 30 | 2021-01-01 -> 2026-01-01 | `asset='macro'` |
| `allium` | 18,270 | 2 | 5 | 2021-01-01 -> 2026-01-01 | BTC and ETH fallback coverage |

Notes:

- `asset_metrics_best` keeps all 9 providers, but CoinMetrics, CoinGecko, and Allium contribute fewer rows there because higher-priority providers win for overlapping points.
- `backfill_progress` and the health views do not currently reflect all live providers. They only track `allium`, `coingecko`, `coinmetrics`, `defillama`, `dune`, and `fred`.

## Roadmap Factor Coverage

This section maps the live warehouse to the roadmap in [Causal PDE-Control Models for Portfolio Optimization.md](../docs/Causal%20PDE-Control%20Models%20for%20Portfolio%20Optimization.md).

| Roadmap Factor | Live Coverage | Current Sources | Status |
|---|---|---|---|
| LiqFlow / LP Flow | TVL and LP-flow style proxies exist | `derived.tvl_net_flow_usd`, DefiLlama `tvl_usd`, Dune UNI V3 LP flow data | Partial |
| StableFlow | Strong stablecoin flow and supply coverage | Dune `stablecoin_minted_usd`, `stablecoin_burned_usd`, `stablecoin_netflow_usd`; Artemis stablecoin metrics; `derived.stablecoin_net_flow_usd` | Strong |
| Funding Basis | Direct funding series present | Hyperliquid `funding_rate_8h`, `funding_premium` | Strong |
| Chain Congestion | Gas-utilization style metrics present for selected chains | Dune `avg_gas_utilization`, `max_gas_utilization` and related chain analytics | Partial |
| Staking Yield | Staking activity present, yield still uneven | Dune staking queries and Artemis staking-related metrics | Partial |
| MEV Pressure | No direct MEV series confirmed in the live warehouse | Congestion and liquidation proxies only | Proxy only |
| CEX / DEX Flow | Bridge and flow proxies present | Dune bridge metrics, Dune CEX/flow queries, `derived.dex_cex_volume_ratio` | Partial to strong |

## Lineage and Legacy Assets

### Polygon lineage

`pol` is not just a post-migration series in the live warehouse. Artemis currently stores:

- `matic`: 4,108 rows, 2021-01-01 -> 2023-10-24
- `pol`: 124,380 Artemis rows, 2021-01-01 -> 2026-01-01

Additional `pol` coverage also exists from:

- `hyperliquid`: 22,594 rows
- `derived`: 9,115 rows
- `defillama`: 3,654 rows
- `coingecko`: 659 rows

This means the warehouse currently contains both a legacy `matic` series and a longer-running `pol` series. Any future cleanup should decide whether `matic` stays as archival lineage or gets merged/re-mapped.

### Legacy non-target data

- `lido` still exists from DefiLlama with 1,708 rows from 2021-04-30 -> 2026-01-01.
- `makerdao` does not appear in the current live table.

## Warehouse vs Repository State

The committed repo and the live warehouse are not fully aligned:

- The current worktree does not contain committed Artemis or Hyperliquid provider source files.
- `backfill_data/src/providers/artemis/` and `backfill_data/src/providers/hyperliquid/` only contain `__pycache__`.
- No committed `config/providers/artemis.json`, `config/providers/hyperliquid.json`, or Artemis/Hyperliquid endpoint JSON files are present.
- Despite that, both providers are clearly present in `asset_metrics` and `asset_metrics_best`.

Interpretation:

- Supabase is the source of truth for live data coverage.
- The repo is the source of truth for what can be re-run from the current committed tree.

## Storage Footprint

Database size right now:

- Total DB size: 1,140,436,115 bytes (`1,088 MB`)
- `public.asset_metrics` total relation size: 1,127,555,072 bytes
- `public.asset_metrics` heap/table size: 405,659,648 bytes
- `public.asset_metrics` index size: 721,747,968 bytes

Largest `asset_metrics` indexes:

| Index | Size | `idx_scan` | Notes |
|---|---:|---:|---|
| `idx_asset_metrics_priority` | 293 MB | 1 | Needed by the best-provider view pattern |
| `asset_metrics_pkey` | 292 MB | 4,113 | Core uniqueness index |
| `idx_asset_metrics_asset_time` | 43 MB | 3 | Lightly used |
| `idx_asset_metrics_time` | 32 MB | 1 | Lightly used |
| `idx_asset_metrics_provider` | 28 MB | 1 | Lightly used |

### Can we get under 1.1 GB?

Yes. This is now done.

Applied on 2026-04-10:

- dropped `public.idx_asset_metrics_metric_time`
- database size moved from about `1,147 MB` to `1,088 MB`

This got the project under the requested `1.1 GB` target without deleting CPCM-relevant data.

Important:

- the checked-in schema file has also been updated so the index is not recreated accidentally
- if more headroom is needed later, the next review candidates are `idx_asset_metrics_provider` and `idx_asset_metrics_time`, but those should be checked against real workloads first

## Recommended Next Actions

1. Treat Supabase fact tables, not `backfill_progress`, as the source of truth for Artemis and Hyperliquid coverage.
2. Decide whether Artemis and Hyperliquid should be restored into the committed provider tree or explicitly documented as warehouse-only historical imports.
3. Keep the current index set stable unless new workload evidence justifies more removals.
4. If more storage pressure appears later, profile `idx_asset_metrics_provider` and `idx_asset_metrics_time` before touching the priority or primary-key indexes.
