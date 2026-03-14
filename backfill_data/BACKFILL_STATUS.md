# Backfill Status — All 26 Coins

**Last updated:** 2026-03-14
**Total rows in `asset_metrics`:** 297,516
**Distinct assets:** 28 (26 target coins + 2 legacy ETH-era names)
**Distinct metrics:** 60
**Distinct providers:** 6

## Database Schema

### Tables (12 total)

| Table | Rows | Primary Keys | Notes |
|-------|------|-------------|-------|
| `asset_metrics` | 297,516 | `(provider, asset, metric, time)` | Main data table — all coin metrics |
| `backfill_progress` | ~130 | `(endpoint_id)` | Checkpoint/resume tracking |
| `exchange_metrics` | 0 | `(provider, exchange, metric, time)` | Empty — not yet used |
| `market_trades` | 0 | `(provider, market, time, trade_id)` | Empty |
| `market_candles` | 0 | `(provider, market, time, frequency)` | Empty |
| `pair_candles` | 0 | `(provider, pair, time, frequency)` | Empty |
| `market_orderbooks` | 0 | `(provider, market, time)` | Empty |
| `market_open_interest` | 0 | `(provider, market, time)` | Empty — requires CoinMetrics Pro |
| `market_funding_rates` | 0 | `(provider, market, time)` | Empty — requires CoinMetrics Pro |
| `market_liquidations` | 0 | `(provider, market, time)` | Empty — requires CoinMetrics Pro |
| `market_implied_volatility` | 0 | `(provider, market, time)` | Empty — requires CoinMetrics Pro |
| `market_greeks` | 0 | `(provider, market, time)` | Empty — requires CoinMetrics Pro |

### Views (6)

| View | Purpose |
|------|---------|
| `asset_metrics_best` | Selects one provider per data point by `provider_priority` |
| `exchange_metrics_best` | Same for exchange metrics |
| `market_orderbooks_best_quotes` | Best bid/ask from orderbooks |
| `data_coverage_by_provider` | Summary of rows per provider/asset |
| `provider_health` | Freshness and error rate per provider |
| `recent_backfill_activity` | Latest backfill_progress entries |

### `asset_metrics` Columns

| Column | Type | Notes |
|--------|------|-------|
| `provider` | text | coinmetrics, coingecko, defillama, allium, dune, derived |
| `asset` | text | Ticker (btc, eth, sol...) |
| `metric` | text | Metric name (PriceUSD, tvl_usd...) |
| `time` | timestamptz | Observation timestamp |
| `value` | text | Metric value (stored as text for precision) |
| `frequency` | text | 1d, 1h, etc. |
| `provider_priority` | integer | Lower = higher priority (1=CoinMetrics, 3=CoinGecko, 99=derived) |
| `metadata` | jsonb | Provider-specific extra fields |
| `created_at` | timestamptz | Row insert time |

---

## Row Counts by Provider

| Provider | Rows | Assets Covered | Metrics | Date Range |
|----------|------|---------------|---------|------------|
| CoinMetrics | 140,679 | 11 (btc, eth, bnb, xrp, doge, zec, aave, uni, link, usdc, usdt) | 13 | 2021-01-01 → 2026-01-01 |
| Dune | 49,752 | 1 (eth only) | 28 | 2021-01-01 → 2025-12-31 |
| DefiLlama | 45,601 | 19 | 4 (tvl_usd, fees_usd, volume_usd, stablecoin_circulating_usd) | 2021-01-01 → 2026-01-01 |
| Derived | 41,613 | 11 | 3 (realized_volatility_7d/30d, dex_cex_volume_ratio) | 2021-01-01 → 2026-01-01 |
| Allium | 18,270 | 2 (btc, eth OHLCV fallback) | 5 | 2021-01-01 → 2026-01-01 |
| CoinGecko | 1,601 | 26 (all coins — snapshots only) | 8 | Snapshots: 2026-03-14 |
| **Total** | **297,516** | | | |

---

## Per-Coin Data Coverage

### Legend

- **CM** = CoinMetrics (priority 1) — community tier, no key needed
- **CG** = CoinGecko (priority 3) — demo key, limited to snapshots only (market_chart 401s on historical range)
- **DL** = DefiLlama (priority 2) — free, no key needed
- **AL** = Allium (priority 4) — fallback OHLCV
- **DU** = Dune — ETH-only on-chain analytics
- **DR** = Derived — computed from other providers

### L1 Major Chains

| Coin | CM Timeseries | CG Snapshot | DL Chain TVL | DL Fees | DL DEX Volume | Other | Total Rows |
|------|--------------|-------------|--------------|---------|---------------|-------|------------|
| **BTC** | 13 metrics (23,751) | 5 metrics | tvl_usd (1,748) | -- | -- | Allium OHLCV (9,135), DR vol 7d/30d (3,617) | **34,644** |
| **ETH** | 13 metrics (23,751) | 7 metrics | tvl_usd (1,827) | -- | -- | Allium (9,135), Dune 28 metrics (49,752), DR 3 metrics (5,443) | **93,107** |
| **SOL** | -- | 4 metrics | tvl_usd (1,751) | -- | -- | -- | **1,759** |
| **BNB** | 5 metrics (9,135) | 5 metrics | tvl_usd (1,827) | -- | -- | DR vol 7d/30d (3,617) | **10,972** |
| **XRP** | 10 metrics (12,789) | 5 metrics | -- | -- | -- | DR vol 7d/30d (3,617) | **12,799** |
| **AVAX** | -- | 5 metrics | tvl_usd (1,794) | -- | -- | -- | **1,804** |
| **POL** | -- | 5 metrics | tvl_usd (1,827) | -- | -- | -- | **1,832** |
| **DOGE** | 10 metrics (18,270) | 4 metrics | -- | -- | -- | DR vol 7d/30d (3,617) | **18,278** |
| **ZEC** | 10 metrics (18,270) | 5 metrics | -- | -- | -- | DR vol 7d/30d (3,617) | **18,280** |

### DeFi Protocols

| Coin | CM Timeseries | CG Snapshot | DL Protocol TVL | DL Fees | DL DEX Volume | DR Volatility | Total Rows |
|------|--------------|-------------|-----------------|---------|---------------|---------------|------------|
| **UNI** | 5 metrics (5,481) | 5 metrics | tvl_usd (1,827) | fees_usd (1,827) | volume_usd (1,827) | 7d/30d (3,617) | **10,972** |
| **AAVE** | 5 metrics (5,481) | 5 metrics | tvl_usd (1,827) | fees_usd (1,827) | -- | 7d/30d (3,617) | **9,145** |
| **CRV** | -- | 5 metrics | tvl_usd (1,820) | fees_usd (1,823) | volume_usd (1,231) | -- | **4,884** |
| **PENDLE** | -- | 4 metrics | tvl_usd (1,633) | fees_usd (909) | -- | -- | **2,550** |
| **MORPHO** | -- | 5 metrics | tvl_usd (732) | fees_usd (687) | -- | -- | **1,429** |
| **AERO** | -- | 4 metrics | tvl_usd (616) | fees_usd (607) | volume_usd (600) | -- | **1,831** |
| **LINK** | 5 metrics (5,481) | 5 metrics | -- | -- | -- | 7d/30d (3,617) | **5,491** |
| **ENA** | -- | 5 metrics | tvl_usd (641) | fees_usd (641) | -- | -- | **1,292** |
| **JUP** | -- | 5 metrics | tvl_usd (704) | fees_usd (732) | volume_usd (72) | -- | **1,518** |

### Stablecoins

| Coin | CM Timeseries | CG Snapshot | DL Stablecoin Supply | DR Volatility | Total Rows |
|------|--------------|-------------|---------------------|---------------|------------|
| **USDC** | 7 metrics (9,135) | 4 metrics | circulating_usd (1,827) | 7d/30d (3,617) | **10,970** |
| **USDT** | 7 metrics (9,135) | 4 metrics | circulating_usd (1,827) | 7d/30d (3,617) | **10,970** |
| **USDe** | -- | 4 metrics | circulating_usd (701), tvl_usd (701) | -- | **1,410** |

### Alt / Meme

| Coin | CM Timeseries | CG Snapshot | DL Chain TVL | Total Rows |
|------|--------------|-------------|--------------|------------|
| **HYPE** | -- | 5 metrics | -- (0 from DL) | **10** |
| **TAO** | -- | 5 metrics | -- | **10** |
| **WLFI** | -- | 5 metrics | -- | **10** |
| **PEPE** | -- | 5 metrics | -- | **10** |
| **SHIB** | -- | 4 metrics | -- | **8** |

---

## Legacy Asset Names (not in 26-coin list)

| `asset` | Provider | Rows | Notes |
|---------|----------|------|-------|
| `lido` | defillama | 3,534 | ETH ecosystem protocol (TVL + fees) |
| `makerdao` | defillama | 1,827 | ETH ecosystem protocol (TVL) |

Previously renamed: `uniswap`→`uni`, `curve`→`crv`, `ethereum`→`eth`, `eth_stablecoins`→`eth` (merged).

---

## What's Missing — Gaps by Provider

### CoinGecko Market Chart (ALL 26 coins — 0 timeseries rows)

**Status:** All `market_chart` endpoints return HTTP 401.
**Root cause:** Demo API key (`CG-` prefix) at `api.coingecko.com` cannot access historical ranges.
**Fix:** Upgrade to CoinGecko Pro key (`pro-api.coingecko.com`, 500 req/min, full history). This would add ~47,500 rows (3 metrics x 1,825 days x ~26 coins).
**Metrics affected:** `price_usd`, `market_cap_usd`, `spot_volume_usd_24h`

### CoinMetrics — Missing Coins (15 coins have no CM data)

CoinMetrics community tier returns 403 for SOL, AVAX, POL (matic), SHIB — these are Tier 2/3 assets.
Configs removed for these. Remaining coins without CoinMetrics data:

| Coin | Reason | Alternative |
|------|--------|-------------|
| SOL, AVAX, POL, SHIB | 403 on community tier (Tier 2/3) | CoinGecko Pro |
| CRV, PENDLE, MORPHO, AERO, ENA, JUP | Not on CoinMetrics community | CoinGecko Pro |
| USDe | Not on CoinMetrics | CoinGecko Pro |
| HYPE, TAO, WLFI, PEPE | Not on CoinMetrics | CoinGecko Pro |

### DefiLlama — Missing Data

| Coin | Expected | Status | Notes |
|------|----------|--------|-------|
| HYPE | chain TVL from Hyperliquid | 0 rows | Chain may not be in DefiLlama |
| LINK | No DeFi protocol | N/A | Chainlink is an oracle, not a DeFi protocol |

### Dune — ETH Only

Dune queries (9 total, 49,752 rows) only exist for ETH. No Dune coverage for other coins.

### Derived Metrics

Realized volatility computed for 11 assets with CoinMetrics PriceUSD: btc, eth, bnb, xrp, doge, zec, aave, uni, link, usdc, usdt.
DEX/CEX volume ratio only for ETH (requires Dune CEX data).

---

## Priority Actions

### Immediate (no cost)

1. **All non-CoinGecko endpoints are now backfilled** for all 26 coins
2. Derived metrics computed for all 11 coins with CoinMetrics PriceUSD

### Requires CoinGecko Pro Key

3. **CoinGecko market_chart for all 26 coins** — unlocks price/mcap/volume timeseries (est. ~47,500 rows)
4. Once CoinGecko prices available, derived volatility can be computed for remaining 15 coins

### Requires CoinMetrics Pro Key

5. **Derivatives data** — open interest, funding rates, liquidations (all empty tables)
6. **Tier 2/3 asset metrics** for SOL, AVAX, POL, SHIB

### Nice-to-Have

7. **Dune queries for other L1s** — staking, burn, whale metrics for SOL, BNB, AVAX
8. **DefiLlama DEX volume/fees** — already done for UNI, CRV, AERO, JUP
