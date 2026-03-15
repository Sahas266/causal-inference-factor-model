# Backfill Status — All 26 Coins

**Last updated:** 2026-03-15
**Total rows in `asset_metrics`:** ~430,300
**Distinct assets:** 29 (26 target coins + macro + 2 legacy ETH-era names)
**Distinct metrics:** 104
**Distinct providers:** 7

## Database Schema

### Tables (12 total)

| Table | Rows | Primary Keys | Notes |
|-------|------|-------------|-------|
| `asset_metrics` | ~430,300 | `(provider, asset, metric, time)` | Main data table — all coin metrics |
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
| `provider` | text | coinmetrics, coingecko, defillama, allium, dune, derived, fred |
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
| Dune | 138,137 | 4 (eth, sol, bnb, avax) | 42 | 2021-01-01 → 2025-12-31 |
| Derived | 49,330 | 26 (all coins) | 3 (realized_volatility_7d/30d, dex_cex_volume_ratio) | 2021-01-01 → 2026-01-01 |
| DefiLlama | 40,667 | 19 | 4 (tvl_usd, fees_usd, volume_usd, stablecoin_circulating_usd) | 2021-01-01 → 2026-01-01 |
| CoinGecko | 22,772 | 26 (all coins — snapshots + market_chart) | 11 | 2025-03-15 → 2026-03-15 |
| FRED | 20,452 | 1 (macro) | 30 | 2021-01-01 → 2026-01-01 |
| Allium | 18,270 | 2 (btc, eth OHLCV fallback) | 5 | 2021-01-01 → 2026-01-01 |
| **Total** | **~430,300** | | | |

---

## Per-Coin Data Coverage

### Legend

- **CM** = CoinMetrics (priority 1) — community tier, no key needed
- **CG** = CoinGecko (priority 3) — snapshots + market_chart (last 365 days via free tier days=365 endpoint)
- **DL** = DefiLlama (priority 2) — free, no key needed
- **AL** = Allium (priority 4) — fallback OHLCV
- **DU** = Dune — on-chain analytics (ETH, SOL, BNB, AVAX)
- **DR** = Derived — computed from other providers (all 26 coins)
- **FR** = FRED — Federal Reserve macro data (asset="macro")

### L1 Major Chains

| Coin | CM Timeseries | CG Snapshot | DL Chain TVL | DL Fees | DL DEX Volume | Other | Total Rows |
|------|--------------|-------------|--------------|---------|---------------|-------|------------|
| **BTC** | 13 metrics (23,751) | 5 metrics | tvl_usd (1,748) | -- | -- | Allium OHLCV (9,135), DR vol 7d/30d (3,617) | **34,644** |
| **ETH** | 13 metrics (23,751) | 7 metrics | tvl_usd (1,827) | -- | -- | Allium (9,135), Dune 42 metrics (72,442), DR 3 metrics (5,443) | **115,797** |
| **SOL** | -- | 4 metrics | tvl_usd (1,751) | -- | -- | -- | **1,759** |
| **BNB** | 5 metrics (9,135) | 5 metrics | tvl_usd (1,827) | -- | -- | Dune 14 metrics (12,755), DR vol 7d/30d (3,617) | **23,727** |
| **XRP** | 10 metrics (12,789) | 5 metrics | -- | -- | -- | DR vol 7d/30d (3,617) | **12,799** |
| **AVAX** | -- | 5 metrics | tvl_usd (1,794) | -- | -- | Dune 14 metrics (22,264) | **24,068** |
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

### FRED Macro Data

| Config | Series | Count |
|--------|--------|-------|
| `fred_interest_rates` | DFF, DGS2, DGS10, DGS30, DFEDTARU, T10Y2Y, T10Y3M | 7 |
| `fred_inflation` | CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, T5YIE, T10YIE, MICH | 7 |
| `fred_money_supply` | M2SL, WALCL, RRPONTSYD | 3 |
| `fred_risk_volatility` | VIXCLS, BAMLH0A0HYM2, TEDRATE | 3 |
| `fred_labor_growth` | UNRATE, PAYEMS, ICSA, GDPC1, INDPRO | 5 |
| `fred_commodities` | DCOILWTICO, PPIACO | 2 |
| `fred_financial_conditions` | NFCI, STLFSI2 | 2 |
| `fred_dollar` | DTWEXBGS | 1 |
| **Total** | | **30 series, 20,452 rows** |

- Provider: `fred`, asset: `macro`, metric: series_id (e.g. `DFF`, `VIXCLS`)
- Date range: 2021-01-01 → 2026-01-01
- 8 endpoint configs in `config/endpoints/macro/`

---

## Legacy Asset Names (not in 26-coin list)

| `asset` | Provider | Rows | Notes |
|---------|----------|------|-------|
| `lido` | defillama | 3,534 | ETH ecosystem protocol (TVL + fees) |
| `makerdao` | defillama | 1,827 | ETH ecosystem protocol (TVL) |

Previously renamed: `uniswap`→`uni`, `curve`→`crv`, `ethereum`→`eth`, `eth_stablecoins`→`eth` (merged).

---

## What's Missing — Gaps by Provider

### CoinGecko Market Chart

**Status:** Snapshots + market_chart working via free tier `days=365` endpoint (22,772 rows total).
**Coverage:** Last 365 days of daily price/mcap/volume for all 26 coins.
**Remaining gap:** Full historical range (2021-01-01 onward) requires CoinGecko Pro key (`pro-api.coingecko.com`, 500 req/min).
**Metrics:** `price_usd`, `market_cap_usd`, `spot_volume_usd_24h`

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

### Dune — ETH + SOL/BNB/AVAX

Dune queries now cover ETH (12 queries, 42 metrics) + SOL (3 queries) + BNB (5 queries) + AVAX (5 queries). Total: 138,137 rows.

**CPCM causal factor queries:**

| Chain | Block Congestion | Liquidations | Flashloans/LP Flow |
|-------|-----------------|--------------|-------------------|
| ETH | 6831658 (11,918 rows) | 6831664 (6,904 rows) | 6831661 (5,478 rows) |
| BNB | 6831685 (8,898 rows) | 6831687 (1,616 rows) | 6831689 (2,241 rows) |
| AVAX | 6831686 (11,842 rows) | 6831688 (5,808 rows) | 6831690 (4,614 rows) |

Metrics: `avg_gas_utilization`, `max_gas_utilization`, `avg_base_fee_gwei`, `stddev_base_fee_gwei`, `max/min_base_fee_gwei`, `block_count`, `liquidation_count`, `unique_liquidators`, `liquidation_volume_usd`, `avg_liquidation_usd`, `flashloan_count`, `flashloan_volume_usd`, `unique_flashloan_users`

**SOL**: No block congestion/liquidation/flashloan queries — Solana has a different block structure (no gas_used/gas_limit/base_fee) and no `lending.borrow`/`lending.flashloans` spellbook coverage.

No Dune coverage for remaining coins.

### Derived Metrics

Realized volatility (7d/30d) computed for all 26 coins (49,330 rows). DEX/CEX volume ratio for ETH (requires Dune CEX data).

---

## Priority Actions

### Completed

1. **All non-CoinGecko endpoints are now backfilled** for all 26 coins
2. Derived metrics (volatility 7d/30d) computed for all 26 coins
3. CoinGecko market_chart working for all 26 coins (last 365 days via free tier)
4. FRED macro data backfilled (30 series, 20,452 rows)
5. Dune expanded to SOL, BNB, AVAX
6. **CPCM causal factor data**: Chain Congestion (gas utilization, base fee), MEV Pressure proxy (base fee volatility), Liquidation Cascades, LP Flow proxy (flashloans) — 3 new Dune queries, 14 new metrics, 24,300 rows

### Requires CoinGecko Pro Key

6. **Full CoinGecko historical range** (2021-01-01 onward) — currently limited to last 365 days

### Requires CoinMetrics Pro Key or Paid API

7. **Funding Basis (perp funding rates)** — no free source available; requires CoinMetrics Pro or Hyperliquid API
8. **Derivatives data** — open interest, funding rates, liquidations (all empty market_* tables)
9. **Tier 2/3 asset metrics** for SOL, AVAX, POL, SHIB

### Nice-to-Have

9. **Dune queries for remaining L1s** — staking, burn, whale metrics for other coins
10. **DefiLlama DEX volume/fees** — already done for UNI, CRV, AERO, JUP
