# ETH Backfill Plan — All Providers

> Generated: 2026-03-09 | **Last updated: 2026-03-11**
> Date range: **2021-01-01 → 2026-01-01** (5 years daily)
> Target table: `asset_metrics` in Supabase (`jnulpcqpftnwvknwuqpa`)
> **Current total: 86,065 rows across 5 providers, 42 distinct metrics**

---

## Provider Access Summary (Verified Live)

| Provider | Auth | Status | Rows | Limits |
|---|---|---|---|---|
| **CoinMetrics** | Community (no key) | **DONE** — 13 core metrics backfilled | 23,751 | 6000 req/20s |
| **DefiLlama** | Free (no key) | **DONE** — chain/protocol TVL, DEX volumes, fees, stablecoins | 21,201 | 200 req/min |
| **Allium** | `ALLIUM_API_KEY` | **DONE** — WETH OHLCV fallback | 9,135 | 60 req/min |
| **CoinGecko** | `COINGECKO_API_KEY` (Demo) | **DONE** — last 365 days only (Demo key limit) | 1,368 | 30 req/min |
| **Dune** | `DUNE_API_KEY` | **DONE** — 5 queries created and backfilled | 30,610 | 40 req/min |

---

## Data Point Assignment by Provider

### Priority Rules
1. Use **CoinMetrics community** for everything it provides free (price, market cap, tx count, addresses, supply, issuance, exchange flows)
2. Use **DefiLlama** for DeFi-specific data (TVL, DEX volumes, fees, stablecoins, coin prices as fallback)
3. Use **Allium** as price fallback only (WETH OHLCV via Developer API)
4. **Dune** and **CoinGecko** deferred — require UI query creation and MCP restart respectively

---

## Phase 1: CoinMetrics Community — Core ETH Market & On-Chain Metrics

### Available metrics (verified working on community tier):

| Metric ID | Description | Canonical Name |
|---|---|---|
| `PriceUSD` | ETH spot price | `price_usd` |
| `CapMrktCurUSD` | Market cap (circulating) | `market_cap_usd` |
| `TxCnt` | Daily transaction count | `tx_count` |
| `AdrActCnt` | Daily active addresses | `active_addresses` |
| `SplyCur` | Circulating supply | `circulating_supply` |
| `BlkCnt` | Block count | `block_count` |
| `HashRate` | Network hash rate | `hash_rate` |
| `ROI30d` | 30-day return on investment | `roi_30d` |
| `FeeTotNtv` | Total fees in ETH | `fees_eth` |
| `IssTotNtv` | Total issuance in ETH | `issuance_eth` |
| `FlowInExNtv` | Exchange inflow (ETH) | `exchange_inflow_eth` |
| `FlowOutExNtv` | Exchange outflow (ETH) | `exchange_outflow_eth` |
| `TxTfrCnt` | Transfer count | `transfer_count` |

### Blocked on community tier (403):
`FeeMeanNtv`, `FeeMedNtv`, `GasUsedTx`, `GasPriceAvg`, `RevNtv`, `DiffMean`,
`VtyDayRet30d`, `VtyDayRet7d`, `SplyAct1d`, `SplyAct30d`, `NVTAdj`, `TxTfrValAdjNtv`

### Endpoint config:
- **File**: `config/endpoints/eth/eth_asset_metrics_primary.json`
- **Action**: Update `metrics` param to include all 13 working metrics
- **Update `date_range`** to `2021-01-01 → 2026-01-01`

### Expected output:
- 13 metrics × ~1,826 days = **~23,738 records** in `asset_metrics`

---

## Phase 2: DefiLlama — DeFi Ecosystem Metrics

### 2a. Ethereum Chain TVL
- **Endpoint**: `GET /v2/historicalChainTvl/Ethereum`
- **Metric**: `tvl_usd` for asset `ethereum`
- **Config**: `config/endpoints/eth/eth_chain_tvl_defillama.json`
- **Action**: Update `date_range` to `2021-01-01 → 2026-01-01`
- **Expected**: ~1,826 records

### 2b. Protocol TVL (Top 5 ETH Protocols)
- **Endpoint**: `GET /protocol/{protocol}` for each
- **Protocols**: `aave`, `uniswap`, `curve-dex`, `lido`, `makerdao`
- **Metric**: `tvl_usd` per protocol
- **Config**: `config/endpoints/eth/eth_protocol_tvl_defillama.json`
- **Action**: Update `date_range` to `2021-01-01 → 2026-01-01`, add `makerdao`
- **Expected**: ~9,130 records (5 × ~1,826)

### 2c. DEX Volumes (Uniswap + Curve)
- **Endpoint**: `GET /summary/dexs/{protocol}`
- **Protocols**: `uniswap`, `curve-dex`
- **Metric**: `volume_usd`
- **Config**: `config/endpoints/eth/eth_dex_volumes_defillama.json`
- **Action**: Update `date_range`, add `curve-dex` as second provider entry
- **Expected**: ~3,652 records

### 2d. Protocol Fees (Uniswap + Lido + Aave)
- **Endpoint**: `GET /summary/fees/{protocol}`
- **Protocols**: `uniswap`, `lido`, `aave`
- **Metric**: `fees_usd`
- **Config**: `config/endpoints/eth/eth_fees_defillama.json`
- **Action**: Update `date_range`, add `lido` and `aave` as provider entries
- **Expected**: ~5,478 records

### 2e. Stablecoin Circulating on Ethereum
- **Endpoint**: `GET /stablecoincharts/Ethereum` (on `stablecoins.llama.fi`)
- **Metric**: `stablecoin_circulating_usd`
- **Config**: `config/endpoints/eth/eth_stablecoin_flow_defillama.json`
- **BUG FIX REQUIRED**: `date` field is a string epoch (`"1511913600"`), not int. The transformer's `_in_range()` calls `datetime.fromtimestamp(ts)` which fails on strings.
- **Fix**: Cast `ts` to `int()` in `_in_range()` and `_unix_to_iso()`
- **Expected**: ~1,826 records

### 2f. ETH Price History (DefiLlama coins API — fallback)
- **Endpoint**: `GET /chart/coingecko:ethereum` on `coins.llama.fi`
- **Metric**: `price_usd` for asset `eth`
- **Config**: `config/endpoints/eth/eth_asset_metrics_primary.json` (already has DefiLlama as priority 2)
- **Note**: CoinMetrics is primary for price; DefiLlama is the fallback
- **Expected**: ~1,826 records (deduplicated against CoinMetrics via `prefer_priority`)

---

## Phase 3: Allium Developer API — WETH OHLCV Fallback

### 3a. WETH Price OHLCV
- **Endpoint**: `POST /developer/prices/history`
- **Token**: WETH `0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2` on Ethereum
- **Metrics**: `price_usd`, `open_usd`, `high_usd`, `low_usd`, `close_usd`
- **Config**: `config/endpoints/eth_fallback/eth_price_allium_fallback.json`
- **Action**: Update `date_range` to `2021-01-01 → 2026-01-01`
- **Expected**: 5 metrics × ~1,826 days = ~9,130 records
- **Priority**: 99 (only used when CoinMetrics + DefiLlama unavailable)

### 3b. DEX Trades (NOT viable)
- **Endpoint**: `GET /developer/ethereum/dex/trades`
- **Status**: Requires `block_number` or `transaction_hash` filter — cannot do time-range bulk queries
- **Action**: Skip for backfill

---

## Phase 4: Dune — COMPLETED (2026-03-11)

5 queries created and backfilled (30,610 total records):

| Query ID | Name | Metrics | Records | Date Range |
|---|---|---|---|---|
| 6811495 | ETH Staking Stats | eth_staked_daily, depositor_count, deposit_count, cumulative_eth_staked | 7,304 | 2021-01-01 → 2026-01-01 |
| 6811496 | ETH Burn/Issuance | eth_burned, block_count | 3,220 | 2021-08-05 → 2026-01-01 |
| 6811497 | Whale Transfers | whale_transfer_usd, whale_transfer_count | 3,652 | 2021-01-01 → 2026-01-01 |
| 6811498 | Bridge Flows | bridge_outflow/inflow/netflow_usd, deposit/withdrawal_count | 9,130 | 2021-01-01 → 2026-01-01 |
| 6811499 | CEX Flows | cex_inflow/outflow/netflow_usd, cex_transfer_count | 7,304 | 2021-01-01 → 2026-01-01 |

Endpoint configs: `config/endpoints/eth/eth_dune_staking.json`, `eth_dune_burn.json`, `eth_dune_whales.json`, `eth_dune_bridges.json`, `eth_dune_cex_flows.json`

Dune CLI installed at `~/.local/bin/dune.exe` for query management.

---

## Phase 5: CoinGecko — COMPLETED (2026-03-11)

Provider built and backfilled (1,368 records):

| Metric | Records | Notes |
|---|---|---|
| `price_usd` | 456 | ~365 daily data points |
| `market_cap_usd` | 456 | ~365 daily data points |
| `spot_volume_usd_24h` | 456 | ~365 daily data points |

**Limitation**: Using Demo API key (`CG-` prefix) which limits data to last 365 days. Pro key needed for full 2021-01-01 range.

**Not yet backfilled** (require Pro key or `coin_data` endpoint):
- `fdv_usd`, `total_supply`, `max_supply`, `ath_usd`, `atl_usd` (snapshot data from `/coins/{id}`)

Endpoint config: `config/endpoints/eth/eth_coingecko.json`

---

## Actual Record Counts (Verified 2026-03-11, updated after gap resolution)

| Provider | Rows | Distinct Metrics |
|---|---|---|
| coinmetrics | 23,751 | 13 |
| defillama | 21,201 | 4 |
| allium | 9,135 | 5 |
| coingecko | 1,372 | 7 (price_usd, market_cap_usd, spot_volume_usd_24h + fdv_usd, total_supply, ath_usd, atl_usd snapshots) |
| dune | 49,752 | 28 (across 9 queries: original 5 + staking_apr, net_issuance, l2_settlement, stablecoin_netflow) |
| derived | 5,443 | 3 (realized_volatility_7d, realized_volatility_30d, dex_cex_volume_ratio) |
| **Total** | **110,654** | **59** |

---

## Resolved Gaps (2026-03-11)

### DefiLlama Stablecoin Transformer Bug — FIXED
- **Bug**: `_unix_to_iso()` and `_in_range()` failed on string-epoch timestamps like `"1511913600"`
- **Fix**: Changed `int(ts)` to `int(float(ts))` in both functions in `src/providers/defillama/transformer.py`
- **Why `float()`**: Ensures robustness for both string integers and potential float-strings

### CoinGecko Snapshot Data — DONE (4 records)
- Created `config/endpoints/eth/eth_coingecko_snapshot.json` (endpoint_type: `coin_data`)
- Backfilled: `fdv_usd`, `total_supply`, `ath_usd`, `atl_usd` (max_supply is null for ETH)
- These are point-in-time snapshots, not historical timeseries

### Dune — 4 New Queries Created & Backfilled (19,142 records)
| Query ID | Name | Source Table | Records | Metrics |
|---|---|---|---|---|
| 6815240 | ETH Staking APR | `staking_ethereum.flows` | 5,478 | daily_rewards_eth, cumulative_staked_eth, staking_apr |
| 6815241 | ETH Net Issuance | `staking_ethereum.flows` + `ethereum.blocks` | 4,830 | consensus_rewards_eth, eth_burned_calc, eth_net_issuance |
| 6815242 | L2 Settlement | `rollup_economics_ethereum.l2_revenue` | 3,356 | l2_settlement_value_usd, l2_chain_count |
| 6815244 | Stablecoin Netflow | `stablecoins_ethereum.transfers` | 5,478 | stablecoin_minted_usd, stablecoin_burned_usd, stablecoin_netflow_usd |

### Derived Metrics — DONE (5,443 records)
Computed from existing Supabase data via `scripts/compute_derived_metrics.py`:
- `realized_volatility_7d` (1,820 records): 7-day rolling annualized % from CoinMetrics PriceUSD log returns
- `realized_volatility_30d` (1,797 records): 30-day rolling annualized % from CoinMetrics PriceUSD log returns
- `dex_cex_volume_ratio` (1,826 records): DefiLlama volume_usd (sum) / abs(Dune cex_netflow_usd)
- All tagged with `provider='derived'`, `provider_priority=99`, `metadata.source` documenting inputs

---

## Remaining Gaps (Require Paid Keys)

| Item | Blocker | Est. Rows |
|---|---|---|
| CoinGecko full history (2021-01-01 → 2025-03-12) | Pro API key (~$130/mo) | ~4,380 |
| CoinMetrics derivatives (candles, OI, funding, liquidations) | Pro API key | ~30,000+ |
| CoinMetrics advanced metrics (GasPriceAvg, VtyDayRet, NVTAdj, etc.) | Pro API key | ~20,000+ |
| Derived: spread_bps, microprice, futures_basis, perp_premium | Requires CoinMetrics derivatives data | N/A |
