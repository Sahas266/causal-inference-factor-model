# ETH Backfill Plan — All Providers

> Generated: 2026-03-09
> Date range: **2021-01-01 → 2026-01-01** (5 years daily)
> Target table: `asset_metrics` in Supabase (`jnulpcqpftnwvknwuqpa`)

---

## Provider Access Summary (Verified Live)

| Provider | Auth | Status | Limits |
|---|---|---|---|
| **CoinMetrics** | Community (no key) | Working — subset of metrics available | 6000 req/20s |
| **DefiLlama** | Free (no key) | Working — all free endpoints | 200 req/min |
| **Allium** | `ALLIUM_API_KEY` | Working — Developer REST API only (Explorer SQL out of credits) | 60 req/min |
| **Dune** | `DUNE_API_KEY` | Key present but API requires pre-saved query IDs — **not viable for custom SQL** without creating queries in the UI first |
| **CoinGecko** | MCP configured (not yet activated) | Not available this session — requires Claude Code restart |

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

## Phase 4: Dune (Deferred)

Dune's API model requires pre-saved queries (created in the Dune UI) referenced by query ID. Custom SQL cannot be run ad-hoc via API alone.

**If/when Dune queries are created in the UI**, the following would be high-value additions:

| Query | Metrics | Priority |
|---|---|---|
| ETH daily staking stats | `eth_staked`, `validator_count`, `staking_apr` | High |
| ETH burn/issuance post-merge | `eth_burned`, `eth_net_issuance` | High |
| ETH L2 settlement value | `l2_settlement_value_usd` | Medium |
| Whale transfers >$1M | `whale_transfer_usd`, `whale_transfer_count` | Medium |

**Action**: Create queries in Dune UI, save them, then build the provider to fetch results by query ID.

---

## Phase 5: CoinGecko (Deferred)

CoinGecko MCP is configured in `.mcp.json` but requires a Claude Code restart to activate.

**Planned data points (when available):**

| Data Point | Metric | Notes |
|---|---|---|
| Market cap (USD) | `market_cap_usd` | Redundant with CoinMetrics |
| Fully diluted valuation | `fdv_usd` | **Unique to CoinGecko** |
| 24h volume | `spot_volume_usd_24h` | Cross-reference with CM |
| Total/max supply | `total_supply`, `max_supply` | Supply context |
| ATH/ATL + drawdowns | `ath_usd`, `atl_usd`, `ath_drawdown_pct` | **Unique** |

**Action**: Restart Claude Code, verify MCP access, then build provider.

---

## Implementation Steps (In Order)

### Step 1: Fix bugs
1. **Fix stablecoin transformer string-epoch bug** in `src/providers/defillama/transformer.py`
   - `_unix_to_iso(ts)` and `_in_range(ts, ...)` must cast `ts = int(ts)` to handle string epochs
2. **Fix CoinMetrics 403 handling** — treat 403 as `fatal: True` instead of retrying

### Step 2: Update endpoint configs
1. `eth_asset_metrics_primary.json` — expand metrics to all 13 working ones, date range → 2021-01-01
2. `eth_chain_tvl_defillama.json` — date range → 2021-01-01
3. `eth_protocol_tvl_defillama.json` — date range → 2021-01-01, add makerdao
4. `eth_dex_volumes_defillama.json` — date range → 2021-01-01, add curve-dex
5. `eth_fees_defillama.json` — date range → 2021-01-01, add lido + aave
6. `eth_stablecoin_flow_defillama.json` — date range → 2021-01-01
7. `eth_price_allium_fallback.json` — date range → 2021-01-01

### Step 3: Reset stale backfill progress
- Delete existing `backfill_progress` rows (they have the old 2025-01-01 range and stale states)

### Step 4: Run backfills in order
```bash
cd backfill_data

# Phase 1: CoinMetrics core metrics (largest, most important)
python backfill.py --config config/endpoints/eth/eth_asset_metrics_primary.json

# Phase 2: DefiLlama DeFi data
python backfill.py --config config/endpoints/eth/eth_chain_tvl_defillama.json
python backfill.py --config config/endpoints/eth/eth_protocol_tvl_defillama.json
python backfill.py --config config/endpoints/eth/eth_dex_volumes_defillama.json
python backfill.py --config config/endpoints/eth/eth_fees_defillama.json
python backfill.py --config config/endpoints/eth/eth_stablecoin_flow_defillama.json

# Phase 3: Allium fallback
python backfill.py --config config/endpoints/eth_fallback/eth_price_allium_fallback.json
```

### Step 5: Verify in Supabase
```sql
SELECT provider, asset, metric, COUNT(*) as rows,
       MIN(time) as earliest, MAX(time) as latest
FROM asset_metrics
GROUP BY provider, asset, metric
ORDER BY provider, asset, metric;
```

---

## Expected Final Record Count

| Provider | Asset | Metrics | Est. Records |
|---|---|---|---|
| coinmetrics | eth | 13 core metrics | ~23,738 |
| defillama | ethereum | tvl_usd | ~1,826 |
| defillama | aave | tvl_usd | ~1,826 |
| defillama | uniswap | tvl_usd, volume_usd, fees_usd | ~5,478 |
| defillama | curve | tvl_usd, volume_usd | ~3,652 |
| defillama | lido | tvl_usd, fees_usd | ~3,652 |
| defillama | makerdao | tvl_usd | ~1,826 |
| defillama | eth_stablecoins | stablecoin_circulating_usd | ~1,826 |
| defillama | eth | price_usd (fallback) | ~1,826 |
| allium | eth | price_usd, open/high/low/close_usd | ~9,130 |
| **Total** | | | **~54,780** |

---

## Files to Modify

| File | Change |
|---|---|
| `src/providers/defillama/transformer.py` | Fix string-epoch bug in `_unix_to_iso` and `_in_range` |
| `src/providers/coinmetrics/provider.py` | Treat 403 as fatal (no retry) |
| `config/endpoints/eth/eth_asset_metrics_primary.json` | Expand metrics, widen date range |
| `config/endpoints/eth/eth_chain_tvl_defillama.json` | Widen date range |
| `config/endpoints/eth/eth_protocol_tvl_defillama.json` | Add makerdao, widen date range |
| `config/endpoints/eth/eth_dex_volumes_defillama.json` | Add curve-dex, widen date range |
| `config/endpoints/eth/eth_fees_defillama.json` | Add lido + aave, widen date range |
| `config/endpoints/eth/eth_stablecoin_flow_defillama.json` | Widen date range |
| `config/endpoints/eth_fallback/eth_price_allium_fallback.json` | Widen date range |
