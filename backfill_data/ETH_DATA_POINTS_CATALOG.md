# ETH Data Points Catalog (Allium, CoinGecko, CoinMetrics, Dune, DeFi Llama)

This is a comprehensive ETH-focused pull list across the five providers.

Status tags:
- `Backfilled`: data exists in Supabase `asset_metrics` table.
- `Implemented`: provider/endpoint wired but not yet backfilled (or partially backfilled).
- `Planned`: provider/endpoint not yet fully wired in this repo, but recommended.

> **Last verified: 2026-03-11** — 86,065 rows across 5 providers, 42 distinct metrics.

## 1. Core Market + Network State (Daily Minimum)

| Data Point | Canonical Metric | Provider(s) | Status | Notes |
|---|---|---|---|---|
| ETH spot price (USD) | `price_usd` | CoinMetrics, CoinGecko, DeFi Llama, Allium | Backfilled (CM/DL/Allium/CG) | Anchor series for all downstream models. |
| Market cap (USD) | `market_cap_usd` | CoinMetrics, CoinGecko | Backfilled (CM/CG) | Use circulating market cap where possible. |
| Fully diluted valuation | `fdv_usd` | CoinGecko | Implemented (coin_data endpoint) | Useful for supply overhang context. |
| 24h spot volume (USD) | `spot_volume_usd_24h` | CoinGecko, CoinMetrics | Backfilled (CG) | Normalize exchange-aggregated vs venue-specific. |
| Daily tx count | `tx_count` | CoinMetrics, Allium, Dune | Backfilled (CM) | ETH L1 usage baseline. |
| Daily active addresses | `active_addresses` | CoinMetrics, Allium, Dune | Backfilled (CM) | EOA + contract split is useful if available. |
| Avg gas price (gwei) | `avg_gas_price_gwei` | Allium, Dune, CoinMetrics | Implemented (Allium), Planned (Dune/CM extension) | Key congestion proxy. |
| Gas used / gas limits | `gas_used`, `gas_limit_tx`, `gas_limit_block` | CoinMetrics, Dune, Allium | Implemented (CM limit metrics), Planned (Dune/Allium extension) | Pair with fee/burn analysis. |
| Total daily fees (ETH/USD) | `fees_eth`, `fees_usd` | Allium, Dune, DeFi Llama | Implemented (Allium), Planned (Dune/DL mapping) | Post-EIP-1559 decomposition preferred. |

## 2. Price Structure + Liquidity

| Data Point | Canonical Metric | Provider(s) | Status | Notes |
|---|---|---|---|---|
| Pair OHLCV (ETH-USD/USDT) | `pair_ohlcv` | CoinMetrics, CoinGecko | Implemented (CM), Planned (CG) | Good for asset-level daily bars. |
| Venue-specific OHLCV | `market_ohlcv` | CoinMetrics | Implemented | Coinbase/Binance venue granularity. |
| Trade-level ticks | `trade_price`, `trade_amount`, `trade_side` | CoinMetrics | Implemented | Needed for realized volatility, microstructure. |
| Orderbook depth levels | `bid_ask_depth_l{n}` | CoinMetrics | Implemented | Build liquidity/impact metrics. |
| Spread and microprice | `spread_bps`, `microprice` | CoinMetrics, Dune | Planned (derived) | Derived from orderbook snapshots. |
| Realized volatility | `rv_1d`, `rv_7d`, `rv_30d` | CoinMetrics, CoinGecko | Planned (derived) | Derived from OHLCV or tick returns. |

## 3. Derivatives + Options

| Data Point | Canonical Metric | Provider(s) | Status | Notes |
|---|---|---|---|---|
| Futures open interest | `open_interest_contracts`, `open_interest_usd` | CoinMetrics, CoinGecko | Implemented (CM), Planned (CG) | Per exchange and aggregated. |
| Funding rates | `funding_rate` | CoinMetrics, CoinGecko | Implemented (CM), Planned (CG) | Keep interval metadata (e.g. 8h). |
| Liquidation flow | `liq_amount`, `liq_notional_usd` | CoinMetrics, CoinGecko | Implemented (CM), Planned (CG) | Separate long vs short liquidations. |
| Basis / premium index | `futures_basis`, `perp_premium` | CoinGecko, Dune, CoinMetrics | Planned | Useful for positioning and stress regimes. |
| Options implied vol | `implied_volatility` | CoinMetrics | Implemented | Per contract and surface bucketing. |
| Options greeks | `delta`, `gamma`, `vega`, `theta`, `rho` | CoinMetrics | Implemented | Use for dealer positioning proxies. |
| Put/Call skew and term structure | `iv_skew_25d`, `iv_term_structure` | CoinMetrics, Dune | Planned (derived) | Derived from options chain snapshots. |

## 4. DeFi + Stablecoins (ETH Ecosystem)

| Data Point | Canonical Metric | Provider(s) | Status | Notes |
|---|---|---|---|---|
| Ethereum chain TVL | `eth_chain_tvl_usd` | DeFi Llama | Implemented | Macro DeFi risk appetite proxy. |
| Protocol TVL (Aave/Uniswap/Curve/Lido/...) | `protocol_tvl_usd` | DeFi Llama | Implemented | Pull top ETH protocols daily. |
| DEX volume by protocol | `dex_volume_usd` | DeFi Llama, Allium, Dune | Implemented (DL/Allium), Planned (Dune) | Reconcile per protocol vs chain total. |
| Protocol fees/revenue | `protocol_fees_usd`, `protocol_revenue_usd` | DeFi Llama, Dune | Implemented (DL fees), Planned (Dune expansion) | Helps identify value accrual vs activity. |
| Stablecoin circulating on Ethereum | `stablecoin_circulating_usd` | DeFi Llama, Dune | Implemented (DL), Planned (Dune) | USD liquidity context for ETH risk. |
| Stablecoin inflow/outflow | `stablecoin_netflow_usd` | DeFi Llama, Dune | Planned | Often predictive for short-term beta moves. |

## 5. Supply, Staking, and Monetary Regime

| Data Point | Canonical Metric | Provider(s) | Status | Notes |
|---|---|---|---|---|
| Circulating supply | `eth_circulating_supply` | CoinMetrics, CoinGecko | Planned (explicit mapping) | Supply-adjusted valuation metrics. |
| Net issuance / burn | `eth_net_issuance`, `eth_burned` | CoinMetrics, Dune, Allium | Backfilled (Dune 6811496) | eth_burned, block_count from EIP-1559 (Aug 2021). |
| Staked ETH | `eth_staked` | Dune, CoinGecko, DeFi Llama | Backfilled (Dune 6811495) | eth_staked_daily, cumulative_eth_staked, depositor_count, deposit_count. |
| Validator count | `validator_count` | Dune | Backfilled (Dune 6811495) | Via depositor_count proxy. |
| Staking yield (APR) | `staking_apr` | CoinGecko, Dune | Planned | Compare against funding/carry signals. |
| Exchange reserves (ETH) | `exchange_reserve_eth` | CoinMetrics, Dune | Planned | Supply-on-exchange risk indicator. |

## 6. Flow + Behavioral Signals

| Data Point | Canonical Metric | Provider(s) | Status | Notes |
|---|---|---|---|---|
| Exchange inflow/outflow | `exchange_inflow_eth`, `exchange_outflow_eth` | CoinMetrics, Dune | Backfilled (CM: FlowInExNtv/FlowOutExNtv, Dune 6811499: cex_inflow/outflow/netflow_usd) | Spot selling/buying pressure proxy. |
| Whale transfer volume | `whale_transfer_usd` | Allium, Dune | Backfilled (Dune 6811497) | whale_transfer_usd, whale_transfer_count (>$1M). |
| CEX vs DEX share | `dex_cex_volume_ratio` | DeFi Llama, CoinMetrics, Dune | Planned (derived) | Market-structure regime indicator. |
| Bridge inflow/outflow to Ethereum | `bridge_netflow_usd` | DeFi Llama, Dune | Backfilled (Dune 6811498) | bridge_outflow/inflow/netflow_usd, deposit/withdrawal_count. |
| L2 to L1 settlement activity | `l2_settlement_value_usd` | Dune, Allium | Planned | Helpful for Ethereum demand decomposition. |

## 7. Recommended ETH Pull Pack by Provider

### Allium
- `active_addresses`
- `tx_count`
- `avg_gas_price_gwei`
- `total_fees_eth`
- `dex_volume_usd`
- `dex_trade_count`
- `eth_token_price_ohlc` (`price_usd`, `open_usd`, `high_usd`, `low_usd`, `close_usd`)

### CoinGecko (backfilled — 1,368 rows, last 365 days)
- **Backfilled**: `price_usd`, `market_cap_usd`, `spot_volume_usd_24h` (via `market_chart`)
- **Implemented but not backfilled**: `fdv_usd`, `total_supply`, `max_supply`, `ath_usd`, `atl_usd` (via `coin_data`)
- **Limitation**: Demo key (`CG-` prefix) restricts to last 365 days; Pro key needed for full range

### CoinMetrics
- Asset metrics: `PriceUSD`, `CapMrktCurUSD`, `TxCnt`, `AdrActCnt`, `GasLmtTx`, `GasLmtBlk`
- Pair candles: ETH pairs (`eth-usd`)
- Market candles: `coinbase-eth-usd-spot`, `binance-eth-usdt-spot`
- Trades: price/amount/side
- Orderbooks: multi-level bid/ask depth
- Perp/futures: open interest, funding rates, liquidations
- Options: implied volatility + greeks

### Dune (5 queries backfilled — 30,610 rows)
- **6811495**: Staking — eth_staked_daily, depositor_count, deposit_count, cumulative_eth_staked
- **6811496**: Burn — eth_burned, block_count (from EIP-1559 Aug 2021)
- **6811497**: Whales — whale_transfer_usd, whale_transfer_count (>$1M)
- **6811498**: Bridges — bridge_outflow/inflow/netflow_usd, deposit/withdrawal_count
- **6811499**: CEX Flows — cex_inflow/outflow/netflow_usd, cex_transfer_count
- Planned: L2 settlement, staking APR, stablecoin flows

### DeFi Llama
- Ethereum chain TVL
- Top protocol TVLs on Ethereum
- DEX volumes (daily)
- Protocol fees/revenue
- Stablecoin circulating and flow
- Coin price history (`coingecko:ethereum`)

## 8. Priority Pull Order for Modeling

1. Price + market cap + volume
2. On-chain activity (tx count, active addresses, gas, fees)
3. Derivatives positioning (OI, funding, liquidations, IV)
4. DeFi state (chain/protocol TVL, DEX volume, fees)
5. Stablecoin and bridge liquidity
6. Staking/issuance regime metrics

