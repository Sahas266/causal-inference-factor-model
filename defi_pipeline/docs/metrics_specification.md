# DeFi Metrics Specification

## Overview

This document specifies the exact data requirements, collection strategies, and validation rules for each of the 7 core DeFi metrics in the pipeline.

## 1. Liquidity Flow (LiqFlow)

### Purpose
Track net capital flows into and out of DEX liquidity pools to understand market sentiment and protocol health.

### Data Sources
- **The Graph**: Subgraph data for DEX protocols
- **DefiLlama**: TVL and volume data
- **Direct RPC**: Pool contract events

### Required Fields
```python
class LiqFlowMetric(BaseMetric):
    protocol: str                    # DEX protocol (uniswap, sushiswap, etc.)
    net_inflow: Decimal             # Net USD inflow (positive = inflow)
    tvl_change: Decimal             # Total Value Locked change in USD
    token_flows: Optional[Dict[str, Decimal]]  # Token-specific flows
    add_liquidity_events: Optional[int]        # Number of add events
    remove_liquidity_events: Optional[int]     # Number of remove events
```

### Collection Strategy
- **Frequency**: Every 15 minutes
- **Historical Window**: 24 hours for trend analysis
- **Calculation**:
  ```python
  net_inflow = total_adds_usd - total_removes_usd
  tvl_change = current_tvl - previous_tvl
  ```

### Validation Rules
- `net_inflow` can be negative (net outflows)
- `tvl_change` should be reasonable (±50% from previous value)
- `protocol` must be from approved list
- Token flows should sum to net_inflow

## 2. Stablecoin Flow (Stableflow)

### Purpose
Monitor stablecoin supply mechanics to detect inflationary/deflationary pressures.

### Data Sources
- **Etherscan**: Mint/burn transaction events
- **Dune Analytics**: Aggregated stablecoin data
- **Direct Contracts**: Stablecoin contract calls

### Required Fields
```python
class StableflowMetric(BaseMetric):
    stablecoin: str                 # Symbol (USDT, USDC, DAI, FRAX)
    minted: Decimal                 # Total minted amount
    burned: Decimal                 # Total burned amount
    net_flow: Decimal               # Net flow (minted - burned)
    mint_events: Optional[int]      # Number of mint transactions
    burn_events: Optional[int]      # Number of burn transactions
```

### Collection Strategy
- **Frequency**: Every 15 minutes
- **Coverage**: All major stablecoins (USDT, USDC, DAI, FRAX, USDP, TUSD)
- **Calculation**:
  ```python
  net_flow = minted - burned
  # Track both CeFi (centralized) and DeFi (decentralized) flows
  ```

### Validation Rules
- `minted` and `burned` must be non-negative
- `net_flow = minted - burned`
- Stablecoin symbol must be from approved list
- Large deviations should trigger alerts

## 3. Funding Basis (FundingBasis)

### Purpose
Track perpetual futures funding rates to identify arbitrage opportunities and market stress.

### Data Sources
- **Exchange APIs**: Binance, Bybit, OKX, etc.
- **Coingecko**: Aggregated funding data
- **Direct Websockets**: Real-time funding updates

### Required Fields
```python
class FundingBasisMetric(BaseMetric):
    exchange: str                   # Exchange name
    symbol: str                     # Trading pair (ETH, BTC)
    funding_rate: Decimal           # Current funding rate (percentage)
    premium_index: Decimal          # Premium index value
    mark_price: Optional[Decimal]   # Mark price of perpetual
    index_price: Optional[Decimal]  # Spot index price
```

### Collection Strategy
- **Frequency**: Every 5 minutes (funding rates change frequently)
- **Pairs**: Focus on ETH and BTC across major exchanges
- **Calculation**:
  ```python
  premium_index = (mark_price - index_price) / index_price * 100
  # Funding rate is exchange-provided
  ```

### Validation Rules
- `funding_rate` typically between -0.5% to +0.5%
- `premium_index` should be small percentage
- Prices should be positive and reasonable
- Cross-exchange validation for consistency

## 4. Chain Congestion (ChainCongestion)

### Purpose
Monitor blockchain network health and gas market dynamics.

### Data Sources
- **Etherscan**: Gas oracle and block data
- **Blockscout**: Alternative block explorer
- **Direct RPC**: Real-time network data
- **Mempool APIs**: Pending transaction data

### Required Fields
```python
class ChainCongestionMetric(BaseMetric):
    avg_gas_price: Decimal          # Average gas price in gwei
    block_fullness: Decimal         # Block utilization (0-1)
    pending_txs: int               # Pending transaction count
    base_fee: Optional[Decimal]    # EIP-1559 base fee
    priority_fee: Optional[Decimal] # EIP-1559 priority fee
    block_time: Optional[Decimal]  # Average block time in seconds
```

### Collection Strategy
- **Frequency**: Every 1 minute (gas prices volatile)
- **Calculation**:
  ```python
  avg_gas_price = (safe + standard + fast) / 3
  block_fullness = gas_used / gas_limit
  ```

### Validation Rules
- `avg_gas_price` > 0 and reasonable (< 1000 gwei)
- `block_fullness` between 0 and 1
- `pending_txs` non-negative
- Base fee should be reasonable for network

## 5. Staking Yield (StakingYield)

### Purpose
Track proof-of-stake yields and validator network health.

### Data Sources
- **Beacon Chain APIs**: Validator and reward data
- **Staking Pools**: Lido, Rocket Pool, etc.
- **DefiLlama**: Staking protocol data
- **Direct Contracts**: Staking contract events

### Required Fields
```python
class StakingYieldMetric(BaseMetric):
    staked_amount: Decimal          # Total staked ETH
    apr: Decimal                    # Annual percentage rate
    validator_count: int            # Active validators
    total_validators: Optional[int] # Total registered validators
    slashing_events: Optional[int]  # Slashing events in period
    rewards_distributed: Optional[Decimal]  # Total rewards distributed
```

### Collection Strategy
- **Frequency**: Every 1 hour (staking data changes slowly)
- **Calculation**:
  ```python
  apr = (rewards_distributed / staked_amount) * (365 / days_in_period) * 100
  ```

### Validation Rules
- `staked_amount` should increase over time
- `apr` typically 4-8% for Ethereum
- `validator_count` reasonable for network size
- Rewards should be positive

## 6. MEV Pressure (MEVPressure)

### Purpose
Quantify Maximal Extractable Value extraction and market impact.

### Data Sources
- **MEV-Boost Relays**: Block and extraction data
- **Flashbots**: MEV transaction data
- **Block Explorers**: MEV transaction identification
- **Mempool Analysis**: Frontrunning detection

### Required Fields
```python
class MEVPressureMetric(BaseMetric):
    mev_extracted: Decimal          # Total MEV extracted in period
    mev_blocks_pct: Decimal         # Percentage of blocks with MEV
    top_searcher_share: Decimal     # Market share of top searcher
    sandwich_attacks: Optional[int] # Number of sandwich attacks
    frontrunning_events: Optional[int]  # Number of frontrunning events
    arbitrage_opportunities: Optional[int]  # Exploited arbitrage ops
```

### Collection Strategy
- **Frequency**: Every 30 minutes
- **Analysis Window**: 100-1000 blocks for statistical significance
- **Identification**: MEV transactions by patterns (sandwiches, liquidations, etc.)

### Validation Rules
- `mev_extracted` non-negative
- `mev_blocks_pct` between 0 and 1
- `top_searcher_share` between 0 and 1
- Event counts should be reasonable

## 7. CEX/DEX Flow (CEXDEXFlow)

### Purpose
Monitor cross-chain capital flows between centralized and decentralized exchanges.

### Data Sources
- **Bridge Protocols**: Across, Arbitrum Bridge, etc.
- **CEX APIs**: Deposit/withdrawal data
- **Dune Analytics**: Cross-chain bridge volumes
- **DefiLlama**: Bridge protocol data

### Required Fields
```python
class CEXDEXFlowMetric(BaseMetric):
    bridge: str                     # Bridge protocol name
    direction: str                  # "inflow" or "outflow"
    volume: Decimal                 # Bridge volume in USD
    unique_users: int               # Number of unique users
    transaction_count: Optional[int]  # Number of bridge transactions
    avg_transaction_size: Optional[Decimal]  # Average transaction size
```

### Collection Strategy
- **Frequency**: Every 1 hour
- **Coverage**: Major bridges (Arbitrum, Polygon, Optimism, etc.)
- **Directions**: CEX→DEX and DEX→CEX flows

### Validation Rules
- `direction` must be "inflow" or "outflow"
- `volume` and `unique_users` non-negative
- Bridge names from approved list
- Volume should be reasonable for bridge capacity

## Data Quality Standards

### Completeness
- **Required Fields**: Must be present and valid
- **Optional Fields**: Can be null but should be collected when available
- **Missing Data**: Implement fallback strategies

### Accuracy
- **Cross-validation**: Compare data from multiple sources
- **Reasonableness Checks**: Values within expected ranges
- **Trend Analysis**: Detect anomalous changes

### Freshness
- **Timestamps**: All metrics must have accurate timestamps
- **Latency**: Data should be no more than 5 minutes old
- **Staleness Alerts**: Automatic alerts for stale data

### Consistency
- **Units**: All values in standard units (USD, gwei, etc.)
- **Naming**: Consistent naming conventions
- **Schemas**: Strict adherence to Pydantic schemas

## Error Handling

### Provider Failures
```python
# Fallback to secondary providers
if primary_provider.failed:
    result = secondary_provider.collect()
```

### Data Validation Failures
```python
# Log and skip invalid data
if not collector.validate_data(data):
    logger.warning(f"Invalid data from {provider.name}: {data}")
    continue
```

### Rate Limiting
```python
# Respect provider limits
await rate_limiter.wait_if_needed()
response = await provider.fetch_data(endpoint)
```

## Monitoring and Alerting

### Data Freshness
- Alert if data > 2x collection interval old
- Dashboard showing last update times
- Automated data backfill for gaps

### Data Quality
- Statistical outlier detection
- Cross-source validation
- Trend anomaly alerts

### System Health
- Provider availability monitoring
- Collection success rates
- API response times

## Implementation Checklist

For each metric, ensure:

- [ ] Provider classes implemented
- [ ] Collector classes implemented
- [ ] Database schema created
- [ ] API endpoints working
- [ ] Data validation rules
- [ ] Error handling
- [ ] Unit tests written
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Monitoring configured
