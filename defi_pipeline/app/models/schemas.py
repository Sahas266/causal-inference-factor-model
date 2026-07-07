"""
Pydantic schemas for DeFi data pipeline metrics.
All schemas follow the Receive an Object, Return an Object (RORO) pattern.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class BaseMetric(BaseModel):
    """Base class for all metric schemas with common fields."""

    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime = Field(..., description="Timestamp of the metric data")
    chain: Optional[str] = Field(None, description="Blockchain network identifier")


class LiqFlowMetric(BaseMetric):
    """DEX liquidity flow metrics."""

    protocol: str = Field(..., description="DEX protocol name (uniswap, sushiswap, etc.)")
    net_inflow: Decimal = Field(..., ge=0, description="Net liquidity inflow in USD")
    tvl_change: Decimal = Field(..., description="Total Value Locked change in USD")
    token_flows: Optional[Dict[str, Decimal]] = Field(
        default=None, description="Token-specific inflows/outflows"
    )
    add_liquidity_events: Optional[int] = Field(
        default=None, ge=0, description="Number of add liquidity events"
    )
    remove_liquidity_events: Optional[int] = Field(
        default=None, ge=0, description="Number of remove liquidity events"
    )


class StableflowMetric(BaseMetric):
    """Stablecoin mint/burn flow metrics."""

    stablecoin: str = Field(..., description="Stablecoin symbol (USDT, USDC, DAI, FRAX)")
    minted: Decimal = Field(..., ge=0, description="Total minted amount")
    burned: Decimal = Field(..., ge=0, description="Total burned amount")
    net_flow: Decimal = Field(..., description="Net flow (minted - burned)")
    mint_events: Optional[int] = Field(
        default=None, ge=0, description="Number of mint events"
    )
    burn_events: Optional[int] = Field(
        default=None, ge=0, description="Number of burn events"
    )


class FundingBasisMetric(BaseMetric):
    """Perpetual futures funding rate metrics."""

    exchange: str = Field(..., description="Exchange name (binance, bybit, etc.)")
    symbol: str = Field(..., description="Trading pair symbol (ETH, BTC)")
    funding_rate: Decimal = Field(..., description="Current funding rate as percentage")
    premium_index: Decimal = Field(..., description="Premium index value")
    mark_price: Optional[Decimal] = Field(
        default=None, description="Mark price of the perpetual contract"
    )
    index_price: Optional[Decimal] = Field(
        default=None, description="Index price of the spot asset"
    )


class ChainCongestionMetric(BaseMetric):
    """Blockchain network congestion metrics."""

    avg_gas_price: Decimal = Field(..., ge=0, description="Average gas price in gwei")
    block_fullness: Decimal = Field(..., ge=0, le=1, description="Block utilization ratio (0-1)")
    pending_txs: int = Field(..., ge=0, description="Number of pending transactions")
    base_fee: Optional[Decimal] = Field(
        default=None, ge=0, description="Base fee for EIP-1559 chains"
    )
    priority_fee: Optional[Decimal] = Field(
        default=None, ge=0, description="Priority fee for EIP-1559 chains"
    )
    block_time: Optional[Decimal] = Field(
        default=None, ge=0, description="Average block time in seconds"
    )


class StakingYieldMetric(BaseMetric):
    """Proof-of-stake staking yield metrics."""

    staked_amount: Decimal = Field(..., ge=0, description="Total staked amount")
    apr: Decimal = Field(..., ge=0, description="Annual percentage rate")
    validator_count: int = Field(..., ge=0, description="Number of active validators")
    total_validators: Optional[int] = Field(
        default=None, ge=0, description="Total number of validators"
    )
    slashing_events: Optional[int] = Field(
        default=None, ge=0, description="Number of slashing events in period"
    )
    rewards_distributed: Optional[Decimal] = Field(
        default=None, ge=0, description="Total rewards distributed in period"
    )


class MEVPressureMetric(BaseMetric):
    """Maximal Extractable Value pressure metrics."""

    mev_extracted: Decimal = Field(..., ge=0, description="Total MEV extracted in period")
    mev_blocks_pct: Decimal = Field(
        ..., ge=0, le=1, description="Percentage of blocks with MEV extraction"
    )
    top_searcher_share: Decimal = Field(
        ..., ge=0, le=1, description="Market share of top MEV searcher"
    )
    sandwich_attacks: Optional[int] = Field(
        default=None, ge=0, description="Number of sandwich attacks"
    )
    frontrunning_events: Optional[int] = Field(
        default=None, ge=0, description="Number of frontrunning events"
    )
    arbitrage_opportunities: Optional[int] = Field(
        default=None, ge=0, description="Number of arbitrage opportunities exploited"
    )


class CEXDEXFlowMetric(BaseMetric):
    """Cross-exchange/DEX bridge flow metrics."""

    bridge: str = Field(..., description="Bridge protocol name")
    direction: str = Field(..., description="Flow direction (inflow/outflow)")
    volume: Decimal = Field(..., ge=0, description="Bridge volume in USD")
    unique_users: int = Field(..., ge=0, description="Number of unique users")
    transaction_count: Optional[int] = Field(
        default=None, ge=0, description="Number of bridge transactions"
    )
    avg_transaction_size: Optional[Decimal] = Field(
        default=None, ge=0, description="Average transaction size in USD"
    )


# API Response schemas
class MetricMetadata(BaseModel):
    """Metadata for API responses."""

    timestamp: datetime = Field(..., description="Response timestamp")
    source: str = Field(..., description="Data source identifier")
    cache_hit: bool = Field(..., description="Whether data came from cache")
    freshness_seconds: Optional[int] = Field(
        default=None, description="How fresh the data is in seconds"
    )


class APIResponse(BaseModel):
    """Standardized API response format."""

    success: bool = Field(..., description="Whether the request was successful")
    data: Any = Field(..., description="Response data")
    metadata: MetricMetadata = Field(..., description="Response metadata")
    error: Optional[str] = Field(default=None, description="Error message if any")


class HealthStatus(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status (healthy/unhealthy)")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class DataFreshnessStatus(BaseModel):
    """Data freshness status for all metrics."""

    metric: str = Field(..., description="Metric name")
    last_updated: Optional[datetime] = Field(
        default=None, description="Last successful data collection"
    )
    is_fresh: bool = Field(..., description="Whether data is considered fresh")
    age_seconds: Optional[int] = Field(
        default=None, description="Age of last data in seconds"
    )


# Request schemas
class MetricQueryParams(BaseModel):
    """Common query parameters for metric endpoints."""

    chain: Optional[str] = Field(default=None, description="Filter by blockchain")
    from_timestamp: Optional[datetime] = Field(
        default=None, description="Start timestamp for historical data"
    )
    to_timestamp: Optional[datetime] = Field(
        default=None, description="End timestamp for historical data"
    )
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of records")


class DashboardSummary(BaseModel):
    """Dashboard summary with latest data for all metrics."""

    liq_flow: Optional[LiqFlowMetric] = Field(default=None, description="Latest liquidity flow data")
    stableflow: Optional[StableflowMetric] = Field(default=None, description="Latest stablecoin flow data")
    funding_basis: Optional[FundingBasisMetric] = Field(default=None, description="Latest funding basis data")
    chain_congestion: Optional[ChainCongestionMetric] = Field(default=None, description="Latest chain congestion data")
    staking_yield: Optional[StakingYieldMetric] = Field(default=None, description="Latest staking yield data")
    mev_pressure: Optional[MEVPressureMetric] = Field(default=None, description="Latest MEV pressure data")
    cex_dex_flow: Optional[CEXDEXFlowMetric] = Field(default=None, description="Latest CEX/DEX flow data")
    metadata: MetricMetadata = Field(..., description="Response metadata")
