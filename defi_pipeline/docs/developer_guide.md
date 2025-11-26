# DeFi Data Pipeline - Developer Guide

## Overview

This document provides detailed guidance for backend engineers on how to implement and extend the DeFi Data Pipeline for the 7 core metrics. The architecture is designed to be modular, extensible, and provider-agnostic.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Providers     │    │   Collectors    │    │   Services      │
│                 │    │                 │    │                 │
│ - External APIs │───▶│ - Data Collection│───▶│ - Processing   │
│ - Rate Limiting │    │ - Validation    │    │ - Aggregation  │
│ - Error Handling│    │ - Transformation│    │ - Caching      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │    │   API Layer     │    │   Tasks         │
│                 │    │                 │    │                 │
│ - TimescaleDB   │◀───│ - REST Endpoints│    │ - Scheduled     │
│ - Hypertables   │    │ - Response Format│    │ - Collection   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Metric Models (`app/models/`)

#### Pydantic Schemas (`schemas.py`)
Each metric has a dedicated Pydantic model for validation and serialization:

```python
class LiqFlowMetric(BaseMetric):
    """DEX liquidity flow metrics."""
    protocol: str
    net_inflow: Decimal
    tvl_change: Decimal
    # ... additional fields

class StableflowMetric(BaseMetric):
    """Stablecoin supply mechanics."""
    stablecoin: str
    minted: Decimal
    burned: Decimal
    net_flow: Decimal
    # ... additional fields
```

#### Database Models (`database.py`)
SQLAlchemy models with TimescaleDB optimizations:

```python
class LiqFlow(Base):
    __tablename__ = "metrics_liq_flow"
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    chain = Column(String(50), primary_key=True)
    protocol = Column(String(100))
    net_inflow = Column(Numeric(36, 18))
    # ... columns with proper indexing
```

### 2. Collector Framework (`app/collectors/`)

#### Base Collector Class (`base.py`)
All metric collectors inherit from `BaseCollector`:

```python
class BaseCollector(ABC):
    def __init__(self, metric_type: MetricType, config: CollectionConfig):
        self.metric_type = metric_type
        self.config = config

    @abstractmethod
    async def collect(self) -> CollectionResult:
        """Collect data for this metric."""
        pass

    @abstractmethod
    async def validate_data(self, data: Any) -> bool:
        """Validate collected data."""
        pass
```

#### Implementing a Metric Collector
Create a new collector for each metric:

```python
from app.collectors.base import BaseCollector, MetricType, CollectionConfig
from app.models.schemas import LiqFlowMetric

class LiqFlowCollector(BaseCollector):
    def __init__(self, config: CollectionConfig, provider):
        super().__init__(MetricType.LIQ_FLOW, config)
        self.provider = provider

    async def collect(self) -> CollectionResult:
        """Collect liquidity flow data."""
        try:
            # Use provider to fetch data
            raw_data = await self.provider.fetch_data("/liquidity")

            # Transform to metric format
            metrics = []
            for item in raw_data:
                metric = LiqFlowMetric(
                    timestamp=datetime.utcnow(),
                    chain=item["chain"],
                    protocol=item["protocol"],
                    net_inflow=Decimal(str(item["net_inflow"])),
                    tvl_change=Decimal(str(item["tvl_change"]))
                )
                metrics.append(metric)

            return CollectionResult(
                success=True,
                data=metrics,
                source=self.provider.config.name
            )

        except Exception as e:
            return CollectionResult(
                success=False,
                error=str(e)
            )

    async def validate_data(self, data: Any) -> bool:
        """Validate liquidity flow data."""
        if not isinstance(data, list):
            return False

        for item in data:
            if not isinstance(item, LiqFlowMetric):
                return False

        return True
```

### 3. Provider Framework (`app/providers/`)

#### Base Provider Classes (`base.py`)
Abstract base classes for different provider types:

```python
class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    @abstractmethod
    async def fetch_data(self, endpoint: str, params=None) -> ProviderResponse:
        """Fetch data from endpoint."""
        pass

class RESTProvider(BaseProvider):
    """REST API provider implementation."""

    async def fetch_data(self, endpoint: str, params=None) -> ProviderResponse:
        return await self.make_request("GET", endpoint, params=params)
```

#### Implementing a Provider
Create providers for specific data sources:

```python
from app.providers.base import RESTProvider, ProviderConfig, ProviderType

class TheGraphProvider(RESTProvider):
    def __init__(self, api_key: str = None):
        config = ProviderConfig(
            name="the_graph",
            provider_type=ProviderType.DEX_PROTOCOL,
            base_url="https://api.thegraph.com/subgraphs/name/",
            api_key=api_key,
            rate_limit=60,  # requests per minute
            timeout=30
        )
        super().__init__(config)

    async def fetch_liquidity_data(self, protocol: str) -> ProviderResponse:
        """Fetch liquidity data for a specific protocol."""
        query = """
        {
          liquidityPools(first: 100) {
            id
            totalValueLockedUSD
            volumeUSD
          }
        }
        """

        return await self.make_request(
            "POST",
            f"/{protocol}",
            json={"query": query}
        )
```

### 4. Data Processing (`app/services/`)

#### Data Processor (`data_processor.py`)
Handles data transformation and API responses:

```python
class DataProcessor:
    def get_latest_metric(self, metric_class, filters=None) -> Optional[Dict]:
        """Get latest data for a metric."""
        with get_db_session() as db:
            query = db.query(metric_class).order_by(metric_class.timestamp.desc())
            result = query.first()
            return self._row_to_dict(result) if result else None

    def get_metric_history(self, metric_class, from_ts=None, to_ts=None, limit=100):
        """Get historical data for a metric."""
        with get_db_session() as db:
            query = db.query(metric_class).order_by(metric_class.timestamp.desc())
            if from_ts:
                query = query.filter(metric_class.timestamp >= from_ts)
            if to_ts:
                query = query.filter(metric_class.timestamp <= to_ts)

            results = query.limit(limit).all()
            return [self._row_to_dict(row) for row in results]
```

#### Aggregator Service (`aggregator.py`)
Provides cross-metric analytics:

```python
class MetricAggregator:
    async def get_market_correlations(self, from_ts, to_ts, chains=None):
        """Calculate correlations between metrics."""
        # Implementation for correlation analysis

    async def get_chain_health_score(self, chain: str, hours: int = 24):
        """Calculate health score for a blockchain."""
        # Implementation for health scoring
```

### 5. API Layer (`app/api/v1/`)

#### Router Structure (`router.py`)
API endpoints for each metric:

```python
@api_router.get("/liq-flow")
async def get_liq_flow(chain: str = None, from_ts: str = None, to_ts: str = None):
    """Get liquidity flow metrics."""
    # Implementation using data_processor

@api_router.get("/dashboard")
async def get_dashboard():
    """Get dashboard summary."""
    dashboard_data = data_processor.get_dashboard_summary()
    return create_api_response(data=dashboard_data)
```

## Implementing the 7 Core Metrics

### 1. Liquidity Flow (LiqFlow)

**Data Sources**: DEX protocols (Uniswap, SushiSwap, etc.)
**Collection Frequency**: 15 minutes
**Key Metrics**: Net inflows, TVL changes, protocol volumes

```python
class LiqFlowCollector(BaseCollector):
    async def collect(self):
        # Fetch from DEX providers (The Graph, etc.)
        # Calculate net inflows and TVL changes
        # Return LiqFlowMetric objects
        pass
```

**Database Schema**:
```sql
CREATE TABLE metrics_liq_flow (
  timestamp TIMESTAMPTZ NOT NULL,
  chain VARCHAR(50) NOT NULL,
  protocol VARCHAR(100) NOT NULL,
  net_inflow NUMERIC(36,18) NOT NULL,
  tvl_change NUMERIC(36,18) NOT NULL,
  -- Additional fields...
  PRIMARY KEY (timestamp, chain, protocol)
);
```

### 2. Stablecoin Flow (Stableflow)

**Data Sources**: Stablecoin contracts, mint/burn events
**Collection Frequency**: 15 minutes
**Key Metrics**: Minted/burned amounts, net flow by stablecoin

```python
class StableflowCollector(BaseCollector):
    async def collect(self):
        # Monitor USDT, USDC, DAI, FRAX contracts
        # Track mint/burn events
        # Calculate net flows
        pass
```

### 3. Funding Basis (FundingBasis)

**Data Sources**: Perpetuals exchanges (Binance, Bybit, etc.)
**Collection Frequency**: 5 minutes
**Key Metrics**: Funding rates, premium index, mark vs index prices

```python
class FundingBasisCollector(BaseCollector):
    async def collect(self):
        # Fetch funding rates from exchanges
        # Calculate premium index
        # Monitor ETH and BTC perpetuals
        pass
```

### 4. Chain Congestion (ChainCongestion)

**Data Sources**: Blockchain explorers (Etherscan, etc.)
**Collection Frequency**: 1 minute
**Key Metrics**: Gas prices, block fullness, pending transactions

```python
class ChainCongestionCollector(BaseCollector):
    async def collect(self):
        # Get current gas prices
        # Monitor block utilization
        # Track pending transaction counts
        pass
```

### 5. Staking Yield (StakingYield)

**Data Sources**: Beacon chain data, validator APIs
**Collection Frequency**: 1 hour
**Key Metrics**: APR, validator counts, rewards distribution

```python
class StakingYieldCollector(BaseCollector):
    async def collect(self):
        # Monitor Ethereum staking
        # Calculate APR from rewards
        # Track validator participation
        pass
```

### 6. MEV Pressure (MEVPressure)

**Data Sources**: MEV relays, block data
**Collection Frequency**: 30 minutes
**Key Metrics**: Extracted value, MEV blocks percentage, searcher activity

```python
class MEVPressureCollector(BaseCollector):
    async def collect(self):
        # Monitor MEV-Boost blocks
        # Track extracted value
        # Analyze searcher behavior
        pass
```

### 7. CEX/DEX Flow (CEXDEXFlow)

**Data Sources**: Bridge protocols, cross-chain data
**Collection Frequency**: 1 hour
**Key Metrics**: Bridge volumes, user activity, direction flows

```python
class CEXDEXFlowCollector(BaseCollector):
    async def collect(self):
        # Monitor bridge contracts
        # Track cross-chain transfers
        # Calculate flow directions
        pass
```

## Data Collection Workflow

### 1. Provider Setup
```python
# Register providers
the_graph = TheGraphProvider(api_key="...")
etherscan = EtherscanProvider(api_key="...")
provider_registry.register(the_graph)
provider_registry.register(etherscan)
```

### 2. Collector Registration
```python
# Create and register collectors
liq_collector = LiqFlowCollector(
    config=CollectionConfig(
        metric_type=MetricType.LIQ_FLOW,
        priority=CollectionPriority.MEDIUM,
        interval_minutes=15
    ),
    provider=the_graph
)
collector_registry.register(liq_collector)
```

### 3. Task Scheduling
```python
# Celery tasks automatically call collectors
@celery_app.task
def collect_liq_flow():
    result = collector_registry.get_collector(MetricType.LIQ_FLOW).collect_with_retry()
    # Store results in database
    store_metrics(result.data)
```

### 4. Data Processing
```python
# API endpoints use data processor
@app.get("/api/v1/liq-flow")
def get_liq_flow(chain: str = None):
    data = data_processor.get_metric_history(LiqFlow, filters={"chain": chain})
    return create_api_response(data)
```

## Error Handling and Validation

### Provider Error Handling
```python
async def collect(self) -> CollectionResult:
    try:
        response = await self.provider.fetch_data(endpoint)
        if not response.success:
            return CollectionResult(
                success=False,
                error=f"Provider error: {response.error}"
            )
        # Process successful response
    except Exception as e:
        return CollectionResult(success=False, error=str(e))
```

### Data Validation
```python
async def validate_data(self, data: Any) -> bool:
    """Validate collected data before storage."""
    try:
        for item in data:
            # Pydantic validation
            LiqFlowMetric(**item)
        return True
    except ValidationError:
        return False
```

## Database Operations

### Storing Metrics
```python
def store_metrics(metrics: List[BaseModel]):
    """Store collected metrics in database."""
    with get_db_session() as db:
        for metric in metrics:
            db_metric = self._schema_to_db_model(metric)
            db.add(db_metric)
        db.commit()
```

### Querying Data
```python
def get_latest_liq_flow(chain: str = None) -> List[LiqFlowMetric]:
    """Get latest liquidity flow data."""
    with get_db_session() as db:
        query = db.query(LiqFlow).order_by(LiqFlow.timestamp.desc())

        if chain:
            query = query.filter(LiqFlow.chain == chain)

        results = query.limit(100).all()
        return [self._db_to_schema(result) for result in results]
```

## Testing Strategy

### Unit Tests
```python
def test_liq_flow_collector():
    # Mock provider response
    mock_provider = Mock()
    collector = LiqFlowCollector(config, mock_provider)

    # Test collection
    result = await collector.collect()
    assert result.success
    assert len(result.data) > 0

def test_liq_flow_validation():
    collector = LiqFlowCollector(config, provider)

    # Test valid data
    valid_data = [LiqFlowMetric(...)]
    assert await collector.validate_data(valid_data)

    # Test invalid data
    invalid_data = [{"invalid": "data"}]
    assert not await collector.validate_data(invalid_data)
```

### Integration Tests
```python
def test_full_data_pipeline():
    # Test end-to-end data flow
    # Provider -> Collector -> Database -> API
    pass
```

## Performance Considerations

### Caching Strategy
```python
# Cache frequently accessed data
await cache_manager.set(f"liq_flow:latest:{chain}", data, ttl=300)

# Check cache before database queries
cached = await cache_manager.get(f"liq_flow:latest:{chain}")
if cached:
    return cached
```

### Database Optimization
```python
# Use TimescaleDB hypertables for time-series data
# Create appropriate indexes
# Implement data retention policies
```

### Async Operations
```python
# Use async/await for I/O operations
async def collect_all_metrics():
    tasks = [
        collector.collect_with_retry()
        for collector in collector_registry.get_all_collectors()
    ]
    results = await asyncio.gather(*tasks)
    return results
```

## Monitoring and Observability

### Health Checks
```python
@app.get("/health/detailed")
def health_check():
    return {
        "database": check_database_health(),
        "cache": check_cache_health(),
        "collectors": collector_registry.get_health_status(),
        "providers": provider_registry.get_health_status()
    }
```

### Metrics Collection
```python
# Track collection success/failure rates
# Monitor API response times
# Alert on data freshness issues
```

## Deployment and Scaling

### Docker Configuration
```yaml
# Use provided docker-compose.yml
# Scale individual services as needed
# Configure environment variables
```

### Production Considerations
```python
# Enable production logging
# Set up proper monitoring
# Configure rate limits
# Set up database backups
# Implement graceful shutdown
```

## Getting Started

1. **Choose a Metric**: Start with one metric (e.g., Chain Congestion)
2. **Implement Provider**: Create or reuse a provider for data source
3. **Create Collector**: Implement the collector class
4. **Register Components**: Add to registry and task scheduler
5. **Test End-to-End**: Verify data flows from source to API
6. **Add Monitoring**: Implement health checks and metrics
7. **Deploy**: Use Docker setup for containerized deployment

This architecture provides a solid foundation for implementing all 7 DeFi metrics while maintaining flexibility for future extensions and modifications.
