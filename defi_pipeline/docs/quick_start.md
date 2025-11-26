# Quick Start Guide for Backend Engineers

## Getting Started

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd defi-data-pipeline
   pip install -r requirements.txt
   ```

2. **Read the Documentation**
   - `docs/developer_guide.md` - Complete architecture overview
   - `docs/metrics_specification.md` - Detailed metric requirements
   - `docs/implementation_example.py` - Working code example

3. **Choose Your Metric**
   Start with one metric from the 7 core metrics. We recommend starting with **Chain Congestion** as it has straightforward data sources.

## Implementation Steps

### Step 1: Create Provider (if needed)

```python
# Create app/providers/etherscan.py
from app.providers.base import RESTProvider, ProviderConfig, ProviderType

class EtherscanProvider(RESTProvider):
    def __init__(self, api_key: str):
        config = ProviderConfig(
            name="etherscan",
            provider_type=ProviderType.BLOCKCHAIN_EXPLORER,
            base_url="https://api.etherscan.io/api",
            api_key=api_key,
            rate_limit=5
        )
        super().__init__(config)

    async def get_gas_oracle(self):
        params = {"module": "gastracker", "action": "gasoracle"}
        return await self.fetch_data("", params=params)
```

### Step 2: Create Collector

```python
# Create app/collectors/chain_congestion.py
from app.collectors.base import BaseCollector, MetricType, CollectionConfig
from app.models.schemas import ChainCongestionMetric

class ChainCongestionCollector(BaseCollector):
    def __init__(self, config: CollectionConfig, provider):
        super().__init__(MetricType.CHAIN_CONGESTION, config)
        self.provider = provider

    async def collect(self):
        # Implement data collection logic
        gas_data = await self.provider.get_gas_oracle()
        # Transform and validate data
        return CollectionResult(success=True, data=[metric])

    async def validate_data(self, data):
        # Implement validation logic
        return True
```

### Step 3: Register Components

```python
# In app/collectors/registry.py
from app.collectors.chain_congestion import ChainCongestionCollector
from app.providers.etherscan import EtherscanProvider

def create_default_collectors():
    provider = EtherscanProvider(api_key=os.getenv("ETHERSCAN_API_KEY"))

    register_collector(
        ChainCongestionCollector,
        MetricType.CHAIN_CONGESTION,
        CollectionPriority.HIGH,
        1,  # minutes
        provider=provider
    )
```

### Step 4: Test Your Implementation

```python
# Create tests/test_chain_congestion.py
import pytest
from app.collectors.chain_congestion import ChainCongestionCollector

@pytest.mark.asyncio
async def test_chain_congestion_collection():
    # Mock provider and test collection
    pass
```

### Step 5: Run and Verify

```bash
# Run tests
pytest tests/test_chain_congestion.py -v

# Start API server
uvicorn app.main:app --reload

# Test endpoint
curl http://localhost:8000/api/v1/chain-congestion
```

## Common Patterns

### Error Handling
```python
async def collect(self):
    try:
        response = await self.provider.fetch_data(endpoint)
        if not response.success:
            return CollectionResult(success=False, error=response.error)
        # Process data
    except Exception as e:
        return CollectionResult(success=False, error=str(e))
```

### Data Validation
```python
async def validate_data(self, data):
    try:
        for item in data:
            MetricSchema(**item)  # Pydantic validation
        return True
    except ValidationError:
        return False
```

### Database Storage
```python
def store_metrics(metrics):
    with get_db_session() as db:
        for metric in metrics:
            db_metric = SchemaToDBModel(metric)
            db.add(db_metric)
        db.commit()
```

## Directory Structure for New Metrics

```
app/
├── collectors/
│   ├── chain_congestion.py      # Your new collector
│   └── registry.py              # Register here
├── providers/
│   └── etherscan.py            # Your new provider (if needed)
├── models/
│   ├── schemas.py               # Add metric schema
│   └── database.py              # Add database model
├── api/v1/
│   ├── endpoints/
│   │   └── chain_congestion.py  # API endpoints
│   └── router.py                # Register routes here
└── tasks/
    └── scheduled.py             # Add Celery task
```

## Environment Setup

Create `.env` file:
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/defi_pipeline
REDIS_URL=redis://localhost:6379/0
ETHERSCAN_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

## Docker Development

```bash
# Start services
docker-compose up -d postgres redis

# Run your code
docker-compose up api
```

## Next Steps

1. **Implement one metric completely** (data collection → storage → API)
2. **Add comprehensive tests** (unit + integration)
3. **Update documentation** with your implementation details
4. **Move to next metric** following the same pattern
5. **Consider performance optimizations** (caching, async operations)

## Need Help?

- Check `docs/implementation_example.py` for working code
- Review existing metric implementations for patterns
- Run tests to verify your implementation
- Check API documentation at `/docs` when server is running

## Success Criteria

✅ **Collector successfully collects data**
✅ **Data validates against schema**
✅ **Data stores in database correctly**
✅ **API endpoint returns data**
✅ **Tests pass**
✅ **Documentation updated**

Once these are met, your metric implementation is complete!
