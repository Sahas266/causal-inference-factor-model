# Multi-Provider Data Backfill System

A production-ready, provider-agnostic system for backfilling cryptocurrency and blockchain data from multiple providers (CoinMetrics, Glassnode, Messari, etc.). Built with a plugin architecture that makes adding new providers trivial.

## Features

- **Provider-Agnostic**: Add new data providers without modifying core logic
- **Plugin Architecture**: Drop-in provider implementations with auto-discovery
- **Multi-Provider Support**: Fetch from multiple providers with priority-based deduplication
- **Checkpoint Resumption**: Resume from last successful point after failures
- **Rate Limiting**: Provider-specific, thread-safe rate limiting
- **Concurrent Execution**: Thread-pool based concurrent backfills
- **Progress Tracking**: Database-backed progress tracking with error logging
- **Memory Efficient**: Generator-based streaming for large datasets
- **Type Safe**: Full type hints with Pydantic schemas

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Interface                           │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                  BackfillOrchestrator                        │
│  - Validates endpoints                                       │
│  - Coordinates providers                                     │
│  - Manages concurrent execution                              │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
┌───────────────▼──────────┐   ┌───────▼──────────────────────┐
│   ProviderRegistry       │   │   Storage Layer              │
│   - Auto-discovers       │   │   - SupabaseManager         │
│   - Registers providers  │   │   - DatabaseWriter          │
└──────────────────────────┘   │   - ProgressTracker         │
                               └─────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Provider Implementations                  │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  CoinMetrics    │  │  Glassnode      │  (Add more...)   │
│  │  - Client       │  │  - Client       │                  │
│  │  - RateLimiter  │  │  - RateLimiter  │                  │
│  │  - Transformer  │  │  - Transformer  │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone repository
cd backfill_data

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys and database credentials
```

### 2. Database Setup

Execute the database schema:

```bash
# Connect to your Supabase database and run:
psql -h your-host -U your-user -d your-db -f config/database_schema.sql
```

Or use Supabase SQL editor to run `config/database_schema.sql`.

### 3. Configure Endpoints

Edit or create endpoint configurations in `config/endpoints/`:

```json
{
  "endpoint_id": "btc_price_metrics",
  "table": "asset_metrics",
  "primary_keys": ["provider", "asset", "metric", "time"],
  "providers": [
    {
      "name": "coinmetrics",
      "enabled": true,
      "priority": 1,
      "config": {
        "endpoint_type": "timeseries/asset-metrics",
        "params": {
          "assets": "btc",
          "metrics": "PriceUSD,CapMrktCurUSD",
          "frequency": "1d"
        }
      }
    }
  ]
}
```

### 4. Run Backfill

```bash
# List available providers
python backfill.py --list-providers

# Validate configuration (no data fetch)
python backfill.py --config config/endpoints/btc_metrics.json --validate-only

# Run backfill
python backfill.py --config config/endpoints/btc_metrics.json

# Backfill all endpoints in directory
python backfill.py --config config/endpoints/

# Resume failed backfills
python backfill.py --resume-failed
```

### 5. Run Provider-Specific Backfill Scripts

Provider-specific scripts are available in `scripts/`:

```bash
# CoinMetrics
python scripts/backfill_coinmetrics.py --validate-only

# DeFi Llama
python scripts/backfill_defillama.py --validate-only

# CoinGecko (script scaffold; requires provider adapter + endpoint configs)
python scripts/backfill_coingecko.py --list-endpoints

# Dune (script scaffold; requires provider adapter + endpoint configs)
python scripts/backfill_dune.py --list-endpoints
```

## Adding a New Provider

Adding a new provider requires just 4 steps:

### 1. Create Provider Directory

```bash
mkdir -p src/providers/glassnode
touch src/providers/glassnode/{__init__.py,provider.py,client.py,rate_limiter.py,transformer.py}
```

### 2. Implement Provider Interface

```python
# src/providers/glassnode/provider.py
from src.core.interfaces import DataProviderInterface

class GlassnodeProvider(DataProviderInterface):
    @property
    def provider_name(self) -> str:
        return "glassnode"
    
    # Implement all required methods...
```

### 3. Register Provider

```python
# src/providers/glassnode/__init__.py
from src.providers.registry import ProviderRegistry
from .provider import GlassnodeProvider

ProviderRegistry.register(GlassnodeProvider)
```

### 4. Add Configuration

```json
// config/providers/glassnode.json
{
  "provider_name": "glassnode",
  "enabled": true,
  "api_config": {
    "base_url": "https://api.glassnode.com",
    "api_key_env": "GLASSNODE_API_KEY"
  },
  "rate_limits": {
    "requests_per_minute": 60
  }
}
```

That's it! The system will auto-discover and use the new provider.

## Configuration

### Provider Configuration

Located in `config/providers/{provider_name}.json`:

- `provider_name`: Unique provider identifier
- `enabled`: Enable/disable provider
- `api_config`: API credentials and base URL
  - CoinMetrics note: `community_base_url` can be set to `https://community-api.coinmetrics.io/v4` for community endpoints (no API key required)
- `rate_limits`: Rate limiting parameters
- `retry_config`: Retry behavior configuration

### Endpoint Configuration

Located in `config/endpoints/{endpoint_name}.json`:

- `endpoint_id`: Unique endpoint identifier
- `table`: Target database table
- `primary_keys`: Columns forming primary key
- `providers`: Array of provider configurations
- `fallback_strategy`: Multi-provider strategy

## CLI Reference

```bash
# Basic usage
python backfill.py --config <path> [options]

# Options
--config PATH              Path to endpoint config file or directory
--config-dir PATH          Path to config directory (default: config/)
--max-workers N            Maximum concurrent workers (default: 10)
--validate-only            Only validate, don't backfill
--list-providers           List available providers
--resume-failed            Resume failed backfills
--stats                    Show provider statistics
--provider NAME            Filter by specific provider
--log-level LEVEL          Logging level (DEBUG, INFO, WARNING, ERROR)
--log-file PATH            Log to file
--json-logs                Use JSON formatted logs
```

## Data Schemas

The system supports the following schemas (implemented in `src/schemas`):
- **Asset Metrics** (`asset_metrics`)
- **Market Trades** (`market_trades`)
- **Exchange Metrics** (`exchange_metrics`)
- **Market Orderbooks** (`market_orderbooks`)
- **Pair Candles** (`pair_candle`)
- **Market Candles** (`market_candle`)
- **Derivatives Data**: Funding Rates, Open Interest, Liquidations (`derivatives_data`)
- **Options Data**: Implied Volatility, Greeks (`options_data`)

### Asset Metrics

```python
{
    "provider": "coinmetrics",
    "provider_priority": 1,
    "asset": "btc",
    "metric": "PriceUSD",
    "time": "2024-01-01T00:00:00Z",
    "value": "42000.50",
    "frequency": "1d",
    "metadata": {}
}
```

### Market Trades

```python
{
    "provider": "coinmetrics",
    "provider_priority": 1,
    "market": "coinbase-btc-usd-spot",
    "time": "2024-01-01T00:00:00Z",
    "trade_id": "123456",
    "price": "42000.50",
    "amount": "0.5",
    "side": "buy",
    "metadata": {}
}
```

### Exchange Metrics

```python
{
    "provider": "coinmetrics",
    "provider_priority": 1,
    "exchange": "binance",
    "metric": "flow_in_btc",
    "time": "2024-01-01T00:00:00Z",
    "value": "1000.5",
    "frequency": "1d",
    "metadata": {}
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/providers/test_coinmetrics.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Monitoring

### Database Views

The system provides several monitoring views:

```sql
-- Provider health
SELECT * FROM provider_health;

-- Data coverage by provider
SELECT * FROM data_coverage_by_provider
WHERE asset = 'btc' AND metric = 'PriceUSD';

-- Recent activity
SELECT * FROM recent_backfill_activity;
```

### Progress Tracking

```sql
-- Check progress of all backfills
SELECT endpoint_id, provider, status, total_records_fetched, last_successful_time
FROM backfill_progress
ORDER BY updated_at DESC;

-- Find failed backfills
SELECT * FROM backfill_progress WHERE status = 'failed';
```

## Production Considerations

### Environment Variables

- Always use environment variables for API keys
- Never commit `.env` file to version control
- Use different keys for dev/staging/production

### Rate Limiting

- Adjust `safety_margin` based on your API tier
- Monitor rate limiter statistics
- Consider using multiple API keys for higher throughput

### Database

- Ensure proper indexes are created (see schema)
- Monitor table sizes and partition if needed
- Regular vacuum and analyze operations
- Consider TimescaleDB for time-series optimization

### Error Handling

- Monitor `backfill_progress` table for failures
- Set up alerts for failed backfills
- Use `--resume-failed` for automatic recovery

### Performance Tuning

- Adjust `max_workers` based on API limits and database capacity
- Tune `batch_size` for optimal write performance
- Use connection pooling for database

## Troubleshooting

### Common Issues

**Issue**: Rate limit errors

```bash
# Solution: Reduce max_workers or adjust safety_margin
python backfill.py --config config/endpoints/ --max-workers 5
```

**Issue**: Database connection errors

```bash
# Solution: Check Supabase credentials and network
# Verify .env variables
cat .env | grep SUPABASE
```

**Issue**: Import errors

```bash
# Solution: Ensure all dependencies are installed
pip install -r requirements.txt
```

### Debug Mode

```bash
# Enable debug logging
python backfill.py --config config/endpoints/ --log-level DEBUG --log-file debug.log
```

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Check documentation in `docs/` directory
- Review project_goals.md for architecture details
- Open an issue on GitHub

## ETH Data Catalog

- ETH provider-by-provider pull checklist:
  - `ETH_DATA_POINTS_CATALOG.md`

