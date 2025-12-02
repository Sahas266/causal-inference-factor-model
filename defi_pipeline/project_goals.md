# DeFi Data Pipeline

A scalable DeFi data pipeline built with FastAPI that collects, processes, and serves real-time metrics from multiple blockchain data providers.

## Overview

This system tracks seven key on-chain indicators to provide comprehensive DeFi market intelligence:

- **Liq Flow**: DEX liquidity dynamics (token flows, TVL changes, add/remove events)
- **Stableflow**: Stablecoin supply mechanics (USDT/USDC/DAI/FRAX mint/burn rates)
- **Funding Basis**: Perpetual futures arbitrage (funding rates for ETH/BTC)
- **Chain Congestion**: Network health (gas prices, block utilization)
- **Staking Yield**: Proof-of-stake economics (ETH staking rewards, validator metrics)
- **MEV Pressure**: Maximal Extractable Value dynamics (MEV-Boost blocks, extracted value)
- **CEX/DEX Flow**: Cross-chain capital movement (bridge volumes by direction)

## Architecture

The pipeline follows a modular, scalable architecture with clear separation of concerns:

```
defi-data-pipeline/
├── app/
│   ├── api/v1/           # REST API endpoints
│   ├── collectors/       # Data collection logic
│   ├── providers/        # External API integrations
│   ├── models/           # Database schemas & Pydantic models
│   ├── services/         # Data processing & aggregation
│   ├── tasks/           # Scheduled data collection
│   └── core/            # Core utilities (DB, cache, config)
├── tests/               # Comprehensive test suite
└── docker/              # Containerization
```

### Key Components

- **FastAPI**: Async web framework for high-performance APIs
- **PostgreSQL + TimescaleDB**: Time-series database for metrics storage
- **Redis(Optional)**: Caching and task queue backend
- **Celery**: Distributed task scheduling
- **Pydantic**: Data validation and serialization

## Features

### Data Collection
- **Provider Agnostic**: Flexible architecture supports multiple data sources
- **Rate Limiting**: Built-in rate limiting with exponential backoff
- **Error Handling**: Robust error handling with retry mechanisms
- **Health Monitoring**: Provider and collector health checks

### API Endpoints
- **RESTful Design**: Clean, versioned API endpoints
- **Real-time Metrics**: Latest data for all 7 metric types
- **Historical Data**: Time-series queries with filtering
- **Dashboard Aggregation**: Combined metrics for dashboard views

### Performance & Scalability
- **Caching**: Redis(Optional)-based caching for improved response times
- **Async Operations**: Fully asynchronous data processing
- **TimescaleDB**: Optimized time-series storage and querying
- **Horizontal Scaling**: Stateless design supports scaling

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL with TimescaleDB extension
- Redis(Optional)
- Poetry (optional, for dependency management)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd defi-data-pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or with Poetry
   poetry install
   ```

3. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize database**
   ```bash
   python scripts/init_db.py
   ```

5. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Latest Metrics
```bash
# Chain congestion
curl "http://localhost:8000/api/v1/chain-congestion?chain=ethereum"

# Liquidity flow
curl "http://localhost:8000/api/v1/liq-flow?chain=ethereum"

# Dashboard summary
curl "http://localhost:8000/api/v1/dashboard"
```

### Historical Data
```bash
# Get funding basis history
curl "http://localhost:8000/api/v1/historical?metrics=funding_basis&from_timestamp=2025-01-01T00:00:00Z"
```

## Configuration

Key environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/defi_pipeline
REDIS_URL=redis://localhost:6379/0

# Application
LOG_LEVEL=INFO
CACHE_TTL=300
MAX_WORKERS=4
RATE_LIMIT_PER_MINUTE=60

# Provider settings (configure as needed)
PROVIDER_TIMEOUT=30
PROVIDER_MAX_RETRIES=3
```

## Development

### Running Tests
```bash
pytest
```

### Code Quality
```bash
# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8 .
```

### Database Migrations
```bash
alembic revision --autogenerate -m "migration message"
alembic upgrade head
```

## Deployment

### Docker
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Production
- Use gunicorn/uvicorn with multiple workers
- Set up proper logging and monitoring
- Configure reverse proxy (nginx)
- Set up database backups
- Enable SSL/TLS

## Monitoring

- **Health Endpoints**: `/health`, `/health/detailed`
- **Metrics**: `/metrics` (Prometheus format)
- **Status**: `/status` (data freshness)
- **Structured Logging**: JSON logs for easy parsing

## Success Criteria

- [ ] All 7 metrics collecting data successfully
- [ ] API serving requests with <200ms latency (cached)
- [ ] Zero data loss over 48-hour period
- [ ] Provider failover working (tested)
- [ ] Documentation complete and accurate
- [ ] Passing CI/CD pipeline (tests + linting)

## For Backend Engineers

🚀 **Ready to implement metrics? Start here:**

1. **Read the Developer Guide**: `docs/developer_guide.md`
2. **Check Metric Specifications**: `docs/metrics_specification.md`
3. **Follow the Quick Start**: `docs/quick_start.md`
4. **Study the Example**: `docs/implementation_example.py`

### Implementation Checklist

For each metric you implement:

- [ ] Create provider class (if needed)
- [ ] Implement collector with validation
- [ ] Add database schema and models
- [ ] Create API endpoints
- [ ] Add Celery scheduled tasks
- [ ] Write comprehensive tests
- [ ] Update documentation
- [ ] Test end-to-end data flow

### Available Metrics to Implement

Choose from these 7 core metrics:

1. **Chain Congestion** ⭐ *Recommended for beginners*
2. **Funding Basis** ⭐ *Good second metric*
3. **Liquidity Flow**
4. **Stablecoin Flow**
5. **Staking Yield**
6. **MEV Pressure**
7. **CEX/DEX Flow**

## Contributing

1. Follow the established code patterns
2. Add tests for new functionality
3. Update documentation
4. Ensure type hints are complete
5. Run the full test suite before submitting

## License

MIT License - see LICENSE file for details.
