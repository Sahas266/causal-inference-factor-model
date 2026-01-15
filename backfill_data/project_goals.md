# Multi-Provider Data Backfill System - Architecture Design

## Overview
Design a **provider-agnostic**, extensible system that supports N data providers (CoinMetrics, Glassnode, Messari, etc.) with a plugin-based architecture. Add new providers without modifying core logic.

## Core Design Philosophy

### Provider Abstraction Layer
Each data provider implements a standardized interface. Core system doesn't know about provider-specific details.

```python
# The system works with ANY provider that implements this interface
class DataProviderInterface(ABC):
    @abstractmethod
    def validate_endpoint(self, endpoint_config) -> ValidationResult:
        """Validate endpoint exists and has data"""
        pass
    
    @abstractmethod
    def fetch_data(self, endpoint_config, start_time, end_time) -> Generator:
        """Fetch data in chunks (generator for memory efficiency)"""
        pass
    
    @abstractmethod
    def get_rate_limiter(self) -> RateLimiter:
        """Return provider-specific rate limiter"""
        pass
    
    @abstractmethod
    def transform_to_standard_format(self, raw_data, endpoint_config) -> List[Dict]:
        """Transform provider format to standard schema"""
        pass
```

## Technical Stack
- **HTTP Client**: `requests` with `requests.Session()` 
- **Concurrency**: `concurrent.futures.ThreadPoolExecutor`
- **Database**: Single shared Supabase client (Singleton)
- **Rate Limiting**: Provider-specific implementations
- **Retry Logic**: `tenacity` library
- **Plugin System**: Python packages with entry points or directory-based discovery

## Architecture

### 1. Directory Structure (Plugin-Based)

```
backfill_system/
├── config/
│   ├── providers/
│   │   ├── coinmetrics.json      # CoinMetrics-specific config
│   │   ├── glassnode.json        # Glassnode-specific config
│   │   ├── messari.json          # Future provider
│   │   └── dune.json             # Future provider
│   ├── endpoints/
│   │   ├── btc_metrics.json      # Multi-provider endpoint definition
│   │   └── eth_markets.json
│   └── database_schema.sql
├── src/
│   ├── core/                      # Provider-agnostic core
│   │   ├── __init__.py
│   │   ├── interfaces.py          # Abstract base classes
│   │   ├── orchestrator.py        # Main coordinator
│   │   ├── storage/
│   │   │   ├── supabase_manager.py
│   │   │   ├── database_writer.py
│   │   │   └── progress_tracker.py
│   │   └── utils/
│   │       ├── logger.py
│   │       └── config_loader.py
│   ├── providers/                 # Provider implementations
│   │   ├── __init__.py
│   │   ├── base_provider.py       # Common provider functionality
│   │   ├── coinmetrics/
│   │   │   ├── __init__.py
│   │   │   ├── provider.py        # Implements DataProviderInterface
│   │   │   ├── client.py
│   │   │   ├── rate_limiter.py
│   │   │   └── transformer.py
│   │   ├── glassnode/
│   │   │   ├── __init__.py
│   │   │   ├── provider.py
│   │   │   ├── client.py
│   │   │   ├── rate_limiter.py
│   │   │   └── transformer.py
│   │   └── registry.py            # Auto-discover providers
│   └── schemas/                   # Standard data schemas
│       ├── asset_metrics.py
│       ├── market_data.py
│       └── exchange_metrics.py
├── tests/
│   ├── core/
│   ├── providers/
│   │   ├── test_coinmetrics.py
│   │   └── test_glassnode.py
│   └── integration/
├── backfill.py                    # CLI entry point
├── requirements.txt
└── README.md
```

### 2. Configuration Schema (Multi-Provider)

```json
// config/endpoints/btc_price_metrics.json
{
  "endpoint_id": "btc_price_metrics",
  "description": "Bitcoin price and market cap metrics",
  "table": "asset_metrics",
  "primary_keys": ["provider", "asset", "metric", "time"],
  "date_range": {
    "start": "2015-01-01",
    "end": "2024-12-31"
  },
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
          "frequency": "1d",
          "page_size": 10000
        }
      }
    },
    {
      "name": "glassnode",
      "enabled": true,
      "priority": 2,
      "config": {
        "endpoint": "/v1/metrics/market/price_usd_close",
        "params": {
          "a": "BTC",
          "i": "24h"
        }
      }
    }
  ],
  "fallback_strategy": "use_all",
  "deduplication": {
    "strategy": "prefer_priority",
    "conflict_resolution": "keep_highest_priority"
  }
}
```

```json
// config/providers/coinmetrics.json
{
  "provider_name": "coinmetrics",
  "display_name": "Coin Metrics",
  "enabled": true,
  "api_config": {
    "base_url": "https://api.coinmetrics.io/v4",
    "api_key_env": "COINMETRICS_API_KEY",
    "tier": "pro"
  },
  "rate_limits": {
    "requests_per_window": 6000,
    "window_seconds": 20,
    "safety_margin": 0.9
  },
  "retry_config": {
    "max_attempts": 3,
    "backoff_multiplier": 2,
    "max_wait_seconds": 60
  },
  "capabilities": [
    "asset_metrics",
    "market_trades",
    "exchange_metrics",
    "market_orderbooks"
  ]
}
```

### 3. Core Interfaces

```python
# src/core/interfaces.py

from abc import ABC, abstractmethod
from typing import Generator, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ValidationResult:
    valid: bool
    adjusted_start_time: Optional[datetime] = None
    adjusted_end_time: Optional[datetime] = None
    reason: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class FetchResult:
    data: List[Dict]
    next_cursor: Optional[str] = None
    has_more: bool = False
    metadata: Optional[Dict] = None

class DataProviderInterface(ABC):
    """
    Every data provider must implement this interface
    Core system only interacts through this interface
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier"""
        pass
    
    @abstractmethod
    def initialize(self, provider_config: Dict) -> None:
        """Initialize with provider-specific config"""
        pass
    
    @abstractmethod
    def validate_endpoint(self, endpoint_config: Dict) -> ValidationResult:
        """
        Validate that endpoint exists and has data
        Returns adjusted date ranges based on actual availability
        """
        pass
    
    @abstractmethod
    def fetch_data_batch(
        self, 
        endpoint_config: Dict, 
        start_time: datetime, 
        end_time: datetime,
        cursor: Optional[str] = None
    ) -> FetchResult:
        """
        Fetch one batch of data
        Returns data + pagination cursor if more exists
        """
        pass
    
    @abstractmethod
    def fetch_data_stream(
        self, 
        endpoint_config: Dict, 
        start_time: datetime, 
        end_time: datetime
    ) -> Generator[List[Dict], None, None]:
        """
        Stream data in chunks (generator)
        Handles pagination internally
        Yields standardized data format
        """
        pass
    
    @abstractmethod
    def get_rate_limiter(self) -> 'RateLimiterInterface':
        """Return provider-specific rate limiter"""
        pass
    
    @abstractmethod
    def transform_to_standard_schema(
        self, 
        raw_data: List[Dict], 
        endpoint_config: Dict,
        schema_type: str
    ) -> List[Dict]:
        """
        Transform provider-specific format to standard schema
        schema_type: 'asset_metrics', 'market_trades', etc.
        """
        pass
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict) -> Dict:
        """
        Provider-specific error handling
        Returns: {'retry': bool, 'wait_seconds': int, 'error_type': str}
        """
        pass

class RateLimiterInterface(ABC):
    """Rate limiting interface - each provider implements differently"""
    
    @abstractmethod
    def acquire(self) -> None:
        """Block until request can be made"""
        pass
    
    @abstractmethod
    def get_current_rate(self) -> float:
        """Return current requests per second"""
        pass
    
    @abstractmethod
    def get_remaining_quota(self) -> Optional[int]:
        """Return remaining requests in current window"""
        pass
```

### 4. Provider Implementation Example

```python
# src/providers/coinmetrics/provider.py

from src.core.interfaces import DataProviderInterface, ValidationResult, FetchResult
from .client import CoinMetricsClient
from .rate_limiter import CoinMetricsRateLimiter
from .transformer import CoinMetricsTransformer
from typing import Generator, Dict, List
from datetime import datetime

class CoinMetricsProvider(DataProviderInterface):
    """
    CoinMetrics-specific implementation
    Plugs into core system via standard interface
    """
    
    @property
    def provider_name(self) -> str:
        return "coinmetrics"
    
    def initialize(self, provider_config: Dict) -> None:
        self.config = provider_config
        self.client = CoinMetricsClient(
            api_key=os.getenv(provider_config['api_config']['api_key_env']),
            base_url=provider_config['api_config']['base_url']
        )
        self.rate_limiter = CoinMetricsRateLimiter(
            requests_per_window=provider_config['rate_limits']['requests_per_window'],
            window_seconds=provider_config['rate_limits']['window_seconds']
        )
        self.transformer = CoinMetricsTransformer()
    
    def validate_endpoint(self, endpoint_config: Dict) -> ValidationResult:
        """Use CoinMetrics catalog API to validate"""
        catalog_endpoint = f"catalog-v2/{endpoint_config['endpoint_type']}"
        
        try:
            catalog_data = self.client.fetch_catalog(
                endpoint=catalog_endpoint,
                params=endpoint_config['params']
            )
            
            # Parse catalog response to find actual data availability
            min_time, max_time = self._parse_catalog_times(catalog_data)
            
            return ValidationResult(
                valid=True,
                adjusted_start_time=min_time,
                adjusted_end_time=max_time,
                metadata={'catalog_data': catalog_data}
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                reason=f"Validation failed: {str(e)}"
            )
    
    def fetch_data_stream(
        self, 
        endpoint_config: Dict, 
        start_time: datetime, 
        end_time: datetime
    ) -> Generator[List[Dict], None, None]:
        """
        Stream data from CoinMetrics
        Handles pagination automatically
        Yields chunks in standard format
        """
        cursor = None
        
        while True:
            # Respect rate limits
            self.rate_limiter.acquire()
            
            # Fetch batch
            result = self.fetch_data_batch(
                endpoint_config, 
                start_time, 
                end_time, 
                cursor
            )
            
            if result.data:
                # Transform to standard schema
                standard_data = self.transform_to_standard_schema(
                    result.data,
                    endpoint_config,
                    schema_type='asset_metrics'  # determined from config
                )
                yield standard_data
            
            if not result.has_more:
                break
            
            cursor = result.next_cursor
    
    def fetch_data_batch(
        self, 
        endpoint_config: Dict, 
        start_time: datetime, 
        end_time: datetime,
        cursor: Optional[str] = None
    ) -> FetchResult:
        """Fetch single batch from CoinMetrics"""
        if cursor:
            # Use next_page_url from previous response
            response = self.client.fetch_by_url(cursor)
        else:
            # Initial request
            response = self.client.fetch(
                endpoint=endpoint_config['endpoint_type'],
                params={
                    **endpoint_config['params'],
                    'start_time': start_time.isoformat() + 'Z',
                    'end_time': end_time.isoformat() + 'Z'
                }
            )
        
        return FetchResult(
            data=response.get('data', []),
            next_cursor=response.get('next_page_url'),
            has_more=bool(response.get('next_page_url')),
            metadata={'response_headers': response.get('headers')}
        )
    
    def transform_to_standard_schema(
        self, 
        raw_data: List[Dict], 
        endpoint_config: Dict,
        schema_type: str
    ) -> List[Dict]:
        """Transform CoinMetrics format to standard"""
        return self.transformer.transform(raw_data, schema_type)
    
    def get_rate_limiter(self) -> RateLimiterInterface:
        return self.rate_limiter
    
    def handle_error(self, error: Exception, context: Dict) -> Dict:
        """CoinMetrics-specific error handling"""
        if isinstance(error, requests.HTTPError):
            if error.response.status_code == 429:
                retry_after = int(error.response.headers.get('Retry-After', 20))
                return {
                    'retry': True,
                    'wait_seconds': retry_after,
                    'error_type': 'rate_limit'
                }
            elif error.response.status_code in [500, 502, 503, 504]:
                return {
                    'retry': True,
                    'wait_seconds': 5,
                    'error_type': 'server_error'
                }
            elif error.response.status_code == 401:
                return {
                    'retry': False,
                    'error_type': 'auth_error',
                    'fatal': True
                }
        
        return {'retry': True, 'wait_seconds': 10, 'error_type': 'unknown'}
```

### 5. Provider Registry (Auto-Discovery)

```python
# src/providers/registry.py

from typing import Dict, Type
from src.core.interfaces import DataProviderInterface

class ProviderRegistry:
    """
    Automatically discovers and registers all providers
    Add new provider by creating a new directory in src/providers/
    """
    
    _providers: Dict[str, Type[DataProviderInterface]] = {}
    
    @classmethod
    def register(cls, provider_class: Type[DataProviderInterface]) -> None:
        """Register a provider implementation"""
        provider_name = provider_class().provider_name
        cls._providers[provider_name] = provider_class
        logger.info(f"Registered provider: {provider_name}")
    
    @classmethod
    def get_provider(cls, provider_name: str) -> Type[DataProviderInterface]:
        """Get provider class by name"""
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return cls._providers[provider_name]
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers"""
        return list(cls._providers.keys())
    
    @classmethod
    def auto_discover(cls) -> None:
        """
        Auto-discover providers in src/providers/ directory
        Each provider must have __init__.py that calls register()
        """
        import importlib
        import pkgutil
        import src.providers as providers_package
        
        for _, name, _ in pkgutil.iter_modules(providers_package.__path__):
            if name not in ['base_provider', 'registry', '__pycache__']:
                try:
                    module = importlib.import_module(f'src.providers.{name}')
                    logger.info(f"Discovered provider module: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load provider {name}: {e}")

# src/providers/coinmetrics/__init__.py
from src.providers.registry import ProviderRegistry
from .provider import CoinMetricsProvider

# Auto-register on import
ProviderRegistry.register(CoinMetricsProvider)

# src/providers/glassnode/__init__.py
from src.providers.registry import ProviderRegistry
from .provider import GlassnodeProvider

# Auto-register on import
ProviderRegistry.register(GlassnodeProvider)
```

### 6. Core Orchestrator (Provider-Agnostic)

```python
# src/core/orchestrator.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from src.providers.registry import ProviderRegistry
from src.core.storage.supabase_manager import SupabaseManager
from src.core.storage.database_writer import DatabaseWriter
from src.core.storage.progress_tracker import ProgressTracker

class BackfillOrchestrator:
    """
    Provider-agnostic orchestrator
    Works with any provider that implements DataProviderInterface
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.supabase_manager = SupabaseManager()
        self.db_writer = DatabaseWriter(self.supabase_manager)
        self.progress_tracker = ProgressTracker(self.supabase_manager)
        
        # Auto-discover and initialize providers
        ProviderRegistry.auto_discover()
        self.providers = self._initialize_providers()
    
    def _initialize_providers(self) -> Dict[str, DataProviderInterface]:
        """Initialize all enabled providers"""
        providers = {}
        
        for provider_name in ProviderRegistry.list_providers():
            provider_config = self._load_provider_config(provider_name)
            
            if provider_config.get('enabled', False):
                provider_class = ProviderRegistry.get_provider(provider_name)
                provider = provider_class()
                provider.initialize(provider_config)
                providers[provider_name] = provider
                logger.info(f"Initialized provider: {provider_name}")
        
        return providers
    
    def run_backfill(
        self, 
        endpoint_configs: List[Dict], 
        max_workers: int = 10
    ) -> Dict:
        """
        Run backfill for all endpoints across all providers
        Each endpoint can specify multiple providers
        """
        results = {
            'total_endpoints': len(endpoint_configs),
            'completed': 0,
            'failed': 0,
            'provider_stats': {}
        }
        
        # Validate all endpoints first
        validated_configs = []
        for config in endpoint_configs:
            validated = self._validate_endpoint_all_providers(config)
            if validated:
                validated_configs.append(validated)
        
        # Execute backfill with threading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for endpoint_config in validated_configs:
                for provider_config in endpoint_config['providers']:
                    if provider_config['enabled']:
                        future = executor.submit(
                            self._backfill_endpoint_provider,
                            endpoint_config,
                            provider_config
                        )
                        futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['success']:
                        results['completed'] += 1
                    else:
                        results['failed'] += 1
                    
                    # Track per-provider stats
                    provider = result['provider']
                    if provider not in results['provider_stats']:
                        results['provider_stats'][provider] = {
                            'completed': 0, 'failed': 0, 'records': 0
                        }
                    
                    results['provider_stats'][provider]['records'] += result.get('records', 0)
                    if result['success']:
                        results['provider_stats'][provider]['completed'] += 1
                    else:
                        results['provider_stats'][provider]['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Backfill task failed: {e}")
                    results['failed'] += 1
        
        return results
    
    def _validate_endpoint_all_providers(
        self, 
        endpoint_config: Dict
    ) -> Optional[Dict]:
        """
        Validate endpoint across all specified providers
        Adjust date ranges based on actual availability
        """
        validated_config = endpoint_config.copy()
        validated_providers = []
        
        for provider_config in endpoint_config['providers']:
            provider_name = provider_config['name']
            
            if provider_name not in self.providers:
                logger.warning(f"Provider {provider_name} not available")
                continue
            
            provider = self.providers[provider_name]
            validation = provider.validate_endpoint(provider_config['config'])
            
            if validation.valid:
                # Adjust date range based on provider's actual availability
                adjusted_provider_config = provider_config.copy()
                adjusted_provider_config['actual_start'] = validation.adjusted_start_time
                adjusted_provider_config['actual_end'] = validation.adjusted_end_time
                validated_providers.append(adjusted_provider_config)
                
                logger.info(
                    f"Validated {endpoint_config['endpoint_id']} "
                    f"for {provider_name}: "
                    f"{validation.adjusted_start_time} to {validation.adjusted_end_time}"
                )
            else:
                logger.warning(
                    f"Validation failed for {endpoint_config['endpoint_id']} "
                    f"on {provider_name}: {validation.reason}"
                )
        
        if not validated_providers:
            logger.error(f"No valid providers for {endpoint_config['endpoint_id']}")
            return None
        
        validated_config['providers'] = validated_providers
        return validated_config
    
    def _backfill_endpoint_provider(
        self, 
        endpoint_config: Dict,
        provider_config: Dict
    ) -> Dict:
        """
        Backfill single endpoint from single provider
        Provider-agnostic - just calls interface methods
        """
        provider_name = provider_config['name']
        provider = self.providers[provider_name]
        endpoint_id = f"{endpoint_config['endpoint_id']}_{provider_name}"
        
        try:
            # Check if already completed
            progress = self.progress_tracker.get_progress(endpoint_id)
            if progress and progress['status'] == 'completed':
                logger.info(f"Skipping completed endpoint: {endpoint_id}")
                return {'success': True, 'provider': provider_name, 'records': 0}
            
            # Get date range
            start_time = provider_config['actual_start']
            end_time = provider_config['actual_end']
            
            # Resume from checkpoint if exists
            if progress and progress['last_successful_time']:
                start_time = progress['last_successful_time']
            
            # Stream data from provider
            total_records = 0
            
            for data_chunk in provider.fetch_data_stream(
                provider_config['config'],
                start_time,
                end_time
            ):
                # Add provider metadata
                for record in data_chunk:
                    record['provider'] = provider_name
                    record['provider_priority'] = provider_config.get('priority', 999)
                
                # Write to database
                self.db_writer.upsert_batch(
                    table=endpoint_config['table'],
                    data=data_chunk,
                    primary_keys=endpoint_config['primary_keys'],
                    batch_size=1000
                )
                
                total_records += len(data_chunk)
                
                # Update progress
                last_time = max(record['time'] for record in data_chunk)
                self.progress_tracker.update_progress(
                    endpoint_id=endpoint_id,
                    last_time=last_time,
                    records_count=len(data_chunk)
                )
                
                logger.info(
                    f"Backfilled {len(data_chunk)} records for {endpoint_id} "
                    f"(total: {total_records})"
                )
            
            # Mark as completed
            self.progress_tracker.mark_completed(endpoint_id)
            
            return {
                'success': True,
                'provider': provider_name,
                'endpoint_id': endpoint_id,
                'records': total_records
            }
            
        except Exception as e:
            logger.error(f"Backfill failed for {endpoint_id}: {e}", exc_info=True)
            self.progress_tracker.log_error(endpoint_id, str(e))
            
            return {
                'success': False,
                'provider': provider_name,
                'endpoint_id': endpoint_id,
                'error': str(e)
            }
```

### 7. Standard Data Schemas

```python
# src/schemas/asset_metrics.py

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal

class AssetMetric(BaseModel):
    """
    Standard schema for asset metrics
    All providers must transform to this format
    """
    provider: str
    provider_priority: int
    asset: str
    metric: str
    time: datetime
    value: Optional[Decimal]
    frequency: str  # '1d', '1h', '5m', etc.
    metadata: Optional[dict] = {}
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }

class MarketTrade(BaseModel):
    """Standard schema for market trades"""
    provider: str
    provider_priority: int
    market: str
    time: datetime
    price: Decimal
    amount: Decimal
    side: str  # 'buy' or 'sell'
    trade_id: str
    metadata: Optional[dict] = {}

class ExchangeMetric(BaseModel):
    """Standard schema for exchange metrics"""
    provider: str
    provider_priority: int
    exchange: str
    metric: str
    time: datetime
    value: Optional[Decimal]
    frequency: str
    metadata: Optional[dict] = {}
```

### 8. Database Schema (Multi-Provider)

```sql
-- Progress tracking (provider-aware)
CREATE TABLE backfill_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint_id TEXT NOT NULL,  -- includes provider: "btc_metrics_coinmetrics"
    provider TEXT NOT NULL,
    endpoint_type TEXT NOT NULL,
    table_name TEXT NOT NULL,
    status TEXT CHECK (status IN ('pending', 'validating', 'running', 'completed', 'failed')),
    last_successful_time TIMESTAMPTZ,
    total_records_fetched BIGINT DEFAULT 0,
    total_api_calls INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(endpoint_id, provider)
);

CREATE INDEX idx_progress_provider ON backfill_progress(provider);
CREATE INDEX idx_progress_status ON backfill_progress(status);

-- Asset metrics (multi-provider with priority)
CREATE TABLE asset_metrics (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    asset TEXT NOT NULL,
    metric TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    value NUMERIC,
    frequency TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (provider, asset, metric, time)
);

-- Indexes for querying
CREATE INDEX idx_asset_metrics_composite ON asset_metrics(asset, metric, time);
CREATE INDEX idx_asset_metrics_provider_priority ON asset_metrics(asset, metric, time, provider_priority);

-- View: Get best data per metric (highest priority provider)
CREATE VIEW asset_metrics_best AS
SELECT DISTINCT ON (asset, metric, time)
    provider,
    asset,
    metric,
    time,
    value,
    frequency,
    metadata
FROM asset_metrics
ORDER BY asset, metric, time, provider_priority ASC;

-- Market trades (multi-provider)
CREATE TABLE market_trades (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    market TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    trade_id TEXT NOT NULL,
    price NUMERIC NOT NULL,
    amount NUMERIC NOT NULL,
    side TEXT CHECK (side IN ('buy', 'sell')),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (provider, market, time, trade_id)
);

CREATE INDEX idx_market_trades_composite ON market_trades(market, time);
```

### 9. Adding a New Provider (Step-by-Step)

**To add Messari (or any new provider):**

1. **Create provider directory:**
```bash
mkdir src/providers/messari
touch src/providers/messari/__init__.py
touch src/providers/messari/provider.py
touch src/providers/messari/client.py
touch src/providers/messari/rate_limiter.py
touch src/providers/messari/transformer.py
```

2. **Implement DataProviderInterface:**
```python
# src/providers/messari/provider.py
from src.core.interfaces import DataProviderInterface

class MessariProvider(DataProviderInterface):
    @property
    def provider_name(self) -> str:
        return "messari"
    
    # Implement all required methods...
```

3. **Register provider:**
```python
# src/providers/messari/__init__.py
from src.providers.registry import ProviderRegistry
from .provider import MessariProvider

ProviderRegistry.register(MessariProvider)
```

4. **Add provider config:**
```json
// config/providers/messari.json
{
  "provider_name": "messari",
  "enabled": true,
  "api_config": {
    "base_url": "https://data.messari.io/api/v1",
    "api_key_env": "MESSARI_API_KEY"
  },
  "rate_limits": {
    "requests_per_minute": 60
  }
}
```

5. **Update endpoint configs to include new provider:**
```json
// config/endpoints/btc_metrics.json
{
  "providers": [
    {"name": "coinmetrics", "priority": 1, ...},
    {"name": "glassnode", "priority": 2, ...},
    {"name": "messari", "priority": 3, "enabled": true, ...}
  ]
}
```

**That's it!** The system auto-discovers and uses the new provider.

### 10. CLI Usage (Provider-Aware)

```bash
# Backfill all endpoints from all enabled providers
python backfill.py --config config/endpoints/

# Backfill specific providers only
python backfill.py --providers coinmetrics,glassnode

# Backfill specific endpoint with specific provider
python backfill.py --endpoint btc_metrics --provider coinmetrics

# List all available providers
python backfill.py --list-providers

# Validate all endpoints across all providers (no backfill)
python backfill.py --validate-only

# Resume failed endpoints
python backfill.py --resume-failed

# Show provider statistics
python backfill.py --stats --provider coinmetrics
```

### 11. Multi-Provider Data Strategy

#### Strategy 1: Redundancy (Use All Providers)
```json
{
  "fallback_strategy": "use_all",
  "deduplication": {
    "strategy": "prefer_priority",
    "conflict_resolution": "keep_highest_priority"
  }
}
```
- Fetch from all providers
- Store all data with provider metadata
- Query uses view to get highest priority data
- Fallback if primary provider fails

#### Strategy 2: Primary with Fallback
```json
{
  "fallback_strategy": "primary_with_fallback",
  "primary_provider": "coinmetrics",
  "fallback_providers": ["glassnode", "messari"]
}
```
- Try primary provider first
- Only use fallback if primary fails or has gaps
- Reduces API costs

#### Strategy 3: Data Comparison/Validation
```json
{
  "strategy": "comparison",
  "min_providers": 2,
  "validation": {
    "max_deviation_percent": 1.0,
    "flag_discrepancies": true
  }
}
```
- Fetch from multiple providers
- Compare values
- Flag discrepancies for review
- Quality assurance

### 12. Monitoring Dashboard Data

```sql
-- Provider health check view
CREATE VIEW provider_health AS
SELECT 
    provider,
    COUNT(*) as total_endpoints,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    SUM(total_records_fetched) as total_records,
    AVG(total_api_calls) as avg_api_calls,
    MAX(updated_at) as last_activity
FROM backfill_progress
GROUP BY provider;

-- Data coverage comparison
CREATE VIEW data_coverage_by_provider AS
SELECT 
    asset,
    metric,
    provider,
    MIN(time) as earliest_data,
    MAX(time) as latest_data,
    COUNT(*) as data_points
FROM asset_metrics
GROUP BY asset, metric, provider;
```

### 13. Testing Strategy

#### Unit Tests (Per Provider)
```python
# tests/providers/test_coinmetrics.py
def test_coinmetrics_implements_interface():
    provider = CoinMetricsProvider()
    assert isinstance(provider, DataProviderInterface)

def test_coinmetrics_validation():
    provider = CoinMetricsProvider()
    result = provider.validate_endpoint(mock_config)
    assert result.valid == True

def test_coinmetrics_pagination():
    provider = CoinMetricsProvider()
    data_chunks = list(provider.fetch_data_stream(mock_config, start, end))
    assert len(data_chunks) > 0
```

#### Integration Tests (Multi-Provider)
```python
# tests/integration/test_multi_provider.py
def test_backfill_with_multiple_providers():
    orchestrator = BackfillOrchestrator(config)
    results = orchestrator.run_backfill(endpoints)
    
    assert 'coinmetrics' in results['provider_stats']
    assert 'glassnode' in results['provider_stats']
    assert results['completed'] > 0

def test_fallback_on_provider_failure():
    # Simulate CoinMetrics failure
    results = orchestrator.run_backfill_with_failure(endpoints)
    
    # Should fallback to Glassnode
    assert results['provider_stats']['glassnode']['completed'] > 0
```

### 14. Key Benefits of This Architecture

✅ **Add New Providers Without Modifying Core**
- Drop in new provider directory
- Implement interface
- Auto-discovered and ready to use

✅ **Provider Independence**
- Each provider is self-contained
- Provider failures don't affect others
- Easy to disable/enable providers

✅ **Data Redundancy & Quality**
- Fetch from multiple providers
- Compare and validate data
- Fallback mechanisms

✅ **Flexible Data Strategy**
- Use all providers for redundancy
- Primary + fallback for cost efficiency
- Comparison mode for quality assurance

✅ **Easy Testing**
- Test providers independently
- Mock individual providers
- Integration tests across providers

✅ **Clear Ownership**
- Each provider team owns their implementation
- Core team owns orchestration only
- Clean separation of concerns

✅ **Scalability**
- Add providers without limits
- Thread pool scales with providers
- Rate limiters per provider

### 15. Implementation Timeline (Revised)

**Phase 1: Core Framework**
- Design and implement interfaces
- Build provider registry system
- Create SupabaseManager singleton
- Design standard schemas
- Database schema with multi-provider support

**Phase 2: First Provider (CoinMetrics)**
- Implement CoinMetricsProvider
- Build catalog validation
- Implement rate limiter
- Test thoroughly

**Phase 3: Core Orchestrator**
- Build provider-agnostic orchestrator
- Multi-provider validation
- Progress tracking
- Error handling
- CLI interface

**Phase 4: Second Provider**
- Implement second provider
- Test multi-provider backfill
- Implement fallback strategies
- Load testing
- Documentation

**Phase 5+: Additional Providers**
- Each new provider: 3-5 steps
- Messari, Dune, etc.
- Continuous improvement

## Questions for Backend Team

1. **Provider Priority**: Which data providers are most important? (determines implementation order)

2. **Data Strategy**: Redundancy (all providers) or Primary+Fallback? (affects architecture decisions)

3. **Data Quality**: Need comparison/validation between providers? (adds complexity but ensures quality)

4. **API Keys**: Do you already have accounts with multiple providers?

5. **Future Providers**: Any specific providers you know you'll need? (helps design decisions)

6. **Existing Schema**: Any existing database tables we need to integrate with?

7. **Team Structure**: Will different engineers own different providers? (clean separation makes this easy)

This architecture is **production-ready for infinite providers**. Add 10 more providers? Just create 10 new directories and implement the interface. The core never changes.