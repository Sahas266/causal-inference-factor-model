"""
Configuration management using Pydantic settings.
Centralized configuration for database, cache, providers, and application settings.
"""

import os
from typing import Optional, List
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    url: str = Field(
        default="postgresql://user:password@localhost:5432/defi_pipeline",
        env="DATABASE_URL",
        description="PostgreSQL connection URL"
    )
    pool_size: int = Field(default=20, env="DB_POOL_SIZE", description="Connection pool size")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW", description="Max overflow connections")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT", description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE", description="Pool recycle time")

    @computed_field
    @property
    def is_timescaledb_enabled(self) -> bool:
        """Check if TimescaleDB is available in the database."""
        # This would be checked at runtime
        return True


class CacheSettings(BaseSettings):
    """Redis cache configuration settings."""

    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL", description="Redis connection URL")
    ttl: int = Field(default=300, env="CACHE_TTL", description="Default cache TTL in seconds")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS", description="Max Redis connections")
    health_check_interval: int = Field(default=30, env="REDIS_HEALTH_CHECK_INTERVAL", description="Health check interval")


class ProviderSettings(BaseSettings):
    """External provider API configuration."""

    timeout: int = Field(default=30, env="PROVIDER_TIMEOUT", description="Request timeout in seconds")
    max_retries: int = Field(default=3, env="PROVIDER_MAX_RETRIES", description="Max retry attempts")
    backoff_factor: float = Field(default=2.0, env="PROVIDER_BACKOFF_FACTOR", description="Exponential backoff factor")
    rate_limit_buffer: float = Field(default=0.1, env="PROVIDER_RATE_LIMIT_BUFFER", description="Rate limit buffer percentage")

    # Provider-specific API keys (optional - providers can be configured without hardcoded keys)
    dune_api_key: Optional[str] = Field(default=None, env="DUNE_API_KEY")
    etherscan_api_key: Optional[str] = Field(default=None, env="ETHERSCAN_API_KEY")
    defillama_api_key: Optional[str] = Field(default=None, env="DEFILLAMA_API_KEY")
    thegraph_api_key: Optional[str] = Field(default=None, env="THEGRAPH_API_KEY")
    coingecko_api_key: Optional[str] = Field(default=None, env="COINGECKO_API_KEY")
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")


class ApplicationSettings(BaseSettings):
    """Main application configuration."""

    log_level: str = Field(default="INFO", env="LOG_LEVEL", description="Logging level")
    debug: bool = Field(default=False, env="DEBUG", description="Debug mode")
    reload: bool = Field(default=False, env="RELOAD", description="Auto-reload on code changes")
    max_workers: int = Field(default=4, env="MAX_WORKERS", description="Max worker processes")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE", description="API rate limit")

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST", description="Server host")
    port: int = Field(default=8000, env="PORT", description="Server port")

    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"], env="CORS_ORIGINS", description="Allowed CORS origins"
    )

    # Health check settings
    health_check_interval: int = Field(default=60, env="HEALTH_CHECK_INTERVAL", description="Health check interval")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""

    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT", description="Prometheus metrics port")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS", description="Enable Prometheus metrics")
    metrics_path: str = Field(default="/metrics", env="METRICS_PATH", description="Metrics endpoint path")

    # Alert thresholds
    data_freshness_threshold: int = Field(
        default=3600, env="DATA_FRESHNESS_THRESHOLD", description="Data freshness alert threshold in seconds"
    )
    collection_failure_threshold: int = Field(
        default=3, env="COLLECTION_FAILURE_THRESHOLD", description="Consecutive failure threshold for alerts"
    )


class TaskSettings(BaseSettings):
    """Celery task configuration."""

    broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL", description="Celery broker URL")
    result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND", description="Celery result backend")
    timezone: str = Field(default="UTC", env="CELERY_TIMEZONE", description="Celery timezone")
    enable_utc: bool = Field(default=True, env="CELERY_ENABLE_UTC", description="Enable UTC timezone")

    # Task execution settings
    task_serializer: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
    result_serializer: str = Field(default="json", env="CELERY_RESULT_SERIALIZER")
    accept_content: List[str] = Field(default=["json"], env="CELERY_ACCEPT_CONTENT")

    # Worker settings
    worker_prefetch_multiplier: int = Field(default=1, env="CELERY_WORKER_PREFETCH_MULTIPLIER")
    worker_max_tasks_per_child: int = Field(default=1000, env="CELERY_WORKER_MAX_TASKS_PER_CHILD")


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections."""

    # Configuration sections
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    app: ApplicationSettings = Field(default_factory=ApplicationSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    task: TaskSettings = Field(default_factory=TaskSettings)

    # Application metadata
    version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Deployment environment"
    )

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()

# Export individual settings for convenience
database_settings = settings.database
cache_settings = settings.cache
provider_settings = settings.provider
app_settings = settings.app
monitoring_settings = settings.monitoring
task_settings = settings.task
