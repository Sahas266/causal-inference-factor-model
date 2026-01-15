"""Standard schema for exchange metrics data"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal


class ExchangeMetric(BaseModel):
    """
    Standard schema for exchange metrics.
    Represents metrics about exchanges (flows, balances, etc).
    
    Examples of metrics:
    - flow_in_btc: Bitcoin flowing into exchange
    - flow_out_btc: Bitcoin flowing out of exchange
    - balance_btc: Total Bitcoin balance on exchange
    - volume_24h: 24-hour trading volume
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority for deduplication")
    exchange: str = Field(..., description="Exchange identifier (e.g., 'binance', 'coinbase')")
    metric: str = Field(..., description="Metric name (e.g., 'flow_in_btc', 'balance_btc')")
    time: datetime = Field(..., description="Timestamp of the metric")
    value: Optional[Decimal] = Field(None, description="Metric value")
    frequency: str = Field(..., description="Data frequency (e.g., '1d', '1h')")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }
    
    def to_db_dict(self) -> dict:
        """Convert to dictionary suitable for database insertion"""
        return {
            'provider': self.provider,
            'provider_priority': self.provider_priority,
            'exchange': self.exchange,
            'metric': self.metric,
            'time': self.time.isoformat(),
            'value': str(self.value) if self.value is not None else None,
            'frequency': self.frequency,
            'metadata': self.metadata
        }

