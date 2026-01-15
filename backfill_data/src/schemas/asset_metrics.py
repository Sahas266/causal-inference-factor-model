"""Standard schema for asset metrics data"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal


class AssetMetric(BaseModel):
    """
    Standard schema for asset metrics.
    All providers must transform to this format.
    
    Examples of metrics:
    - PriceUSD: Asset price in USD
    - CapMrktCurUSD: Market capitalization
    - TxCnt: Transaction count
    - AdrActCnt: Active addresses
    """
    provider: str = Field(..., description="Data provider name (e.g., 'coinmetrics', 'glassnode')")
    provider_priority: int = Field(..., description="Provider priority for deduplication (lower is better)")
    asset: str = Field(..., description="Asset identifier (e.g., 'BTC', 'ETH')")
    metric: str = Field(..., description="Metric name (e.g., 'PriceUSD', 'CapMrktCurUSD')")
    time: datetime = Field(..., description="Timestamp of the metric")
    value: Optional[Decimal] = Field(None, description="Metric value (can be null for missing data)")
    frequency: str = Field(..., description="Data frequency (e.g., '1d', '1h', '5m')")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional provider-specific metadata")
    
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
            'asset': self.asset,
            'metric': self.metric,
            'time': self.time.isoformat(),
            'value': str(self.value) if self.value is not None else None,
            'frequency': self.frequency,
            'metadata': self.metadata
        }

