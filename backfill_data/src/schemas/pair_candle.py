"""Standard schema for pair candle (OHLCV) data"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal


class PairCandle(BaseModel):
    """
    Standard schema for pair candles (OHLCV).
    Represents candlestick data for asset pairs (e.g., btc-usd, eth-usd).
    
    CoinMetrics response fields:
    - pair, time, price_open, price_close, price_high, price_low, vwap, volume, candle_usd_volume
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority for deduplication")
    pair: str = Field(..., description="Asset pair identifier (e.g., 'btc-usd', 'eth-usd')")
    time: datetime = Field(..., description="Candle open timestamp")
    price_open: Decimal = Field(..., description="Opening price")
    price_high: Decimal = Field(..., description="Highest price in period")
    price_low: Decimal = Field(..., description="Lowest price in period")
    price_close: Decimal = Field(..., description="Closing price")
    volume: Optional[Decimal] = Field(None, description="Trading volume")
    vwap: Optional[Decimal] = Field(None, description="Volume-weighted average price")
    candle_usd_volume: Optional[Decimal] = Field(None, description="Candle volume in USD")
    frequency: str = Field(default="1d", description="Candle frequency")
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
            'pair': self.pair,
            'time': self.time.isoformat(),
            'price_open': str(self.price_open),
            'price_high': str(self.price_high),
            'price_low': str(self.price_low),
            'price_close': str(self.price_close),
            'volume': str(self.volume) if self.volume is not None else None,
            'vwap': str(self.vwap) if self.vwap is not None else None,
            'candle_usd_volume': str(self.candle_usd_volume) if self.candle_usd_volume is not None else None,
            'frequency': self.frequency,
            'metadata': self.metadata
        }
