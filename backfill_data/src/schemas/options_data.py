"""Standard schemas for options market data (implied volatility, greeks)"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal


class MarketImpliedVolatility(BaseModel):
    """
    Standard schema for market implied volatility.
    Represents implied volatility for options markets.
    
    CoinMetrics response fields:
    - market, time, implied_volatility, database_time, coin_metrics_id
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority for deduplication")
    market: str = Field(..., description="Market identifier (e.g., 'deribit-BTC-PERPETUAL-future')")
    time: datetime = Field(..., description="Snapshot timestamp")
    implied_volatility: Optional[Decimal] = Field(None, description="Implied volatility value")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }

    def to_db_dict(self) -> dict:
        return {
            'provider': self.provider,
            'provider_priority': self.provider_priority,
            'market': self.market,
            'time': self.time.isoformat(),
            'implied_volatility': str(self.implied_volatility) if self.implied_volatility is not None else None,
            'metadata': self.metadata
        }


class MarketGreeks(BaseModel):
    """
    Standard schema for option greeks.
    Represents delta, gamma, vega, theta, rho for options markets.
    
    CoinMetrics response fields:
    - market, time, delta, gamma, vega, theta, rho, database_time, coin_metrics_id
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority for deduplication")
    market: str = Field(..., description="Market identifier")
    time: datetime = Field(..., description="Snapshot timestamp")
    delta: Optional[Decimal] = Field(None, description="Option delta")
    gamma: Optional[Decimal] = Field(None, description="Option gamma")
    vega: Optional[Decimal] = Field(None, description="Option vega")
    theta: Optional[Decimal] = Field(None, description="Option theta")
    rho: Optional[Decimal] = Field(None, description="Option rho")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }

    def to_db_dict(self) -> dict:
        return {
            'provider': self.provider,
            'provider_priority': self.provider_priority,
            'market': self.market,
            'time': self.time.isoformat(),
            'delta': str(self.delta) if self.delta is not None else None,
            'gamma': str(self.gamma) if self.gamma is not None else None,
            'vega': str(self.vega) if self.vega is not None else None,
            'theta': str(self.theta) if self.theta is not None else None,
            'rho': str(self.rho) if self.rho is not None else None,
            'metadata': self.metadata
        }
