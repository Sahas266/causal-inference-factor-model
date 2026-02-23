"""Standard schemas for derivatives market data (open interest, liquidations, funding rates)"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal


class MarketOpenInterest(BaseModel):
    """
    Standard schema for market open interest.
    Represents open interest for futures markets.
    
    CoinMetrics response fields:
    - market, time, contract_count, value_usd, database_time, coin_metrics_id
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority for deduplication")
    market: str = Field(..., description="Market identifier (e.g., 'binance-BTCUSDT-future')")
    time: datetime = Field(..., description="Snapshot timestamp")
    contract_count: Optional[Decimal] = Field(None, description="Number of open contracts")
    value_usd: Optional[Decimal] = Field(None, description="Open interest value in USD (deprecated)")
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
            'contract_count': str(self.contract_count) if self.contract_count is not None else None,
            'value_usd': str(self.value_usd) if self.value_usd is not None else None,
            'metadata': self.metadata
        }


class MarketLiquidation(BaseModel):
    """
    Standard schema for market liquidations.
    Represents individual liquidation events in futures markets.
    
    CoinMetrics response fields:
    - market, time, coin_metrics_id, amount, price, type, side, database_time
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority for deduplication")
    market: str = Field(..., description="Market identifier")
    time: datetime = Field(..., description="Liquidation timestamp")
    coin_metrics_id: str = Field(..., description="Unique identifier from CoinMetrics")
    amount: Optional[Decimal] = Field(None, description="Liquidation amount")
    price: Optional[Decimal] = Field(None, description="Liquidation price")
    side: Optional[str] = Field(None, description="Liquidation side: 'buy' or 'sell'")
    type: Optional[str] = Field(None, description="Liquidation type")
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
            'coin_metrics_id': self.coin_metrics_id,
            'amount': str(self.amount) if self.amount is not None else None,
            'price': str(self.price) if self.price is not None else None,
            'side': self.side,
            'type': self.type,
            'metadata': self.metadata
        }


class MarketFundingRate(BaseModel):
    """
    Standard schema for market funding rates.
    Represents funding rates for perpetual futures markets.
    
    CoinMetrics response fields:
    - market, time, rate, period, database_time, coin_metrics_id
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority for deduplication")
    market: str = Field(..., description="Market identifier")
    time: datetime = Field(..., description="Funding rate timestamp")
    rate: Optional[Decimal] = Field(None, description="Funding rate value")
    period: Optional[str] = Field(None, description="Funding period")
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
            'rate': str(self.rate) if self.rate is not None else None,
            'period': self.period,
            'metadata': self.metadata
        }
