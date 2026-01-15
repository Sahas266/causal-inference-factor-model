"""Standard schema for market data (trades, orderbooks)"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal


class MarketTrade(BaseModel):
    """
    Standard schema for market trades.
    Represents individual trades on exchanges.
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority for deduplication")
    market: str = Field(..., description="Market identifier (e.g., 'coinbase-btc-usd-spot')")
    time: datetime = Field(..., description="Trade timestamp")
    price: Decimal = Field(..., description="Trade price")
    amount: Decimal = Field(..., description="Trade amount/volume")
    side: str = Field(..., description="Trade side: 'buy' or 'sell'")
    trade_id: str = Field(..., description="Unique trade identifier from provider")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional trade metadata")
    
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
            'market': self.market,
            'time': self.time.isoformat(),
            'price': str(self.price),
            'amount': str(self.amount),
            'side': self.side,
            'trade_id': self.trade_id,
            'metadata': self.metadata
        }


class MarketOrderbook(BaseModel):
    """
    Standard schema for market orderbook snapshots.
    Represents bid/ask levels at a point in time.
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority")
    market: str = Field(..., description="Market identifier")
    time: datetime = Field(..., description="Snapshot timestamp")
    side: str = Field(..., description="Order side: 'bid' or 'ask'")
    price: Decimal = Field(..., description="Price level")
    amount: Decimal = Field(..., description="Amount at this price level")
    metadata: Optional[dict] = Field(default_factory=dict)
    
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
            'market': self.market,
            'time': self.time.isoformat(),
            'side': self.side,
            'price': str(self.price),
            'amount': str(self.amount),
            'metadata': self.metadata
        }

