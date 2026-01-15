"""Standard schema for orderbook data"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal


class OrderbookLevel(BaseModel):
    """
    Standard schema for a single orderbook level (bid or ask).
    Each row represents one price level in the orderbook.
    
    For a snapshot with 10 bids and 10 asks, this generates 20 records.
    """
    provider: str = Field(..., description="Data provider name")
    provider_priority: int = Field(..., description="Provider priority for deduplication")
    market: str = Field(..., description="Market identifier (e.g., 'coinbase-btc-usd-spot')")
    time: datetime = Field(..., description="Snapshot timestamp")
    snapshot_id: str = Field(..., description="Unique snapshot identifier (coin_metrics_id)")
    side: str = Field(..., description="Order side: 'bid' or 'ask'")
    price: Decimal = Field(..., description="Price at this level")
    size: Decimal = Field(..., description="Size/volume at this level")
    level: int = Field(..., description="Orderbook depth level (0 = best bid/ask, 1 = second best, etc.)")
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
            'market': self.market,
            'time': self.time.isoformat(),
            'snapshot_id': self.snapshot_id,
            'side': self.side,
            'price': str(self.price),
            'size': str(self.size),
            'level': self.level,
            'metadata': self.metadata
        }


class OrderbookSnapshot(BaseModel):
    """
    Full orderbook snapshot containing all bids and asks.
    This is a convenience model for API responses.
    """
    provider: str
    provider_priority: int
    market: str
    time: datetime
    snapshot_id: str
    bids: list[OrderbookLevel] = Field(default_factory=list, description="All bid levels")
    asks: list[OrderbookLevel] = Field(default_factory=list, description="All ask levels")
    metadata: Optional[dict] = Field(default_factory=dict)
    
    @property
    def best_bid(self) -> Optional[OrderbookLevel]:
        """Get the best bid (highest price)"""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderbookLevel]:
        """Get the best ask (lowest price)"""
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate the bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate the mid price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / Decimal('2')
        return None

