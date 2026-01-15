"""Core interfaces for provider abstraction layer"""

from abc import ABC, abstractmethod
from typing import Generator, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of endpoint validation"""
    valid: bool
    adjusted_start_time: Optional[datetime] = None
    adjusted_end_time: Optional[datetime] = None
    reason: Optional[str] = None
    metadata: Optional[Dict] = field(default_factory=dict)


@dataclass
class FetchResult:
    """Result of a data fetch operation"""
    data: List[Dict]
    next_cursor: Optional[str] = None
    has_more: bool = False
    metadata: Optional[Dict] = field(default_factory=dict)


class DataProviderInterface(ABC):
    """
    Abstract interface that every data provider must implement.
    Core system only interacts through this interface.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier"""
        pass
    
    @abstractmethod
    def initialize(self, provider_config: Dict) -> None:
        """
        Initialize provider with provider-specific configuration.
        
        Args:
            provider_config: Configuration dictionary for the provider
        """
        pass
    
    @abstractmethod
    def validate_endpoint(self, endpoint_config: Dict) -> ValidationResult:
        """
        Validate that endpoint exists and has data.
        Returns adjusted date ranges based on actual availability.
        
        Args:
            endpoint_config: Configuration for specific endpoint
            
        Returns:
            ValidationResult with availability information
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
        Fetch one batch of data.
        
        Args:
            endpoint_config: Configuration for the endpoint
            start_time: Start of time range
            end_time: End of time range
            cursor: Pagination cursor from previous batch
            
        Returns:
            FetchResult with data and pagination info
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
        Stream data in chunks (generator pattern).
        Handles pagination internally.
        Yields standardized data format.
        
        Args:
            endpoint_config: Configuration for the endpoint
            start_time: Start of time range
            end_time: End of time range
            
        Yields:
            Lists of records in standard format
        """
        pass
    
    @abstractmethod
    def get_rate_limiter(self) -> 'RateLimiterInterface':
        """
        Return provider-specific rate limiter instance.
        
        Returns:
            RateLimiterInterface implementation
        """
        pass
    
    @abstractmethod
    def transform_to_standard_schema(
        self, 
        raw_data: List[Dict], 
        endpoint_config: Dict,
        schema_type: str
    ) -> List[Dict]:
        """
        Transform provider-specific format to standard schema.
        
        Args:
            raw_data: Raw data from provider API
            endpoint_config: Endpoint configuration
            schema_type: Type of schema ('asset_metrics', 'market_trades', 'exchange_metrics')
            
        Returns:
            List of records in standard format
        """
        pass
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict) -> Dict:
        """
        Provider-specific error handling logic.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            
        Returns:
            Dictionary with error handling instructions:
                - retry: bool - whether to retry
                - wait_seconds: int - how long to wait before retry
                - error_type: str - classification of error
                - fatal: bool (optional) - whether error is unrecoverable
        """
        pass


class RateLimiterInterface(ABC):
    """
    Abstract interface for rate limiting.
    Each provider implements rate limiting differently.
    """
    
    @abstractmethod
    def acquire(self) -> None:
        """
        Block until a request can be made.
        Implements rate limiting logic.
        """
        pass
    
    @abstractmethod
    def get_current_rate(self) -> float:
        """
        Get current request rate.
        
        Returns:
            Current requests per second
        """
        pass
    
    @abstractmethod
    def get_remaining_quota(self) -> Optional[int]:
        """
        Get remaining requests in current window.
        
        Returns:
            Number of remaining requests, or None if not applicable
        """
        pass

