"""
Custom exceptions for the DeFi data pipeline.
Provides structured error handling throughout the application.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


class DeFiPipelineError(Exception):
    """Base exception for all DeFi pipeline errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataCollectionError(DeFiPipelineError):
    """Error during data collection operations."""
    pass


class ProviderError(DeFiPipelineError):
    """Error from external data providers."""
    pass


class ProviderUnavailableError(ProviderError):
    """Provider is temporarily unavailable."""
    pass


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded."""
    pass


class ProviderAuthError(ProviderError):
    """Provider authentication failed."""
    pass


class DataValidationError(DeFiPipelineError):
    """Data validation failed."""
    pass


class DatabaseError(DeFiPipelineError):
    """Database operation failed."""
    pass


class CacheError(DeFiPipelineError):
    """Cache operation failed."""
    pass


class ConfigurationError(DeFiPipelineError):
    """Configuration error."""
    pass


class MetricNotFoundError(DeFiPipelineError):
    """Requested metric not found."""
    pass


class InvalidRequestError(DeFiPipelineError):
    """Invalid API request."""
    pass


@dataclass
class ErrorResponse:
    """Structured error response for API endpoints."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    status_code: int = 500

    def to_dict(self) -> Dict[str, Any]:
        """Convert error response to dictionary."""
        result = {
            "error": self.error,
            "message": self.message,
            "status_code": self.status_code
        }
        if self.details:
            result["details"] = self.details
        return result


def create_error_response(
    exception: Exception,
    status_code: Optional[int] = None
) -> ErrorResponse:
    """
    Create an ErrorResponse from an exception.

    Args:
        exception: The exception to convert
        status_code: Optional status code override

    Returns:
        ErrorResponse: Structured error response
    """

    # Map exception types to status codes and error types
    error_mappings = {
        DataCollectionError: ("data_collection_error", 503),
        ProviderError: ("provider_error", 502),
        ProviderUnavailableError: ("provider_unavailable", 503),
        ProviderRateLimitError: ("rate_limit_exceeded", 429),
        ProviderAuthError: ("authentication_error", 401),
        DataValidationError: ("validation_error", 422),
        DatabaseError: ("database_error", 500),
        CacheError: ("cache_error", 500),
        ConfigurationError: ("configuration_error", 500),
        MetricNotFoundError: ("metric_not_found", 404),
        InvalidRequestError: ("invalid_request", 400),
    }

    error_type, default_status = error_mappings.get(
        type(exception),
        ("internal_error", 500)
    )

    # Use provided status code or default
    status_code = status_code or default_status

    # Extract details if available
    details = None
    if hasattr(exception, 'details') and exception.details:
        details = exception.details

    return ErrorResponse(
        error=error_type,
        message=str(exception),
        details=details,
        status_code=status_code
    )


def handle_exceptions(func):
    """
    Decorator to handle exceptions and convert them to ErrorResponse.

    Usage:
        @handle_exceptions
        async def my_endpoint():
            # Your code here
            pass
    """
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_response = create_error_response(e)
            return error_response

    return wrapper
