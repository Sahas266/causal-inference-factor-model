"""Tests for core interfaces"""

import pytest
from datetime import datetime
from src.core.interfaces import (
    DataProviderInterface,
    RateLimiterInterface,
    ValidationResult,
    FetchResult
)


def test_validation_result():
    """Test ValidationResult dataclass"""
    result = ValidationResult(
        valid=True,
        adjusted_start_time=datetime(2024, 1, 1),
        adjusted_end_time=datetime(2024, 12, 31),
        metadata={'test': 'data'}
    )
    
    assert result.valid is True
    assert result.adjusted_start_time.year == 2024
    assert result.metadata['test'] == 'data'


def test_fetch_result():
    """Test FetchResult dataclass"""
    result = FetchResult(
        data=[{'key': 'value'}],
        next_cursor='abc123',
        has_more=True,
        metadata={'count': 1}
    )
    
    assert len(result.data) == 1
    assert result.has_more is True
    assert result.next_cursor == 'abc123'


def test_provider_interface_is_abstract():
    """Test that DataProviderInterface cannot be instantiated"""
    with pytest.raises(TypeError):
        DataProviderInterface()


def test_rate_limiter_interface_is_abstract():
    """Test that RateLimiterInterface cannot be instantiated"""
    with pytest.raises(TypeError):
        RateLimiterInterface()


class MockProvider(DataProviderInterface):
    """Mock provider for testing"""
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    def initialize(self, provider_config: dict) -> None:
        self.config = provider_config
    
    def validate_endpoint(self, endpoint_config: dict) -> ValidationResult:
        return ValidationResult(valid=True)
    
    def fetch_data_batch(self, endpoint_config, start_time, end_time, cursor=None) -> FetchResult:
        return FetchResult(data=[])
    
    def fetch_data_stream(self, endpoint_config, start_time, end_time):
        yield []
    
    def get_rate_limiter(self) -> RateLimiterInterface:
        return None
    
    def transform_to_standard_schema(self, raw_data, endpoint_config, schema_type) -> list:
        return []
    
    def handle_error(self, error, context) -> dict:
        return {'retry': False}


def test_mock_provider_instantiation():
    """Test that mock provider can be instantiated"""
    provider = MockProvider()
    assert provider.provider_name == "mock"
    
    provider.initialize({'test': 'config'})
    assert provider.config['test'] == 'config'

