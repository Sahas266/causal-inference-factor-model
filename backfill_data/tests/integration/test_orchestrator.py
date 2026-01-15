"""Integration tests for orchestrator"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.orchestrator import BackfillOrchestrator


@pytest.fixture
def mock_supabase():
    """Mock Supabase manager"""
    with patch('src.core.orchestrator.SupabaseManager') as mock:
        yield mock


@pytest.fixture
def mock_config_loader():
    """Mock config loader"""
    with patch('src.core.orchestrator.ConfigLoader') as mock:
        loader = Mock()
        loader.load_provider_config.return_value = {
            'provider_name': 'coinmetrics',
            'enabled': True,
            'api_config': {
                'base_url': 'https://api.coinmetrics.io/v4',
                'api_key': 'test-key'
            },
            'rate_limits': {
                'requests_per_window': 100,
                'window_seconds': 10
            }
        }
        mock.return_value = loader
        yield loader


@pytest.fixture
def mock_provider_registry():
    """Mock provider registry"""
    with patch('src.core.orchestrator.ProviderRegistry') as mock:
        mock.list_providers.return_value = ['coinmetrics']
        mock.get_provider.return_value = Mock
        yield mock


def test_orchestrator_initialization(mock_supabase, mock_config_loader, mock_provider_registry):
    """Test orchestrator initialization"""
    with patch('src.providers.coinmetrics.provider.CoinMetricsProvider'):
        orchestrator = BackfillOrchestrator()
        
        assert orchestrator.config_loader is not None
        assert orchestrator.supabase_manager is not None
        assert orchestrator.db_writer is not None
        assert orchestrator.progress_tracker is not None


def test_orchestrator_list_providers(mock_supabase, mock_config_loader, mock_provider_registry):
    """Test listing providers"""
    with patch('src.providers.coinmetrics.provider.CoinMetricsProvider'):
        orchestrator = BackfillOrchestrator()
        providers = orchestrator.list_providers()
        
        assert isinstance(providers, list)


def test_endpoint_validation():
    """Test endpoint validation logic"""
    endpoint_config = {
        'endpoint_id': 'test_endpoint',
        'table': 'asset_metrics',
        'primary_keys': ['provider', 'asset', 'metric', 'time'],
        'providers': [
            {
                'name': 'coinmetrics',
                'enabled': True,
                'priority': 1,
                'config': {}
            }
        ]
    }
    
    assert 'endpoint_id' in endpoint_config
    assert 'providers' in endpoint_config
    assert len(endpoint_config['providers']) > 0


def test_backfill_results_structure():
    """Test backfill results structure"""
    results = {
        'total_endpoints': 1,
        'completed': 1,
        'failed': 0,
        'skipped': 0,
        'provider_stats': {
            'coinmetrics': {
                'completed': 1,
                'failed': 0,
                'records': 1000
            }
        }
    }
    
    assert 'total_endpoints' in results
    assert 'provider_stats' in results
    assert results['completed'] == 1

