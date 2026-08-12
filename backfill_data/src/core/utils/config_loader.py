"""Configuration loading and validation utilities"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger('backfill_system.config')


def iter_endpoint_config_files(endpoints_dir: Path):
    """Yield active endpoint configs recursively in deterministic order.

    Directories whose name ends in ``_disabled`` are intentionally omitted
    from broad discovery. An operator can still load one of those files by
    specifying its path explicitly.
    """
    for config_file in sorted(endpoints_dir.rglob('*.json')):
        relative = config_file.relative_to(endpoints_dir)
        if any(part.lower().endswith('_disabled') for part in relative.parts[:-1]):
            continue
        yield config_file


class ConfigLoader:
    """Utility class for loading and validating configuration files"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_dir: Path to config directory. Defaults to ./config
        """
        if config_dir is None:
            # Default to config directory relative to project root
            config_dir = Path(__file__).parent.parent.parent.parent / 'config'
        
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        
        logger.info(f"Config loader initialized with directory: {self.config_dir}")
    
    def load_provider_config(self, provider_name: str) -> Dict:
        """
        Load configuration for a specific provider.
        
        Args:
            provider_name: Name of the provider (e.g., 'coinmetrics')
            
        Returns:
            Provider configuration dictionary
            
        Raises:
            FileNotFoundError: If provider config file doesn't exist
            ValueError: If config is invalid
        """
        config_file = self.config_dir / 'providers' / f'{provider_name}.json'
        
        if not config_file.exists():
            raise FileNotFoundError(f"Provider config not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['provider_name', 'enabled', 'api_config']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {config_file}")
        
        # Resolve API key from environment variable
        if 'api_key_env' in config['api_config']:
            env_var = config['api_config']['api_key_env']
            api_key = os.getenv(env_var)
            if not api_key:
                logger.warning(f"API key environment variable '{env_var}' not set for {provider_name}")
            config['api_config']['api_key'] = api_key
        
        logger.info(f"Loaded config for provider: {provider_name}")
        return config
    
    def load_endpoint_config(self, endpoint_file: str) -> Dict:
        """
        Load configuration for a specific endpoint.
        
        Args:
            endpoint_file: Name of endpoint config file (e.g., 'btc_metrics.json')
            
        Returns:
            Endpoint configuration dictionary
            
        Raises:
            FileNotFoundError: If endpoint config file doesn't exist
            ValueError: If config is invalid
        """
        # Handle both full path and just filename
        if os.path.isabs(endpoint_file):
            config_file = Path(endpoint_file)
        else:
            candidate = Path(endpoint_file)
            if candidate.exists():
                config_file = candidate
            else:
                config_file = self.config_dir / 'endpoints' / endpoint_file
        
        if not config_file.exists():
            raise FileNotFoundError(f"Endpoint config not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['endpoint_id', 'table', 'primary_keys', 'providers']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {config_file}")
        
        logger.info(f"Loaded config for endpoint: {config['endpoint_id']}")
        return config
    
    def load_all_endpoint_configs(self) -> List[Dict]:
        """
        Load all endpoint configurations from the endpoints directory.
        
        Returns:
            List of endpoint configuration dictionaries
        """
        endpoints_dir = self.config_dir / 'endpoints'
        if not endpoints_dir.exists():
            logger.warning(f"Endpoints directory not found: {endpoints_dir}")
            return []
        
        configs = []
        for config_file in iter_endpoint_config_files(endpoints_dir):
            try:
                config = self.load_endpoint_config(str(config_file))
                configs.append(config)
            except Exception as e:
                logger.error(f"Failed to load endpoint config {config_file}: {e}")
        
        logger.info(f"Loaded {len(configs)} endpoint configurations")
        return configs
    
    def list_available_providers(self) -> List[str]:
        """
        List all available provider configurations.
        
        Returns:
            List of provider names
        """
        providers_dir = self.config_dir / 'providers'
        if not providers_dir.exists():
            return []
        
        providers = [
            f.stem for f in providers_dir.glob('*.json')
        ]
        return providers
    
    def list_available_endpoints(self) -> List[str]:
        """
        List all available endpoint configurations.
        
        Returns:
            List of endpoint IDs
        """
        endpoints_dir = self.config_dir / 'endpoints'
        if not endpoints_dir.exists():
            return []
        
        endpoints = [
            str(f.relative_to(endpoints_dir).with_suffix('')).replace('\\', '/')
            for f in iter_endpoint_config_files(endpoints_dir)
        ]
        return endpoints


# Convenience function
def load_config() -> ConfigLoader:
    """Create and return a ConfigLoader instance"""
    return ConfigLoader()

