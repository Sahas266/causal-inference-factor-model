"""Base provider class with common functionality"""

from typing import Dict, Optional
import logging
from src.core.interfaces import DataProviderInterface

logger = logging.getLogger('backfill_system.providers')


class BaseProvider(DataProviderInterface):
    """
    Base class for providers with common functionality.
    Providers can extend this instead of implementing DataProviderInterface directly.
    """
    
    def __init__(self):
        """Initialize base provider"""
        self.config: Optional[Dict] = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """
        Check that provider is initialized.
        
        Raises:
            RuntimeError: If provider not initialized
        """
        if not self._initialized:
            raise RuntimeError(
                f"{self.provider_name} provider not initialized. "
                "Call initialize() first."
            )
    
    def _validate_config(self, required_fields: list) -> None:
        """
        Validate that required config fields are present.
        
        Args:
            required_fields: List of required field paths (e.g., ['api_config.api_key'])
            
        Raises:
            ValueError: If required fields are missing
        """
        for field_path in required_fields:
            parts = field_path.split('.')
            current = self.config
            
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    raise ValueError(
                        f"Missing required config field: {field_path}"
                    )
                current = current[part]
    
    def get_config_value(self, key_path: str, default=None):
        """
        Get configuration value by dot-notation path.
        
        Args:
            key_path: Dot-notation path (e.g., 'api_config.base_url')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        parts = key_path.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        
        return current

