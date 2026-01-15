"""Provider registry for auto-discovery and management"""

from typing import Dict, Type, List
import logging
from src.core.interfaces import DataProviderInterface

logger = logging.getLogger('backfill_system.providers')


class ProviderRegistry:
    """
    Registry for data provider implementations.
    Supports auto-discovery and dynamic registration.
    """
    
    _providers: Dict[str, Type[DataProviderInterface]] = {}
    
    @classmethod
    def register(cls, provider_class: Type[DataProviderInterface]) -> None:
        """
        Register a provider implementation.
        
        Args:
            provider_class: Provider class that implements DataProviderInterface
            
        Raises:
            TypeError: If provider_class doesn't implement DataProviderInterface
        """
        # Validate that it's a DataProviderInterface implementation
        if not issubclass(provider_class, DataProviderInterface):
            raise TypeError(
                f"{provider_class.__name__} must implement DataProviderInterface"
            )
        
        # Get provider name from instance
        try:
            provider_instance = provider_class()
            provider_name = provider_instance.provider_name
        except Exception as e:
            logger.error(f"Failed to instantiate provider {provider_class.__name__}: {e}")
            raise
        
        # Register
        cls._providers[provider_name] = provider_class
        logger.info(f"Registered provider: {provider_name} ({provider_class.__name__})")
    
    @classmethod
    def get_provider(cls, provider_name: str) -> Type[DataProviderInterface]:
        """
        Get provider class by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider class
            
        Raises:
            ValueError: If provider not found
        """
        if provider_name not in cls._providers:
            available = ', '.join(cls._providers.keys()) if cls._providers else 'none'
            raise ValueError(
                f"Unknown provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
        
        return cls._providers[provider_name]
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider names.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def is_registered(cls, provider_name: str) -> bool:
        """
        Check if a provider is registered.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if registered, False otherwise
        """
        return provider_name in cls._providers
    
    @classmethod
    def auto_discover(cls) -> None:
        """
        Auto-discover and import all providers.
        Searches src/providers/ directory for provider modules.
        Each provider module should auto-register on import.
        """
        import importlib
        import pkgutil
        import src.providers as providers_package
        
        logger.info("Starting provider auto-discovery...")
        
        # Iterate through all modules in src/providers/
        for importer, modname, ispkg in pkgutil.iter_modules(providers_package.__path__):
            # Skip non-package modules and special modules
            if not ispkg or modname in ['__pycache__']:
                continue
            
            # Skip the registry itself
            if modname == 'registry':
                continue
            
            try:
                # Import the provider module (triggers auto-registration)
                full_module_name = f'src.providers.{modname}'
                importlib.import_module(full_module_name)
                logger.info(f"Discovered and loaded provider module: {modname}")
                
            except Exception as e:
                logger.warning(f"Failed to load provider module '{modname}': {e}")
        
        logger.info(
            f"Provider auto-discovery complete. "
            f"Registered providers: {', '.join(cls.list_providers())}"
        )
    
    @classmethod
    def unregister(cls, provider_name: str) -> None:
        """
        Unregister a provider (mainly for testing).
        
        Args:
            provider_name: Name of the provider to unregister
        """
        if provider_name in cls._providers:
            del cls._providers[provider_name]
            logger.info(f"Unregistered provider: {provider_name}")
    
    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered providers (mainly for testing).
        """
        cls._providers.clear()
        logger.info("Provider registry cleared")
    
    @classmethod
    def get_provider_info(cls) -> Dict[str, Dict]:
        """
        Get information about all registered providers.
        
        Returns:
            Dictionary mapping provider names to their info
        """
        info = {}
        
        for provider_name, provider_class in cls._providers.items():
            info[provider_name] = {
                'name': provider_name,
                'class': provider_class.__name__,
                'module': provider_class.__module__
            }
        
        return info

