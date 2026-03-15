"""FRED (Federal Reserve Economic Data) provider implementation"""

from src.providers.registry import ProviderRegistry
from .provider import FredProvider

ProviderRegistry.register(FredProvider)
