"""DefiLlama data provider implementation"""

from src.providers.registry import ProviderRegistry
from .provider import DefiLlamaProvider

ProviderRegistry.register(DefiLlamaProvider)
