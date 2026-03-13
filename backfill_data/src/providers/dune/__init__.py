"""Dune Analytics data provider implementation"""

# Auto-register provider when imported
from src.providers.registry import ProviderRegistry
from .provider import DuneProvider

ProviderRegistry.register(DuneProvider)
