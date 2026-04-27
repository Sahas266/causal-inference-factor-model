"""CPCM data loaders.

`get_loader()` is the recommended entry point. It returns DuckDBCPCMDataLoader
when CPCM_LOCAL_DB is set, otherwise falls back to the cloud Supabase loader.
"""

from __future__ import annotations

import os
from typing import Optional


def get_loader(env_path: Optional[str] = None):
    """Return the appropriate loader based on CPCM_LOCAL_DB env var."""
    if os.environ.get("CPCM_LOCAL_DB"):
        from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader
        return DuckDBCPCMDataLoader()
    from causal_portfolio.data.supabase_loader import CPCMDataLoader
    return CPCMDataLoader(env_path=env_path)
