"""CPCM data loaders.

`get_loader()` is the recommended entry point. Resolution order:

1. `CPCM_FORCE_REMOTE=1` env var → use Supabase (escape hatch)
2. `CPCM_LOCAL_DB` env var → DuckDB at that path
3. Default local snapshot at `<repo>/causal_portfolio/data/cpcm_local.duckdb`
   if the file exists → DuckDB
4. Fall back to Supabase

This makes "local-first" the default once a snapshot exists, so reads don't
silently consume the Supabase quota. Create the local snapshot with:

    python -m causal_portfolio.data.snapshot
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cpcm.data")

DEFAULT_LOCAL_DB = Path(__file__).parent / "cpcm_local.duckdb"


def _resolve_local_db_path() -> Optional[str]:
    """Return a usable local DuckDB path, or None to fall back to Supabase."""
    if os.environ.get("CPCM_FORCE_REMOTE") == "1":
        return None
    explicit = os.environ.get("CPCM_LOCAL_DB")
    if explicit:
        return explicit
    if DEFAULT_LOCAL_DB.exists():
        return str(DEFAULT_LOCAL_DB)
    return None


def get_loader(env_path: Optional[str] = None):
    """Return the appropriate loader. Local DuckDB preferred when available."""
    local = _resolve_local_db_path()
    if local:
        from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader
        logger.info("Using local DuckDB loader: %s", local)
        return DuckDBCPCMDataLoader(db_path=local)

    from causal_portfolio.data.supabase_loader import CPCMDataLoader
    logger.warning(
        "No local DuckDB snapshot found — falling back to Supabase. "
        "To avoid burning the read quota, run: "
        "python -m causal_portfolio.data.snapshot"
    )
    return CPCMDataLoader(env_path=env_path)
