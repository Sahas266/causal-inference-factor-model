"""Tests for the DuckDB local loader.

Uses tempfile-backed DuckDB instances seeded with synthetic data — no network
or Supabase dependency. Verifies:
  - schema creation
  - asset_metrics_best view picks lowest provider_priority
  - load_panel pivots correctly with daily resampling
  - load_returns computes log returns with correct column naming + fallback
  - load_macro pivots, ffills, lowercases columns
  - get_loader factory routes by CPCM_LOCAL_DB
  - PK upsert behavior (INSERT OR REPLACE)
  - empty-result handling
  - cache round-trip
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.data.duckdb_loader import (
    DuckDBCPCMDataLoader,
    ensure_schema,
)


# ── fixtures ────────────────────────────────────────────────────────


def _row(provider, priority, asset, metric, day, value, freq="1d"):
    return {
        "provider": provider,
        "provider_priority": priority,
        "asset": asset,
        "metric": metric,
        "time": datetime(2024, 1, day, tzinfo=timezone.utc),
        "value": float(value),
        "frequency": freq,
    }


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Yield a writable DuckDB loader on a fresh temp file."""
    db = tmp_path / "test.duckdb"
    loader = DuckDBCPCMDataLoader(db_path=str(db), read_only=False)
    yield loader
    loader.close()


@pytest.fixture
def seeded_db(tmp_path: Path):
    """DuckDB with two providers per metric (priority 1 wins) for btc/eth."""
    db = tmp_path / "seeded.duckdb"
    loader = DuckDBCPCMDataLoader(db_path=str(db), read_only=False)

    rows = []
    for day in range(1, 11):
        # btc: provider A (priority 1) wins over provider B (priority 5)
        rows.append(_row("provA", 1, "btc", "PriceUSD", day, 50000 + day * 100))
        rows.append(_row("provB", 5, "btc", "PriceUSD", day, 99999.0))  # should lose
        # eth: only one provider
        rows.append(_row("provA", 1, "eth", "PriceUSD", day, 3000 + day * 10))
        # macro
        rows.append(_row("fred", 1, "macro", "DFF", day, 5.0 + day * 0.01))
        rows.append(_row("fred", 1, "macro", "VIXCLS", day, 15.0 + day * 0.1))

    loader.upsert_rows(rows)
    yield loader
    loader.close()


# ── schema + view ───────────────────────────────────────────────────


def test_ensure_schema_creates_table_and_view(tmp_db):
    tables = {r[0] for r in tmp_db._con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()}
    assert "asset_metrics" in tables
    assert "asset_metrics_best" in tables


def test_view_picks_lowest_priority(seeded_db):
    """asset_metrics_best should yield only the priority=1 provider for btc."""
    df = seeded_db._con.execute(
        "SELECT provider, value FROM asset_metrics_best "
        "WHERE asset='btc' AND metric='PriceUSD' ORDER BY time"
    ).fetchdf()
    assert len(df) == 10  # one row per day, not two
    assert (df["provider"] == "provA").all()
    # provB's 99999 sentinel must never appear
    assert (df["value"] != 99999.0).all()


def test_view_keeps_unique_provider(seeded_db):
    df = seeded_db._con.execute(
        "SELECT COUNT(*) FROM asset_metrics_best WHERE asset='eth'"
    ).fetchone()
    assert df[0] == 10


# ── load_panel ──────────────────────────────────────────────────────


def test_load_panel_pivots_wide(seeded_db):
    df = seeded_db.load_panel(
        ["btc", "eth"], ["PriceUSD"],
        "2024-01-01", "2024-01-15",
        use_cache=False,
    )
    assert not df.empty
    assert "btc_PriceUSD" in df.columns
    assert "eth_PriceUSD" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    # 10 days seeded
    assert len(df) == 10
    # Sanity check on first BTC value (priority 1 = 50100, not 99999)
    assert df["btc_PriceUSD"].iloc[0] == pytest.approx(50100.0)


def test_load_panel_date_range_filter(seeded_db):
    df = seeded_db.load_panel(
        ["btc"], ["PriceUSD"],
        "2024-01-03", "2024-01-07",
        use_cache=False,
    )
    assert len(df) == 5  # days 3-7 inclusive


def test_load_panel_empty_when_no_match(seeded_db):
    df = seeded_db.load_panel(
        ["btc"], ["NonexistentMetric"],
        "2024-01-01", "2024-01-31",
        use_cache=False,
    )
    assert df.empty


def test_load_panel_unknown_asset(seeded_db):
    df = seeded_db.load_panel(
        ["doge"], ["PriceUSD"],
        "2024-01-01", "2024-01-31",
        use_cache=False,
    )
    assert df.empty


# ── load_returns ────────────────────────────────────────────────────


def test_load_returns_computes_log_returns(seeded_db):
    rets = seeded_db.load_returns(
        ["btc", "eth"],
        "2024-01-01", "2024-01-31",
        use_cache=False,
    )
    assert "btc_return" in rets.columns
    assert "eth_return" in rets.columns
    # First row dropped (NaN from shift)
    assert len(rets) == 9
    # Manual check: log(50200/50100) for day-2 BTC
    expected = np.log(50200.0 / 50100.0)
    assert rets["btc_return"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_load_returns_falls_back_to_lowercase_price(tmp_path: Path):
    """If PriceUSD is absent, should retry with 'price'."""
    loader = DuckDBCPCMDataLoader(db_path=str(tmp_path / "fallback.duckdb"), read_only=False)
    rows = [_row("provA", 1, "btc", "price", d, 100 + d) for d in range(1, 6)]
    loader.upsert_rows(rows)
    rets = loader.load_returns(["btc"], "2024-01-01", "2024-01-31", use_cache=False)
    loader.close()
    assert "btc_return" in rets.columns
    assert len(rets) == 4


def test_load_returns_empty_when_no_price(tmp_db):
    rets = tmp_db.load_returns(["btc"], "2024-01-01", "2024-01-31", use_cache=False)
    assert rets.empty


# ── load_macro ──────────────────────────────────────────────────────


def test_load_macro_pivots_and_lowercases(seeded_db):
    macro = seeded_db.load_macro(
        ["DFF", "VIXCLS"],
        "2024-01-01", "2024-01-31",
        use_cache=False,
    )
    assert "dff" in macro.columns
    assert "vixcls" in macro.columns
    # Resampled to daily — covers full range, ffilled
    assert macro["dff"].iloc[0] == pytest.approx(5.01)


def test_load_macro_empty(tmp_db):
    macro = tmp_db.load_macro(["DFF"], "2024-01-01", "2024-01-31", use_cache=False)
    assert macro.empty


# ── upsert behavior ─────────────────────────────────────────────────


def test_upsert_replaces_on_pk_collision(tmp_db):
    """INSERT OR REPLACE: same PK → second value wins."""
    tmp_db.upsert_rows([_row("provA", 1, "btc", "PriceUSD", 1, 100.0)])
    tmp_db.upsert_rows([_row("provA", 1, "btc", "PriceUSD", 1, 200.0)])
    val = tmp_db._con.execute(
        "SELECT value FROM asset_metrics WHERE asset='btc'"
    ).fetchone()
    assert val[0] == 200.0


def test_upsert_empty_is_noop(tmp_db):
    assert tmp_db.upsert_rows([]) == 0


# ── cache round-trip ────────────────────────────────────────────────


def test_panel_cache_round_trip(seeded_db, monkeypatch, tmp_path):
    """Second call with use_cache=True returns the cached frame."""
    from causal_portfolio.data import duckdb_loader as dl

    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()

    df1 = seeded_db.load_panel(["btc"], ["PriceUSD"], "2024-01-01", "2024-01-31", use_cache=True)
    # Wipe DB → second call must come from cache
    seeded_db._con.execute("DELETE FROM asset_metrics")
    df2 = seeded_db.load_panel(["btc"], ["PriceUSD"], "2024-01-01", "2024-01-31", use_cache=True)
    # check_freq=False because parquet round-trip drops DatetimeIndex.freq
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)


# ── error handling ──────────────────────────────────────────────────


def test_missing_db_file_raises_in_readonly(tmp_path):
    with pytest.raises(FileNotFoundError):
        DuckDBCPCMDataLoader(db_path=str(tmp_path / "nope.duckdb"), read_only=True)


def test_no_path_no_env_raises(monkeypatch):
    monkeypatch.delenv("CPCM_LOCAL_DB", raising=False)
    with pytest.raises(ValueError, match="CPCM_LOCAL_DB"):
        DuckDBCPCMDataLoader()


# ── factory ────────────────────────────────────────────────────────


def test_get_loader_routes_to_duckdb(tmp_path, monkeypatch):
    db = tmp_path / "factory.duckdb"
    # Pre-create with schema
    DuckDBCPCMDataLoader(db_path=str(db), read_only=False).close()

    monkeypatch.setenv("CPCM_LOCAL_DB", str(db))
    from causal_portfolio.data import get_loader
    loader = get_loader()
    assert isinstance(loader, DuckDBCPCMDataLoader)
    loader.close()


def test_get_loader_routes_to_supabase_when_unset(monkeypatch):
    """Without CPCM_LOCAL_DB, factory must instantiate the Supabase loader, not DuckDB."""
    monkeypatch.delenv("CPCM_LOCAL_DB", raising=False)
    from causal_portfolio.data import get_loader
    from causal_portfolio.data.supabase_loader import CPCMDataLoader
    try:
        loader = get_loader()
    except (KeyError, FileNotFoundError):
        # No Supabase creds in test env — that's fine, it confirms we routed
        # to the Supabase branch rather than DuckDB.
        return
    assert isinstance(loader, CPCMDataLoader)
    assert not isinstance(loader, DuckDBCPCMDataLoader)
