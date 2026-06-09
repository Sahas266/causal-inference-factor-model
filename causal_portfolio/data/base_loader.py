"""Shared CPCM loader behavior (pivoting, returns, caching).

Backends (Supabase REST, local DuckDB) implement the two long-format fetch
hooks; everything else — wide pivoting, daily resampling, timezone
normalization, price→return conversion, parquet caching — lives here so the
two loaders cannot drift apart.

Conventions enforced here:
- All DatetimeIndexes are tz-naive UTC (timestamps normalized to UTC, then
  tz dropped) so daily resampling is deterministic on any machine and frames
  from either backend share an index dtype.
- ``load_returns`` yields SIMPLE returns by default (p_t/p_{t-1} - 1) — the
  basis every portfolio aggregation in this repo assumes (w·r daily sums,
  cumprod(1+r) compounding). Pass ``kind="log"`` for log returns.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("cpcm.data")

CACHE_DIR = Path(__file__).parent / "cache"

# Bump to invalidate all on-disk parquet caches when load semantics change.
# v2: simple returns by default; tz-naive UTC index from both backends.
CACHE_VERSION = 2


class BaseCPCMDataLoader(ABC):
    """Shared load_panel / load_prices / load_returns / load_macro logic."""

    cache_tag: str = ""  # distinguishes backends in cache keys

    # ── backend hooks ───────────────────────────────────────────────

    @abstractmethod
    def _fetch_panel_long(
        self, assets: list[str], metrics: list[str], start: str, end: str,
    ) -> pd.DataFrame:
        """Long-format rows with columns (asset, metric, time, value)."""

    @abstractmethod
    def _fetch_macro_long(
        self, series_ids: list[str], start: str, end: str,
    ) -> pd.DataFrame:
        """Long-format macro rows (asset='macro') with columns (metric, time, value)."""

    # ── public API ──────────────────────────────────────────────────

    def load_panel(
        self,
        assets: list[str],
        metrics: list[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Query asset_metrics_best and pivot to wide format.

        Returns DataFrame with DatetimeIndex and columns ``{asset}_{metric}``.
        """
        cache_key = self._cache_key("panel", assets, metrics, start, end)
        if use_cache and (cached := self._read_cache(cache_key)) is not None:
            return cached

        df = self._pivot(self._fetch_panel_long(assets, metrics, start, end))
        if use_cache and not df.empty:
            self._write_cache(cache_key, df)
        return df

    def load_prices(
        self,
        assets: list[str],
        start: str,
        end: str,
        price_metric: str = "PriceUSD",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Daily price levels, one column per asset (PriceUSD → 'price' fallback)."""
        panel = self.load_panel(assets, [price_metric], start, end, use_cache=use_cache)
        if panel.empty and price_metric == "PriceUSD":
            panel = self.load_panel(assets, ["price"], start, end, use_cache=use_cache)
        if panel.empty:
            return panel
        panel = panel.copy()
        panel.columns = [c.rsplit("_", 1)[0] for c in panel.columns]
        return panel

    def load_returns(
        self,
        assets: list[str],
        start: str,
        end: str,
        price_metric: str = "PriceUSD",
        kind: str = "simple",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Daily returns with ``{asset}_return`` columns.

        kind="simple" (default): p_t/p_{t-1} - 1, suitable for portfolio
        aggregation (w·r) and compounding via cumprod(1+r).
        kind="log": ln(p_t/p_{t-1}), for additive/statistical work.
        """
        prices = self.load_prices(
            assets, start, end, price_metric=price_metric, use_cache=use_cache)
        if prices.empty:
            return pd.DataFrame()
        if kind == "simple":
            returns = prices / prices.shift(1) - 1.0
        elif kind == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"unknown return kind: {kind!r} (use 'simple' or 'log')")
        returns = returns.dropna(how="all")
        returns.columns = [f"{c}_return" for c in returns.columns]
        return returns

    def load_macro(
        self,
        series_ids: list[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load FRED macro series (asset='macro'), lowercase columns, ffilled."""
        cache_key = self._cache_key("macro", ["macro"], series_ids, start, end)
        if use_cache and (cached := self._read_cache(cache_key)) is not None:
            return cached

        long = self._fetch_macro_long(series_ids, start, end)
        if long.empty:
            return pd.DataFrame()

        df = long.copy()
        df["time"] = self._utc_naive(df["time"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.pivot_table(index="time", columns="metric", values="value")
        df = df.sort_index().resample("D").last().ffill()
        df.columns = [c.lower() for c in df.columns]

        if use_cache and not df.empty:
            self._write_cache(cache_key, df)
        return df

    # ── internals ───────────────────────────────────────────────────

    @staticmethod
    def _utc_naive(times) -> pd.Series:
        """Normalize timestamps to UTC, then drop tz (deterministic resampling)."""
        return pd.to_datetime(times, utc=True).dt.tz_convert(None)

    def _pivot(self, long: pd.DataFrame) -> pd.DataFrame:
        """Convert long-format (asset, metric, time, value) rows to wide daily."""
        if long.empty:
            return pd.DataFrame()
        df = long.copy()
        df["time"] = self._utc_naive(df["time"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["col"] = df["asset"] + "_" + df["metric"]
        df = df.pivot_table(index="time", columns="col", values="value")
        return df.sort_index().resample("D").last()

    def _cache_key(self, prefix: str, *parts) -> str:
        raw = f"v{CACHE_VERSION}:{self.cache_tag}:{prefix}:{parts}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _read_cache(key: str) -> Optional[pd.DataFrame]:
        path = CACHE_DIR / f"{key}.parquet"
        if path.exists():
            logger.debug(f"Cache hit: {path}")
            return pd.read_parquet(path)
        return None

    @staticmethod
    def _write_cache(key: str, df: pd.DataFrame) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{key}.parquet"
        df.to_parquet(path)
        logger.debug(f"Cached: {path}")
