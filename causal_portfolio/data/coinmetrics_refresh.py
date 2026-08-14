"""Direct Coin Metrics refresh for the frozen causal model inputs."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader

API_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
API_HOST = urlparse(API_URL).hostname
FACTOR_ASSETS = ("btc", "doge", "eth")


def _metric_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        asset = str(record.get("asset", "")).casefold()
        timestamp = record.get("time")
        if asset not in FACTOR_ASSETS or not timestamp:
            continue
        fee = record.get("FeeTotNtv")
        if fee not in (None, ""):
            value = float(fee)
            if not math.isfinite(value):
                raise ValueError(f"non-finite {asset} FeeTotNtv at {timestamp}")
            rows.append(
                {
                    "provider": "coinmetrics",
                    "provider_priority": 1,
                    "asset": asset,
                    "metric": "FeeTotNtv",
                    "time": timestamp,
                    "value": value,
                    "frequency": "1d",
                    "metadata": None,
                }
            )
        price = record.get("PriceUSD")
        if asset == "btc" and price not in (None, ""):
            value = float(price)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid BTC PriceUSD at {timestamp}: {price}")
            rows.append(
                {
                    "provider": "coinmetrics",
                    "provider_priority": 3,
                    "asset": asset,
                    "metric": "price",
                    "time": timestamp,
                    "value": value,
                    "frequency": "1d",
                    "metadata": None,
                }
            )
    return rows


def refresh_causal_sources(
    db_path: str | Path,
    *,
    start: str,
    end: str,
    session: requests.Session | None = None,
    timeout_seconds: float = 30.0,
) -> int:
    """Upsert exact daily causal inputs directly from Coin Metrics.

    This operational path intentionally bypasses the Supabase mirror: model
    availability must not depend on a stale or retired warehouse endpoint.
    ``end`` is inclusive and should name the latest fully completed UTC day.
    """
    if datetime.fromisoformat(start) > datetime.fromisoformat(end):
        raise ValueError(f"causal refresh start {start} is after end {end}")
    client = session or requests.Session()
    url: str | None = API_URL
    params = {
        "assets": ",".join(FACTOR_ASSETS),
        "metrics": "FeeTotNtv,PriceUSD",
        "frequency": "1d",
        "start_time": start,
        "end_time": end,
        "page_size": 10_000,
    }
    records: list[dict] = []
    while url:
        if urlparse(url).hostname != API_HOST:
            raise ValueError(f"unexpected Coin Metrics pagination host: {url}")
        response = client.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("data")
        if not isinstance(page, list):
            raise ValueError("Coin Metrics response has no data list")
        records.extend(page)
        next_url = payload.get("next_page_url")
        url = str(next_url) if next_url else None
        params = None

    rows = _metric_rows(records)
    if not rows:
        raise ValueError("Coin Metrics returned no causal source rows")
    end_date = datetime.fromisoformat(end).date()
    latest = {}
    for row in rows:
        key = (row["asset"], row["metric"])
        date = datetime.fromisoformat(
            str(row["time"]).replace("Z", "+00:00")
        ).date()
        latest[key] = max(latest.get(key, date), date)
    required = {(asset, "FeeTotNtv") for asset in FACTOR_ASSETS} | {
        ("btc", "price")
    }
    missing = sorted(required - set(latest))
    stale = sorted(
        key for key in required if key in latest and latest[key] != end_date
    )
    if missing or stale:
        raise ValueError(
            "Coin Metrics causal refresh incomplete: "
            f"missing={missing}, not_at_end={stale}, end={end}"
        )

    loader = DuckDBCPCMDataLoader(db_path=str(Path(db_path)), read_only=False)
    try:
        return loader.upsert_rows(rows)
    finally:
        loader.close()
