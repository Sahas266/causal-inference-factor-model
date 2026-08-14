"""Direct Coin Metrics refresh for the frozen causal model inputs."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
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
    max_lag_days: int = 1,
) -> int:
    """Upsert exact daily causal inputs directly from Coin Metrics.

    This operational path intentionally bypasses the Supabase mirror: model
    availability must not depend on a stale or retired warehouse endpoint.
    ``end`` is inclusive and should name the latest fully completed UTC day.

    ``max_lag_days`` tolerates the provider's own publication delay. Coin
    Metrics does not publish a UTC day the instant it closes, so requiring
    data exactly at ``end`` fails every run launched shortly after midnight
    UTC — and, worse, discards the rows it did fetch. A series that falls
    further behind than this is still a hard failure, because that is a real
    gap rather than normal lag. Whether the model may then trade on the
    result is decided separately by its own freshness gate.
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
    start_date = datetime.fromisoformat(start).date()
    end_date = datetime.fromisoformat(end).date()
    latest = {}
    observed = set()
    for row in rows:
        key = (row["asset"], row["metric"])
        date = datetime.fromisoformat(
            str(row["time"]).replace("Z", "+00:00")
        ).date()
        observed.add((*key, date))
        latest[key] = max(latest.get(key, date), date)
    required = {(asset, "FeeTotNtv") for asset in FACTOR_ASSETS} | {
        ("btc", "price")
    }
    missing = sorted(required - set(latest))
    earliest_allowed = end_date - timedelta(days=max(0, max_lag_days))
    expected_dates = (
        {
            start_date + timedelta(days=offset)
            for offset in range((earliest_allowed - start_date).days + 1)
        }
        if earliest_allowed >= start_date
        else set()
    )
    gaps = sorted(
        (asset, metric, date.isoformat())
        for asset, metric in required
        for date in expected_dates
        if (asset, metric, date) not in observed
    )
    stale = sorted(
        key for key in required
        if key in latest and latest[key] < earliest_allowed
    )
    if missing or stale or gaps:
        raise ValueError(
            "Coin Metrics causal refresh incomplete: "
            f"missing={missing}, behind={stale}, gaps={gaps[:8]}, end={end}, "
            f"max_lag_days={max_lag_days}"
        )

    loader = DuckDBCPCMDataLoader(db_path=str(Path(db_path)), read_only=False)
    try:
        return loader.upsert_rows(rows)
    finally:
        loader.close()
