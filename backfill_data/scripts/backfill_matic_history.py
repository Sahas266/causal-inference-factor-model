"""
Backfill MATIC price history (pre-POL migration, 2021-01-01 to 2023-10-24).
Stores as asset='matic' in asset_metrics.

Strategy (in order):
  1. CoinGecko Pro API  (x-cg-pro-api-key, pro-api.coingecko.com)
  2. CoinGecko free     (api.coingecko.com, days=max, no key)
  3. Artemis            (1-month chunks to avoid 504s)
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import requests
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
CG_API_KEY   = os.environ.get("COINGECKO_API_KEY", "")
ARTEMIS_KEY  = os.environ.get("ARTEMIS_API_KEY", "")

COIN_ID = "matic-network"
START   = datetime(2021, 1, 1, tzinfo=timezone.utc)
END     = datetime(2023, 10, 24, tzinfo=timezone.utc)


# ── helpers ────────────────────────────────────────────────────────────────

def upsert_batch(supabase, rows, batch_size=500):
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        supabase.table("asset_metrics").upsert(
            batch, on_conflict="provider,asset,metric,time"
        ).execute()
        total += len(batch)
        logger.info("  upserted %d / %d rows", total, len(rows))
    return total


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        k = (r["provider"], r["asset"], r["metric"], r["time"][:10])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def to_asset_metrics_rows(prices, market_caps, total_volumes, provider, priority):
    rows = []
    for series, metric in [(prices, "price"), (market_caps, "mc"), (total_volumes, "volume_24h")]:
        for ts_ms, val in series:
            if val is None or val == 0:
                continue
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            rows.append({
                "provider": provider,
                "asset": "matic",
                "metric": metric,
                "time": dt.strftime("%Y-%m-%dT00:00:00+00:00"),
                "value": float(val),
                "provider_priority": priority,
                "frequency": "1d",
            })
    return rows


# ── strategy 1: CoinGecko Pro ──────────────────────────────────────────────

def try_coingecko_pro():
    if not CG_API_KEY:
        return []
    logger.info("Trying CoinGecko Pro API...")
    base    = "https://pro-api.coingecko.com/api/v3"
    headers = {"x-cg-pro-api-key": CG_API_KEY}
    url     = f"{base}/coins/{COIN_ID}/market_chart/range"
    params  = {"vs_currency": "usd",
               "from": int(START.timestamp()),
               "to":   int(END.timestamp())}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        rows = to_asset_metrics_rows(
            data.get("prices", []),
            data.get("market_caps", []),
            data.get("total_volumes", []),
            "coingecko", 3,
        )
        logger.info("CoinGecko Pro: %d rows", len(rows))
        return rows
    except requests.HTTPError as e:
        logger.warning("CoinGecko Pro failed: %s", e)
        return []


# ── strategy 2: CoinGecko free (days=max) ─────────────────────────────────

def try_coingecko_free():
    logger.info("Trying CoinGecko free API (days=max)...")
    base   = "https://api.coingecko.com/api/v3"
    headers = {"x-cg-demo-api-key": CG_API_KEY} if CG_API_KEY else {}
    url    = f"{base}/coins/{COIN_ID}/market_chart"
    params = {"vs_currency": "usd", "days": "max", "interval": "daily"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        rows = to_asset_metrics_rows(
            data.get("prices", []),
            data.get("market_caps", []),
            data.get("total_volumes", []),
            "coingecko", 3,
        )
        # filter to pre-migration only
        cutoff = END.strftime("%Y-%m-%d")
        rows = [r for r in rows if r["time"][:10] <= cutoff]
        logger.info("CoinGecko free: %d rows (after date filter)", len(rows))
        return rows
    except requests.HTTPError as e:
        logger.warning("CoinGecko free failed: %s", e)
        return []


# ── strategy 3: Artemis (1-month chunks) ──────────────────────────────────

def try_artemis():
    if not ARTEMIS_KEY:
        return []
    logger.info("Trying Artemis 'matic' (1-month chunks)...")
    base   = "https://data-svc.artemisxyz.com"
    rows   = []
    cursor = START
    while cursor < END:
        chunk_end = min(cursor + timedelta(days=30), END)
        cs = cursor.strftime("%Y-%m-%d")
        ce = chunk_end.strftime("%Y-%m-%d")
        for metric in ["price", "mc", "fdmc", "24h_volume"]:
            url    = f"{base}/data/api/{metric}/"
            params = {"APIKey": ARTEMIS_KEY, "symbols": "matic",
                      "startDate": cs, "endDate": ce}
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                points = (data.get("data", {})
                              .get("symbols", {})
                              .get("matic", {})
                              .get(metric, []))
                m_name = "volume_24h" if metric == "24h_volume" else metric
                for p in points:
                    if p.get("val") is None:
                        continue
                    rows.append({
                        "provider": "artemis",
                        "asset": "matic",
                        "metric": m_name,
                        "time": f"{p['date']}T00:00:00+00:00",
                        "value": float(p["val"]),
                        "provider_priority": 2,
                        "frequency": "1d",
                    })
            except Exception as e:
                logger.warning("  Artemis %s %s-%s: %s", metric, cs, ce, e)
            time.sleep(1)
        logger.info("  chunk %s–%s: %d rows so far", cs, ce, len(rows))
        cursor = chunk_end + timedelta(days=1)
    logger.info("Artemis total: %d rows", len(rows))
    return rows


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # CoinGecko free (demo key) works for full history via days=max
    # Pro key (non-CG- prefix) also tried first if available
    rows = try_coingecko_pro()
    if not rows:
        rows = try_coingecko_free()
    if not rows:
        rows = try_artemis()  # fallback: 1-month chunks

    if not rows:
        logger.error("All strategies failed. No data to upsert.")
        return

    rows = dedup(rows)
    dates   = sorted(set(r["time"][:10] for r in rows))
    metrics = sorted(set(r["metric"] for r in rows))
    logger.info("Final: %d rows | %d dates (%s → %s) | metrics: %s",
                len(rows), len(dates), dates[0], dates[-1], metrics)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    total    = upsert_batch(supabase, rows)
    logger.info("Done! Upserted %d MATIC rows.", total)


if __name__ == "__main__":
    main()
