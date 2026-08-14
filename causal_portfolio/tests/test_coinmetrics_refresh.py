from __future__ import annotations

from types import SimpleNamespace

import pytest

from causal_portfolio.data.coinmetrics_refresh import refresh_causal_sources
from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _record(asset: str, date: str, fee: float, price: float = 1.0) -> dict:
    return {
        "asset": asset,
        "time": f"{date}T00:00:00.000000000Z",
        "FeeTotNtv": str(fee),
        "PriceUSD": str(price),
    }


def test_direct_refresh_upserts_exact_causal_inputs(tmp_path):
    session = SimpleNamespace()
    session.get = lambda *_args, **_kwargs: _Response(
        {
            "data": [
                _record("btc", "2026-08-11", 2.8, 63_500.0),
                _record("doge", "2026-08-11", 8_800.0, 0.07),
                _record("eth", "2026-08-11", 118.0, 1_880.0),
            ]
        }
    )
    db = tmp_path / "causal.duckdb"

    count = refresh_causal_sources(
        db,
        start="2026-08-11",
        end="2026-08-11",
        session=session,
    )

    assert count == 4
    loader = DuckDBCPCMDataLoader(str(db), read_only=True)
    try:
        rows = loader._con.execute(
            "SELECT provider, provider_priority, asset, metric, value "
            "FROM asset_metrics ORDER BY asset, metric"
        ).fetchall()
    finally:
        loader.close()
    assert rows == [
        ("coinmetrics", 1, "btc", "FeeTotNtv", 2.8),
        ("coinmetrics", 3, "btc", "price", 63_500.0),
        ("coinmetrics", 1, "doge", "FeeTotNtv", 8_800.0),
        ("coinmetrics", 1, "eth", "FeeTotNtv", 118.0),
    ]


def _session_with_latest(dates: dict[str, str]):
    session = SimpleNamespace()
    session.get = lambda *_args, **_kwargs: _Response(
        {"data": [
            _record("btc", dates["btc"], 2.8, 63_500.0),
            _record("doge", dates["doge"], 8_800.0, 0.07),
            _record("eth", dates["eth"], 118.0, 1_880.0),
        ]}
    )
    return session


def test_one_day_provider_lag_is_tolerated(tmp_path):
    """Coin Metrics does not publish a UTC day the moment it closes.

    Requiring data exactly at `end` failed every run launched shortly after
    midnight UTC and threw away the rows it had already fetched.
    """
    session = _session_with_latest(
        {"btc": "2026-08-11", "doge": "2026-08-10", "eth": "2026-08-11"}
    )

    written = refresh_causal_sources(
        tmp_path / "causal.duckdb",
        start="2026-08-10",
        end="2026-08-11",
        session=session,
    )

    assert written > 0


def test_a_series_further_behind_than_the_lag_still_fails(tmp_path):
    """Real staleness must remain a hard failure, not be waved through."""
    session = _session_with_latest(
        {"btc": "2026-08-11", "doge": "2026-08-08", "eth": "2026-08-11"}
    )

    with pytest.raises(ValueError, match="behind"):
        refresh_causal_sources(
            tmp_path / "causal.duckdb",
            start="2026-08-08",
            end="2026-08-11",
            session=session,
        )


def test_zero_lag_restores_the_strict_at_end_requirement(tmp_path):
    session = _session_with_latest(
        {"btc": "2026-08-11", "doge": "2026-08-10", "eth": "2026-08-11"}
    )

    with pytest.raises(ValueError, match="behind"):
        refresh_causal_sources(
            tmp_path / "causal.duckdb",
            start="2026-08-10",
            end="2026-08-11",
            session=session,
            max_lag_days=0,
        )
