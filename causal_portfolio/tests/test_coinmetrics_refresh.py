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


def test_direct_refresh_fails_when_one_source_is_not_at_requested_end(tmp_path):
    session = SimpleNamespace()
    session.get = lambda *_args, **_kwargs: _Response(
        {
            "data": [
                _record("btc", "2026-08-11", 2.8, 63_500.0),
                _record("doge", "2026-08-10", 8_800.0, 0.07),
                _record("eth", "2026-08-11", 118.0, 1_880.0),
            ]
        }
    )

    with pytest.raises(ValueError, match="not_at_end"):
        refresh_causal_sources(
            tmp_path / "causal.duckdb",
            start="2026-08-10",
            end="2026-08-11",
            session=session,
        )
