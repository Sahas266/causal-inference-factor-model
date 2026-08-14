"""Correctness regressions for local derived-metric continuations."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace

import duckdb

from scripts import compute_derived_extra_local as extra
from scripts import compute_derived_metrics_local as metrics
from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader


def _database(tmp_path):
    path = tmp_path / "metrics.duckdb"
    DuckDBCPCMDataLoader(str(path), read_only=False).close()
    return path


def _source_row(provider, day, value, *, hour=0, asset="btc", metric="price"):
    return {
        "provider": provider,
        "provider_priority": 3,
        "asset": asset,
        "metric": metric,
        "time": datetime.combine(day, datetime.min.time(), timezone.utc)
        + timedelta(hours=hour),
        "value": value,
        "frequency": "1d",
        "metadata": None,
    }


def _insert(path, rows):
    loader = DuckDBCPCMDataLoader(str(path), read_only=False)
    try:
        loader.upsert_rows(rows)
    finally:
        loader.close()


def test_metrics_price_source_is_deterministic_daily_and_complete(tmp_path):
    path = _database(tmp_path)
    today = datetime.now(timezone.utc).date()
    rows = []
    for offset in range(31):
        day = today - timedelta(days=31 - offset)
        rows.extend(
            [
                _source_row("coinmetrics", day, 100 + offset),
                _source_row("coingecko", day, 200 + offset),
                _source_row("artemis", day - timedelta(days=10), 300 + offset),
            ]
        )
    rows.append(_source_row("coinmetrics", today - timedelta(days=1), 999, hour=12))
    rows.append(_source_row("coingecko", today, 9999, hour=1))
    _insert(path, rows)

    module = SimpleNamespace(
        PRICE_SOURCES=[
            ("artemis", "price"),
            ("coinmetrics", "price"),
            ("coingecko", "price"),
        ],
        compute_realized_volatility=lambda _asset: [],
    )
    metrics.install_duckdb_io(module, str(path))

    selected, source = module.fetch_price_data("btc")

    assert source == "coinmetrics:price"
    assert len(selected) == 31
    assert len({row["time"][:10] for row in selected}) == 31
    assert all(row["time"].endswith("T00:00:00+00:00") for row in selected)
    assert selected[-1]["value"] == 999
    assert selected[-1]["time"][:10] == str(today - timedelta(days=1))


def test_metrics_realized_volatility_does_not_bridge_a_gap(tmp_path):
    path = _database(tmp_path)
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=45)
    gap = start + timedelta(days=20)
    _insert(
        path,
        [
            _source_row("coinmetrics", start + timedelta(days=offset), 100 + offset**2)
            for offset in range(45)
            if start + timedelta(days=offset) != gap
        ],
    )

    module = SimpleNamespace(PRICE_SOURCES=[("coinmetrics", "price")])

    def compute(asset):
        selected, source = module.fetch_price_data(asset)
        return [
            {
                "asset": asset,
                "metric": "realized_volatility_7d",
                "time": row["time"],
                "metadata": {"source": source, "window": 7},
            }
            for row in selected[7:]
        ]

    module.compute_realized_volatility = compute
    metrics.install_duckdb_io(module, str(path))

    days = {row["time"][:10] for row in module.compute_realized_volatility("btc")}

    assert str(gap + timedelta(days=3)) not in days
    assert str(gap + timedelta(days=8)) in days


def test_metrics_flush_is_append_only(tmp_path):
    path = _database(tmp_path)
    today = datetime.now(timezone.utc).date()
    old_day = today - timedelta(days=3)
    new_day = today - timedelta(days=2)
    existing = {
        **_source_row("derived", old_day, 1.0, metric="realized_volatility_7d"),
        "provider_priority": 99,
    }
    _insert(path, [existing])
    replacement = {**existing, "value": 999.0}
    added = {**existing, "time": f"{new_day}T00:00:00+00:00", "value": 2.0}

    assert metrics.flush([replacement, added], str(path)) == 1

    con = duckdb.connect(str(path), read_only=True)
    try:
        con.execute("SET TimeZone='UTC'")
        values = con.execute(
            "select time::date, value from asset_metrics where provider='derived' order by time"
        ).fetchall()
    finally:
        con.close()
    assert values == [(old_day, 1.0), (new_day, 2.0)]


def test_extra_series_uses_last_completed_utc_observation(tmp_path):
    path = _database(tmp_path)
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    _insert(
        path,
        [
            _source_row("coingecko", yesterday, 1.0),
            _source_row("coingecko", yesterday, 2.0, hour=12),
            _source_row("coingecko", today, 3.0, hour=1),
        ],
    )
    con = extra._connect(str(path), read_only=True)
    try:
        series = extra._series(con, "btc", "price", "coingecko")
    finally:
        con.close()

    assert series == [(str(yesterday), 2.0)]


def test_extra_windows_and_flows_do_not_bridge_calendar_gaps(monkeypatch):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc).date()
    gap = start + timedelta(days=20)
    series = [
        (str(start + timedelta(days=offset)), float(100 + offset**2))
        for offset in range(71)
        if start + timedelta(days=offset) != gap
    ]
    monkeypatch.setattr(extra, "_price_series", lambda _con, _asset: (series, "test"))
    monkeypatch.setattr(extra, "_series", lambda *_args: series)

    records = extra.log_returns_and_sharpe(None, "btc")
    flows = extra.net_flow(None, "btc", "stock", "flow")

    for record in records:
        window = 7 if record["metric"] == "log_return_7d" else 30
        end = datetime.fromisoformat(record["time"]).date()
        assert not (gap < end <= gap + timedelta(days=window))
    assert any(
        record["time"].startswith(str(gap + timedelta(days=31)))
        for record in records
        if record["metric"] == "sharpe_30d"
    )
    assert str(gap + timedelta(days=1)) not in {row["time"][:10] for row in flows}


def test_metrics_main_fails_closed(monkeypatch):
    base = ModuleType("compute_derived_metrics")
    base.compute_realized_volatility = lambda asset: (
        (_ for _ in ()).throw(RuntimeError("broken")) if asset == "eth" else [{}]
    )
    base.compute_dex_cex_volume_ratio = lambda: []
    monkeypatch.setitem(sys.modules, "compute_derived_metrics", base)
    monkeypatch.setattr(metrics, "install_duckdb_io", lambda _base, _path: [])
    writes = []
    monkeypatch.setattr(metrics, "flush", lambda records, _path: writes.extend(records))

    assert metrics.main(["--asset", "btc,eth"]) == 1
    assert writes == []


def test_extra_main_fails_closed(monkeypatch):
    class Connection:
        def close(self):
            pass

    monkeypatch.setattr(extra, "_connect", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(
        extra,
        "log_returns_and_sharpe",
        lambda _con, asset: (_ for _ in ()).throw(RuntimeError("broken"))
        if asset == "eth"
        else [{}],
    )
    monkeypatch.setattr(extra, "net_flow", lambda *_args: [])

    assert extra.main(["--asset", "btc,eth", "--dry-run"]) == 1
