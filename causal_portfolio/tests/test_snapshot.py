"""Regression tests for incremental Supabase-to-DuckDB snapshots."""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace

import pytest

from causal_portfolio.data import snapshot as snapshot_module
from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader


def _row(
    *,
    provider: str = "source",
    asset: str = "btc",
    metric: str = "price",
    time: str = "2024-01-01T00:00:00+00:00",
    value: float = 1.0,
    created_at: str = "2024-01-02T00:00:00+00:00",
    updated_at: str = "2024-01-02T00:00:00+00:00",
) -> dict:
    return {
        "provider": provider,
        "provider_priority": 1,
        "asset": asset,
        "metric": metric,
        "time": time,
        "value": value,
        "frequency": "1d",
        "metadata": None,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _as_datetime(value):
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class _FakeQuery:
    def __init__(self, client):
        self.client = client
        self.columns = ""
        self.filters = []
        self.orders = []
        self.bounds = (0, snapshot_module.PAGE_SIZE - 1)
        self.limit_count = None
        self.or_filter = None

    def select(self, columns):
        self.columns = columns
        return self

    def in_(self, field, values):
        self.filters.append(("in", field, values))
        return self

    def gte(self, field, value):
        self.filters.append(("gte", field, value))
        return self

    def lt(self, field, value):
        self.filters.append(("lt", field, value))
        return self

    def or_(self, expression):
        self.or_filter = expression
        return self

    def order(self, field):
        self.orders.append(field)
        return self

    def range(self, start, end):
        self.bounds = (start, end)
        return self

    def limit(self, count):
        self.limit_count = count
        return self

    def execute(self):
        call = {
            "columns": self.columns,
            "filters": list(self.filters),
            "orders": list(self.orders),
            "bounds": self.bounds,
            "limit": self.limit_count,
            "or_filter": self.or_filter,
        }
        self.client.calls.append(call)
        if self.client.fail_late and any(
            field == "updated_at" for _, field, _ in self.filters
        ):
            raise RuntimeError("late sweep unavailable")

        rows = list(self.client.rows)
        for operation, field, expected in self.filters:
            if operation == "in":
                rows = [row for row in rows if row[field] in expected]
                continue
            bound = _as_datetime(expected)
            if operation == "gte":
                rows = [row for row in rows if _as_datetime(row[field]) >= bound]
            else:
                rows = [row for row in rows if _as_datetime(row[field]) < bound]

        if self.or_filter:
            cursor_values = {}
            for field, operation, value in re.findall(
                r'(updated_at|provider|asset|metric|time)\.(eq|gt)\."((?:\\.|[^"])*)"',
                self.or_filter,
            ):
                if operation == "gt" and field not in cursor_values:
                    cursor_values[field] = value.replace('\\"', '"').replace(
                        "\\\\", "\\"
                    )
            cursor = tuple(
                _as_datetime(cursor_values[field])
                if field in {"updated_at", "time"}
                else cursor_values[field]
                for field in ("updated_at", "provider", "asset", "metric", "time")
            )
            rows = [
                row
                for row in rows
                if tuple(
                    _as_datetime(row[field])
                    if field in {"updated_at", "time"}
                    else row[field]
                    for field in ("updated_at", "provider", "asset", "metric", "time")
                )
                > cursor
            ]

        rows.sort(
            key=lambda row: tuple(
                _as_datetime(row[field])
                if field in {"time", "created_at", "updated_at"}
                else row[field]
                for field in self.orders
            )
        )
        start, end = self.bounds
        selected = self.columns.split(",")
        page = rows[start : end + 1]
        if self.limit_count is not None:
            page = page[: self.limit_count]
        data = [{field: row[field] for field in selected} for row in page]
        if self.client.after_execute is not None:
            self.client.after_execute(self.client, call)
        return SimpleNamespace(data=data)


class _FakeClient:
    def __init__(self, rows, *, fail_late=False, after_execute=None):
        self.rows = rows
        self.fail_late = fail_late
        self.after_execute = after_execute
        self.calls = []

    def table(self, name):
        assert name == "asset_metrics"
        return _FakeQuery(self)


def _install_supabase(monkeypatch, client):
    module = ModuleType("supabase")
    module.create_client = lambda _url, _key: client
    monkeypatch.setitem(sys.modules, "supabase", module)
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")


def _seed_resume_db(path, previous_sync):
    loader = DuckDBCPCMDataLoader(db_path=str(path), read_only=False)
    loader.upsert_rows(
        [
            _row(value=1.0),
            _row(
                asset="eth",
                time="2024-02-01T00:00:00+00:00",
                value=10.0,
            ),
        ]
    )
    snapshot_module._record_sync_time(loader, previous_sync)
    loader.close()


def test_resume_applies_correction_to_existing_historical_row(tmp_path, monkeypatch):
    previous_sync = datetime(2024, 6, 1, tzinfo=timezone.utc)
    db_path = tmp_path / "snapshot.duckdb"
    _seed_resume_db(db_path, previous_sync)
    corrected = _row(
        value=2.0,
        created_at="2024-01-02T00:00:00+00:00",
        updated_at="2024-06-02T00:00:00+00:00",
    )
    client = _FakeClient([corrected])
    _install_supabase(monkeypatch, client)

    assert snapshot_module.snapshot(str(db_path), end="2024-02-02") == 1

    loader = DuckDBCPCMDataLoader(db_path=str(db_path), read_only=False)
    value = loader._con.execute(
        "SELECT value FROM asset_metrics "
        "WHERE provider='source' AND asset='btc' AND metric='price'"
    ).fetchone()[0]
    checkpoint = snapshot_module._last_sync_time(loader)
    loader.close()
    assert value == 2.0

    late_call = next(
        call
        for call in client.calls
        if any(field == "updated_at" for _, field, _ in call["filters"])
    )
    assert "created_at,updated_at" in late_call["columns"]
    assert late_call["orders"] == [
        "updated_at",
        "provider",
        "asset",
        "metric",
        "time",
    ]
    cutoff = next(
        expected
        for operation, field, expected in late_call["filters"]
        if operation == "lt" and field == "updated_at"
    )
    assert checkpoint == cutoff


def test_failed_historical_sweep_does_not_advance_checkpoint(tmp_path, monkeypatch):
    previous_sync = datetime(2024, 6, 1, tzinfo=timezone.utc)
    db_path = tmp_path / "snapshot.duckdb"
    _seed_resume_db(db_path, previous_sync)
    client = _FakeClient([], fail_late=True)
    _install_supabase(monkeypatch, client)

    with pytest.raises(RuntimeError, match="late sweep unavailable"):
        snapshot_module.snapshot(str(db_path), end="2024-02-02")

    loader = DuckDBCPCMDataLoader(db_path=str(db_path), read_only=False)
    assert snapshot_module._last_sync_time(loader) == previous_sync.isoformat()
    loader.close()


def test_historical_sweep_paginates_total_order_without_boundary_loss(monkeypatch):
    monkeypatch.setattr(snapshot_module, "PAGE_SIZE", 2)
    changed_at = "2024-06-02T00:00:00+00:00"
    rows = [
        _row(provider="z", asset="btc", metric="price", updated_at=changed_at),
        _row(provider="a", asset="eth", metric="price", updated_at=changed_at),
        _row(provider="a", asset="btc", metric="volume", updated_at=changed_at),
        _row(provider="a", asset="btc", metric="price", updated_at=changed_at),
        _row(provider="b", asset="btc", metric="price", updated_at=changed_at),
    ]
    client = _FakeClient(rows)

    fetched = snapshot_module._fetch_backfilled(
        client,
        "2024-06-01T00:00:00+00:00",
        datetime(2024, 2, 1, tzinfo=timezone.utc),
        None,
        changed_before=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )

    assert [(row["provider"], row["asset"], row["metric"]) for row in fetched] == [
        ("a", "btc", "price"),
        ("a", "btc", "volume"),
        ("a", "eth", "price"),
        ("b", "btc", "price"),
        ("z", "btc", "price"),
    ]
    assert [call["limit"] for call in client.calls] == [2, 2, 2]
    assert client.calls[0]["or_filter"] is None
    assert all(call["or_filter"] for call in client.calls[1:])
    assert all(
        call["orders"] == ["updated_at", "provider", "asset", "metric", "time"]
        for call in client.calls
    )
    assert all(
        ("lt", "updated_at", "2024-07-01T00:00:00+00:00") in call["filters"]
        for call in client.calls
    )


def test_historical_sweep_keyset_survives_update_between_pages(monkeypatch):
    monkeypatch.setattr(snapshot_module, "PAGE_SIZE", 2)
    rows = [
        _row(provider="a", updated_at="2024-06-02T00:00:00+00:00"),
        _row(provider="b", updated_at="2024-06-03T00:00:00+00:00"),
        _row(provider="c", updated_at="2024-06-04T00:00:00+00:00"),
    ]

    def move_first_row_after_cutoff(client, _call):
        if len(client.calls) == 1:
            client.rows[0]["updated_at"] = "2024-08-01T00:00:00+00:00"

    client = _FakeClient(rows, after_execute=move_first_row_after_cutoff)
    fetched = snapshot_module._fetch_backfilled(
        client,
        "2024-06-01T00:00:00+00:00",
        datetime(2024, 2, 1, tzinfo=timezone.utc),
        None,
        changed_before=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )

    assert [row["provider"] for row in fetched] == ["a", "b", "c"]
    assert len(client.calls) == 2
    assert client.calls[1]["or_filter"] is not None
    assert all(call["bounds"][0] == 0 for call in client.calls)
