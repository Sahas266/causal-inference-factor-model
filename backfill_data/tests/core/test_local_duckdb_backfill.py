"""Regression tests for the local DuckDB backfill command."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts import backfill_local_duckdb as local_backfill


class _Provider:
    def __init__(self, *, initialization_error: bool = False, fetch_error: bool = False):
        self.initialization_error = initialization_error
        self.fetch_error = fetch_error
        self.calls = []

    def initialize(self, _settings):
        if self.initialization_error:
            raise RuntimeError("initialization failed")

    def fetch_data_stream(self, config, start, end):
        self.calls.append((config, start, end))
        if self.fetch_error:
            raise RuntimeError("fetch failed")
        return iter([])


class _Loader:
    def __init__(self):
        self.closed = False

    def upsert_rows(self, rows):
        return len(rows)

    def close(self):
        self.closed = True


def _job(start: str = "2021-01-01"):
    endpoint = {
        "endpoint_id": "test_endpoint",
        "date_range": {"start": start},
    }
    provider_config = {"name": "test", "config": {"endpoint_type": "test"}}
    return "test", endpoint, provider_config


def _install_run_fakes(monkeypatch, provider, loader, jobs=None):
    monkeypatch.setattr(local_backfill.ProviderRegistry, "auto_discover", lambda: None)
    monkeypatch.setattr(
        local_backfill.ProviderRegistry,
        "get_provider",
        lambda _name: lambda: provider,
    )
    monkeypatch.setattr(
        local_backfill,
        "ConfigLoader",
        lambda _path: SimpleNamespace(load_provider_config=lambda _name: {}),
    )
    monkeypatch.setattr(local_backfill, "discover", lambda _path: jobs or [_job()])
    monkeypatch.setattr(local_backfill, "_local_loader", lambda _path: loader)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (None, datetime(2021, 1, 1, tzinfo=timezone.utc)),
        ("2026-08-01", datetime(2026, 8, 1, tzinfo=timezone.utc)),
    ],
)
def test_run_uses_endpoint_start_unless_explicitly_overridden(
    monkeypatch, tmp_path, override, expected
):
    provider = _Provider()
    loader = _Loader()
    _install_run_fakes(monkeypatch, provider, loader)

    assert local_backfill.run(
        {"test"},
        start=override,
        end="2026-08-14",
        db_path=None,
        config_dir=tmp_path,
        limit=None,
    ) == 0

    assert provider.calls[0][1] == expected
    assert loader.closed is True


def test_run_fails_when_no_provider_initializes(monkeypatch, tmp_path):
    provider = _Provider(initialization_error=True)
    loader = _Loader()
    _install_run_fakes(monkeypatch, provider, loader)

    with pytest.raises(RuntimeError, match="no provider could be initialized"):
        local_backfill.run(
            {"test"},
            start=None,
            end="2026-08-14",
            db_path=None,
            config_dir=tmp_path,
            limit=None,
        )

    assert loader.closed is False


def test_run_fails_after_an_endpoint_error_and_closes_loader(monkeypatch, tmp_path):
    provider = _Provider(fetch_error=True)
    loader = _Loader()
    _install_run_fakes(monkeypatch, provider, loader)

    with pytest.raises(RuntimeError, match=r"failed for 1 endpoint\(s\)"):
        local_backfill.run(
            {"test"},
            start=None,
            end="2026-08-14",
            db_path=None,
            config_dir=tmp_path,
            limit=None,
        )

    assert loader.closed is True


def test_main_returns_nonzero_when_run_fails(monkeypatch):
    monkeypatch.setattr(local_backfill, "discover", lambda _path: [_job()])

    def fail(*_args, **_kwargs):
        raise RuntimeError("endpoint failed")

    monkeypatch.setattr(local_backfill, "run", fail)

    assert local_backfill.main(["--providers", "test"]) == 1
