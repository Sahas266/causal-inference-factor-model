"""Recurring ingestion must actually recur, and must find nested configs.

These cover the two defects that froze the causal price feed: completed
endpoints were permanently terminal regardless of a later requested end, and
provider-wide discovery used a non-recursive glob that never saw the per-coin
directories where the causal sources live.

The orchestrator-level tests are the ones that pin the regression — the pure
helpers can all be right while `_backfill_endpoint_provider` still returns
early. The unit tests below them cover the edges those two cannot reach.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core.orchestrator import (
    BackfillOrchestrator,
    _completed_range_covers,
    _parse_date_range_bound,
)
from src.core.utils.config_loader import ConfigLoader, iter_endpoint_config_files

NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


def _write_endpoint(path: Path, endpoint_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "endpoint_id": endpoint_id,
            "table": "asset_metrics",
            "primary_keys": ["provider", "asset", "metric", "time"],
            "providers": [],
        }),
        encoding="utf-8",
    )


# ── the regression: a completed job must resume, not stop forever ────────

def _orchestrator_for_range_test(progress: dict):
    orchestrator = BackfillOrchestrator.__new__(BackfillOrchestrator)
    provider = Mock()
    provider.fetch_data_stream.return_value = iter([])
    orchestrator.providers = {"coinmetrics": provider}
    orchestrator.progress_tracker = Mock()
    orchestrator.progress_tracker.get_progress.return_value = progress
    orchestrator.db_writer = Mock()
    return orchestrator, provider


def _range_configs(requested_end: datetime):
    endpoint = {
        "endpoint_id": "btc_asset_metrics",
        "table": "asset_metrics",
        "primary_keys": ["provider", "asset", "metric", "time"],
    }
    provider = {
        "name": "coinmetrics",
        "provider_instance_id": "0",
        "actual_start": datetime(2021, 1, 1, tzinfo=timezone.utc),
        "actual_end": requested_end,
        "config": {"endpoint_type": "timeseries/asset-metrics"},
    }
    return endpoint, provider


def test_completed_same_range_is_a_no_fetch_skip():
    end = datetime(2026, 8, 11, tzinfo=timezone.utc)
    progress = {
        "status": "completed",
        "config": {"actual_end": end.isoformat()},
        "last_successful_time": "2026-08-10T00:00:00+00:00",
    }
    orchestrator, provider = _orchestrator_for_range_test(progress)
    endpoint, provider_config = _range_configs(end)

    result = orchestrator._backfill_endpoint_provider(endpoint, provider_config)

    assert result["skipped"] is True
    provider.fetch_data_stream.assert_not_called()
    orchestrator.progress_tracker.reopen_progress.assert_not_called()


def test_completed_advanced_range_reopens_at_checkpoint():
    """The frozen-feed regression: resume incrementally, do not replay."""
    old_end = datetime(2026, 8, 10, tzinfo=timezone.utc)
    new_end = datetime(2026, 8, 11, tzinfo=timezone.utc)
    checkpoint = datetime(2026, 8, 9, tzinfo=timezone.utc)
    progress = {
        "status": "completed",
        "config": {"actual_end": old_end.isoformat()},
        "last_successful_time": checkpoint.isoformat(),
    }
    orchestrator, provider = _orchestrator_for_range_test(progress)
    endpoint, provider_config = _range_configs(new_end)

    result = orchestrator._backfill_endpoint_provider(endpoint, provider_config)

    assert result["success"] is True
    assert result.get("skipped") is not True
    orchestrator.progress_tracker.reopen_progress.assert_called_once_with(
        "btc_asset_metrics_coinmetrics_0", provider_config
    )
    # Resumes from the checkpoint, not from actual_start — a progress reset
    # would replay the full history instead.
    provider.fetch_data_stream.assert_called_once_with(
        provider_config["config"], checkpoint, new_end
    )
    orchestrator.progress_tracker.mark_completed.assert_called_once()


# ── completed-range coverage edges ───────────────────────────────────────

def test_completed_range_only_skips_when_requested_end_is_covered():
    progress = {
        "status": "completed",
        "config": {"actual_end": "2026-08-10T00:00:00+00:00"},
    }
    assert _completed_range_covers(progress, datetime(2026, 8, 9, tzinfo=timezone.utc))
    assert not _completed_range_covers(progress, datetime(2026, 8, 11, tzinfo=timezone.utc))


@pytest.mark.parametrize("progress", [
    {"status": "completed", "config": {}},                                  # no bound
    {"status": "completed"},                                                # no config
    {"status": "completed", "config": {"actual_end": "not-a-date"}},        # unparseable
    {"status": "in_progress", "config": {"actual_end": "2027-01-01"}},      # unfinished
])
def test_uncertain_or_unfinished_progress_never_claims_coverage(progress):
    """Absent or unusable provenance must re-run rather than assume coverage."""
    assert _completed_range_covers(progress, NOW) is False


def test_no_requested_end_never_claims_coverage():
    progress = {"status": "completed", "config": {"actual_end": "2026-01-01"}}
    assert _completed_range_covers(progress, None) is False


# ── the "latest" sentinel ────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["latest", "LATEST", "  LaTeSt  "])
def test_latest_resolves_to_aware_utc_now(raw):
    assert _parse_date_range_bound(raw, 23, now=NOW) == NOW


def test_fixed_dates_and_empty_values_are_unchanged():
    """The sentinel must not disturb existing bounds."""
    assert _parse_date_range_bound("2026-01-01", 23) == datetime(
        2026, 1, 1, 23, tzinfo=timezone.utc
    )
    assert _parse_date_range_bound(None, 23) is None
    assert _parse_date_range_bound("", 23) is None


def test_datetime_input_is_normalised_to_utc():
    assert _parse_date_range_bound(datetime(2026, 8, 12, 12, 0), 23).tzinfo is timezone.utc


# ── recursive discovery, minus deliberately disabled trees ───────────────

def test_loader_finds_nested_configs_and_skips_disabled_dirs(tmp_path):
    endpoints = tmp_path / "endpoints"
    _write_endpoint(endpoints / "top.json", "top")
    _write_endpoint(endpoints / "btc" / "fees.json", "btc_fees")
    _write_endpoint(endpoints / "eth_disabled" / "old.json", "disabled")

    loader = ConfigLoader(str(tmp_path))

    assert [c["endpoint_id"] for c in loader.load_all_endpoint_configs()] == [
        "btc_fees", "top",
    ]
    assert loader.list_available_endpoints() == ["btc/fees", "top"]
    # An operator can still run a disabled config by naming its path.
    assert loader.load_endpoint_config(
        str(endpoints / "eth_disabled" / "old.json")
    )["endpoint_id"] == "disabled"


def test_discovery_excludes_disabled_at_any_depth(tmp_path):
    _write_endpoint(tmp_path / "eth" / "keep.json", "keep")
    _write_endpoint(tmp_path / "eth" / "nested_disabled" / "deep.json", "deep")

    found = [p.relative_to(tmp_path).as_posix() for p in iter_endpoint_config_files(tmp_path)]

    assert found == ["eth/keep.json"]


def test_discovery_order_is_deterministic(tmp_path):
    for name in ["c/c.json", "a/a.json", "b.json"]:
        _write_endpoint(tmp_path / name, name)
    first = [p.name for p in iter_endpoint_config_files(tmp_path)]
    assert first == [p.name for p in iter_endpoint_config_files(tmp_path)] == sorted(first)


# ── the real causal sources ──────────────────────────────────────────────

def test_real_causal_sources_are_discoverable():
    """They live in per-coin subdirectories the old non-recursive glob missed."""
    endpoints = Path(__file__).resolve().parents[2] / "config" / "endpoints"
    if not endpoints.is_dir():
        pytest.skip("endpoint configs not present in this checkout")
    found = {p.relative_to(endpoints).as_posix() for p in iter_endpoint_config_files(endpoints)}
    for required in (
        "btc/btc_coinmetrics.json",
        "doge/doge_coinmetrics.json",
        "eth/eth_asset_metrics_primary.json",
    ):
        assert required in found, f"{required} not discovered"


def test_causal_configs_keep_dynamic_end_after_generation():
    """A hardcoded end date is what froze ingestion; the manifest must agree.

    The manifest is checked too because regenerating configs from it would
    otherwise silently reinstate a fixed past bound.
    """
    config_root = Path(__file__).resolve().parents[2] / "config"
    if not config_root.is_dir():
        pytest.skip("configs not present in this checkout")
    manifest = json.loads((config_root / "coin_manifest.json").read_text(encoding="utf-8"))
    entries = {coin["ticker"]: coin for coin in manifest["coins"]}

    for asset, relative in {
        "btc": "btc/btc_coinmetrics.json",
        "doge": "doge/doge_coinmetrics.json",
        "eth": "eth/eth_asset_metrics_primary.json",
    }.items():
        endpoint = json.loads(
            (config_root / "endpoints" / relative).read_text(encoding="utf-8")
        )
        assert endpoint["date_range"]["end"] == "latest"
        assert entries[asset]["date_range"]["end"] == "latest"
