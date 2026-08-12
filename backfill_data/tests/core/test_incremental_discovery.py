"""Recurring ingestion must actually recur, and must find nested configs.

These cover the two defects that froze the causal price feed: completed
endpoints were permanently terminal regardless of a later requested end, and
provider-wide discovery used a non-recursive glob that never saw the per-coin
directories where the causal sources live.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.orchestrator import _completed_range_covers, _parse_date_range_bound
from src.core.utils.config_loader import iter_endpoint_config_files

NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


# ── the "latest" sentinel ────────────────────────────────────────────────

def test_latest_resolves_to_an_aware_utc_time():
    resolved = _parse_date_range_bound("latest", 23, now=NOW)
    assert resolved == NOW
    assert resolved.tzinfo is not None


@pytest.mark.parametrize("raw", ["latest", "LATEST", "  LaTeSt  "])
def test_latest_is_case_and_whitespace_insensitive(raw):
    assert _parse_date_range_bound(raw, 23, now=NOW) == NOW


def test_fixed_dates_and_empty_values_are_unchanged():
    """The sentinel must not disturb existing bounds."""
    assert _parse_date_range_bound("2026-01-01", 23) == datetime(
        2026, 1, 1, 23, tzinfo=timezone.utc
    )
    assert _parse_date_range_bound(None, 23) is None
    assert _parse_date_range_bound("", 23) is None


def test_datetime_input_is_normalised_to_utc():
    naive = datetime(2026, 8, 12, 12, 0)
    assert _parse_date_range_bound(naive, 23).tzinfo is timezone.utc


# ── completed jobs must reopen when the bound advances ───────────────────

def test_completed_endpoint_reopens_when_requested_end_advances():
    """The regression that made recurring ingestion one-shot.

    A completed job whose stored bound predates the new request must NOT be
    skipped, or the feed silently freezes at the old end date.
    """
    progress = {"status": "completed", "config": {"actual_end": "2026-01-01"}}
    assert _completed_range_covers(progress, NOW) is False


def test_completed_endpoint_skips_when_bound_already_covers_request():
    progress = {"status": "completed", "config": {"actual_end": "2026-08-12"}}
    assert _completed_range_covers(progress, NOW) is True


def test_completed_endpoint_without_a_prior_bound_reruns():
    """Absent provenance, re-run rather than assume coverage."""
    assert _completed_range_covers({"status": "completed", "config": {}}, NOW) is False
    assert _completed_range_covers({"status": "completed"}, NOW) is False


def test_unfinished_endpoint_is_never_treated_as_covered():
    progress = {"status": "in_progress", "config": {"actual_end": "2027-01-01"}}
    assert _completed_range_covers(progress, NOW) is False


def test_unparseable_stored_bound_reruns_instead_of_raising():
    progress = {"status": "completed", "config": {"actual_end": "not-a-date"}}
    assert _completed_range_covers(progress, NOW) is False


def test_no_requested_end_never_claims_coverage():
    progress = {"status": "completed", "config": {"actual_end": "2026-01-01"}}
    assert _completed_range_covers(progress, None) is False


# ── recursive discovery, minus deliberately disabled trees ───────────────

def _write(path: Path, endpoint_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"endpoint_id": endpoint_id}), encoding="utf-8")


def test_discovery_finds_nested_configs_and_skips_disabled_dirs(tmp_path):
    _write(tmp_path / "top_level.json", "top")
    _write(tmp_path / "btc" / "btc_coinmetrics.json", "btc")
    _write(tmp_path / "eth" / "eth_asset_metrics_primary.json", "eth")
    _write(tmp_path / "eth_disabled" / "eth_old.json", "disabled")
    _write(tmp_path / "eth" / "nested_disabled" / "deep.json", "deep-disabled")

    found = [p.relative_to(tmp_path).as_posix() for p in iter_endpoint_config_files(tmp_path)]

    assert "btc/btc_coinmetrics.json" in found
    assert "eth/eth_asset_metrics_primary.json" in found
    assert "top_level.json" in found
    assert not [f for f in found if "_disabled" in f], f"disabled tree leaked: {found}"


def test_discovery_order_is_deterministic(tmp_path):
    for name in ["c/c.json", "a/a.json", "b.json"]:
        _write(tmp_path / name, name)
    first = [p.name for p in iter_endpoint_config_files(tmp_path)]
    assert first == [p.name for p in iter_endpoint_config_files(tmp_path)]
    assert first == sorted(first)


def test_real_causal_sources_are_discoverable():
    """The three configs the causal model depends on must be reachable.

    They live in per-coin subdirectories, which the previous non-recursive
    glob could not see.
    """
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


def test_causal_sources_use_a_dynamic_end_bound():
    """A hardcoded end date is what froze ingestion; keep these dynamic."""
    endpoints = Path(__file__).resolve().parents[2] / "config" / "endpoints"
    if not endpoints.is_dir():
        pytest.skip("endpoint configs not present in this checkout")
    for rel in (
        "btc/btc_coinmetrics.json",
        "doge/doge_coinmetrics.json",
        "eth/eth_asset_metrics_primary.json",
    ):
        config = json.loads((endpoints / rel).read_text(encoding="utf-8"))
        end = config.get("date_range", {}).get("end")
        assert end == "latest", f"{rel} end bound is {end!r}, expected 'latest'"
