"""Tests for the driver-stability diagnostic.

Skips the data-loading path — tests the rendering, interpretation, and doc
append logic with synthetic StabilityReport objects.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from causal_portfolio.diagnostics.driver_stability import (
    StabilityReport,
    WindowResult,
    _interpretation,
    append_to_doc,
    render_markdown,
)


def _make_report(pct_optimal: float, win_counts: dict, driver_counts: dict,
                 n_windows: int = 10) -> StabilityReport:
    global_winner = next(iter(win_counts))
    return StabilityReport(
        n_windows=n_windows,
        global_winner=global_winner,
        global_score=0.5,
        window_results=[],
        subset_win_counts=Counter(win_counts),
        driver_appearance_counts=Counter(driver_counts),
        global_winner_avg_rank=2.0,
        global_winner_pct_optimal=pct_optimal,
    )


def test_interpretation_stable():
    r = _make_report(pct_optimal=0.80, win_counts={("a", "b"): 8},
                     driver_counts={"a": 8, "b": 8})
    msg = _interpretation(r)
    assert "Skip Option 3" in msg


def test_interpretation_partial():
    r = _make_report(pct_optimal=0.50, win_counts={("a", "b"): 5},
                     driver_counts={"a": 5, "b": 5})
    msg = _interpretation(r)
    assert "lightweight" in msg.lower()


def test_interpretation_unstable():
    r = _make_report(pct_optimal=0.10, win_counts={("a", "b"): 1},
                     driver_counts={"a": 1})
    msg = _interpretation(r)
    assert "justified" in msg.lower()


def test_render_markdown_contains_essentials():
    r = _make_report(
        pct_optimal=0.50,
        win_counts={("a", "b"): 5, ("c", "d"): 3, ("a", "c"): 2},
        driver_counts={"a": 7, "b": 5, "c": 5, "d": 3},
    )
    # Add one window result so the per-window table isn't empty
    r.window_results.append(WindowResult(
        start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-06-01"),
        n_obs=120, n_candidates=15,
        winning_subset=("a", "b"), winning_score=0.4,
        runner_up=("c", "d"), runner_up_score=0.5,
        global_subset_rank=1, global_subset_score=0.4,
    ))
    md = render_markdown(r, run_args={
        "assets": ["btc"], "start": "2024-01-01", "end": "2024-12-31",
        "m": 2, "window_days": 180, "step_days": 30,
    })
    assert "Run parameters" in md
    assert "Global winner" in md
    assert "Subset win frequency" in md
    assert "Per-window winners" in md
    assert "Interpretation" in md
    # Global pick marker appears
    assert "← global" in md
    assert "← in global pick" in md


def test_append_to_doc_creates_new_file(tmp_path):
    out = tmp_path / "doc.md"
    append_to_doc("hello world\n", str(out))
    assert out.exists()
    assert "Option 2 results" in out.read_text()
    assert "hello world" in out.read_text()


def test_append_to_doc_replaces_existing_results(tmp_path):
    out = tmp_path / "doc.md"
    out.write_text("preamble\n\n## Option 2 results\n\nOLD CONTENT\n")
    append_to_doc("NEW CONTENT\n", str(out))
    text = out.read_text()
    assert "preamble" in text         # preamble preserved
    assert "NEW CONTENT" in text      # new content added
    assert "OLD CONTENT" not in text  # old content replaced


def test_append_to_doc_preserves_preamble(tmp_path):
    """Appending must not damage the description sections above the sentinel."""
    out = tmp_path / "doc.md"
    out.write_text("# Title\n\nbig long preamble here\n\n## Option 2 results\n\nold\n")
    append_to_doc("fresh results\n", str(out))
    text = out.read_text()
    assert "big long preamble here" in text
