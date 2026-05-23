"""Tests for the per-regime stability diagnostic.

Synthetic RegimeReport objects exercise the rendering and verdict logic
without the data-loading + HMM-fit path.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_portfolio.diagnostics.regime_stability import (
    RegimeReport,
    RegimeWinner,
    _verdict,
    append_to_doc,
    render_markdown,
)


def _make_winner(regime: int, winner: tuple, score: float = 5.0,
                 global_rank: int = 50) -> RegimeWinner:
    return RegimeWinner(
        regime=regime, n_obs=300, pct_of_sample=0.5,
        winner=winner, score=score,
        runner_up=("x", "y", "z"), runner_up_score=score + 0.1,
        global_pick_rank=global_rank, global_pick_score=score + 0.5,
    )


def _make_report(winners: list[RegimeWinner],
                 global_winner: tuple = ("a", "b", "c")) -> RegimeReport:
    return RegimeReport(
        n_states=2,
        feature_columns=["vix", "btc_vol"],
        transition_matrix=np.array([[0.95, 0.05], [0.10, 0.90]]),
        state_means=np.array([[-0.5, -0.5], [0.5, 0.5]]),
        global_winner=global_winner,
        global_score=4.5,
        dwell={0: {"days": 500, "pct": 0.6, "n_runs": 5,
                   "mean_run_length": 100, "max_run_length": 200},
               1: {"days": 333, "pct": 0.4, "n_runs": 5,
                   "mean_run_length": 66, "max_run_length": 150}},
        regime_winners=winners,
        n_total_obs=833,
    )


# ── verdict logic ───────────────────────────────────────────────────


def test_verdict_phase_b_justified():
    r = _make_report([
        _make_winner(0, ("chain_congestion", "mev_pressure", "vixcls")),
        _make_winner(1, ("liq_flow", "funding_basis", "stable_flow")),
    ])
    msg = _verdict(r)
    assert "JUSTIFIED" in msg


def test_verdict_phase_b_not_justified_when_all_match_global():
    r = _make_report([
        _make_winner(0, ("a", "b", "c")),
        _make_winner(1, ("a", "b", "c")),
    ])
    msg = _verdict(r)
    assert "NOT justified" in msg


def test_verdict_partial_when_all_same_but_differ_from_global():
    r = _make_report([
        _make_winner(0, ("x", "y", "z")),
        _make_winner(1, ("x", "y", "z")),
    ])
    msg = _verdict(r)
    assert "partially justified" in msg.lower()


def test_verdict_inconclusive_when_only_one_regime():
    r = _make_report([_make_winner(0, ("a", "b", "c"))])
    msg = _verdict(r)
    assert "inconclusive" in msg.lower()


# ── markdown rendering ──────────────────────────────────────────────


def test_render_markdown_has_all_sections():
    r = _make_report([
        _make_winner(0, ("chain_congestion", "mev_pressure", "vixcls")),
        _make_winner(1, ("liq_flow", "funding_basis", "stable_flow")),
    ])
    md = render_markdown(r, run_args={
        "assets": ["btc", "eth"], "start": "2022-01-01", "end": "2025-12-31",
        "m": 3, "n_states": 2,
    })
    for section in ["Run parameters", "HMM characterization",
                    "Transition matrix", "Regime dwell",
                    "Per-regime winning subsets", "Driver overlap",
                    "Verdict"]:
        assert section in md, f"section missing: {section}"


def test_render_markdown_marks_drivers_in_global_pick():
    r = _make_report(
        winners=[
            _make_winner(0, ("a", "b", "c")),  # all in global
            _make_winner(1, ("x", "y", "z")),  # none in global
        ],
        global_winner=("a", "b", "c"),
    )
    md = render_markdown(r, run_args={
        "assets": ["btc"], "start": "2022-01-01", "end": "2025-12-31",
        "m": 3, "n_states": 2,
    })
    # Each row in the overlap table should mark global-pick membership
    assert "| `a` |" in md
    assert "| `x` |" in md


# ── doc append ──────────────────────────────────────────────────────


def test_append_to_doc_creates_phase_a_section(tmp_path):
    out = tmp_path / "doc.md"
    out.write_text("# Doc\n\nsome preamble\n")
    append_to_doc("new phase A content\n", str(out))
    text = out.read_text()
    assert "some preamble" in text
    assert "Phase A: HMM regime-conditional selection results" in text
    assert "new phase A content" in text


def test_append_to_doc_replaces_existing_phase_a(tmp_path):
    out = tmp_path / "doc.md"
    out.write_text("# Doc\n\n## Phase A: HMM regime-conditional selection results\n\nOLD\n")
    append_to_doc("NEW\n", str(out))
    text = out.read_text()
    assert "NEW" in text
    assert "OLD" not in text
