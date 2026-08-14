"""Operator DAG edits must never produce a graph the SCM cannot reason about."""

from __future__ import annotations

import networkx as nx
import pytest

from causal_portfolio.scm.dag_edits import (
    DagEdits,
    apply_edits,
    diff_identification,
    edge_rows,
    validate_new_edge,
)
from causal_portfolio.scm.graph import EdgeKind, NodeKind


def _dag() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("f1", kind=NodeKind.GLOBAL_FACTOR)
    g.add_node("f2", kind=NodeKind.GLOBAL_FACTOR)
    g.add_node("btc_return", kind=NodeKind.ASSET_RETURN)
    g.add_edge("f1", "btc_return", kind=EdgeKind.CAUSAL)
    g.add_edge("f2", "btc_return", kind=EdgeKind.CAUSAL)
    return g


def test_apply_edits_does_not_mutate_the_source_graph():
    dag = _dag()
    apply_edits(dag, DagEdits(removed=(("f1", "btc_return"),)))
    assert dag.has_edge("f1", "btc_return"), "caller's DAG was mutated"


def test_add_and_remove_round_trip():
    dag = _dag()
    edits = DagEdits().with_added("f1", "f2").with_removed("f2", "btc_return")
    edited = apply_edits(dag, edits)

    assert edited.has_edge("f1", "f2")
    assert not edited.has_edge("f2", "btc_return")
    assert edited["f1"]["f2"]["operator_added"] is True


def test_readding_a_removed_edge_restores_it_instead_of_stacking():
    edits = DagEdits().with_removed("f1", "btc_return").with_added("f1", "btc_return")
    assert edits.is_empty
    assert apply_edits(_dag(), edits).has_edge("f1", "btc_return")


def test_removing_a_just_added_edge_drops_the_addition():
    edits = DagEdits().with_added("f1", "f2").with_removed("f1", "f2")
    assert edits.is_empty


def test_repeated_edits_are_idempotent():
    edits = DagEdits().with_added("f1", "f2").with_added("f1", "f2")
    assert edits.added == (("f1", "f2"),)


def test_edits_survive_a_regenerated_dag_that_lost_those_nodes():
    """A stored edit must not crash after the asset/driver selection changes."""
    smaller = nx.DiGraph()
    smaller.add_node("f1", kind=NodeKind.GLOBAL_FACTOR)
    edits = DagEdits(added=(("f1", "gone"),), removed=(("also", "gone"),))

    edited = apply_edits(smaller, edits)

    assert list(edited.nodes()) == ["f1"]
    assert edited.number_of_edges() == 0


# ── the acyclicity guard ─────────────────────────────────────────────────

def test_cycle_creating_edge_is_rejected():
    """Identification assumes a DAG; a cycle makes its output meaningless."""
    dag = _dag()
    assert validate_new_edge(dag, "btc_return", "f1") is not None
    assert "cycle" in validate_new_edge(dag, "btc_return", "f1")


def test_indirect_cycle_is_rejected():
    dag = _dag()
    dag.add_edge("f2", "f1", kind=EdgeKind.CAUSAL)   # f2 -> f1 -> btc_return
    assert "cycle" in validate_new_edge(dag, "btc_return", "f2")


def test_a_legal_edge_validates_clean_and_keeps_the_graph_acyclic():
    dag = _dag()
    assert validate_new_edge(dag, "f1", "f2") is None
    edited = apply_edits(dag, DagEdits().with_added("f1", "f2"))
    assert nx.is_directed_acyclic_graph(edited)


@pytest.mark.parametrize("source,target,fragment", [
    ("f1", "f1", "own node"),
    ("nope", "f1", "unknown node"),
    ("f1", "nope", "unknown node"),
    ("f1", "btc_return", "already exists"),
])
def test_illegal_edges_are_named(source, target, fragment):
    assert fragment in validate_new_edge(_dag(), source, target)


# ── display + diff ───────────────────────────────────────────────────────

def test_edge_rows_tag_operator_edges():
    rows = edge_rows(_dag(), DagEdits().with_added("f1", "f2"))
    origins = {(r["source"], r["target"]): r["origin"] for r in rows}

    assert origins[("f1", "f2")] == "operator"
    assert origins[("f1", "btc_return")] == "generated"


class _Result:
    def __init__(self, treatment, outcome, identifiable, strategy):
        self.treatment, self.outcome = treatment, outcome
        self.identifiable, self.strategy = identifiable, strategy


def test_diff_reports_gained_lost_and_strategy_changes():
    before = [
        _Result("f1", "btc_return", True, "backdoor"),
        _Result("f2", "btc_return", False, None),
    ]
    after = [
        _Result("f1", "btc_return", True, "iv"),
        _Result("f2", "btc_return", True, "backdoor"),
    ]

    diff = diff_identification(before, after)

    assert diff["gained"] == ["f2 -> btc_return"]
    assert diff["lost"] == []
    assert diff["changed"] == ["f1 -> btc_return: backdoor -> iv"]


def test_diff_is_empty_when_nothing_changed():
    results = [_Result("f1", "btc_return", True, "backdoor")]
    assert diff_identification(results, results) == {
        "gained": [], "lost": [], "changed": [],
    }
