"""Operator DAG edits must never produce a graph the SCM cannot reason about."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from causal_portfolio.scm.dag_edits import (
    DAG_SOURCES,
    build_base_dag,
    delete_workspace,
    load_workspaces,
    read_workspace,
    save_workspace,
    validate_new_node,
    DagEdits,
    apply_edits,
    diff_identification,
    edge_rows,
    validate_new_edge,
)
from causal_portfolio.scm.graph import EdgeKind, NodeKind
from causal_portfolio.scm.identification import (
    IdentificationMethod,
    IdentificationResult,
)


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


def test_stale_saved_edge_cannot_cycle_a_regenerated_base():
    dag = _dag()
    dag.add_edge("f2", "f1", kind=EdgeKind.CAUSAL)

    with pytest.raises(ValueError, match="would create a cycle"):
        apply_edits(dag, DagEdits(added=(("f1", "f2"),)))


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
    def __init__(self, treatment, outcome, identified, method):
        self.treatment, self.outcome = treatment, outcome
        self.identified, self.method = identified, method


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


def test_diff_uses_the_real_identification_result_contract():
    before = [IdentificationResult(
        "f1", "btc_return", False, IdentificationMethod.NOT_IDENTIFIED,
    )]
    after = [IdentificationResult(
        "f1", "btc_return", True, IdentificationMethod.BACKDOOR,
    )]

    assert diff_identification(before, after) == {
        "gained": ["f1 -> btc_return"], "lost": [], "changed": [],
    }


# ── custom nodes ─────────────────────────────────────────────────────────

def test_a_dag_can_be_built_from_the_blank_base():
    """The point of 'Blank': a structure with no generated nodes to subtract."""
    edits = (
        DagEdits()
        .with_node("my_factor", "GLOBAL_FACTOR")
        .with_node("my_return", "ASSET_RETURN")
        .with_added("my_factor", "my_return")
    )
    built = apply_edits(nx.DiGraph(), edits)

    assert set(built.nodes()) == {"my_factor", "my_return"}
    assert built.has_edge("my_factor", "my_return")
    assert built.nodes["my_factor"]["kind"] is NodeKind.GLOBAL_FACTOR
    assert nx.is_directed_acyclic_graph(built)


def test_edge_edits_preserve_custom_nodes():
    """Regression: a positional DagEdits constructor silently dropped them."""
    edits = DagEdits().with_node("n1", "GLOBAL_FACTOR").with_added("f1", "f2")
    assert edits.added_nodes == (("n1", "GLOBAL_FACTOR"),)

    edits = edits.with_removed("f1", "btc_return")
    assert edits.added_nodes == (("n1", "GLOBAL_FACTOR"),)


def test_removing_a_custom_node_drops_edits_that_referenced_it():
    edits = (
        DagEdits()
        .with_node("tmp", "GLOBAL_FACTOR")
        .with_added("tmp", "btc_return")
        .without_node("tmp")
    )
    assert edits.is_empty


def test_duplicate_node_is_ignored_and_unknown_kind_falls_back():
    edits = DagEdits().with_node("n", "GLOBAL_FACTOR").with_node("n", "ASSET_RETURN")
    assert edits.added_nodes == (("n", "GLOBAL_FACTOR"),)

    built = apply_edits(nx.DiGraph(), DagEdits(added_nodes=(("x", "NOT_A_KIND"),)))
    assert built.nodes["x"]["kind"] is NodeKind.GLOBAL_FACTOR


@pytest.mark.parametrize("name,fragment", [
    ("", "empty"),
    ("   ", "empty"),
    (" pad", "whitespace"),
    ("f1", "already exists"),
])
def test_illegal_node_names_are_named(name, fragment):
    assert fragment in validate_new_node(_dag(), name)


# ── base DAG sources ─────────────────────────────────────────────────────

def test_every_registered_source_builds():
    for source in DAG_SOURCES.values():
        dag = build_base_dag(source, ["btc", "eth"])
        assert isinstance(dag, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(dag)
    assert build_base_dag("blank", ["btc"]).number_of_nodes() == 0
    assert build_base_dag("cpcm", ["btc"]).number_of_nodes() > 0


# ── saved workspaces ─────────────────────────────────────────────────────

def test_workspace_round_trip(tmp_path):
    path = tmp_path / "ws.json"
    edits = DagEdits().with_node("n", "GLOBAL_FACTOR").with_added("f1", "f2")

    save_workspace("mine", "discovered", edits, path=path)
    source, loaded = read_workspace("mine", path)

    assert source == "discovered"
    assert loaded == edits


def test_saving_one_workspace_preserves_the_others(tmp_path):
    path = tmp_path / "ws.json"
    save_workspace("a", "cpcm", DagEdits().with_added("f1", "f2"), path=path)
    save_workspace("b", "blank", DagEdits().with_node("n", "GLOBAL_FACTOR"), path=path)

    assert set(load_workspaces(path)) == {"a", "b"}
    assert read_workspace("a", path)[1].added == (("f1", "f2"),)


def test_missing_and_corrupt_workspace_files_read_as_empty(tmp_path):
    """A bad scratch file must not stop the dashboard from rendering."""
    assert load_workspaces(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_workspaces(bad) == {}
    assert read_workspace("anything", bad) is None


@pytest.mark.parametrize("edits", [
    [],
    {"added": [["only-one"]]},
    {"removed": "not-a-list"},
    {"added_nodes": [["node", 123]]},
])
def test_malformed_workspace_schema_is_ignored(tmp_path, edits):
    path = tmp_path / "bad-schema.json"
    path.write_text(
        json.dumps({"bad": {"source": "cpcm", "edits": edits}}),
        encoding="utf-8",
    )

    assert load_workspaces(path) == {}
    assert read_workspace("bad", path) is None


def test_delete_workspace(tmp_path):
    path = tmp_path / "ws.json"
    save_workspace("gone", "cpcm", DagEdits(), path=path)

    assert delete_workspace("gone", path) is True
    assert delete_workspace("gone", path) is False
    assert load_workspaces(path) == {}


def test_blank_workspace_name_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_workspace("  ", "cpcm", DagEdits(), path=tmp_path / "ws.json")
