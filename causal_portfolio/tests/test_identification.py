"""Validate the identification port against the Rust dsep.rs + identify.rs tests.

Every test here mirrors a Rust unit test one-to-one, so a green run means the
Python d-separation / backdoor / IV logic matches the reference engine.
"""

from __future__ import annotations

import networkx as nx
import pytest

from causal_portfolio.scm.graph import NodeKind, EdgeKind, build_cpcm_dag
from causal_portfolio.scm.identification import (
    d_separated, d_connected_set,
    backdoor_adjustment_set, check_iv_validity, identify_all_effects,
    IdentificationMethod,
)


def _dag():
    return nx.DiGraph()


def _add(g, name, kind):
    g.add_node(name, kind=kind, asset=None)


def _edge(g, u, v):
    g.add_edge(u, v, kind=EdgeKind.CAUSAL, lag=0)


# ── d-separation (mirrors dsep.rs tests) ─────────────────────────────


def test_chain_unconditioned():
    g = _dag()
    _add(g, "A", NodeKind.GLOBAL_FACTOR); _add(g, "B", NodeKind.GLOBAL_FACTOR)
    _add(g, "C", NodeKind.ASSET_RETURN)
    _edge(g, "A", "B"); _edge(g, "B", "C")
    assert not d_separated(g, "A", "C", [])


def test_chain_conditioned_on_mediator():
    g = _dag()
    _add(g, "A", NodeKind.GLOBAL_FACTOR); _add(g, "B", NodeKind.GLOBAL_FACTOR)
    _add(g, "C", NodeKind.ASSET_RETURN)
    _edge(g, "A", "B"); _edge(g, "B", "C")
    assert d_separated(g, "A", "C", ["B"])


def test_fork_unconditioned():
    g = _dag()
    _add(g, "A", NodeKind.ASSET_RETURN); _add(g, "B", NodeKind.GLOBAL_FACTOR)
    _add(g, "C", NodeKind.ASSET_RETURN)
    _edge(g, "B", "A"); _edge(g, "B", "C")
    assert not d_separated(g, "A", "C", [])


def test_fork_conditioned_on_common_cause():
    g = _dag()
    _add(g, "A", NodeKind.ASSET_RETURN); _add(g, "B", NodeKind.GLOBAL_FACTOR)
    _add(g, "C", NodeKind.ASSET_RETURN)
    _edge(g, "B", "A"); _edge(g, "B", "C")
    assert d_separated(g, "A", "C", ["B"])


def test_collider_unconditioned():
    g = _dag()
    _add(g, "A", NodeKind.GLOBAL_FACTOR); _add(g, "B", NodeKind.ASSET_RETURN)
    _add(g, "C", NodeKind.GLOBAL_FACTOR)
    _edge(g, "A", "B"); _edge(g, "C", "B")
    assert d_separated(g, "A", "C", [])


def test_collider_conditioned_on_collider():
    g = _dag()
    _add(g, "A", NodeKind.GLOBAL_FACTOR); _add(g, "B", NodeKind.ASSET_RETURN)
    _add(g, "C", NodeKind.GLOBAL_FACTOR)
    _edge(g, "A", "B"); _edge(g, "C", "B")
    assert not d_separated(g, "A", "C", ["B"])


def test_collider_conditioned_on_descendant():
    g = _dag()
    _add(g, "A", NodeKind.GLOBAL_FACTOR); _add(g, "B", NodeKind.ASSET_RETURN)
    _add(g, "C", NodeKind.GLOBAL_FACTOR); _add(g, "D", NodeKind.ASSET_COVARIATE)
    _edge(g, "A", "B"); _edge(g, "C", "B"); _edge(g, "B", "D")
    assert not d_separated(g, "A", "C", ["D"])


def test_disconnected_graph():
    g = _dag()
    for n in "ABC":
        _add(g, n, NodeKind.GLOBAL_FACTOR)
    assert d_separated(g, "A", "B", [])
    assert d_separated(g, "A", "C", [])
    assert d_separated(g, "B", "C", [])


def test_diamond():
    g = _dag()
    _add(g, "A", NodeKind.GLOBAL_FACTOR); _add(g, "B", NodeKind.GLOBAL_FACTOR)
    _add(g, "C", NodeKind.GLOBAL_FACTOR); _add(g, "D", NodeKind.ASSET_RETURN)
    _edge(g, "A", "B"); _edge(g, "A", "C"); _edge(g, "B", "D"); _edge(g, "C", "D")
    assert not d_separated(g, "A", "D", [])
    assert not d_separated(g, "A", "D", ["B"])
    assert d_separated(g, "A", "D", ["B", "C"])


def test_d_connected_set():
    g = _dag()
    _add(g, "A", NodeKind.GLOBAL_FACTOR); _add(g, "B", NodeKind.GLOBAL_FACTOR)
    _add(g, "C", NodeKind.ASSET_RETURN); _add(g, "D", NodeKind.ASSET_RETURN)
    _edge(g, "A", "B"); _edge(g, "B", "C")
    connected = d_connected_set(g, "A", [])
    assert "B" in connected and "C" in connected and "D" not in connected


def test_self_not_separated():
    g = _dag()
    _add(g, "A", NodeKind.GLOBAL_FACTOR)
    assert not d_separated(g, "A", "A", [])


def test_nonexistent_node_separated():
    g = _dag()
    _add(g, "A", NodeKind.GLOBAL_FACTOR)
    assert d_separated(g, "A", "ZZZ", [])


# ── identification (mirrors identify.rs tests) ───────────────────────


def test_backdoor_no_confounders():
    g = _dag()
    _add(g, "X", NodeKind.GLOBAL_FACTOR)
    g.add_node("Y", kind=NodeKind.ASSET_RETURN, asset="eth")
    _edge(g, "X", "Y")
    adj = backdoor_adjustment_set(g, "X", "Y")
    assert adj == []


def test_backdoor_with_confounder():
    g = _dag()
    _add(g, "X", NodeKind.GLOBAL_FACTOR)
    g.add_node("Y", kind=NodeKind.ASSET_RETURN, asset="eth")
    _add(g, "Z", NodeKind.MACRO_FACTOR)
    _edge(g, "X", "Y"); _edge(g, "Z", "X"); _edge(g, "Z", "Y")
    adj = backdoor_adjustment_set(g, "X", "Y")
    assert adj is not None and "Z" in adj


def test_iv_valid():
    g = _dag()
    _add(g, "Z", NodeKind.INSTRUMENT); _add(g, "X", NodeKind.GLOBAL_FACTOR)
    g.add_node("Y", kind=NodeKind.ASSET_RETURN, asset="eth")
    g.add_edge("Z", "X", kind=EdgeKind.INSTRUMENTAL, lag=1)
    _edge(g, "X", "Y")
    r = check_iv_validity(g, "Z", "X", "Y")
    assert r.valid and r.relevant and r.excludable


def test_iv_exclusion_violated():
    g = _dag()
    _add(g, "Z", NodeKind.INSTRUMENT); _add(g, "X", NodeKind.GLOBAL_FACTOR)
    g.add_node("Y", kind=NodeKind.ASSET_RETURN, asset="eth")
    _edge(g, "Z", "X"); _edge(g, "X", "Y"); _edge(g, "Z", "Y")
    r = check_iv_validity(g, "Z", "X", "Y")
    assert not r.valid and r.relevant and not r.excludable


def test_iv_no_relevance():
    g = _dag()
    _add(g, "Z", NodeKind.INSTRUMENT); _add(g, "X", NodeKind.GLOBAL_FACTOR)
    g.add_node("Y", kind=NodeKind.ASSET_RETURN, asset="eth")
    _edge(g, "X", "Y")  # Z disconnected
    r = check_iv_validity(g, "Z", "X", "Y")
    assert not r.valid and not r.relevant


def test_identify_all_simple():
    g = _dag()
    _add(g, "F1", NodeKind.GLOBAL_FACTOR)
    g.add_node("eth_return", kind=NodeKind.ASSET_RETURN, asset="eth")
    _edge(g, "F1", "eth_return")
    results = identify_all_effects(g)
    assert len(results) == 1
    assert results[0].identified
    assert results[0].method is IdentificationMethod.BACKDOOR


# ── integration: the real CPCM DAG ───────────────────────────────────


def test_cpcm_dag_identifies_effects():
    g = build_cpcm_dag(["btc", "eth"])
    results = identify_all_effects(g)
    # Every factor->return pair should be classified (identified or not)
    assert len(results) > 0
    # The CPCM star-DAG has factors directly into returns with shock confounding
    # only via the unobserved shock (which can't open a backdoor), so factors are
    # backdoor-identified with the empty set in this structure.
    identified = [r for r in results if r.identified]
    assert len(identified) > 0
