"""Causal identification — Python port of cpcm-core dsep.rs + identify.rs.

Operates on the networkx DiGraph built by ``scm/graph.py`` (node attr ``kind``
is a ``NodeKind`` enum, edges carry ``kind`` + ``lag``).

Provides:
  - ``d_separated`` — Bayes-Ball (Shachter 1998), faithful to dsep.rs.
  - ``backdoor_adjustment_set`` — minimal backdoor set search (identify.rs).
  - ``check_iv_validity`` — graph-based IV relevance + exclusion.
  - ``identify_all_effects`` — backdoor-first, IV-fallback over all
    factor→return pairs.

The point of identification: before running 2SLS, the graph tells us *whether*
the causal effect of a factor on a return is even estimable from observed data
(is there a valid adjustment set, or a valid instrument?). Only then does the
numeric estimate (estimators.py) carry a causal interpretation.

Validated against the Rust unit tests (mirrored in tests/test_identification.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import networkx as nx

from causal_portfolio.scm.graph import NodeKind


class _Arrival(Enum):
    FROM_PARENT = auto()   # travelling downward along an edge
    FROM_CHILD = auto()    # travelling upward against an edge


def _kind(dag: nx.DiGraph, node: str) -> NodeKind | None:
    return dag.nodes[node].get("kind") if node in dag.nodes else None


def _parents(dag: nx.DiGraph, node: str) -> list[str]:
    return list(dag.predecessors(node))


def _children(dag: nx.DiGraph, node: str) -> list[str]:
    return list(dag.successors(node))


# ── d-separation (Bayes-Ball) ────────────────────────────────────────


def d_separated(dag: nx.DiGraph, x: str, y: str, z: list[str]) -> bool:
    """True iff every path between x and y is blocked given conditioning set z.

    Faithful port of dsep.rs::d_separated (Bayes-Ball with explicit collider
    descendant handling). Non-existent nodes are trivially separated; a node is
    never d-separated from itself.
    """
    if x not in dag.nodes or y not in dag.nodes:
        return True
    if x == y:
        return False

    z_set = {n for n in z if n in dag.nodes}
    descendant_in_z = _compute_descendant_in_z(dag, z_set)

    visited: set[tuple[str, _Arrival]] = set()
    queue: deque[tuple[str, _Arrival]] = deque()

    x_conditioned = x in z_set
    if not x_conditioned:
        for child in _children(dag, x):
            st = (child, _Arrival.FROM_PARENT)
            if st not in visited:
                visited.add(st)
                queue.append(st)
        for parent in _parents(dag, x):
            st = (parent, _Arrival.FROM_CHILD)
            if st not in visited:
                visited.add(st)
                queue.append(st)
    else:
        if x in descendant_in_z:
            for parent in _parents(dag, x):
                st = (parent, _Arrival.FROM_CHILD)
                if st not in visited:
                    visited.add(st)
                    queue.append(st)

    while queue:
        current, arrival = queue.popleft()
        if current == y:
            return False
        is_conditioned = current in z_set

        if arrival is _Arrival.FROM_PARENT:
            # arrived downward (from a parent)
            if not is_conditioned:
                for child in _children(dag, current):
                    st = (child, _Arrival.FROM_PARENT)
                    if st not in visited:
                        visited.add(st)
                        queue.append(st)
            if is_conditioned or current in descendant_in_z:
                for parent in _parents(dag, current):
                    st = (parent, _Arrival.FROM_CHILD)
                    if st not in visited:
                        visited.add(st)
                        queue.append(st)
        else:  # FROM_CHILD: arrived upward (from a child)
            if not is_conditioned:
                for child in _children(dag, current):
                    st = (child, _Arrival.FROM_PARENT)
                    if st not in visited:
                        visited.add(st)
                        queue.append(st)
                for parent in _parents(dag, current):
                    st = (parent, _Arrival.FROM_CHILD)
                    if st not in visited:
                        visited.add(st)
                        queue.append(st)

    return True


def _compute_descendant_in_z(dag: nx.DiGraph, z_set: set[str]) -> set[str]:
    """Nodes that are in Z or have a descendant in Z (walk up ancestors of Z)."""
    result: set[str] = set(z_set)
    for z_node in z_set:
        stack = [z_node]
        while stack:
            current = stack.pop()
            for parent in _parents(dag, current):
                if parent not in result:
                    result.add(parent)
                    stack.append(parent)
    return result


def d_connected_set(dag: nx.DiGraph, x: str, z: list[str]) -> list[str]:
    """All nodes not d-separated from x given z."""
    return [n for n in dag.nodes if n != x and not d_separated(dag, x, n, z)]


# ── identification ───────────────────────────────────────────────────


class IdentificationMethod(Enum):
    BACKDOOR = auto()
    IV = auto()
    NOT_IDENTIFIED = auto()


@dataclass
class IdentificationResult:
    treatment: str
    outcome: str
    identified: bool
    method: IdentificationMethod
    adjustment_set: list[str] = field(default_factory=list)
    instrument: str | None = None
    reason: str = ""


def backdoor_adjustment_set(
    dag: nx.DiGraph, treatment: str, outcome: str,
) -> list[str] | None:
    """Find a valid backdoor adjustment set, or None if unidentified.

    Port of identify.rs::backdoor_adjustment_set: try empty set, then singles,
    then pairs, then the full candidate set. Candidates exclude descendants of
    treatment and unobserved shocks.
    """
    descendants = nx.descendants(dag, treatment) if treatment in dag.nodes else set()
    candidates = [
        n for n in dag.nodes
        if n != treatment and n != outcome
        and n not in descendants
        and _kind(dag, n) != NodeKind.UNOBSERVED_SHOCK
    ]

    if _blocks_all_backdoor_paths(dag, treatment, outcome, []):
        return []
    for c in candidates:
        if _blocks_all_backdoor_paths(dag, treatment, outcome, [c]):
            return [c]
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            pair = [candidates[i], candidates[j]]
            if _blocks_all_backdoor_paths(dag, treatment, outcome, pair):
                return pair
    if _blocks_all_backdoor_paths(dag, treatment, outcome, candidates):
        return candidates
    return None


def _blocks_all_backdoor_paths(
    dag: nx.DiGraph, treatment: str, outcome: str, adjustment_set: list[str],
) -> bool:
    """Port of identify.rs::blocks_all_backdoor_paths.

    For each parent P of treatment, P must be d-separated from outcome given
    {adjustment_set ∪ treatment}.
    """
    parents = _parents(dag, treatment)
    if not parents:
        return True
    cond = list(adjustment_set)
    if treatment not in cond:
        cond.append(treatment)
    for parent in parents:
        if _kind(dag, parent) == NodeKind.UNOBSERVED_SHOCK:
            if not d_separated(dag, parent, outcome, cond):
                return False
            continue
        if not d_separated(dag, parent, outcome, cond):
            return False
    return True


@dataclass
class IvValidityResult:
    instrument: str
    treatment: str
    outcome: str
    relevant: bool
    excludable: bool
    valid: bool
    reason: str


def check_iv_validity(
    dag: nx.DiGraph, instrument: str, treatment: str, outcome: str,
) -> IvValidityResult:
    """Graph-based IV check (port of identify.rs::check_iv_validity).

    Relevance: Z is NOT d-separated from treatment (unconditionally).
    Exclusion: Z IS d-separated from outcome given treatment.
    """
    relevant = not d_separated(dag, instrument, treatment, [])
    excludable = d_separated(dag, instrument, outcome, [treatment])
    valid = relevant and excludable
    if valid:
        reason = "Valid IV: relevant and excludable"
    elif not relevant:
        reason = f"Invalid IV: {instrument} is d-separated from {treatment} (no relevance)"
    else:
        reason = (f"Invalid IV: {instrument} is NOT d-separated from {outcome} "
                  f"given {treatment} (exclusion violated)")
    return IvValidityResult(instrument, treatment, outcome, relevant, excludable, valid, reason)


def identify_all_effects(dag: nx.DiGraph) -> list[IdentificationResult]:
    """Backdoor-first, IV-fallback identification over all factor→return pairs.

    Port of identify.rs::identify_all_effects.
    """
    treatments = [
        n for n in dag.nodes
        if _kind(dag, n) in (NodeKind.GLOBAL_FACTOR, NodeKind.MACRO_FACTOR)
    ]
    outcomes = [n for n in dag.nodes if _kind(dag, n) == NodeKind.ASSET_RETURN]
    instruments = [n for n in dag.nodes if _kind(dag, n) == NodeKind.INSTRUMENT]

    results: list[IdentificationResult] = []
    for treatment in treatments:
        for outcome in outcomes:
            adj = backdoor_adjustment_set(dag, treatment, outcome)
            if adj is not None:
                results.append(IdentificationResult(
                    treatment, outcome, True, IdentificationMethod.BACKDOOR,
                    adjustment_set=adj, reason="Identified via backdoor criterion",
                ))
                continue
            found = False
            for iv in instruments:
                if check_iv_validity(dag, iv, treatment, outcome).valid:
                    results.append(IdentificationResult(
                        treatment, outcome, True, IdentificationMethod.IV,
                        adjustment_set=[iv], instrument=iv,
                        reason=f"Identified via IV: {iv}",
                    ))
                    found = True
                    break
            if not found:
                results.append(IdentificationResult(
                    treatment, outcome, False, IdentificationMethod.NOT_IDENTIFIED,
                    reason="No valid backdoor set or instrument found",
                ))
    return results
