"""Operator edits to a CPCM DAG, kept Streamlit-free so they can be tested.

The dashboard lets an operator add and remove edges to ask "what if this
arrow were not there?" without editing `graph.py`. Edits are stored as a
small declarative record rather than a mutated graph, so they survive a
pipeline re-run, can be shown back to the operator, and can be exported.

Nothing here mutates the caller's graph — `apply_edits` returns a copy. The
acyclicity check matters: identification, d-separation and backdoor search
all assume a DAG, and a cycle would make `identify_all_effects` meaningless
rather than merely wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from causal_portfolio.scm.graph import EdgeKind


@dataclass(frozen=True)
class DagEdits:
    """Edges an operator added or removed, relative to the generated DAG."""

    added: tuple[tuple[str, str], ...] = ()
    removed: tuple[tuple[str, str], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.added and not self.removed

    def with_added(self, source: str, target: str) -> "DagEdits":
        edge = (source, target)
        # Re-adding a previously removed edge is a restore, not a new edge.
        if edge in self.removed:
            return DagEdits(
                added=self.added,
                removed=tuple(e for e in self.removed if e != edge),
            )
        if edge in self.added:
            return self
        return DagEdits(added=self.added + (edge,), removed=self.removed)

    def with_removed(self, source: str, target: str) -> "DagEdits":
        edge = (source, target)
        # Removing an edge the operator just added simply drops the addition.
        if edge in self.added:
            return DagEdits(
                added=tuple(e for e in self.added if e != edge),
                removed=self.removed,
            )
        if edge in self.removed:
            return self
        return DagEdits(added=self.added, removed=self.removed + (edge,))

    def cleared(self) -> "DagEdits":
        return DagEdits()


def apply_edits(dag: nx.DiGraph, edits: DagEdits) -> nx.DiGraph:
    """Return a copy of `dag` with the operator's edits applied.

    Removals are applied before additions so that a re-added edge wins, and
    both are tolerant of edges that no longer exist in a regenerated DAG —
    an edit recorded against an older asset/driver selection must not crash
    the dashboard.
    """
    edited = dag.copy()
    for source, target in edits.removed:
        if edited.has_edge(source, target):
            edited.remove_edge(source, target)
    for source, target in edits.added:
        if source in edited and target in edited:
            edited.add_edge(source, target, kind=EdgeKind.CAUSAL, operator_added=True)
    return edited


def validate_new_edge(dag: nx.DiGraph, source: str, target: str) -> str | None:
    """Return why this edge cannot be added, or None when it is legal."""
    if source == target:
        return "an edge cannot point at its own node"
    if source not in dag:
        return f"unknown node: {source}"
    if target not in dag:
        return f"unknown node: {target}"
    if dag.has_edge(source, target):
        return f"{source} -> {target} already exists"
    # nx.has_path is the cheap acyclicity test here: adding source -> target
    # creates a cycle exactly when target already reaches source.
    if nx.has_path(dag, target, source):
        return (
            f"{source} -> {target} would create a cycle "
            f"({target} already reaches {source})"
        )
    return None


def edge_rows(dag: nx.DiGraph, edits: DagEdits) -> list[dict]:
    """Edges of the edited DAG, tagged with their origin for display."""
    added = set(edits.added)
    rows = []
    for source, target, data in apply_edits(dag, edits).edges(data=True):
        kind = data.get("kind")
        rows.append({
            "source": source,
            "target": target,
            "kind": kind.name.title() if hasattr(kind, "name") else str(kind),
            "lag": data.get("lag", ""),
            "origin": "operator" if (source, target) in added else "generated",
        })
    return sorted(rows, key=lambda r: (r["source"], r["target"]))


def diff_identification(before: list, after: list) -> dict[str, list[str]]:
    """Summarise how identification changed between two effect lists.

    Takes `IdentificationResult` sequences from `identify_all_effects`. The
    point of editing a DAG is to see what an arrow buys you, so the answer
    that matters is which effects became or stopped being identifiable, and
    which switched strategy (backdoor vs IV).
    """
    def index(results):
        out = {}
        for r in results:
            key = f"{getattr(r, 'treatment', '?')} -> {getattr(r, 'outcome', '?')}"
            out[key] = (
                getattr(r, "identifiable", None),
                getattr(r, "strategy", None),
            )
        return out

    a, b = index(before), index(after)
    gained, lost, changed = [], [], []
    for key in sorted(set(a) | set(b)):
        old, new = a.get(key), b.get(key)
        if old == new:
            continue
        if old is None:
            gained.append(f"{key} (new)")
        elif new is None:
            lost.append(f"{key} (gone)")
        elif old[0] != new[0]:
            (gained if new[0] else lost).append(key)
        else:
            changed.append(f"{key}: {old[1]} -> {new[1]}")
    return {"gained": gained, "lost": lost, "changed": changed}
