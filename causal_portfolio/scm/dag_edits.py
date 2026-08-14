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

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import networkx as nx

from causal_portfolio.scm.graph import EdgeKind, NodeKind


@dataclass(frozen=True)
class DagEdits:
    """Nodes and edges an operator added or removed, vs the generated DAG."""

    added: tuple[tuple[str, str], ...] = ()
    removed: tuple[tuple[str, str], ...] = ()
    # (name, NodeKind.name) — needed to build a DAG from the blank base, where
    # there is no generated structure to draw nodes from.
    added_nodes: tuple[tuple[str, str], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.added and not self.removed and not self.added_nodes

    def with_node(self, name: str, kind: str) -> "DagEdits":
        if any(n == name for n, _ in self.added_nodes):
            return self
        return DagEdits(
            added=self.added,
            removed=self.removed,
            added_nodes=self.added_nodes + ((name, kind),),
        )

    def without_node(self, name: str) -> "DagEdits":
        """Drop a custom node and any edits that referenced it."""
        return DagEdits(
            added=tuple(e for e in self.added if name not in e),
            removed=tuple(e for e in self.removed if name not in e),
            added_nodes=tuple(n for n in self.added_nodes if n[0] != name),
        )

    def with_added(self, source: str, target: str) -> "DagEdits":
        edge = (source, target)
        # `replace` rather than a fresh DagEdits: every field must survive an
        # edge edit, and a positional constructor silently drops new ones.
        if edge in self.removed:   # re-adding a removed edge is a restore
            return replace(self, removed=tuple(e for e in self.removed if e != edge))
        if edge in self.added:
            return self
        return replace(self, added=self.added + (edge,))

    def with_removed(self, source: str, target: str) -> "DagEdits":
        edge = (source, target)
        # Removing an edge the operator just added drops the addition.
        if edge in self.added:
            return replace(self, added=tuple(e for e in self.added if e != edge))
        if edge in self.removed:
            return self
        return replace(self, removed=self.removed + (edge,))

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
    for name, kind in edits.added_nodes:
        if name not in edited:
            edited.add_node(
                name,
                kind=_node_kind(kind),
                asset=None,
                operator_added=True,
            )
    for source, target in edits.removed:
        if edited.has_edge(source, target):
            edited.remove_edge(source, target)
    for source, target in edits.added:
        if source not in edited or target not in edited:
            continue
        if edited.has_edge(source, target):
            continue
        problem = validate_new_edge(edited, source, target)
        if problem:
            raise ValueError(f"cannot apply saved edge {source} -> {target}: {problem}")
        edited.add_edge(source, target, kind=EdgeKind.CAUSAL, operator_added=True)
    return edited


def _node_kind(name: str) -> NodeKind:
    """Resolve a stored kind name, tolerating one written by an older build."""
    try:
        return NodeKind[name]
    except KeyError:
        return NodeKind.GLOBAL_FACTOR


def validate_new_node(dag: nx.DiGraph, name: str) -> str | None:
    """Return why this node cannot be added, or None when it is legal."""
    cleaned = name.strip()
    if not cleaned:
        return "node name cannot be empty"
    if cleaned != name:
        return "node name cannot start or end with whitespace"
    if cleaned in dag:
        return f"node {cleaned} already exists"
    return None


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
            method = getattr(r, "method", None)
            if hasattr(method, "name"):
                method = method.name.lower()
            out[key] = (
                getattr(r, "identified", None),
                method,
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


# ── DAG sources and saved workspaces ─────────────────────────────────────

#: Named base graphs the dashboard can edit. "Blank" exists so a structure can
#: be built from nothing rather than only by subtracting from a generated one.
DAG_SOURCES: dict[str, str] = {
    "Star DAG (v1, hand-drawn)": "cpcm",
    "Discovered DAG (v2, data-supported)": "discovered",
    "Blank (build from scratch)": "blank",
}

DEFAULT_WORKSPACE_PATH = Path("causal_portfolio/data/dag_workspaces.json")


def build_base_dag(source: str, assets: list[str],
                   selected_drivers: list[str] | None = None) -> nx.DiGraph:
    """Build one of the named base graphs.

    Imported lazily so `dag_edits` stays cheap to import and testable without
    pulling the whole factor registry.
    """
    from causal_portfolio.scm.graph import build_cpcm_dag, build_discovered_dag

    if source == "discovered":
        return build_discovered_dag(assets)
    if source == "blank":
        return nx.DiGraph()
    return build_cpcm_dag(assets, selected_drivers)


def _edits_to_dict(edits: DagEdits) -> dict:
    return {
        "added": [list(e) for e in edits.added],
        "removed": [list(e) for e in edits.removed],
        "added_nodes": [list(n) for n in edits.added_nodes],
    }


def _edits_from_dict(payload: dict) -> DagEdits:
    return DagEdits(
        added=tuple(tuple(e) for e in payload.get("added", [])),
        removed=tuple(tuple(e) for e in payload.get("removed", [])),
        added_nodes=tuple(tuple(n) for n in payload.get("added_nodes", [])),
    )


def _valid_workspace_entry(entry: object) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("edits"), dict):
        return False
    if "source" in entry and not isinstance(entry["source"], str):
        return False
    edits = entry["edits"]
    for field_name in ("added", "removed", "added_nodes"):
        pairs = edits.get(field_name, [])
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(value, str) for value in pair)
            for pair in pairs
        ):
            return False
    return True


def load_workspaces(path: Path | None = None) -> dict[str, dict]:
    """Read saved workspaces. A missing or unreadable file is simply empty.

    Never raises: a corrupt scratch file must not stop the dashboard from
    rendering the generated DAG.
    """
    target = Path(path or DEFAULT_WORKSPACE_PATH)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        name: entry for name, entry in raw.items()
        if _valid_workspace_entry(entry)
    }


def save_workspace(
    name: str,
    source: str,
    edits: DagEdits,
    *,
    path: Path | None = None,
) -> Path:
    """Persist one named workspace, leaving the others intact."""
    if not name.strip():
        raise ValueError("workspace name cannot be empty")
    target = Path(path or DEFAULT_WORKSPACE_PATH)
    workspaces = load_workspaces(target)
    workspaces[name.strip()] = {"source": source, "edits": _edits_to_dict(edits)}
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write via a temp file so an interrupted save cannot truncate saved work.
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(workspaces, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return target


def read_workspace(name: str, path: Path | None = None) -> tuple[str, DagEdits] | None:
    """Return (source, edits) for a saved workspace, or None if absent."""
    entry = load_workspaces(path).get(name)
    if entry is None:
        return None
    return entry.get("source", "cpcm"), _edits_from_dict(entry["edits"])


def delete_workspace(name: str, path: Path | None = None) -> bool:
    target = Path(path or DEFAULT_WORKSPACE_PATH)
    workspaces = load_workspaces(target)
    if name not in workspaces:
        return False
    del workspaces[name]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(workspaces, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return True
