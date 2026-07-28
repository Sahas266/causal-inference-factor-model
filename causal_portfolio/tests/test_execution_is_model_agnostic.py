"""The execution layer must stay model-agnostic.

Any model reaches the exchange through execute_target(TargetSnapshot,
ExecutionConfig). If execution starts importing model, research, or numeric
code, that contract quietly becomes "works for the model that happens to be
vendored alongside it", and the installable library stops standing alone.

These are static import checks on purpose: they fail on the offending line
without needing the Hyperliquid SDK, a network, or a model to be installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXECUTION_DIR = Path(__file__).resolve().parents[1] / "execution"

# Importing any of these from causal_portfolio.execution breaks the boundary.
# numpy/pandas are banned outright: they are not declared dependencies of the
# wheel, so importing one makes an installed cpcm-execution fail at runtime.
FORBIDDEN_ROOTS = {
    "causal_portfolio.data",
    "causal_portfolio.factors",
    "causal_portfolio.solvers",
    "causal_portfolio.backtest",
    "causal_portfolio.models",
    "causal_portfolio.regimes",
    "causal_portfolio.scm",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "streamlit",
    "duckdb",
}


def _module_files() -> list[Path]:
    files = sorted(EXECUTION_DIR.glob("*.py"))
    assert files, f"no execution modules found under {EXECUTION_DIR}"
    return files


def _imported_roots(path: Path) -> set[tuple[str, int]]:
    """Every module name imported by `path`, including inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # Relative imports stay inside the package by construction.
            if node.level == 0 and node.module:
                found.add((node.module, node.lineno))
    return found


def _violations(module: str, lineno: int) -> str | None:
    for root in FORBIDDEN_ROOTS:
        if module == root or module.startswith(root + "."):
            return root
    return None


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_execution_module_imports_no_model_or_research_code(path):
    offenders = [
        f"{path.name}:{lineno} imports {module!r} (banned root {root!r})"
        for module, lineno in sorted(_imported_roots(path))
        if (root := _violations(module, lineno))
    ]
    assert not offenders, (
        "causal_portfolio.execution must stay model-agnostic; move this code "
        "into causal_portfolio.models (or the repo-level dashboard) and reach "
        "execution through execute_target():\n  " + "\n  ".join(offenders)
    )


def test_execution_only_imports_itself_within_the_repo():
    """Execution may import causal_portfolio.execution.* and nothing else here."""
    offenders = []
    for path in _module_files():
        for module, lineno in sorted(_imported_roots(path)):
            if module.startswith("causal_portfolio.") and not module.startswith(
                "causal_portfolio.execution"
            ):
                offenders.append(f"{path.name}:{lineno} imports {module!r}")
    assert not offenders, (
        "execution may only depend on causal_portfolio.execution.*:\n  "
        + "\n  ".join(offenders)
    )


def test_models_are_not_shipped_inside_the_execution_package():
    """A model vendored into the package would ride along in the wheel."""
    stray = [p.name for p in _module_files() if "rppca" in p.name or "model" in p.name]
    assert not stray, (
        f"model code found inside the execution package: {stray}. "
        "Models belong in causal_portfolio/models/."
    )
