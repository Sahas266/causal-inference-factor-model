"""Tests for backfill CLI path handling."""

import sys
from pathlib import Path

import backfill


def test_config_dir_default_uses_config_loader_default(monkeypatch):
    """The documented repo-root command should not force cwd/config."""
    monkeypatch.setattr(sys, "argv", ["backfill.py", "--list-providers"])

    args = backfill.parse_args()

    assert args.config_dir is None


def test_script_relative_paths_resolve_from_repo_root(monkeypatch):
    """Paths from backfill.py examples work when invoked from the repo root."""
    repo_root = backfill.BACKFILL_DIR.parent
    monkeypatch.chdir(repo_root)

    resolved_config_dir = backfill.resolve_config_dir("config")
    resolved_endpoint_dir = backfill.resolve_cli_path("config/endpoints/btc")

    assert Path(resolved_config_dir).resolve() == (
        backfill.BACKFILL_DIR / "config"
    ).resolve()
    assert resolved_endpoint_dir.resolve() == (
        backfill.BACKFILL_DIR / "config" / "endpoints" / "btc"
    ).resolve()
