import re

import pytest


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    """Never let tests post to a real Telegram channel.

    notify._credentials() falls back to repo-root .env, so a developer who
    configures live notifications would otherwise have every execute_plan
    test spam the channel. Empty values read as unconfigured.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")


@pytest.fixture(scope="session")
def _runtime_state_root(tmp_path_factory):
    """One session directory for the redirected execution runtime tree."""
    return tmp_path_factory.mktemp("cpcm-runtime")


@pytest.fixture(autouse=True)
def _isolate_runtime_state(monkeypatch, _runtime_state_root, request):
    """Keep every test out of the operator's live ~/.cpcm-execution tree.

    execute_plan()/send_pnl_update() call trace.* and write_control_panel()
    without an explicit directory, so they resolve audit.LOG_DIR (read
    dynamically by trace.active_path) and CPCM_EXECUTION_CONTROL_PANEL_DIR.
    Without this fixture a test run rotates the live rebalance-current
    database and overwrites the real control panel with synthetic data.
    Tests that need their own directory still monkeypatch it themselves;
    this is the floor, not a replacement.

    The per-test slot is only a path, never created here: the code under test
    mkdirs lazily, so the vast majority of tests that never touch this tree
    cost nothing. (Depending on `tmp_path` instead would materialize a
    directory for all ~560 tests on every run, which measurably slowed the
    suite and piled up thousands of retained directories.)
    """
    from causal_portfolio.execution import audit, run_logging

    slot = _runtime_state_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)
    monkeypatch.setattr(audit, "LOG_DIR", slot / "logs")
    monkeypatch.setattr(run_logging, "LOG_DIR", slot / "logs")
    monkeypatch.setenv("CPCM_EXECUTION_CONTROL_PANEL_DIR", str(slot / "panel"))
