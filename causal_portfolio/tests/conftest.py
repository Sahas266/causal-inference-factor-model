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


@pytest.fixture(autouse=True)
def _isolate_runtime_state(monkeypatch, tmp_path):
    """Keep every test out of the operator's live ~/.cpcm-execution tree.

    execute_plan()/send_pnl_update() call trace.* and write_control_panel()
    without an explicit directory, so they resolve audit.LOG_DIR (read
    dynamically by trace.active_path) and CPCM_EXECUTION_CONTROL_PANEL_DIR.
    Without this fixture a test run rotates the live rebalance-current
    database and overwrites the real control panel with synthetic data.
    Tests that need their own directory still monkeypatch it themselves;
    this is the floor, not a replacement.
    """
    from causal_portfolio.execution import audit

    monkeypatch.setattr(audit, "LOG_DIR", tmp_path / "cpcm-logs")
    monkeypatch.setenv(
        "CPCM_EXECUTION_CONTROL_PANEL_DIR", str(tmp_path / "cpcm-panel")
    )
