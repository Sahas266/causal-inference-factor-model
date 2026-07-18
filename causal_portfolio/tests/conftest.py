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
