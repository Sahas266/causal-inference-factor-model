"""Tests for the narrow DAG v2 upstream refresh command."""

from scripts.backfill_causal_sources import load_causal_source_configs


def test_causal_source_refresh_is_exact_and_dynamic():
    configs = load_causal_source_configs()

    assert len(configs) == 3
    assets = set()
    for config in configs:
        assert config["date_range"]["end"] == "latest"
        assert len(config["providers"]) == 1
        provider = config["providers"][0]
        assert provider["name"] == "coinmetrics"
        params = provider["config"]["params"]
        assert "FeeTotNtv" in params["metrics"].split(",")
        assets.add(params["assets"])
    assert assets == {"btc", "doge", "eth"}
