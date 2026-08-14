from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

from scripts import run_provider_backfill, warm_dune_queries
from src.core.utils.config_loader import ConfigLoader


def _write_endpoint(path, endpoint_id, providers):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "endpoint_id": endpoint_id,
            "table": "asset_metrics",
            "primary_keys": ["provider", "asset", "metric", "time"],
            "date_range": {"start": "2021-01-01", "end": "latest"},
            "providers": providers,
        }),
        encoding="utf-8",
    )


def test_provider_filter_preserves_original_instance_ids():
    config = {
        "providers": [
            {"name": "coinmetrics"},
            {"name": "dune", "config": {"query": 1}},
            {
                "name": "dune",
                "provider_instance_id": "named",
                "config": {"query": 2},
            },
        ],
    }

    filtered = run_provider_backfill._filter_for_provider(config, "dune")

    assert [p["provider_instance_id"] for p in filtered["providers"]] == ["1", "named"]
    assert "provider_instance_id" not in config["providers"][1]


def test_recursive_loader_deduplicates_exact_provider_jobs(tmp_path):
    shared = {
        "name": "defillama",
        "enabled": True,
        "priority": 1,
        "config": {"endpoint_type": "protocol/tvl", "params": {"protocol": "aave"}},
    }
    unique = {
        "name": "defillama",
        "enabled": True,
        "priority": 2,
        "config": {"endpoint_type": "protocol/tvl", "params": {"protocol": "lido"}},
    }
    _write_endpoint(tmp_path / "endpoints" / "a.json", "first", [shared])
    _write_endpoint(tmp_path / "endpoints" / "nested" / "b.json", "second", [shared, unique])

    configs = ConfigLoader(str(tmp_path)).load_all_endpoint_configs()

    assert [c["endpoint_id"] for c in configs] == ["first", "second"]
    assert [p["config"]["params"]["protocol"] for p in configs[1]["providers"]] == ["lido"]
    assert configs[1]["providers"][0]["provider_instance_id"] == "1"


def test_warm_dune_returns_failure_when_any_query_fails(monkeypatch):
    client = Mock()
    client.get_query_results.side_effect = RuntimeError("not cached")
    provider = SimpleNamespace(client=client, initialize=lambda _config: None)

    monkeypatch.setattr(warm_dune_queries, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        warm_dune_queries,
        "dune_query_ids",
        lambda _config_dir: [(1, "ok"), (2, "bad")],
    )
    monkeypatch.setattr(warm_dune_queries.ProviderRegistry, "auto_discover", lambda: None)
    monkeypatch.setattr(
        warm_dune_queries.ProviderRegistry,
        "get_provider",
        lambda _name: lambda: provider,
    )
    monkeypatch.setattr(
        warm_dune_queries.ConfigLoader,
        "load_provider_config",
        lambda _self, _name: {},
    )
    monkeypatch.setattr(
        warm_dune_queries,
        "warm",
        lambda _client, query_id, **_kwargs: (
            "QUERY_STATE_COMPLETED" if query_id == 1 else "QUERY_STATE_FAILED"
        ),
    )

    assert warm_dune_queries.main([]) == 1
