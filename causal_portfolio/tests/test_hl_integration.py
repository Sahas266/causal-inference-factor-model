"""Live integration tests against Hyperliquid testnet.

These hit the real testnet API (https://api.hyperliquid-testnet.xyz). They
are READ-ONLY — never submit orders. Skipped automatically if:
  - hyperliquid-python-sdk is not installed
  - HL_ADDRESS/HYPERLIQUID_WALLET_ADDRESS env var is not set

To run:
    # In repo root, with .env containing HYPERLIQUID_WALLET_ADDRESS
    python -m pytest causal_portfolio/tests/test_hl_integration.py -v
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("hyperliquid", reason="hyperliquid-python-sdk not installed")

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.hyperliquid import (
    ADDRESS_ENV_VARS,
    HLAdapter,
    _load_dotenv_once,
    _read_env_chain,
)
from causal_portfolio.execution.rebalancer import plan_rebalance
from causal_portfolio.execution.types import AccountState


_load_dotenv_once()
_HAS_CREDS = bool(_read_env_chain(ADDRESS_ENV_VARS))

requires_creds = pytest.mark.skipif(
    not _HAS_CREDS,
    reason="No HYPERLIQUID_WALLET_ADDRESS in env",
)


@pytest.fixture(scope="module")
def testnet_adapter():
    cfg = ExecutionConfig(testnet=True, dry_run=True)
    return HLAdapter(cfg)


@requires_creds
def test_meta_returns_known_assets(testnet_adapter):
    meta = testnet_adapter.fetch_meta()
    assert len(meta) > 50, f"expected many assets, got {len(meta)}"
    # These three have been on HL since launch and should always be there
    for coin in ["BTC", "ETH", "SOL"]:
        assert coin in meta, f"{coin} missing from testnet listings"
        assert meta[coin].sz_decimals >= 0
        assert meta[coin].max_leverage > 0


@requires_creds
def test_mids_have_sane_prices(testnet_adapter):
    mids = testnet_adapter.fetch_mids()
    btc = mids.get("BTC")
    eth = mids.get("ETH")
    assert btc is not None and btc > 1000, f"BTC mid looks wrong: {btc}"
    assert eth is not None and eth > 100, f"ETH mid looks wrong: {eth}"


@requires_creds
def test_account_state_shape(testnet_adapter):
    """Equity and positions parse correctly even on an empty account."""
    state = testnet_adapter.fetch_state()
    assert state.address.startswith("0x")
    assert state.account_value_usd >= 0
    assert state.margin_used_usd >= 0
    # positions dict should be a dict (possibly empty)
    assert isinstance(state.positions, dict)


@requires_creds
def test_planner_produces_orders_against_live_meta(testnet_adapter):
    """Rebalancer + adapter wired together produce valid orders."""
    state = testnet_adapter.fetch_state()
    mids = testnet_adapter.fetch_mids()
    meta = testnet_adapter.fetch_meta()

    # Use synthetic equity so the test works on an empty testnet account.
    synthetic = AccountState(
        address=state.address,
        account_value_usd=10_000.0,
        margin_used_usd=0.0,
    )
    cfg = ExecutionConfig(
        testnet=True, dry_run=True,
        max_position_pct=1.0, max_single_trade_pct=1.0,
    )
    plan = plan_rebalance(
        target_weights={"btc": 0.05, "eth": 0.04},
        state=synthetic, mids=mids, meta=meta, config=cfg,
    )
    coins_traded = {o.coin for o in plan.orders}
    assert "BTC" in coins_traded, f"BTC missing from orders: {plan.orders}"
    assert "ETH" in coins_traded
    # Limit prices should be within 1% of mids for reasonable slippage
    for o in plan.orders:
        mid = mids[o.coin]
        assert abs(o.limit_px - mid) / mid < 0.01


def test_no_creds_means_skipped():
    """Sanity check: if creds are missing this whole module skips gracefully."""
    if not _HAS_CREDS:
        pytest.skip("expected — no creds")
