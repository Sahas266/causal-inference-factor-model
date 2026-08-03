"""Tests for the pure rebalancer logic.

No network. All inputs are synthetic AccountState/AssetMeta/mids constructed
in-test. Covers:
  - asset mapping + unlisted-coin handling
  - position caps
  - long-only enforcement
  - dust filtering at both pre- and post-rounding stages
  - size rounding to sz_decimals
  - single-trade cap enforcement
  - close-out behavior for coins held but not in target weights
  - slippage factor sign convention
  - cloid determinism + uniqueness
  - empty inputs
"""

from __future__ import annotations

import pytest

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.rebalancer import (
    _apply_position_caps,
    _make_cloid,
    _round_size,
    plan_rebalance,
)
from causal_portfolio.execution.types import (
    AccountState,
    AssetMeta,
    Position,
    SkipReason,
)


# ── helpers ─────────────────────────────────────────────────────────


def _meta(coin: str, sz_decimals: int = 4) -> AssetMeta:
    return AssetMeta(coin=coin, sz_decimals=sz_decimals,
                     max_leverage=10, min_size=10 ** -sz_decimals)


def _state(equity: float = 10_000.0, positions: dict | None = None) -> AccountState:
    return AccountState(
        address="0xtest",
        account_value_usd=equity,
        margin_used_usd=0.0,
        positions=positions or {},
    )


def _pos(coin: str, size: float, px: float) -> Position:
    return Position(coin=coin, size=size, entry_px=px, notional_usd=size * px)


@pytest.fixture
def basic_meta():
    return {
        "BTC": _meta("BTC", sz_decimals=5),
        "ETH": _meta("ETH", sz_decimals=4),
        "SOL": _meta("SOL", sz_decimals=2),
    }


@pytest.fixture
def basic_mids():
    return {"BTC": 60_000.0, "ETH": 3_000.0, "SOL": 150.0}


# ── _round_size ─────────────────────────────────────────────────────


def test_round_size_truncates_toward_zero():
    assert _round_size(0.123456, 4) == 0.1234
    assert _round_size(-0.123456, 4) == -0.1234
    assert _round_size(0.99999, 4) == 0.9999  # truncates, doesn't round up
    assert _round_size(0, 4) == 0.0


def test_round_size_zero_decimals():
    assert _round_size(3.7, 0) == 3.0
    assert _round_size(-3.7, 0) == -3.0


# ── _apply_position_caps ────────────────────────────────────────────


def test_position_caps_preserves_sign():
    weights = {"BTC": 0.5, "ETH": -0.4, "SOL": 0.1}
    capped, hit = _apply_position_caps(weights, cap_pct=0.30)
    assert capped == {"BTC": 0.30, "ETH": -0.30, "SOL": 0.1}
    assert set(hit) == {"BTC", "ETH"}


def test_position_caps_no_op_when_under():
    weights = {"BTC": 0.1, "ETH": -0.1}
    capped, hit = _apply_position_caps(weights, cap_pct=0.30)
    assert capped == weights
    assert hit == []


# ── _make_cloid ─────────────────────────────────────────────────────


def test_cloid_is_deterministic():
    a = _make_cloid(1234, "BTC", True, 0.001)
    b = _make_cloid(1234, "BTC", True, 0.001)
    assert a == b
    assert a.startswith("0x") and len(a) == 34


def test_cloid_changes_with_inputs():
    base = _make_cloid(1234, "BTC", True, 0.001)
    assert _make_cloid(1235, "BTC", True, 0.001) != base
    assert _make_cloid(1234, "ETH", True, 0.001) != base
    assert _make_cloid(1234, "BTC", False, 0.001) != base
    assert _make_cloid(1234, "BTC", True, 0.002) != base


# ── plan_rebalance ──────────────────────────────────────────────────


def test_basic_long_only_rebalance(basic_meta, basic_mids):
    """Going from cash to 50% BTC + 30% ETH on $10k equity."""
    cfg = ExecutionConfig(dry_run=True, max_position_pct=1.0, max_single_trade_pct=1.0)
    plan = plan_rebalance(
        target_weights={"btc": 0.5, "eth": 0.3},
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    coins = {o.coin for o in plan.orders}
    assert coins == {"BTC", "ETH"}
    btc_order = next(o for o in plan.orders if o.coin == "BTC")
    eth_order = next(o for o in plan.orders if o.coin == "ETH")
    assert btc_order.is_buy and eth_order.is_buy
    assert btc_order.size == pytest.approx(0.0833, rel=1e-3)  # $5000 / $60k
    assert eth_order.size == pytest.approx(1.0, rel=1e-3)     # $3000 / $3000


def test_unlisted_coin_skipped(basic_meta, basic_mids):
    cfg = ExecutionConfig(dry_run=True, max_position_pct=1.0, max_single_trade_pct=1.0)
    plan = plan_rebalance(
        target_weights={"btc": 0.3, "wlfi": 0.2},  # wlfi not in HL universe
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    skipped_coins = {coin for coin, _, _ in plan.skipped}
    assert "wlfi" in skipped_coins
    reasons = {r for c, r, _ in plan.skipped if c == "wlfi"}
    assert SkipReason.NOT_LISTED in reasons


def test_tiny_weight_fires_order_no_dust_filter(basic_meta, basic_mids):
    """Without a dust filter, even a tiny weight should fire an order
    (subject to single-trade cap, exchange minimum notional, and tick size)."""
    cfg = ExecutionConfig(dry_run=True)
    plan = plan_rebalance(
        target_weights={"btc": 0.0015},  # $15 on 10k equity, safely above min notional
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    # No arbitrary dust filter: the trade goes out once exchange floors are met.
    assert len(plan.orders) == 1
    assert plan.orders[0].coin == "BTC"


def test_no_trade_band_skips_tiny_same_direction_resize(basic_meta, basic_mids):
    cfg = ExecutionConfig(
        dry_run=True,
        max_position_pct=1.0,
        max_single_trade_pct=1.0,
        min_position_change_pct=0.10,
    )
    state = _state(10_000, positions={"BTC": _pos("BTC", size=0.1, px=60_000)})

    plan = plan_rebalance(
        target_weights={"btc": 0.59},
        state=state,
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )

    assert plan.orders == []
    assert ("BTC", SkipReason.BELOW_NO_TRADE_BAND) in {
        (coin, reason) for coin, reason, _ in plan.skipped
    }


def test_no_trade_band_default_preserves_same_direction_resize(basic_meta, basic_mids):
    cfg = ExecutionConfig(
        dry_run=True,
        max_position_pct=1.0,
        max_single_trade_pct=1.0,
    )
    state = _state(10_000, positions={"BTC": _pos("BTC", size=0.1, px=60_000)})

    plan = plan_rebalance(
        target_weights={"btc": 0.59},
        state=state,
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )

    assert [order.coin for order in plan.orders] == ["BTC"]


def test_no_trade_band_never_blocks_direction_flip(basic_meta, basic_mids):
    cfg = ExecutionConfig(
        dry_run=True,
        max_position_pct=1.0,
        max_single_trade_pct=1.0,
        min_order_notional_usd=0.0,
        min_position_change_pct=10.0,
    )
    state = _state(10_000, positions={"ETH": _pos("ETH", size=0.001, px=3_000)})

    plan = plan_rebalance(
        target_weights={"eth": -0.0001},
        state=state,
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )

    eth_order = next(order for order in plan.orders if order.coin == "ETH")
    assert not eth_order.is_buy
    assert not eth_order.reduce_only


def test_below_min_notional_is_skipped_before_exchange_reject(basic_meta, basic_mids):
    """Small non-reducing sleeves should be visible in dry-run as skips."""
    cfg = ExecutionConfig(dry_run=True, min_order_notional_usd=10.0)
    plan = plan_rebalance(
        target_weights={"eth": 0.0005},  # $5 on $10k equity
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )

    assert plan.orders == []
    assert ("ETH", SkipReason.BELOW_MIN_NOTIONAL) in {
        (coin, reason) for coin, reason, _ in plan.skipped
    }


def test_full_reduce_close_allowed_below_min_notional(basic_meta, basic_mids):
    """Tiny full closes are allowed so cleanup can flatten dust positions."""
    cfg = ExecutionConfig(
        dry_run=True,
        min_order_notional_usd=10.0,
        max_single_trade_pct=1.0,
        min_position_change_pct=10.0,
    )
    state = _state(10_000, positions={"ETH": _pos("ETH", size=0.001, px=3_000)})
    plan = plan_rebalance(
        target_weights={"eth": 0.0},
        state=state,
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )

    eth_order = next(o for o in plan.orders if o.coin == "ETH")
    assert not eth_order.is_buy
    assert eth_order.reduce_only
    assert eth_order.size == pytest.approx(0.001)


def test_long_only_zeros_negative_weights(basic_meta, basic_mids):
    cfg = ExecutionConfig(dry_run=True, long_only=True,
                          max_position_pct=1.0, max_single_trade_pct=1.0)
    plan = plan_rebalance(
        target_weights={"btc": 0.3, "eth": -0.2},
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    coins = {o.coin for o in plan.orders}
    assert "ETH" not in coins  # negative weight zeroed
    assert "BTC" in coins
    skip_reasons = {(c, r) for c, r, _ in plan.skipped}
    assert ("ETH", SkipReason.LONG_ONLY_VIOLATION) in skip_reasons


def test_position_cap_clips_oversize_weight(basic_meta, basic_mids):
    cfg = ExecutionConfig(dry_run=True, max_position_pct=0.20, max_single_trade_pct=1.0)
    plan = plan_rebalance(
        target_weights={"btc": 0.9},   # would be $9000 → capped to $2000
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    btc_order = next(o for o in plan.orders if o.coin == "BTC")
    expected_size = (10_000 * 0.20) / 60_000
    # Truncated to 5 decimals → 0.03333
    assert btc_order.size == pytest.approx(0.03333, abs=1e-5)


def test_close_out_position_not_in_targets(basic_meta, basic_mids):
    """Holding SOL, target weights only mention BTC → SOL must be closed."""
    cfg = ExecutionConfig(dry_run=True, max_position_pct=1.0, max_single_trade_pct=1.0)
    state = _state(10_000, positions={"SOL": _pos("SOL", size=10, px=150)})
    plan = plan_rebalance(
        target_weights={"btc": 0.3},
        state=state,
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    sol_order = next((o for o in plan.orders if o.coin == "SOL"), None)
    assert sol_order is not None, "should emit a sell to close SOL"
    assert not sol_order.is_buy
    assert sol_order.size == pytest.approx(10.0, abs=1e-2)


def test_long_to_short_flip(basic_meta, basic_mids):
    """Holding +1 BTC long, target -50% BTC on $200k equity → flip to short."""
    cfg = ExecutionConfig(
        dry_run=True,
        max_position_pct=0.50,
        max_single_trade_pct=1.0,
    )
    # Use $200k equity so the $160k flip trade fits under 100% of equity cap
    state = _state(200_000, positions={"BTC": _pos("BTC", size=1.0, px=60_000)})
    plan = plan_rebalance(
        target_weights={"btc": -0.5},
        state=state,
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    btc_order = next(o for o in plan.orders if o.coin == "BTC")
    assert not btc_order.is_buy
    # Current notional = +$60k; target = -$100k (50% of $200k); delta = -$160k
    # → size 160k/60k = 2.6667 BTC sold
    assert btc_order.size == pytest.approx(2.66666, abs=1e-4)


def test_single_trade_cap_blocks_oversize_trade(basic_meta, basic_mids):
    cfg = ExecutionConfig(
        dry_run=True,
        max_single_trade_pct=0.05,  # $500 cap on $10k equity
        max_position_pct=1.0,       # don't let position cap be the limiter
    )
    plan = plan_rebalance(
        target_weights={"btc": 0.3},  # $3000 trade → exceeds $500 cap
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    assert plan.orders == []
    skip_reasons = {(c, r) for c, r, _ in plan.skipped}
    assert ("BTC", SkipReason.EXCEEDS_TRADE_CAP) in skip_reasons


def test_precision_warning_when_rounding_shrinks_trade(basic_mids):
    """Coarse sz_decimals → trade shrinks >10% from target → warning in notes."""
    # Whole-BTC granularity. Target $80k → 1.3333 BTC → rounded to 1.0 BTC ($60k).
    # That's a 25% shrink — over the 10% threshold.
    meta = {"BTC": AssetMeta("BTC", sz_decimals=0, max_leverage=10, min_size=1.0)}
    cfg = ExecutionConfig(
        dry_run=True, max_position_pct=1.0, max_single_trade_pct=1.0,
    )
    plan = plan_rebalance(
        target_weights={"btc": 0.8},
        state=_state(100_000),
        mids=basic_mids,
        meta=meta,
        config=cfg,
        timestamp_ms=1000,
    )
    assert any("precision warning" in n and "BTC" in n for n in plan.notes), (
        f"expected precision warning, got notes: {plan.notes}"
    )
    # Trade still goes out at the rounded size, not skipped
    btc_order = next(o for o in plan.orders if o.coin == "BTC")
    assert btc_order.size == 1.0


def test_no_precision_warning_when_rounding_is_minor(basic_meta, basic_mids):
    """Normal sz_decimals → tiny rounding shouldn't trigger the warning."""
    cfg = ExecutionConfig(dry_run=True, max_position_pct=1.0, max_single_trade_pct=1.0)
    plan = plan_rebalance(
        target_weights={"btc": 0.5},
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,  # BTC sz_decimals=5
        config=cfg,
        timestamp_ms=1000,
    )
    assert not any("precision warning" in n for n in plan.notes)


def test_size_rounded_to_zero_skipped(basic_mids):
    """If sz_decimals is too coarse, tiny notional rounds to 0."""
    meta = {"BTC": _meta("BTC", sz_decimals=0)}  # whole BTC only
    cfg = ExecutionConfig(dry_run=True)
    plan = plan_rebalance(
        target_weights={"btc": 0.005},  # $50 → 0.000833 BTC → rounds to 0
        state=_state(10_000),
        mids=basic_mids,
        meta=meta,
        config=cfg,
        timestamp_ms=1000,
    )
    assert plan.orders == []
    skip_reasons = {(c, r) for c, r, _ in plan.skipped}
    assert any(r == SkipReason.BELOW_MIN_SIZE for _, r in skip_reasons)


def test_slippage_factor_sign_convention(basic_meta, basic_mids):
    cfg = ExecutionConfig(dry_run=True, slippage_bps=50,
                          max_position_pct=1.0, max_single_trade_pct=1.0)
    plan = plan_rebalance(
        target_weights={"btc": 0.5},
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    btc_order = next(o for o in plan.orders if o.coin == "BTC")
    # buy → limit > mid by 50 bps
    assert btc_order.limit_px == pytest.approx(60_000 * 1.005)


def test_slippage_for_sell(basic_meta, basic_mids):
    cfg = ExecutionConfig(dry_run=True, slippage_bps=50, max_position_pct=1.0,
                          max_single_trade_pct=1.0)
    state = _state(10_000, positions={"BTC": _pos("BTC", size=0.1, px=60_000)})
    plan = plan_rebalance(
        target_weights={"btc": 0.0},  # close existing
        state=state,
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    btc_order = next(o for o in plan.orders if o.coin == "BTC")
    assert not btc_order.is_buy
    assert btc_order.limit_px == pytest.approx(60_000 * 0.995)


def test_empty_target_with_existing_positions(basic_meta, basic_mids):
    """No targets supplied → close everything."""
    cfg = ExecutionConfig(dry_run=True, max_single_trade_pct=1.0)
    state = _state(10_000, positions={
        "BTC": _pos("BTC", size=0.05, px=60_000),
        "ETH": _pos("ETH", size=1.0, px=3_000),
    })
    plan = plan_rebalance(
        target_weights={},
        state=state, mids=basic_mids, meta=basic_meta,
        config=cfg, timestamp_ms=1000,
    )
    coins = {o.coin for o in plan.orders}
    assert coins == {"BTC", "ETH"}
    assert all(not o.is_buy for o in plan.orders)


def test_no_op_when_already_on_target(basic_meta, basic_mids):
    """If current state matches target, no orders emitted."""
    cfg = ExecutionConfig(dry_run=True)
    # 0.1 BTC * $60k = $6000 = 0.6 weight on $10k equity
    state = _state(10_000, positions={"BTC": _pos("BTC", size=0.1, px=60_000)})
    plan = plan_rebalance(
        target_weights={"btc": 0.6},
        state=state, mids=basic_mids, meta=basic_meta,
        config=cfg, timestamp_ms=1000,
    )
    # Pos cap of 0.30 will clip down to $3000 target → emit a sell.
    # With max_position_pct=1.0 we should get nothing.
    cfg_ok = ExecutionConfig(dry_run=True, max_position_pct=1.0)
    plan_ok = plan_rebalance(
        target_weights={"btc": 0.6},
        state=state, mids=basic_mids, meta=basic_meta,
        config=cfg_ok, timestamp_ms=1000,
    )
    assert plan_ok.orders == []


def test_leverage_multiplies_notional(basic_meta, basic_mids):
    cfg = ExecutionConfig(dry_run=True, leverage=2.0, max_position_pct=1.0,
                          max_single_trade_pct=1.0)
    plan = plan_rebalance(
        target_weights={"btc": 0.5},
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    btc_order = next(o for o in plan.orders if o.coin == "BTC")
    # With 2x leverage and 0.5 weight on $10k → $10k notional → 0.16667 BTC
    assert btc_order.size == pytest.approx(0.16666, abs=1e-4)


def test_asset_leverage_limit_blocks_risk_increase(basic_mids):
    """Do not submit an order whose target exposure exceeds HL's asset cap."""
    meta = {"BTC": AssetMeta("BTC", sz_decimals=5, max_leverage=2, min_size=1e-5)}
    cfg = ExecutionConfig(
        dry_run=True,
        leverage=5.0,
        max_position_pct=1.0,
        max_single_trade_pct=1.0,
    )
    plan = plan_rebalance(
        target_weights={"btc": 0.8},  # $40k target on $10k equity = 4x
        state=_state(10_000),
        mids=basic_mids,
        meta=meta,
        config=cfg,
        timestamp_ms=1000,
    )

    assert plan.orders == []
    assert ("BTC", SkipReason.EXCEEDS_LEVERAGE_LIMIT) in {
        (coin, reason) for coin, reason, _ in plan.skipped
    }


def test_asset_leverage_limit_allows_reduce_only_deleveraging(basic_mids):
    """Existing over-limit exposure can still be reduced with reduce-only orders."""
    meta = {"BTC": AssetMeta("BTC", sz_decimals=5, max_leverage=2, min_size=1e-5)}
    state = _state(
        10_000,
        positions={"BTC": _pos("BTC", size=0.5, px=60_000)},  # $30k = 3x
    )
    cfg = ExecutionConfig(
        dry_run=True,
        leverage=5.0,
        max_position_pct=1.0,
        max_single_trade_pct=1.0,
    )
    plan = plan_rebalance(
        target_weights={"btc": 0.5},  # $25k = 2.5x, still above cap but lower risk
        state=state,
        mids=basic_mids,
        meta=meta,
        config=cfg,
        timestamp_ms=1000,
    )

    btc_order = next(o for o in plan.orders if o.coin == "BTC")
    assert not btc_order.is_buy
    assert btc_order.reduce_only
    assert not any(reason == SkipReason.EXCEEDS_LEVERAGE_LIMIT for _, reason, _ in plan.skipped)


def test_closing_position_without_asset_meta_is_skipped_not_crashed(basic_mids):
    """A stale/unsupported held coin should surface as skipped, not KeyError."""
    cfg = ExecutionConfig(dry_run=True, max_single_trade_pct=1.0)
    state = _state(10_000, positions={"DOGE": _pos("DOGE", size=100.0, px=0.1)})
    mids = {**basic_mids, "DOGE": 0.1}
    plan = plan_rebalance(
        target_weights={},
        state=state,
        mids=mids,
        meta={},
        config=cfg,
        timestamp_ms=1000,
    )

    assert plan.orders == []
    assert ("DOGE", SkipReason.NOT_LISTED) in {
        (coin, reason) for coin, reason, _ in plan.skipped
    }


def test_plan_summary_string(basic_meta, basic_mids):
    cfg = ExecutionConfig(dry_run=True)
    plan = plan_rebalance(
        target_weights={"btc": 0.3, "eth": 0.2},
        state=_state(10_000),
        mids=basic_mids,
        meta=basic_meta,
        config=cfg,
        timestamp_ms=1000,
    )
    s = plan.summary()
    assert "RebalancePlan" in s
    assert "$5,000" in s or "$5000" in s  # $3k + $2k gross


def test_config_validation():
    with pytest.raises(ValueError, match="leverage"):
        ExecutionConfig(leverage=0)
    with pytest.raises(ValueError, match="max_position_pct"):
        ExecutionConfig(max_position_pct=1.5)
    with pytest.raises(ValueError, match="slippage_bps"):
        ExecutionConfig(slippage_bps=-1)
    with pytest.raises(ValueError, match="min_order_notional_usd"):
        ExecutionConfig(min_order_notional_usd=-1)
    with pytest.raises(ValueError, match="min_position_change_pct"):
        ExecutionConfig(min_position_change_pct=-1)
    with pytest.raises(ValueError, match="min_position_change_pct"):
        ExecutionConfig(min_position_change_pct=float("nan"))
    with pytest.raises(ValueError, match="min_position_change_pct"):
        ExecutionConfig(min_position_change_pct=float("inf"))


def test_twap_config_validation():
    ExecutionConfig(twap_minutes=5.0, twap_slices=3)
    with pytest.raises(ValueError, match="twap_minutes"):
        ExecutionConfig(twap_minutes=-1.0)
    with pytest.raises(ValueError, match="twap_slices"):
        ExecutionConfig(twap_slices=0)
    with pytest.raises(ValueError, match="max_twap_minutes"):
        ExecutionConfig(twap_minutes=31.0)


def test_is_live_mainnet_gate():
    assert not ExecutionConfig(dry_run=True, testnet=True).is_live_mainnet()
    assert not ExecutionConfig(dry_run=False, testnet=True).is_live_mainnet()
    assert not ExecutionConfig(dry_run=True, testnet=False).is_live_mainnet()
    assert ExecutionConfig(dry_run=False, testnet=False).is_live_mainnet()
