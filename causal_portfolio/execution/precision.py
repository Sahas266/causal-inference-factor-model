"""Exchange precision rules shared by every order-producing module.

Hyperliquid enforces per-asset size decimals and price tick rules. These
helpers are the single home for that arithmetic; the rebalancer, both
adapters, the maker engine, and the testnet benchmarks all consume them
instead of re-implementing rounding locally.

Two size conventions exist deliberately:

- ``round_size`` truncates toward zero with no epsilon. Used when sizing
  from a fresh float computation (planner USD -> size conversion).
- ``floor_size`` floors with a tiny epsilon (1e-12) so a value that is
  exactly N units up to float error (e.g. 0.00019999999998) still counts
  as N units. Used when re-deriving a size from accumulated fills or
  positions, where representation error would otherwise strand dust.
"""

from __future__ import annotations

import math


def round_price(px: float, sz_decimals: int, is_spot: bool = False) -> float:
    """Round a price to HL's tick rules.

    Mirrors `hyperliquid.exchange.Exchange._slippage_price`:
      - 5 significant figures total
      - Then at most (6 - sz_decimals) decimal places for perps,
        (8 - sz_decimals) for spot.

    Without this, HL rejects orders with "Order has invalid price".
    """
    if px <= 0:
        return px
    five_sig = float(f"{px:.5g}")
    max_decimals = (8 if is_spot else 6) - sz_decimals
    return round(five_sig, max_decimals)


def round_size(size: float, sz_decimals: int) -> float:
    """Round a size to the asset's allowed precision, toward zero."""
    if size == 0:
        return 0.0
    factor = 10 ** sz_decimals
    sign = 1.0 if size > 0 else -1.0
    return sign * (int(abs(size) * factor)) / factor


def floor_size(size: float, sz_decimals: int) -> float:
    """Floor a non-negative size to precision, tolerant of float error."""
    factor = 10 ** sz_decimals
    return math.floor(size * factor + 1e-12) / factor
