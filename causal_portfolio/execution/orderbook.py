"""Order-book-aware pricing — pure functions, no network, no SDK.

The single-shot `mid ± fixed_slippage_bps` pricing in the rebalancer fails in
two ways we observed live on Hyperliquid:

  1. **Too far from oracle** — the limit price was outside HL's oracle-price
     band (rejected before matching).
  2. **Could not immediately match** — the limit price didn't cross the spread
     (IOC found no resting orders).

On a normal book there's a wide band of prices that both cross AND sit inside
the oracle band. On a thin/divergent book that band can be empty — in which
case no order can fill and the only correct action is to wait and retry on a
fresh book. This module computes a marketable price from the actual L2 book
and tells the caller when no valid price exists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class L2Level:
    px: float
    sz: float


@dataclass(frozen=True)
class L2Book:
    """Top-of-book snapshot. bids descending by px, asks ascending by px."""
    coin: str
    bids: list[L2Level]
    asks: list[L2Level]

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].px if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].px if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return 0.5 * (self.best_bid + self.best_ask)


def marketable_price(
    book: L2Book,
    is_buy: bool,
    size: float,
    *,
    max_band_bps: float = 200.0,
    reference_mid: float | None = None,
) -> float | None:
    """Compute an IOC limit price that crosses the book to fill `size`,
    clamped to within `max_band_bps` of the reference mid.

    For a BUY we walk the asks (ascending) accumulating depth until we cover
    `size`; the price of the deepest level needed is the sweep price. We then
    clamp UP to `mid * (1 + band)`. The returned price must be >= best_ask to
    cross — if the band cap is below best_ask, no crossable in-band price
    exists and we return None.

    For a SELL the mirror: walk bids (descending), clamp DOWN to
    `mid * (1 - band)`, require <= best_bid.

    Returns:
        A float limit price, or None when no price both crosses and sits in
        the oracle band (caller should wait and retry on a fresh book).
    """
    if size <= 0:
        return None
    mid = reference_mid if reference_mid is not None else book.mid
    if mid is None or mid <= 0:
        return None
    band = max_band_bps / 10_000.0

    if is_buy:
        if not book.asks:
            return None
        best_ask = book.asks[0].px
        # Walk asks to find the sweep price that covers `size`.
        cum = 0.0
        sweep_px = best_ask
        for lvl in book.asks:
            sweep_px = lvl.px
            cum += lvl.sz
            if cum >= size:
                break
        band_cap = mid * (1.0 + band)
        if band_cap < best_ask:
            # Even the best ask is outside the band → cannot cross in-band.
            return None
        # Aggressive enough to sweep, but no further than the band cap.
        return min(sweep_px, band_cap)
    else:
        if not book.bids:
            return None
        best_bid = book.bids[0].px
        cum = 0.0
        sweep_px = best_bid
        for lvl in book.bids:
            sweep_px = lvl.px
            cum += lvl.sz
            if cum >= size:
                break
        band_floor = mid * (1.0 - band)
        if band_floor > best_bid:
            return None
        return max(sweep_px, band_floor)


@dataclass(frozen=True)
class FillResult:
    """Parsed outcome of a single IOC submission."""
    filled_size: float
    avg_px: float | None
    error: str | None

    @property
    def is_oracle_reject(self) -> bool:
        return self.error is not None and "oracle" in self.error.lower()

    @property
    def is_no_match(self) -> bool:
        return self.error is not None and "match" in self.error.lower()


def parse_fill_response(resp: dict) -> FillResult:
    """Extract fill size / avg price / error from an HL order response.

    HL response shape:
      {"status": "ok", "response": {"type": "order",
        "data": {"statuses": [{"filled": {"totalSz": "0.001", "avgPx": "...", ...}}]}}}
    or a status entry of {"error": "..."}, or {"resting": {...}} (no immediate fill).
    """
    try:
        statuses = resp["response"]["data"]["statuses"]
    except (KeyError, TypeError):
        return FillResult(0.0, None, f"unparseable response: {resp}")
    if not statuses:
        return FillResult(0.0, None, "empty statuses")
    st = statuses[0]
    if "error" in st:
        return FillResult(0.0, None, str(st["error"]))
    if "filled" in st:
        f = st["filled"]
        return FillResult(float(f.get("totalSz", 0.0)),
                          float(f["avgPx"]) if "avgPx" in f else None,
                          None)
    if "resting" in st:
        # IOC shouldn't rest, but treat as zero-fill / no error.
        return FillResult(0.0, None, None)
    return FillResult(0.0, None, f"unknown status: {st}")
