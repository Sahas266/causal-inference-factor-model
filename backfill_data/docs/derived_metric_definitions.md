# Derived metric definitions

Every `provider='derived'` series in `asset_metrics`, with the exact formula
and input source. Written because these had to be **reverse-engineered from
stored values** — `compute_derived_metrics.py` in the tree only covers two of
them, and the fuller script that produced the rest is not committed.

Each formula below was confirmed by reproducing a known historical value
exactly. Where a definition could not be reproduced it says so; do not guess
one, because a plausible-but-different formula appends a silently divergent
series to its own history and nothing fails.

Validation anchor unless stated otherwise: **btc, 2025-12-30**.

## Conventions that bit us

- **Timestamps are UTC midnight.** Always `SET TimeZone='UTC'` before using
  `time::date` in DuckDB. Without it the session renders local time and every
  date label shifts back a day — all four metrics below mismatched until this
  was fixed, with no error raised.
- **Window offsets are ROWS, not calendar days.** `log(P[i] / P[i-7])`, not
  `price(today) - price(today minus 7 days)`. These agree only when the series
  has no gaps.
- **Price source matters.** Different metrics were computed from different
  providers. Using the "obvious" one gives a close-but-wrong answer: an
  Artemis-based `log_return_7d` gives 1.1469 where the stored value is 1.2088.

## Computable, definitions confirmed

| Metric | Formula | Price/source | Anchor value |
|---|---|---|---|
| `log_return_7d` | `ln(P[i] / P[i-7]) * 100` | CoinMetrics `price` | `1.2088` |
| `log_return_30d` | `ln(P[i] / P[i-30]) * 100` | CoinMetrics `price` | `-2.4345` |
| `sharpe_30d` | `mean(r) / stdev(r, ddof=1) * sqrt(365)` over the last 30 daily log returns | CoinMetrics `price` | `-0.7738` |
| `realized_volatility_7d` | `stdev(daily log returns, 7) * sqrt(365) * 100` | see note | `12.0892` |
| `realized_volatility_30d` | `stdev(daily log returns, 30) * sqrt(365) * 100` | see note | `37.9548` |
| `tvl_net_flow_usd` | `tvl_usd(t) - tvl_usd(t-1)` | DefiLlama `tvl_usd` | `-51,014,710.00` |

`sharpe_30d` uses the **sample** standard deviation (`ddof=1`); the population
form gives `-0.7870` and is wrong against history.

Realized volatility does **not** reproduce from CoinMetrics prices (sample
gives 38.2804, population 37.6370 against a stored 37.9548). It was computed
from a different source — most likely Artemis, which `PRICE_SOURCES` prefers.
Treat its source as unconfirmed. Note this means the historical derived series
were **not all computed from one price provider**.

Producer: `scripts/compute_derived_extra_local.py` (log returns, Sharpe, net
flows) and `scripts/compute_derived_metrics_local.py` (realized volatility,
which reuses the committed `compute_derived_metrics.py` formulas).

## Not computable, and why

| Metric | Definition | Blocker |
|---|---|---|
| `volume_mcap_ratio` | `24h_volume / mc` (confirmed exactly: `39,734,225,800.08 / 1,765,833,387,995.62 = 0.0225020`) | Both inputs are **Artemis-only**. Artemis is frozen at 2026-01-01 and its provider source is not in the tree. Substituting `spot_volume_usd_24h / market_cap_usd` changes the numerator materially (53.96B vs 39.73B) — same name, different meaning. |
| `dex_cex_volume_ratio` | `sum(ETH DEX volume) / abs(eth cex_netflow_usd)` per the committed docstring | The **denominator is now current** (Dune query 6811499 fixed, see below). The **numerator cannot be reproduced**: it required `provider=defillama, asset=eth, metric=volume_usd`, which has zero rows locally. Back-solving `dex = ratio * abs(cex)` on 2025-12-31 gives an implied 1,502,347,908, which matches neither `uni+crv` (1,074,454,905) nor `uniswap+crv` (1,602,724,545). The protocol set is unknown — do not guess it. |
| `beta_to_btc_30d` | presumably `cov(r_asset, r_btc) / var(r_btc)` over 30d | Formula not verified against a stored value; source and window convention unconfirmed. |
| `stablecoin_net_flow_usd` | presumably `Δ` of a stablecoin supply series | Source metric name not identified locally. |

## DefiLlama asset-naming drift

DEX volume is stored under **protocol-named assets**, and the names duplicate:

- `uni` ≡ `uniswap_v3` — Uniswap **v3 only** (746,571,524 on 2026-08-10)
- `uniswap` — **all** Uniswap versions (1,574,205,794 same day)
- `crv` ≡ `curve` — identical values

`uniswap`, `curve` and `uniswap_v3` first appear on **2026-07-06**; the older
canonical names are `uni` and `crv` (see the legacy rename note in CLAUDE.md:
`uniswap→uni`, `curve→crv`). Summing across these labels double-counts.
`backfill_local_duckdb.py` does not apply an asset-name normalization, so new
rows keep whatever `asset` the endpoint config declares.

## Dune: how a series goes stale

Three independent causes, all seen here. Check them in this order.

### 1. Hardcoded date bound in the saved SQL — fixed

Sixteen queries ended at `TIMESTAMP '2026-01-01'` (the UNI one twice). All now
use `CURRENT_DATE`, which also excludes the partial current day:

- ETH: `6811495` `6811496` `6811497` `6811498` `6811499` `6815240` `6815241`
  `6815242` `6815244` `6831658` `6831661` `6831664`
- BNB: `6831685` `6831687` `6831689`
- UNI: `6854380`

### 2. A stale cache, even when the SQL is open-ended

`6830998` / `6830999` already used `CURRENT_DATE` yet BNB was still frozen.
**Editing or re-saving a Dune query does not invalidate its cached results**,
and the provider only reads `GET /query/{id}/results`. `warm_dune_queries.py`
skips anything already cached — right for resuming an interrupted run, wrong
after a SQL edit. Use `--force` whenever the SQL changed.

### 3. No endpoint config, so nothing ever re-fetches it

`6854380` (Uniswap V3 daily LP flows) had 1,703 historical rows per metric but
**no file under `config/endpoints/`**, so no backfill touched it and `uni`
stayed at 2025-12-31. Config added at `config/endpoints/uni/uni_dune_lp_flows.json`.
If a series is stale and its query looks healthy, check that a config exists at
all before debugging anything else.

### 4. Upstream table retired: `6815244` — migrated, with a value change

ETH stablecoin netflow failed on execution before any edit here:

    delta_prod.stablecoins_ethereum.transfers does not exist or it is private

The table was **not deleted** — it was consolidated into `stablecoins_evm.transfers`
and `stablecoins_multichain.transfers`. But both, and `stablecoins_multichain.tokens`,
carry a `prod_exclude` tag: they appear in the catalog and are marked public,
yet executing against them raises the same "does not exist or it is private".
The whole curated stablecoin family is unavailable in the production database,
so a straight rename is not possible.

Migrated instead to `tokens.transfers` (queryable, and already used by query
`6811497`; keeps `from`/`to` as `varbinary`, so the zero-address comparisons are
unchanged) joined to the published contract list
`dune.bnbchain.dataset_stablecoin_tokens_multichain`, whose own description
states it *replaces the private `tokens.erc20_stablecoins`*. The stablecoin
universe therefore comes from a maintained upstream list rather than a
hand-written symbol set.

**This changes the values.** The DefiLlama-derived list is broader than the
retired spell's set, so mint/burn totals run roughly 10–20% higher:

| date | old value | new value |
|---|--:|--:|
| 2025-12-31 | 3,111,842,596 | 3,490,267,067 |
| 2025-12-30 | 3,203,959,576 | 3,655,202,767 |

The series was re-fetched from 2021-01-01 so the **entire history is on the new
definition**. A partial refresh would have left a silent 10–20% step mid-series,
which is the failure mode this document exists to prevent. Anything comparing
`stablecoin_*_usd` against numbers recorded before 2026-08-14 is comparing two
different definitions.

### Operational notes

- The Dune **MCP server dropped mid-call** twice on `getDuneQuery` for
  `6854380`. The REST API with `X-Dune-API-Key` worked; note that REST returns
  the SQL under `query_sql`, while the MCP tool returns it under `query`.
- `POST /execute` with `{"performance": "free"}` is rejected on this plan
  ("performance tier is not available with your subscription"); omit the body
  and let it default.
- Execution is free on this plan but slow — the Uniswap V3 LP-flow query takes
  several minutes, so warm it out of band rather than inside a timed run.

## Providers that cannot be refreshed at all

`artemis` (1.56M rows, frozen 2026-01-01) and `hyperliquid` (621k rows, frozen
2025-12-31) have only `__pycache__` in `src/providers/` — no source, no configs.
Nothing local can advance them, which is also why `volume_mcap_ratio` is stuck.
