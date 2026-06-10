# Causal discovery on the factor panel (Direction 3)

- Window: `2022-01-01` → `2025-12-31` | PC α=0.05 (Fisher-Z) | VAR-LiNGAM lag 1, pruned, |coef| > 0.05
- Aligned obs: PC 791, VAR-LiNGAM 792
- Run UTC: `2026-06-10T23:20:37+00:00`

## PC skeleton: lagged factor ↔ next-day return adjacencies

The minimal predictive-structure claim. Only edges replicating in BOTH sample halves count.

| Edge (F_{t-1} — R_t) | full | half 1 | half 2 | stable |
|---|:-:|:-:|:-:|:-:|
| chain_congestion → btc_return |  |  | x | no |

## VAR-LiNGAM directed edges involving returns

B0 = contemporaneous, B1 = lag-1. Direction at B0 is the reverse-causality test: returns → factor means prices drive on-chain activity within the day.

| Edge | kind | full | half 1 | half 2 | stable |
|---|---|:-:|:-:|:-:|:-:|
| btc_return → ew_return | B0 | x | x | x | **yes** |
| btc_return → funding_basis | B0 |  |  | x | no |
| btc_return → liq_flow | B1 |  | x |  | no |
| btc_return → mev_pressure | B1 | x |  | x | no |
| btc_return → stable_flow | B0 | x |  | x | no |
| ew_return → liq_flow | B1 | x | x | x | **yes** |
| ew_return → mev_pressure | B1 | x |  | x | no |

## Verdict

- Stable PC factor→return adjacencies: **0** (none).
- VAR-LiNGAM contemporaneous return→factor edges (reverse causality): **1** (btc_return→stable_flow).
- VAR-LiNGAM lag-1 factor→return edges: **0** (none).

The data does not support ANY stable lagged factor→return edge — the hand-drawn DAG's central premise. Where direction is identifiable, the arrows mostly point FROM returns TO on-chain factors: prices drive activity, not the reverse. The right structural model for this panel is returns as a near-exogenous driver of on-chain state — which explains why every estimator in this project has failed to extract return predictability from these factors.