# Event studies on on-chain shocks (Direction 2)

Forward returns and forward 5d realized vol after event days vs a moving-block bootstrap null (block=10, 1000 reps, two-sided empirical p). Events defined at day t from data available at day t; outcomes start at t+1 — no overlap.

- Window: `2022-01-01` → `2025-12-31` | basket: `b, t, c, ,, e, t, h, ,, s, o, l, ,, b, n, b, ,, a, v, a, x, ,, u, n, i, ,, a, a, v, e, ,, l, i, n, k, ,, d, o, g, e`
- Run UTC: `2026-06-10T23:18:28+00:00`

| Event | n | Exogeneity | BTC +5d (p) | EW +5d (p) | fwd vol ratio (p) |
|---|---:|---|---:|---:|---:|
| blob_spike | 41 | good | -0.90% (0.44) | -1.50% (0.42) | 1.05 (0.753) |
| stable_mint_shock | 59 | moderate | +1.06% (0.75) | +1.20% (0.70) | 1.01 (0.963) |
| stable_burn_shock | 17 | moderate | +5.11% (0.11) | +3.49% (0.48) | 1.11 (0.684) |
| etf_inflow | 26 | moderate | -0.07% (0.78) | +1.65% (0.67) | 0.91 (0.645) |
| etf_outflow | 9 | moderate | -0.24% (0.82) | +2.87% (0.64) | 1.04 (0.906) |
| funding_extreme_pos | 40 | poor | +1.71% (0.54) | +3.75% (0.21) | 1.04 (0.837) |
| funding_extreme_neg | 40 | poor | +2.93% (0.19) | +4.33% (0.13) | 1.17 (0.309) |
| gas_spike | 82 | moderate | +0.56% (0.97) | +0.69% (0.88) | 1.11 (0.432) |

## Full horizon detail

### blob_spike — n=41 (good: blob demand is L2-driven)

| Outcome | h | event mean | null mean | p |
|---|---:|---:|---:|---:|
| btc | 1d | -0.33% | +0.08% | 0.213 |
| btc | 3d | -0.32% | +0.23% | 0.554 |
| btc | 5d | -0.90% | +0.35% | 0.443 |
| btc | 10d | -0.74% | +0.76% | 0.556 |
| ew | 1d | -0.44% | +0.04% | 0.376 |
| ew | 3d | -0.53% | +0.27% | 0.598 |
| ew | 5d | -1.50% | +0.33% | 0.423 |
| ew | 10d | -1.75% | +0.59% | 0.588 |

Forward 5d vol ratio (event/uncond): **1.05** (p=0.753)

### stable_mint_shock — n=59 (moderate: issuance responds to demand)

| Outcome | h | event mean | null mean | p |
|---|---:|---:|---:|---:|
| btc | 1d | +0.02% | +0.10% | 0.783 |
| btc | 3d | +0.69% | +0.22% | 0.610 |
| btc | 5d | +1.06% | +0.56% | 0.747 |
| btc | 10d | +1.15% | +1.07% | 0.976 |
| ew | 1d | +0.24% | +0.07% | 0.705 |
| ew | 3d | +0.83% | +0.18% | 0.604 |
| ew | 5d | +1.20% | +0.47% | 0.701 |
| ew | 10d | +1.90% | +0.83% | 0.777 |

Forward 5d vol ratio (event/uncond): **1.01** (p=0.963)

### stable_burn_shock — n=17 (moderate: redemptions respond to stress)

| Outcome | h | event mean | null mean | p |
|---|---:|---:|---:|---:|
| btc | 1d | +1.95% ** | +0.14% | 0.013 |
| btc | 3d | +3.44% | +0.15% | 0.081 |
| btc | 5d | +5.11% | +0.53% | 0.108 |
| btc | 10d | +6.23% | +1.04% | 0.282 |
| ew | 1d | +1.87% ** | +0.05% | 0.046 |
| ew | 3d | +2.58% | +0.23% | 0.331 |
| ew | 5d | +3.49% | +0.77% | 0.477 |
| ew | 10d | +4.22% | +0.96% | 0.614 |

Forward 5d vol ratio (event/uncond): **1.11** (p=0.684)

### etf_inflow — n=26 (moderate: TradFi flow shocks (2024+, short))

| Outcome | h | event mean | null mean | p |
|---|---:|---:|---:|---:|
| btc | 1d | +0.16% | +0.09% | 0.849 |
| btc | 3d | -0.31% | +0.23% | 0.633 |
| btc | 5d | -0.07% | +0.50% | 0.779 |
| btc | 10d | -1.39% | +0.93% | 0.466 |
| ew | 1d | +0.81% | +0.10% | 0.310 |
| ew | 3d | +0.69% | +0.32% | 0.836 |
| ew | 5d | +1.65% | +0.40% | 0.674 |
| ew | 10d | -0.38% | +1.11% | 0.782 |

Forward 5d vol ratio (event/uncond): **0.91** (p=0.645)

### etf_outflow — n=9 (moderate: TradFi flow shocks (2024+, short))

| Outcome | h | event mean | null mean | p |
|---|---:|---:|---:|---:|
| btc | 1d | -0.79% | +0.08% | 0.187 |
| btc | 3d | -0.39% | +0.24% | 0.725 |
| btc | 5d | -0.24% | +0.49% | 0.820 |
| btc | 10d | -1.83% | +0.90% | 0.609 |
| ew | 1d | -1.16% | +0.01% | 0.324 |
| ew | 3d | +0.44% | +0.25% | 0.966 |
| ew | 5d | +2.87% | +0.41% | 0.643 |
| ew | 10d | -0.38% | +1.38% | 0.828 |

Forward 5d vol ratio (event/uncond): **1.04** (p=0.906)

### funding_extreme_pos — n=40 (poor: funding reacts to price/positioning)

| Outcome | h | event mean | null mean | p |
|---|---:|---:|---:|---:|
| btc | 1d | +0.84% ** | +0.15% | 0.047 |
| btc | 3d | +1.50% | +0.48% | 0.301 |
| btc | 5d | +1.71% | +0.72% | 0.537 |
| btc | 10d | +3.29% | +1.52% | 0.547 |
| ew | 1d | +0.97% | +0.18% | 0.119 |
| ew | 3d | +2.35% | +0.51% | 0.221 |
| ew | 5d | +3.75% | +0.78% | 0.212 |
| ew | 10d | +6.35% | +1.61% | 0.277 |

Forward 5d vol ratio (event/uncond): **1.04** (p=0.837)

### funding_extreme_neg — n=40 (poor: funding reacts to price/positioning)

| Outcome | h | event mean | null mean | p |
|---|---:|---:|---:|---:|
| btc | 1d | +0.61% | +0.15% | 0.211 |
| btc | 3d | +2.22% | +0.48% | 0.100 |
| btc | 5d | +2.93% | +0.72% | 0.191 |
| btc | 10d | +4.24% | +1.52% | 0.344 |
| ew | 1d | +1.02% | +0.18% | 0.095 |
| ew | 3d | +3.53% ** | +0.51% | 0.045 |
| ew | 5d | +4.33% | +0.78% | 0.134 |
| ew | 10d | +5.85% | +1.61% | 0.328 |

Forward 5d vol ratio (event/uncond): **1.17** (p=0.309)

### gas_spike — n=82 (moderate: usage bursts)

| Outcome | h | event mean | null mean | p |
|---|---:|---:|---:|---:|
| btc | 1d | +0.04% | +0.10% | 0.857 |
| btc | 3d | +0.58% | +0.29% | 0.740 |
| btc | 5d | +0.56% | +0.52% | 0.971 |
| btc | 10d | -0.19% | +0.81% | 0.662 |
| ew | 1d | +0.13% | +0.08% | 0.896 |
| ew | 3d | +0.70% | +0.27% | 0.682 |
| ew | 5d | +0.69% | +0.44% | 0.882 |
| ew | 10d | +0.35% | +0.71% | 0.920 |

Forward 5d vol ratio (event/uncond): **1.11** (p=0.432)

## Verdict

- Return effects with p<0.05: 4 of 64 (outcome,horizon) cells across 8 events — at α=0.05, ~3 expected by chance. Treat isolated hits accordingly.
- No event moves forward volatility beyond the block-bootstrap null either.