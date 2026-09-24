# PokeRL — Results

Generated 2026-08-06 06:54. All figures regenerate from `results/*.csv` via `scripts/analyse.py`.

Format: **win rate % ± std across seeds**, 500 battles per opponent per seed, deterministic action selection. Battle format `gen9randombattle`.

> **Scope.** This covers the 300k-step study matrix only. The three 5M-step coach training
> arms (`ppo_v1_long`, `ppo_v1_entropy`, `ppo_v1_selfplay`) were run later, deliberately
> outside the matrix. They are reported in the [readme](../readme.md#coach-training-arms) and
> in [COACH_FINDINGS.md](COACH_FINDINGS.md).

### Axis A — Algorithm (encoder `v0`, shaped reward, vs heuristic)

| Configuration | Random | MaxBasePower | SimpleHeuristics | Seeds |
|---|---|---|---|---|
| `ppo_masked_v0` | 96.7 ± 0.6 | 72.5 ± 1.7 | 31.1 ± 2.9 | 3 |
| `ppo_unmasked_v0` | 96.1 ± 0.3 | 70.5 ± 3.9 | 24.7 ± 1.5 | 3 |
| `dqn_masked_v0` | 89.0 ± 2.8 | 45.4 ± 1.6 | 11.9 ± 0.5 | 3 |
| `a2c_masked_v0` | 79.1 ± 5.9 | 34.9 ± 7.0 | 5.7 ± 2.5 | 3 |
| `tabular_q_v0` | 54.9 ± 2.0 | 9.1 ± 0.8 | 1.8 ± 0.5 | 3 |

### Axis B — State encoding

| Configuration | Random | MaxBasePower | SimpleHeuristics | Seeds |
|---|---|---|---|---|
| `ppo_masked_v0` | 96.7 ± 0.6 | 72.5 ± 1.7 | 31.1 ± 2.9 | 3 |
| `ppo_masked_v1` | 95.9 ± 0.4 | 66.1 ± 1.8 | 26.5 ± 1.6 | 3 |

### Axis C — Reward shaping

| Configuration | Random | MaxBasePower | SimpleHeuristics | Seeds |
|---|---|---|---|---|
| `ppo_masked_v0_sparse` | 77.5 ± 34.2 | 52.1 ± 40.1 | 22.3 ± 18.6 | 3 |
| `ppo_masked_v0` | 96.7 ± 0.6 | 72.5 ± 1.7 | 31.1 ± 2.9 | 3 |
| `ppo_masked_v0_aggressive` | 97.5 ± 0.6 | 70.3 ± 1.5 | 31.1 ± 1.8 | 3 |

### Axis D — Training opponent

| Configuration | Random | MaxBasePower | SimpleHeuristics | Seeds |
|---|---|---|---|---|
| `ppo_masked_v0_vs_random` | 97.1 ± 0.5 | 71.6 ± 2.2 | 28.9 ± 4.7 | 3 |
| `ppo_masked_v0_vs_maxpower` | 97.1 ± 0.8 | 72.5 ± 1.4 | 32.1 ± 1.9 | 3 |
| `ppo_masked_v0` | 96.7 ± 0.6 | 72.5 ± 1.7 | 31.1 ± 2.9 | 3 |
| `ppo_masked_v0_vs_mixed` | 97.4 ± 0.6 | 73.1 ± 3.0 | 30.1 ± 2.7 | 3 |

### Overall ranking (vs SimpleHeuristics, the discriminating opponent)

| Rank | Configuration | Win rate | Seeds |
|---|---|---|---|
| 1 | `ppo_masked_v0_vs_maxpower` | 32.1 ± 1.9 | 3 |
| 2 | `ppo_masked_v0` | 31.1 ± 2.9 | 3 |
| 3 | `ppo_masked_v0_aggressive` | 31.1 ± 1.8 | 3 |
| 4 | `ppo_masked_v0_vs_mixed` | 30.1 ± 2.7 | 3 |
| 5 | `ppo_masked_v0_vs_random` | 28.9 ± 4.7 | 3 |
| 6 | `ppo_masked_v1` | 26.5 ± 1.6 | 3 |
| 7 | `ppo_unmasked_v0` | 24.7 ± 1.5 | 3 |
| 8 | `ppo_masked_v0_sparse` | 22.3 ± 18.6 | 3 |
| 9 | `sanity_ppo_masked` | 20.0 | 1 |
| 10 | `dqn_masked_v0` | 11.9 ± 0.5 | 3 |
| 11 | `a2c_masked_v0` | 5.7 ± 2.5 | 3 |
| 12 | `tabular_q_v0` | 1.8 ± 0.5 | 3 |

### Cross-evaluation

Row player's win rate (%) against the column player.

| | a2c_masked_v0_seed0 | dqn_masked_v0_seed0 | ppo_masked_v0_aggressive_seed0 | ppo_masked_v0_seed0 | ppo_masked_v0_sparse_seed0 | ppo_masked_v0_vs_maxpower_seed0 | ppo_masked_v0_vs_mixed_seed0 | ppo_masked_v0_vs_random_seed0 | ppo_masked_v1_seed0 | ppo_unmasked_v0_seed0 | tabular_q_v0_seed0 | random | max_power | heuristic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **a2c_masked_v0_seed0** | — | 40 | 17 | 15 | 11 | 15 | 16 | 13 | 21 | 14 | 83 | 75 | 30 | 7 |
| **dqn_masked_v0_seed0** | 60 | — | 24 | 19 | 24 | 34 | 23 | 20 | 36 | 27 | 89 | 92 | 57 | 7 |
| **ppo_masked_v0_aggressive_seed0** | 83 | 76 | — | 51 | 45 | 51 | 49 | 54 | 48 | 52 | 95 | 100 | 79 | 28 |
| **ppo_masked_v0_seed0** | 85 | 81 | 49 | — | 45 | 54 | 47 | 49 | 63 | 64 | 97 | 97 | 71 | 35 |
| **ppo_masked_v0_sparse_seed0** | 89 | 76 | 55 | 55 | — | 48 | 53 | 49 | 52 | 54 | 96 | 97 | 75 | 44 |
| **ppo_masked_v0_vs_maxpower_seed0** | 85 | 66 | 49 | 46 | 52 | — | 51 | 47 | 60 | 55 | 96 | 95 | 77 | 37 |
| **ppo_masked_v0_vs_mixed_seed0** | 84 | 77 | 51 | 53 | 47 | 49 | — | 54 | 67 | 64 | 100 | 98 | 69 | 35 |
| **ppo_masked_v0_vs_random_seed0** | 87 | 80 | 46 | 51 | 51 | 53 | 46 | — | 62 | 52 | 97 | 95 | 69 | 33 |
| **ppo_masked_v1_seed0** | 79 | 64 | 52 | 37 | 48 | 40 | 33 | 38 | — | 47 | 93 | 97 | 71 | 28 |
| **ppo_unmasked_v0_seed0** | 86 | 73 | 48 | 36 | 46 | 45 | 36 | 48 | 53 | — | 99 | 98 | 72 | 21 |
| **tabular_q_v0_seed0** | 17 | 11 | 5 | 3 | 4 | 4 | 0 | 3 | 7 | 1 | — | 49 | 11 | 2 |
| **random** | 25 | 8 | 0 | 3 | 3 | 5 | 2 | 5 | 3 | 2 | 51 | — | 9 | 2 |
| **max_power** | 70 | 43 | 21 | 29 | 25 | 23 | 31 | 31 | 29 | 28 | 89 | 91 | — | 6 |
| **heuristic** | 93 | 93 | 72 | 65 | 56 | 63 | 65 | 67 | 72 | 79 | 98 | 98 | 94 | — |

### Figures

- `report/figures/win_rates.png` — win rate by configuration and opponent
- `report/figures/learning_curves.png` — episode return vs environment steps
- `report/figures/cross_evaluation.png` — pairwise win-rate matrix

### Run inventory

34 evaluated runs:

- `a2c_masked_v0_seed0`
- `a2c_masked_v0_seed1`
- `a2c_masked_v0_seed2`
- `dqn_masked_v0_seed0`
- `dqn_masked_v0_seed1`
- `dqn_masked_v0_seed2`
- `ppo_masked_v0_aggressive_seed0`
- `ppo_masked_v0_aggressive_seed1`
- `ppo_masked_v0_aggressive_seed2`
- `ppo_masked_v0_seed0`
- `ppo_masked_v0_seed1`
- `ppo_masked_v0_seed2`
- `ppo_masked_v0_sparse_seed0`
- `ppo_masked_v0_sparse_seed1`
- `ppo_masked_v0_sparse_seed2`
- `ppo_masked_v0_vs_maxpower_seed0`
- `ppo_masked_v0_vs_maxpower_seed1`
- `ppo_masked_v0_vs_maxpower_seed2`
- `ppo_masked_v0_vs_mixed_seed0`
- `ppo_masked_v0_vs_mixed_seed1`
- `ppo_masked_v0_vs_mixed_seed2`
- `ppo_masked_v0_vs_random_seed0`
- `ppo_masked_v0_vs_random_seed1`
- `ppo_masked_v0_vs_random_seed2`
- `ppo_masked_v1_seed0`
- `ppo_masked_v1_seed1`
- `ppo_masked_v1_seed2`
- `ppo_unmasked_v0_seed0`
- `ppo_unmasked_v0_seed1`
- `ppo_unmasked_v0_seed2`
- `sanity_ppo_masked_seed0`
- `tabular_q_v0_seed0`
- `tabular_q_v0_seed1`
- `tabular_q_v0_seed2`
