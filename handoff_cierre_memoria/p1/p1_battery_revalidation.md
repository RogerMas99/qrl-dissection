> SIMULATOR RESULTS (agent types sv / su2) - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL (docs/SIMULATED-RESULTS-POLICY.md; the validation-battery figures are the only approved use, recorded 2026-11-01)

| kind | env | simulator arm | real arm | n sim / real | IQM final sim / real | P(sim>real) final / greedy / auc | perm. p final | d | informative |
|---|---|---|---|---|---|---|---|---|---|
| sim-vs-real | CartPole | `sv_cartpole_L5` | `hybrid_fig4` | 10 / 10 | 35.989 / 35.801 | 0.53 / 0.62 / 0.50 | 0.85 | +0.07 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L5` | `hybrid_DR5` | 10 / 10 | 35.989 / 37.685 | 0.50 / 0.44 / 0.47 | 1.00 | -0.05 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR4` | `hybrid_OR4` | 10 / 10 | 99.010 / 94.660 | 0.53 / 0.53 / 0.54 | 0.85 | +0.13 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR8` | `hybrid_OR8` | 10 / 10 | 115.901 / 109.849 | 0.58 / 0.51 / 0.62 | 0.57 | +0.09 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR16` | `hybrid_OR16` | 10 / 10 | 119.379 / 104.209 | 0.66 / 0.53 / 0.55 | 0.26 | +0.49 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR32` | `hybrid_OR32` | 10 / 10 | 169.686 / 137.599 | 0.65 / 0.67 / 0.55 | 0.28 | +0.51 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L1` | `hybrid_DR1` | 10 / 10 | 10.319 / 10.137 | 0.60 / 0.72 / 0.67 | 0.46 |  | no |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR4` | `hybrid_DR1_OR4` | 10 / 10 | 10.644 / 10.647 | 0.52 / 0.49 / 0.45 | 0.91 |  | no |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR8` | `hybrid_DR1_OR8` | 10 / 10 | 11.509 / 11.184 | 0.63 / 0.42 / 0.51 | 0.35 |  | no |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR16` | `hybrid_DR1_OR16` | 10 / 10 | 11.025 / 11.589 | 0.18 / 0.52 / 0.45 | 0.01 |  | no |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR32` | `hybrid_DR1_OR32` | 10 / 10 | 10.615 / 10.830 | 0.46 / 0.57 / 0.45 | 0.79 |  | no |
| sim-vs-real | CartPole | `sv_cartpole_L2` | `hybrid_DR2` | 10 / 10 | 31.531 / 30.151 | 0.53 / 0.36 / 0.46 | 0.84 | +0.04 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR4` | `hybrid_DR2_OR4` | 10 / 10 | 46.511 / 44.614 | 0.50 / 0.47 / 0.51 | 1.00 | +0.03 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR8` | `hybrid_DR2_OR8` | 10 / 10 | 57.090 / 54.551 | 0.54 / 0.45 / 0.50 | 0.78 | +0.04 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR16` | `hybrid_DR2_OR16` | 10 / 10 | 73.532 / 67.884 | 0.58 / 0.58 / 0.53 | 0.58 | +0.38 | yes |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR32` | `hybrid_DR2_OR32` | 10 / 10 | 74.990 / 77.099 | 0.48 / 0.42 / 0.53 | 0.91 | -0.03 | yes |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L1` | `paper_skolik_4q_L1` | 10 / 10 | 9.624 / 9.634 | 0.44 / 0.51 / 0.36 | 0.66 |  | no |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L2` | `paper_skolik_4q_L2` | 10 / 10 | 9.616 / 9.621 | 0.43 / 0.50 / 0.43 | 0.63 |  | no |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L5` | `paper_skolik_4q_L5` | 10 / 10 | 9.624 / 9.605 | 0.63 / 0.57 / 0.48 | 0.35 |  | no |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L1` | `paper_skolik_8q_L1` | 10 / 10 | 11.205 / 10.915 | 0.53 / 0.64 / 0.56 | 0.85 |  | no |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L2` | `paper_skolik_8q_L2` | 10 / 10 | 13.392 / 12.944 | 0.56 / 0.64 / 0.58 | 0.68 | +0.14 | yes |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L5` | `paper_skolik_8q_L5` | 10 / 10 | 18.443 / 19.753 | 0.46 / 0.49 / 0.46 | 0.80 | -0.36 | yes |
| sim-vs-real | CartPole | `su2_cartpole_L1` | `hybrid_DR1_noent` | 10 / 10 | 10.183 / 9.950 | 0.67 / 0.57 / 0.54 | 0.22 |  | no |
| sim-vs-real | CartPole | `su2_cartpole_L2` | `hybrid_DR2_noent` | 10 / 10 | 36.112 / 34.547 | 0.54 / 0.72 / 0.56 | 0.78 | +0.22 | yes |
| sim-vs-real | CartPole | `su2_cartpole_L5 [init2pi]` | `hybrid_DR5_noent` | 10 / 10 | 43.673 / 41.789 | 0.66 / 0.59 / 0.52 | 0.26 | +0.10 | yes |
| sim-vs-real | CartPole | `su2_cartpole_L5 [init2pi]` | `hybrid_fig4_noent` | 10 / 10 | 43.673 / 43.503 | 0.54 / 0.62 / 0.58 | 0.80 | +0.02 | yes |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR4` | `hybrid_or_r4_noent` | 10 / 10 | 64.355 / 60.434 | 0.52 / 0.65 / 0.59 | 0.92 | +0.10 | yes |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR8` | `hybrid_or_r8_noent` | 10 / 10 | 73.461 / 69.134 | 0.57 / 0.51 / 0.50 | 0.62 | +0.04 | yes |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR16` | `hybrid_or_r16_noent` | 10 / 10 | 86.047 / 89.465 | 0.45 / 0.47 / 0.64 | 0.75 | -0.11 | yes |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR32` | `hybrid_or_r32_noent` | 10 / 10 | 75.852 / 77.880 | 0.46 / 0.53 / 0.58 | 0.80 | +0.13 | yes |
| sim-vs-real | CartPole | `su2_paper_skolik_4q_L2_noent` | `paper_skolik_4q_L2_noent` | 10 / 10 | 9.738 / 9.710 | 0.63 / 0.51 / 0.48 | 0.36 |  | no |
| sim-vs-real | CartPole | `su2_paper_skolik_4q_L5_noent` | `paper_skolik_4q_L5_noent` | 10 / 10 | 9.679 / 9.693 | 0.47 / 0.52 / 0.57 | 0.85 |  | no |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L1 (seeds<=10)` | `frozen_binary_4q_L1` | 10 / 10 | 0.000 / 0.000 | 0.55 / 0.50 / 0.56 | 1.00 |  | no |
| sim-vs-real | FrozenLake | `su2_frozen_binary_4q_L1` | `frozen_binary_4q_noent_L1` | 10 / 10 | 0.000 / 0.000 | 0.49 / 0.50 / 0.50 | 1.00 |  | no |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L5 (seeds<=10)` | `frozen_binary_4q_L5` | 10 / 10 | 0.942 / 0.952 | 0.21 / 0.43 / 0.48 | 0.03 | -0.10 | yes |
| sim-vs-real | FrozenLake | `su2_frozen_binary_4q_L5` | `frozen_binary_4q_noent_L5` | 10 / 10 | 0.000 / 0.000 | 0.55 / 0.50 / 0.58 | 1.00 |  | no |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L5` | `frozen_scalar_1q_L5` | 10 / 10 | 0.102 / 0.125 | 0.49 / 0.50 / 0.53 | 0.99 | +0.10 | yes |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L10` | `frozen_scalar_1q_L10` | 10 / 10 | 0.046 / 0.081 | 0.56 / 0.45 / 0.45 | 0.69 | -0.12 | yes |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L15` | `frozen_scalar_1q_L15` | 10 / 10 | 0.010 / 0.011 | 0.46 / 0.50 / 0.47 | 0.79 |  | no |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L5` | `frozen_binary_4q_L5` | 20 / 10 | 0.928 / 0.952 | 0.27 / 0.40 / 0.41 | 0.04 | -0.45 | yes |
| twin | CartPole | `sv_cartpole_L5_noent` | `su2_cartpole_L5 [init2pi]` | 10 / 10 | 42.942 / 43.673 | 0.41 / 0.38 / 0.54 | 0.55 | -0.14 | yes |
| twin | FrozenLake | `sv_frozen_binary_4q_noent_L1` | `su2_frozen_binary_4q_L1` | 10 / 10 | 0.000 / 0.000 | 0.56 / 0.50 / 0.33 | 0.48 |  | no |
| twin | FrozenLake | `sv_frozen_binary_4q_noent_L5` | `su2_frozen_binary_4q_L5` | 10 / 10 | 0.000 / 0.000 | 0.45 / 0.50 / 0.54 | 1.00 |  | no |
| real-vs-real | CartPole | `hybrid_fig4` | `hybrid_DR5` | 10 / 10 | 35.801 / 37.685 | 0.49 / 0.34 / 0.48 | 0.97 | -0.11 | yes |

**Criterion 1 (no global bias):** mean d = +0.090, 95% CI (+0.008, +0.175), 21 informative pairs -> FAIL
**Criterion 2 (no excess of disagreements):** 1 of 63 cells outside [0.25, 0.75] = 2% (limit 25%) -> PASS
**Criterion 3 (the two simulators agree):** 0 of 9 twin cells outside [0.25, 0.75] -> PASS
**Overall: NOT ACCEPTED** (still under docs/SIMULATED-RESULTS-POLICY.md in either case).

Cells outside [0.25, 0.75] (pairs inside the global rule):

- `sv_frozen_binary_4q_L5 (seeds<=10)` vs `frozen_binary_4q_L5`, final: P(sim>real) = 0.21