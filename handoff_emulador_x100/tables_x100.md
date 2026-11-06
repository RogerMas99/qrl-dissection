> SIMULATOR RESULTS - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL (docs/SIMULATED-RESULTS-POLICY.md). Nothing in this file is an approved use.

## Arms and seeds available

| arm | n | new (exp10) | reused (exp09 / exp01) |
|---|---|---|---|
| `su2_cartpole_L5` | 100 | 90 | 10 |
| `su2_cartpole_L5_OR16` | 100 | 90 | 10 |
| `su2_cartpole_L5_OR32` | 100 | 90 | 10 |
| `su2_cartpole_L5_OR4` | 100 | 90 | 10 |
| `su2_cartpole_L5_OR8` | 100 | 90 | 10 |
| `sv_cartpole_L1` | 100 | 90 | 10 |
| `sv_cartpole_L1_noent` | 100 | 100 | 0 |
| `sv_cartpole_L2` | 100 | 90 | 10 |
| `sv_cartpole_L2_OR16` | 100 | 90 | 10 |
| `sv_cartpole_L2_OR32` | 100 | 90 | 10 |
| `sv_cartpole_L2_OR4` | 100 | 90 | 10 |
| `sv_cartpole_L2_OR8` | 100 | 90 | 10 |
| `sv_cartpole_L2_noent` | 100 | 100 | 0 |
| `sv_cartpole_L5` | 100 | 90 | 10 |
| `sv_cartpole_L5_OR16` | 100 | 90 | 10 |
| `sv_cartpole_L5_OR16_noent` | 100 | 100 | 0 |
| `sv_cartpole_L5_OR32` | 100 | 90 | 10 |
| `sv_cartpole_L5_OR32_noent` | 100 | 100 | 0 |
| `sv_cartpole_L5_OR4` | 100 | 90 | 10 |
| `sv_cartpole_L5_OR4_noent` | 100 | 100 | 0 |
| `sv_cartpole_L5_OR8` | 100 | 90 | 10 |
| `sv_cartpole_L5_OR8_noent` | 100 | 100 | 0 |
| `sv_cartpole_L5_noent` | 100 | 90 | 10 |
| `sv_frozen_binary_4q_L5` | 100 | 80 | 20 |
| `sv_frozen_binary_4q_noent_L5` | 100 | 90 | 10 |

## C1 - entanglement x Output Reuse (CartPole, L=5)

Primary (same tool on both sides):

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | strong? |
|---|---|---|---|---|---|---|
| R=1: sv entangled vs sv unentangled | 100 / 100 | 42.3 / 45.6 | 0.43 | 54.0 / 55.5 | 0.48 | no |
| R=4: sv entangled vs sv unentangled | 100 / 100 | 81.0 / 60.7 | 0.70 | 95.2 / 70.5 | 0.69 | no |
| R=8: sv entangled vs sv unentangled | 100 / 100 | 106.1 / 64.8 | 0.85 | 119.4 / 71.7 | 0.82 | no |
| R=16: sv entangled vs sv unentangled | 100 / 100 | 115.4 / 72.5 | 0.83 | 134.5 / 82.3 | 0.82 | no |
| R=32: sv entangled vs sv unentangled | 100 / 100 | 133.8 / 75.6 | 0.84 | 154.9 / 85.2 | 0.85 | no |

Secondary (against the SU(2) emulator):

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | strong? |
|---|---|---|---|---|---|---|
| R=1: sv entangled vs su2 unentangled | 100 / 100 | 42.3 / 45.4 | 0.43 | 54.0 / 52.4 | 0.52 | no |
| R=4: sv entangled vs su2 unentangled | 100 / 100 | 81.0 / 64.5 | 0.67 | 95.2 / 78.1 | 0.63 | no |
| R=8: sv entangled vs su2 unentangled | 100 / 100 | 106.1 / 63.8 | 0.84 | 119.4 / 71.8 | 0.81 | no |
| R=16: sv entangled vs su2 unentangled | 100 / 100 | 115.4 / 71.7 | 0.83 | 134.5 / 79.8 | 0.81 | no |
| R=32: sv entangled vs su2 unentangled | 100 / 100 | 133.8 / 75.4 | 0.81 | 154.9 / 84.9 | 0.83 | no |

Tool agreement (sv ent=False against su2, same circuit):

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | strong? |
|---|---|---|---|---|---|---|
| R=1: sv unentangled vs su2 (same circuit, two tools) | 100 / 100 | 45.6 / 45.4 | 0.50 | 55.5 / 52.4 | 0.53 | no |
| R=4: sv unentangled vs su2 (same circuit, two tools) | 100 / 100 | 60.7 / 64.5 | 0.46 | 70.5 / 78.1 | 0.43 | no |
| R=8: sv unentangled vs su2 (same circuit, two tools) | 100 / 100 | 64.8 / 63.8 | 0.51 | 71.7 / 71.8 | 0.51 | no |
| R=16: sv unentangled vs su2 (same circuit, two tools) | 100 / 100 | 72.5 / 71.7 | 0.52 | 82.3 / 79.8 | 0.51 | no |
| R=32: sv unentangled vs su2 (same circuit, two tools) | 100 / 100 | 75.6 / 75.4 | 0.50 | 85.2 / 84.9 | 0.49 | no |

## C2 - depth (CartPole, R=1)

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | strong? |
|---|---|---|---|---|---|---|
| entangled: L2 > L1 | 100 / 100 | 26.2 / 10.3 | 0.97 | 28.6 / 9.9 | 0.92 | no |
| entangled: L5 > L2 | 100 / 100 | 42.3 / 26.2 | 0.76 | 54.0 / 28.6 | 0.79 | no |
| unentangled: L2 > L1 | 100 / 100 | 35.0 / 10.2 | 1.00 | 37.2 / 9.6 | 0.99 | yes |
| unentangled: L5 > L2 | 100 / 100 | 45.6 / 35.0 | 0.74 | 55.5 / 37.2 | 0.74 | no |

## C3 - depth x Output Reuse (CartPole)

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | strong? |
|---|---|---|---|---|---|---|
| L2 with OR4 > L2 | 100 / 100 | 52.1 / 26.2 | 0.91 | 55.5 / 28.6 | 0.85 | no |
| L2 with OR8 > L2 | 100 / 100 | 63.4 / 26.2 | 0.96 | 66.3 / 28.6 | 0.92 | no |
| L2 with OR16 > L2 | 100 / 100 | 71.2 / 26.2 | 0.98 | 75.0 / 28.6 | 0.96 | yes |
| L2 with OR32 > L2 | 100 / 100 | 71.6 / 26.2 | 0.97 | 76.8 / 28.6 | 0.95 | yes |
| L5_OR32 > L2_OR32 | 100 / 100 | 133.8 / 71.6 | 0.88 | 154.9 / 76.8 | 0.88 | no |

## C4 - FrozenLake Config B, L=5

| arm | n | alive seeds (greedy_final > 0) | final_perf IQM (CI) | greedy_final IQM (CI) |
|---|---|---|---|---|
| `sv_frozen_binary_4q_L5` | 100 | 82 of 100 | 0.946 (0.917-0.948) | 0.933 (0.813-0.993) |
| `sv_frozen_binary_4q_noent_L5` | 100 | 0 of 100 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) |

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | strong? |
|---|---|---|---|---|---|---|
| entangled > unentangled | 100 / 100 | 0.946 / 0.000 | 1.00 | 0.933 / 0.000 | 0.91 | no |

## Coherence check 1 - seeds 1-10 against seeds 11-100 (same arm; should be indistinguishable)

| arm | n 1-10 / 11-100 | IQM final 1-10 / 11-100 | P(1-10 > 11-100) final / greedy | outside [0.25, 0.75]? |
|---|---|---|---|---|
| `su2_cartpole_L5` | 10 / 90 | 43.673 / 45.854 | 0.49 / 0.56 | no |
| `su2_cartpole_L5_OR16` | 10 / 90 | 86.047 / 69.983 | 0.72 / 0.76 | yes |
| `su2_cartpole_L5_OR32` | 10 / 90 | 75.852 / 75.440 | 0.48 / 0.59 | no |
| `su2_cartpole_L5_OR4` | 10 / 90 | 64.355 / 64.602 | 0.48 / 0.50 | no |
| `su2_cartpole_L5_OR8` | 10 / 90 | 73.461 / 62.731 | 0.61 / 0.62 | no |
| `sv_cartpole_L1` | 10 / 90 | 10.319 / 10.265 | 0.49 / 0.63 | no |
| `sv_cartpole_L1_noent` | 10 / 90 | 10.074 / 10.198 | 0.43 / 0.65 | no |
| `sv_cartpole_L2` | 10 / 90 | 31.531 / 25.340 | 0.64 / 0.57 | no |
| `sv_cartpole_L2_OR16` | 10 / 90 | 73.532 / 70.829 | 0.56 / 0.59 | no |
| `sv_cartpole_L2_OR32` | 10 / 90 | 74.990 / 71.033 | 0.58 / 0.72 | no |
| `sv_cartpole_L2_OR4` | 10 / 90 | 46.511 / 52.710 | 0.38 / 0.39 | no |
| `sv_cartpole_L2_OR8` | 10 / 90 | 57.090 / 64.034 | 0.37 / 0.40 | no |
| `sv_cartpole_L2_noent` | 10 / 90 | 35.372 / 34.838 | 0.54 / 0.57 | no |
| `sv_cartpole_L5` | 10 / 90 | 35.989 / 43.170 | 0.38 / 0.55 | no |
| `sv_cartpole_L5_OR16` | 10 / 90 | 119.379 / 115.036 | 0.51 / 0.58 | no |
| `sv_cartpole_L5_OR16_noent` | 10 / 90 | 83.072 / 71.333 | 0.71 / 0.78 | yes |
| `sv_cartpole_L5_OR32` | 10 / 90 | 169.686 / 129.655 | 0.71 / 0.70 | no |
| `sv_cartpole_L5_OR32_noent` | 10 / 90 | 92.330 / 73.904 | 0.71 / 0.75 | yes |
| `sv_cartpole_L5_OR4` | 10 / 90 | 99.010 / 79.014 | 0.68 / 0.66 | no |
| `sv_cartpole_L5_OR4_noent` | 10 / 90 | 61.574 / 60.560 | 0.50 / 0.52 | no |
| `sv_cartpole_L5_OR8` | 10 / 90 | 115.901 / 104.826 | 0.57 / 0.57 | no |
| `sv_cartpole_L5_OR8_noent` | 10 / 90 | 73.743 / 64.058 | 0.62 / 0.63 | no |
| `sv_cartpole_L5_noent` | 10 / 90 | 42.942 / 46.103 | 0.46 / 0.44 | no |
| `sv_frozen_binary_4q_L5` | 10 / 90 | 0.942 / 0.946 | 0.39 / 0.49 | no |
| `sv_frozen_binary_4q_noent_L5` | 10 / 90 | 0.000 / 0.000 | 0.48 / 0.50 | no |

## Coherence check 2 - simulator arm (all its seeds) against its real arm (n = 10)

| simulator arm | real arm | n sim / real | IQM final sim / real | P(sim>real) final / greedy |
|---|---|---|---|---|
| `sv_cartpole_L5` | `hybrid_fig4` | 100 / 10 | 42.338 / 35.801 | 0.59 / 0.58 |
| `sv_cartpole_L5_noent` | `hybrid_fig4_noent` | 100 / 10 | 45.562 / 43.503 | 0.51 / 0.57 |
| `su2_cartpole_L5` | `hybrid_fig4_noent` | 100 / 10 | 45.421 / 43.503 | 0.53 / 0.55 |
| `sv_cartpole_L2` | `hybrid_DR2` | 100 / 10 | 26.163 / 30.151 | 0.39 / 0.32 |
| `sv_cartpole_L5_OR4` | `hybrid_OR4` | 100 / 10 | 80.953 / 94.660 | 0.36 / 0.36 |
| `sv_cartpole_L5_OR4_noent` | `hybrid_or_r4_noent` | 100 / 10 | 60.680 / 60.434 | 0.52 / 0.58 |
| `su2_cartpole_L5_OR4` | `hybrid_or_r4_noent` | 100 / 10 | 64.519 / 60.434 | 0.56 / 0.63 |
| `sv_cartpole_L2_OR4` | `hybrid_DR2_OR4` | 100 / 10 | 52.138 / 44.614 | 0.63 / 0.56 |
| `sv_cartpole_L5_OR8` | `hybrid_OR8` | 100 / 10 | 106.079 / 109.849 | 0.47 / 0.44 |
| `sv_cartpole_L5_OR8_noent` | `hybrid_or_r8_noent` | 100 / 10 | 64.751 / 69.134 | 0.44 / 0.36 |
| `su2_cartpole_L5_OR8` | `hybrid_or_r8_noent` | 100 / 10 | 63.790 / 69.134 | 0.44 / 0.39 |
| `sv_cartpole_L2_OR8` | `hybrid_DR2_OR8` | 100 / 10 | 63.431 / 54.551 | 0.65 / 0.47 |
| `sv_cartpole_L5_OR16` | `hybrid_OR16` | 100 / 10 | 115.431 / 104.209 | 0.61 / 0.44 |
| `sv_cartpole_L5_OR16_noent` | `hybrid_or_r16_noent` | 100 / 10 | 72.454 / 89.465 | 0.29 / 0.28 |
| `su2_cartpole_L5_OR16` | `hybrid_or_r16_noent` | 100 / 10 | 71.654 / 89.465 | 0.29 / 0.30 |
| `sv_cartpole_L2_OR16` | `hybrid_DR2_OR16` | 100 / 10 | 71.153 / 67.884 | 0.54 / 0.50 |
| `sv_cartpole_L5_OR32` | `hybrid_OR32` | 100 / 10 | 133.782 / 137.599 | 0.47 / 0.47 |
| `sv_cartpole_L5_OR32_noent` | `hybrid_or_r32_noent` | 100 / 10 | 75.599 / 77.880 | 0.50 / 0.41 |
| `su2_cartpole_L5_OR32` | `hybrid_or_r32_noent` | 100 / 10 | 75.425 / 77.880 | 0.49 / 0.42 |
| `sv_cartpole_L2_OR32` | `hybrid_DR2_OR32` | 100 / 10 | 71.641 / 77.099 | 0.39 / 0.26 |
| `sv_cartpole_L1` | `hybrid_DR1` | 100 / 10 | 10.270 / 10.137 | 0.58 / 0.58 |
| `sv_cartpole_L1_noent` | `hybrid_DR1_noent` | 100 / 10 | 10.183 / 9.950 | 0.62 / 0.35 |
| `sv_cartpole_L2_noent` | `hybrid_DR2_noent` | 100 / 10 | 35.017 / 34.547 | 0.47 / 0.50 |
| `sv_frozen_binary_4q_L5` | `frozen_binary_4q_L5` | 100 / 10 | 0.946 / 0.952 | 0.31 / 0.46 |
| `sv_frozen_binary_4q_noent_L5` | `frozen_binary_4q_noent_L5` | 100 / 10 | 0.000 / 0.000 | 0.52 / 0.50 |

No multiple-comparison correction is applied to any of the contrasts above.