# Experiment 09 - validation battery of the simulators: results

> **SIMULATOR RESULTS - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL**
> ([SIMULATED-RESULTS-POLICY.md](SIMULATED-RESULTS-POLICY.md)). Design and acceptance rule:
> [EXPERIMENT-09.md](EXPERIMENT-09.md) section 7, fixed before any battery cell finished.
> Generated 2026-10-21 by `battery_report.py` from the manifests on disk.

**Verdict under the rule as written: NOT ACCEPTED, narrowly.** Criteria 2 and 3 pass; criterion 1
fails because the 95 % CI of the mean standardised difference is (+0.010, +0.250): it excludes 0
by 0.010, although the mean (+0.119) is well inside the 0.30 bound. The simulators run slightly
high on average against the real runs. Consequence stated in advance: nothing read from the
simulators is interpreted until the author decides.

Note on method: a first run of this report used all 20 seeds of `sv_frozen_binary_4q_L5` and
gave criterion 1 PASS (mean d = +0.102, CI (-0.020, +0.243)); the rule is written for 10 seeds
per simulator arm, so the rule is evaluated with seeds 1-10 and the 20-seed pair is shown as an extra row.

> SIMULATOR RESULTS (agent types sv / su2) - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL (docs/SIMULATED-RESULTS-POLICY.md)

| kind | env | simulator arm | real arm | n sim / real | IQM final_perf sim / real | P(sim>real) final / greedy / auc | perm. p final | d | informative | note |
|---|---|---|---|---|---|---|---|---|---|---|
| sim-vs-real | CartPole | `sv_cartpole_L5` | `hybrid_fig4` | 10 / 10 | 35.989 / 35.801 | 0.53 / 0.62 / 0.50 | 0.85 | +0.07 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L5` | `hybrid_DR5` | 10 / 10 | 35.989 / 37.685 | 0.50 / 0.44 / 0.47 | 1.00 | -0.05 | yes | second reference (not in rule) |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR4` | `hybrid_OR4` | 10 / 10 | 99.010 / 94.660 | 0.53 / 0.53 / 0.54 | 0.85 | +0.13 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR8` | `hybrid_OR8` | 10 / 10 | 115.901 / 109.849 | 0.58 / 0.51 / 0.62 | 0.57 | +0.09 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR16` | `hybrid_OR16` | 10 / 10 | 119.379 / 104.209 | 0.66 / 0.53 / 0.55 | 0.26 | +0.49 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR32` | `hybrid_OR32` | 10 / 10 | 169.686 / 137.599 | 0.65 / 0.67 / 0.55 | 0.28 | +0.51 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L1` | `hybrid_DR1` | 10 / 10 | 10.319 / 10.137 | 0.60 / 0.72 / 0.67 | 0.46 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR4` | `hybrid_DR1_OR4` | 10 / 9 | 10.644 / 10.669 | 0.53 / 0.46 / 0.43 | 0.84 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR8` | `hybrid_DR1_OR8` | 10 / 9 | 11.509 / 11.224 | 0.62 / 0.39 / 0.53 | 0.40 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR16` | `hybrid_DR1_OR16` | 10 / 9 | 11.025 / 11.662 | 0.18 / 0.55 / 0.42 | 0.02 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR32` | `hybrid_DR1_OR32` | 10 / 9 | 10.615 / 10.751 | 0.49 / 0.64 / 0.49 | 0.97 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L2` | `hybrid_DR2` | 10 / 10 | 31.531 / 30.151 | 0.53 / 0.36 / 0.46 | 0.86 | +0.04 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR4` | `hybrid_DR2_OR4` | 10 / 9 | 46.511 / 46.275 | 0.47 / 0.43 / 0.48 | 0.83 | -0.04 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR8` | `hybrid_DR2_OR8` | 10 / 9 | 57.090 / 56.794 | 0.51 / 0.39 / 0.46 | 0.97 | -0.09 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR16` | `hybrid_DR2_OR16` | 10 / 9 | 73.532 / 70.519 | 0.54 / 0.53 / 0.48 | 0.78 | +0.23 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR32` | `hybrid_DR2_OR32` | 10 / 9 | 74.990 / 77.760 | 0.43 / 0.42 / 0.54 | 0.67 | -0.12 | yes |  |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L1` | `paper_skolik_4q_L1` | 10 / 7 | 9.624 / 9.634 | 0.46 / 0.55 / 0.37 | 0.80 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L2` | `paper_skolik_4q_L2` | 10 / 7 | 9.616 / 9.642 | 0.41 / 0.41 / 0.50 | 0.60 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L5` | `paper_skolik_4q_L5` | 10 / 5 | 9.624 / 9.627 | 0.54 / 0.53 / 0.44 | 0.87 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L1` | `paper_skolik_8q_L1` | 10 / 5 | 11.205 / 10.547 | 0.60 / 0.55 / 0.64 | 0.58 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L2` | `paper_skolik_8q_L2` | 10 / 4 | 13.392 / 10.194 | 0.82 / 0.62 / 0.80 | 0.07 | +1.05 | yes |  |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L5` | `paper_skolik_8q_L5` | 10 / 3 | 18.443 / 19.604 | 0.43 / 0.40 / 0.40 | 0.80 | -0.12 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L1` | `hybrid_DR1_noent` | 10 / 10 | 10.183 / 9.950 | 0.67 / 0.57 / 0.54 | 0.24 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `su2_cartpole_L2` | `hybrid_DR2_noent` | 10 / 10 | 36.112 / 34.547 | 0.54 / 0.72 / 0.56 | 0.79 | +0.22 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5 [init2pi]` | `hybrid_DR5_noent` | 10 / 10 | 43.673 / 41.789 | 0.66 / 0.59 / 0.52 | 0.23 | +0.10 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5 [init2pi]` | `hybrid_fig4_noent` | 10 / 9 | 43.673 / 42.528 | 0.59 / 0.58 / 0.63 | 0.54 | +0.05 | yes | second reference (not in rule) |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR4` | `hybrid_or_r4_noent` | 10 / 8 | 64.355 / 60.337 | 0.51 / 0.66 / 0.57 | 0.96 | +0.03 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR8` | `hybrid_or_r8_noent` | 10 / 9 | 73.461 / 68.745 | 0.56 / 0.50 / 0.52 | 0.74 | +0.04 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR16` | `hybrid_or_r16_noent` | 10 / 7 | 86.047 / 81.099 | 0.56 / 0.64 / 0.63 | 0.74 | +0.24 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR32` | `hybrid_or_r32_noent` | 10 / 5 | 75.852 / 86.655 | 0.36 / 0.42 / 0.48 | 0.44 | -0.26 | yes |  |
| sim-vs-real | CartPole | `su2_paper_skolik_4q_L2_noent` | `paper_skolik_4q_L2_noent` | 10 / 5 | 9.738 / 9.720 | 0.60 / 0.50 / 0.40 | 0.60 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `su2_paper_skolik_4q_L5_noent` | `paper_skolik_4q_L5_noent` | 10 / 4 | 9.679 / 9.662 | 0.45 / 0.46 / 0.60 | 0.83 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L1 (seeds<=10)` | `frozen_binary_4q_L1` | 10 / 10 | 0.000 / 0.000 | 0.55 / 0.50 / 0.56 | 1.00 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `su2_frozen_binary_4q_L1` | `frozen_binary_4q_noent_L1` | 10 / 10 | 0.000 / 0.000 | 0.49 / 0.50 / 0.50 | 1.00 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L5 (seeds<=10)` | `frozen_binary_4q_L5` | 10 / 10 | 0.942 / 0.952 | 0.21 / 0.43 / 0.48 | 0.03 | -0.10 | yes |  |
| sim-vs-real | FrozenLake | `su2_frozen_binary_4q_L5` | `frozen_binary_4q_noent_L5` | 10 / 10 | 0.000 / 0.000 | 0.55 / 0.50 / 0.58 | 1.00 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L5` | `frozen_scalar_1q_L5` | 10 / 10 | 0.102 / 0.125 | 0.49 / 0.50 / 0.53 | 0.99 | +0.10 | yes |  |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L10` | `frozen_scalar_1q_L10` | 10 / 10 | 0.046 / 0.081 | 0.56 / 0.45 / 0.45 | 0.67 | -0.12 | yes |  |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L15` | `frozen_scalar_1q_L15` | 10 / 10 | 0.010 / 0.011 | 0.46 / 0.50 / 0.47 | 0.81 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L5` | `frozen_binary_4q_L5` | 20 / 10 | 0.928 / 0.952 | 0.27 / 0.40 / 0.41 | 0.04 | -0.45 | yes | extra, not in rule (all sv seeds) |
| twin | CartPole | `sv_cartpole_L5_noent` | `su2_cartpole_L5 [init2pi]` | 10 / 10 | 42.942 / 43.673 | 0.41 / 0.38 / 0.54 | 0.55 | -0.14 | yes |  |
| twin | FrozenLake | `sv_frozen_binary_4q_noent_L1` | `su2_frozen_binary_4q_L1` | 10 / 10 | 0.000 / 0.000 | 0.56 / 0.50 / 0.33 | 0.47 |  | no |  |
| twin | FrozenLake | `sv_frozen_binary_4q_noent_L5` | `su2_frozen_binary_4q_L5` | 10 / 10 | 0.000 / 0.000 | 0.45 / 0.50 / 0.54 | 1.00 |  | no |  |
| real-vs-real | CartPole | `hybrid_fig4` | `hybrid_DR5` | 10 / 10 | 35.801 / 37.685 | 0.49 / 0.34 / 0.48 | 0.97 | -0.11 | yes | noise floor (not in rule) |

**Criterion 1 (no global bias):** mean d = +0.119, 95% CI (+0.010, +0.250), 21 informative pairs -> FAIL
**Criterion 2 (no excess of disagreements):** 3 of 63 cells outside [0.25, 0.75] = 5% (limit 25%) -> PASS
**Criterion 3 (the two simulators agree):** 0 of 9 twin cells outside [0.25, 0.75] -> PASS
**Overall: NOT ACCEPTED** (still under docs/SIMULATED-RESULTS-POLICY.md in either case).