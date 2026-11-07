# Experiment 13, blocks E, F, G - tables

> All arms are classical on the real environment; the reference rows (`real_*`) are the REAL circuits (10 seeds, other pipeline: upstream's exploration sampler was not seeded there).


## E - Fourier ceiling (CartPole)

| arm | params | n | final_perf_iqm | final_ci_lo | final_ci_hi | greedy_final_iqm | greedy_ci_lo | greedy_ci_hi | does_not_learn |
|---|---|---|---|---|---|---|---|---|---|
| ceiling_dup | 162.0 | 30 | 78.6 | 73.6 | 84.1 | 92.9 | 86.5 | 98.1 | False |
| ceiling_unique | 82.0 | 30 | 62.6 | 59.3 | 64.8 | 59.8 | 56.0 | 65.7 | False |
| ceiling_unique_lrx2 | 82.0 | 30 | 83.1 | 76.1 | 90.0 | 94.3 | 87.5 | 101.0 | False |
| ceiling_unique_lrx2_collapsed | 162.0 | 30 | 78.6 | 73.6 | 84.1 | 92.9 | 86.5 | 98.1 | False |
| real_ceiling_L5 |  | 10 | 76.5 | 69.6 | 94.1 | 84.7 | 77.1 | 104.7 | False |
| real_hybrid_fig4 |  | 10 | 35.8 | 24.0 | 48.2 | 47.3 | 31.3 | 68.4 | False |
| real_hybrid_fig4_noent |  | 10 | 43.5 | 39.7 | 51.3 | 46.4 | 38.4 | 80.4 | False |
| real_matched_classical |  | 10 | 11.8 | 10.6 | 12.6 | 11.1 | 10.1 | 12.5 | True |


### Contrasts E

| family | prediction | n_a | n_b | iqm_final_a | iqm_final_b | p_final | iqm_greedy_a | iqm_greedy_b | p_greedy | status |
|---|---|---|---|---|---|---|---|---|---|---|
| E_equivalence | E1: lrx2 (fresh init) = dup | 30 | 30 | 83.1 | 78.6 | 0.569 | 94.3 | 92.9 | 0.546 | equivalent |
| E_equivalence | E1c: lrx2 collapsed (V0 = sum of the blocks) = dup | 30 | 30 | 78.6 | 78.6 | 0.500 | 92.9 | 92.9 | 0.500 | equivalent |
| E_equivalence | E1d: fresh init = collapsed init, both lr x2 | 30 | 30 | 83.1 | 78.6 | 0.569 | 94.3 | 92.9 | 0.546 | equivalent |
| E_effect | E2: no prediction (effect of the x2): unique vs dup | 30 | 30 | 62.6 | 78.6 | 0.143 | 59.8 | 92.9 | 0.109 | not equivalent (indicates a direction (b > a)) |
| E_effect | E2b: unique vs unique lrx2 | 30 | 30 | 62.6 | 83.1 | 0.154 | 59.8 | 94.3 | 0.112 | not equivalent (indicates a direction (b > a)) |
| E_consistency | E0: dup rerun vs the ceiling of exp01 (other pipeline: seeded sampler) | 30 | 10 | 78.6 | 76.5 | 0.513 | 92.9 | 84.7 | 0.567 | equivalent |
| E_vs_real_hybrid | P(ceiling_dup > real hybrid no entanglement) | 30 | 10 | 78.6 | 43.5 | 0.963 | 92.9 | 46.4 | 0.827 | not equivalent (indicates a direction (a > b)) |
| E_vs_real_hybrid | P(ceiling_dup > real hybrid entangled) | 30 | 10 | 78.6 | 35.8 | 1.000 | 92.9 | 47.3 | 0.913 | not equivalent (indicates a direction (a > b)) |
| E_vs_real_hybrid | P(ceiling_unique > real hybrid no entanglement) | 30 | 10 | 62.6 | 43.5 | 0.850 | 59.8 | 46.4 | 0.730 | not equivalent (indicates a direction (a > b)) |
| E_vs_real_hybrid | P(ceiling_unique > real hybrid entangled) | 30 | 10 | 62.6 | 35.8 | 0.930 | 59.8 | 47.3 | 0.693 | not equivalent (indicates a direction (a > b)) |
| E_vs_real_hybrid | P(ceiling_unique_lrx2 > real hybrid no entanglement) | 30 | 10 | 83.1 | 43.5 | 0.960 | 94.3 | 46.4 | 0.813 | not equivalent (indicates a direction (a > b)) |
| E_vs_real_hybrid | P(ceiling_unique_lrx2 > real hybrid entangled) | 30 | 10 | 83.1 | 35.8 | 0.993 | 94.3 | 47.3 | 0.923 | not equivalent (indicates a direction (a > b)) |
| E_vs_real_hybrid | P(ceiling_unique_lrx2_collapsed > real hybrid no entanglement) | 30 | 10 | 78.6 | 43.5 | 0.963 | 92.9 | 46.4 | 0.827 | not equivalent (indicates a direction (a > b)) |
| E_vs_real_hybrid | P(ceiling_unique_lrx2_collapsed > real hybrid entangled) | 30 | 10 | 78.6 | 35.8 | 1.000 | 92.9 | 47.3 | 0.913 | not equivalent (indicates a direction (a > b)) |


## F - capacity-matched control (CartPole)

| arm | params | n | final_perf_iqm | final_ci_lo | final_ci_hi | greedy_final_iqm | greedy_ci_lo | greedy_ci_hi | does_not_learn |
|---|---|---|---|---|---|---|---|---|---|
| matched_rep4_w7 | 135.0 | 30 | 10.8 | 10.3 | 11.3 | 10.3 | 9.935 | 10.9 | True |
| matched_norep_w7_lrx4 | 51.0 | 30 | 11.0 | 10.4 | 11.8 | 10.7 | 10.1 | 11.5 | True |
| matched_norep_w7_lrx4_collapsed | 135.0 | 30 | 10.8 | 10.3 | 11.3 | 10.3 | 9.968 | 10.9 | True |
| matched_norep_w18 | 128.0 | 30 | 14.1 | 12.2 | 17.8 | 17.9 | 13.9 | 22.7 | False |
| matched_norep_w18_lrx4 | 128.0 | 30 | 21.5 | 17.9 | 26.3 | 24.9 | 20.4 | 30.0 | False |
| real_ceiling_L5 |  | 10 | 76.5 | 69.6 | 94.1 | 84.7 | 77.1 | 104.7 | False |
| real_hybrid_fig4 |  | 10 | 35.8 | 24.0 | 48.2 | 47.3 | 31.3 | 68.4 | False |
| real_hybrid_fig4_noent |  | 10 | 43.5 | 39.7 | 51.3 | 46.4 | 38.4 | 80.4 | False |
| real_matched_classical |  | 10 | 11.8 | 10.6 | 12.6 | 11.1 | 10.1 | 12.5 | True |


### Contrasts F

| family | prediction | n_a | n_b | iqm_final_a | iqm_final_b | p_final | iqm_greedy_a | iqm_greedy_b | p_greedy | status |
|---|---|---|---|---|---|---|---|---|---|---|
| F_equivalence | F1: norep w7 lrx4 (fresh init) = rep4 w7 | 30 | 30 | 11.0 | 10.8 | 0.537 | 10.7 | 10.3 | 0.525 | equivalent |
| F_equivalence | F1c: norep w7 lrx4 collapsed = rep4 w7 | 30 | 30 | 10.8 | 10.8 | 0.499 | 10.3 | 10.3 | 0.505 | equivalent |
| F_equivalence | F1d: fresh = collapsed init, both lr x4 | 30 | 30 | 11.0 | 10.8 | 0.537 | 10.7 | 10.3 | 0.524 | equivalent |
| F_consistency | F0: rep4 rerun vs matched_classical of exp01 (other pipeline: seeded sampler) | 30 | 10 | 10.8 | 11.8 | 0.350 | 10.3 | 11.1 | 0.387 | equivalent |
| F_four_arms | no prediction: matched_rep4_w7 vs matched_norep_w18 | 30 | 30 | 10.8 | 14.1 | 0.172 | 10.3 | 17.9 | 0.229 | not equivalent (indicates a direction (b > a)) |
| F_four_arms | no prediction: matched_rep4_w7 vs matched_norep_w18_lrx4 | 30 | 30 | 10.8 | 21.5 | 0.030 | 10.3 | 24.9 | 0.048 | not equivalent (strong (reverse)) |
| F_four_arms | no prediction: matched_norep_w7_lrx4 vs matched_norep_w18 | 30 | 30 | 11.0 | 14.1 | 0.236 | 10.7 | 17.9 | 0.220 | not equivalent (indicates a direction (b > a)) |
| F_four_arms | no prediction: matched_norep_w7_lrx4 vs matched_norep_w18_lrx4 | 30 | 30 | 11.0 | 21.5 | 0.048 | 10.7 | 24.9 | 0.039 | not equivalent (strong (reverse)) |
| F_four_arms | no prediction: matched_norep_w18 vs matched_norep_w18_lrx4 | 30 | 30 | 14.1 | 21.5 | 0.254 | 17.9 | 24.9 | 0.323 | inconclusive |
| F_vs_real_hybrid | P(real hybrid without reuse > matched_rep4_w7) | 10 | 30 | 35.8 | 10.8 | 0.993 | 47.3 | 10.3 | 0.993 | not equivalent (strong) |
| F_vs_real_hybrid_noent | P(real hybrid noent > matched_rep4_w7) | 10 | 30 | 43.5 | 10.8 | 1.000 | 46.4 | 10.3 | 0.997 | not equivalent (strong) |
| F_vs_real_hybrid | P(real hybrid without reuse > matched_norep_w7_lrx4) | 10 | 30 | 35.8 | 11.0 | 0.987 | 47.3 | 10.7 | 1.000 | not equivalent (strong) |
| F_vs_real_hybrid_noent | P(real hybrid noent > matched_norep_w7_lrx4) | 10 | 30 | 43.5 | 11.0 | 1.000 | 46.4 | 10.7 | 1.000 | not equivalent (strong) |
| F_vs_real_hybrid | P(real hybrid without reuse > matched_norep_w7_lrx4_collapsed) | 10 | 30 | 35.8 | 10.8 | 0.993 | 47.3 | 10.3 | 0.993 | not equivalent (strong) |
| F_vs_real_hybrid_noent | P(real hybrid noent > matched_norep_w7_lrx4_collapsed) | 10 | 30 | 43.5 | 10.8 | 1.000 | 46.4 | 10.3 | 0.997 | not equivalent (strong) |
| F_vs_real_hybrid | P(real hybrid without reuse > matched_norep_w18) | 10 | 30 | 35.8 | 14.1 | 0.877 | 47.3 | 17.9 | 0.870 | not equivalent (indicates a direction (a > b)) |
| F_vs_real_hybrid_noent | P(real hybrid noent > matched_norep_w18) | 10 | 30 | 43.5 | 14.1 | 0.997 | 46.4 | 17.9 | 0.943 | not equivalent (indicates a direction (a > b)) |
| F_vs_real_hybrid | P(real hybrid without reuse > matched_norep_w18_lrx4) | 10 | 30 | 35.8 | 21.5 | 0.743 | 47.3 | 24.9 | 0.793 | not equivalent (indicates a direction (a > b)) |
| F_vs_real_hybrid_noent | P(real hybrid noent > matched_norep_w18_lrx4) | 10 | 30 | 43.5 | 21.5 | 0.957 | 46.4 | 24.9 | 0.890 | not equivalent (indicates a direction (a > b)) |


## G - MLP on the 4 bits (FrozenLake)

| arm | params | n | final_perf_iqm | final_ci_lo | final_ci_hi | greedy_final_iqm | greedy_ci_lo | greedy_ci_hi | n_solved_seeds |
|---|---|---|---|---|---|---|---|---|---|
| mlp_bits_w6 | 58.0 | 30 | 0.038 | 0.013 | 0.079 | 0.000 | 0.000 | 0.000 | 0.000 |
| mlp_bits_w12 | 112.0 | 30 | 0.750 | 0.486 | 0.948 | 0.625 | 0.312 | 0.917 | 15.0 |
| real_frozen_binary_4q_L1 |  | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real_frozen_binary_4q_L5 |  | 10 | 0.952 | 0.806 | 0.957 | 0.944 | 0.722 | 1.000 | 7.000 |
| real_frozen_binary_4q_noent_L1 |  | 10 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| real_frozen_binary_4q_noent_L5 |  | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real_frozen_matched_scalar |  | 10 | 0.001 | 0.000 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 |


### Contrasts G

| family | prediction | n_a | n_b | iqm_final_a | iqm_final_b | p_final | iqm_greedy_a | iqm_greedy_b | p_greedy | status |
|---|---|---|---|---|---|---|---|---|---|---|
| G_vs_real_circuit | P(mlp_bits_w6 > binary entangled circuit L=1) | 30 | 10 | 0.038 | 0.000 | 0.917 | 0.000 | 0.000 | 0.500 | not equivalent (indicates a direction (a > b)) |
| G_vs_real_circuit | P(mlp_bits_w6 > binary entangled circuit L=5) | 30 | 10 | 0.038 | 0.952 | 0.030 | 0.000 | 0.944 | 0.050 | not equivalent (strong (reverse)) |
| G_vs_real_circuit | P(mlp_bits_w12 > binary entangled circuit L=1) | 30 | 10 | 0.750 | 0.000 | 1.000 | 0.625 | 0.000 | 0.817 | not equivalent (indicates a direction (a > b)) |
| G_vs_real_circuit | P(mlp_bits_w12 > binary entangled circuit L=5) | 30 | 10 | 0.750 | 0.952 | 0.310 | 0.625 | 0.944 | 0.363 | inconclusive |
| G_pair | no prediction: w12 vs w6 | 30 | 30 | 0.750 | 0.038 | 0.888 | 0.625 | 0.000 | 0.817 | not equivalent (indicates a direction (a > b)) |


## float64 traces (E, F)

| pair | seed | first_diff_step | max_param_diff_to_first_diff | max_q_diff_final_2000_states | greedy_disagreement_final | n_steps |
|---|---|---|---|---|---|---|
| E_ceiling | 1 | None | 9.769962616701378e-15 | 3.019806626980426e-14 | 0.0 | 100001 |
| E_ceiling | 2 | None | 6.661338147750939e-15 | 2.842170943040401e-14 | 0.0 | 100001 |
| E_ceiling | 3 | None | 9.103828801926284e-15 | 3.375077994860476e-14 | 0.0 | 100001 |
| F_matched | 1 | None | 1.1546319456101628e-14 | 2.842170943040401e-14 | 0.0 | 100001 |
| F_matched | 2 | None | 9.325873406851315e-15 | 4.263256414560601e-14 | 0.0 | 100001 |
| F_matched | 3 | None | 1.2434497875801753e-14 | 6.039613253960852e-14 | 0.0 | 100001 |
