> Real CartPole environment, classical control arms. No simulator is involved. Protocol, contrasts and reading rule: docs/EXPERIMENT-11-REPLICATION-CONTROL.md (written before any cell ran).

## Widths and measured parameters (CartPole-v1)

| R | hybrid_OR total | classical_or_matched_r (replication fixed at 4): width / params | classical_or_matched_rep_r (replication R): width / params |
|---|---|---|---|
| 4 | 222 | 12 / 230 | 12 / 230 (identical to the previous column: not re-run) |
| 8 | 350 | 18 / 344 | 10 / 352 |
| 16 | 606 | 32 / 610 | 9 / 605 |
| 32 | 1118 | 59 / 1123 | 9 / 1181 |

R = 4 identical to classical_or_matched_r4: True.

## IQM per arm (95 % bootstrap CI)

| arm | exp | n | params | final_perf IQM (CI) | greedy_final IQM (CI) |
|---|---|---|---|---|---|
| `hybrid_OR4` | exp02 | 10 | 222 | 94.7 (81.0-111.5) | 112.0 (91.3-152.4) |
| `hybrid_OR8` | exp02 | 10 | 350 | 109.8 (93.8-125.1) | 124.3 (107.2-189.9) |
| `hybrid_OR16` | exp02 | 10 | 606 | 104.2 (76.7-123.6) | 146.2 (109.7-180.0) |
| `hybrid_OR32` | exp02 | 10 | 1118 | 137.6 (106.3-179.7) | 161.1 (120.0-224.0) |
| `hybrid_or_r4_noent` | exp08 | 10 |  | 60.4 (46.6-74.3) | 65.4 (50.4-83.7) |
| `hybrid_or_r8_noent` | exp08 | 10 |  | 69.1 (55.3-82.0) | 86.9 (66.3-106.5) |
| `hybrid_or_r16_noent` | exp08 | 10 |  | 89.5 (72.6-104.0) | 103.9 (83.8-123.8) |
| `hybrid_or_r32_noent` | exp08 | 10 |  | 77.9 (64.3-91.5) | 96.6 (80.2-106.3) |
| `classical_or_matched_r4` | exp02 | 10 | 230 | 12.4 (11.5-15.5) | 13.8 (10.9-17.5) |
| `classical_or_matched_r8` | exp02 | 10 | 344 | 23.8 (20.5-27.8) | 24.8 (18.0-36.9) |
| `classical_or_matched_r16` | exp02 | 10 | 610 | 36.3 (28.4-43.5) | 43.0 (32.5-59.3) |
| `classical_or_matched_r32` | exp02 | 10 | 1123 | 61.2 (46.0-71.5) | 88.5 (62.3-117.4) |
| `classical_or_matched_rep_r8` | exp11 | 10 | 352 | 10.0 (9.7-11.1) | 9.5 (9.3-10.6) |
| `classical_or_matched_rep_r16` | exp11 | 10 | 605 | 10.6 (9.8-14.7) | 9.8 (9.4-13.3) |
| `classical_or_matched_rep_r32` | exp11 | 10 | 1181 | 13.0 (9.8-17.3) | 15.0 (9.8-22.4) |
| `classical_OR4` | exp02 | 10 |  | 9.6 (9.6-9.7) | 9.5 (9.3-9.7) |
| `classical_OR8` | exp02 | 10 |  | 9.6 (9.6-9.6) | 9.5 (9.3-9.6) |
| `classical_OR16` | exp02 | 10 |  | 9.6 (9.6-27.7) | 9.6 (9.4-28.6) |
| `classical_OR32` | exp02 | 10 |  | 10.2 (10.2-10.2) | 10.0 (10.0-10.1) |

## C1 - effect of replication in a classical control at equal budget

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| R=8: replication R vs replication 4 (same budget) | 10 / 10 | 10.0 / 23.8 | 0.00 | 9.5 / 24.8 | 0.02 | strong (reverse) |
| R=16: replication R vs replication 4 (same budget) | 10 / 10 | 10.6 / 36.3 | 0.03 | 9.8 / 43.0 | 0.00 | strong (reverse) |
| R=32: replication R vs replication 4 (same budget) | 10 / 10 | 13.0 / 61.2 | 0.00 | 15.0 / 88.5 | 0.00 | strong (reverse) |

## C2 - the circuit against the control with the same replication

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| R=4: hybrid_OR4 vs control replicating R | 10 / 10 | 94.7 / 12.4 | 1.00 | 112.0 / 13.8 | 1.00 | strong |
| R=8: hybrid_OR8 vs control replicating R | 10 / 10 | 109.8 / 10.0 | 1.00 | 124.3 / 9.5 | 1.00 | strong |
| R=16: hybrid_OR16 vs control replicating R | 10 / 10 | 104.2 / 10.6 | 1.00 | 146.2 / 9.8 | 1.00 | strong |
| R=32: hybrid_OR32 vs control replicating R | 10 / 10 | 137.6 / 13.0 | 1.00 | 161.1 / 15.0 | 1.00 | strong |

## C3 - trend with R inside the new control

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| rep_r32 vs rep_r4 (= matched_r4) | 10 / 10 | 13.0 / 12.4 | 0.43 | 15.0 / 13.8 | 0.49 | near 0.5 |
| rep_r16 vs rep_r4 (= matched_r4) | 10 / 10 | 10.6 / 12.4 | 0.24 | 9.8 / 13.8 | 0.17 | indicates a direction (b > a) |

## C4 (context) - the unentangled circuit against the control with the same replication

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| R=4: hybrid_or_r4_noent vs control replicating R | 10 / 10 | 60.4 / 12.4 | 1.00 | 65.4 / 13.8 | 1.00 | strong |
| R=8: hybrid_or_r8_noent vs control replicating R | 10 / 10 | 69.1 / 10.0 | 1.00 | 86.9 / 9.5 | 1.00 | strong |
| R=16: hybrid_or_r16_noent vs control replicating R | 10 / 10 | 89.5 / 10.6 | 1.00 | 103.9 / 9.8 | 1.00 | strong |
| R=32: hybrid_or_r32_noent vs control replicating R | 10 / 10 | 77.9 / 13.0 | 1.00 | 96.6 / 15.0 | 1.00 | strong |

## Mechanical application of the reading rule

| check | result |
|---|---|
| C1 and C3 all `near 0.5` | False |
| any C1 or C3 `strong` (replication helps the control) | False |
| every C2 `strong` (circuit above the control) | True |
| control does not learn (IQM final_perf <= 12 and IQM greedy_final <= 12 at every R in 4, 8, 16, 32) | False |
| R=4 control: IQM final_perf / greedy_final | 12.4 / 13.8 - at the limit of the threshold |
| R=8 control: IQM final_perf / greedy_final | 10.0 / 9.5 - at the limit of the threshold |
| R=16 control: IQM final_perf / greedy_final | 10.6 / 9.8 - at the limit of the threshold |
| R=32 control: IQM final_perf / greedy_final | 13.0 / 15.0 - at the limit of the threshold |

No multiple-comparison correction is applied to any of the contrasts above.