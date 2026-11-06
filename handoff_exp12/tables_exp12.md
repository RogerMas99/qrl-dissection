> Real CartPole environment, classical twin arms. No simulator is involved. Protocol, contrasts and reading rule: docs/EXPERIMENT-12-CLASSICAL-TWIN.md (written before any cell ran).

## Parameters (CartPole-v1, measured by building each arm)

| R | hybrid total | twin total (bounded and unbounded) | extra vs hybrid | head | extractor |
|---|---|---|---|---|---|
| 1 | 126 | 132 | +6 | 46 | 86 |
| 4 | 222 | 228 | +6 | 142 | 86 |
| 8 | 350 | 356 | +6 | 270 | 86 |
| 16 | 606 | 612 | +6 | 526 | 86 |
| 32 | 1118 | 1124 | +6 | 1038 | 86 |

Head and OutputReuse of the twin are the hybrid's (same code, same shapes, same initial weights under one seed; `tests/test_exp12_arms.py`).

## IQM per arm (95 % bootstrap CI)

| arm | exp | n | params | final_perf IQM (CI) | greedy_final IQM (CI) |
|---|---|---|---|---|---|
| `hybrid_fig4` | exp01 | 10 | 126 | 35.8 (24.0-48.2) | 47.3 (31.3-68.4) |
| `hybrid_OR4` | exp02 | 10 | 222 | 94.7 (81.0-111.5) | 112.0 (91.3-152.4) |
| `hybrid_OR8` | exp02 | 10 | 350 | 109.8 (93.8-125.1) | 124.3 (107.2-189.9) |
| `hybrid_OR16` | exp02 | 10 | 606 | 104.2 (76.7-123.6) | 146.2 (109.7-180.0) |
| `hybrid_OR32` | exp02 | 10 | 1118 | 137.6 (106.3-179.7) | 161.1 (120.0-224.0) |
| `classical_twin_r1` | exp12 | 10 | 132 | 13.9 (11.4-30.2) | 29.8 (14.8-79.9) |
| `classical_twin_r4` | exp12 | 10 | 228 | 120.6 (79.5-166.7) | 212.2 (154.2-299.7) |
| `classical_twin_r8` | exp12 | 10 | 356 | 171.0 (91.6-262.5) | 292.1 (152.8-425.2) |
| `classical_twin_r16` | exp12 | 10 | 612 | 212.3 (152.5-288.6) | 276.6 (214.7-395.2) |
| `classical_twin_r32` | exp12 | 10 | 1124 | 198.6 (157.6-235.4) | 230.7 (159.4-299.4) |
| `classical_twin_unbounded_r1` | exp12 | 10 | 132 | 37.4 (17.0-63.2) | 70.7 (22.2-117.9) |
| `classical_twin_unbounded_r4` | exp12 | 10 | 228 | 120.0 (72.0-163.5) | 179.3 (138.2-248.3) |
| `classical_twin_unbounded_r8` | exp12 | 10 | 356 | 112.2 (72.5-146.6) | 175.1 (119.0-209.0) |
| `classical_twin_unbounded_r16` | exp12 | 10 | 612 | 173.7 (127.2-235.5) | 210.3 (149.7-297.1) |
| `classical_twin_unbounded_r32` | exp12 | 10 | 1124 | 184.5 (130.0-279.1) | 275.7 (204.6-337.8) |

## G1 - effect of Output Reuse in the bounded twin

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| R=4: twin R vs twin R=1 | 10 / 10 | 120.6 / 13.9 | 0.97 | 212.2 / 29.8 | 0.93 | indicates a direction (a > b) |
| R=8: twin R vs twin R=1 | 10 / 10 | 171.0 / 13.9 | 0.96 | 292.1 / 29.8 | 0.94 | indicates a direction (a > b) |
| R=16: twin R vs twin R=1 | 10 / 10 | 212.3 / 13.9 | 1.00 | 276.6 / 29.8 | 0.96 | strong |
| R=32: twin R vs twin R=1 | 10 / 10 | 198.6 / 13.9 | 0.99 | 230.7 / 29.8 | 0.95 | strong |

## G2 - effect of Output Reuse in the hybrid (reference, same scripts)

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| R=4: hybrid_OR4 vs hybrid_fig4 | 10 / 10 | 94.7 / 35.8 | 0.97 | 112.0 / 47.3 | 0.96 | strong |
| R=8: hybrid_OR8 vs hybrid_fig4 | 10 / 10 | 109.8 / 35.8 | 1.00 | 124.3 / 47.3 | 0.99 | strong |
| R=16: hybrid_OR16 vs hybrid_fig4 | 10 / 10 | 104.2 / 35.8 | 0.96 | 146.2 / 47.3 | 0.96 | strong |
| R=32: hybrid_OR32 vs hybrid_fig4 | 10 / 10 | 137.6 / 35.8 | 1.00 | 161.1 / 47.3 | 0.99 | strong |

## G3 - circuit against twin at equal R

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| R=1: hybrid_fig4 vs twin R=1 | 10 / 10 | 35.8 / 13.9 | 0.79 | 47.3 / 29.8 | 0.65 | indicates a direction (a > b) |
| R=4: hybrid_OR4 vs twin R=4 | 10 / 10 | 94.7 / 120.6 | 0.38 | 112.0 / 212.2 | 0.20 | indicates a direction (b > a) |
| R=8: hybrid_OR8 vs twin R=8 | 10 / 10 | 109.8 / 171.0 | 0.36 | 124.3 / 292.1 | 0.27 | near 0.5 |
| R=16: hybrid_OR16 vs twin R=16 | 10 / 10 | 104.2 / 212.3 | 0.09 | 146.2 / 276.6 | 0.06 | indicates a direction (b > a) |
| R=32: hybrid_OR32 vs twin R=32 | 10 / 10 | 137.6 / 198.6 | 0.26 | 161.1 / 230.7 | 0.35 | near 0.5 |

## G4 - role of the bounded range: bounded twin vs unbounded twin, per R

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| R=1: bounded twin vs unbounded twin | 10 / 10 | 13.9 / 37.4 | 0.34 | 29.8 / 70.7 | 0.40 | near 0.5 |
| R=4: bounded twin vs unbounded twin | 10 / 10 | 120.6 / 120.0 | 0.57 | 212.2 / 179.3 | 0.58 | near 0.5 |
| R=8: bounded twin vs unbounded twin | 10 / 10 | 171.0 / 112.2 | 0.65 | 292.1 / 175.1 | 0.72 | near 0.5 |
| R=16: bounded twin vs unbounded twin | 10 / 10 | 212.3 / 173.7 | 0.62 | 276.6 / 210.3 | 0.69 | near 0.5 |
| R=32: bounded twin vs unbounded twin | 10 / 10 | 198.6 / 184.5 | 0.55 | 230.7 / 275.7 | 0.40 | near 0.5 |

## G4 (repeated G1) - effect of Output Reuse in the unbounded twin

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| R=4: unbounded twin R vs unbounded twin R=1 | 10 / 10 | 120.0 / 37.4 | 0.84 | 179.3 / 70.7 | 0.91 | indicates a direction (a > b) |
| R=8: unbounded twin R vs unbounded twin R=1 | 10 / 10 | 112.2 / 37.4 | 0.88 | 175.1 / 70.7 | 0.86 | indicates a direction (a > b) |
| R=16: unbounded twin R vs unbounded twin R=1 | 10 / 10 | 173.7 / 37.4 | 1.00 | 210.3 / 70.7 | 0.92 | indicates a direction (a > b) |
| R=32: unbounded twin R vs unbounded twin R=1 | 10 / 10 | 184.5 / 37.4 | 0.97 | 275.7 / 70.7 | 0.95 | strong |

## Context (not part of the reading rule) - circuit against the unbounded twin

| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |
|---|---|---|---|---|---|---|
| R=1: hybrid_fig4 vs unbounded twin R=1 | 10 / 10 | 35.8 / 37.4 | 0.50 | 47.3 / 70.7 | 0.41 | near 0.5 |
| R=4: hybrid_OR4 vs unbounded twin R=4 | 10 / 10 | 94.7 / 120.0 | 0.34 | 112.0 / 179.3 | 0.23 | indicates a direction (b > a) |
| R=8: hybrid_OR8 vs unbounded twin R=8 | 10 / 10 | 109.8 / 112.2 | 0.47 | 124.3 / 175.1 | 0.31 | near 0.5 |
| R=16: hybrid_OR16 vs unbounded twin R=16 | 10 / 10 | 104.2 / 173.7 | 0.14 | 146.2 / 210.3 | 0.22 | indicates a direction (b > a) |
| R=32: hybrid_OR32 vs unbounded twin R=32 | 10 / 10 | 137.6 / 184.5 | 0.37 | 161.1 / 275.7 | 0.24 | indicates a direction (b > a) |

## Descriptive - gain of Output Reuse relative to R = 1

| family | R | ratio final_perf IQM(R)/IQM(1) (95 % CI) | ratio greedy_final IQM(R)/IQM(1) (95 % CI) |
|---|---|---|---|
| bounded twin | 4 | 8.66 (3.52-12.62) | 7.13 (2.56-15.49) |
| bounded twin | 8 | 12.28 (4.49-19.56) | 9.82 (2.99-21.32) |
| bounded twin | 16 | 15.25 (6.31-21.83) | 9.30 (3.34-20.25) |
| bounded twin | 32 | 14.26 (6.13-18.47) | 7.75 (2.66-16.03) |
| unbounded twin | 4 | 3.21 (1.55-7.47) | 2.54 (1.42-7.92) |
| unbounded twin | 8 | 3.00 (1.52-6.83) | 2.48 (1.32-7.10) |
| unbounded twin | 16 | 4.65 (2.45-10.78) | 2.97 (1.59-9.04) |
| unbounded twin | 32 | 4.94 (2.61-11.86) | 3.90 (2.17-11.66) |
| hybrid | 4 | 2.64 (1.88-4.06) | 2.37 (1.53-3.98) |
| hybrid | 8 | 3.07 (2.21-4.68) | 2.63 (1.77-4.73) |
| hybrid | 16 | 2.91 (1.91-4.47) | 3.09 (1.95-4.86) |
| hybrid | 32 | 3.84 (2.61-6.28) | 3.41 (2.13-5.80) |

## Mechanical application of the reading rule

| check | result |
|---|---|
| bounded twin R=1: IQM final_perf / greedy_final | 13.9 / 29.8 (learns: True) |
| G1 per R = 4, 8, 16, 32 | indicates a direction (a > b), indicates a direction (a > b), strong, strong |
| G1 near 0.5 at all four R | False |
| G1 strong at least at one R | True |
| G2 per R = 4, 8, 16, 32 | strong, strong, strong, strong |
| G2 strong at least at one R (the hybrid gains) | True |
| G4 repeated (unbounded twin, R vs R=1) per R | indicates a direction (a > b), indicates a direction (a > b), indicates a direction (a > b), strong |
| G4 (bounded vs unbounded) per R = 4, 8, 16, 32 | near 0.5, near 0.5, near 0.5, near 0.5 |
| bounded twin does not learn at any R (IQM <= 12 in both metrics at R = 1, 4, 8, 16, 32) | False |
| unbounded twin does not learn at any R | False |

- **Branch 1** (twin learns without OR, OR gives it nothing, OR gives the hybrid a gain): does not hold (learns at R=1: True; G1 near 0.5 at all R: False; hybrid gain: True).
- **Branch 2** (OR gives the twin a strong gain): holds (G1 strong at at least one R: True).
- **Branch 4** (the bounded twin does not learn at any R): does not hold.
- **Branch 3** (G4), per R: R=4: not evaluable (OR does not help either twin strongly); R=8: not evaluable (OR does not help either twin strongly); R=16: tied to the bounded range; R=32: does not depend on the range.

Branches of the rule that hold mechanically: 2 (branch 3 is conditional, see its per-R line). Intermediate values are not forced into a branch; the dossier states which branch holds, which do not and why, and says so if none holds cleanly.

No multiple-comparison correction is applied to any of the contrasts above.