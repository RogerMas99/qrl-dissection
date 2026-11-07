# Experiment 13 - Output Reuse against the learning rate of the head

> Agent H runs on the batched statevector EMULATOR: simulated results (docs/SIMULATED-RESULTS-POLICY.md; approval of 2026-11-06 recorded there). Agent T and block D are classical arms on the real environment.

Design, predictions and criteria: docs/EXPERIMENT-13-OR-VS-LR.md (written before any cell ran). No multiple-comparison correction. Equivalence = P(a>b) in [0.35, 0.65] in both metrics and overlapping 95 % CIs of both IQMs. 'Strong' = P >= 0.95 in both metrics.

## Cells per arm

| agent | arms | cells |
|---|---|---|
| T | 22 | 660 |
| H | 18 | 540 |
| PPO | 20 | 600 |

## Agent T: IQM per arm
### Block A

| arm | n | IQM final_perf (95 % CI) | IQM greedy_final (95 % CI) | learns |
|---|---|---|---|---|
| classical_twin_reuse_r1 | 30 | 16.0 (13.4-21.5) | 24.2 (17.9-37.9) | yes |
| classical_twin_reuse_r4 | 30 | 112.5 (89.1-132.6) | 169.8 (136.9-212.6) | yes |
| classical_twin_reuse_r8 | 30 | 151.5 (116.8-190.2) | 223.5 (173.3-276.5) | yes |
| classical_twin_reuse_r16 | 30 | 188.0 (152.2-225.4) | 273.3 (225.5-327.6) | yes |
| classical_twin_reuse_r32 | 30 | 189.5 (154.8-238.1) | 278.8 (214.9-349.5) | yes |
| classical_twin_collapsed_r4 | 30 | 110.5 (88.5-131.3) | 195.5 (152.1-254.7) | yes |
| classical_twin_collapsed_r8 | 30 | 153.1 (120.5-191.0) | 223.2 (175.1-277.0) | yes |
| classical_twin_collapsed_r16 | 30 | 190.4 (155.6-228.7) | 258.5 (205.3-329.2) | yes |
| classical_twin_collapsed_r32 | 30 | 189.0 (155.0-237.9) | 262.1 (204.5-330.9) | yes |
| classical_twin_gradx8 | 30 | 15.0 (12.8-19.5) | 22.9 (18.1-34.3) | yes |
| classical_twin_gradx32 | 30 | 15.7 (13.0-20.1) | 20.8 (16.8-30.9) | yes |

### Block B (fresh, lr of head.weight x k)

| arm | n | IQM final_perf (95 % CI) | IQM greedy_final (95 % CI) | learns |
|---|---|---|---|---|
| classical_twin_headlr_x2 | 30 | 44.4 (30.6-63.4) | 72.7 (51.3-101.6) | yes |
| classical_twin_headlr_x4 | 30 | 86.2 (65.1-106.7) | 148.1 (119.2-179.7) | yes |
| classical_twin_headlr_x8 | 30 | 134.2 (109.9-165.2) | 193.7 (157.0-246.2) | yes |
| classical_twin_headlr_x16 | 30 | 180.3 (149.5-216.6) | 269.0 (214.4-329.0) | yes |
| classical_twin_headlr_x32 | 30 | 185.1 (140.1-240.8) | 247.2 (190.7-313.6) | yes |
| classical_twin_headlr_x64 | 30 | 216.8 (161.7-277.4) | 265.2 (205.1-337.2) | yes |
| classical_twin_headlr_x128 | 30 | 240.7 (196.2-282.7) | 309.9 (272.2-343.1) | yes |

### Block C (global lr x k, exploratory)

| arm | n | IQM final_perf (95 % CI) | IQM greedy_final (95 % CI) | learns |
|---|---|---|---|---|
| classical_twin_globallr_x0p5 | 30 | 12.9 (11.4-14.6) | 13.4 (11.3-16.0) | yes |
| classical_twin_globallr_x2 | 30 | 30.2 (16.5-60.6) | 56.3 (29.9-102.6) | yes |
| classical_twin_globallr_x4 | 30 | 138.4 (81.8-193.5) | 160.9 (105.9-214.5) | yes |
| classical_twin_globallr_x8 | 30 | 207.5 (183.5-239.5) | 234.6 (198.2-290.8) | yes |

## Agent T: contrasts

| prediction | a vs b | n | IQM final a / b | P final | IQM greedy a / b | P greedy | CI overlap (f/g) | status |
|---|---|---|---|---|---|---|---|---|
| P1/P5 | classical_twin_collapsed_r4 vs classical_twin_reuse_r4 | 30/30 | 110.5 / 112.5 | 0.49 | 195.5 / 169.8 | 0.56 | yes / yes | equivalent |
| P2/P5 | classical_twin_headlr_x4 vs classical_twin_reuse_r4 | 30/30 | 86.2 / 112.5 | 0.38 | 148.1 / 169.8 | 0.42 | yes / yes | equivalent |
| P1/P5 | classical_twin_collapsed_r8 vs classical_twin_reuse_r8 | 30/30 | 153.1 / 151.5 | 0.52 | 223.2 / 223.5 | 0.51 | yes / yes | equivalent |
| P2/P5 | classical_twin_headlr_x8 vs classical_twin_reuse_r8 | 30/30 | 134.2 / 151.5 | 0.46 | 193.7 / 223.5 | 0.46 | yes / yes | equivalent |
| P1/P5 | classical_twin_collapsed_r16 vs classical_twin_reuse_r16 | 30/30 | 190.4 / 188.0 | 0.51 | 258.5 / 273.3 | 0.48 | yes / yes | equivalent |
| P2/P5 | classical_twin_headlr_x16 vs classical_twin_reuse_r16 | 30/30 | 180.3 / 188.0 | 0.49 | 269.0 / 273.3 | 0.49 | yes / yes | equivalent |
| P1/P5 | classical_twin_collapsed_r32 vs classical_twin_reuse_r32 | 30/30 | 189.0 / 189.5 | 0.50 | 262.1 / 278.8 | 0.49 | yes / yes | equivalent |
| P2/P5 | classical_twin_headlr_x32 vs classical_twin_reuse_r32 | 30/30 | 185.1 / 189.5 | 0.46 | 247.2 / 278.8 | 0.45 | yes / yes | equivalent |
| P3/P5 | classical_twin_gradx8 vs classical_twin_reuse_r1 | 30/30 | 15.0 / 16.0 | 0.47 | 22.9 / 24.2 | 0.48 | yes / yes | equivalent |
| P3/P5 | classical_twin_gradx32 vs classical_twin_reuse_r1 | 30/30 | 15.7 / 16.0 | 0.48 | 20.8 / 24.2 | 0.45 | yes / yes | equivalent |
| effect of OR | classical_twin_reuse_r4 vs classical_twin_reuse_r1 | 30/30 | 112.5 / 16.0 | 0.95 | 169.8 / 24.2 | 0.95 | no / no | not equivalent (indicates a direction (a > b)) |
| effect of OR | classical_twin_reuse_r8 vs classical_twin_reuse_r1 | 30/30 | 151.5 / 16.0 | 0.97 | 223.5 / 24.2 | 0.97 | no / no | not equivalent (strong) |
| effect of OR | classical_twin_reuse_r16 vs classical_twin_reuse_r1 | 30/30 | 188.0 / 16.0 | 0.99 | 273.3 / 24.2 | 0.99 | no / no | not equivalent (strong) |
| effect of OR | classical_twin_reuse_r32 vs classical_twin_reuse_r1 | 30/30 | 189.5 / 16.0 | 0.99 | 278.8 / 24.2 | 0.99 | no / no | not equivalent (strong) |
| P4 | classical_twin_headlr_x2 vs classical_twin_reuse_r1 | 30/30 | 44.4 / 16.0 | 0.75 | 72.7 / 24.2 | 0.75 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | classical_twin_headlr_x4 vs classical_twin_reuse_r1 | 30/30 | 86.2 / 16.0 | 0.88 | 148.1 / 24.2 | 0.88 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | classical_twin_headlr_x8 vs classical_twin_reuse_r1 | 30/30 | 134.2 / 16.0 | 0.99 | 193.7 / 24.2 | 0.98 | no / no | not equivalent (strong) |
| P4 | classical_twin_headlr_x16 vs classical_twin_reuse_r1 | 30/30 | 180.3 / 16.0 | 0.99 | 269.0 / 24.2 | 0.99 | no / no | not equivalent (strong) |
| P4 | classical_twin_headlr_x32 vs classical_twin_reuse_r1 | 30/30 | 185.1 / 16.0 | 0.99 | 247.2 / 24.2 | 0.99 | no / no | not equivalent (strong) |
| P4 | classical_twin_headlr_x64 vs classical_twin_reuse_r1 | 30/30 | 216.8 / 16.0 | 0.99 | 265.2 / 24.2 | 0.99 | no / no | not equivalent (strong) |
| P4 | classical_twin_headlr_x128 vs classical_twin_reuse_r1 | 30/30 | 240.7 / 16.0 | 1.00 | 309.9 / 24.2 | 0.99 | no / no | not equivalent (strong) |
| P4 | classical_twin_headlr_x128 vs classical_twin_reuse_r32 (best fresh (classical_twin_headlr_x128) vs best reuse (classical_twin_reuse_r32), chosen by IQM final_perf) | 30/30 | 240.7 / 189.5 | 0.60 | 309.9 / 278.8 | 0.54 | yes / yes | equivalent |
| P6 | classical_twin_globallr_x0p5 vs classical_twin_reuse_r1 | 30/30 | 12.9 / 16.0 | 0.33 | 13.4 / 24.2 | 0.27 | yes / no | inconclusive |
| P6 | classical_twin_globallr_x2 vs classical_twin_reuse_r1 | 30/30 | 30.2 / 16.0 | 0.66 | 56.3 / 24.2 | 0.69 | yes / yes | inconclusive |
| P6 | classical_twin_globallr_x4 vs classical_twin_reuse_r1 | 30/30 | 138.4 / 16.0 | 0.86 | 160.9 / 24.2 | 0.85 | no / no | not equivalent (indicates a direction (a > b)) |
| P6 | classical_twin_globallr_x8 vs classical_twin_reuse_r1 | 30/30 | 207.5 / 16.0 | 0.99 | 234.6 / 24.2 | 0.99 | no / no | not equivalent (strong) |

P1 collapsed vs reuse: 4 pairs: 4 equivalent  
P2 fresh x R vs reuse R: 4 pairs: 4 equivalent  
P3 gradient x R vs R = 1: 2 pairs: 2 equivalent

### Performance against k (fresh) and against R (reuse) - P4

| k or R | IQM final: fresh x k | IQM final: reuse R | IQM greedy: fresh x k | IQM greedy: reuse R |
|---|---|---|---|---|
| 1 | 16.0 (13.4-21.5) | 16.0 (13.4-21.5) | 24.2 (17.9-37.9) | 24.2 (17.9-37.9) |
| 2 | 44.4 (30.6-63.4) | - | 72.7 (51.3-101.6) | - |
| 4 | 86.2 (65.1-106.7) | 112.5 (89.1-132.6) | 148.1 (119.2-179.7) | 169.8 (136.9-212.6) |
| 8 | 134.2 (109.9-165.2) | 151.5 (116.8-190.2) | 193.7 (157.0-246.2) | 223.5 (173.3-276.5) |
| 16 | 180.3 (149.5-216.6) | 188.0 (152.2-225.4) | 269.0 (214.4-329.0) | 273.3 (225.5-327.6) |
| 32 | 185.1 (140.1-240.8) | 189.5 (154.8-238.1) | 247.2 (190.7-313.6) | 278.8 (214.9-349.5) |
| 64 | 216.8 (161.7-277.4) | - | 265.2 (205.1-337.2) | - |
| 128 | 240.7 (196.2-282.7) | - | 309.9 (272.2-343.1) | - |

### Shift of the IQM, reuse R minus fresh x R, with 95 % bootstrap CI

| R | shift_final | shift_final_lo | shift_final_hi | shift_greedy | shift_greedy_lo | shift_greedy_hi |
|---|---|---|---|---|---|---|
| 4 | 26.3 | -5.4 | 54.9 | 21.7 | -25.2 | 73.0 |
| 8 | 17.3 | -29.1 | 63.5 | 29.8 | -42.3 | 94.3 |
| 16 | 7.8 | -42.9 | 56.3 | 4.3 | -71.7 | 83.2 |
| 32 | 4.4 | -61.2 | 70.1 | 31.7 | -61.0 | 121.2 |

### Scale trajectory (median over seeds of ||V_eff||_F and mean |Q|, from the 1 000-step diagnostics)

| arm | V@10k | |Q|@10k | V@30k | |Q|@30k | V@60k | |Q|@60k | V@100k | |Q|@100k |
|---|---|---|---|---|---|---|---|---|
| classical_twin_reuse_r1 | 1.17 | 0.246 | 3.47 | 13 | 3.24 | 10.6 | 3.58 | 13.4 |
| classical_twin_reuse_r4 | 1.14 | 0.305 | 7.68 | 17 | 11.9 | 30.1 | 14.8 | 48.7 |
| classical_twin_reuse_r8 | 1.15 | 0.246 | 11.1 | 18.6 | 17.8 | 40.3 | 22.2 | 53.6 |
| classical_twin_reuse_r16 | 1.15 | 0.265 | 18.3 | 21.8 | 27.1 | 42.9 | 33.5 | 59.8 |
| classical_twin_reuse_r32 | 1.13 | 0.269 | 28.7 | 25.8 | 42.1 | 53.3 | 50 | 69.5 |
| classical_twin_collapsed_r4 | 1.14 | 0.305 | 7.68 | 17 | 11.9 | 30.1 | 14.9 | 49 |
| classical_twin_collapsed_r8 | 1.15 | 0.246 | 11.1 | 18.6 | 17.8 | 40.6 | 22.2 | 54.7 |
| classical_twin_collapsed_r16 | 1.15 | 0.265 | 18.3 | 21.8 | 27.1 | 43.6 | 33.7 | 61.8 |
| classical_twin_collapsed_r32 | 1.13 | 0.269 | 28.7 | 25.8 | 42.1 | 53.3 | 50 | 67.6 |
| classical_twin_headlr_x4 | 1.17 | 0.246 | 7.59 | 16.4 | 11.7 | 32.1 | 14.5 | 45.4 |
| classical_twin_headlr_x8 | 1.17 | 0.246 | 12.1 | 19.3 | 18.1 | 40.3 | 22.1 | 56.7 |
| classical_twin_headlr_x16 | 1.17 | 0.246 | 19.3 | 21.8 | 28.4 | 46.2 | 34.6 | 70.9 |
| classical_twin_headlr_x32 | 1.17 | 0.246 | 30.1 | 25.6 | 43.9 | 51.3 | 53.1 | 62.4 |


## Agent H: IQM per arm (EMULATOR, simulated)
### Block A

| arm | n | IQM final_perf (95 % CI) | IQM greedy_final (95 % CI) | learns |
|---|---|---|---|---|
| hybrid_emu_reuse_r1 | 30 | 43.9 (35.0-51.1) | 59.2 (46.4-73.6) | yes |
| hybrid_emu_reuse_r4 | 30 | 83.5 (71.6-93.6) | 93.8 (81.8-113.3) | yes |
| hybrid_emu_reuse_r8 | 30 | 100.2 (85.5-114.5) | 110.3 (92.7-129.7) | yes |
| hybrid_emu_reuse_r16 | 30 | 126.4 (115.1-141.4) | 145.5 (127.4-166.4) | yes |
| hybrid_emu_reuse_r32 | 30 | 127.5 (108.9-146.4) | 157.3 (132.7-184.2) | yes |
| hybrid_emu_collapsed_r4 | 30 | 84.5 (72.1-96.5) | 94.3 (82.3-114.6) | yes |
| hybrid_emu_collapsed_r8 | 30 | 100.2 (85.5-114.5) | 110.3 (92.7-129.7) | yes |
| hybrid_emu_collapsed_r16 | 30 | 125.2 (114.9-139.3) | 142.4 (126.6-162.1) | yes |
| hybrid_emu_collapsed_r32 | 30 | 126.6 (108.5-145.5) | 157.4 (132.9-184.3) | yes |
| hybrid_emu_gradx8 | 30 | 43.7 (34.9-51.1) | 55.1 (43.0-70.1) | yes |
| hybrid_emu_gradx32 | 30 | 43.8 (35.8-51.0) | 56.5 (43.8-73.4) | yes |

### Block B (fresh, lr of head.weight x k)

| arm | n | IQM final_perf (95 % CI) | IQM greedy_final (95 % CI) | learns |
|---|---|---|---|---|
| hybrid_emu_headlr_x2 | 30 | 59.6 (55.1-68.0) | 86.5 (74.3-101.6) | yes |
| hybrid_emu_headlr_x4 | 30 | 102.5 (84.4-117.6) | 114.2 (97.0-130.2) | yes |
| hybrid_emu_headlr_x8 | 30 | 107.8 (96.1-121.4) | 117.3 (105.8-131.1) | yes |
| hybrid_emu_headlr_x16 | 30 | 107.5 (96.1-120.9) | 125.5 (112.1-143.3) | yes |
| hybrid_emu_headlr_x32 | 30 | 113.9 (100.8-128.7) | 127.4 (112.8-147.1) | yes |
| hybrid_emu_headlr_x64 | 30 | 115.9 (98.5-138.6) | 140.2 (124.0-159.0) | yes |
| hybrid_emu_headlr_x128 | 30 | 137.0 (117.0-164.5) | 168.9 (137.5-205.2) | yes |

### Block C (global lr x k, exploratory)

| arm | n | IQM final_perf (95 % CI) | IQM greedy_final (95 % CI) | learns |
|---|---|---|---|---|

## Agent H: contrasts

| prediction | a vs b | n | IQM final a / b | P final | IQM greedy a / b | P greedy | CI overlap (f/g) | status |
|---|---|---|---|---|---|---|---|---|
| P1/P5 | hybrid_emu_collapsed_r4 vs hybrid_emu_reuse_r4 | 30/30 | 84.5 / 83.5 | 0.51 | 94.3 / 93.8 | 0.50 | yes / yes | equivalent |
| P2/P5 | hybrid_emu_headlr_x4 vs hybrid_emu_reuse_r4 | 30/30 | 102.5 / 83.5 | 0.63 | 114.2 / 93.8 | 0.59 | yes / yes | equivalent |
| P1/P5 | hybrid_emu_collapsed_r8 vs hybrid_emu_reuse_r8 | 30/30 | 100.2 / 100.2 | 0.50 | 110.3 / 110.3 | 0.50 | yes / yes | equivalent |
| P2/P5 | hybrid_emu_headlr_x8 vs hybrid_emu_reuse_r8 | 30/30 | 107.8 / 100.2 | 0.57 | 117.3 / 110.3 | 0.56 | yes / yes | equivalent |
| P1/P5 | hybrid_emu_collapsed_r16 vs hybrid_emu_reuse_r16 | 30/30 | 125.2 / 126.4 | 0.49 | 142.4 / 145.5 | 0.49 | yes / yes | equivalent |
| P2/P5 | hybrid_emu_headlr_x16 vs hybrid_emu_reuse_r16 | 30/30 | 107.5 / 126.4 | 0.35 | 125.5 / 145.5 | 0.40 | yes / yes | equivalent |
| P1/P5 | hybrid_emu_collapsed_r32 vs hybrid_emu_reuse_r32 | 30/30 | 126.6 / 127.5 | 0.49 | 157.4 / 157.3 | 0.50 | yes / yes | equivalent |
| P2/P5 | hybrid_emu_headlr_x32 vs hybrid_emu_reuse_r32 | 30/30 | 113.9 / 127.5 | 0.42 | 127.4 / 157.3 | 0.38 | yes / yes | equivalent |
| P3/P5 | hybrid_emu_gradx8 vs hybrid_emu_reuse_r1 | 30/30 | 43.7 / 43.9 | 0.50 | 55.1 / 59.2 | 0.47 | yes / yes | equivalent |
| P3/P5 | hybrid_emu_gradx32 vs hybrid_emu_reuse_r1 | 30/30 | 43.8 / 43.9 | 0.51 | 56.5 / 59.2 | 0.48 | yes / yes | equivalent |
| effect of OR | hybrid_emu_reuse_r4 vs hybrid_emu_reuse_r1 | 30/30 | 83.5 / 43.9 | 0.87 | 93.8 / 59.2 | 0.75 | no / no | not equivalent (indicates a direction (a > b)) |
| effect of OR | hybrid_emu_reuse_r8 vs hybrid_emu_reuse_r1 | 30/30 | 100.2 / 43.9 | 0.95 | 110.3 / 59.2 | 0.79 | no / no | not equivalent (indicates a direction (a > b)) |
| effect of OR | hybrid_emu_reuse_r16 vs hybrid_emu_reuse_r1 | 30/30 | 126.4 / 43.9 | 0.99 | 145.5 / 59.2 | 0.89 | no / no | not equivalent (indicates a direction (a > b)) |
| effect of OR | hybrid_emu_reuse_r32 vs hybrid_emu_reuse_r1 | 30/30 | 127.5 / 43.9 | 0.97 | 157.3 / 59.2 | 0.88 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | hybrid_emu_headlr_x2 vs hybrid_emu_reuse_r1 | 30/30 | 59.6 / 43.9 | 0.78 | 86.5 / 59.2 | 0.70 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | hybrid_emu_headlr_x4 vs hybrid_emu_reuse_r1 | 30/30 | 102.5 / 43.9 | 0.89 | 114.2 / 59.2 | 0.79 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | hybrid_emu_headlr_x8 vs hybrid_emu_reuse_r1 | 30/30 | 107.8 / 43.9 | 0.98 | 117.3 / 59.2 | 0.85 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | hybrid_emu_headlr_x16 vs hybrid_emu_reuse_r1 | 30/30 | 107.5 / 43.9 | 0.98 | 125.5 / 59.2 | 0.87 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | hybrid_emu_headlr_x32 vs hybrid_emu_reuse_r1 | 30/30 | 113.9 / 43.9 | 0.99 | 127.4 / 59.2 | 0.86 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | hybrid_emu_headlr_x64 vs hybrid_emu_reuse_r1 | 30/30 | 115.9 / 43.9 | 0.96 | 140.2 / 59.2 | 0.85 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | hybrid_emu_headlr_x128 vs hybrid_emu_reuse_r1 | 30/30 | 137.0 / 43.9 | 0.99 | 168.9 / 59.2 | 0.92 | no / no | not equivalent (indicates a direction (a > b)) |
| P4 | hybrid_emu_headlr_x128 vs hybrid_emu_reuse_r32 (best fresh (hybrid_emu_headlr_x128) vs best reuse (hybrid_emu_reuse_r32), chosen by IQM final_perf) | 30/30 | 137.0 / 127.5 | 0.57 | 168.9 / 157.3 | 0.55 | yes / yes | equivalent |
| P6 | hybrid_emu_globallr_x0p5 vs hybrid_emu_reuse_r1 | 0/30 | (not available) | | | | | |
| P6 | hybrid_emu_globallr_x2 vs hybrid_emu_reuse_r1 | 0/30 | (not available) | | | | | |
| P6 | hybrid_emu_globallr_x4 vs hybrid_emu_reuse_r1 | 0/30 | (not available) | | | | | |
| P6 | hybrid_emu_globallr_x8 vs hybrid_emu_reuse_r1 | 0/30 | (not available) | | | | | |

P1 collapsed vs reuse: 4 pairs: 4 equivalent  
P2 fresh x R vs reuse R: 4 pairs: 4 equivalent  
P3 gradient x R vs R = 1: 2 pairs: 2 equivalent

### Performance against k (fresh) and against R (reuse) - P4

| k or R | IQM final: fresh x k | IQM final: reuse R | IQM greedy: fresh x k | IQM greedy: reuse R |
|---|---|---|---|---|
| 1 | 43.9 (35.0-51.1) | 43.9 (35.0-51.1) | 59.2 (46.4-73.6) | 59.2 (46.4-73.6) |
| 2 | 59.6 (55.1-68.0) | - | 86.5 (74.3-101.6) | - |
| 4 | 102.5 (84.4-117.6) | 83.5 (71.6-93.6) | 114.2 (97.0-130.2) | 93.8 (81.8-113.3) |
| 8 | 107.8 (96.1-121.4) | 100.2 (85.5-114.5) | 117.3 (105.8-131.1) | 110.3 (92.7-129.7) |
| 16 | 107.5 (96.1-120.9) | 126.4 (115.1-141.4) | 125.5 (112.1-143.3) | 145.5 (127.4-166.4) |
| 32 | 113.9 (100.8-128.7) | 127.5 (108.9-146.4) | 127.4 (112.8-147.1) | 157.3 (132.7-184.2) |
| 64 | 115.9 (98.5-138.6) | - | 140.2 (124.0-159.0) | - |
| 128 | 137.0 (117.0-164.5) | - | 168.9 (137.5-205.2) | - |

### Shift of the IQM, reuse R minus fresh x R, with 95 % bootstrap CI

| R | shift_final | shift_final_lo | shift_final_hi | shift_greedy | shift_greedy_lo | shift_greedy_hi |
|---|---|---|---|---|---|---|
| 4 | -19.0 | -38.3 | 2.1 | -20.4 | -40.7 | 5.5 |
| 8 | -7.7 | -27.7 | 11.0 | -7.0 | -29.4 | 15.7 |
| 16 | 18.9 | 1.2 | 37.5 | 20.0 | -5.2 | 44.5 |
| 32 | 13.6 | -9.9 | 36.2 | 29.9 | -1.1 | 60.2 |

### Scale trajectory (median over seeds of ||V_eff||_F and mean |Q|, from the 1 000-step diagnostics)

| arm | V@10k | |Q|@10k | V@30k | |Q|@30k | V@60k | |Q|@60k | V@100k | |Q|@100k |
|---|---|---|---|---|---|---|---|---|
| hybrid_emu_reuse_r1 | 1.17 | 0.23 | 3.56 | 9.86 | 4.86 | 19 | 7.18 | 25 |
| hybrid_emu_reuse_r4 | 1.14 | 0.25 | 8.88 | 18.5 | 14.6 | 36.1 | 19.4 | 38.4 |
| hybrid_emu_reuse_r8 | 1.15 | 0.224 | 14.7 | 24.2 | 22.6 | 38.3 | 29.3 | 45.5 |
| hybrid_emu_reuse_r16 | 1.15 | 0.258 | 23.2 | 27.8 | 35.3 | 43.4 | 44.4 | 48 |
| hybrid_emu_reuse_r32 | 1.13 | 0.264 | 35.6 | 30.5 | 52.4 | 43.2 | 62.2 | 48.7 |
| hybrid_emu_collapsed_r4 | 1.14 | 0.25 | 8.88 | 18.5 | 14.6 | 35.8 | 19.4 | 39.7 |
| hybrid_emu_collapsed_r8 | 1.15 | 0.224 | 14.7 | 24.2 | 22.6 | 38.3 | 29.3 | 45.5 |
| hybrid_emu_collapsed_r16 | 1.15 | 0.258 | 23.2 | 27.8 | 35.3 | 43.4 | 44.4 | 48 |
| hybrid_emu_collapsed_r32 | 1.13 | 0.264 | 35.6 | 30.5 | 52.4 | 43.2 | 62.2 | 48.8 |
| hybrid_emu_headlr_x4 | 1.17 | 0.23 | 9.08 | 18 | 14.6 | 33.7 | 19.3 | 37.5 |
| hybrid_emu_headlr_x8 | 1.17 | 0.23 | 14.6 | 23.9 | 23 | 39.2 | 30.6 | 39.5 |
| hybrid_emu_headlr_x16 | 1.17 | 0.23 | 22.7 | 27.6 | 34.6 | 43.3 | 45.3 | 45.2 |
| hybrid_emu_headlr_x32 | 1.17 | 0.23 | 35.9 | 30.1 | 52 | 45.1 | 61.8 | 44.6 |


## Block D: PPO (classical control of the reference study)

Metrics: final_perf, greedy_final and best_ma50 (maximum of the 50-episode moving average, the reference study's metric, which gives 264-382 for its classical control at R = 4-32).

| arm | n | IQM final_perf | IQM greedy_final | IQM best_ma50 (95 % CI) |
|---|---|---|---|---|
| ppo_reuse_clip_r1 | 30 | 61.5 (60.4-62.5) | 455.2 (436.9-468.7) | 72.3 (71.1-73.3) |
| ppo_reuse_clip_r4 | 30 | 244.8 (238.1-251.2) | 500.0 (500.0-500.0) | 267.9 (260.1-275.1) |
| ppo_reuse_clip_r8 | 30 | 315.3 (291.6-338.2) | 500.0 (500.0-500.0) | 356.2 (336.8-369.4) |
| ppo_reuse_clip_r16 | 30 | 217.6 (203.1-243.6) | 449.6 (395.1-482.6) | 341.5 (318.9-366.3) |
| ppo_reuse_clip_r32 | 30 | 170.7 (162.1-180.3) | 222.9 (200.6-269.8) | 328.0 (303.3-353.7) |
| ppo_reuse_noclip_r1 | 30 | 64.3 (63.2-65.7) | 464.3 (447.9-475.9) | 74.7 (73.9-75.9) |
| ppo_reuse_noclip_r4 | 30 | 257.1 (249.2-265.9) | 500.0 (500.0-500.0) | 275.6 (265.9-286.7) |
| ppo_reuse_noclip_r8 | 30 | 269.0 (246.0-296.8) | 498.0 (485.5-500.0) | 320.5 (302.6-338.5) |
| ppo_reuse_noclip_r16 | 30 | 197.0 (182.7-216.7) | 366.4 (288.5-440.3) | 344.0 (316.1-372.4) |
| ppo_reuse_noclip_r32 | 30 | 162.7 (157.3-168.9) | 197.2 (184.3-226.8) | 319.6 (293.9-348.5) |
| ppo_lrx_clip_r4 | 30 | 247.8 (238.9-255.7) | 500.0 (500.0-500.0) | 270.9 (260.8-281.6) |
| ppo_lrx_clip_r8 | 30 | 284.2 (261.7-307.1) | 500.0 (499.3-500.0) | 328.4 (311.2-345.5) |
| ppo_lrx_clip_r16 | 30 | 220.0 (201.7-259.8) | 430.5 (365.6-483.5) | 357.8 (335.7-381.0) |
| ppo_lrx_clip_r32 | 30 | 162.5 (155.7-168.9) | 201.3 (188.8-216.5) | 320.6 (303.5-335.0) |
| ppo_lrx_noclip_r4 | 30 | 256.9 (249.6-265.6) | 500.0 (500.0-500.0) | 277.5 (267.5-288.7) |
| ppo_lrx_noclip_r8 | 30 | 276.6 (253.0-302.5) | 500.0 (496.0-500.0) | 322.0 (308.7-336.6) |
| ppo_lrx_noclip_r16 | 30 | 193.4 (177.7-221.0) | 350.6 (281.6-421.7) | 343.4 (318.2-367.1) |
| ppo_lrx_noclip_r32 | 30 | 159.6 (155.1-165.3) | 189.2 (176.9-212.7) | 312.5 (287.2-342.3) |
| ppo_lrx_modclip_r8 | 30 | 302.0 (280.6-324.7) | 500.0 (499.6-500.0) | 337.5 (322.1-352.5) |
| ppo_lrx_modclip_r32 | 30 | 169.1 (162.2-177.2) | 226.0 (195.1-283.4) | 346.1 (312.1-372.8) |

### Contrasts

| prediction | a vs b | n | IQM final a / b | P final | IQM greedy a / b | P greedy | CI overlap (f/g) | status |
|---|---|---|---|---|---|---|---|---|
| P7 (clip) | ppo_lrx_clip_r4 vs ppo_reuse_clip_r4 | 30/30 | 247.8 / 244.8 | 0.55 | 500.0 / 500.0 | 0.50 | yes / yes | equivalent |
| P7 (no clip) | ppo_lrx_noclip_r4 vs ppo_reuse_noclip_r4 | 30/30 | 256.9 / 257.1 | 0.51 | 500.0 / 500.0 | 0.50 | yes / yes | equivalent |
| P7 (clip) | ppo_lrx_clip_r8 vs ppo_reuse_clip_r8 | 30/30 | 284.2 / 315.3 | 0.34 | 500.0 / 500.0 | 0.45 | yes / yes | inconclusive |
| P7 (no clip) | ppo_lrx_noclip_r8 vs ppo_reuse_noclip_r8 | 30/30 | 276.6 / 269.0 | 0.54 | 500.0 / 498.0 | 0.55 | yes / yes | equivalent |
| P7 (clip) | ppo_lrx_clip_r16 vs ppo_reuse_clip_r16 | 30/30 | 220.0 / 217.6 | 0.50 | 430.5 / 449.6 | 0.50 | yes / yes | equivalent |
| P7 (no clip) | ppo_lrx_noclip_r16 vs ppo_reuse_noclip_r16 | 30/30 | 193.4 / 197.0 | 0.47 | 350.6 / 366.4 | 0.50 | yes / yes | equivalent |
| P7 (clip) | ppo_lrx_clip_r32 vs ppo_reuse_clip_r32 | 30/30 | 162.5 / 170.7 | 0.38 | 201.3 / 222.9 | 0.36 | yes / yes | equivalent |
| P7 (no clip) | ppo_lrx_noclip_r32 vs ppo_reuse_noclip_r32 | 30/30 | 159.6 / 162.7 | 0.45 | 189.2 / 197.2 | 0.43 | yes / yes | equivalent |
| P8 | ppo_lrx_modclip_r8 vs ppo_reuse_clip_r8 | 30/30 | 302.0 / 315.3 | 0.43 | 500.0 / 500.0 | 0.47 | yes / yes | equivalent |
| P8 (usual rule) | ppo_lrx_clip_r8 vs ppo_lrx_modclip_r8 (usual vs modified clipping on the same lr x R network) | 30/30 | 284.2 / 302.0 | 0.41 | 500.0 / 500.0 | 0.48 | yes / yes | equivalent |
| P8 | ppo_lrx_modclip_r32 vs ppo_reuse_clip_r32 | 30/30 | 169.1 / 170.7 | 0.48 | 226.0 / 222.9 | 0.48 | yes / yes | equivalent |
| P8 (usual rule) | ppo_lrx_clip_r32 vs ppo_lrx_modclip_r32 (usual vs modified clipping on the same lr x R network) | 30/30 | 162.5 / 169.1 | 0.39 | 201.3 / 226.0 | 0.40 | yes / yes | equivalent |
| clip vs no clip (reuse) | ppo_reuse_clip_r1 vs ppo_reuse_noclip_r1 | 30/30 | 61.5 / 64.3 | 0.23 | 455.2 / 464.3 | 0.42 | no / yes | not equivalent (indicates a direction (b > a)) |
| clip vs no clip (reuse) | ppo_reuse_clip_r4 vs ppo_reuse_noclip_r4 | 30/30 | 244.8 / 257.1 | 0.32 | 500.0 / 500.0 | 0.52 | yes / yes | inconclusive |
| clip vs no clip (reuse) | ppo_reuse_clip_r8 vs ppo_reuse_noclip_r8 | 30/30 | 315.3 / 269.0 | 0.72 | 500.0 / 498.0 | 0.63 | yes / yes | inconclusive |
| clip vs no clip (reuse) | ppo_reuse_clip_r16 vs ppo_reuse_noclip_r16 | 30/30 | 217.6 / 197.0 | 0.65 | 449.6 / 366.4 | 0.64 | yes / yes | equivalent |
| clip vs no clip (reuse) | ppo_reuse_clip_r32 vs ppo_reuse_noclip_r32 | 30/30 | 170.7 / 162.7 | 0.60 | 222.9 / 197.2 | 0.62 | yes / yes | equivalent |
| effect of OR (clip) | ppo_reuse_clip_r4 vs ppo_reuse_clip_r1 | 30/30 | 244.8 / 61.5 | 1.00 | 500.0 / 455.2 | 0.98 | no / no | not equivalent (strong) |
| effect of OR (no clip) | ppo_reuse_noclip_r4 vs ppo_reuse_noclip_r1 | 30/30 | 257.1 / 64.3 | 1.00 | 500.0 / 464.3 | 0.95 | no / no | not equivalent (indicates a direction (a > b)) |
| effect of OR (clip) | ppo_reuse_clip_r8 vs ppo_reuse_clip_r1 | 30/30 | 315.3 / 61.5 | 1.00 | 500.0 / 455.2 | 0.97 | no / no | not equivalent (strong) |
| effect of OR (no clip) | ppo_reuse_noclip_r8 vs ppo_reuse_noclip_r1 | 30/30 | 269.0 / 64.3 | 1.00 | 498.0 / 464.3 | 0.78 | no / no | not equivalent (indicates a direction (a > b)) |
| effect of OR (clip) | ppo_reuse_clip_r16 vs ppo_reuse_clip_r1 | 30/30 | 217.6 / 61.5 | 1.00 | 449.6 / 455.2 | 0.54 | no / yes | not equivalent (indicates a direction (a > b)) |
| effect of OR (no clip) | ppo_reuse_noclip_r16 vs ppo_reuse_noclip_r1 | 30/30 | 197.0 / 64.3 | 1.00 | 366.4 / 464.3 | 0.34 | no / no | not equivalent (mixed) |
| effect of OR (clip) | ppo_reuse_clip_r32 vs ppo_reuse_clip_r1 | 30/30 | 170.7 / 61.5 | 1.00 | 222.9 / 455.2 | 0.07 | no / no | not equivalent (mixed) |
| effect of OR (no clip) | ppo_reuse_noclip_r32 vs ppo_reuse_noclip_r1 | 30/30 | 162.7 / 64.3 | 1.00 | 197.2 / 464.3 | 0.09 | no / no | not equivalent (mixed) |

P7 no clipping: 4 pairs: 4 equivalent  
P7 with clipping 0.5: 4 pairs: 3 equivalent, 1 inconclusive  
P8 modified rule: 2 pairs: 2 equivalent

### Clipping diagnostic (fraction of the updates in which the clipping fired, mean over seeds; median factor min(1, 0.5/(norm + 1e-6)))

| arm | n | clip_fraction_mean | clip_fraction_min | median_factor | median_norm |
|---|---|---|---|---|---|
| ppo_lrx_clip_r16 | 30 | 1 | 1 | 0.0654 | 7.65 |
| ppo_lrx_clip_r32 | 30 | 1 | 1 | 0.0738 | 6.78 |
| ppo_lrx_clip_r4 | 30 | 1 | 1 | 0.0663 | 7.54 |
| ppo_lrx_clip_r8 | 30 | 1 | 1 | 0.0645 | 7.75 |
| ppo_lrx_modclip_r32 | 30 | 1 | 1 | 0.05 | 10 |
| ppo_lrx_modclip_r8 | 30 | 1 | 1 | 0.058 | 8.62 |
| ppo_lrx_noclip_r16 | 30 | 1 | 1 | 0.0662 | 7.55 |
| ppo_lrx_noclip_r32 | 30 | 1 | 1 | 0.0761 | 6.57 |
| ppo_lrx_noclip_r4 | 30 | 1 | 1 | 0.0659 | 7.58 |
| ppo_lrx_noclip_r8 | 30 | 1 | 1 | 0.0647 | 7.73 |
| ppo_reuse_clip_r1 | 30 | 1 | 1 | 0.0893 | 5.6 |
| ppo_reuse_clip_r16 | 30 | 1 | 1 | 0.0513 | 9.75 |
| ppo_reuse_clip_r32 | 30 | 1 | 1 | 0.0498 | 10 |
| ppo_reuse_clip_r4 | 30 | 1 | 1 | 0.063 | 7.93 |
| ppo_reuse_clip_r8 | 30 | 1 | 1 | 0.0581 | 8.61 |
| ppo_reuse_noclip_r1 | 30 | 1 | 1 | 0.0885 | 5.65 |
| ppo_reuse_noclip_r16 | 30 | 1 | 1 | 0.051 | 9.8 |
| ppo_reuse_noclip_r32 | 30 | 1 | 1 | 0.0514 | 9.73 |
| ppo_reuse_noclip_r4 | 30 | 1 | 1 | 0.0626 | 7.98 |
| ppo_reuse_noclip_r8 | 30 | 1 | 1 | 0.0576 | 8.68 |

### Scale trajectory of the actor / critic V (median over seeds)

| arm | V_actor@20k | V_critic@20k | V_actor@60k | V_critic@60k | V_actor@100k | V_critic@100k |
|---|---|---|---|---|---|---|
| ppo_reuse_clip_r1 | 0.622 | 0.984 | 1.31 | 0.981 | 1.52 | 0.956 |
| ppo_reuse_clip_r4 | 1.99 | 0.906 | 3.56 | 0.983 | 3.88 | 1.01 |
| ppo_reuse_clip_r8 | 3.19 | 1.07 | 4.68 | 1.45 | 4.92 | 2.13 |
| ppo_reuse_clip_r16 | 4.32 | 1.56 | 5.53 | 8.34 | 5.71 | 13.2 |
| ppo_reuse_clip_r32 | 5.24 | 3.29 | 6.39 | 25 | 6.74 | 33.9 |
| ppo_reuse_noclip_r1 | 0.647 | 0.96 | 1.35 | 0.935 | 1.58 | 0.932 |
| ppo_reuse_noclip_r4 | 2.2 | 0.827 | 3.73 | 0.953 | 4 | 1.04 |
| ppo_reuse_noclip_r8 | 3.46 | 1.12 | 4.71 | 2.66 | 4.98 | 3.94 |
| ppo_reuse_noclip_r16 | 4.61 | 2.33 | 5.78 | 11.7 | 5.85 | 17.1 |
| ppo_reuse_noclip_r32 | 5.54 | 4.59 | 6.75 | 29.1 | 7.07 | 37.3 |
| ppo_lrx_clip_r4 | 2 | 0.976 | 3.61 | 1.03 | 3.93 | 1.19 |
| ppo_lrx_clip_r8 | 3.27 | 1.09 | 4.7 | 2.76 | 4.9 | 4.33 |
| ppo_lrx_clip_r16 | 4.42 | 1.62 | 5.56 | 8.62 | 5.78 | 13.9 |
| ppo_lrx_clip_r32 | 5.41 | 3.67 | 6.47 | 26.7 | 6.85 | 35.5 |
| ppo_lrx_noclip_r4 | 2.2 | 0.827 | 3.74 | 0.953 | 4.01 | 1.04 |
| ppo_lrx_noclip_r8 | 3.46 | 1.12 | 4.77 | 2.39 | 5.02 | 3.44 |
| ppo_lrx_noclip_r16 | 4.63 | 2.1 | 5.72 | 11.2 | 5.93 | 16.8 |
| ppo_lrx_noclip_r32 | 5.51 | 4.82 | 6.68 | 29.9 | 6.91 | 37.8 |
| ppo_lrx_modclip_r8 | 3.19 | 1.07 | 4.66 | 1.91 | 4.91 | 2.46 |
| ppo_lrx_modclip_r32 | 5.34 | 3.33 | 6.29 | 24.9 | 6.66 | 33.9 |

## Float64 traces (P1, P5, P7, P8)

first_diff_step = first environment step at which the action taken differs between the two runs (null = never in 100 000 steps); max_param_diff = max over the traced steps up to that point of the maximum difference of the effective parameters (V = sum_r W_r).

| key | first_diff_step | max_param_diff_to_first_diff | max_param_diff_final | n_steps | wall_s |
|---|---|---|---|---|---|
| hybrid_R32_s1 |  | 2.69e-08 | 1.95e-08 | 100001 | 3.23e+03 |
| hybrid_R32_s2 |  | 2.10e-08 | 1.67e-08 | 100001 | 3.56e+03 |
| hybrid_R32_s3 |  | 4.17e-08 | 4.09e-08 | 100001 | 3.58e+03 |
| hybrid_R8_s1 |  | 1.20e-08 | 9.23e-09 | 100001 | 2.4e+03 |
| hybrid_R8_s2 |  | 1.79e-08 | 1.40e-08 | 100001 | 3.04e+03 |
| hybrid_R8_s3 |  | 5.74e-09 | 3.82e-09 | 100001 | 3.42e+03 |
| P8_clip_vs_modclip_R8_s1 |  | 4.52e-12 | 4.52e-12 | 100352 | 188 |
| P8_clip_vs_modclip_R8_s2 |  | 8.79e-11 | 8.79e-11 | 100352 | 182 |
| P8_clip_vs_modclip_R8_s3 |  | 3.30e-11 | 3.30e-11 | 100352 | 211 |
| P8_clip_vs_modclip_R32_s1 | 4.06e+04 | 5.51e-03 | 2.07e+00 | 100352 | 197 |
| P8_clip_vs_modclip_R32_s2 | 3.33e+04 | 2.14e-03 | 2.53e+00 | 100352 | 179 |
| P8_clip_vs_modclip_R32_s3 | 5.14e+04 | 8.84e-03 | 3.88e-01 | 100352 | 177 |
| P7_noclip_R8_s1 |  | 1.54e-08 | 1.54e-08 | 100352 | 176 |
| P7_noclip_R8_s2 |  | 2.43e-11 | 2.43e-11 | 100352 | 169 |
| P7_noclip_R8_s3 |  | 7.16e-08 | 7.16e-08 | 100352 | 182 |
| P7_noclip_R32_s1 | 5.06e+04 | 1.63e-03 | 6.28e-01 | 100352 | 197 |
| P7_noclip_R32_s2 | 2.7e+04 | 5.50e-04 | 1.82e+00 | 100352 | 211 |
| P7_noclip_R32_s3 | 4.35e+04 | 2.47e-04 | 9.80e-01 | 100352 | 212 |
| ref_clip_vs_usualclip_R8_s1 | 1.85e+03 | 8.54e-03 | 4.27e+00 | 100352 | 203 |
| ref_clip_vs_usualclip_R8_s2 | 2.62e+03 | 5.54e-03 | 5.88e+00 | 100352 | 204 |
| ref_clip_vs_usualclip_R8_s3 | 627 | 3.69e-03 | 1.49e+00 | 100352 | 244 |
| ref_clip_vs_usualclip_R32_s1 | 2.18e+03 | 7.67e-02 | 6.12e+01 | 100352 | 248 |
| ref_clip_vs_usualclip_R32_s2 | 573 | 2.23e-02 | 6.57e+01 | 100352 | 253 |
| ref_clip_vs_usualclip_R32_s3 | 649 | 3.71e-02 | 3.10e+01 | 100352 | 240 |
| perturb_noclip_R32_s1 | 2.67e+04 | 1.89e-03 | 1.26e+00 | 100352 | 301 |
| perturb_noclip_R32_s2 | 2.61e+04 | 1.84e-02 | 2.28e+00 | 100352 | 325 |
| perturb_noclip_R32_s3 | 4.09e+04 | 3.89e-03 | 1.10e+00 | 100352 | 321 |
| perturb_noclip_R8_s1 |  | 2.43e-06 | 2.43e-06 | 100352 | 306 |
| perturb_noclip_R8_s2 |  | 1.98e-09 | 1.98e-09 | 100352 | 290 |
| perturb_noclip_R8_s3 |  | 1.28e-06 | 1.28e-06 | 100352 | 305 |
| twin_R8_s1 |  | 2.04e-14 | 1.87e-14 | 100001 | 121 |
| twin_R8_s2 |  | 1.87e-14 | 1.87e-14 | 100001 | 100 |
| twin_R8_s3 |  | 1.03e-13 | 8.62e-14 | 100001 | 101 |
| twin_R32_s1 |  | 5.51e-14 | 3.55e-14 | 100001 | 103 |
| twin_R32_s2 |  | 4.80e-14 | 4.62e-14 | 100001 | 102 |
| twin_R32_s3 |  | 4.62e-14 | 3.55e-14 | 100001 | 104 |
