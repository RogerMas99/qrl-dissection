# qrl-dissection: per-run data (manifests and CSV)

Data behind the thesis *A Component-Based Evaluation of Parametrized Quantum Circuits: from Proximal Policy Optimization to Deep Q-Learning Across Configurations and Environments* (R. Mas Bosch). Code: <https://github.com/RogerMas99/qrl-dissection> (MIT). Data: CC BY 4.0.

## What is here

| file | contents | runs |
|---|---|---|
| `qrl-dissection-data-core.zip` | real PennyLane circuit (`hybrid`) and classical networks, DQN and PPO, CartPole and FrozenLake | 2 608 |
| `qrl-dissection-data-simulators.zip` | runs of two **simulators** of the circuit: the SU(2) emulator (`su2`) and a batched statevector simulator (`sv`) | 3 748 |

**About the simulator archive.** These runs are not the real circuit. They are the additional emulator-computed results of the thesis (appendix on the validation of the simulators, a 100-seed extension, and the emulator arms of the experiment on the learning rate of the head). Anyone reusing them should say that they were computed with a simulator. The archive also carries the `NOT_FOR_THESIS.md` notices of the result folders.

Not included, on purpose: TensorBoard event files (3.7 GB, not read by any analysis), the `.npy` Q-value traces and `.pt` weights of each run, backup files, and the 8 runs of `exp10_emulator_x100/_quarantine_dup_writers` (duplicate writers, discarded).

## Layout

Each archive mirrors the `results/` folder of the repository:

```
<experiment>/<run_name>.manifest.json        one per run: spec, agent type and configuration, git revision, python, outcome (wall seconds, probe of the FIX-01 correction)
<experiment>/runs/<run_name>.csv             training episodes: episode_n, ep_reward, global_step
<experiment>/runs/<run_name>_eval.csv        greedy evaluation every 10 000 steps: global_step, greedy_return
<experiment>/runs/<run_name>_diag.csv        diagnostics (experiment 13 only)
<experiment>/design_params.json, reused_seeds.json   design records, where they exist
exp13_traces/*.json                          float64 trace tests of experiment 13 (core archive)
MANIFEST_INDEX.csv                           one row per run: experiment, run name, agent type, seed, files, sha256 of the manifest
```

Manifests keep the absolute paths of the machine that produced the run (`/content/drive/MyDrive/...` for Colab, `C:\...\Mi unidad\...` for the local PC). The analysis code resolves the files by name inside `runs/`, so the paths do not need to exist.

Run names: `<arm>__fix01on|off__s<seed>[__<tag>]` (older folders: `<arm>__s<seed>`). `fix01on` = the correction of the autoreset phantom transition (FIX-01) is applied; 25 legacy manifests of exp02/exp03 carry no flag and record it in `outcome.probe.fix01_active` instead (all `true`).

## Experiments and research questions

| folder | what it holds | research question of the thesis |
|---|---|---|
| `exp01_dqn_cartpole_capacity` (+ `_60k_…`) | capacity ladder: 26-parameter reference control, matched classical, hybrid circuit, oversized network, additive Fourier ceiling | RQ2 (circuit against matched controls); RQ4 (ceiling) |
| `exp02_dqn_cartpole_output_reuse` | Output Reuse R = 4, 8, 16, 32: hybrid, reference control, matched control | RQ3 (components); RQ2 |
| `exp03_…`, `exp03b_…` | Data Reuploading depth L = 1, 2, 5, entangled and unentangled | RQ3 |
| `exp04_dqn_frozenlake_embeddings` | FrozenLake: scalar and binary encodings, depth, entanglement, classical controls and ceilings | RQ5 (and RQ4) |
| `exp07_dqn_cartpole_paper_replication` | the 24 configurations of the reference study ported to DQN | RQ1 |
| `exp08_dqn_cartpole_frozenlake_followups` | unentangled Output Reuse, depth crossed with reuse, further controls | RQ3 |
| `exp11_replication_control` | classical control that replicates its input like the hybrid agent | RQ2, RQ3 |
| `exp12_classical_twin` | classical twin sharing the head and Output Reuse of the hybrid agent | RQ3, RQ2 |
| `exp13_lr_equivalence` | Output Reuse against the learning rate of the head (DQN): classical twin (core) and hybrid on the emulator (simulators) | RQ3 |
| `exp13_ppo` | the same equivalence under PPO, with and without gradient clipping | RQ3 |
| `exp13_efg` | classical models that contain Output Reuse (Fourier ceiling with duplicated columns, matched control ×4) and an MLP on bits for FrozenLake | RQ4, RQ2, RQ5 |
| `exp09_dqn_emulator_sweep`, `exp10_emulator_x100` | simulator batteries: validation of the simulators and a 100-seed extension (simulators archive) | validation of the simulators (appendix) |

(This table is a reading aid assembled from the design documents of the repository, `docs/EXPERIMENT-*.md`; the arms that answer each question are defined in the thesis, chapter 3.)

## Regenerating the tables

Unzip both archives in the same folder (`<results>`), install the environment (`environment/requirements-lock.txt` of the repository) and run, from the repository root:

| output | command |
|---|---|
| tables of exp01–exp08 (`per_seed_metrics.csv`, `summary_by_arm.csv`, curves) | `python scripts/build_dossier_v3.py --results-root <results> --out <dir>` |
| validation of the simulators (appendix) | `python experiments/exp09c_battery_report.py --results-root <results> --out <report.md>` |
| exp10, exp11, exp12 | `python experiments/exp10_analysis.py\|exp11_analysis.py\|exp12_analysis.py --results-root <results> --out <dir>` |
| exp13 (blocks A–D) and E/F/G | `python experiments/exp13_analysis.py --results-root <results> --out <dir>` and `exp13_analysis_efg.py` |
| greedy-evaluation episodes per group | `python experiments/audit_eval_episodes.py --results-root <results> --out <dir>` |
| figures of chapter 4 | `python scripts/figures/figs.py <out>` and `scripts/figures/fig_lr.py <out> <exp13 summary dir>` (see `scripts/figures/README.md`) |

Checked on these archives (unzipped into one folder and analysed with the scripts above): the exp01–exp08 tables match the ones delivered for the thesis (888 per-seed rows, 93 arm groups; differences at floating-point round-off in the text of three columns), and the output files of the analyses of exp10, exp11, exp12 and exp13 (27 CSV files, including per-seed tables, curves, summaries, contrasts and diagnostics) are **identical as text** to the ones delivered with each experiment. The validation of the simulators gives the same verdict and figures (mean d = +0.090, CI (+0.008, +0.175), 1 of 63, 0 of 9).

## Hardware and timing

Wall-clock seconds per run are in `outcome.wall_seconds`. The machine is **not** recorded in the manifests. Local runs: AMD Ryzen AI 7 350, 16 threads, 31 GB, CPU only; Colab runs: runtime type not recorded. Times are not comparable between experiments.

## Citation and licence

Cite the thesis and this record. Data: Creative Commons Attribution 4.0 (CC BY 4.0). The code in the repository is MIT; `src/simplyqrl/` is unmodified third-party code (MIT, J. Lázaro González). The curves of the reference study in `data/` of the repository belong to the repository of that study (MIT) and are not part of this record.
