# Zenodo record: text for review (not published)

**Nothing has been uploaded.** Fields marked `[CONFIRM]` need the author's decision or a value only the author has.

## Resource type
Dataset

## Title
Per-run data of "A Component-Based Evaluation of Parametrized Quantum Circuits: from Proximal Policy Optimization to Deep Q-Learning Across Configurations and Environments" (manifests, training and evaluation records of 6,356 runs)

## Authors
Mas Bosch, Roger — Universitat Oberta de Catalunya (UOC) `[CONFIRM affiliation, ORCID]`

## Description (HTML-free text)
This record contains the per-run data behind the master's thesis *A Component-Based Evaluation of Parametrized Quantum Circuits: from Proximal Policy Optimization to Deep Q-Learning Across Configurations and Environments*. The thesis ports the components of a hybrid quantum-classical agent (output reuse, data reuploading depth, entanglement) from PPO to DQN on CartPole and FrozenLake, compares them with classical controls matched in parameters and in information, and tests whether a classical model can represent what the circuit represents.

Each run has a manifest (specification, agent configuration, code revision, Python version, wall-clock seconds, a probe of the autoreset correction), its training-episode returns, and its greedy evaluations every 10,000 steps (experiment 13 also has per-1,000-step diagnostics). There are 2,608 runs of the real PennyLane circuit and of classical networks (archive `core`) and 3,748 runs of two simulators of the circuit (archive `simulators`). Seeds, training length (100,000 steps), the statistics used and the commands that regenerate the tables and figures are described in the README of the record and in the repository.

The simulator runs are not the real circuit: they are additional results computed with a statevector simulator and a classical emulator of the unentangled circuit, used in the thesis to validate the simulators, to extend some contrasts to 100 seeds and to run one experiment on the emulator. Anyone reusing them should state that they were computed with a simulator. TensorBoard logs, Q-value traces and network weights are not included (they are not needed to regenerate any table or figure).

Code: https://github.com/RogerMas99/qrl-dissection (MIT licence; `src/simplyqrl/` is unmodified third-party code, MIT, J. Lázaro González).

## Files
- `README_DATA.md`
- `qrl-dissection-data-core.zip` (≈ 67 MiB)
- `qrl-dissection-data-simulators.zip` (≈ 82 MiB) `[CONFIRM whether to publish this archive in the same record, in a separate one, or not at all]`

## License
Creative Commons Attribution 4.0 International (CC BY 4.0) (proposed) `[CONFIRM]`

## Keywords
quantum reinforcement learning; parametrized quantum circuits; deep Q-learning; proximal policy optimization; output reuse; data reuploading; CartPole; FrozenLake; ablation study; reproducibility

## Related identifiers
- Is supplemented by (software): https://github.com/RogerMas99/qrl-dissection — the code revision/tag and its Zenodo DOI go here after the release `[CONFIRM tag name, e.g. v1.0-memoria]`
- Is supplement to (thesis): `[CONFIRM thesis URL/DOI when it exists]`
- References: arXiv:2511.17112 (reference study); DOI 10.1007/978-3-032-08462-0_19 (SimplyQRL)

## Version
1.0

## Language
English (documentation); the thesis is in English

## Funding / Communities / Subjects
`[CONFIRM or leave empty]`

## Notes for the reviewer
- Run counts: core 2,608; simulators 3,748; total 6,356 (8 duplicate-writer runs of one folder and 2 probe runs were left out).
- Hardware: not recorded in the manifests; the README says so.
- Checksums of every manifest are in `MANIFEST_INDEX.csv` inside each archive.
