# qrl-dissection

Cross-algorithm extension of *Dissecting Quantum Reinforcement Learning: A
Systematic Evaluation of Key Components* (Lazaro, Vazquez & Garcia Bringas,
arXiv:2511.17112).

The paper evaluates three pipeline blocks - post-PQC inference, observation
embedding, ansatz design - under one fixed protocol: PPO on CartPole-v1. This
repository asks how those conclusions behave as the **RL algorithm** and the
**environment** change, and finds that the question cannot be answered without
first correcting and re-specifying parts of the experimental apparatus.

**Start here:** [`docs/ROADMAP.md`](docs/ROADMAP.md) for the plan,
[`docs/EXPERIMENT-01.md`](docs/EXPERIMENT-01.md) for the first experiment, and
[`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) for every change made to upstream
code, with evidence.

## Design

This is **not** a fork of SimplyQRL. Upstream is pinned by revision and imported;
corrections are applied at import time by `qrl_dissection.core.compat`, each
guarded so it raises rather than silently no-op if upstream changes. The diff
between "their code" and "our code" is exactly `core/compat.py` (SimplyQRL
patches) plus `dqn/safe.py` (the off-policy autoreset correction).

The package is organised around the two axes the study grows along - algorithm
and environment - not a flat list of experiments:

```
src/qrl_dissection/
    core/          algorithm-agnostic, shared by everything
        compat.py       FIX-02, FIX-03 - guarded patches to SimplyQRL
        capacity.py     NEW-02 - parameter accounting and capacity matching
        configs.py      experiment ARMS and ENVIRONMENTS registries
        analysis.py     loading, summary tables, learning-curve plots
    dqn/           off-policy specifics
        safe.py         FIX-01 + NEW-01 (autoreset probe) + NEW-03 (greedy eval)
        runner.py       NEW-04 - resumable grids, manifests, Drive-safe output
    ppo/           on-policy specifics - scaffolded, see ppo/README.md
experiments/       one script per experiment: expNN_<algo>_<env>_<topic>.py
notebooks/         Colab drivers (thin: they call the package, not vice versa)
scripts/           verify_env.py - FIX-04, run this first
tests/             fast checks that need neither GPU nor a trained model
docs/              roadmap, per-experiment write-ups, corrections, results log
```

`import qrl_dissection` applies only the algorithm-agnostic corrections and
exposes `core`. The `dqn` and `ppo` subpackages are imported explicitly -
`from qrl_dissection.dqn import SafeDQN` - so an on-policy experiment never
pulls in off-policy machinery.

## Corrections

| ID | What | Affects the paper's published results? |
|----|------|----------------------------------------|
| FIX-01 | Autoreset phantom transition poisons the replay buffer | No - PPO has no replay buffer |
| FIX-02 | `OutputScale` never reaches the model | No - DQN branch only; softmax is scale-invariant |
| FIX-03 | `agent_type="classic"` is unresolvable | Their experiment script 2 does not run as published |
| FIX-04 | Dependency pins do not import | The published artefact cannot be installed as specified |

Full evidence and scope in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).

## Quick start

```bash
git clone https://github.com/<you>/qrl-dissection.git
cd qrl-dissection
pip install -e ".[dev]"      # installs SimplyQRL at the pinned revision too

python scripts/verify_env.py         # FIX-04: fails early and clearly
pytest -q                            # ~10 s, no GPU needed

# Parameter accounting only - trains nothing, seconds.
python experiments/exp01_dqn_cartpole_capacity.py --ladder-only

# The classical grid: 3 arms x FIX-01 on/off x 3 seeds.
python experiments/exp01_dqn_cartpole_capacity.py --outdir results/exp01
```

On Colab, open `notebooks/01_dqn_runner.ipynb` from the GitHub tab (do not upload
it). It clones this repository, verifies the environment and writes results to
Drive.

## Metric convention

Report `best_ma50` (best 50-episode moving average), not the tail mean. DQN on
CartPole decays from its peak even when healthy - deadly triad, epsilon pinned at
`end_e` - so a last-N-episodes statistic reports failure on runs that reached
200-400 mid-training. Where the greedy-eval hook (NEW-03) is enabled,
`greedy_best` is primary.

A random policy on CartPole-v1 returns ~22. A run at ~9.5 is not "failing to
learn": it is learning a degenerate constant-action policy - a different
diagnosis.

## Status

Experiment 01 (DQN, CartPole, capacity-matched control) specified and
implemented; results pending. Corrections verified against pinned upstream
revision b534cc9. Nothing has been executed end to end on target hardware yet -
see the checklist at the end of `docs/EXPERIMENT-01.md`.

## Citation

> J. Lazaro, J.-I. Vazquez, P. Garcia Bringas. *Dissecting Quantum Reinforcement
> Learning: A Systematic Evaluation of Key Components.* arXiv:2511.17112, 2025.

> J. Lazaro, J.-I. Vazquez, P. Garcia Bringas. *SimplyQRL: A Modular Benchmarking
> Library for Hybrid Quantum Reinforcement Learning.* Springer, 2025.
> doi:10.1007/978-3-032-08462-0_19
