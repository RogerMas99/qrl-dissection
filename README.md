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
        obs_adapters.py FIX-05 - observation adapters for Discrete spaces
        baselines.py    the paper's own 10-seed PPO results, for comparison
    data/               the paper's 360 logged runs: summaries and curves
    src/simplyqrl/      VENDORED upstream b534cc9, unmodified (see VENDORED.md)
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
| FIX-05 | `Discrete` observation spaces unusable under DQN | No - the chapter's FrozenLake run is PPO |
| FIX-06 | Chapter's transformer signature does not run as printed | Documentation only |
| FIX-07 | `ent` is a no-op at depth 1 on the `skolik` template | YES - one published contrast is vacuous |
| FIX-08 | a shorter finished run silently satisfied a longer request | ours, not upstream's |
| FIX-09 | exp02's migrated manifests could not distinguish hybrid from classical | ours, not upstream's |

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

# Second environment (exp04, FrozenLake). Accounting and baseline first.
python experiments/exp04_dqn_frozenlake_embeddings.py --ladder-only
python experiments/exp04_dqn_frozenlake_embeddings.py --stage 1 --smoke
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

On FrozenLake the episodic return is a single bit, so a rolling mean **is** the
success rate. Use a 100-episode window there rather than 50 - on a binary signal
the shorter window is noisy - and note that `summarise_run`'s `mean_ep_len` field
is `rewards.mean()`, which equals episode length on CartPole but equals the
success rate on FrozenLake. A random policy reaches the goal 1.5% of the time.

## Running it

```bash
python scripts/run_dqn_suite.py --plan                 # status of every grid
python scripts/run_dqn_suite.py --pass coverage        # 3 seeds, everything
python scripts/run_dqn_suite.py --pass robustness      # 10 seeds - the real numbers
python scripts/run_dqn_suite.py --only exp03 exp03b    # the pair that matters most
```

Resumable at cell granularity: interrupt whenever, rerun the same command, and
only missing cells are computed - and a cell is only reused if its stored spec
matches what you asked for (FIX-08). `--budget-minutes` bounds a session safely.
What is and is not reusable: `docs/REUSE.md`. To see what a results folder
already contains before adding to it:

```bash
python scripts/inventory_results.py /content/drive/MyDrive/tfm_qrl -v
``` In
Colab, use `notebooks/10_dqn_suite_runner.ipynb` and then
`notebooks/20_dqn_results.ipynb`; see `notebooks/README.md`.

Stack choice, and why upgrading gymnasium does not remove FIX-01:
`docs/ENVIRONMENTS.md`.

## Relationship to the paper's own repository

SimplyQRL is vendored here at `src/simplyqrl/`, byte-identical to upstream
`b534cc9` and to the copy the paper's companion repository ships - so the two
artefacts can be diffed directly. It is never edited; corrections live in
`qrl_dissection/` and are applied at runtime. `tests/test_vendored_integrity.py`
enforces that.

`data/` holds their 360 logged runs (3 blocks, 36 configurations, 10 seeds),
extracted from 140 MB of TensorBoard events into 430 KB of CSV, including full
learning curves. `core/baselines.py` reads them, so every "the paper reports X"
statement can now cite logged returns with standard deviations instead of a
number read off a figure.

A full structural comparison, including the one place where our configuration
diverges from theirs and what that does and does not invalidate, is in
`docs/COMPARISON-WITH-PAPER-REPO.md`.

`docs/LITERATURE.md` places the work in its lineage - Skolik et al. (2022) did
PQC + deep Q-learning with architectural ablations, on CartPole *and* FrozenLake,
three years before the dissection paper moved the same questions to PPO. This
thesis takes them back off-policy, which makes it a closed loop rather than an
extension - and explains why the library's off-policy path needed seven
corrections to get there.

`docs/STATISTICS.md` records what we report and why, following Agarwal et al.
(2021). It also documents a bias in the metric this repository was using: `best_ma50`
and `greedy_best` are maxima over training, which reward a noisier arm for being
noisier. Recompute before writing anything up.

## Status

Experiments 01 and 03 have coverage results (3 seeds); 02 is scripted. Experiment
04 extends the study to a **second environment**, FrozenLake-v1, which required
FIX-05 before it could run under DQN at all - see
`notebooks/00_fix05_verification.ipynb` for the reproduction. Its stage-0
accounting and baselines are measured and logged; stages 1 and 2 are pending.

Note the two papers, because the distinction governs what each experiment can
claim: the **dissection paper** (arXiv:2511.17112) is CartPole-only and PPO-only,
while the **SimplyQRL library chapter** additionally demonstrates FrozenLake as a
single-seed illustrative run with no controls. exp04 is therefore not a transfer
test against a dissected baseline; see `docs/EXPERIMENT-04.md` section 0.

No experiment has had its 8-10 seed robustness pass yet (plan B in
`docs/ROADMAP.md`). exp04 is the first where that is affordable.

## Citation

> J. Lazaro, J.-I. Vazquez, P. Garcia Bringas. *Dissecting Quantum Reinforcement
> Learning: A Systematic Evaluation of Key Components.* arXiv:2511.17112, 2025.

> J. Lazaro, J.-I. Vazquez, P. Garcia Bringas. *SimplyQRL: A Modular Benchmarking
> Library for Hybrid Quantum Reinforcement Learning.* Springer, 2025.
> doi:10.1007/978-3-032-08462-0_19
