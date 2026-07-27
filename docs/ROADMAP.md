# Roadmap

The study grows along two axes: **RL algorithm** and **environment**. This file
tracks the grid so a glance shows what is covered and what is next. Update it
when an experiment lands.

## !! STANDING COMMITMENT - robustness pass (plan "B")

**Every block sweep in this repo is currently a COVERAGE pass at 3 seeds.** This
is deliberate: it lets the dissection advance across all three blocks (OR, DR,
entanglement) without waiting on expensive runs. But 3 seeds cannot separate
"equivalent" from "different" - exp01 already showed the metrics crossing within
noise, and seed dispersion at the top of the range is a factor of two.

Before any block result is written up as a conclusion (not just coverage), it
MUST be re-run at 8-10 seeds, matching the paper's PPO protocol. This applies to
exp01 (entanglement / capacity), exp02 (OR) and exp03 (DR) alike. Do not let the
3-seed numbers become the final numbers by default. Track completion in the
"robustness" column of the coverage grid below.

## Axes

**Algorithm.** The paper is PPO-only. The open question is whether its
block-level conclusions (post-PQC inference, embeddings, ansatz) survive a
change of algorithm.

- `dqn`  - off-policy. Underway. Carries FIX-01 (autoreset) and the probe.
- `ppo`  - on-policy. The paper's own setting; used both to extend the study and
  to cross-check against published numbers. Scaffolded in `src/qrl_dissection/ppo/`.
- `sac` / others - not planned yet.

**Environment.** The paper is CartPole-only. Higher-dimensional observation
spaces are where scalability and the barren-plateau story get interesting.

- `cartpole`    - registered, in use.
- `acrobot`     - candidate. Register in `core/configs.py :: ENVIRONMENTS`.
- `lunarlander` - candidate. Larger observation space; good stress test for DR
  width vs depth.

## Coverage grid

By block x algorithm x environment. "cov" = 3-seed coverage done; "rob" = 8-10
seed robustness pass done (plan B); "-" = not started.

| block (dqn, cartpole) | coverage | robustness (B) |
|---|---|---|
| 3 - ansatz/entanglement (exp01) | done (v2, fair control) | pending |
| 1 - Output Reuse (exp02) | scripted | pending |
| 2 - Data Reuploading (exp03) | scripted | pending |

| algorithm x env | cartpole | acrobot | lunarlander |
|---|---|---|---|
| **dqn** | exp01/02/03 in progress | - | - |
| **ppo** | cross-check vs paper - planned | - | - |

## Adding an experiment

Three coordinated pieces, always:

1. `experiments/expNN_<algo>_<env>_<topic>.py` - declares the grid.
2. `docs/EXPERIMENT-NN.md` - the question and the design.
3. A section in `docs/RESULTS-LOG.md` - filled on completion, then committed.

Naming encodes both axes: `exp02_ppo_acrobot_embeddings`, not `exp02_embeddings`.
When the log has fifteen entries, the axis-encoded names are what let you read
coverage at a glance.

## Where things go

- A new **arm** or **environment** -> `core/configs.py`. Shared by all algorithms.
- Anything an algorithm needs that another does not -> its own subpackage
  (`dqn/`, `ppo/`). If you are about to copy a function from `dqn/` to `ppo/`, it
  belonged in `core/`.
- Run artefacts -> Drive or `results/`, never git. Only summary tables in
  `docs/RESULTS-LOG.md` are committed.
