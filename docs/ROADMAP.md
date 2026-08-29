# Roadmap

The study grows along two axes: **RL algorithm** and **environment**. This file
tracks the grid so a glance shows what is covered and what is next. Update it
when an experiment lands.

## !! STANDING NOTE - environment versions (reproduce vs innovate)

The pinned June-2025 stack (gymnasium 1.1.1, pennylane 0.41.1, autoray 0.7.1,
simplyqrl@b534cc9) is deliberate and MUST stay fixed for the entire
reproduce-and-extend phase (exp01, exp02, exp03). Reason: a comparison is only
valid if both sides share the environment. While we compare against the paper's
PPO results or across our own DQN arms, we share the paper's environment.

Do NOT "upgrade to the newest versions because newer is better" during this
phase. Newer is not the same as compatible - autoray 0.8.0, jax 0.6.0 and
gymnasium 1.0 each REMOVED or CHANGED something the code depends on (that is
where FIX-01..04 come from).

**When to modernise (a later, deliberate step).** When the work shifts from
reproducing the paper to a genuinely new method or environment, and we are no
longer comparing against their numbers: create a fresh branch, move to the modern
stack, and re-verify the baseline from scratch before building on it. That is a
controlled jump with its own before/after evidence, not a drift.

Crucial nuance: modernising fixes the easy dependency bugs (FIX-04, jax, autoray)
but NOT FIX-01. The autoreset NEXT_STEP behaviour is an intentional gymnasium
design change - newer gymnasium entrenches it. Only the code adapting to it fixes
it, so the scientifically interesting bug survives any upgrade.

## !! OPEN - does FrozenLake trigger the modernisation branch?

**Confirm this explicitly; do not inherit it from a document.** The note above
says the moment to branch and modernise is when the work stops comparing against
the paper's numbers. FrozenLake sits between the two cases and an honest reading
has to say so: it is a new environment for the *dissection*, so on the letter of
the rule it counts - but it is one the library itself ships and demonstrates, so
there is a published curve to sanity-check against.

Current recommendation: **keep the pinned stack**, on a ground that has nothing
to do with which paper FrozenLake appears in. exp04's headline (H3) is a
*cross-experiment* comparison - FIX-01's effect on FrozenLake against exp01's
null on CartPole - and that is valid only if both sides share the stack. Changing
gymnasium between them would make the central claim unfalsifiable. Upgrading
would not help anyway: the autoreset behaviour is an intentional gymnasium design
decision that newer versions entrench.

Acrobot and LunarLander have no such tether. They remain the right place for the
branch, and the reasons to defer them are, in order of weight: neither appears in
either paper, so there is nothing to transfer and the question degrades to "does
this agent work here?"; they trigger the commitment cleanly, with none of
FrozenLake's ambiguity; and no experiment in this repo has had its robustness
pass yet, so a third environment before then buys width at the cost of depth.

## !! STANDING NOTE - the framing, and the metric

**Framing.** See docs/LITERATURE.md. Skolik, Jerbi & Dunjko (2022) already ran
PQC + deep Q-learning with architectural ablations on CartPole AND FrozenLake.
The 2025 dissection paper is more systematic about the block decomposition but
moved to PPO and to CartPole alone, without commenting on the change. This work
takes the blocks back to DQN - a closed loop, not an extension - and finds that
the library's off-policy path had not been maintained for the journey. Write the
introduction from that, not from "we re-ran their experiments".

Consequence for exp04: FrozenLake DOES have a published DQN reference, in Skolik
et al., whose sweep is n_layers in {5, 10, 15}. exp04 is a partial replication of
a foundational result plus the things it lacked - a matched classical control,
seeds, and FIX-05.

**Metric.** See docs/STATISTICS.md. `best_ma50` and `greedy_best` are maxima over
training, which are positively biased by an amount that grows with variance. Two
arms with identical true performance and different noise score 567 and 306 under
that protocol. The quantum arms are ~3x noisier than the classical ones in the
paper's own data, so it flatters them systematically. Every number currently in
RESULTS-LOG.md was produced this way. Recompute with
core.stats.final_performance - it needs no retraining - before any of it is
written up as a conclusion.

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

**Environment.** The dissection paper (arXiv:2511.17112) is **CartPole-only**.
The SimplyQRL library chapter additionally demonstrates FrozenLake, but as a
single-seed illustrative run with no controls and no block-level conclusion - so
there is no dissected FrozenLake result to transfer from. Keep the two papers
apart when writing up; it changes what each experiment can claim.

- `cartpole`    - registered, in use (exp01/02/03). The dissection paper's only
  environment, and the anchor for every transfer claim.
- `frozenlake`  - registered, in use (exp04). 4x4 non-slippery, via the adapted
  ids `FrozenLake4x4Scalar-v0` / `FrozenLake4x4OneHot-v0`. NOT in the dissection
  paper. Its purposes are (a) testing whether exp03's DR result generalises
  beyond one environment and (b) providing the first regime in which FIX-01 is
  measurable - the phantom fraction ~ 1/mean_episode_length does not shrink as
  the agent improves. MEASURED: 8.6-9.9% here against < 1% on CartPole at
  convergence. Required FIX-05 before it could run under DQN at all.
- `acrobot`     - candidate, DEFERRED. In neither paper.
- `lunarlander` - candidate, deferred for the same reason.

## Coverage grid

By block x algorithm x environment. "cov" = 3-seed coverage done; "rob" = 8-10
seed robustness pass done (plan B); "-" = not started.

| block | algorithm x env | coverage | robustness (B) |
|---|---|---|---|
| 3 - ansatz/entanglement (exp01) | dqn x cartpole | done (v2, fair control) | pending |
| 1 - Output Reuse (exp02) | dqn x cartpole | scripted | pending |
| 2 - Data Reuploading (exp03) | dqn x cartpole | done (100k, DR transfers) | pending |
| 2 - embedding / DR (exp04) | dqn x frozenlake | specified | pending - AFFORDABLE (1-4 qubits) |

| algorithm x env | cartpole | frozenlake | acrobot | lunarlander |
|---|---|---|---|---|
| **dqn** | exp01/02/03 in progress | exp04 specified | deferred | deferred |
| **ppo** | dissection paper's own setting | chapter Exp 3, single seed | - | - |

exp04 is the first experiment where the plan-B robustness pass is genuinely
affordable: Config A is a one-qubit circuit and Config B four, against CartPole's
eight. Measured throughput ~50 and ~25 steps/s on a CPU runtime.

## Named follow-ups created by the merge

- **exp03b - DR sweep with `ent=False`.** IMPLEMENTED:
  `experiments/exp03b_dqn_cartpole_dr_unentangled.py`, and wired into
  `scripts/run_dqn_suite.py`. exp03's depth axis is confounded with effective
  entanglement depth (CORRECTIONS.md#fix-07); this separates the two by changing
  one boolean. Run it with exp03 or the repo's only clean positive stays
  confounded.
- **Rebuild every "the paper reports X" statement on `data/paper_ppo_summary.csv`**
  rather than on figures. The spreads are large enough to change several
  readings - see docs/PAPER-BASELINES.md.
- **exp02 comparison target.** The paper's OR block is now available at 10 seeds
  (`Quantum_r{4,8,16,32}`, `Classical_r{4,8,16,32}`), so exp02 has a real
  comparator instead of a figure.
- **SU(2) emulator (NEW-05).** Infrastructure and verification, not an
  experimental arm. `core/su2_emulator.py` reproduces
  `build_skolik_qlayer(ent=False)` exactly - see docs/CORRECTIONS.md#new-05
  for why an unentangled, single-generator circuit is a product state and
  therefore classically simulable in O(n_qubits x n_layers) via real Bloch
  vectors instead of 2^n amplitudes. Depends on nothing and cheapens
  everything downstream that trains an unentangled skolik-family arm, so it
  goes BEFORE the Fourier ceiling below. The deliverable at this point is
  `tests/test_su2_equivalence.py` (forward + gradient agreement to ~1e-6,
  against a real `ent=True` circuit as the negative control) - not training
  curves. Registering arms that use it to train faster is Phase B, deferred
  until `core/configs.py` is next safe to edit (see the standing note above
  on not touching the shared registry mid-grid).
- **Additive Fourier ceiling (NEW-06).** A classical control arm, not
  infrastructure - see docs/CORRECTIONS.md#new-06 for the derivation (Schuld,
  Sweke & Meyer 2021, plus NEW-05's separability result) and the guard
  against configurations it does not apply to (`hsiao`, and `dr` whenever
  `n_qubits < n_data` - which includes the 2-qubit Salinas arm, not only the
  1-qubit one). Depends on NEW-05's separability argument, not on NEW-05's
  code. **FrozenLake Config B before CartPole**: the linear-model-on-bits
  degeneracy (`linear_on_bits_ceiling`, pre-registered prediction P2 - depth
  cannot rescue an unentangled circuit there, verified against the real
  circuit in `tests/test_frozenlake_additive_ceiling.py`) makes it the
  sharpest comparison in the study, with the hypothesis class characterised
  analytically rather than by analogy. CartPole's version (exp06) is a
  cheaper, lower-priority control - expect it to be inconclusive, since
  CartPole is largely solvable by near-linear controllers already, and report
  that as "the environment does not discriminate" rather than as a finding.
  Arm registration and the two experiment scripts (exp05, exp06) are Phase B.
- **Trained-agent spectrum (derived from NEW-06's spectrum measurement, not
  implemented).** The per-frequency magnitude table in
  `docs/CORRECTIONS.md#new-06` is measured at RANDOM, UNTRAINED weights - a
  candidate explanation for exp03's depth-curve saturation (L=2 -> L=5), not
  an established one, precisely because training could redistribute energy
  toward high k instead of leaving the initialisation profile in place.
  Extracting the same per-frequency coefficients from a TRAINED agent's
  weights (any completed exp03/exp03b checkpoint that has them - training
  currently does not save weights, see `docs/REUSE.md`'s "Model weights are
  not saved" limitation, so this may need a small addition first) and
  comparing against the initialisation spectrum is what would convert the
  candidate explanation into a measurement.
- **Input-scaling sweep (derived from NEW-06, not implemented).** Repeat the
  spectrum extraction with an explicit input-scaling weight `w` fixed at
  several values (the embedding becomes e.g. `2*atan(w*x)` in place of
  `CartPoleNormalizationTransformer`'s fixed `2*atan(x)`), to test whether the
  dominant harmonic shifts with `w`. Hypothesis: the scale of `arctan(w*x)`
  determines which harmonic dominates. If confirmed, this measures the
  relationship between a trainable input scaling and the effectively-used
  spectrum - one of the reasons this mechanism is treated as important in the
  expressivity literature (`docs/LITERATURE.md`, Schuld/Sweke/Meyer 2021), not
  only an abstract accessibility argument.

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
