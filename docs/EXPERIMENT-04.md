# Experiment 04 — Data Reuploading and observation embedding under DQN (FrozenLake-v1)

**Axes:** algorithm `dqn` × environment `frozenlake` × block 2 (embedding / DR)
**Script:** `experiments/exp04_dqn_frozenlake_embeddings.py`
**Notebooks:** `notebooks/00_fix05_verification.ipynb` (gate), `notebooks/04_dqn_frozenlake_runner.ipynb`
**Status:** specified and integrated; stage 0 measured, stages 1-2 not yet run

---

## 0. Two papers, and which one says what

This distinction governs every claim below, so it goes first.

| | **Dissection paper** (arXiv:2511.17112) | **SimplyQRL chapter** (Springer, 2025) |
|---|---|---|
| what it is | systematic evaluation of three pipeline blocks | library announcement with illustrative demos |
| algorithm | PPO only | PPO (Exp 1–3), one hybrid DQN curve (Exp 1, Fig. 4) |
| environment | **CartPole-v1 only** | CartPole (Exp 1–2), **FrozenLake (Exp 3)** |
| protocol | 100k steps, 10 seeds, controls | Exp 3: 500k steps, **single seed**, no controls |
| blocks dissected | OR, DR, ansatz | none — the experiments demonstrate configurability |

**The dissection paper never touches FrozenLake.** The FrozenLake material lives
in the library chapter, where it is an illustrative demonstration that the
configuration dictionary works: two built-in transforms, a DR-depth sweep, one
seed, no classical control, no block-level conclusion drawn.

This corrects the framing an earlier draft of this document used. The standing
question of this thesis — *do the dissection paper's block-level conclusions
survive a change of algorithm?* — **cannot be asked of FrozenLake**, because
there is no dissected FrozenLake result to transfer from. Fig. 6 of the chapter
is a sanity anchor, nothing more, and a single-seed one at that.

## 1. So what is this experiment actually for

Three things, none of which needs a FrozenLake dissection baseline.

**(a) Generality of exp03, the one block that transferred.** exp01 returned a
negative under DQN (no clear circuit advantage at equal budget and information)
and exp02 is still running, but **exp03 found strong monotone DR transfer**:
greedy 15 → 35 → 199 for depths L = 1/2/5 at 100k steps on CartPole. That is the
thesis's one clean positive so far, and it rests on a single environment. A DR
sweep on a second, structurally different environment — sparse reward, discrete
state, short episodes — is what turns "DR transfers to DQN" into "DR transfers to
DQN, and not only on CartPole". If it fails to replicate, that is equally worth
knowing and bounds the exp03 claim rather than inflating it.

**(b) FIX-01 in a regime where it is measurable.** The strongest reason, and it
depends on no paper at all. The phantom fraction FIX-01 removes is approximately
`1 / mean_episode_length`, so on CartPole it *shrinks as the agent improves* —
under 1% once an arm learns well, which is exactly why exp01 found no significant
FIX-01 effect in any live arm despite verifying the mechanism. FrozenLake inverts
this: the optimal path is six steps and episodes end in a hole or at the goal, so
mean length stays near 6–10 for the entire run and the phantom fraction stays
near 10–15% regardless of performance.

The poison is also better aimed. The phantom transition is
`(terminal_state, arbitrary_action, r=0, s'=start, done=False)`. Under dense
reward it is one corrupted sample among many informative ones. On FrozenLake the
reward is a single bit delivered only at the goal, and the phantom attaches to
precisely the transition that carries it — teaching the network that reaching the
goal is followed by bootstrapping from the start with zero reward.

A large FIX-01 effect here against the CartPole null is a **mechanism result**:
the bug's impact scales with episode turnover, predicted quantitatively before
measurement. That converts exp01's negative into a positive with a story.

**(c) A properly controlled version of the chapter's Exp 3.** Config A and
Config B were run once each, single seed, no classical control. Running them with
matched-budget controls and 8–10 seeds is a contribution in itself, and it is
cheap: Config A is a **one-qubit** circuit and Config B four, against CartPole's
eight. The robustness pass plan B has deferred on exp01/02/03 is affordable here
for the first time.

## 2. Open decision: does this trigger the modernisation commitment?

`docs/ROADMAP.md` carries a standing commitment: moving from reproducing the
paper to a genuinely new setting is the moment to branch, update the stack and
re-verify the baseline from scratch. Its trigger was described as "a new method,
or taking it to Acrobot/LunarLander".

FrozenLake sits between the two cases and an honest reading has to say so. It is a
new environment for the *dissection*, so on the letter of the rule it counts; but
it is an environment the library itself ships and demonstrates, so there is a
published curve to sanity-check against.

**Recommendation: keep the pinned stack — for a reason that has nothing to do
with which paper FrozenLake appears in.** Point (b), the headline, is a
*cross-experiment* comparison: FIX-01's effect here against exp01's null on
CartPole. That comparison is valid only if both sides share the stack. Changing
gymnasium between them would make the central claim unfalsifiable, and upgrading
would not help anyway, since the autoreset behaviour is an intentional gymnasium
design decision that newer versions entrench.

Acrobot has no such tether and remains the right place for the branch.

*This is a judgement call and should be confirmed explicitly rather than inherited
from this document.*

## 3. Blocker: FIX-05

Upstream SimplyQRL cannot run FrozenLake under DQN at all. Three symptoms, two of
them silent. See `docs/CORRECTIONS.md#fix-05`, and run
`notebooks/00_fix05_verification.ipynb` — which reproduces the bug from scratch,
verifies the fix and prints the capacity ladder — before anything else.

The fix is an environment adapter, not a patch to upstream internals:
`Box(shape=(1,), float32)`, state index unchanged in value, reproducing PPO's
effective representation exactly.

## 4. Design

Environment: `FrozenLake-v1`, `map_name="4x4"`, `is_slippery=False`,
`max_episode_steps=100`, through `DiscreteToBoxObs` (scalar arms) or `OneHotObs`
(classical reference arm).

### Arms

| arm | type | observation | params | role |
|---|---|---|---|---|
| `frozen_scalar_1q_L{1,5,10,15}` | hybrid, 1 qubit, `ent=False` | scalar → phase | 10 / 18 / 28 / 38 | chapter Config A |
| `frozen_binary_4q_L{1,5}` | hybrid, 4 qubits, `ent=True` | binary basis | 28 / 60 | chapter Config B |
| `frozen_binary_4q_noent_L{1,5}` | hybrid, 4 qubits, `ent=False` | binary basis | 28 / 60 | ablation, ours |
| `frozen_matched_scalar` | classic | scalar | 22 | **fair control**: matched budget and information |
| `frozen_scalar_mlp_large` | classic, oversized | scalar | 4548 | separates encoding from capacity |
| `frozen_onehot_mlp` | classic, oversized | one-hot (16) | 5508 | **liveness guard** |

All arms are registered in `core/configs.py :: ARMS`, per the repo's own
placement rule, so exp04 uses `RunSpec` / `run_grid` exactly like exp01 - **no
changes to `SafeDQN`, `runner.py`, `capacity.py`, `analysis.py` or `compat.py`**.
The matched control is computed from the reference hybrid's measured budget, then
rounded *up*: `match_hidden_width` returns the closest width, which at these tiny
budgets lands under (18 → width 2 → 16 params). A control that loses while
carrying fewer parameters proves nothing, so the conservative direction wins.

A structural note that belongs in the write-up: Config A's head is
`Linear(1, 4)`, so all four Q-values are affine functions of a single expectation
value. That is a limit of the chapter's configuration, not of this set-up.

`ent=True` is invalid at one qubit — the circular CZ becomes `CZ(wires=[0, 0])`.
Config A is therefore *necessarily* unentangled. This matters: comparing Config A
against Config B confounds embedding with entanglement, and the chapter does not
flag it. The `noent` ablation separates them, and it is one flag away.

### Stage 0, measured

Run during integration on the pinned stack, and already in `RESULTS-LOG.md`:

| quantity | value | CartPole reference |
|---|---|---|
| random-policy success rate | 0.0150 | — |
| mean episode length (random) | 7.69 | ~22 at random, 200-500 once learning |
| predicted phantom fraction `1/len` | 0.130 | < 0.01 at convergence |
| **measured** phantom, one-hot arm, 6k steps | 0.086 off / 0.099 on | < 0.01 |

The premise holds: the phantom fraction here is an order of magnitude above
CartPole's, and unlike CartPole it does not shrink as the agent improves.

Throughput, same runtime: ~1600 steps/s classical, ~50 at one qubit, ~25 at four.

### Hyper-parameters

exp01 defaults except where FrozenLake forces a change. Each deviation recorded,
because an unexplained hyper-parameter difference between environments is a
confound a reviewer will find.

| parameter | value | why it differs |
|---|---|---|
| `train_frequency` | 1 | exp01: the upstream default of 10 gradient-starves small networks enough to mask every other effect |
| `buffer_size` | 50 000 | with one bit of reward, a buffer that forgets rare successes starves the only signal there is |
| `learning_starts` | 1 000 | 10 000 is a large fraction of a run at these episode lengths |
| `batch_size`, `gamma`, `lr`, `target_network_frequency` | 128, 0.99, 2.5e-4, 500 | unchanged |

### Metric

FrozenLake's episodic return is a single bit, so `best_ma50` is a noisy estimator
here. The primary training metric is the **success rate**: a 100-episode rolling
mean of `ep_reward`, directly interpretable as the fraction of recent episodes
reaching the goal. `greedy_best` stays the primary reported metric, as in exp01.

### Staging

Compute is spent only after the regime is shown alive — the failure that cost
exp01 two experiments.

- **Stage 0.** Gate notebook, capacity ladder, random-policy baseline,
  phantom-fraction prediction. Minutes.
- **Stage 1.** Three classical arms × FIX-01 on/off × 5 seeds. Cheap, and where
  the headline is decided. **Gate:** if `frozen_onehot_mlp` does not reach a
  success rate near 1.0, stop and repair the regime.
- **Stage 2.** Hybrid sweeps, FIX-01 on, 3 seeds coverage, then 8–10 robustness.

## 5. Hypotheses and readings

**H1 — DR depth (the exp03 replication).** Rising then saturating success rate
with `n_layers_q` replicates exp03's CartPole finding in a second environment and
generalises the thesis's one clean positive. Flat or falling bounds that claim to
CartPole — also a result, and a more interesting one than a second confirmation.
Falling with depth indicates optimisation difficulty, not expressivity, and
should not be called a barren plateau without gradient-norm evidence.

**H2 — embedding.** Binary-basis above scalar-to-phase at equal depth. Note this
is *not* a transfer claim: the chapter's single-seed Fig. 6 is too weak a
reference to transfer from. It is a first controlled measurement. The `noent`
ablation decides whether any gap is embedding or entanglement.

**H3 — FIX-01.** From stage 1. A large effect where the phantom fraction is ~10×
CartPole's, against the CartPole null, is the mechanism result. A null here too
means the bug is real but practically inconsequential — publishable, honest, and
it would substantially reduce how much weight the correction can carry.

**H4 — control, and the caveat that may swallow it.** If `frozen_matched_scalar`
reaches Config A, the circuit buys nothing over an equal-budget classical net on
the same information, and the honest claim concerns the *embedding*, not the
circuit. That would echo exp01.

But the smoke run (1 seed, 5k steps) suggests something more awkward: both scalar
arms sit at success 0.03-0.07 against a random floor of 0.015, while the one-hot
arm reaches 0.86-0.96. A network fed the raw state **index** has almost nothing to
work with - the ordering is row-major and carries no usable metric, so it must fit
a near-arbitrary 16-point function through one input.

If `frozen_matched_scalar` stays at chance, **H4 is unanswerable**: a hybrid
beating a dead control proves nothing, and claiming otherwise would repeat exactly
the error exp01 caught. Config A runs on that same scalar input and may be dead
for the same reason. In that case the informative comparison is Config B against
the one-hot MLP, and it must be reported as a result about the *encoding* rather
than dressed up as circuit-vs-classical.

**FIX-02 bonus.** Output scaling should be unnecessary here: with a 0/1 terminal
reward and γ = 0.99 the optimal action value is bounded by 1, while PauliZ
expectations already span [−1, 1]. On CartPole the same readout must reach
Q ≈ 100. If the hybrid learns here without scaling, the CartPole scaling
requirement is a *range* problem, not a representational one.

## 6. What would falsify the premise

Random-policy success above ~5% means the task is near-trivial and arms will not
separate. `frozen_onehot_mlp` failing means the regime is dead. Either invalidates
everything downstream — which is why stage 0 and the stage-1 gate exist and must
not be skipped.
