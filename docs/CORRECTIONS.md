# Corrections registry

Every change made to upstream behaviour, with the evidence that motivated it and
the scope of what it does and does not affect.

Two rules govern this file. **Every correction is guarded**: the patch first
asserts that upstream still looks the way it expects, and raises otherwise -
silently applying a no-op is the exact failure mode being corrected here.
**Every correction states its scope**, because most of them do not touch the
paper's published results at all.

| ID | What | Affects the paper's published results? |
|----|------|----------------------------------------|
| [FIX-01](#fix-01) | Autoreset phantom transition poisons the replay buffer | No - PPO has no replay buffer |
| [FIX-02](#fix-02) | `OutputScale` never reaches the model | No - DQN branch only; softmax is scale-invariant |
| [FIX-03](#fix-03) | `agent_type="classic"` is unresolvable | Their experiment script 2 does not run as published |
| [FIX-04](#fix-04) | Dependency pins do not import | The published artefact cannot be installed as specified |
| [FIX-05](#fix-05) | `Discrete` observation spaces unusable under DQN | No - the chapter's FrozenLake run is PPO |
| [FIX-06](#fix-06) | Chapter's transformer signature does not run as printed | Documentation only |
| [FIX-07](#fix-07) | `ent` is a no-op at depth 1 on the `skolik` template | YES - one published contrast is vacuous |
| [FIX-08](#fix-08) | a shorter finished run silently satisfied a longer request | ours, not upstream's |

---

## FIX-01

**Autoreset `NEXT_STEP` phantom transition enters the replay buffer.**

Since gymnasium >= 1.0 the default autoreset mode of `SyncVectorEnv` is
`AutoresetMode.NEXT_STEP`: the `step()` following a termination ignores the
action, resets, and returns `reward=0, terminated=False` with the reset
observation. Upstream `simplyqrl/dqn.py` stores it unconditionally:

```python
real_next_obs = next_obs.copy()
# Save to replay buffer
self.rb.add(self.obs, real_next_obs, actions, rewards, terminations, infos)
```

Observed directly under gymnasium 1.2.1:

```
t=25  r=1.0  term=True   next_obs=[-0.197 -0.814  0.229  1.388]   TERMINAL (correct)
t=26  r=0.0  term=False  next_obs=[ 0.031  0.041  0.011  0.023]   PHANTOM
```

**Mechanism.** The stored phantom teaches `Q(terminal) ~= gamma * V(s0)` - a
value leak through death: dying leads, according to the buffer, to a fresh
initial state carrying all its future value. This is consistent with a stable
loss (targets are finite and mutually consistent, so the symptom is not
numerical instability) and with a collapse that deepens as the agent improves,
since the leak is worth `gamma*V(s0)` and `V(s0)` grows.

**Detection.** The phantom is always the `rb.add` immediately following a step
that reported `terminated or truncated`. `NEW-01` wraps `envs.step` rather than
reading the `done` argument `rb.add` receives - the latter carries only
`terminations`, so it would miss phantoms following truncation, which matter as
soon as the agent reaches the 500-step cap. In CartPole a second, independent
detector is available: reward is always 1.0 while stepping, so `reward == 0`
marks the phantom exactly. `tests/test_probe.py` asserts the two agree.

**Scope.** Off-policy only. PPO produces the same phantom step but has no replay
buffer, so the effect is a bounded point bias in GAE at the episode boundary
rather than cumulative poisoning. The paper's published results are PPO.

**Present since when.** Both upstream locks pin gymnasium >= 1.0 (1.1.1 in June
2025, 1.2.1 at publication), `rb.add` is unconditional in the earliest public
revision of `dqn.py`, and neither `final_observation` nor `autoreset` appears in
any commit of the public history. There is no public state of the code in which
this was correct. History before 2025-06-19 is not public.

**Measured effect, so far.** Across every configuration measured to date the
correction produced no significant change: an oversized MLP learns with or
without it, and the paper's linear arm dies with or without it at
`train_frequency` of both 10 and 1. A delta measured on a dead agent is
uninformative; only configurations that learn can test the correction. This is
one of the motivations for `NEW-02`.

---

## FIX-02

**`OutputScale` is appended to a list that `nn.Sequential` already consumed.**

Upstream `simplyqrl/agents.py`, `HybridAgent.__init__`, `is_qnet` branch:

```python
qnet_layers.append(nn.Linear(prev_dim, act_dim))
self.network = nn.Sequential(*qnet_layers)      # <- list consumed here

# Insert the output scaling layer if desired.
if config.get("use_output_scaling", False):
    init_val = config.get("output_scale_init", 2.0)
    qnet_layers.append(OutputScale(act_dim, init_val))   # <- too late
```

`use_output_scaling=True` therefore has no effect on the model.

**Why it matters.** A PQC readout is bounded in [-1, 1] while CartPole Q-values
reach `1/(1-gamma) = 100`. Without scaling, the linear head must learn very
large weights - precisely what output scaling exists to avoid (Skolik et al.,
2022). This is a live suspect for the underperformance of the hybrid arm under
DQN.

**Scope.** `is_qnet` branch only, i.e. DQN. Under PPO the head feeds a softmax,
which is scale-invariant, so the published results do not depend on it.

---

## FIX-03

**`agent_type="classic"` cannot be resolved.**

`simplyqrl/agents.py :: build_agent` dispatches on `"mlp"` and `"hybrid"` with
no `else` and no `raise` - verified by walking the AST, not by reading. Any
other name returns `None`.

`src/experiments/post-pqc-inference.py:53`, the paper's classical control arm,
passes `agent_type="classic"`. `PPO.__init__` (line 122) forwards it straight to
`build_agent` without normalising, and the following line, `self.agent.to(...)`,
raises `AttributeError: 'NoneType' object has no attribute 'to'`.

The string `"classic"` appears nowhere in `src/simplyqrl/` as an agent type -
the only matches are English comments. Of the five published experiment scripts,
this is the only one using a type other than `"hybrid"`; the other four run.

**Consequence.** Experiment 2 of the paper - the classical control with Output
Reuse - is not executable as distributed. Most likely a `classic` -> `mlp`
rename in the library that was not propagated to the scripts.

**Fix.** Alias `classic`/`classical` -> `mlp`, and raise on unknown names so the
next such mismatch fails loudly.

---

## FIX-04

**The published dependency pins do not import.**

`requirements.txt` in `qrl-dissection` pins `autoray==0.8.0`. That release
removed `autoray.autoray.NumpyMimic`, which PennyLane 0.41.1 requires at import,
so `import pennylane` raises `AttributeError` before any project code runs.
Verified by installing both versions: 0.7.1 has the symbol, 0.8.0 does not.

The same export also drifted `pennylane-lightning` to 0.43.0 against
`pennylane` 0.41.1.

Comparing the two locks:

| | upstream, 2025-06-19 (Fig. 4) | qrl-dissection, 2025-10-31 (published) |
|---|---|---|
| gymnasium | 1.1.1 | 1.2.1 |
| pennylane | 0.41.1 | 0.41.1 |
| pennylane-lightning | **0.41.1** | **0.43.0** |
| autoray | **0.7.1** | **0.8.0** |
| numpy | 2.3.0 | 2.3.4 |

The published `requirements.txt` is a poetry export made at publication time
that picked up newer transitive versions and was never executed as specified.
This repository pins the June set.

**Note on gymnasium.** Both locks are >= 1.0, so `NEXT_STEP` autoreset - and
therefore the phantom transition of FIX-01 - was present in the environment that
produced the published Fig. 4. No pre-1.0 environment exists in the public
history.

**jax.** Absent from both locks, so not part of the authors' environment. Colab
preinstalls it, PennyLane imports it opportunistically, and jax >= 0.6.0 removed
`jax.core.Primitive`. Removing jax is fidelity to the original environment, not
a workaround. `scripts/verify_env.py` checks all of the above and fails with an
actionable message.

---

# Additions

## NEW-01 - autoreset instrumentation

`AutoresetProbe` (`dqn/safe.py`) wraps `envs.step` and `rb.add` on a constructed
`DQN`. The same object both counts phantom transitions and optionally skips
them, so every run reports the poisoning rate it was exposed to - including runs
with the correction disabled. A patch you cannot measure is a patch you cannot
defend.

Useful identity: the phantom rate tracks `1 / E[episode length]`, so it can be
estimated from an episode CSV alone when no probe was attached.

## NEW-02 - capacity-matched classical control

See `docs/EXPERIMENT-01.md`. The hidden width is derived from the reference
hybrid arm's *measured* parameter count, never from an analytic formula, because
the formula depends on circuit-template details.

**Input parity (revision after exp01 v1).** The first version of this arm
inherited the paper's classical input policy - `reuse_indices=[1,2,3]`, cart
position discarded - because it was copied from `config_classical`. That is
correct for `paper_linear` (a faithful replication) but WRONG for a fair control
against the hybrid, which sees all four observations. Cart position is one of the
two termination conditions of CartPole (|x| > 2.4), so the amputated arm may fail
from blindness rather than from being classical, confounding the main
circuit-vs-classical comparison. The fair control now uses the full observation
(`observation="full"`, in_dim = 4 * n_repeats = 16, width 7, ~135 params). The
amputated version is kept behind `observation="paper"` for an explicit ablation
that measures the cost of the amputation itself. exp01 v1 results with the
amputated input are therefore an ablation data point, not the main comparison;
exp01 v2 re-runs the matched arm with full observation.

## NEW-03 - greedy evaluation hook

Periodic evaluation of the greedy policy (epsilon = 0) on a separate
environment, logged to its own CSV. Training return under epsilon-greedy
conflates policy quality with exploration noise, and DQN on CartPole decays from
its peak even when healthy, so a tail statistic misreports runs that reached
200-400 mid-training.

## NEW-04 - resumable run orchestration

Manifest per run, skip-if-done on restart, results written outside the
repository. Upstream flushes its episode CSV once per episode, so an interrupted
run still leaves a usable partial learning curve.

---

## FIX-05

**A `Discrete` observation space cannot be run by upstream's DQN at all.**

Found while specifying exp04 (FrozenLake, the second environment). Affects
FrozenLake, Taxi, CliffWalking and any other discrete-observation task. PPO is
unaffected, which turns out to be the informative part.

### Mechanism

`gym.spaces.Discrete(n).shape` is `()`, not `(1,)`, and its dtype is `int64`.
Upstream derives every tensor shape from that attribute, so three things go wrong
at once.

**1. Action selection raises.** `self.obs` from the single-env `SyncVectorEnv`
has shape `(1,)`. `torch.Tensor(obs)` is therefore 1-D; `nn.Linear` reads it as
one sample with one feature and returns a 1-D `q_values`; and
`torch.argmax(q_values, dim=1)` raises `IndexError` on the first step.

**2. The replay buffer allocates without a feature axis.** In
`simplyqrl/buffers.py`:

```python
obs_shape = observation_space.shape
self.obs_buf = np.zeros((buffer_size, *obs_shape), dtype=observation_space.dtype)
```

For `Discrete(16)` that is `(buffer_size,)`, `int64`. A sampled batch is `[B]`,
one-dimensional and integer, where every downstream consumer expects `[B, 1]`
float.

**3. The consequence of (2) is silent, which is the dangerous part.** The
built-in `FrozenNormalizationTransformer` and `FrozenBasisToAngleTransformer`
branch on `data.dim() == 1` to handle a single observation. Given a `[B]` batch
they take that branch and return **one sample's angles for the entire batch**.
No exception. Separately, `FrozenNormalizationTransformer` does
`transformed = data.clone()` and then writes a float into it; on an `int64`
tensor that write truncates, quantising the encoding angle to whole radians.

Verified, not inferred:

```python
>>> space = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False).observation_space
>>> space.shape, space.dtype
((), dtype('int64'))
>>> np.zeros((1000, *space.shape), dtype=space.dtype).shape
(1000,)                      # CartPole gives (1000, 4)
```

### Why PPO escapes it, and what that tells us

`ppo.py` reshapes explicitly - `next_obs_np.reshape(self.num_envs, -1)` on
collection, `self.obs.reshape(self.batch_size, -1)` on the update - and stores
observations in a float tensor. Immediately above the second of those lines the
CartPole-only version survives as a comment:

```python
#b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
b_obs = self.obs.reshape(self.batch_size, -1)
```

That comment is the clearest available evidence for the reading FIX-01 already
suggested: **the on-policy path was hardened when the library moved beyond
CartPole, and the off-policy path was never re-validated.** FIX-01 and FIX-05 are
two independent instances of the same history, in two different subsystems. The
library chapter's own FrozenLake results (Fig. 6) are PPO and unaffected.

A third fingerprint, for completeness: `dqn.py` hardcodes
`test_obs = torch.randn(8, 4)` in its `verbose` diagnostic branch, which raises
on any observation dimension other than 4. Not patched - it is diagnostic-only
and `verbose=False` is our default - but cited here because three independent
CartPole assumptions in the off-policy path is a pattern, not an accident.

### Our fix

`src/qrl_dissection/core/obs_adapters.py`. We do **not** patch upstream's buffer. We
adapt the environment so it presents its observation in the form the rest of the
stack already assumes: `Box(shape=(1,), dtype=float32)`, the state index
unchanged in value. Three properties earn it the choice:

- it reproduces PPO's effective representation exactly, so a future
  cross-algorithm comparison sees the same thing on both sides;
- `SafeDQN` does `gym.make(self.env_id)` and `run_arm`/`run_grid` take an
  `env_id` string, so registering gym ids means **no runner code changes at all**;
- it is testable in isolation, which a monkeypatch of upstream internals is not.

Placement is `core/`, not `dqn/`. Only DQN currently needs it, but it is an
environment concern and PPO can use it unharmed - and a thing that would be
copied from `dqn/` to `ppo/` belonged in `core/` to begin with.

Registered ids: `FrozenLake4x4Scalar-v0` (adapter) and `FrozenLake4x4OneHot-v0`
(one-hot, for the classical reference arm). `max_episode_steps=100` is pinned
explicitly rather than inherited, because truncation is where the FIX-01 phantom
transition is generated and that boundary should be ours.

### Scope and guard

`tests/test_frozenlake_envs.py` pins the upstream bug as well as the fix: the
`test_upstream_*` cases assert that raw FrozenLake still produces the malformed
shape and dtype. If one of them ever fails, upstream has repaired the problem and
`core/obs_adapters.py` may be removable. `notebooks/00_fix05_verification.ipynb`
reproduces all three symptoms from scratch, standalone.

---

## FIX-06

**The library chapter's transformer constructors do not run as printed.**

Documentation only; no code defect.

The SimplyQRL chapter's Experiment 3 lists the configurations as
`transform_fn = FrozenNormalizationTransformer()` and
`FrozenBasisToAngleTransformer()`, with no arguments. Both shipped classes
require a `grid_size` string (e.g. `"4x4"`) to derive the state count and qubit
count, and raise `TypeError` when called as printed. The repository's own
`examples/frozenlake.py` uses the correct form.

Recorded because a reader reproducing from the chapter text alone hits it
immediately, and because it belongs in any correspondence with the authors.
Note this concerns the *library chapter*, not the dissection paper - which never
runs FrozenLake at all.

---

## FIX-07

**On the `skolik` template the `ent` flag cannot affect the output at
`n_layers_q = 1`, so one published entanglement contrast compares a circuit
against itself.**

Found while merging the paper's companion repository, which ships the raw
TensorBoard logs behind every figure. `Skolik_DR_L1_Entangled` and
`Skolik_DR_L1_Unentangled` carry *identical* returns on all ten seeds. The
innocent explanation turned out to be the correct one, and it is more
interesting than the alternative.

### Mechanism

`build_skolik_qlayer` closes **each** layer with a circular ring of CZ gates,
and the circuit is then measured in the PauliZ basis:

```python
for layer in range(n_layers):
    angle_embedding(...)                 # Rx
    for i, wire in enumerate(range(n_qubits)):
        qml.RY(weights[layer, 0, i], wires=wire)
        qml.RZ(weights[layer, 1, i], wires=wire)
    if ent == True:                      # <-- last thing in the layer
        for i in range(n_qubits):
            qml.CZ(wires=[i, (i + 1) % n_qubits])
return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

CZ is diagonal in the computational basis, and diagonal unitaries commute with
Z. The entangling ring of the **final** layer therefore cannot change any
`<Z_i>`. At depth 1 that is the only ring, and `ent` is a complete no-op.

Verified by construction, not argued (`tests/test_entanglement_noop.py`):

```
n_layers= 1  max |ent - unent| = 0.000e+00   IDENTICAL
n_layers= 2  max |ent - unent| = 3.426e-01   differs
n_layers= 5  max |ent - unent| = 1.412e+00   differs
```

Scope is narrow and deliberately checked: the **Hsiao** template is unaffected
and its ablation is valid at every depth, so the paper's Hsiao entanglement
result stands. The Salinas/UQC rows at one qubit cannot entangle for the obvious
reason, which the authors' own comments already note.

### Consequences

**1. One published cell is vacuous.** The `Skolik_DR_L1` entangled/unentangled
comparison is between identical circuits. Their logged returns coincide on all
ten seeds because they *must*. This is an experimental-design artefact and
explicitly **not** duplicated data - a distinction worth stating plainly, since
identical numbers across ten seeds invite the harsher reading.

**2. Depth L carries only L-1 effective entangling blocks.** "Five layers of
entanglement" is four. Any Skolik depth sweep with `ent=True` therefore varies
entanglement and data reuploading together.

**3. It touches our own exp03.** That sweep runs `hybrid_dr_config` over
L = 1, 2, 5 on the Skolik template with `ent=True`, so its depth axis carries
0, 1 and 4 effective entangling blocks. The monotone result (greedy 15 -> 35 ->
199) is therefore **confounded**: part of the gain may be entanglement coming
online rather than depth. The separation is cheap - rerun the same sweep with
`ent=False` - and is now a named follow-up in docs/ROADMAP.md.

### Not a fix, a caveat

Nothing is patched. The circuit is a faithful implementation of the published
template; the issue is what a sweep over it means. Recorded here because this
registry exists for anything that changes how a published number should be read.

---

## FIX-08

**Our own bug, not upstream's. A finished cell was reused whenever a file with
the same name existed, regardless of whether it answered the same question.**

`run_name` is `arm__fix01{on,off}__s{seed}[__tag]` - readable, and an incomplete
key. It encodes neither the step budget nor `dqn_kwargs` nor the environment. The
skip logic was `if manifest.exists(): return it`, so a 1,500-step smoke cell
satisfied a 100,000-step request, with a different batch size and buffer size,
and reported the smoke numbers as the production result. Nothing was printed.

Reproduced before fixing:

```
run 1 name: oversized_mlp__fix01on__s1 | steps asked: 1500
run 2 name: oversized_mlp__fix01on__s1 | steps asked: 100000
>>> manifest on disk says total_timesteps = 1500 and batch_size = 32
```

This is the worst failure mode available to this project - not a crash, but a
plausible number attached to the wrong experiment, in the one mechanism the whole
suite relies on to be cheap.

**Fix:** `dqn/runner.py :: reuse_or_none` compares the stored spec against the
requested one across `arm`, `seed`, `fix_autoreset`, `total_timesteps`,
`dqn_kwargs`, `tag` and `env_id`. A mismatch raises and names the offending
fields. `--smoke` writes to `<outdir>/_smoke` so the ordinary workflow does not
trip it. Pinned by `tests/test_reuse_guard.py`, including the symmetric case: a
longer finished run does not satisfy a shorter request either, since every metric
here is computed over the whole trace.

Full account of what is and is not reusable: `docs/REUSE.md`.
