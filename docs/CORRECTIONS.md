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
