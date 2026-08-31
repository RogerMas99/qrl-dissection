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
| [FIX-09](#fix-09) | exp02's migrated manifests could not distinguish hybrid from classical | ours, not upstream's - and it silently defeats the OR block's own comparison |
| [FIX-10](#fix-10) | `DQN(seed=...)` never seeds epsilon-greedy's action sampler | No - reproducibility here was always statistical (N seeds, IQM, CI), never bit-identical |
| [FIX-11](#fix-11) | `SafeDQN`'s weight-saving crashed after training finished, on a relative `--outdir` | Ours, not upstream's - would have silently wasted every cell's compute on the FIRST real run using any experiment script's documented default invocation |

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

## NEW-05 - exact classical (SU(2)) emulator of the unentangled `skolik` circuit

**Not a correction. Infrastructure and verification for exp05/exp06.**

### Why this is possible at all

`build_skolik_qlayer(..., ent=False)` (`src/simplyqrl/qlayers.py`) applies
only single-wire gates - the embedding rotation, `RY`, `RZ` - to a circuit
that starts in the product state `|0>^n`. A product state acted on only by
single-qubit unitaries stays a product state: there is no two-qubit gate for
it to entangle through. Each qubit's state is therefore exactly described,
throughout, by its Bloch vector - a real 3-vector - and `<Z_i>` is that
vector's z-component. `core/su2_emulator.py::SU2SkolikEmulator` implements
this directly: `n_qubits` independent Bloch vectors updated by real 3x3
rotations, O(n_qubits * n_layers), no complex amplitudes, no 2^n scaling.

It deliberately does **not** accept an `ent` argument. `ent=True` inserts CZ
gates, which entangle - an entangled two-qubit state cannot be described by
two independent Bloch vectors, so there is no flag to add here. That absence
is what makes the negative control below meaningful.

### Evidence

`tests/test_su2_equivalence.py` copies weights from a real
`qml.qnn.TorchLayer` (built by `build_skolik_qlayer(ent=False)`) into the
emulator and compares, on random batches:

| check (batch=16, `torch.manual_seed(0)`) | `skolik_8q_cartpole_L5` | `frozen_binary_4q_L1` | `frozen_binary_4q_L5` | `frozen_scalar_1q_L5` |
|---|---|---|---|---|
| forward, max abs diff | 2.38e-7 | 0.0 (exact) | 1.79e-7 | 8.94e-8 |
| gradient wrt weights, max abs diff (`.mean()` reduction over the batch - `.sum()` inflates float32 rounding by the same factor without testing anything different) | 1.12e-8 | 5.82e-11 | 2.98e-8 | 5.96e-8 |
| negative control: same weights, REAL `ent=True` circuit vs the emulator, max abs diff | 1.2974 | 0.9034 | 1.6308 | skipped - `ent=True` is not a valid circuit at 1 qubit (`CZ(wires=[0,0])`, a self-loop PennyLane rejects; see `core/configs.py`'s comment on Config A) |

- **Feature-selection paths**: both the cycling branch (CartPole's 4 features
  onto Skolik's 8 wires, `idx = arange(8) % 4`) and the explicit-`emb_indices`
  branch are covered (a fifth, 3-qubit case not shown above).
- **Negative control** disagrees by 0.90-1.63 depending on configuration -
  four to five orders of magnitude above the positive-case forward tolerance,
  which is what gives the positive checks their power.
- **Throughput**, measured on the machine that ran this: skolik_8q_L5,
  batch 32, PennyLane `lightning.qubit` 188 ms/call vs the emulator 0.53
  ms/call - about 353x. Reported, not asserted against a threshold, and it
  moves run to run with system load (an earlier measurement this session read
  445x under a lighter load); PQC throughput varies several-fold across this
  project's own machines regardless (see `docs/REUSE.md`), so a number from
  one run is a data point, not a claim.

### Claim discipline - state this exactly, nowhere stronger

Proven: agreement of forward output and gradients, per call, to numerical
tolerance. **Not** proven or claimed: bitwise-identical training curves over
a full run. Epsilon-greedy argmax ties and replay-buffer sampling amplify
last-bit differences over tens of thousands of steps, so two
functionally-identical Q-networks can still diverge in which actions get
sampled and which transitions get stored. The training-level claim this
module supports is equivalence **in distribution over seeds**, never
per-trajectory identity.

### Scope

Verification and infrastructure, primarily. Numbers in the main results
tables (exp05, exp06) come from the real PennyLane path; anything the
emulator itself produces is labelled as such wherever it is reported, never
left to be inferred from context. **UPDATE 2026-08-30 - arms registered.**
See "Arm registration (Phase B)" below, under NEW-06, for both NEW-05's and
NEW-06's arms together (they were wired into `core/configs.py` and
`core/compat.py` in the same change).

## NEW-06 - the additive Fourier ceiling

**Not a correction. A classical control arm, sharpening what exp05/exp06 can
attribute to the circuit versus to its embedding's accessible frequency
spectrum. Nothing here criticises the reference paper, SimplyQRL or Hsiao et
al. - these are additional controls, not an audit.**

### The argument

Schuld, Sweke & Meyer (2021) - see `docs/LITERATURE.md` - establish that a
data-reuploading circuit is, as a function of one input feature, a truncated
Fourier series: the ENCODING gate's generator fixes which frequencies are
reachable, the variational layers only fix the coefficients. Combined with
NEW-05's separability result: on a `skolik`-style circuit whose embedding
puts one feature per wire, at reuploading depth L, the UNENTANGLED circuit's
accessible function class per wire is exactly

    f(z) = w_0 + sum_{k=1}^{L} [ a_k cos(k z) + b_k sin(k z) ]

`core/fourier_ceiling.py::FourierAdditiveCeiling` builds that basis directly
(a linear head over `{cos(kz), sin(kz)}_{k=1..L}`, per wire, concatenated -
the head's own bias supplies the k=0 term). Any unentangled, one-feature-per-
wire circuit's reachable class is a SUBSET of what this head represents (same
frequencies, and the head's coefficients are free real numbers where the
circuit's are constrained by unitarity), so if a circuit ever beats this
ceiling on held-out performance, the honest reading is inductive bias or an
optimisation effect - **not** expressivity. State it that way; see
`docs/LITERATURE.md` for the full citation and the barren-plateau /
NISQ-robustness motivation for unentangled circuits, argued fairly rather
than dismissed.

### Guarded, not assumed applicable

`check_additive_embedding(circ_type, n_qubits, n_data)` raises rather than
silently building a ceiling that does not bound anything:

- `circ_type="hsiao"` (`emb_type="multi"`): every wire receives all three
  selected features via a non-commuting Z-Y-Z composition
  (`embeddings.multiple_rotation_embedding`) - not additive across features,
  at any qubit count.
- `circ_type="dr"` **whenever `n_qubits < n_data`** - verified against
  `build_dr_qlayer`'s own branch condition, not assumed from the brief's
  "1-qubit" shorthand: this is the SAME `multiple_rotation_embedding` path,
  and it fires for BOTH `paper_salinas_1q_*` and `paper_salinas_2q_*`
  (`n_data` is CartPole's raw 4, computed before any index selection, so
  `n_qubits=2 < 4` also qualifies). The guard checks the structural
  condition, not a name list, so this does not need updating if a third
  Salinas qubit count is ever registered.

### The FrozenLake Config B degeneracy - verified against the real circuit,
### not just algebraically

`FrozenBasisToAngleTransformer` maps each bit to `{0, pi}`. On that
two-point domain `sin(k*{0,pi}) = 0` for every integer k - the sine features
carry no information at any L - and `cos(k*{0,pi})` is `1 - 2b` for odd k,
the constant `1` for even k. The entire `2L`-dimensional feature set per wire
collapses to ONE informative direction, independent of L: pre-registered
prediction **P2**. `core/fourier_ceiling.py::linear_on_bits_ceiling` builds
that degenerate case directly - 5 parameters per action (4 bits + bias),
never `2*L*n_qubits+1` carrying mostly-zero, mostly-redundant columns.

`tests/test_frozenlake_additive_ceiling.py` checks this against the REAL
circuit, not against the ceiling module (which is that hypothesis class by
construction and would prove nothing about `build_skolik_qlayer` itself):
enumerates FrozenLake's 16 states, evaluates the 4-qubit unentangled circuit
on all of them, and fits `Z = B.W + c` by least squares. Measured:

| check | result |
|---|---|
| each `<Z_i>` takes exactly two values, keyed by `b_i` | holds at L = 1, 2, 5 |
| affine-in-bits residual, L=1 | 1.78e-15 |
| affine-in-bits residual, L=2 | 1.89e-15 |
| affine-in-bits residual, L=5 | 8.88e-16 |
| residual growth with L (**P2**) | none - all three at lstsq machine-precision level, no trend |
| negative control, `ent=True`, L=2 | 0.4149 |
| negative control, `ent=True`, L=5 | 0.7978 |

(L=1's `ent=True` negative control is not meaningful: FIX-07 makes `ent` a
no-op at depth 1 on this template, so there is no entangled circuit to
contrast against there - both L=2 and L=5 above sit thirteen to fourteen
orders of magnitude above the positive cases.)

**P1 and P2 both hold, against the real circuit, before exp05 spends any
compute on them.**

### The general (continuous-domain) claim, verified separately

The FrozenLake check above is a two-point special case. The underlying claim
is stronger and domain-independent (Schuld, Sweke & Meyer 2021,
`docs/LITERATURE.md`): for FIXED weights, `<Z_i>(x)` is EXACTLY a degree-L
trigonometric polynomial in the embedded feature x, for any x, not only
`x in {0, pi}`. `tests/test_fourier_ceiling_spectrum.py` checks this directly
- sample x densely over one full RX period, fit `{1, cos(kx), sin(kx) : k =
1..L}` by least squares, and require the residual to (a) be near-zero for the
full basis and (b) become clearly nonzero once the TOP frequency `k=L` is
dropped.

**Revised after an initial version of this test gave a marginal ~1e-3 on the
`ent=True` negative control** (fixed weight seed, all 4 wires driven by the
SAME shared x). Diagnosis: with every wire seeing an identical scalar, the
whole system has only one true degree of freedom, so even the ENTANGLED
circuit collapses back to *some* Fourier series in x - a higher-degree one,
but the fit could still partially absorb it (residual dropped from 0.19 to
0.0053 just by raising the fit degree from 5 to 10, confirming it was "too
few independent inputs to probe entanglement", not "not a Fourier series").
Fixed by sweeping ONE wire's feature while drawing the OTHER wires'
features INDEPENDENTLY AT RANDOM per sample - the design that actually lets
entanglement show up as a genuine failure of the univariate fit - and by
requiring the assertion to hold across the WORST of `N_DRAWS=8` fixed weight
seeds (mean over the 4 swept-qubit measurements per draw), not one draw:
a wider 30-draw calibration found individual (draw, qubit) pairs with a
near-zero top-frequency coefficient purely by chance (as low as 3.3e-5 on the
truncated-basis check), which a single-seed test could have hit by luck in
either direction.

| check (4 qubits, `N_SAMPLES=300` per sweep, `N_DRAWS=8` fixed seeds, relative residual = RMS(fit residual) / std(signal), worst draw reported) | result |
|---|---|
| full basis `k=1..L`, `ent=False` (positive) | 4.8e-8 to 1.29e-7 across L in {1,2,5} |
| truncated basis `k=1..L-1`, `ent=False` (negative control 1: proves frequency L is really used) | 0.0211 (L=5) to 1.0000 (L=1) |
| full basis `k=1..L`, `ent=True`, independent-others sampling (negative control 2) | 0.2908 (L=2) to 0.9771 (L=5) - three orders of magnitude firmer than the pre-fix ~1e-3 |
| unentangled circuit, independent-others sampling, sanity check | 4.8e-8 to 1.29e-7 - identical to the shared-x case, confirming the fix isolates entanglement rather than just making every test harder |

This is the test that matters most for exp06 (CartPole): FrozenLake's domain
is exactly two points, so its degeneracy is a special case worth knowing but
not evidence the general basis is sized correctly. This one is.

### Measured mechanism: the spectrum is concentrated at low-to-mid frequency, not flat

Motivated by a pattern in the table above: the truncated-basis negative
control weakens sharply with depth (1.00 at L=1, down to 0.021-0.072 at
L=5) even though it always stays well clear of the floor. That is consistent
with the circuit reaching frequency L (the positive check already proves it
does) while spending LESS amplitude on that top frequency as L grows -
checked directly by extracting the fitted Fourier coefficients from the
same full-basis fit `test_fourier_ceiling_spectrum.py` already performs
(`sqrt(a_k^2 + b_k^2)` per frequency, `L=5`, mean and spread over the 8
weight draws x 4 swept qubits = 32 values per `k`):

| k (frequency) | mean magnitude | std | min | max |
|---|---|---|---|---|
| 1 | 0.3703 | 0.1941 | 0.0313 | 0.7921 |
| 2 | 0.4427 | 0.1886 | 0.0906 | 0.7971 |
| 3 | 0.2788 | 0.1270 | 0.0903 | 0.6438 |
| 4 | 0.1364 | 0.1146 | 0.0093 | 0.4758 |
| 5 | 0.0305 | 0.0323 | 0.0005 | 0.1307 |

**Read min-max before mean, not instead of it - the two stated patterns are
not equally solid.**

- **k=4 -> k=5 decay: robust.** The MAXIMUM of k=5 across all 32 draws
  (0.1307) sits BELOW the MEAN of k=4 (0.1364) - the luckiest k=5 draw still
  does not reach the average k=4 draw. That is a real separation, not an
  artefact of the mean hiding spread.
- **k=2 above k=1: NOT established.** Their ranges overlap almost entirely
  (k=1: 0.0313-0.7921; k=2: 0.0906-0.7971) - the mean ordering (0.4427 vs
  0.3703) is consistent with a peak at k=2, but with 8 draws this sits inside
  the overlap and must not be stated as a finding. "Amplitude peaks at
  low-to-mid frequency and tapers toward the top one" is defensible from this
  table; "the peak is at k=2 specifically" is not.

This is a **measured mechanism**, not a hypothesis, for the tapering part:
with random weights, `n_layers` reuploads of a fixed-generator rotation make
frequency L *accessible* (the positive check), but do not allocate it equal
*weight* - amplitude concentrates at low-to-mid frequency and tapers toward
the newest one, and the k=4/k=5 step is the part of that claim with a clean
statistical margin behind it.

**The scope-limiting caveat, and the one that matters most: these are RANDOM,
UNTRAINED weights.** The table describes the spectrum the architecture
produces AT INITIALISATION, not the spectrum a trained agent actually uses -
training could redistribute energy toward high k rather than leave the
initialisation profile in place. **The defensible claim is that the
accessible spectrum is populated very unevenly at initialisation, and that
this is a CANDIDATE explanation for the depth-curve saturation (L=2 -> L=5 in
exp03), not an established one.** Confirming it needs the two follow-ups now
recorded in `docs/ROADMAP.md`'s Named follow-ups (extracting the spectrum
from TRAINED agents, and sweeping an input-scaling weight to test whether the
dominant harmonic moves) - neither implemented yet. Belongs in the framing
chapter alongside the Schuld/Sweke/Meyer citation (`docs/LITERATURE.md`), with
this same qualification carried over, not left behind - a reader of the
memoria's chapter 2 should not walk away with a stronger claim than the data
supports.

### Open sizing question - flagged, not decided here

NEW-02's matching recipe (`core/capacity.py::match_hidden_width`) solves for
a hidden width that spends AT LEAST a reference arm's measured budget,
because that control has a free capacity knob. This ceiling does not: its
parameter count (`2*n_qubits*n_layers*n_actions` in general, `5*n_actions` on
the FrozenLake Config B degeneracy) is not a design choice, it is the exact
size of the circuit's accessible hypothesis class - inflating it would stop
the comparison from bounding THIS circuit's expressivity. Confirm before
exp05/exp06 report anything: the ceiling's parameter count is reported
alongside the hybrid arm's for context, not matched to it.

### Arm registration (Phase B) - UPDATE 2026-08-30

Both NEW-05 and NEW-06 are now wired into the same registry every other arm
uses, with **six new arms, zero existing arms touched** (`core/configs.py`'s
`ARMS` dict is append-only here, matching the standing rule against editing
an already-run arm's spec - see the NEW-02/FIX-09 discussion above for why).

**The dispatch mechanism.** `core/compat.py::_patch_new_agent_types` follows
FIX-03's own pattern exactly: it wraps whatever `simplyqrl.agents.build_agent`
already is (after FIX-03's alias patch) and intercepts three NEW agent_type
strings this project defines - `"su2"`, `"fourier_additive"`,
`"linear_on_bits"` - constructing `SU2HybridAgent` / `FourierAdditiveCeiling`
/ `linear_on_bits_ceiling` directly. Anything else is passed straight through
UNCHANGED to the function `build_agent` already was, so FIX-03's own code is
never touched and no existing agent_type string's behaviour can change. This
is a structural guarantee, not just an intention: `run_arm`'s reuse guard
(`dqn/runner.py::reuse_or_none`) compares stored specs to the requested one
and returns a cached manifest WITHOUT ever calling `build_arm_config` or
`build_agent` on a reuse hit - so this patch cannot affect whether an
already-completed cell reuses, only what a NEW cell with a NEW agent_type
builds. Empirically checked, not just argued: replaying all 83
already-completed exp04/exp04b manifests' own stored specs through
`run_arm(force=False)` after this change landed - 83/83 reuse hits, zero
mismatches, zero unexpected writes (a run that would have retrained).

**`SU2HybridAgent`** (`core/su2_emulator.py`) mirrors `HybridAgent`'s
`is_qnet=True` construction line for line - PQC layer, `OutputReuse` if
`reuse_repetitions > 1`, `pi_arch` linear+activation stack, final `Linear`,
`OutputScale` if requested - reusing upstream's own `OutputReuse`/
`OutputScale` rather than reimplementing them, with `SU2SkolikEmulator` in
place of the real `TorchLayer`. Guarded, not permissive: raises on
`circ_type != "skolik"` or `ent=True` rather than silently emulating a
circuit it cannot represent (see NEW-05 above for why `ent=True` has no
emulator path at all).

**The six arms**, and measured trainable parameter counts (`--ladder-only`,
this machine):

| arm | agent_type | params | mirrors / bounds |
|---|---|---|---|
| `su2_cartpole_L5` | su2 | 126 | `hybrid_fig4` (8q, L5) with `ent` forced False - **exactly matches** `hybrid_fig4`'s own 126, since it is the identical architecture |
| `su2_frozen_scalar_1q_L5` | su2 | 18 | `frozen_scalar_1q_L5` (1q, L5, already `ent=False`) - matches its 18 exactly |
| `su2_frozen_binary_4q_L1` | su2 | 28 | `frozen_binary_4q_noent_L1` (4q, L1, `ent=False`) - matches its 28 exactly |
| `su2_frozen_binary_4q_L5` | su2 | 60 | `frozen_binary_4q_noent_L5` (4q, L5, `ent=False`) - matches its 60 exactly |
| `cartpole_fourier_ceiling_L5` | fourier_additive | 162 | `hybrid_fig4` (126 params) - NOT matched, see "Open sizing question" above; `2*8*5*2+2` |
| `frozen_binary_4q_fourier_ceiling` | linear_on_bits | 20 | every `frozen_binary_4q_{noent_,}L{1,5}` - **no L suffix on purpose**: `5*n_actions=20`, provably independent of L (the Config B degeneracy above) |

The exact-match column for the four `su2_*` arms is not a coincidence to note
in passing - it is `tests/test_new_agent_types.py`'s own regression pin
(`test_su2_arm_matches_reference_hybrid_param_count`), because a future
change to either construction path that broke this equality would mean the
"same architecture, no quantum simulator" claim had quietly stopped being
true.

`frozen_binary_4q_fourier_ceiling`'s config carries FrozenLake's
`transform_fn` marker (`"frozen_binary"`, resolved to
`FrozenBasisToAngleTransformer` by `core/configs.py::_resolve_transform`,
same as every other `frozen_*` arm) even though `linear_on_bits_ceiling`'s
own docstring says it wants raw bits `{0,1}`, not the transformer's
`{0, pi}`-scaled output: `linear_on_bits_ceiling` now takes an optional
`transform_fn` (new parameter, default `None` - the bare-`nn.Linear` case
used by nothing else stays unchanged) and wraps the head in
`_TransformedLinear` when given one, so the arm can consume FrozenLake's raw
scalar state like every other arm rather than requiring a caller to
pre-extract bits. The `{0,pi}` vs `{0,1}` rescaling does not change the
hypothesis class - an affine model absorbs a fixed input scale into its
weights - so reusing the same transformer the real circuit uses is exact, not
an approximation.

Everything above is verified in `tests/test_new_agent_types.py` (18 tests:
existing-dispatch non-regression, guard behaviour for all three new types,
the six arms' resolution, the parameter-count equalities in the table) plus
the standalone 83/83 reuse-guard replay described above.

### Scope

Additions only - a new classical control, not a correction to upstream or to
any already-published number in this repo.

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

## FIX-09

**exp02's manifests, once migrated, could not be told apart by arm - both the
hybrid and the classical control were labelled `hybrid_fig4` - and the analysis
notebook averaged them together without anyone asking it to.**

Found while auditing the Drive results tree on a new machine, before launching
exp04b - not by inspecting the numbers first, but by asking why a directory
that mixes two agent types reported only one arm name.

### Mechanism

`experiments/exp02_dqn_cartpole_output_reuse.py::run_one` writes its manifest by
hand:

```python
m = {"name": name, "agent_type": agent_type, "seed": seed,
     "fix_autoreset": fix_autoreset, "outcome": out.__dict__,
     "config": {k: str(v) for k, v in cfg.items()},
     "total_timesteps": steps, "dqn_kwargs": kw}
```

No `spec` wrapper, no `arm` field - unlike `dqn/runner.py::run_arm`, this script
never went through the shared pipeline at all. That is tolerable on its own:
`docs/REUSE.md`'s migration step exists exactly to back-fill manifests like
this. The failure came from how it was migrated. `docs/REUSE.md`'s own example
command is

```bash
python scripts/migrate_manifests.py /content/drive/MyDrive/tfm_qrl/exp03 \
    --arm hybrid_fig4 --dqn-kwargs '{"batch_size":128,"buffer_size":10000,"train_frequency":10}'
```

written for **exp03**, whose every cell is a Skolik-depth sweep on
`hybrid_fig4` - the override is harmless there because there is only one true
arm in the directory. The same override, run against **exp02**, is wrong:
exp02's directory holds both `hybrid_OR{R}` cells and the eight `classical_OR{R}`
control cells added by the amendment in `docs/RESULTS-LOG.md#experiment-02`
(`agent_type="classic"`, a completely different network). `migrate_manifests.py`
does not know that difference; it takes the override on trust and writes it into
every manifest in the directory. All 80 exp02 manifests ended up with
`spec.arm = "hybrid_fig4"`, independent of their real `agent_type` or `config`.

### Detection and reproduction

```
$ python -c "... load every exp02 manifest, print distinct spec.arm values ..."
distinct 'arm' field values: ['hybrid_fig4']          # should be two values, not one
```

`agent_type` and `config` inside the same manifests were never touched by the
migration and remained correct throughout - `classical_OR16__s1.manifest.json`
carries `"agent_type": "classic"`, `"config": {"reuse_indices": "[1, 2, 3]",
"n_repeats": "16", "net_arch": "[]"}` - so the corruption is confined to
`spec.arm`, the one field the migration script writes and the one field
`notebooks/20_dqn_results.ipynb` reads for grouping.

### Consequence: the notebook pools two arms into one number

`notebooks/20_dqn_results.ipynb`'s exp02 cell does not, in fact, use the
corrupted `spec.arm` directly - it derives `R` from the run name by regex and
groups by `R` alone, which is a second, independent way to arrive at the same
failure: pooling every `classical_OR{R}` and `hybrid_OR{R}` cell of the same `R`
into one mean.

```
notebook's groupby("R") - what it reports today:
    R=4   -> 108.8   (mean of hybrid 203.0 and classical 14.6)
    R=8   -> 127.9   (mean of hybrid 226.0 and classical 29.8)
    R=16  -> 162.5   (mean of hybrid 216.8 and classical 108.1)
    R=32  -> 162.4   (mean of hybrid 306.1 and classical 18.8)

split correctly, by real arm:
                 greedy   n
is_classical R
False        4    203.0  10
             8    226.0  10
             16   216.8  10
             32   306.1  10
True         4     14.6  10
             8     29.8  10
             16   108.1  10
             32    18.8  10
```

Either bug alone would have been enough to make exp02's headline figure - the
one comparison the whole experiment exists to run - report a single meaningless
average instead of the hybrid-vs-classical contrast. Having both meant the
corrupted `spec.arm` was never even exercised by the code that would have made
the mistake visible.

### A second, independent finding surfaced while diagnosing this: the classical control's failure is bimodal, not merely undersized

Fixing the metadata revealed the split numbers above, and the natural next
question - is `classical_OR{R}` merely too small, or dead - was answered by
reading the ten per-seed learning curves for `classical_OR16` (the arm's best
performing point) rather than trusting the mean:

```
classical_OR16, early (first 10% of episodes) vs late (last 10%) mean return:
  s1   early=17.6   late= 10.1   peak= 72.0   n_eps=8091
  s2   early=23.0   late=116.6   peak=500.0   n_eps=1167   <- solves CartPole
  s3   early=20.9   late=  9.7   peak= 88.0   n_eps=8133
  s4   early=20.0   late=  9.6   peak= 75.0   n_eps=8178
  s5   early=28.3   late=  9.6   peak=130.0   n_eps=7651
  s6   early=17.9   late=  9.6   peak= 69.0   n_eps=8296
  s7   early=19.7   late=  9.6   peak= 97.0   n_eps=8222
  s8   early=19.5   late=  9.6   peak= 67.0   n_eps=8213
  s9   early=17.8   late=  9.6   peak= 58.0   n_eps=8300
  s10  early=18.8   late=  9.6   peak= 60.0   n_eps=8267
```

Nine of ten seeds show the identical signature already characterised for
`paper_linear` in exp01 section 2: a transient peak (58-130) followed by decay
to the ~9.6 degenerate constant-action floor - not "learns slowly", but "learns,
then collapses", the deadly-triad pattern of a linear Q-function bootstrapping
against a tiny parameter budget. One seed (s2) is qualitatively different: it
runs only 1,167 episodes in the same 100k steps (i.e. episodes ~7x longer on
average) and its late-training mean is 116.6 with a peak of 500 - it solves
CartPole outright. A mean over the ten (108.1) is therefore not "the classical
arm performs moderately" - it is nine dead runs and one solved run, averaged,
which reads as moderate performance for none of the ten seeds individually.
This is the concrete case `docs/STATISTICS.md` argues from in the abstract: a
raw mean is dominated by a single outlier, and IQM (`stats.iqm`) recovers the
typical-case reading (dead) that the mean obscures.

This bimodality is evidence about the **amputated observation**
(`reuse_indices=[1,2,3]`, cart position discarded), not only about network
size: `paper_linear` in exp01 - identical design at a fixed `n_repeats=4` -
showed the same collapse independent of any capacity difference, so the
capacity gap documented above compounds a pre-existing failure mode rather than
being its sole cause. A capacity-matched control (`classical_or_matched_r{R}`,
added below) changes both variables at once; whether it changes the outcome is
now an open, separately testable question rather than an assumption.

### Fix

Two independent corrections, one for each mechanism above.

**Metadata.** `migrate_manifests.py::arm_for` now recognises `classical_OR*` and
`hybrid_OR*` as distinct arms (`classical_or` / `hybrid_or`) instead of folding
both into `hybrid_fig4`; its docstring records why a directory-wide `--arm`
override is safe for exp03 and wrong for exp02. The 80 already-migrated exp02
manifests were corrected in place (`spec.arm` rewritten from the run-name
prefix; `_arm_corrected` records the previous value for audit).

**Analysis.** `notebooks/20_dqn_results.ipynb` section 5 now groups by
`(arm, R)`, never `R` alone, and reports IQM alongside the mean for exactly the
reason demonstrated above.

**Capacity.** `core/configs.py::matched_classical_or_config` /
`build_arm_config("classical_or_matched_r{R}")` - NEW-02's recipe (full
observation, budget matched to the reference hybrid's *measured* total) applied
per-R against `hybrid_or_config(R)`'s own budget, which grows with R (222 / 350
/ 606 / 1118 for R = 4/8/16/32) rather than the fixed ~126 of `HYBRID_FIG4`
alone - `classical_OR{R}` itself is left untouched, exactly as `paper_linear`
was left untouched when `matched_classical` was added for exp01.

### Scope

Metadata and analysis-code only for the migration and notebook halves -
`agent_type` and `config` were always correct in every exp02 manifest, no
episode data is affected, and nothing needs retraining. The bimodal-collapse
finding is a reading of data that already existed; it changes how `classical_OR`
should be described, not what it measures. Pinned by
`tests/test_paper_arms.py::test_classical_or_matched_control_scales_with_r`
(the capacity-matching code) and `migrate_manifests.py::arm_for`'s updated
docstring (the metadata fix); the bimodal-collapse observation itself is
empirical evidence from already-completed runs, not a code property a test can
pin, the same status FIX-01's "measured effect" section has.

**Same mechanism, a second field.** The identical failure recurred on
`dqn_kwargs`: 68 of exp04's 72 manifests carried CartPole's
`{train_frequency: 10, buffer_size: 10000}` instead of exp04's own
`{train_frequency: 1, buffer_size: 50000, learning_starts: 1000}`, from a
migration that overrode `dqn_kwargs` rather than recognising the directory via
`KNOWN_KWARGS` (which already lists exp04's correct values). `outcome`
(`wall_seconds`, `probe`, the episode CSVs) was never touched, so no training
was ever wrong and nothing here depends on the field - corrected on disk
2026-08-23. Noted because it is the second time the same override-collapses-a-
directory pattern has hit this project on a different field; a third
occurrence would be worth its own entry.

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

---

## FIX-10

**`DQN(seed=...)` does not fix every source of randomness a training run
draws on. Two runs at the same nominal seed can diverge - this project's
reproducibility is statistical (N seeds, IQM, a bootstrap CI), never
bit-identical, and no claim here should be read as promising the latter.**

Found while trying to elevate an informal diagnosis (docs/RESULTS-LOG.md's
exp04 stage-2 greedy-loop finding) into cited evidence: a from-scratch
retrain at the identical spec - same arm, same `seed=1`, same `DQN_KWARGS`,
same `total_timesteps` - landed the greedy policy in a **different** trap
than an earlier session's now-unreproducible description (`0<->4` DOWN/UP vs.
`0->1->2->3` then a self-loop at 3). A third retrain, same spec again,
landed back on `0<->4` (DOWN/UP), this time with the Q-gaps measured
directly: 0.002 at state 0, 0.018 at state 4 - matching the original
description's "~0.01-0.02" closely. That discrepancy, and the partial
recurrence, is itself the finding, not a mistake to quietly paper over. See
docs/RESULTS-LOG.md's exp04 stage-2 update for all three runs' evidence
side by side, and the explicit caveat this entry motivates: three runs from
one machine, in one project, are not equivalent samples of one population -
nothing here records what else may have differed between them, and this
very entry is why "same nominal seed" cannot be assumed to equalise them
either. What is defensible is narrower: the MECHANISM is stable (a
razor-thin argmax margin traps the greedy policy), the specific trajectory
is not, and of the two trajectories seen so far, `0<->4` has recurred
(twice) while the state-3 self-loop has not (once) - stated as a count, not
a rate.

### Mechanism

`simplyqrl.dqn.DQN.__init__` seeds three generators up front:

```python
random.seed(self.seed)
np.random.seed(self.seed)
torch.manual_seed(self.seed)
```

then builds the vectorised environment via `simplyqrl.envs.make_vec_env`,
whose per-env factory calls `env.reset(seed=seed + idx)` - which seeds the
environment's OWN internal RNG (`env.np_random`, used for stochastic
dynamics and random initial states) - but never `env.action_space.seed(...)`.
`gymnasium.spaces.Discrete` does not inherit its RNG from `env.np_random`,
`np.random`, or `random`: it lazily creates its own `np.random.Generator`
from **OS entropy** the first time `.sample()` (or `.np_random`) is touched,
unless `.seed()` was called on it explicitly. Epsilon-greedy exploration
draws exactly that way:

```python
# simplyqrl/dqn.py, the training loop
if random.random() < epsilon:
    actions = np.array(
        [self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)]
    )
```

so **every exploratory action DQN takes is drawn from a generator no `seed=`
argument reaches.** An older, unused factory in the same file
(`make_env`, lines 22-34) does call `env.action_space.seed(seed)` - but
`DQN.__init__` calls `make_vec_env`, not `make_env`, so that seeding line is
dead code on the path actually taken.

### Evidence

Three separate process invocations, identical seeding calls, identical
`make_vec_env` construction, comparing the resulting action-space samples:

```python
random.seed(1); np.random.seed(1); torch.manual_seed(1)
envs = make_vec_env(base="FrozenLake-v1", num_envs=1, seed=1)
[envs.single_action_space.sample() for _ in range(10)]

run 1: [3, 0, 2, 1, 0, 2, 2, 0, 0, 0]
run 2: [1, 3, 2, 3, 1, 1, 0, 2, 0, 0]
run 3: [3, 2, 2, 1, 2, 1, 0, 3, 3, 0]
```

Three different sequences from the identical nominal seed - `_np_random is
not None` confirms the Space WAS seeded, just not from anything `seed=1`
reaches. The other three candidate culprits checked and ruled out by reading
the code (not assumed clean):

| candidate | seeded? | where |
|---|---|---|
| weight initialisation | yes | `torch.manual_seed` runs before `build_agent(...)` constructs the network; nothing consumes torch's RNG in between |
| replay buffer sampling order | yes | `ReplayBuffer.sample` uses `np.random.randint`, the seeded global NumPy stream |
| argmax tie-breaking | not RNG-based | `torch.argmax` is a deterministic function of its input tensor |
| the environment's own dynamics / initial state | yes | `env.reset(seed=...)` seeds `env.np_random`, the generator `gym.Env.reset`/`step` actually use |
| **action-space sampling (epsilon-greedy exploration)** | **no** | confirmed above |

### Scope

**DQN only.** `simplyqrl.ppo.PPO` never calls `.action_space.sample()` -
PPO's actions come from its own trained `Categorical` distribution, which
IS seeded via `torch.manual_seed`. Grepped, not assumed:
`grep -n "action_space.sample" src/simplyqrl/ppo.py` returns nothing.

**Temporal scope: exploration-only, and it shrinks over the run.** The
unseeded sampler is reached exclusively through the `if random.random() <
epsilon:` branch - it is never touched on a greedy step. `epsilon` itself
follows `linear_schedule(start_e, end_e, duration, t)` with this project's
defaults (`start_e=1.0`, `end_e=0.05`, `exploration_fraction=0.5`, none
overridden by exp04's `DQN_KWARGS`): epsilon decays linearly from 1.0 to
0.05 over the first HALF of `total_timesteps`, then sits flat at 0.05 for
the rest. So the divergence between two nominally-identical-seed runs is
**maximal at the start of training** (epsilon near 1.0, almost every action
is the unseeded draw) and **narrows as epsilon decays** (only ~1 in 20
actions is unseeded once epsilon reaches 0.05) - it is not a constant-rate
drift across the whole run.

**Candidate (not established) explanation for a result already in this
repo: Config A's bimodal spread at L=5.** `frozen_scalar_1q_L5`'s three
coverage seeds land at best_ma 0.08, 0.80, 0.04
(`docs/RESULTS-LOG.md`'s exp04 stage-2 table: IQM 0.307, CI 0.040-0.800) -
one seed roughly ten times the other two, not a tight cluster. FrozenLake
delivers its only reward bit exclusively at the goal, so WHICH early
successes happen to land in the replay buffer plausibly shapes everything
that follows - and, by the mechanism above, that is exactly the part of
training (epsilon near 1.0, near the start) where this unseeded sampler's
influence is largest. This is a **candidate** explanation, explicitly not
an established one: it would need instrumenting when the first successful
transition enters the buffer for each seed and correlating that timing
against final performance, which has not been done. Do not cite this as the
cause of the spread without that check. Follow-up recorded in
`docs/ROADMAP.md`, not implemented here.

**Does not invalidate any reported number.** Every claim in this project is
already IQM + a percentile bootstrap CI over N seeds (`docs/STATISTICS.md`),
never a single run's trajectory - and because the unseeded draw comes from OS
entropy rather than anything correlated with the requested `seed=`, each
nominal seed's exploration is still a genuinely independent sample, which is
exactly what the N-seed statistics need. What it removes is a narrower
guarantee nothing in this project actually depended on: that re-running
`seed=3` later reproduces the SAME trajectory. It does not. State this
explicitly rather than let a reader assume it from the word "seed":
**reproducibility here is statistical (the distribution over 10 seeds, with
its interval), not bit-identical (one seed's own curve, replayed).**

Left unfixed on purpose, per the brief that produced this entry: this is a
registry entry, not a patch. Fixing it (calling
`self.envs.single_action_space.seed(seed)` early in `DQN.__init__`) would
also retroactively change what "seed=N" means for every number already in
`docs/RESULTS-LOG.md` - worth doing deliberately, in its own change, not as
a side effect of writing this entry.

---

## FIX-11

**`SafeDQN.train()` crashed AFTER training finished, on a relative `--outdir`
- the documented default for every experiment script in this repo - losing
all of that cell's compute with nothing recorded.**

Found by the mechanism that exists to find exactly this: `experiments/
exp05_dqn_frozenlake_classical_ceiling.py --smoke` (no `--outdir` given, so
the argparse default `results/exp04_dqn_frozenlake_embeddings/_smoke`, a
RELATIVE path, was in effect) crashed on its very first cell, right after
upstream printed `Training finished!`. Every prior check of this session's
weight-saving change (`tests/test_weight_saving.py`, a hand-run diagnostic
retrain) happened to use an ABSOLUTE `outdir` - pytest's `tmp_path` fixture,
or an explicit absolute scratch path - so none of them exercised the branch
that actually broke.

### Mechanism

`SafeDQN.train()` changes into `self.outdir` for the duration of training,
because upstream writes `runs/{run_name}.csv` relative to cwd (the
comment on that line explains why: a per-episode flush that survives a
Colab disconnect):

```python
try:
    os.chdir(self.outdir)
    ...
    dqn.train(total_timesteps=total_timesteps, progress_bar=progress_bar)

    weights_path = self.outdir / f"{self.run_name}_weights.pt"   # <- bug
    torch.save(dqn.q_network.state_dict(), weights_path)
    ...
finally:
    os.chdir(cwd)
```

`weights_path` is built INSIDE the `os.chdir`'d window, from `self.outdir` -
which, before this fix, was stored exactly as passed in, relative or not. A
RELATIVE `self.outdir` is a path relative to the ORIGINAL cwd; but by the
time `weights_path` is constructed, cwd IS `self.outdir` already. `pathlib`
does not re-resolve on `os.chdir` (a `Path` object is just a string), so
`self.outdir / filename` silently produces a path that means "join
`self.outdir` onto ITSELF" once `torch.save` actually opens it - a
doubly-nested path whose parent directory was never created, hence
`RuntimeError: Parent directory ... does not exist`, raised AFTER the full
training run had already completed.

### Evidence

```
$ python experiments/exp05_dqn_frozenlake_classical_ceiling.py --smoke
[1/6] frozen_binary_4q_L1__fix01on__s1
Training finished!
    FAILED: RuntimeError: Parent directory results\exp04_dqn_frozenlake_embeddings\_smoke does not exist.
```

Reproduced directly and minimally, isolated from the smoke script:
`SafeDQN(..., outdir=pathlib.Path("results/_bugcheck_relative_outdir"), ...)`
- a 500-step classical cell - raised the identical error after
`Training finished!`. Fixed by resolving `self.outdir` to an absolute path
ONCE, at construction:

```python
self.outdir = (pathlib.Path(outdir) if outdir else pathlib.Path.cwd()).resolve()
```

`os.chdir` accepts an absolute path exactly as well as a relative one, and
every `self.outdir`-relative expression built anywhere in the class -
`weights_path`, and (though never buggy, since they run AFTER the `finally`
restores cwd) `trace_path`, `eval_path` - is now correct regardless of when
in the method it is evaluated. Re-ran the identical reproduction after the
fix: succeeds, `weights_path.is_absolute()` and `.exists()`. Pinned by
`tests/test_weight_saving.py::test_weights_save_with_a_relative_outdir`,
which `monkeypatch.chdir`s to a fresh `tmp_path` and passes a genuinely
relative `outdir` - confirmed to FAIL with the exact reported error against
the pre-fix code, and to pass against the fix.

### Scope

**Every cell of every experiment run so far in this project is unaffected**:
`run_dqn_suite.py` and every experiment script pass `--outdir` explicitly
when actually launching a real pass, and this session's own relaunches
(exp04's `--dr-a --dr-b ...`, the queued exp05/exp06 coverage runs) all used
absolute Drive paths. The bug needed a RELATIVE `outdir` specifically, which
only happens when a script's `--outdir` default is used as-is - exactly what
every script's own `Usage` docstring shows as the first, simplest example
(`python experiments/exp0N_....py` with no flags). Had any future session
followed that documented usage literally for a real (non-smoke) pass, it
would have spent the FULL step budget of every cell before crashing on the
weight-save, with nothing recorded to disk (no manifest - `train()` raises
before `run_arm` can write one) and nothing to show for the compute. Caught
before that happened, by the smoke run doing exactly its job.
