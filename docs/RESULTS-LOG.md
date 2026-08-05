# Results log

Append-only. One entry per grid, with the summary table and the reading. Raw
artefacts stay out of git - only what is needed to follow the argument goes here.

Metric convention: `best_ma50` unless stated. Random policy ~22; degenerate
constant-action policy ~9.5.

---

## Prior work (before this repository)

Carried over from the diagnosis phase for context. Configurations differ from
the arms defined here; see `docs/EXPERIMENT-01.md` section 3.

| configuration | seeds | learns? | FIX-01 delta |
|---|---|---|---|
| MLP 120-84, batch 16, 60k | 3 | yes, ~130 | -11 |
| MLP 120-84, batch 128, 60k | 3 | yes, ~212 | +1 |
| MLP 120-84, batch 128, 60k (repeat) | 3 | yes, ~226 | +39 (sd 64-86) |
| paper linear, tf=10, 60k | 3 | no, ~9.8 | -0.0 |
| paper linear, tf=1, 60k | 3 | no, ~10 | -0.4 |
| hybrid Fig. 4 config, 103k, seed 1 | 1 | weakly, peak 57 | not tested |

Reading: FIX-01 shows no significant effect in any configuration where the
effect is measurable at all. The two linear rows are uninformative - the agent
is dead for an unrelated reason.

---

## Experiment 01 - capacity-matched classical control

Status: **specified, not yet run.**

Fill in on completion:

```
grid:        arms x FIX-01 {off, on} x seeds {1,2,3}
steps:       60,000
dqn:         batch_size=128, buffer_size=10000, train_frequency=10
commit:      <git rev>
environment: <output of scripts/verify_env.py>
```

| arm | FIX-01 off | FIX-01 on | delta |
|---|---|---|---|
| paper_linear | | | |
| matched_classical | | | |
| oversized_mlp | | | |

Reading:


---

## Experiment 01 v1 - hybrid vs matched (AMPUTATED input) - SUPERSEDED as main comparison

Grid: 4 arms x FIX-01 {off,on} x 3 seeds, 60k steps, batch 128, tf 10.

| arm | params | best_ma50 (off/on) | greedy_best |
|---|---|---|---|
| paper_linear | 26 | 22.3 / 22.8 | ~10 |
| matched_classical (amputated) | 122 | 23.9 / 22.2 | ~10 |
| hybrid_fig4 | 126 | 54.9 / 53.0 | ~44 |
| oversized_mlp | 10,934 | 253.9 / 285.5 | ~400 |

**Caveat that demotes this from the main result:** matched_classical here used
the paper's amputated input (cart position discarded), while the hybrid saw all
four observations. So the arm may have died from missing a termination variable,
not from being classical. Treated as an ABLATION (cost of the amputation), not
the circuit-vs-classical comparison.

Standing conclusions unaffected by the caveat:
- FIX-01 shows no significant effect in any live arm (hybrid 54.9->53.0;
  oversized_mlp +31.6; both within noise, n=3).
- oversized_mlp confirms the DQN+CartPole setup is sound.

## Experiment 01 v2 - hybrid vs matched (FULL input) - to run

matched_classical rebuilt with observation="full" (in_dim 16, width 7, ~135
params). This is the clean circuit-vs-classical comparison. Re-run only the
matched arm; the other three are unchanged and their manifests are reused.

| arm | FIX-01 off | FIX-01 on |
|---|---|---|
| matched_classical (full) | | |

Reading:
- matched (full) still dies -> the hybrid result stands: at equal budget AND
  equal information, the circuit learns where the classical block does not.
- matched (full) now learns -> the v1 death was largely the amputation; the
  circuit's advantage is smaller or absent. Compare best_ma50 to the hybrid's 54.


---

## Experiment 02 - Output Reuse (OR) under DQN - to run

Paper block 1. Sweep R in {4,8,16,32} on the hybrid arm, **100k steps** (to
match exp03/DR for cross-block comparability), 3 seeds (coverage). Paper (PPO): OR helps hybrid, not classical -> genuine quantum
interaction. Question: does that transfer to DQN?

| R | best_ma50 | greedy_best |
|---|---|---|
| 4 | | |
| 8 | | |
| 16 | | |
| 32 | | |

Reading: monotone gains with R (as paper) -> OR transfers. Flat/noisy -> OR
effect not measurable in this weak-learning DQN regime (report as such).
ROBUSTNESS (B): re-run at 8-10 seeds before concluding.

## Experiment 03 - Data Reuploading (DR) under DQN - COVERAGE DONE

!! TWO AMENDMENTS from merging the paper's companion repository (docs/PAPER-BASELINES.md).

(1) COMPARISON AGAINST THE PAPER'S OWN LOGS, not figures. Skolik_8Q, PPO, n=10:
    L1 32.2 +/- 2.6, L2 144.0 +/- 37.9, L5 199.3 +/- 60.1. Ours (DQN): 15, 35,
    199. Endpoint indistinguishable; the intermediate points are far outside the
    paper's spread. Both curves rise monotonically, so "DR transfers" survives -
    but the SHAPE does not. Under PPO most of the gain has arrived by L2; under
    DQN it arrives late. The defensible claim is narrower: DR helps DQN too, but
    DQN needs more depth for the same benefit. Note also that our L1 value of 15
    sits below CartPole's random baseline of ~22, i.e. a degenerate policy.

(2) CONFOUND, see CORRECTIONS.md#fix-07. This sweep runs the Skolik template
    with ent=True, and on that template the final layer's CZ ring cannot affect
    a PauliZ readout. Depths 1/2/5 therefore carry 0/1/4 EFFECTIVE entangling
    blocks: the depth axis moves entanglement too. Part of the monotone gain may
    be entanglement coming online rather than reuploading. Rerunning the sweep
    with ent=False separates them and costs one flag.

Paper block 2. Sweep depth L in {1,2,5} on the Skolik template, **100k steps**
(raised from 60k - DR needed more budget to unfold), 3 seeds. FIX-01 on.

| L | best_ma50 (mean) | greedy_best (mean) | greedy per seed |
|---|---|---|---|
| 1 | 24.1 | 15.2 | 12.8 / 22.0 / 10.8 |
| 2 | 36.3 | 35.3 | 51.8 / 21.4 / 32.8 |
| 5 | 79.4 | 199.3 | 408.2 / 61.0 / 128.8 |

**Reading: DR transfers to DQN, clearly.** Monotone rise with depth on both
metrics; greedy 15 -> 35 -> 199, a >10x jump from L=1 to L=5. One L=5 seed hit
greedy 408 (near-solved; CartPole max 500) - the first hybrid in the whole
project that learns strongly rather than barely. This matches the paper's PPO
finding (deeper reuploading improves trainability) and, unlike the circuit block
(exp01, no clear advantage), it DOES transfer to the off-policy setting. A clean
cross-block contrast: DR transfers, the ansatz/entanglement block does not.

Caveats:
- Measured at 100k steps, NOT 60k like exp01. State this in any comparison. OR
  (exp02) should also run at 100k for cross-block comparability.
- L=5 variance is high (greedy 408/61/129). "Depth helps monotonically" is safe;
  "L=5 gives greedy ~200" is not precise at n=3.
- ROBUSTNESS (B): re-run at 8-10 seeds before writing the final numbers.

---

## Experiment 04 - embedding & DR under DQN, FrozenLake-v1 - to run

Block 2 (embedding / DR), second environment. Design: `docs/EXPERIMENT-04.md`.

ATTRIBUTION: the dissection paper is CartPole-only. FrozenLake comes from the
SimplyQRL library chapter (Exp 3: PPO, 500k steps, single seed, no controls), so
this is NOT a transfer test against a dissected baseline. Its purposes are (a)
testing whether exp03's DR result generalises beyond one environment and (b)
measuring FIX-01 where the phantom fraction is large.

Blocked on FIX-05 until `tests/test_frozenlake_envs.py` passes: upstream cannot
run a Discrete observation space under DQN at all.

### Stage 0 - accounting and liveness (MEASURED during integration)

Recorded before any grid, which is the point: H3 claims the FIX-01 effect tracks
the phantom fraction, so the prediction must pre-date the measurement.

| quantity | value | reference |
|---|---|---|
| random-policy success rate | 0.0150 | floor the learning curves must clear |
| mean episode length (random) | 7.69 | |
| predicted phantom fraction 1/len | 0.130 | CartPole at convergence: < 0.01 |
| MEASURED phantom, one-hot arm 6k steps | 0.086 (fix off) / 0.099 (on) | ~10x CartPole |

Capacity ladder (verified on the pinned stack):

| arm | type | params | note |
|---|---|---|---|
| frozen_scalar_1q_L1 / L5 / L10 / L15 | hybrid | 10 / 18 / 28 / 38 | q = 2/10/20/30, head Linear(1,4) = 8 |
| frozen_binary_4q_L1 / L5 | hybrid | 28 / 60 | q = 8/40, head Linear(4,4) = 20 |
| frozen_matched_scalar | classic | 22 (width 3) | >= hybrid 18, rounded up deliberately |
| frozen_scalar_mlp_large | classic | 4548 | encoding vs capacity |
| frozen_onehot_mlp | classic | 5508 | liveness guard |

Structural note for the write-up: Config A's head is `Linear(1, 4)`, so all four
Q-values are affine in a single expectation value. That is a limit of the
chapter's configuration, not of our set-up.

### Stage 1 - classical arms x FIX-01, 5 seeds, 100k steps

SMOKE (1 seed, 5k steps) already run during integration - not the result, but it
decides how to read the result:

| arm | success (MA-100) | reading |
|---|---|---|
| frozen_onehot_mlp | 0.86 - 0.96 | regime is ALIVE and fast |
| frozen_scalar_mlp_large | 0.03 - 0.07 | barely above random (0.015) |
| frozen_matched_scalar | 0.04 - 0.07 | same |

| arm | FIX-01 | best success (MA-100) | greedy_best | phantom frac |
|---|---|---|---|---|
| frozen_onehot_mlp | off | | | |
| frozen_onehot_mlp | on | | | |
| frozen_scalar_mlp_large | off | | | |
| frozen_scalar_mlp_large | on | | | |
| frozen_matched_scalar | off | | | |
| frozen_matched_scalar | on | | | |

GATE: `frozen_onehot_mlp` must reach a success rate near 1.0 or nothing below is
interpretable. The smoke run says it does, comfortably.

WARNING carried forward from the smoke run: the scalar-input arms may be DEAD. A
network fed the raw state index has almost nothing to work with - row-major
ordering carries no usable metric. If `frozen_matched_scalar` stays at chance,
H4 is unanswerable (a hybrid beating a dead control proves nothing) and the
honest claim becomes one about the ENCODING, not the circuit. Config A runs on
that same scalar input and may be dead for the same reason; in that case the
informative comparison is Config B against the one-hot MLP.

### Stage 2 - hybrid sweeps, FIX-01 on, 3 seeds (coverage)

| config | n_qubits | DR depth | best success | greedy_best | sd |
|---|---|---|---|---|---|
| A scalar-to-phase | 1 | 1 | | | |
| A scalar-to-phase | 1 | 5 | | | |
| A scalar-to-phase | 1 | 10 | | | |
| A scalar-to-phase | 1 | 15 | | | |
| B binary-basis | 4 | 1 | | | |
| B binary-basis | 4 | 5 | | | |

Ablation (`--with-ablation`, `ent=False` on Config B). Needed because Config A
cannot be entangled at one qubit, so A vs B otherwise confounds embedding with
entanglement:

| config | DR depth | best success | greedy_best |
|---|---|---|---|
| B binary-basis, no ent | 1 | | |
| B binary-basis, no ent | 5 | | |

### ROBUSTNESS (plan B)

Three seeds is a COVERAGE pass. At 1-4 qubits (~50 and ~25 steps/s measured) the
8-10 seed pass deferred on exp01/02/03 is affordable here for the first time. Do
not write any of this up as a conclusion before it is done.

| pass | seeds | status |
|---|---|---|
| coverage | 3 | pending |
| robustness | 8-10 | pending |

