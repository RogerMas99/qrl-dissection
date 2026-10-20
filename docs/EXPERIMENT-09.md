# Experiment 09 — Output Reuse x depth on FrozenLake, on the SU(2) emulator

> **STATUS: EXPLORATORY EXTENSION — NOT PART OF THE THESIS CORE.**
> This experiment explores extra options that the core design (exp01–exp08,
> and the controls around them) did not cover. It is run because it is cheap,
> not because the thesis depends on it. Nothing here is a pre-registered test
> of the core hypotheses; results are to be reported separately and labelled
> exploratory, unless the supervisor decides to promote any of it into the
> core. Created 2026-10-20.

**Axes:** algorithm `dqn` x environment `frozenlake` x blocks 1 (Output Reuse) and 2 (DR / embedding)
**Script:** `experiments/exp09_dqn_emulator_sweep.py`
**Arms:** `su2_frozen_{scalar_1q,binary_4q}_L{L}[_OR{R}]` (24, `core/configs.py`, section "[exp09]")
**Results:** `results/exp09_dqn_emulator_sweep/` (Drive)

---

## 0. Why this exists

- On CartPole, Output Reuse is the strongest single lever in our data
  (`hybrid_OR{4,..,32}` against `hybrid_fig4`: P between 0.96 and 1.00), and the only
  setting in which entanglement showed an advantage over its unentangled twin
  (provisional: the unentangled OR arms are still at n = 5–9).
- It has never been tried on FrozenLake. That is a matter of timing, not of a
  decision against it: exp04 was designed while exp02 (OR) was still running and
  exp03 (DR) was the block that had transferred (`docs/EXPERIMENT-04.md`, section 1).
- `OutputReuse` is `x.repeat(1, R)` (`simplyqrl/agents.py`): R identical copies
  of the measured expectation values. It does not enlarge the function class of
  the head; it re-parametrises it. Whatever it does on FrozenLake is therefore
  an optimisation effect, a hypothesis this experiment can inform but not settle.

## 1. What the emulator can and cannot say

`SU2SkolikEmulator` reproduces the UNENTANGLED skolik circuit exactly (NEW-05;
initialisation matched in FIX-12) at 3–8 minutes per 100k-step FrozenLake cell,
against 7–25 hours for the real circuit. It cannot represent an entangled
circuit. **This sweep says nothing about entanglement.** The crossing Output
Reuse x entanglement needs the real circuit (about 25 h per seed on Config B,
L = 5) or a purpose-built batched statevector simulator: phase 2, gated on what
this phase shows.

## 2. Design (wave 1)

| group | what | arms | seeds | cells | est. compute |
|---|---|---|---|---|---|
| **F** | Config A (1 qubit, scalar embedding): L in {1, 5, 10, 15} x R in {1, 4, 16} | 12 | 1–10 | 120 | ~14 h |
| **G** | Config B (4 qubits, binary embedding, ent=False): L in {1, 2, 3, 5} x R in {1, 4, 16} | 12 | 1–10 | 120 | ~8 h |
| **I** | protocol sensitivity: train_frequency {1, 10} x learning_starts {1k, 10k} x buffer {50k, 10k} on `su2_frozen_scalar_1q_L5` and `su2_frozen_binary_4q_L5` | 2 x 8 variants | 101–103 | 48 | ~5 h |

- R = 1 means no Output Reuse. At L = 1 the entangled and unentangled circuits are
  the same circuit (FIX-07), so Config B L = 1 is fully covered.
- **Protocol.** F and G use exp04's own FrozenLake protocol (batch 128, buffer
  50 000, train_frequency 1, learning_starts 1 000) so that they stay comparable
  with every FrozenLake result already on disk. I is a **sensitivity analysis**
  run on separate seeds (101–103); it is not used to choose the protocol of F
  and G. If it shows strong sensitivity (as the L = 5 Fourier ceiling did, FIX-13),
  that is reported as a caveat and, if worthwhile, F/G are repeated under the
  better protocol as a clearly labelled secondary set.
- 100 000 steps, FIX-01 on, greedy evaluation every 10 000 steps (20 episodes).
- Cells are run by several workers of one group with `--claim` (cooperative
  locks, TTL 2 h).

## 3. Contrasts fixed before any cell runs

Primary outcome `greedy_final`; secondary `final_perf`; probability of
improvement P(a > b) with bootstrap CIs on the IQM, as in the rest of the repo.
Six contrasts are treated as the confirmatory set of this exploration; every
other comparison is reported as exploratory.

1. Config A, L = 5: R = 16 against R = 1.
2. Config A, L = 10 and L = 15: R = 16 against R = 1.
3. Config B (ent=False), L = 5: R = 16 against R = 1.
4. Config B (ent=False), L = 1: R = 16 against R = 1.
5. Dose–response: R in {1, 4, 16} at each (config, L) — monotone or not.
6. Emulated Config A circuit against the Fourier ceiling (`__flkw`, FIX-13) at
   matched L, to see whether OR closes any of the ceiling gap.

Reading rules: an effect on a configuration that is dead at R = 1 and alive at
R > 1 is the interesting outcome; no effect anywhere is a result and is reported
as such. A gain on `final_perf` with no gain on `greedy_final` is the value-vs-
policy dissociation (greedy evaluation on deterministic FrozenLake has one
effective sample per checkpoint) and is not read as learning.

## 4. Not in wave 1

- **Control: a classical MLP on the four bits with a matched budget.** Missing
  from the FrozenLake design since exp04; it needs a new observation adapter
  (state as bits) and is not part of this wave.
- **CartPole mechanism of OR** (head size, learning rate, a classical control that
  repeats its inputs), new DR x OR points and per-depth Fourier ceilings: a second
  wave, only if wave 1 or the supervisor makes them worth the time.
- **Output Reuse x entanglement on the real circuit** (phase 2).

## 5. Results

Filled in when the groups finish.
