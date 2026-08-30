# Experiment 06 — the additive Fourier ceiling on the CartPole Skolik sweep

**Axes:** algorithm `dqn` × environment `cartpole` × block 3 (ansatz/entanglement)
**Script:** `experiments/exp06_dqn_cartpole_classical_ceiling.py` (Phase B, not
yet written)
**Status:** stub — priority 2, behind exp05. See `docs/ROADMAP.md`'s Named
follow-ups entry for the dependency ordering (NEW-05 before NEW-06 before
either experiment; exp05 before exp06). **UPDATE 2026-08-30: arm
registration done** — `cartpole_fourier_ceiling_L5` (`fourier_additive`, 162
params) and `su2_cartpole_L5` (`su2`, 126 params — see §1 below for why this
serves as the "noent counterpart" this doc originally called for) are
registered in `core/configs.py`. The grid script is the only piece left.

---

## 0. Why this is priority 2, not priority 1

Same construction as exp05 (`core/fourier_ceiling.py::FourierAdditiveCeiling`,
guarded by `check_additive_embedding`), applied to `hybrid_fig4`'s Skolik
8-qubit circuit instead of FrozenLake's 4-qubit one. The prediction
(docs/CORRECTIONS.md#new-06, H3) is that the ceiling and the unentangled
circuit sit close together on CartPole, because CartPole is largely solvable
by near-linear controllers already (`paper_linear` at 26 parameters is the
only arm in this repo that reliably dies, and `matched_classical` at 135
barely clears it — see `docs/RESULTS-LOG.md`, exp01) — so the additive ceiling
is probably not the binding constraint here, unlike FrozenLake Config B where
it is the whole story.

**Expect an inconclusive result and report it as such**: "the environment
does not discriminate between these hypothesis classes", not as a null
finding dressed up as informative. Run it because it is cheap (CartPole's
own throughput, not FrozenLake's), not because it is expected to be the
sharp comparison — that is exp05.

## 1. Design (sketch, to be filled out before Phase B)

- Arms: `hybrid_fig4` (the reference Skolik 8q circuit, `ent=True`), its
  `noent` counterpart, and the Fourier ceiling sized to `n_qubits=8`,
  `n_layers=hybrid_fig4`'s `n_layers_q`, `n_actions=2`
  (`cartpole_fourier_ceiling_L5`).

  **The `noent` counterpart is `su2_cartpole_L5`, not a second real-PQC
  arm.** `hybrid_fig4` is 8 qubits at 100k steps — the expensive regime in
  this repo ("can run for hours", `docs/REUSE.md`). NEW-05's whole point is
  that an unentangled `skolik` circuit is exactly reproducible by
  `SU2SkolikEmulator` without a quantum simulator
  (docs/CORRECTIONS.md#new-05, ~350x faster measured on this machine), so
  spending real PQC compute on an 8-qubit `ent=False` run here would be
  paying the expensive regime's cost for a result NEW-05 already predicts to
  1e-6 per call. `su2_cartpole_L5` is `HYBRID_FIG4` with `ent` forced False,
  same `net_arch`/`activation`/`n_qubits`/`n_layers_q` — same architecture,
  no quantum simulator underneath (`core/configs.py`, measured 126 trainable
  parameters, identical to `hybrid_fig4`'s own count). If a reviewer wants a
  real-PQC noent anchor point too, `paper_skolik_8q_L5` (`ARMS`, already
  registered, `ent=True` by its own default — pass `ent=False` via
  `paper_skolik_config` before reusing it) is the closest existing
  alternative; not included in this design by default because it would
  reintroduce the expensive-regime cost NEW-05 exists to avoid.
- `check_additive_embedding("skolik", 8, n_data=4)` — passes; CartPole's
  Skolik embedding is the cycling case NEW-05's equivalence test already
  covers (`skolik_8q_cartpole_L5`), so no new embedding logic is needed here,
  only the arm.
- Pre-declared reading (**H3**, docs/CORRECTIONS.md#new-06): ceiling >=
  hybrid `noent` within noise is the expected outcome. **If the hybrid beats
  the ceiling, do not call it expressivity** — the classical class has freer
  coefficients (the circuit's are constrained by unitarity), so the honest
  reading is inductive bias or an optimisation effect, stated as such.

## 2. What this does not need

No new derivation — the Fourier-ceiling argument (§2 of
`docs/CORRECTIONS.md#new-06`) is embedding-specific, not
environment-specific: CartPole's Skolik circuit uses the same `skolik`
template and the same `angle_embedding` (cycling) path NEW-05/NEW-06 already
verify. What is new here is only the CartPole-specific degeneracy question —
unlike FrozenLake's exact two-point domain, CartPole's continuous
observations do not collapse the Fourier basis to a fixed low-dimensional
form, so the general `FourierAdditiveCeiling` (not a degenerate variant like
`linear_on_bits_ceiling`) is the right tool here. State that difference
explicitly when this doc is filled out, so a reader does not expect the same
5-parameter collapse exp05 gets.

**This is already checked on a continuous domain, ahead of exp05's**
(`tests/test_fourier_ceiling_spectrum.py`, docs/CORRECTIONS.md#new-06): the
degree-L trigonometric-polynomial claim holds against the real `skolik`
circuit for x sampled densely over a full period, not only at FrozenLake's
two encoded points, with a negative control proving the circuit genuinely
reaches frequency L (residual jumps 4-6 orders of magnitude when the top
frequency is dropped from the fit basis). exp06 inherits this directly; no
new verification is owed before Phase B, only the arm registration.

## 3. Filled out on completion

Grid, seeds, results table — added when Phase B starts and this stub is
promoted to a full design doc, following `docs/EXPERIMENT-05.md`'s format.
