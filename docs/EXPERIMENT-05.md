# Experiment 05 — the additive Fourier ceiling on FrozenLake Config B

**Axes:** algorithm `dqn` × environment `frozenlake` × block 3 (ansatz/entanglement),
a classical-ceiling control on exp04's Config B arms
**Script:** `experiments/exp05_dqn_frozenlake_classical_ceiling.py` (Phase B, not
yet written)
**Modules:** `core/su2_emulator.py` (NEW-05, infrastructure), `core/fourier_ceiling.py`
(NEW-06, the control arm) — both exist and are tested; see docs/CORRECTIONS.md#new-05
and #new-06.
**Status:** design and analytical verification done (Phase A). **UPDATE
2026-08-30: arm registration done** — `frozen_binary_4q_fourier_ceiling`
(`linear_on_bits`, 20 params, no L suffix — see docs/CORRECTIONS.md#new-06)
is registered in `core/configs.py` and dry-run verified via `--ladder-only`
and `tests/test_new_agent_types.py`. The grid script is the only piece left.

---

## 0. Framing

This extends the reference study; it does not audit it. Nothing here is a
criticism of the dissection paper, of SimplyQRL, or of Hsiao et al. — the
Fourier ceiling is an additional classical control that sharpens what our own
numbers can support, on top of the controls exp04 already has
(`frozen_matched_scalar`, capacity-matched via NEW-02).

## 1. Where this sits

exp04's Config B (`frozen_binary_4q_L{1,5}`, `ent=True`) already measures a
capacity-matched classical control (`frozen_matched_scalar`) against the
circuit. That answers "does the circuit beat an equal-budget classical net on
the same information?" It does not answer a narrower, sharper question this
experiment adds: **does the circuit beat the exact function class an
UNENTANGLED version of itself could already express?** — because if it does
not, any gap to `frozen_matched_scalar` is about entanglement (or about the
embedding), not about the PQC being fundamentally more expressive than a
classical model of the same accessible frequencies.

Two things make FrozenLake Config B the sharpest place to ask this, ahead of
CartPole (exp06):

- **The ceiling's hypothesis class is known exactly, not estimated.**
  `FrozenBasisToAngleTransformer` maps each bit to `{0, pi}` — a two-point
  domain on which `sin(k*z)` vanishes identically and `cos(k*z)` collapses to
  an affine function of the bit, at every reuploading depth L
  (docs/CORRECTIONS.md#new-06, verified against the real circuit in
  `tests/test_frozenlake_additive_ceiling.py`, not just derived). The ceiling
  is therefore a fixed, 5-parameters-per-action linear model on the bits —
  nothing to tune, nothing to match by capacity, an exact target.
- **The prediction is falsifiable and pre-registered before any compute is
  spent** (§5). If the entangled arm also sits at the ceiling, the
  entanglement contrast in this environment is null and that is reported as
  a bound on the claim, not smoothed into a positive finding.

## 2. The analytical result (docs/CORRECTIONS.md#new-06, §2 of the design brief)

> **FrozenLake Config B without entanglement is exactly a linear model on the
> four bits, at any reuploading depth.**
>
>     Q_a(s) = sum_i w_ai (alpha_i + beta_i b_i) + c_a
>
> — 5 free parameters per action, independent of L.

Verified against `build_skolik_qlayer` directly (not against the ceiling
module, which is that hypothesis class by construction and would prove
nothing about the real circuit):

| check | result |
|---|---|
| each `<Z_i>` takes exactly two values, keyed by `b_i` | holds at L = 1, 2, 5 |
| affine-in-bits residual (max abs, float32) | < 1e-4 at every tested L |
| residual growth with L | none — same order of magnitude at L=1 and L=5 |
| negative control, `ent=True`, L=5 | residual > 0.05 |

## 3. Design

### Arms

| arm | status | role |
|---|---|---|
| `frozen_binary_4q_L1` / `L5` | **exists, exp04b** | entangled — the object of study |
| `frozen_binary_4q_noent_L1` / `L5` | registered (`core/configs.py`), not yet run | unentangled circuit — the thing the ceiling bounds |
| `frozen_binary_4q_fourier_ceiling` | **registered 2026-08-30**, not yet run | the classical ceiling — degenerate `linear_on_bits_ceiling`, 5 params/action, no L suffix (L-independent) |
| `frozen_matched_scalar` | **exists, exp04b** | capacity-matched classical control (NEW-02), already answers the budget-matched question |

`frozen_binary_4q_noent_L{1,5}` is not a new arm to design — it already
exists, registered when exp04 was specified, for exactly this ablation
(embedding-vs-entanglement, see `docs/EXPERIMENT-04.md`). exp05 is what
finally spends the compute on it, alongside the ceiling.

### Grid

Coverage first (3 seeds), then the robustness pass (8–10 seeds) — same
discipline as every other block sweep in this repo. FrozenLake Config B is
cheap enough (measured ~25 steps/s at 4 qubits) that the robustness pass is
affordable from the start, as it was for exp04's own Config A/B arms.

| arm | FIX-01 | seeds (coverage) |
|---|---|---|
| `frozen_binary_4q_L1` | on | reuses exp04b's 3 seeds |
| `frozen_binary_4q_L5` | on | reuses exp04b's 3 seeds |
| `frozen_binary_4q_noent_L1` | on | 3, new |
| `frozen_binary_4q_noent_L5` | on | 3, new |
| `frozen_binary_4q_fourier_ceiling` | n/a — classical, no PQC | 3, new (cheap) |
| `frozen_matched_scalar` | on | reuses exp04b's 10 seeds |

## 4. Pre-declared hypotheses and readings

Fixed before any exp05 cell runs, so the analysis cannot become "find a story
that fits."

**H1.** `SU2SkolikEmulator` ≡ the real `noent` circuit at 1e-6, forward and
gradient. **Already checked** (`tests/test_su2_equivalence.py`) — if it ever
fails after a `simplyqrl` version bump, the emulator is wrong; fix it, do not
report the discrepancy as a finding about the circuit.

**H2.** `noent` ≈ ceiling ≈ chance at every depth; the entangled arm sits
above both. This is the interesting outcome — it would mean entanglement
(not depth, not the embedding) is what lets Config B escape the dead floor
`frozen_matched_scalar` and `frozen_scalar_1q` already showed at low depth
(see `docs/RESULTS-LOG.md`, exp04). **If the entangled arm is ALSO at
chance**, the contrast is null in this environment — report it as a bound on
the claim, not as an inconclusive footnote to a positive story.

**H3.** Not the primary question here (CartPole is exp06's job), but worth
watching: if `noent` or the ceiling ever beat the entangled arm, that is not
evidence against entanglement mattering elsewhere — it would say this
particular depth/qubit-count combination does not need it, a narrower and
still-useful claim.

## 5. What would falsify the premise

If `frozen_binary_4q_noent_L5`'s measured output is NOT affine in the bits
(contradicting §2's already-verified analytical result), something has
changed upstream — re-run `tests/test_frozenlake_additive_ceiling.py` before
trusting any exp05 number, the same discipline `docs/EXPERIMENT-04.md`
applies to its own liveness gate.

## 6. Open item carried from Phase A

The Fourier ceiling's parameter count is not a free design choice (see
`docs/CORRECTIONS.md#new-06`, "Open sizing question") — report it alongside
each hybrid arm's measured budget for context, do not attempt to match it via
NEW-02's recipe. Confirm this reading before the robustness pass, not after.
