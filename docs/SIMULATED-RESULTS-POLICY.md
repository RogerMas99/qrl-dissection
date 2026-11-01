# Simulated results: policy

> **Results produced by a simulator are EXPLORATORY and must not enter the final work
> unless the author has approved it explicitly, for that specific use.**
> Not in a table, not in a figure, not as a number in the text, not as an argument.

Set by the author on 2026-10-21. It applies to this repository, to every document
generated from it (briefs, summaries for the supervisor included, until the author says
otherwise for that document) and to every assistant working on it.

## What counts as a simulator result

A run whose `agent_type` is:

| agent_type | what it is | module |
|---|---|---|
| `su2` | exact classical emulator of the UNENTANGLED `skolik` circuit | `core/su2_emulator.py` (NEW-05, FIX-12) |
| `sv` | batched torch statevector simulator of the `skolik` circuit, entangled or not | `core/statevector.py` (NEW-09) |

Arms with these types are named `su2_*` / `sv_*`. The real PennyLane circuit (`hybrid`),
the classical networks and the Fourier ceilings are NOT simulator results and are not
affected by this policy.

## How a result is marked

1. **Every new manifest** written by a simulator carries a `provenance` block
   (`simulated: true`, the simulator, the status text and a pointer to this file).
2. **Older manifests** are recognised by their `agent_type`, so the rule covers them too.
3. **Tables:** `analysis.to_dataframe` adds a `simulated` column and warns loudly whenever
   a table contains simulated rows. `core/provenance.exclude_simulated(df)` drops them and is
   the guard to put in front of any table meant for the final work; passing `approved=True`
   is a statement that the author has approved that use.
4. **Result folders:** `results/exp09_dqn_emulator_sweep/` carries a notice
   (`NOT_FOR_THESIS.md`).
5. **Documents:** every document that reports simulator numbers repeats this status.

## Existing results affected (as of 2026-10-21)

| where | what |
|---|---|
| `results/exp01_dqn_cartpole_capacity/` | `su2_cartpole_L5` (original, zero init) and `su2_cartpole_L5__init2pi` (corrected init): exp06's emulator arm and its FIX-12 rerun |
| `results/exp09_dqn_emulator_sweep/` | everything: groups F, G, I (wave 1) and P, V, K, H and the diagnostics (wave 2) |
| `docs/CORRECTIONS.md` | the FIX-12 evidence table (emulator against the real circuit) and the NEW-09 benchmark: they document the tools, they are not thesis results |
| `docs/EXPERIMENT-06.md`, `-09.md` | notes and results of emulator arms |

**exp06 in particular.** Its design (section 1) uses `su2_cartpole_L5` as the
"unentangled counterpart". Under this policy that arm is exploratory; the real-circuit
counterpart that can go in the final work is `hybrid_fig4_noent` (exp08, PennyLane).

## What the simulators ARE for

Developing and checking ideas cheaply, choosing which real-circuit runs are worth their
hours, and diagnostics. A claim that is to appear in the final work is made from the real
circuit (or from classical arms), possibly guided by what a simulator suggested.

## Approvals

Explicit approvals by the author are recorded here, one line per use.

| date | what is approved (result, table or figure, and where it appears) |
|---|---|
| 2026-11-01 | Apéndice C de la memoria: descripción y resultado de la batería de validación de los simuladores (los tres criterios, la media de d con su IC, el 3 de 63, el 0 de 9 y el suelo de ruido real-real d = -0,11). Ningún otro resultado de simulador. |

This is the ONLY approved use. Everything else produced by a simulator (including
`docs/EXPERIMENT-10-X100.md` and its outputs) remains exploratory. Recorded on the
author's written instruction of 2026-11-01. The validation figures cited in that
appendix are the ones that stand at the time of writing: the original 2026-10-21
evaluation and, if the appendix is to quote it, the re-evaluation with complete real
references in `docs/EXPERIMENT-09-BATTERY.md`; the author decides which set the text uses.
