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
