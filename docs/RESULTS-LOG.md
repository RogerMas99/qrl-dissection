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

Paper block 1. Sweep R in {4,8,16,32} on the hybrid arm, 60k steps, 3 seeds
(coverage). Paper (PPO): OR helps hybrid, not classical -> genuine quantum
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

## Experiment 03 - Data Reuploading (DR) under DQN - to run

Paper block 2. Sweep depth L in {1,2,5} on the Skolik template, 60k steps, 3
seeds (coverage). exp01's single-seed 103k run suggested depth matters (L=1
failed), so DR may show a visible effect even here.

| L | best_ma50 | greedy_best |
|---|---|---|
| 1 | | |
| 2 | | |
| 5 | | |

Reading: performance rises with L (as paper) -> DR transfers. ROBUSTNESS (B):
re-run at 8-10 seeds before concluding.
