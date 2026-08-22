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

> !! EVERY NUMBER BELOW WAS COMPUTED WITH A BIASED STATISTIC.
>
> `best_ma50` and `greedy_best` are maxima over the training curve. A maximum
> over a noisy curve is positively biased and the bias grows with variance, so a
> noisier arm scores higher for nothing - measured at 567 vs 306 for two arms of
> identical true performance. The quantum arms here are several times noisier
> than the classical ones, so the protocol flatters them.
>
> Recompute with `core.stats.final_performance` (mean over the final 10% of
> episodes) before treating any of these as conclusions. It needs no retraining:
> every episode is in the CSVs. Notebook 20 section 2b reports both side by side.
> Rationale and checklist: docs/STATISTICS.md.
>
> The "UPDATE 2026-08-21" subsections added below this point report both
> statistics side by side, with IQM and 95% bootstrap CIs in place of mean+-sd,
> per the STATISTICS.md checklist. Entries without such an update still carry
> only the original, biased number - recompute before citing them.

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

### UPDATE 2026-08-21 - run at 100k steps (not the 60k above), n=10 for 3 of 4 arms

Found already computed on the synced Drive results tree when auditing before
exp04b; not previously transcribed here. Step budget raised to 100k for
cross-block comparability with exp02/exp03 (ROADMAP's own named follow-up).
`hybrid_fig4` is still at n=3 (coverage) - PQC training is the expensive part
of this grid and was not re-run at the same depth as the three classical arms.
Both metrics reported per the STATISTICS.md checklist: `greedy_best` (legacy,
a maximum over training - biased upward, more so for a noisier arm) and IQM of
`final_performance` (mean of the last 10% of episodes - unbiased, needs no
retraining). 95% percentile bootstrap CIs in parentheses.

| arm | FIX-01 | n | greedy_best IQM (CI) | final_performance IQM (CI) |
|---|---|---|---|---|
| paper_linear | off | 10 | 11.7 (9.4, 20.2) | 9.6 (9.6, 9.7) |
| paper_linear | on | 10 | 11.7 (9.4, 19.0) | 9.6 (9.6, 9.7) |
| matched_classical (full) | off | 10 | 12.7 (9.7, 32.1) | 9.9 (9.7, 10.3) |
| matched_classical (full) | on | 10 | 20.9 (12.6, 40.9) | 11.8 (10.7, 12.6) |
| hybrid_fig4 | off | 3 | 97.7 (51.2, 140.0) | 38.1 (22.8, 66.6) |
| hybrid_fig4 | on | 3 | 107.7 (51.6, 139.4) | 44.7 (25.9, 54.6) |
| oversized_mlp | off | 10 | 461.1 (391.8, 492.8) | 198.3 (166.4, 231.5) |
| oversized_mlp | on | 10 | 449.4 (406.6, 476.3) | 252.0 (234.8, 279.4) |

**Reading, with the honest metric.**

- **`matched_classical` is ALSO at the degenerate floor** by `final_performance`
  (IQM 9.9-11.8, indistinguishable from `paper_linear`'s 9.6-9.7), even though
  its `greedy_best` (12.7-20.9) looks marginally alive. This resolves v2's own
  open question from above: the amputation was not the whole story, but neither
  is a fair, capacity-matched control enough to make the classical side live at
  this training budget - the same collapse `paper_linear` shows, just slightly
  less severe. NEW-02 makes the comparison fair; it does not make the classical
  arm learn.
- **The hybrid's advantage over the matched control survives on the honest
  metric** (final IQM 38.1-44.7 vs matched's 9.9-11.8) - a real, if modest, gap,
  not an artefact of `greedy_best`'s variance bias. This is the strongest
  available claim about the circuit under DQN so far, and it is now measured
  with the metric that does not favour a noisier arm.

  > **STATUS: UNDER REVIEW, 2026-08-21 - do not read this bullet as settled in
  > either direction.** Open question: is `matched_classical` dead because its
  > TOTAL budget (135) is insufficient, or because that budget is spent as a
  > single NARROW hidden layer (width 7)? These are confounded by construction
  > in the current recipe - `capacity.match_hidden_width` solves a single
  > `Linear(in,h)->Linear(h,out)` shape, where `cost(h)` is strictly monotone
  > in `h` given fixed in/out dims, so "wider, same total budget" is not
  > expressible without a different shape (more layers, a different in_dim, or
  > similar) - a genuine architecture decision, not yet made. Until it is
  > resolved, treat "the hybrid's advantage survives" as provisional: it may
  > reflect the circuit, or it may reflect that the matched control was never
  > given a fair shape, only a fair total count. See the open item in
  > `docs/CORRECTIONS.md` if/when this is investigated further.
- **But "the hybrid learns" needs qualifying.** Final-performance IQM of 38-45
  is far from `oversized_mlp`'s 198-258 - the hybrid clears the dead-classical
  floor decisively but is nowhere near a converged CartPole policy. Both
  statements are true and belong together, not separately.
- **FIX-01 on `oversized_mlp` no longer looks flatly null.** At n=3 (exp01 v1)
  the delta was "+31.6, within noise". At n=10 with `final_performance`: off IQM
  198.3 (166.4, 231.5) vs on IQM 252.0 (234.8, 279.4) - the CIs are adjacent
  rather than overlapping. Not yet a confirmed effect (adjacent CIs are not a
  significance test), but it is the first live arm anywhere in this project
  where the previously-null FIX-01 reading deserves a second look rather than
  being repeated by default.
- **`hybrid_fig4` remains the one arm not yet re-run at n=10** - it is the
  object of study, and the top-level "seeds" count for this directory (10) does
  NOT reflect that, because it is computed over all arms pooled. Check
  per-arm, always.

---

## Experiment 02 - Output Reuse (OR) under DQN - to run

!! AMENDMENT. The classical control was specified in this experiment's docstring
   from the start and never implemented: the script swept R on the hybrid arm
   only. That made the experiment unable to test its own question. The paper's
   finding is not "OR helps" but "OR helps hybrid agents and NOT classical ones"
   - a statement about the DIFFERENCE between two arms - so a hybrid-only sweep
   can show OR helping and still say nothing about whether the help is quantum.
   Same lesson as exp01's broken control, on a different axis.

   The control now runs by default (`classical_OR{R}__s{seed}`, transcribed from
   post-pqc-inference.py :: config_classical). It costs almost nothing: measured
   0.4s per cell against 32.7s for its hybrid counterpart at the same budget, and
   ~9h at full budget. The 16 hybrid cells already on Drive stay valid; the
   control is added alongside them.

   AND NOTE WHAT THE PAPER'S OWN DATA SAYS, now that we have it at 10 seeds:

   | R | Quantum | Classical |
   |---|---|---|
   | 4 | 116.5 +/- 28.9 | 263.9 +/- 18.4 |
   | 8 | 230.0 +/- 88.1 | 320.0 +/- 36.3 |
   | 16 | 354.8 +/- 124.3 | 381.7 +/- 37.3 |
   | 32 | 430.5 +/- 115.6 | 330.2 +/- 63.4 |

   The classical arm is AHEAD at three of the four reuse factors, and the single
   point where the hybrid leads (r32) has a standard deviation of 115 against a
   gap of 100 - inside noise. Whatever exp02 finds under DQN, the comparison it
   is being measured against is weaker than the published framing suggests, and
   the write-up should say so.

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

### UPDATE 2026-08-21 - n=10 found on Drive; FIX-09 metadata bug found and fixed first

Found already computed (n=10, 100k steps) when auditing before exp04b - not
previously transcribed. **Before reading these numbers, see
`docs/CORRECTIONS.md#fix-09`**: both arms' manifests were migrated with the
same `spec.arm = "hybrid_fig4"` label, and the notebook's own grouping (by `R`
alone, from the run name) pooled hybrid and classical cells of the same `R`
into one meaningless average. Both are now fixed; the numbers below are the
corrected, separated read. `greedy_best` (legacy, biased) and IQM of
`final_performance` (unbiased) both reported, 95% bootstrap CIs in
parentheses, per the STATISTICS.md checklist.

| arm | R | n | greedy_best IQM (CI) | final_performance IQM (CI) |
|---|---|---|---|---|
| hybrid_or | 4 | 10 | 178.8 (127.9, 274.9) | 94.7 (81.0, 111.9) |
| hybrid_or | 8 | 10 | 187.2 (146.2, 301.3) | 109.8 (93.9, 124.9) |
| hybrid_or | 16 | 10 | 199.1 (157.9, 265.2) | 104.2 (77.1, 123.1) |
| hybrid_or | 32 | 10 | 293.0 (199.8, 407.7) | 137.6 (106.0, 179.6) |
| classical_or | 4 | 10 | 11.7 (9.4, 19.6) | 9.6 (9.6, 9.7) |
| classical_or | 8 | 10 | 18.5 (9.4, 49.7) | 9.6 (9.6, 9.6) |
| classical_or | 16 | 10 | 16.5 (9.4, 243.7) | 9.6 (9.6, 27.6) |
| classical_or | 32 | 10 | 10.2 (10.0, 28.7) | 10.2 (10.2, 10.2) |

**Reading.**

- **`hybrid_or` rises with R** on both metrics (roughly 179→187→199→293 greedy,
  95→110→104→138 final) - directionally consistent with the paper's "OR helps
  the hybrid" finding, though the effect is not perfectly monotone at R=16
  (both metrics dip slightly below R=8) and every CI at adjacent R values
  overlaps heavily. Read as "rises over the full range", not "rises at every
  step".
- **`classical_or` (the paper's own design, transcribed as-is) does not merely
  underperform - it is dead**, final_performance IQM 9.6-10.2 at every R,
  indistinguishable from the ~9.6 degenerate-policy floor documented since
  exp01. The `greedy_best` column looks noisier (up to IQM 18.5, mean 108.1 at
  R=16) only because it is a maximum over a training curve that spikes before
  collapsing - see the bimodal-collapse evidence in
  `docs/CORRECTIONS.md#fix-09`: 9 of `classical_OR16`'s 10 seeds show the exact
  transient-peak-then-collapse signature of `paper_linear` in exp01, and the
  10th (a genuine outlier that solves CartPole) is what drags the mean to
  108.1 while IQM correctly reports 16.5.
- **exp02 cannot yet say whether OR's benefit is quantum-specific** from
  `hybrid_or`/`classical_or` alone. The contrast the paper draws its conclusion
  from is "OR helps hybrid AND NOT classical" - a live-vs-live comparison. Here
  it was live-vs-dead, exactly the confound NEW-02 was built to remove from
  exp01. `classical_or_matched_r{R}` (NEW-02's recipe applied per-R) closes
  that gap - see below.

#### classical_or_matched_r{R} - coverage (n=3), completed 2026-08-21

The capacity-matched, full-observation control, run through `RunSpec`/`run_grid`
(correct `spec.arm` and `git_revision` from the start - no migration needed).
3 seeds, 100k steps, matching the rest of exp02.

| R | n | greedy_best IQM (CI) | final_performance IQM (CI) |
|---|---|---|---|
| 4 | 3 | 27.4 (13.4, 39.0) | 17.5 (13.7, 20.1) |
| 8 | 3 | 36.3 (20.6, 64.4) | 20.7 (19.0, 23.8) |
| 16 | 3 | 59.5 (40.4, 83.2) | 29.5 (26.0, 33.2) |
| 32 | 3 | 121.2 (64.2, 193.8) | 38.8 (24.4, 55.8) |

Three-way read against `hybrid_or` and the unmatched `classical_or`
(`greedy_best` IQM; `P(matched > unmatched)` from
`stats.probability_of_improvement`). **N differs across the row - stated
explicitly because it matters for what can and cannot be read off this table:**

| R | hybrid_or (n=10) | classical_or unmatched (n=10) | classical_or_matched (n=3) | P(matched > unmatched) |
|---|---|---|---|---|
| 4 | 178.8 | 11.7 | 27.4 | 0.87 |
| 8 | 187.2 | 18.5 | 36.3 | 0.73 |
| 16 | 199.1 | 16.5 | 59.5 | 0.77 |
| 32 | 293.0 | 10.2 | 121.2 | 0.97 |

**Reading - corrected 2026-08-21 (later same day) after inspecting per-episode
tail data, not just the summary means below. The first version of this
reading called the matched control "alive... rising monotonically" - too
optimistic. What the tail episodes actually show, per seed, for R=4:**

```
last 10% of episodes, n>50 (a return well above the ~9.6-13 dead floor):
  s1: n>50 = 0/763    - flat, never leaves the dead floor
  s2: n>50 = 5/668     - occasional, mild bumps (13-25 range)
  s3: n>50 = 25/732   - more frequent moderate elevation (15-53), still oscillating
```

**One seed of three never leaves the degenerate floor at all; the other two
show intermittent, unstable bumps, not sustained learning** - and the pattern
repeats at every R: `classical_or_matched_r32`'s s2, for instance, has a
late-training stretch around 200-300 (a partial, unstable near-solve) while s3
stays flat around 25-35 - the same kind of single-seed-driven inflation
documented for the UNMATCHED control's `classical_OR16` above, just less
extreme. **A second, related correction: at n=3, `stats.iqm` degrades to the
plain mean** (its own docstring: trimming 25% from each side of three values
discards everything), so the "IQM" column above carries none of the
outlier-robustness the notebook and this file rely on elsewhere - it is a
mean of three points, one of which is often the one doing the intermittent
climbing.

- **What this data DOES support:** the matched control clears the strict
  degenerate floor in most (not all) seeds and does better than the unmatched
  control, which is uniformly dead. "Less dead than the unmatched control" is
  the accurate claim.
- **What it does NOT support:** "alive", "learning", or "rises monotonically
  with R" - the per-episode pattern is intermittent and seed-dependent, not a
  stable climb, and one seed per R shows no elevation at all. Any rise in the
  R=4->32 IQM figures should be read as "the mean including whichever seed
  spiked that round", not as a population-level trend, until more seeds exist
  and IQM can do the trimming it is meant to do (n >= 4).
- **The hybrid-vs-matched comparison across R is not yet readable for shape**
  (see below) for the same n=3-vs-n=10 reason as before - only "substantially
  below the hybrid, at every R measured so far" is supported.
- **n=3 - coverage only**, and the next candidate for the robustness pass (see
  the dated plan note below) - specifically because unlike `classical_or`, it
  is no longer uniformly dead, so more seeds here would let IQM start doing
  its job rather than repeat a null reading.
- ROBUSTNESS (B) is done for `hybrid_or`/`classical_or` (n=10). It is NOT yet
  done for `classical_or_matched_r{R}` (n=3, today) - and see the open design
  question below before that pass is run.

### UPDATE 2026-08-21 (later same day) - exp01's `matched_classical` is architecturally SMALLER than exp02's `classical_or_matched_r4`, not the same control at a different R

Asked directly: if capacity-matching resurrected exp02's control, why does
exp01's `matched_classical` (also capacity-matched, also full observation) stay
dead? Checked architecture-to-architecture rather than assumed:

| | `matched_classical` (exp01) | `classical_or_matched_r4` (exp02) |
|---|---|---|
| input policy | `reuse_indices=[0,1,2,3]`, `n_repeats=4` (in_dim=16) | identical |
| recipe | `Linear(16,w) -> ReLU -> Linear(w,2)` | identical |
| matched to | `hybrid_fig4` total = **126** | `hybrid_or_config(4)` total = **222** |
| hidden width | **7** | **12** |
| total params | **135** | **230** |
| DQN kwargs, steps | batch 128 / buffer 10000 / tf 10, 100k | identical |

**They are not the same control at a different R - they are matched against
different reference budgets, because `hybrid_or_config(4)`'s own total (222)
is already ~1.8x `HYBRID_FIG4`'s (126) before any larger R is considered.**
Output Reuse replicates the PQC's readout before the classical head, so even
at the smallest swept value (R=4) the head alone grows from 46 to 142 params
(quantum stays fixed at 80). `HYBRID_FIG4` has no OR applied at all - it is
not "R=1" of the sweep, it is a different config outside it - so there was
never a shared reference budget to match both controls to in the first place.

**Per-seed curves (all 10 `matched_classical` seeds vs all 3
`classical_or_matched_r4` seeds, both `fix01=on`, 100k steps):**

- `matched_classical` (width 7): all ten seeds show the SAME shape - a modest
  peak (69-127) followed by decay to late-training means of 9.5-13.3. This is
  uniform across all ten seeds, not the bimodal 9-collapse/1-solve pattern
  `classical_OR16` (the UNMATCHED control) showed - it is a milder, more
  consistent decay, not a dramatic collapse.
- `classical_or_matched_r4` (width 12, n=3): similar peaks (79-107) but
  noticeably less late-training decay - late means of 13.7-20.1 vs
  `matched_classical`'s 9.5-13.3. Correction after inspecting the actual
  per-episode tail (not just this summary mean): the improvement is not
  sustained learning either - one seed (s1) never leaves the dead floor at
  all (0/763 tail episodes above 50), the other two show intermittent,
  unstable bumps rather than a stable plateau. "Decays less severely" is the
  accurate claim; neither arm is close to its own peak by the end of training.

**Reading.** The two "matched" controls differing in outcome is consistent
with them differing in capacity (135 vs 230 params, width 7 vs 12) - a
confound this repository exists to avoid, now found in its own follow-up
correction. Neither control is healthy (neither approaches `oversized_mlp`'s
~200-258 final-performance range), so this is not "capacity alone resurrects
the classical arm" - but the milder decay at the larger width is directionally
consistent with capacity mattering, on top of whatever the observation-fairness
fix already contributes. At n=3 for the wider arm this is suggestive, not
established; it is exactly the kind of question the robustness pass on
`classical_or_matched_r{R}` (see plan below) will sharpen, and it means
`matched_classical` (exp01) and `classical_or_matched_r{R}` (exp02) should be
read as separate, non-interchangeable controls rather than as one design
applied twice.

### OPEN QUESTION, raised and not yet resolved: which reference budget?

No `docs/EXPERIMENT-02.md` exists, and neither the exp02 script's docstring nor
the RESULTS-LOG amendment above specifies a capacity-matched control recipe -
they describe a DIFFERENT control (`classical_OR{R}`, which applies OR's own
"repeat the observation R times" mechanism to a classical net, testing whether
OR is a general scaling trick rather than quantum-specific). `classical_or_matched_r{R}`
is a fresh NEW-02-style addition, so the reference budget it targets was not
already settled by prior documentation - stated explicitly rather than assumed.

Two candidate references answer different questions:

- **(a) `hybrid_or_config(R)`'s own total** (what is currently implemented) -
  "does the circuit help at equal total budget, where the budget is whatever
  OR gives the hybrid AT THIS R"? Consistent with the precedent already set
  by `capacity.py`/`EXPERIMENT-01.md` section 5 ("same total budget as the
  model it is meant to rival") and by `frozen_matched_scalar` in exp04
  (matched to a specific reference hybrid *of that experiment*, not a
  fixed/foreign one).
- **(b) `HYBRID_FIG4`'s total alone** (126, no OR) - "does the circuit help at
  a FIXED classical capacity, isolating what OR's head-inflation contributes
  separately"? This would leave the classical control's budget flat while the
  real hybrid arm's budget keeps growing with R - reintroducing a capacity gap
  that WIDENS with R, which is the confound NEW-02 exists to remove, not a way
  to isolate it further.

(a) is the current implementation and is consistent with established
precedent. (b) answers a real but different question (closer to
`frozen_scalar_mlp_large`'s role in exp04: separating capacity from
mechanism).

**RESOLVED 2026-08-21: (a) confirmed as the primary control - no change, no
relaunch needed for it.** (b) is registered as a deferred, separately-named
ablation for later - `classical_or_matched_fixed126` (fixed budget = 126,
`HYBRID_FIG4`'s bare total, not growing with R) - not implemented, not
scheduled, just recorded so the idea is not lost.

### PLAN, confirmed and NOT yet launched

`classical_or_matched_r{R}` is the cleanest live-vs-live signal in the
repository (see the reading above) and is already flagged as the next
robustness candidate. **Agreed order once exp04b finishes and frees compute:
`classical_or_matched_r{R}` to 8-10 seeds is the next thing run, ahead of any
other pending item.** Not launched yet - awaiting explicit go-ahead.

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

### UPDATE 2026-08-21 - ROBUSTNESS DONE (n=10), and exp03b alongside it

Both exp03 (`ent=True`) and exp03b (`ent=False`) found already computed at
n=10, 100k steps, on the synced Drive tree - not previously transcribed. This
is the pair FIX-07 named as the one that matters most: exp03's depth axis
confounds reuploading with entanglement (Skolik template, final layer's CZ ring
is inert against a PauliZ readout - CORRECTIONS.md#fix-07), and exp03b is the
same sweep with `ent=False`, where every depth has zero effective entangling
blocks. Both metrics reported, IQM with 95% bootstrap CI, per the
STATISTICS.md checklist.

| L | ent | n | greedy_best IQM (CI) | final_performance IQM (CI) |
|---|---|---|---|---|
| 1 | True  | 10 | 16.6 (11.3, 25.2) | 10.1 (9.9, 10.5) |
| 1 | False | 10 | 23.6 (13.8, 38.7) | 10.0 (9.7, 11.7) |
| 2 | True  | 10 | 55.5 (42.3, 75.6) | 30.2 (23.8, 37.2) |
| 2 | False | 10 | 60.5 (46.0, 94.7) | 34.5 (28.7, 41.1) |
| 5 | True  | 10 | 121.3 (60.6, 189.7) | 37.7 (24.6, 49.7) |
| 5 | False | 10 | 85.9 (55.9, 196.1) | 41.8 (38.2, 50.6) |

P(a random exp03 [ent=True] run > a random exp03b [ent=False] run), on
`greedy_best`, n=10 vs n=10: **L=1: 0.39, L=2: 0.42, L=5: 0.54.**

**Reading - this changes two things at once, in different directions.**

- **The FIX-07 question is answered, and cleanly: entanglement is not what was
  driving the depth effect.** `ent=False` performs at least as well as
  `ent=True` at every depth (P(True>False) <= 0.54 everywhere, and below 0.5 at
  L=1 and L=2 - i.e. removing entanglement did not cost anything, if anything
  it edged ahead). That is the answer exp03b exists to give, and it is good
  news for the "DR transfers" claim's mechanism: the monotone rise with depth
  is not an artefact of entanglement quietly coming online, because the
  no-entanglement curve rises just as much on its own.
- **But the honest metric shrinks the headline considerably.** The original
  n=3 report (this file, above) was `greedy_best` 15 -> 35 -> 199, described as
  "DR transfers to DQN, clearly" and ">10x jump". At n=10 on the same biased
  statistic the rise is more modest (16.6 -> 55.5 -> 121.3) and n=3's L=5 figure
  of ~199 turns out to have been pulled up by exactly the kind of high-variance
  seed IQM exists to discount (the n=3 set included the 408 outlier at face
  value). On `final_performance` - the metric that does not reward a noisier
  arm - the rise is smaller still: 10.1 -> 30.2 -> 37.7 (ent=True) and
  10.0 -> 34.5 -> 41.8 (ent=False). **L=1 sits at the ~9.6-10 degenerate floor
  on both entanglement settings** - it was never really "learning weakly", it
  was dead, matching the reading already on record for L=1 against the paper's
  own Skolik_8Q_L1 baseline.
- **The defensible claim, updated:** DR (not entanglement) drives a real,
  monotone improvement from L=1 to L=5 under DQN, confirmed at n=10 and with an
  unbiased metric - but the improvement is from "dead" to "modestly alive"
  (final IQM ~38-42), not from "dead" to "near-solved". The "greedy 408, near
  CartPole max 500" seed from the n=3 pass was the single most extreme draw in
  what is now a 10-seed sample, not a representative outcome.
- ROBUSTNESS (B) for this pair is now DONE (n=10, both entanglement settings).
  This is the most complete, best-behaved dataset in the repository - no
  metadata issues, no missing cells, both arms of the comparison it was
  designed to resolve present at full depth.

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

### UPDATE 2026-08-21 - STAGE 1 DONE (n=10), GATE PASSES; H4 warning CONFIRMED with real power

Found already computed at n=10, 100k steps, on the synced Drive tree. Success
rate is a 100-episode rolling mean (the return is a single bit, so this IS the
success rate); `greedy_best` here is the greedy-policy success rate over 20
eval episodes, same 0-1 scale. IQM with 95% bootstrap CI.

| arm | FIX-01 | n | success IQM (CI) | greedy IQM (CI) | phantom frac |
|---|---|---|---|---|---|
| frozen_onehot_mlp | off | 10 | 1.000 (0.998, 1.000) | 1.0 (1.0, 1.0) | 0.1323 |
| frozen_onehot_mlp | on | 10 | 1.000 (0.998, 1.000) | 1.0 (1.0, 1.0) | 0.1321 |
| frozen_scalar_mlp_large | off | 10 | 0.448 (0.318, 0.590) | 0.0 (0.0, 0.2) | 0.0456 |
| frozen_scalar_mlp_large | on | 10 | 0.362 (0.322, 0.438) | 0.0 (0.0, 0.0) | 0.0471 |
| frozen_matched_scalar | off | 10 | 0.055 (0.040, 0.072) | 0.0 (0.0, 0.0) | 0.0365 |
| frozen_matched_scalar | on | 10 | 0.052 (0.037, 0.087) | 0.0 (0.0, 0.0) | 0.0383 |

**GATE: PASSES, comfortably.** `frozen_onehot_mlp` reaches success 1.000 at
every seed (min observed 0.99) - the regime is unambiguously alive, and stage 2
does not need stage 1 repeated.

**H4 (fair scalar control) - the smoke-run warning is CONFIRMED, not merely
repeated, now with n=10 instead of a single seed:** `frozen_matched_scalar`
sits at 0.052-0.055, statistically indistinguishable from the ~0.015 random
floor. A capacity-matched classical control fed the raw state index cannot
learn FrozenLake in this stack - the encoding, not the budget, is the
bottleneck for this arm. See the dated update in `docs/EXPERIMENT-04.md`
section 5 for what this changes about stage 2's primary comparison.

**H3 (FIX-01 where the phantom fraction is large) - measured, and NOT
confirmed in the predicted direction.** The only live, informative arm is
`frozen_scalar_mlp_large`: success IQM 0.448 (off) vs 0.362 (on) - the CIs
barely overlap, and the direction is opposite to what FIX-01's mechanism
predicts (removing the phantom transition should help or be neutral, not
hurt). `final_performance`-style checks (last-10%-of-episodes, matching
STATISTICS.md's recommended statistic) confirm this is not a metric artefact:
`P(fix01=on success > fix01=off success)` computed directly from per-seed
`final_performance` scores is **0.39** - below 0.5 on both the max-based and
final-performance readings. `frozen_matched_scalar`'s FIX-01 delta (0.055 vs
0.052) is within noise and uninformative on a dead arm, as expected.
**H3 is not confirmed at n=10; if anything the one arm where it is measurable
points mildly the other way, though the CIs are wide enough that this should
be read as "not confirmed", not as "disconfirmed".** This tempers a load-bearing
part of exp04's own motivation (section 1b) and should be written up as such,
not smoothed over.

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

### UPDATE 2026-08-21 - stage 2 status before exp04b

Found on Drive before launching exp04b: `frozen_scalar_1q_L1` (Config A, L=1)
already had 8 of 10 seeds (3-10), FIX-01 on. At n=8:

| config | n_qubits | DR depth | n | success IQM | greedy IQM |
|---|---|---|---|---|---|
| A scalar-to-phase | 1 | 1 | 8 | 0.050 | ~0.0 |

**Same reading as `frozen_matched_scalar` above: chance level.** See the dated
update in `docs/EXPERIMENT-04.md` section 5 - Config A at L=1 and its
capacity-matched classical control are statistically indistinguishable from
each other and from a random policy. This does not resolve whether depth
rescues the encoding (that is what the rest of stage 2 tests); it does mean
"circuit vs classical" is not the live question at L=1.

L=5/10/15 (Config A) and all of Config B (4 qubits, L=1/5) had zero cells
before today. Launched as "exp04b": full stage-2 grid, 3-seed coverage per
`docs/EXPERIMENT-04.md` section 4's own staging ("Stage 2. Hybrid sweeps,
FIX-01 on, 3 seeds coverage, then 8-10 robustness") - not skipped ahead to
robustness. This run also fills `frozen_scalar_1q_L1`'s two missing seeds
(1, 2) as a side effect of requesting seeds 1-3 with the reuse guard active.
Results pending; this section will be updated on completion.

