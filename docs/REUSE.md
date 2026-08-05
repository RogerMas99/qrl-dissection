# What gets saved, and what can be reused

Written because "resumable" was doing too much work in earlier descriptions, and
because the guarantee it implied did not hold until FIX-08.

## What one cell writes

A *cell* is one (arm, seed, FIX-01 state, tag) combination. Running it produces
four files:

| file | content | size |
|---|---|---|
| `<run>.manifest.json` | the full spec, the outcome, agent config, `env_id`, **git revision**, python version | ~1 KB |
| `runs/<run>.csv` | every episode: step, return, length | ~200 KB |
| `runs/<run>_eval.csv` | greedy evaluation at each checkpoint | ~1 KB |
| `<run>_trace.npy` | the FIX-01 probe trace, per-step phantom flags | ~100 KB |

The git revision matters more than it looks: it is what lets you tell, months
later, whether a number was produced before or after a correction landed.

**Model weights are not saved.** That is a deliberate limit, discussed below.

## Three kinds of reuse, and which ones exist

**1. Cell-level reuse — yes, and it is the one that matters.**
A finished cell is never recomputed. Interrupt a grid at any point, rerun the
same command, and only the missing cells run. This is what makes
`--budget-minutes` safe and what makes a robustness pass cheap: seeds 1–3 from
the coverage pass are simply skipped, so going from 3 to 10 seeds costs seven
runs per cell, not ten.

**2. Analysis-level reuse — yes, unlimited.**
The episode CSVs hold every episode, so any metric can be recomputed without
retraining: a different moving-average window, steps-to-threshold instead of
best return, a different aggregation across seeds. `20_dqn_results.ipynb` never
trains anything. If you decide next month that the right statistic was
steps-to-200 rather than best MA-50, that is a re-read, not a re-run.

**3. Mid-run continuation — no, and deliberately not.**
You cannot take a finished 60k run and extend it to 100k. Adding that would mean
persisting the replay buffer, the optimiser state, the epsilon-schedule position
and the RNG streams. Persist anything less and the resumed run is *not* the run
you would have got uninterrupted — a cold buffer after a restart changes the
learning dynamics — while still being labelled as if it were. Given that this
repository exists because of silently wrong off-policy behaviour, shipping a
resume that quietly alters the algorithm would be an odd thing to do.

So: choose the step budget before starting a grid, not during.

## FIX-08 — the guard that makes (1) trustworthy

`run_name` is `arm__fix01{on,off}__s{seed}[__tag]`. Readable, and an **incomplete
key**: it does not encode the step budget, the batch size, the buffer size or the
environment. The original skip logic was `if manifest.exists(): reuse`, so:

```python
run_arm(RunSpec(arm="oversized_mlp", seed=1, total_timesteps=1_500,  ...))   # smoke
run_arm(RunSpec(arm="oversized_mlp", seed=1, total_timesteps=100_000, ...))  # real
# -> second call returned the FIRST run's manifest. Silently. Different batch
#    size, different buffer, 1.5k steps reported as a 100k result.
```

Now every skip compares the stored spec against the requested one across
`arm`, `seed`, `fix_autoreset`, `total_timesteps`, `dqn_kwargs`, `tag` and
`env_id`, and a mismatch raises with the offending fields named:

```
oversized_mlp__fix01on__s1.manifest.json exists but was produced by a different run:
    total_timesteps: on disk 1500 != requested 100000
    dqn_kwargs: on disk {'batch_size': 32, ...} != requested {'batch_size': 128, ...}
  Reusing it would report one experiment's numbers as another's.
```

`--smoke` now writes to `<outdir>/_smoke`, so the ordinary workflow — smoke, then
the real pass — does not trip the guard. `tests/test_reuse_guard.py` pins all of
it, including the symmetric case: a *longer* finished run does not satisfy a
shorter request either, because every metric here is computed over the whole
trace.

## Runs made before FIX-08 — are they still usable?

Yes, and the answer has two halves.

**The data is there.** `SafeDQN` chdirs into its `outdir` before training, so
upstream writes `runs/<run>.csv` inside it, and every exploratory notebook pointed
`RESULTS` at `/content/drive/MyDrive/tfm_qrl/expNN`. So the full episode CSVs,
the eval CSVs and the probe traces for exp01, exp02 and exp03 are on Drive, not
in the repository - `results/` and `runs/` are gitignored on purpose. Nothing was
lost; it was simply never committed.

**The manifests are thinner.** The early scripts wrote
`{name, seed, fix_autoreset, outcome, config}` - no `total_timesteps`, no
`dqn_kwargs`. The FIX-08 guard cannot check a field that was never recorded, so
without help it would refuse them and recompute hours of finished simulation.

So the guard distinguishes two cases:

| case | meaning | what happens |
|---|---|---|
| field **differs** | definitely a different run | raises, names the field |
| field **absent** | legacy manifest, unverifiable | raises with a *different* message pointing at the migration |

and `total_timesteps` is recovered from `outcome.total_timesteps` before either
branch is taken, so a genuine budget mismatch is still caught in a legacy file
rather than excused.

### Migrating

```bash
python scripts/migrate_manifests.py /content/drive/MyDrive/tfm_qrl/exp03 --dry-run
python scripts/migrate_manifests.py /content/drive/MyDrive/tfm_qrl/exp03 \
    --arm hybrid_fig4 \
    --dqn-kwargs '{"batch_size":128,"buffer_size":10000,"train_frequency":10}'
```

Recovered automatically: `total_timesteps` (from the outcome, or the last
`global_step` in the CSV), `seed`, `fix_autoreset`, `arm` and `tag` from the run
name. Backups are written as `*.manifest.json.bak`.

Not recoverable from any artefact: `dqn_kwargs`. Supply it from the script or
notebook that produced the runs - **not from memory**, because whatever you write
becomes the value every future run is checked against. exp03 used
`{"batch_size":128,"buffer_size":10000,"train_frequency":10}`; exp02's values are
in its script.

### What you can do with them once migrated

Everything in categories 1 and 2 above. In particular: exp03's episode CSVs let
you recompute the metric without retraining, which matters because the FIX-07
amendment suggests the interesting statistic there may be steps-to-threshold
rather than best return. That is a re-read of files you already have.

## Sizing a session when one cell runs for hours

`--budget-minutes` is a wall clock checked **between** cells, never during one.
That is deliberate: terminating a cell mid-training loses all of its compute,
because the manifest is only written on completion. The consequence is that a
session overshoots by up to one cell - fine when cells are minutes, useless when
a cell is nine hours and the runtime disconnects at twelve.

Use `--max-cells N` instead. It stops after N cells complete, deterministically,
and skipped cells do not count against the allowance. `--plan` reports the median
wall time per cell measured from the manifests already on this machine, so after
one cell of each experiment you can size the rest honestly:

```
experiment   done  target  per cell  remaining  what
exp03b          4       6       12s        24s  exp03 with ent=False ...
estimated time remaining (measured on THIS machine): 24s
```

Do not trust a throughput figure quoted from someone else's hardware, including
any in these documents. PQC simulation speed varies several-fold between a Colab
CPU runtime, a GPU runtime and a laptop, and it depends strongly on
`train_frequency`, `n_qubits` and `learning_starts` - a short probe that never
reaches `learning_starts` measures the environment loop and no gradient steps at
all, which overstates throughput badly.

## Practical consequences

- **Changing hyper-parameters means a new directory or a new tag.** The guard
  will tell you; it will not guess.
- **Re-running after a correction lands** requires deleting the affected
  manifests. Check `git_revision` in the manifest to find them.
- **Copying results between machines works**: the manifests carry absolute paths
  to their CSVs, so if you move a results tree, rerun the analysis notebook from
  the new location rather than trusting stale paths.
