# Notebooks

Three are canonical. Everything else is history, kept for provenance.

| notebook | role |
|---|---|
| `00_fix05_verification.ipynb` | **Evidence.** Reproduces FIX-05 from scratch, standalone — needs nothing from this repo. Cite it in the memoria; it is the artefact that proves the correction rather than asserting it. |
| `10_dqn_suite_runner.ipynb` | **The run.** All five DQN grids, resumable, seeds as a flag. Coverage pass, then robustness pass. |
| `20_dqn_results.ipynb` | **The results.** Every final table and figure, with the paper's own 10-seed PPO curves overlaid. Trains nothing. |

## exploratory/

Six notebooks from building the experiments: single-arm smoke runs, capacity
matching sanity checks, the first FrozenLake driver. They are superseded by the
three above and are **not** how final results should be produced — several use
short budgets or single seeds and would silently give you a coverage-grade number
where a robustness-grade one is needed.

They are kept rather than deleted because they document how the corrections were
found, and `01c_fair_matched.ipynb` in particular is where the cart-position
confound (NEW-02) surfaced. That history is worth more than a tidy directory.

## Order of work

0. In `10`, run section **2b — What do I already have?** first. It inventories
   the Drive folder: how many cells finished, at what step budget, whether the
   episode CSVs survived, and whether the manifests predate the reuse guard.
   Previous sessions left real results there; know their shape before adding to
   them.
1. `00` once, to confirm the environment and the FIX-05 gate.
2. `10` with `--pass coverage` until every grid is full.
3. `20` to see whether anything separates.
4. `10` with `--pass robustness` — seeds 1–3 are skipped, so this adds seven per
   cell rather than redoing ten.
5. `20` again, then copy the tables into `docs/RESULTS-LOG.md` and commit.

### Where the files live

`drive.mount()` does not copy anything — it makes Drive appear as a folder inside
Colab, so `/content/drive/MyDrive/tfm_qrl/...` **is** Drive. Writing there is
writing to Drive, with no upload step. `/content/...` without the `drive/` prefix
works perfectly for one session and is then gone. Full explanation, including how
to get the files onto your own machine if you want them: `docs/REUSE.md`.

Three seeds is not a conclusion. The paper's own spreads put several of its
published contrasts inside one standard deviation at **ten** seeds; see
`docs/PAPER-BASELINES.md`.
