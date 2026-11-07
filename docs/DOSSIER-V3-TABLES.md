# Regenerating the tables of dossier v3 (exp01-exp08)

`scripts/build_dossier_v3.py --results-root <results> --out <dir>` writes `per_seed_metrics.csv`, `summary_by_arm.csv`,
`curves_eval_greedy.csv` and `curves_train_ma50.csv` from the manifests and per-run CSVs. The original scripts of 2026-11-01 were
scratch files of an earlier session and were never committed (they were found in that session's scratchpad: `brief_gen.py`,
`_assemble.py`, `_curves.py`); this script reproduces their logic. Nothing was adjusted to make numbers agree.

`scripts/compare_dossier_tables.py --new <dir> --old <dir>` reports row counts, arms and the maximum absolute difference per column.

Check against the delivered package v3.3 (run on the results folder as it is today):

| file | rows new / delivered | arms | max abs. difference |
|---|---|---|---|
| `per_seed_metrics.csv` | 888 / 888 | 80 / 80 | steps, best_ma50, final_ma50, phantom: 0; final 5.7e-14; greedy 2.8e-14; wall_h 3.6e-15 (float text round-off) |
| `summary_by_arm.csv` | 93 / 93 | 80 / 80 | 0 in every column |
| `curves_eval_greedy.csv` | 8 808 / 8 808 | 80 / 80 | 0 |
| `curves_train_ma50.csv` | 88 080 / 88 080 | 80 / 80 | 0 |

Same rows, same columns, same arms. The 25 legacy manifests of exp02 (16) and exp03 (9) that carry no `fix_autoreset` field take their FIX-01
flag from `outcome.probe.fix01_active` (all 25: `True`, with `phantoms_seen > 0` and `phantoms_stored = 0`), which is how the original
script labelled them (`on`).
