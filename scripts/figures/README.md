# Figures of chapter 4

Scripts copied from the thesis project (`scripts/` of `TFM_LaTeX_project_v175`); only the **input paths** were changed.

    python scripts/figures/figs.py    <out_dir>                             # ch4_capacity, ch4_or, ch4_depth, ch4_frozenlake, ch4_or_curves
    python scripts/figures/fig_lr.py  <out_dir> handoff_exp13/exp13_A-D     # ch4_lr
    python scripts/figures/or_identity_check.py                             # numerical check of the Output Reuse = lr x R equivalence (no figure)

Inputs (all in the repository):

| file | used for |
|---|---|
| `handoff_memoria/01_DOSSIER_RESULTADOS_v3.md` (section 3 tables), `per_seed_metrics.csv`, `curves_eval_greedy.csv`, `curves_train_ma50.csv` | exp01-exp08 arms (`data.py`; override the folder with `FIG_TABLES`) |
| `handoff_exp11/summary_exp11.csv`, `handoff_exp12/summary_exp12.csv` | matched control with replication and classical twin (fig. `ch4_or`) |
| `handoff_exp13/exp13_A-D/summary_exp13.csv` | `ch4_lr` |
| `data/paper_ppo_curves.csv.gz` | PPO curves of the reference study (fig. `ch4_depth`; override with `FIG_PAPER_CURVES`) |

The curves CSVs are produced by `scripts/build_dossier_v3.py` from the manifests in `results/` (identical to the ones of package v3.3).
The reference-study means and standard deviations hard-coded in `fig_lr.py` (263.94 ± 18.38, 320.0 ± 36.29, 381.7 ± 37.32,
330.2 ± 63.44) match `data/paper_ppo_summary.csv` (rows `Classical_r4/8/16/32`).

Checked: the six figures regenerated from the repository render pixel-identical to those in the thesis project (rendered at 110 dpi with
pdftoppm; the PDFs differ only in the Matplotlib version string of their metadata).
