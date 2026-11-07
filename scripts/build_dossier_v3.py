"""Rebuild the data tables of the results dossier v3 (exp01-exp08, real circuit + classical arms) from the manifests.

Recovered logic. The 2026-11-01 tables (handoff_memoria/per_seed_metrics.csv, summary_by_arm.csv, curves) were produced by
scratch scripts of an earlier session that were never committed (brief_gen.py: per-seed table; _assemble.py: removal of simulator
arms and per-arm summary; _curves.py: curves). They are reproduced here with the same filters and definitions:

  * folders: exp01, exp01 (60k folder), exp02, exp03, exp03b, exp04, exp07, exp08 under <results-root>;
  * one row per manifest that has `spec` and `outcome`, without `error`, and whose episode CSV can be found;
  * arm label = run name without the seed (and tag) suffix and without the `__fix01on/off` suffix; tag appended if present;
  * FIX-01 flag = spec.fix_autoreset, else manifest.fix_autoreset, else (legacy manifests) outcome.probe.fix01_active;
  * final = stats.final_performance(training returns, last 10 %); greedy = stats.greedy_final(greedy checkpoints, last 3);
    best_ma50 / final_ma50 from the 50-episode moving average; wall_h = outcome.wall_seconds / 3600; phantom = probe.frac_poison;
  * simulator arms (su2_*, sv_*) are dropped (docs/SIMULATED-RESULTS-POLICY.md);
  * summary: per (exp, arm, fix, tag): IQM and 95 % bootstrap CI (stats.bootstrap_ci, default seed 0, 10 000 resamples).

    python scripts/build_dossier_v3.py --results-root <results> --out <dir>
"""
import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd

from qrl_dissection.core import analysis, stats

DIRS = {
    "exp01": "exp01_dqn_cartpole_capacity",
    "exp01_60k": "exp01_dqn_cartpole_capacity_60k_20260805",
    "exp02": "exp02_dqn_cartpole_output_reuse",
    "exp03": "exp03_dqn_cartpole_data_reuploading",
    "exp03b": "exp03b_dqn_cartpole_dr_unentangled",
    "exp04": "exp04_dqn_frozenlake_embeddings",
    "exp07": "exp07_dqn_cartpole_paper_replication",
    "exp08": "exp08_dqn_cartpole_frozenlake_followups",
}


def local(p, d):
    if not p:
        return None
    name = pathlib.Path(str(p).replace("\\", "/")).name
    for q in (d / "runs" / name, d / name):
        if q.exists():
            return q
    return None


def label_and_fix(m, mp):
    sp, oc = m["spec"], m["outcome"]
    rn = oc.get("run_name") or m.get("run_name") or m.get("name") or mp.stem
    tag = sp.get("tag", "")
    label = re.sub(r"__fix01(on|off)$", "", re.sub(r"__s\d+(__\w+)?$", "", rn))
    if tag:
        label = f"{label}__{tag}"
    fx = sp.get("fix_autoreset", m.get("fix_autoreset"))
    if fx is None:
        fx = (oc.get("probe") or {}).get("fix01_active")
    return label, ("on" if fx else "off"), tag


def build(results_root):
    base = pathlib.Path(results_root)
    recs, ev_rows, tr_rows = [], [], []
    for key, dn in DIRS.items():
        d = base / dn
        for mp in sorted(d.glob("*.manifest.json")):
            try:
                m = json.loads(mp.read_text())
            except Exception:
                continue
            if "error" in m or "spec" not in m or "outcome" not in m:
                continue
            sp, oc = m["spec"], m["outcome"]
            ep = local(oc.get("episodes_csv"), d)
            if ep is None:
                continue
            rew, steps = analysis.load_episodes(ep)
            ma50 = analysis.moving_average(rew, 50)
            evp = local(oc.get("eval_csv"), d)
            gf, es, sc = None, [], []
            if evp is not None:
                es, sc = analysis.load_eval(evp)
                if len(sc):
                    gf = stats.greedy_final(sc)
            label, fx, tag = label_and_fix(m, mp)
            if label.startswith(("su2_", "sv_")):
                continue
            recs.append(dict(exp=key, arm=label, fix=fx, tag=sp.get("tag", ""), steps=sp["total_timesteps"], seed=sp["seed"],
                             final=stats.final_performance(rew, 0.1), greedy=gf, best_ma50=float(np.nanmax(ma50)),
                             final_ma50=float(ma50[-1]), wall_h=(oc.get("wall_seconds") or 0) / 3600.0,
                             phantom=(oc.get("probe") or {}).get("frac_poison")))
            ma = np.array([rew[max(0, i - 49): i + 1].mean() for i in range(len(rew))])
            for edge in range(1000, int(sp["total_timesteps"]) + 1, 1000):
                idx = np.searchsorted(steps, edge, side="right") - 1
                if idx >= 0:
                    tr_rows.append((key, label, fx, tag, sp["seed"], edge, round(float(ma[idx]), 3)))
            for s_, v_ in zip(es, sc):
                ev_rows.append((key, label, fx, tag, sp["seed"], int(s_), round(float(v_), 3)))
    df = pd.DataFrame(recs).sort_values(["exp", "arm", "fix", "seed"])
    df["tag"] = df["tag"].fillna("")
    rows = []
    for (exp, arm, fx, tag), g in df.groupby(["exp", "arm", "fix", "tag"]):
        f, gr = g.final.dropna().tolist(), g.greedy.dropna().tolist()
        flo, fhi = stats.bootstrap_ci(f) if len(f) > 1 else (np.nan, np.nan)
        glo, ghi = stats.bootstrap_ci(gr) if len(gr) > 1 else (np.nan, np.nan)
        rows.append(dict(exp=exp, arm=arm, fix01=fx, tag=tag, n=len(g), final_perf_iqm=stats.iqm(f), final_perf_ci_lo=flo,
                         final_perf_ci_hi=fhi, greedy_final_iqm=stats.iqm(gr) if gr else np.nan, greedy_ci_lo=glo, greedy_ci_hi=ghi,
                         best_ma50_mean=g.best_ma50.mean(), final_ma50_mean=g.final_ma50.mean(),
                         seeds_with_greedy_gt0=int((g.greedy > 0).sum()), wall_h_mean=g.wall_h.mean()))
    summ = pd.DataFrame(rows).round(4)
    cols = ["exp", "arm", "fix", "tag", "seed", "global_step"]
    ev = pd.DataFrame(ev_rows, columns=cols + ["greedy_return"])
    tr = pd.DataFrame(tr_rows, columns=cols + ["train_ma50"])
    return df, summ, ev, tr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, summ, ev, tr = build(a.results_root)
    df.to_csv(out / "per_seed_metrics.csv", index=False)
    summ.to_csv(out / "summary_by_arm.csv", index=False)
    ev.to_csv(out / "curves_eval_greedy.csv", index=False)
    tr.to_csv(out / "curves_train_ma50.csv", index=False)
    print(len(df), "per-seed rows;", len(summ), "arm groups;", len(ev), "eval rows;", len(tr), "train rows")


if __name__ == "__main__":
    main()
