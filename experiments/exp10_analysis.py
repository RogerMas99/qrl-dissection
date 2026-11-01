"""Experiment 10 - analysis: per-seed table, curves, summary and the contrast tables C1-C4.

SIMULATOR RESULTS - EXPLORATORY - NOT FOR THE FINAL WORK without the author's explicit approval
(docs/SIMULATED-RESULTS-POLICY.md). The contrasts and reading rules are the ones fixed in
docs/EXPERIMENT-10-X100.md before any cell ran.

Reads the runs written by experiments/exp10_emulator_x100.py (tag x100) plus the seeds it reused from
exp09 / exp01 (reused_seeds.json), and writes into --out:

    per_seed_x100.csv            exp, arm, fix, tag, steps, seed, final, greedy, best_ma50, final_ma50, wall_h, phantom
                                 (`exp` is the folder the run comes from: exp10 / exp09 / exp01)
    curves_x100_train_ma50.csv   50-episode moving average every 1000 steps
    curves_x100_eval_greedy.csv  greedy evaluation every 10000 steps
    summary_x100.csv             per arm: n, IQM and 95 % CI of final and greedy, alive seeds
    tables_x100.md               contrast tables C1-C4 + coherence checks (the reading is written by hand)

    python experiments/exp10_analysis.py --results-root <results> --out <dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from qrl_dissection.core import analysis, stats

HERE = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e10 = _load("exp10_emulator_x100")
b09 = _load("exp09c_battery_report")
SRC_LABEL = {"exp09_dqn_emulator_sweep": "exp09", "exp01_dqn_cartpole_capacity": "exp01", e10.SUBDIR: "exp10"}
HEAD = ("> SIMULATOR RESULTS - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL "
        "(docs/SIMULATED-RESULTS-POLICY.md). Nothing in this file is an approved use.\n")


def local(p, d):
    if not p:
        return None
    n = pathlib.Path(str(p).replace("\\", "/")).name
    for q in (d / "runs" / n, d / n):
        if q.exists():
            return q
    return None


def index_folder(root, folder):
    out = {}
    for mp in (root / folder).glob("*.manifest.json"):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        sp = m.get("spec")
        if sp and m.get("outcome"):
            out[(sp["arm"], sp["seed"], sp.get("tag", ""))] = m
    return out


def run_record(m, d, exp, arm):
    oc, sp = m["outcome"], m["spec"]
    ep = local(oc.get("episodes_csv"), d)
    if ep is None:
        return None
    rew, steps = analysis.load_episodes(ep)
    ma = np.array([rew[max(0, i - 49): i + 1].mean() for i in range(len(rew))])
    evp = local(oc.get("eval_csv"), d)
    es, sc = (analysis.load_eval(evp) if evp is not None else (np.array([]), np.array([])))
    gf = stats.greedy_final(sc) if len(sc) else None
    ma50 = analysis.moving_average(rew, 50)
    return dict(
        row=dict(exp=exp, arm=arm, fix="on", tag=sp.get("tag", ""), steps=sp["total_timesteps"], seed=sp["seed"],
                 final=stats.final_performance(rew, 0.1), greedy=gf, best_ma50=float(np.nanmax(ma50)),
                 final_ma50=float(ma50[-1]), wall_h=(oc.get("wall_seconds") or 0) / 3600.0,
                 phantom=(oc.get("probe") or {}).get("frac_poison")),
        train=[(edge, round(float(ma[idx]), 3)) for edge in range(1000, int(sp["total_timesteps"]) + 1, 1000)
               for idx in [np.searchsorted(steps, edge, side="right") - 1] if idx >= 0],
        evals=[(int(a), round(float(b), 3)) for a, b in zip(es, sc)])


def collect(root, out_dir):
    reused = {}
    rf = root / e10.SUBDIR / "reused_seeds.json"
    if rf.exists():
        reused = json.loads(rf.read_text())
    idx = {f: index_folder(root, f) for f in SRC_LABEL}
    rows, tr, ev = [], [], []
    arms = [a for g in e10.GROUPS for a, _k, _e in e10.group_arms(g)]
    for arm in dict.fromkeys(arms):
        # new runs
        for (a, seed, tag), m in idx[e10.SUBDIR].items():
            if a == arm and tag == e10.TAG:
                r = run_record(m, root / e10.SUBDIR, "exp10", arm)
                if r:
                    rows.append(r)
        # reused runs
        for seed_s, (folder, tag) in reused.get(arm, {}).items():
            m = idx.get(folder, {}).get((arm, int(seed_s), tag))
            if m:
                r = run_record(m, root / folder, SRC_LABEL[folder], arm)
                if r:
                    rows.append(r)
    per = pd.DataFrame([r["row"] for r in rows]).sort_values(["arm", "seed"]).reset_index(drop=True)
    cols = ["exp", "arm", "fix", "tag", "seed", "global_step"]
    for r in rows:
        k = [r["row"][c] for c in cols[:5]]
        tr += [k + [s, v] for s, v in r["train"]]
        ev += [k + [s, v] for s, v in r["evals"]]
    per.to_csv(out_dir / "per_seed_x100.csv", index=False)
    pd.DataFrame(tr, columns=cols + ["train_ma50"]).to_csv(out_dir / "curves_x100_train_ma50.csv", index=False)
    pd.DataFrame(ev, columns=cols + ["greedy_return"]).to_csv(out_dir / "curves_x100_eval_greedy.csv", index=False)
    return per


def ci_str(v, nd=1):
    v = [x for x in v if x is not None and not np.isnan(x)]
    if len(v) < 2:
        return "-"
    lo, hi = stats.bootstrap_ci(v)
    return f"{stats.iqm(v):.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def summary(per, out_dir):
    rows = []
    for arm, g in per.groupby("arm"):
        f, gr = g.final.dropna().tolist(), g.greedy.dropna().tolist()
        flo, fhi = stats.bootstrap_ci(f) if len(f) > 1 else (np.nan, np.nan)
        glo, ghi = stats.bootstrap_ci(gr) if len(gr) > 1 else (np.nan, np.nan)
        rows.append(dict(arm=arm, n=len(g), n_new=int((g.exp == "exp10").sum()),
                         final_perf_iqm=stats.iqm(f), final_ci_lo=flo, final_ci_hi=fhi,
                         greedy_final_iqm=stats.iqm(gr) if gr else np.nan, greedy_ci_lo=glo, greedy_ci_hi=ghi,
                         seeds_alive_greedy_gt0=int((g.greedy > 0).sum())))
    s = pd.DataFrame(rows).round(4)
    s.to_csv(out_dir / "summary_x100.csv", index=False)
    return s


def P(per, a, b, col):
    x = per[per.arm == a][col].dropna().tolist()
    y = per[per.arm == b][col].dropna().tolist()
    return (stats.probability_of_improvement(x, y) if x and y else float("nan")), len(x), len(y)


def contrast_rows(per, pairs, nd=1):
    out = ["| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | strong? |",
           "|---|---|---|---|---|---|---|"]
    for label, a, b in pairs:
        pf, na, nb = P(per, a, b, "final")
        pg, _, _ = P(per, a, b, "greedy")
        if na == 0 or nb == 0:
            out.append(f"| {label} | {na} / {nb} | (not run yet) | | | | |")
            continue
        ia = lambda c, arm: stats.iqm(per[per.arm == arm][c].dropna().tolist())          # noqa: E731
        strong = "yes" if (pf >= 0.95 and pg >= 0.95) else ("yes (reverse)" if (pf <= 0.05 and pg <= 0.05) else "no")
        out.append(f"| {label} | {na} / {nb} | {ia('final', a):.{nd}f} / {ia('final', b):.{nd}f} | {pf:.2f} | "
                   f"{ia('greedy', a):.{nd}f} / {ia('greedy', b):.{nd}f} | {pg:.2f} | {strong} |")
    return "\n".join(out)


def c1_pairs():
    nm = lambda kind, R, sfx="": __import__("qrl_dissection.core.configs", fromlist=["x"]).cartpole_replica_name(kind, R, sfx)  # noqa: E731
    prim, sec, tool = [], [], []
    for R in (1, 4, 8, 16, 32):
        prim.append((f"R={R}: sv entangled vs sv unentangled", nm("sv", R), nm("sv", R, "_noent")))
        sec.append((f"R={R}: sv entangled vs su2 unentangled", nm("sv", R), nm("su2", R)))
        tool.append((f"R={R}: sv unentangled vs su2 (same circuit, two tools)", nm("sv", R, "_noent"), nm("su2", R)))
    return prim, sec, tool


def tables(per, root):
    from qrl_dissection.core.configs import cartpole_replica_name as nm
    out = [HEAD]
    out.append("## Arms and seeds available\n")
    cnt = per.groupby("arm").agg(n=("seed", "count"), n_new=("exp", lambda s: int((s == "exp10").sum())),
                                 n_reused=("exp", lambda s: int((s != "exp10").sum())))
    out.append("| arm | n | new (exp10) | reused (exp09 / exp01) |\n|---|---|---|---|")
    out += [f"| `{a}` | {r.n} | {r.n_new} | {r.n_reused} |" for a, r in cnt.iterrows()]
    prim, sec, tool = c1_pairs()
    out += ["", "## C1 - entanglement x Output Reuse (CartPole, L=5)\n", "Primary (same tool on both sides):\n",
            contrast_rows(per, prim), "", "Secondary (against the SU(2) emulator):\n", contrast_rows(per, sec), "",
            "Tool agreement (sv ent=False against su2, same circuit):\n", contrast_rows(per, tool)]
    c2 = []
    for lab, s in (("entangled", ""), ("unentangled", "_noent")):
        c2.append((f"{lab}: L2 > L1", nm("sv", 1, s, n_layers=2), nm("sv", 1, s, n_layers=1)))
        c2.append((f"{lab}: L5 > L2", nm("sv", 1, s, n_layers=5), nm("sv", 1, s, n_layers=2)))
    out += ["", "## C2 - depth (CartPole, R=1)\n", contrast_rows(per, c2)]
    c3 = [(f"L2 with OR{R} > L2", nm("sv", R, n_layers=2), nm("sv", 1, n_layers=2)) for R in (4, 8, 16, 32)]
    c3.append(("L5_OR32 > L2_OR32", nm("sv", 32, n_layers=5), nm("sv", 32, n_layers=2)))
    out += ["", "## C3 - depth x Output Reuse (CartPole)\n", contrast_rows(per, c3)]
    fl = per[per.arm.str.startswith("sv_frozen")]
    out += ["", "## C4 - FrozenLake Config B, L=5\n"]
    out.append("| arm | n | alive seeds (greedy_final > 0) | final_perf IQM (CI) | greedy_final IQM (CI) |\n|---|---|---|---|---|")
    for arm in ("sv_frozen_binary_4q_L5", "sv_frozen_binary_4q_noent_L5"):
        g = fl[fl.arm == arm]
        out.append(f"| `{arm}` | {len(g)} | {int((g.greedy > 0).sum())} of {len(g)} | "
                   f"{ci_str(g.final.tolist(), 3)} | {ci_str(g.greedy.dropna().tolist(), 3)} |")
    out += ["", contrast_rows(per, [("entangled > unentangled", "sv_frozen_binary_4q_L5", "sv_frozen_binary_4q_noent_L5")], 3)]

    out += ["", "## Coherence check 1 - seeds 1-10 against seeds 11-100 (same arm; should be indistinguishable)\n",
            "| arm | n 1-10 / 11-100 | IQM final 1-10 / 11-100 | P(1-10 > 11-100) final / greedy | outside [0.25, 0.75]? |",
            "|---|---|---|---|---|"]
    for arm, g in per.groupby("arm"):
        a, b = g[g.seed <= 10], g[g.seed > 10]
        if len(b) < 5:
            continue
        pf = stats.probability_of_improvement(a.final.dropna().tolist(), b.final.dropna().tolist())
        ga, gb = a.greedy.dropna().tolist(), b.greedy.dropna().tolist()
        pg = stats.probability_of_improvement(ga, gb) if ga and gb else float("nan")
        flag = "yes" if not (0.25 <= pf <= 0.75) or (not np.isnan(pg) and not (0.25 <= pg <= 0.75)) else "no"
        out.append(f"| `{arm}` | {len(a)} / {len(b)} | {stats.iqm(a.final.tolist()):.3f} / {stats.iqm(b.final.tolist()):.3f} | "
                   f"{pf:.2f} / {pg:.2f} | {flag} |")

    # simulator against its real arm (n = 10)
    b09.D = {k: root / v for k, v in b09.DIRS.items()}
    real = {}
    for R in (1, 4, 8, 16, 32):
        real[nm("sv", R)] = ("d01", "hybrid_fig4") if R == 1 else ("d02", f"hybrid_OR{R}")
        real[nm("sv", R, "_noent")] = ("d08", "hybrid_fig4_noent") if R == 1 else ("d08", f"hybrid_or_r{R}_noent")
        real[nm("su2", R)] = real[nm("sv", R, "_noent")]
        real[nm("sv", R, n_layers=2)] = ("d03", "hybrid_DR2") if R == 1 else ("d08", f"hybrid_DR2_OR{R}")
    real[nm("sv", 1, n_layers=1)] = ("d03", "hybrid_DR1")
    real[nm("sv", 1, "_noent", n_layers=1)] = ("d03b", "hybrid_DR1_noent")
    real[nm("sv", 1, "_noent", n_layers=2)] = ("d03b", "hybrid_DR2_noent")
    real["sv_frozen_binary_4q_L5"] = ("d04", "frozen_binary_4q_L5")
    real["sv_frozen_binary_4q_noent_L5"] = ("d04", "frozen_binary_4q_noent_L5")
    out += ["", "## Coherence check 2 - simulator arm (all its seeds) against its real arm (n = 10)\n",
            "| simulator arm | real arm | n sim / real | IQM final sim / real | P(sim>real) final / greedy |", "|---|---|---|---|---|"]
    for arm, (dk, ra) in real.items():
        s = per[per.arm == arm]
        r = b09.load(dk, ra, "")
        if s.empty or not r:
            continue
        rf, rg = b09.vals(r, "final"), b09.vals(r, "greedy")
        pf = stats.probability_of_improvement(s.final.dropna().tolist(), rf)
        pg = stats.probability_of_improvement(s.greedy.dropna().tolist(), rg) if rg else float("nan")
        out.append(f"| `{arm}` | `{ra}` | {len(s)} / {len(r)} | {stats.iqm(s.final.tolist()):.3f} / {stats.iqm(rf):.3f} | {pf:.2f} / {pg:.2f} |")
    out += ["", "No multiple-comparison correction is applied to any of the contrasts above."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = pathlib.Path(a.results_root), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per = collect(root, out)
    s = summary(per, out)
    (out / "tables_x100.md").write_text(tables(per, root), encoding="utf-8")
    print(f"{len(per)} runs, {s.shape[0]} arms; wrote per_seed_x100.csv, curves_*, summary_x100.csv, tables_x100.md to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
