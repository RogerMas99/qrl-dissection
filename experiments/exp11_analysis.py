"""Experiment 11 - analysis: per-seed table, curves, summary and the contrast tables C1-C4.

Real environment, classical control arms (no simulator). Contrasts and reading rule are the ones fixed in
docs/EXPERIMENT-11-REPLICATION-CONTROL.md before any cell ran.

Reads the new runs (exp11_replication_control, tag exp11) and the existing references (exp02: hybrid_OR{R},
classical_or_matched_r{R}, classical_OR{R}; exp08: hybrid_or_r{R}_noent) and writes into --out:

    per_seed_exp11.csv           exp, arm, fix, tag, steps, seed, final, greedy, best_ma50, final_ma50, wall_h, phantom
    curves_exp11_train_ma50.csv  50-episode moving average every 1000 steps
    curves_exp11_eval_greedy.csv greedy evaluation every 10000 steps
    summary_exp11.csv            per arm: n, params, hidden width, IQM and 95 % CI of final and greedy
    tables_exp11.md              widths and parameters, IQM table, contrasts C1-C4, mechanical reading

    python experiments/exp11_analysis.py --results-root <results> --out <dir>

`classical_or_matched_rep_r4` is the existing `classical_or_matched_r4` (same specification, checked by the
runner), so it is not a separate set of runs: wherever the design says rep_r4 the table uses matched_r4.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
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


e11 = _load("exp11_replication_control")
a10 = _load("exp10_analysis")          # run_record(): the same per-run metrics and curves as everywhere else

RS = (4, 8, 16, 32)
EXP02, EXP08 = "exp02_dqn_cartpole_output_reuse", "exp08_dqn_cartpole_frozenlake_followups"
REP = {4: "classical_or_matched_r4", 8: "classical_or_matched_rep_r8", 16: "classical_or_matched_rep_r16",
       32: "classical_or_matched_rep_r32"}
HEAD = ("> Real CartPole environment, classical control arms. No simulator is involved. Protocol, contrasts and "
        "reading rule: docs/EXPERIMENT-11-REPLICATION-CONTROL.md (written before any cell ran).\n")


def sources():
    """(label, folder, tag, short experiment name)."""
    s = []
    for R in RS:
        s += [(f"hybrid_OR{R}", EXP02, "", "exp02"), (f"classical_or_matched_r{R}", EXP02, "", "exp02"),
              (f"classical_OR{R}", EXP02, "", "exp02"), (f"hybrid_or_r{R}_noent", EXP08, "", "exp08")]
    s += [(REP[R], e11.SUBDIR, e11.TAG, "exp11") for R in (8, 16, 32)]
    return s


def runs_of(root, label, folder, tag, short):
    d, out = root / folder, []
    for mp in sorted(d.glob("*.manifest.json")):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        sp, oc = m.get("spec"), m.get("outcome")
        if not sp or not oc:
            continue
        rn = oc.get("run_name") or m.get("run_name") or mp.stem
        lab = re.sub(r"__fix01(on|off)$", "", re.sub(r"__s\d+(__[\w.]+)?$", "", rn))
        fx = sp.get("fix_autoreset", m.get("fix_autoreset"))
        if fx is None:
            fx = (oc.get("probe") or {}).get("fix01_active")
        if lab != label or sp.get("tag", "") != tag or not fx:
            continue
        r = a10.run_record(m, d, short, label)
        if r:
            out.append(r)
    return out


def collect(root, out_dir):
    rows, tr, ev = [], [], []
    for label, folder, tag, short in sources():
        rows += runs_of(root, label, folder, tag, short)
    per = pd.DataFrame([r["row"] for r in rows]).sort_values(["arm", "seed"]).reset_index(drop=True)
    cols = ["exp", "arm", "fix", "tag", "seed", "global_step"]
    for r in rows:
        k = [r["row"][c] for c in cols[:5]]
        tr += [k + [s, v] for s, v in r["train"]]
        ev += [k + [s, v] for s, v in r["evals"]]
    per.to_csv(out_dir / "per_seed_exp11.csv", index=False)
    pd.DataFrame(tr, columns=cols + ["train_ma50"]).to_csv(out_dir / "curves_exp11_train_ma50.csv", index=False)
    pd.DataFrame(ev, columns=cols + ["greedy_return"]).to_csv(out_dir / "curves_exp11_eval_greedy.csv", index=False)
    return per


def widths_params(root):
    rec = e11.design_record()
    wp = {}
    for R in RS:
        wp[f"classical_or_matched_r{R}"] = (rec[str(R)]["old_matched"]["width"], rec[str(R)]["old_matched"]["params"])
        wp[f"hybrid_OR{R}"] = (None, rec[str(R)]["hybrid_OR_total"])
        wp[f"classical_or_matched_rep_r{R}"] = (rec[str(R)]["new_rep"]["width"], rec[str(R)]["new_rep"]["params"])
    return rec, wp


def ci_str(v, nd=1):
    v = [x for x in v if x is not None and not np.isnan(x)]
    if len(v) < 2:
        return "-"
    lo, hi = stats.bootstrap_ci(v)
    return f"{stats.iqm(v):.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def summary(per, wp, out_dir):
    rows = []
    for arm, g in per.groupby("arm"):
        f, gr = g.final.dropna().tolist(), g.greedy.dropna().tolist()
        flo, fhi = stats.bootstrap_ci(f) if len(f) > 1 else (np.nan, np.nan)
        glo, ghi = stats.bootstrap_ci(gr) if len(gr) > 1 else (np.nan, np.nan)
        w, p = wp.get(arm, (None, None))
        rows.append(dict(arm=arm, exp=g.exp.iloc[0], n=len(g), hidden_width=w, params=p,
                         final_perf_iqm=stats.iqm(f), final_ci_lo=flo, final_ci_hi=fhi,
                         greedy_final_iqm=stats.iqm(gr) if gr else np.nan, greedy_ci_lo=glo, greedy_ci_hi=ghi,
                         seeds_alive_greedy_gt0=int((g.greedy > 0).sum())))
    s = pd.DataFrame(rows).round(4)
    s.to_csv(out_dir / "summary_exp11.csv", index=False)
    return s


def P(per, a, b, col):
    x = per[per.arm == a][col].dropna().tolist()
    y = per[per.arm == b][col].dropna().tolist()
    return (stats.probability_of_improvement(x, y) if x and y else float("nan")), len(x), len(y)


def verdict(pf, pg):
    if pf >= 0.95 and pg >= 0.95:
        return "strong"
    if pf <= 0.05 and pg <= 0.05:
        return "strong (reverse)"
    if 0.25 <= pf <= 0.75 and 0.25 <= pg <= 0.75:
        return "near 0.5"
    if pf >= 0.5 and pg >= 0.5:
        return "indicates a direction (a > b)"
    if pf <= 0.5 and pg <= 0.5:
        return "indicates a direction (b > a)"
    return "mixed"


def contrast_rows(per, pairs):
    out = ["| contrast (a vs b) | n_a / n_b | IQM final a / b | P(a>b) final | IQM greedy a / b | P(a>b) greedy | reading |",
           "|---|---|---|---|---|---|---|"]
    for label, a, b in pairs:
        pf, na, nb = P(per, a, b, "final")
        pg, _, _ = P(per, a, b, "greedy")
        if na == 0 or nb == 0:
            out.append(f"| {label} | {na} / {nb} | (not available) | | | | |")
            continue
        ia = lambda c, arm: stats.iqm(per[per.arm == arm][c].dropna().tolist())          # noqa: E731
        out.append(f"| {label} | {na} / {nb} | {ia('final', a):.1f} / {ia('final', b):.1f} | {pf:.2f} | "
                   f"{ia('greedy', a):.1f} / {ia('greedy', b):.1f} | {pg:.2f} | {verdict(pf, pg)} |")
    return "\n".join(out)


def tables(per, rec, wp):
    out = [HEAD, "## Widths and measured parameters (CartPole-v1)\n",
           "| R | hybrid_OR total | classical_or_matched_r (replication fixed at 4): width / params | "
           "classical_or_matched_rep_r (replication R): width / params |", "|---|---|---|---|"]
    for R in RS:
        r = rec[str(R)]
        out.append(f"| {R} | {r['hybrid_OR_total']} | {r['old_matched']['width']} / {r['old_matched']['params']} | "
                   f"{r['new_rep']['width']} / {r['new_rep']['params']}"
                   + (" (identical to the previous column: not re-run)" if R == 4 else "") + " |")
    out += ["", f"R = 4 identical to classical_or_matched_r4: {rec['r4_identical_to_classical_or_matched_r4']}.", "",
            "## IQM per arm (95 % bootstrap CI)\n",
            "| arm | exp | n | params | final_perf IQM (CI) | greedy_final IQM (CI) |", "|---|---|---|---|---|---|"]
    order = ([f"hybrid_OR{R}" for R in RS] + [f"hybrid_or_r{R}_noent" for R in RS]
             + [f"classical_or_matched_r{R}" for R in RS] + [REP[R] for R in (8, 16, 32)]
             + [f"classical_OR{R}" for R in RS])
    for arm in order:
        g = per[per.arm == arm]
        if g.empty:
            continue
        out.append(f"| `{arm}` | {g.exp.iloc[0]} | {len(g)} | {wp.get(arm, (None, ''))[1] or ''} | "
                   f"{ci_str(g.final.tolist())} | {ci_str(g.greedy.dropna().tolist())} |")
    c1 = [(f"R={R}: replication R vs replication 4 (same budget)", REP[R], f"classical_or_matched_r{R}")
          for R in (8, 16, 32)]
    c2 = [(f"R={R}: hybrid_OR{R} vs control replicating R", f"hybrid_OR{R}", REP[R]) for R in RS]
    c3 = [("rep_r32 vs rep_r4 (= matched_r4)", REP[32], REP[4]), ("rep_r16 vs rep_r4 (= matched_r4)", REP[16], REP[4])]
    c4 = [(f"R={R}: hybrid_or_r{R}_noent vs control replicating R", f"hybrid_or_r{R}_noent", REP[R]) for R in RS]
    out += ["", "## C1 - effect of replication in a classical control at equal budget\n", contrast_rows(per, c1),
            "", "## C2 - the circuit against the control with the same replication\n", contrast_rows(per, c2),
            "", "## C3 - trend with R inside the new control\n", contrast_rows(per, c3),
            "", "## C4 (context) - the unentangled circuit against the control with the same replication\n",
            contrast_rows(per, c4)]
    # mechanical application of the reading rule (the operationalisation is in the design document)
    dead = []
    for R in RS:                                   # R = 4 is classical_or_matched_r4 (REP[4])
        g = per[per.arm == REP[R]]
        if len(g):
            dead.append((R, stats.iqm(g.final.tolist()), stats.iqm(g.greedy.dropna().tolist())))
    out += ["", "## Mechanical application of the reading rule\n",
            "| check | result |", "|---|---|"]
    near = []
    for lab, a, b in c1 + c3:
        pf, _, _ = P(per, a, b, "final")
        pg, _, _ = P(per, a, b, "greedy")
        near.append(verdict(pf, pg))
    out.append(f"| C1 and C3 all `near 0.5` | {all(v == 'near 0.5' for v in near)} |")
    out.append(f"| any C1 or C3 `strong` (replication helps the control) | {any(v == 'strong' for v in near)} |")
    c2v = []
    for lab, a, b in c2:
        pf, _, _ = P(per, a, b, "final")
        pg, _, _ = P(per, a, b, "greedy")
        c2v.append(verdict(pf, pg))
    out.append(f"| every C2 `strong` (circuit above the control) | {all(v == 'strong' for v in c2v)} |")
    out.append("| control does not learn (IQM final_perf <= 12 and IQM greedy_final <= 12 at every R in 4, 8, 16, 32) | "
               f"{all(f <= 12 and g <= 12 for _, f, g in dead) if dead else 'n/a'} |")
    for R, f, g in dead:
        edge = " - at the limit of the threshold" if min(abs(f - 12), abs(g - 12)) <= 2 else ""
        out.append(f"| R={R} control: IQM final_perf / greedy_final | {f:.1f} / {g:.1f}{edge} |")
    out += ["", "No multiple-comparison correction is applied to any of the contrasts above."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = pathlib.Path(args.results_root).resolve(), pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    per = collect(root, out)
    rec, wp = widths_params(root)
    summary(per, wp, out)
    (out / "tables_exp11.md").write_text(tables(per, rec, wp), encoding="utf-8")
    print(f"{len(per)} runs, {per.arm.nunique()} arms; wrote per_seed_exp11.csv, curves_*, summary_exp11.csv, "
          f"tables_exp11.md to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
