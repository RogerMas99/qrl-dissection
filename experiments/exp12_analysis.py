"""Experiment 12 - analysis: per-seed table, curves, summary and the contrast tables G1-G4 + ratios.

Real environment, classical twin arms (no simulator). Contrasts and reading rule are the ones fixed in
docs/EXPERIMENT-12-CLASSICAL-TWIN.md before any cell ran (quantifiers operationalised in its section 6).

Reads the new runs (exp12_classical_twin, tag exp12) and the existing references (exp01: hybrid_fig4; exp02:
hybrid_OR{R}) and writes into --out:

    per_seed_exp12.csv           exp, arm, fix, tag, steps, seed, final, greedy, best_ma50, final_ma50, wall_h, phantom
    curves_exp12_train_ma50.csv  50-episode moving average every 1000 steps
    curves_exp12_eval_greedy.csv greedy evaluation every 10000 steps
    summary_exp12.csv            per arm: n, params, IQM and 95 % CI of final and greedy
    tables_exp12.md              parameters, IQM table, contrasts G1-G4, ratios, mechanical reading

    python experiments/exp12_analysis.py --results-root <results> --out <dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

from qrl_dissection.core import stats

HERE = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e12 = _load("exp12_classical_twin")
a11 = _load("exp11_analysis")         # runs_of(), verdict(), P(), contrast_rows(), ci_str(): shared with exp11

RS = e12.RS                           # (1, 4, 8, 16, 32)
RS_OR = (4, 8, 16, 32)
EXP01, EXP02 = "exp01_dqn_cartpole_capacity", "exp02_dqn_cartpole_output_reuse"
H = {R: ("hybrid_fig4" if R == 1 else f"hybrid_OR{R}") for R in RS}
T = {R: f"classical_twin_r{R}" for R in RS}
U = {R: f"classical_twin_unbounded_r{R}" for R in RS}
HEAD = ("> Real CartPole environment, classical twin arms. No simulator is involved. Protocol, contrasts and reading "
        "rule: docs/EXPERIMENT-12-CLASSICAL-TWIN.md (written before any cell ran).\n")


def sources():
    s = [(H[1], EXP01, "", "exp01")] + [(H[R], EXP02, "", "exp02") for R in RS_OR]
    s += [(T[R], e12.SUBDIR, e12.TAG, "exp12") for R in RS] + [(U[R], e12.SUBDIR, e12.TAG, "exp12") for R in RS]
    return s


def collect(root, out_dir):
    rows, tr, ev = [], [], []
    for label, folder, tag, short in sources():
        rows += a11.runs_of(root, label, folder, tag, short)
    per = pd.DataFrame([r["row"] for r in rows]).sort_values(["arm", "seed"]).reset_index(drop=True)
    cols = ["exp", "arm", "fix", "tag", "seed", "global_step"]
    for r in rows:
        k = [r["row"][c] for c in cols[:5]]
        tr += [k + [s, v] for s, v in r["train"]]
        ev += [k + [s, v] for s, v in r["evals"]]
    per.to_csv(out_dir / "per_seed_exp12.csv", index=False)
    pd.DataFrame(tr, columns=cols + ["train_ma50"]).to_csv(out_dir / "curves_exp12_train_ma50.csv", index=False)
    pd.DataFrame(ev, columns=cols + ["greedy_return"]).to_csv(out_dir / "curves_exp12_eval_greedy.csv", index=False)
    return per


def params_table(rec):
    p = {}
    for R in RS:
        p[H[R]] = rec[str(R)]["hybrid_total"]
        p[T[R]] = rec[str(R)]["classical_twin"]["total"]
        p[U[R]] = rec[str(R)]["classical_twin_unbounded"]["total"]
    return p


def summary(per, params, out_dir):
    rows = []
    for arm, g in per.groupby("arm"):
        f, gr = g.final.dropna().tolist(), g.greedy.dropna().tolist()
        flo, fhi = stats.bootstrap_ci(f) if len(f) > 1 else (np.nan, np.nan)
        glo, ghi = stats.bootstrap_ci(gr) if len(gr) > 1 else (np.nan, np.nan)
        rows.append(dict(arm=arm, exp=g.exp.iloc[0], n=len(g), params=params.get(arm),
                         final_perf_iqm=stats.iqm(f), final_ci_lo=flo, final_ci_hi=fhi,
                         greedy_final_iqm=stats.iqm(gr) if gr else np.nan, greedy_ci_lo=glo, greedy_ci_hi=ghi,
                         seeds_alive_greedy_gt0=int((g.greedy > 0).sum())))
    s = pd.DataFrame(rows).round(4)
    s.to_csv(out_dir / "summary_exp12.csv", index=False)
    return s


def ratio_ci(x, x1, n_boot=10_000, seed=0):
    """IQM(x) / IQM(x1) with a percentile bootstrap CI (both groups resampled independently)."""
    x, x1 = np.asarray(x, float), np.asarray(x1, float)
    x, x1 = x[np.isfinite(x)], x1[np.isfinite(x1)]
    rng = np.random.default_rng(seed)
    pt = stats.iqm(x) / stats.iqm(x1)
    b = [stats.iqm(rng.choice(x, x.size)) / stats.iqm(rng.choice(x1, x1.size)) for _ in range(n_boot)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return pt, float(lo), float(hi)


def ratio_rows(per):
    out = ["| family | R | ratio final_perf IQM(R)/IQM(1) (95 % CI) | ratio greedy_final IQM(R)/IQM(1) (95 % CI) |",
           "|---|---|---|---|"]
    for name, fam in (("bounded twin", T), ("unbounded twin", U), ("hybrid", H)):
        for R in RS_OR:
            cells = []
            for col in ("final", "greedy"):
                a = per[per.arm == fam[R]][col].dropna().tolist()
                b = per[per.arm == fam[1]][col].dropna().tolist()
                cells.append("(n/a)" if not a or not b else "{:.2f} ({:.2f}-{:.2f})".format(*ratio_ci(a, b)))
            out.append(f"| {name} | {R} | {cells[0]} | {cells[1]} |")
    return "\n".join(out)


def verdicts(per, pairs):
    res = []
    for _lab, a, b in pairs:
        pf, na, nb = a11.P(per, a, b, "final")
        pg, _, _ = a11.P(per, a, b, "greedy")
        res.append(a11.verdict(pf, pg) if na and nb else "n/a")
    return res


def tables(per, rec, params):
    out = [HEAD, "## Parameters (CartPole-v1, measured by building each arm)\n",
           "| R | hybrid total | twin total (bounded and unbounded) | extra vs hybrid | head | extractor |",
           "|---|---|---|---|---|---|"]
    for R in RS:
        r = rec[str(R)]
        out.append(f"| {R} | {r['hybrid_total']} | {r['classical_twin']['total']} | +{r['classical_twin']['extra_vs_hybrid']} "
                   f"| {r['classical_twin']['head']} | {r['classical_twin']['extractor']} |")
    out += ["", "Head and OutputReuse of the twin are the hybrid's (same code, same shapes, same initial weights under one "
            "seed; `tests/test_exp12_arms.py`).", "", "## IQM per arm (95 % bootstrap CI)\n",
            "| arm | exp | n | params | final_perf IQM (CI) | greedy_final IQM (CI) |", "|---|---|---|---|---|---|"]
    order = [H[R] for R in RS] + [T[R] for R in RS] + [U[R] for R in RS]
    for arm in order:
        g = per[per.arm == arm]
        if g.empty:
            continue
        out.append(f"| `{arm}` | {g.exp.iloc[0]} | {len(g)} | {params.get(arm, '')} | "
                   f"{a11.ci_str(g.final.tolist())} | {a11.ci_str(g.greedy.dropna().tolist())} |")

    g1 = [(f"R={R}: twin R vs twin R=1", T[R], T[1]) for R in RS_OR]
    g2 = [(f"R={R}: hybrid_OR{R} vs hybrid_fig4", H[R], H[1]) for R in RS_OR]
    g3 = [(f"R={R}: {H[R]} vs twin R={R}", H[R], T[R]) for R in RS]
    g4 = [(f"R={R}: bounded twin vs unbounded twin", T[R], U[R]) for R in RS]
    g1u = [(f"R={R}: unbounded twin R vs unbounded twin R=1", U[R], U[1]) for R in RS_OR]
    g3u = [(f"R={R}: {H[R]} vs unbounded twin R={R}", H[R], U[R]) for R in RS]       # context, not in the rule
    out += ["", "## G1 - effect of Output Reuse in the bounded twin\n", a11.contrast_rows(per, g1),
            "", "## G2 - effect of Output Reuse in the hybrid (reference, same scripts)\n", a11.contrast_rows(per, g2),
            "", "## G3 - circuit against twin at equal R\n", a11.contrast_rows(per, g3),
            "", "## G4 - role of the bounded range: bounded twin vs unbounded twin, per R\n", a11.contrast_rows(per, g4),
            "", "## G4 (repeated G1) - effect of Output Reuse in the unbounded twin\n", a11.contrast_rows(per, g1u),
            "", "## Context (not part of the reading rule) - circuit against the unbounded twin\n",
            a11.contrast_rows(per, g3u),
            "", "## Descriptive - gain of Output Reuse relative to R = 1\n", ratio_rows(per)]

    # ---- mechanical application of the reading rule (operationalisation: design document, section 6)
    def iqms(arm):
        g = per[per.arm == arm]
        return (stats.iqm(g.final.tolist()), stats.iqm(g.greedy.dropna().tolist())) if len(g) else (np.nan, np.nan)

    v1, v2, v4, v1u = verdicts(per, g1), verdicts(per, g2), verdicts(per, g4[1:]), verdicts(per, g1u)
    f1, gr1 = iqms(T[1])
    learns_r1 = bool(f1 > 12 and gr1 > 12)
    g1_near_all = all(v == "near 0.5" for v in v1)
    g1_strong_any = any(v == "strong" for v in v1)
    g2_strong_any = any(v == "strong" for v in v2)
    twin_dead = all(all(x <= 12 for x in iqms(T[R])) for R in RS)
    unb_dead = all(all(x <= 12 for x in iqms(U[R])) for R in RS)

    out += ["", "## Mechanical application of the reading rule\n", "| check | result |", "|---|---|",
            f"| bounded twin R=1: IQM final_perf / greedy_final | {f1:.1f} / {gr1:.1f} (learns: {learns_r1}) |",
            f"| G1 per R = 4, 8, 16, 32 | {', '.join(v1)} |",
            f"| G1 near 0.5 at all four R | {g1_near_all} |",
            f"| G1 strong at least at one R | {g1_strong_any} |",
            f"| G2 per R = 4, 8, 16, 32 | {', '.join(v2)} |",
            f"| G2 strong at least at one R (the hybrid gains) | {g2_strong_any} |",
            f"| G4 repeated (unbounded twin, R vs R=1) per R | {', '.join(v1u)} |",
            f"| G4 (bounded vs unbounded) per R = 4, 8, 16, 32 | {', '.join(v4)} |",
            f"| bounded twin does not learn at any R (IQM <= 12 in both metrics at R = 1, 4, 8, 16, 32) | {twin_dead} |",
            f"| unbounded twin does not learn at any R | {unb_dead} |"]
    b1 = learns_r1 and g1_near_all and g2_strong_any
    out += ["", f"- **Branch 1** (twin learns without OR, OR gives it nothing, OR gives the hybrid a gain): "
                f"{'holds' if b1 else 'does not hold'} (learns at R=1: {learns_r1}; G1 near 0.5 at all R: {g1_near_all}; "
                f"hybrid gain: {g2_strong_any}).",
            f"- **Branch 2** (OR gives the twin a strong gain): {'holds' if g1_strong_any else 'does not hold'} "
            f"(G1 strong at at least one R: {g1_strong_any}).",
            f"- **Branch 4** (the bounded twin does not learn at any R): {'holds' if twin_dead else 'does not hold'}."]
    b3 = []
    for i, R in enumerate(RS_OR):
        helps_b = v1[i] == "strong"
        helps_u = v1u[i] == "strong"
        if helps_b and (not helps_u or v4[i] == "strong"):
            b3.append(f"R={R}: tied to the bounded range")
        elif helps_b and helps_u and v4[i] == "near 0.5":
            b3.append(f"R={R}: does not depend on the range")
        elif helps_b:
            b3.append(f"R={R}: helps both, G4 {v4[i]} (neither reading applies cleanly)")
        elif helps_u:
            b3.append(f"R={R}: helps the unbounded twin but not the bounded one (G1 {v1[i]})")
        else:
            b3.append(f"R={R}: not evaluable (OR does not help either twin strongly)")
    out.append("- **Branch 3** (G4), per R: " + "; ".join(b3) + ".")
    holding = [n for n, ok in (("1", b1), ("2", g1_strong_any), ("4", twin_dead)) if ok]
    out += ["", f"Branches of the rule that hold mechanically: {', '.join(holding) if holding else 'none'} "
                "(branch 3 is conditional, see its per-R line). Intermediate values are not forced into a branch; "
                "the dossier states which branch holds, which do not and why, and says so if none holds cleanly.",
            "", "No multiple-comparison correction is applied to any of the contrasts above."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = pathlib.Path(args.results_root).resolve(), pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    per = collect(root, out)
    rec = e12.design_record()
    params = params_table(rec)
    summary(per, params, out)
    (out / "tables_exp12.md").write_text(tables(per, rec, params), encoding="utf-8")
    (out / "design_params.json").write_text(__import__("json").dumps(rec, indent=1))
    print(f"{len(per)} runs, {per.arm.nunique()} arms; wrote per_seed_exp12.csv, curves_*, summary_exp12.csv, "
          f"tables_exp12.md to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
