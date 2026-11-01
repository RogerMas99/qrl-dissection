"""Validation battery of the simulators against the real runs: evaluation script.

> SIMULATOR RESULTS (agent types sv / su2) - EXPLORATORY. Only the validation figures
> approved on 2026-11-01 may be quoted in the final work (docs/SIMULATED-RESULTS-POLICY.md).

Acceptance rule, fixed in docs/EXPERIMENT-09.md section 7 BEFORE any battery cell finished:
 1. mean standardised difference d over informative pairs: 95 % bootstrap CI contains 0 and |mean d| <= 0.30
 2. at most 25 % of the (informative pair, metric) cells with P(sim > real) outside [0.25, 0.75]
 3. the sv(ent=False) / su2 twins agree: P in [0.25, 0.75] on every metric
Informative pair: IQM final_perf of at least one arm above 12 (CartPole) or 0.05 (FrozenLake).
Pairs reusing the same simulator arm against a second real reference, and the real-vs-real noise
floor, are reported but excluded from the global rule.

This script is the 2026-10-21 evaluation code, unchanged in logic (same pairs, same seeds 1-10 on the
simulator side, same thresholds, same RNG seed), moved into the repository so that the evaluation can
be repeated. It loads whatever real runs exist on disk at the time it is run. With `--compare-with`
it also parses the table of an earlier report (docs/EXPERIMENT-09-BATTERY.md) and prints, per pair,
the earlier and the current real n, d, informative flag and P(sim > real).

Usage:
    python experiments/exp09c_battery_report.py --results-root <dir with exp01.. exp09 folders> \
        --out <report.md> [--compare-with docs/EXPERIMENT-09-BATTERY.md]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

from qrl_dissection.core import analysis, stats

DIRS = dict(
    d01="exp01_dqn_cartpole_capacity", d02="exp02_dqn_cartpole_output_reuse",
    d03="exp03_dqn_cartpole_data_reuploading", d03b="exp03b_dqn_cartpole_dr_unentangled",
    d04="exp04_dqn_frozenlake_embeddings", d07="exp07_dqn_cartpole_paper_replication",
    d08="exp08_dqn_cartpole_frozenlake_followups", d09="exp09_dqn_emulator_sweep")
METRICS = ("final", "greedy", "auc")
CP, FL = "CartPole", "FrozenLake"
HEAD = ("> SIMULATOR RESULTS (agent types sv / su2) - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE "
        "AUTHOR'S EXPLICIT APPROVAL (docs/SIMULATED-RESULTS-POLICY.md; the validation-battery figures are "
        "the only approved use, recorded 2026-11-01)\n")

RNG = np.random.default_rng(0)
_cache: dict = {}
D: dict = {}


def local(p, d):
    name = pathlib.Path(str(p).replace("\\", "/")).name
    for q in (d / "runs" / name, d / name):
        if q.exists():
            return q
    return None


def load(dkey, arm, tag="", max_seed=None):
    key = (dkey, arm, tag)
    if key not in _cache:
        d, out = D[dkey], {}
        for mp in d.glob("*.manifest.json"):
            m = json.loads(mp.read_text())
            sp, oc = m.get("spec"), m.get("outcome")
            if not sp or not oc:
                continue
            rn = oc.get("run_name") or m.get("run_name") or mp.stem
            label = re.sub(r"__s\d+(__[\w.]+)?$", "", rn)
            label = re.sub(r"__fix01(on|off)$", "", label)
            fx = sp.get("fix_autoreset", m.get("fix_autoreset"))
            if fx is None:
                fx = (oc.get("probe") or {}).get("fix01_active")
            if label != arm or sp.get("tag", "") != tag or not fx:
                continue
            ep = local(oc["episodes_csv"], d)
            if ep is None:
                continue
            rew, steps = analysis.load_episodes(ep)
            ma = analysis.moving_average(rew, 50)
            st = np.asarray(steps)[len(rew) - len(ma):] if len(ma) < len(rew) else np.asarray(steps)
            auc = float(np.trapezoid(ma, st) / max(st[-1] - st[0], 1)) if len(ma) > 1 else float("nan")
            ev = local(oc.get("eval_csv", ""), d) if oc.get("eval_csv") else None
            g = None
            if ev is not None:
                _, sc = analysis.load_eval(ev)
                if len(sc):
                    g = stats.greedy_final(sc)
            out[sp["seed"]] = dict(final=stats.final_performance(rew, 0.1), greedy=g, auc=auc)
        _cache[key] = out
    out = _cache[key]
    return {k: v for k, v in out.items() if max_seed is None or k <= max_seed}


def vals(d, k):
    return [r[k] for r in d.values() if r[k] is not None and not np.isnan(r[k])]


def perm_p(x, y, n=2000):
    obs = abs(stats.probability_of_improvement(x, y) - 0.5)
    pool, k, hits = np.array(list(x) + list(y)), len(x), 0
    for _ in range(n):
        RNG.shuffle(pool)
        if abs(stats.probability_of_improvement(pool[:k], pool[k:]) - 0.5) >= obs - 1e-12:
            hits += 1
    return hits / n


def std_diff(x, y):
    sd = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return (np.mean(x) - np.mean(y)) / sd if sd > 1e-9 else float("nan")


def build_pairs():
    pairs = []

    def add(env, sim, real, glob=True, kind="sim-vs-real"):
        pairs.append((env, sim, real, glob, kind))

    add(CP, ("d09", "sv_cartpole_L5", ""), ("d01", "hybrid_fig4", ""))
    add(CP, ("d09", "sv_cartpole_L5", ""), ("d03", "hybrid_DR5", ""), glob=False)
    for R in (4, 8, 16, 32):
        add(CP, ("d09", f"sv_cartpole_L5_OR{R}", ""), ("d02", f"hybrid_OR{R}", ""))
    for L in (1, 2):
        add(CP, ("d09", f"sv_cartpole_L{L}", ""), ("d03", f"hybrid_DR{L}", ""))
        for R in (4, 8, 16, 32):
            add(CP, ("d09", f"sv_cartpole_L{L}_OR{R}", ""), ("d08", f"hybrid_DR{L}_OR{R}", ""))
    for nq in (4, 8):
        for L in (1, 2, 5):
            add(CP, ("d09", f"sv_paper_skolik_{nq}q_L{L}", ""), ("d07", f"paper_skolik_{nq}q_L{L}", ""))
    for L in (1, 2):
        add(CP, ("d09", f"su2_cartpole_L{L}", ""), ("d03b", f"hybrid_DR{L}_noent", ""))
    add(CP, ("d01", "su2_cartpole_L5", "init2pi"), ("d03b", "hybrid_DR5_noent", ""))
    add(CP, ("d01", "su2_cartpole_L5", "init2pi"), ("d08", "hybrid_fig4_noent", ""), glob=False)
    for R in (4, 8, 16, 32):
        add(CP, ("d09", f"su2_cartpole_L5_OR{R}", ""), ("d08", f"hybrid_or_r{R}_noent", ""))
    for L in (2, 5):
        add(CP, ("d09", f"su2_paper_skolik_4q_L{L}_noent", ""), ("d07", f"paper_skolik_4q_L{L}_noent", ""))
    for L in (1, 5):
        add(FL, ("d09", f"sv_frozen_binary_4q_L{L}", "", 10), ("d04", f"frozen_binary_4q_L{L}", ""))  # rule: seeds 1-10
        add(FL, ("d09", f"su2_frozen_binary_4q_L{L}", ""), ("d04", f"frozen_binary_4q_noent_L{L}", ""))
    for L in (5, 10, 15):
        add(FL, ("d09", f"su2_frozen_scalar_1q_L{L}", ""), ("d04", f"frozen_scalar_1q_L{L}", ""))
    add(FL, ("d09", "sv_frozen_binary_4q_L5", ""), ("d04", "frozen_binary_4q_L5", ""), glob=False)  # all sv seeds, extra
    add(CP, ("d09", "sv_cartpole_L5_noent", ""), ("d01", "su2_cartpole_L5", "init2pi"), kind="twin")
    for L in (1, 5):
        add(FL, ("d09", f"sv_frozen_binary_4q_noent_L{L}", ""), ("d09", f"su2_frozen_binary_4q_L{L}", ""), kind="twin")
    add(CP, ("d01", "hybrid_fig4", ""), ("d03", "hybrid_DR5", ""), glob=False, kind="real-vs-real")
    return pairs


def name(t):
    return f"{t[1]}" + (f" [{t[2]}]" if t[2] else "") + (f" (seeds<={t[3]})" if len(t) > 3 else "")


def evaluate():
    """Return (results, verdict). `results` is a list of dicts, one per pair."""
    res, dlist, outside, cells, twin_bad, twin_cells = [], [], 0, 0, 0, 0
    outside_cells = []
    for env, sim, real, glob, kind in build_pairs():
        s, r = load(*sim), load(*real)
        row = dict(kind=kind, env=env, sim=name(sim), real=name(real), n_sim=len(s), n_real=len(r), glob=glob)
        if not s or not r:
            row["pending"] = True
            res.append(row)
            continue
        thr = 12.0 if env == CP else 0.05
        iq_s, iq_r = stats.iqm(vals(s, "final")), stats.iqm(vals(r, "final"))
        informative = (iq_s > thr) or (iq_r > thr)
        ps = {}
        for k in METRICS:
            x, y = vals(s, k), vals(r, k)
            ps[k] = stats.probability_of_improvement(x, y) if x and y else float("nan")
        pp = perm_p(vals(s, "final"), vals(r, "final"))
        d = std_diff(vals(s, "final"), vals(r, "final")) if informative else float("nan")
        in_rule = kind == "sim-vs-real" and glob and informative
        if in_rule:
            dlist.append(d)
            for k in METRICS:
                if not np.isnan(ps[k]):
                    cells += 1
                    bad = not (0.25 <= ps[k] <= 0.75)
                    outside += int(bad)
                    if bad:
                        outside_cells.append((name(sim), name(real), k, ps[k]))
        if kind == "twin":
            for k in METRICS:
                if not np.isnan(ps[k]):
                    twin_cells += 1
                    twin_bad += int(not (0.25 <= ps[k] <= 0.75))
        row.update(iq_s=iq_s, iq_r=iq_r, ps=ps, pp=pp, d=d, informative=informative, in_rule=in_rule)
        res.append(row)
    verdict = {}
    if dlist:
        boot = [np.mean(RNG.choice(dlist, len(dlist))) for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        md = float(np.mean(dlist))
        verdict.update(c1=(lo <= 0 <= hi) and abs(md) <= 0.30, md=md, lo=float(lo), hi=float(hi), n_inf=len(dlist))
        verdict.update(c2=(outside / max(cells, 1)) <= 0.25, outside=outside, cells=cells)
    else:
        verdict.update(c1=False, c2=False)
    verdict.update(c3=twin_cells > 0 and twin_bad == 0, twin_bad=twin_bad, twin_cells=twin_cells,
                   outside_cells=outside_cells)
    verdict["accepted"] = bool(verdict["c1"] and verdict["c2"] and verdict["c3"])
    return res, verdict


def verdict_lines(v):
    out = []
    if "md" in v:
        out.append(f"**Criterion 1 (no global bias):** mean d = {v['md']:+.3f}, 95% CI ({v['lo']:+.3f}, {v['hi']:+.3f}), "
                   f"{v['n_inf']} informative pairs -> {'PASS' if v['c1'] else 'FAIL'}")
        out.append(f"**Criterion 2 (no excess of disagreements):** {v['outside']} of {v['cells']} cells outside "
                   f"[0.25, 0.75] = {v['outside'] / max(v['cells'], 1):.0%} (limit 25%) -> {'PASS' if v['c2'] else 'FAIL'}")
    else:
        out.append("Criteria 1 and 2: no informative pair loaded.")
    out.append(f"**Criterion 3 (the two simulators agree):** {v['twin_bad']} of {v['twin_cells']} twin cells outside "
               f"[0.25, 0.75] -> {'PASS' if v['c3'] else 'FAIL'}")
    out.append(f"**Overall: {'ACCEPTED as exploratory tools' if v['accepted'] else 'NOT ACCEPTED'}** "
               "(still under docs/SIMULATED-RESULTS-POLICY.md in either case).")
    return out


def parse_old(path):
    """Parse the pair table of an earlier report: {(sim arm, real arm): dict}."""
    old = {}
    for ln in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.startswith(("| sim-vs-real", "| twin", "| real-vs-real")):
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 10:
            continue
        sim, real = c[2].strip("`"), c[3].strip("`")
        n_sim, n_real = (int(x) for x in c[4].split("/"))
        ps = [float(x) for x in c[6].split("/")]
        d = float(c[8]) if c[8] else float("nan")
        old[(sim, real)] = dict(n_real=n_real, n_sim=n_sim, ps=ps, d=d, informative=(c[9] == "yes"))
    return old


def main():
    global D
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-with", default=None)
    a = ap.parse_args()
    root = pathlib.Path(a.results_root)
    D = {k: root / v for k, v in DIRS.items()}
    res, v = evaluate()
    old = parse_old(a.compare_with) if a.compare_with else {}

    lines = [HEAD]
    if old:
        lines += ["| kind | env | simulator arm | real arm | n sim | n real old -> new | IQM final sim / real (new) | "
                  "P(sim>real) final / greedy / auc, old | P(sim>real) final / greedy / auc, new | d old -> new | "
                  "informative old -> new |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    else:
        lines += ["| kind | env | simulator arm | real arm | n sim / real | IQM final sim / real | "
                  "P(sim>real) final / greedy / auc | perm. p final | d | informative |", "|---|---|---|---|---|---|---|---|---|---|"]
    changed = []
    for r in res:
        if r.get("pending"):
            lines.append(f"| {r['kind']} | {r['env']} | `{r['sim']}` | `{r['real']}` | {r['n_sim']} / {r['n_real']} | (pending) |")
            continue
        fd = lambda x: "" if x is None or np.isnan(x) else f"{x:+.2f}"
        pnew = " / ".join(f"{r['ps'][k]:.2f}" for k in METRICS)
        inf_new = "yes" if r["informative"] else "no"
        if old:
            o = old.get((r["sim"], r["real"]))
            if o:
                pold = " / ".join(f"{x:.2f}" for x in o["ps"])
                inf_old = "yes" if o["informative"] else "no"
                if inf_old != inf_new:
                    changed.append((r["sim"], r["real"], inf_old, inf_new))
                lines.append(f"| {r['kind']} | {r['env']} | `{r['sim']}` | `{r['real']}` | {r['n_sim']} | {o['n_real']} -> {r['n_real']} | "
                             f"{r['iq_s']:.3f} / {r['iq_r']:.3f} | {pold} | {pnew} | {fd(o['d']) or '-'} -> {fd(r['d']) or '-'} | {inf_old} -> {inf_new} |")
            else:
                lines.append(f"| {r['kind']} | {r['env']} | `{r['sim']}` | `{r['real']}` | {r['n_sim']} | (not in old table) -> {r['n_real']} | "
                             f"{r['iq_s']:.3f} / {r['iq_r']:.3f} | - | {pnew} | - -> {fd(r['d']) or '-'} | - -> {inf_new} |")
        else:
            lines.append(f"| {r['kind']} | {r['env']} | `{r['sim']}` | `{r['real']}` | {r['n_sim']} / {r['n_real']} | "
                         f"{r['iq_s']:.3f} / {r['iq_r']:.3f} | {pnew} | {r['pp']:.2f} | {fd(r['d'])} | {inf_new} |")
    lines.append("")
    lines += verdict_lines(v)
    if v.get("outside_cells"):
        lines += ["", "Cells outside [0.25, 0.75] (pairs inside the global rule):", ""]
        lines += [f"- `{s}` vs `{rl}`, {k}: P(sim>real) = {p:.2f}" for s, rl, k, p in v["outside_cells"]]
    if old:
        lines += ["", "Pairs whose informative flag changed: " + ("none." if not changed else
                  "; ".join(f"`{s}` vs `{rl}` ({o} -> {n})" for s, rl, o, n in changed))]
    pathlib.Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
