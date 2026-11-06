"""Audit: how many episodes did each greedy evaluation use?

`GreedyEvalConfig.n_episodes` defaults to 5 and several experiments never passed a number, while others pass
20 explicitly. The manifests do not record it, so it is read from the scores themselves: the greedy score of a
checkpoint is the mean return of n episodes, and CartPole / FrozenLake returns are integers, so a 5-episode
mean is a multiple of 0.2 and a 20-episode mean is a multiple of 0.05.

Two verdicts only, because only these are safe:

  * "not 5": at least one score of the group is not a multiple of 0.2. Certain (a 5-episode mean cannot be).
  * "5": every score of every run of the group is a multiple of 0.2 AND the runs together show at least 8
    distinct non-integer scores (counted per run: a dead agent that returns 9.4 at every checkpoint is ONE
    observation, not ten). An integer score (0, 1, 9.0, ...) is a multiple of 0.2 whatever the number of
    episodes, so it says nothing; a non-integer score is a multiple of 0.2 with certainty under 5 episodes
    but only with probability about 4/19 under 20, so 8 of them have probability about 4e-6 by chance.
  * otherwise "undetermined by data": e.g. FrozenLake returns that are always 0 or 1, or an agent whose score
    is the same integer at every checkpoint, cannot be told apart, and the script that launched the group is
    the only evidence (column `code says`).

A group that has both kinds ("not 5" runs and runs with all scores multiples of 0.2 and at least 8 distinct non-integer
scores in that single run) is reported as MIXED.

    python experiments/audit_eval_episodes.py --results-root <results> --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import sys


def local(p, d):
    name = pathlib.Path(str(p).replace("\\", "/")).name
    for q in (d / "runs" / name, d / name):
        if q.exists():
            return q
    return None


def scores(path):
    out = []
    for r in list(csv.reader(path.open()))[1:]:
        try:
            out.append(float(r[1]))
        except (IndexError, ValueError):
            pass
    return out


def mult(v, step):
    return abs(v / step - round(v / step)) < 1e-6


def classify(sc):
    """(kind, informative): kind in {"not5", "mult02", "none"}; informative = number of DISTINCT non-integer scores."""
    if not sc:
        return "none", 0
    if any(not mult(v, 0.2) for v in sc):
        return "not5", 0
    return "mult02", len({round(v, 6) for v in sc if not mult(v, 1.0)})


def code_says(folder):
    f = folder.lower()
    if f.startswith(("exp04", "exp05", "exp06", "exp09", "exp10")):
        return "20 (explicit in the script)"
    if f.startswith("exp11"):
        return "5 (explicit in the runner)"
    if f.startswith(("exp01", "exp02", "exp03")):
        return "5 (library default, not passed); 20 in rerun_controlled_arms.py"
    return "no script in the repository (notebook)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = pathlib.Path(a.results_root), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = collections.defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("exp") and not p.name.startswith("exp10_emulator_x100/_")):
        for mp in sorted(d.glob("*.manifest.json")):
            try:
                m = json.loads(mp.read_text())
            except Exception:
                continue
            sp, oc = m.get("spec") or {}, m.get("outcome") or {}
            rn = oc.get("run_name") or m.get("run_name") or mp.stem
            label = re.sub(r"__fix01(on|off)$", "", re.sub(r"__s\d+(__[\w.]+)?$", "", rn))
            fx = sp.get("fix_autoreset", m.get("fix_autoreset"))
            if fx is None:
                fx = (oc.get("probe") or {}).get("fix01_active")
            ev = local(oc.get("eval_csv", ""), d) if oc.get("eval_csv") else None
            groups[(d.name, label, "on" if fx else "off", sp.get("tag", ""))].append(classify(scores(ev)) if ev else ("none", 0))

    rows = []
    for (exp, arm, fx, tag), runs in sorted(groups.items()):
        n = len(runs)
        not5 = sum(1 for k, _v in runs if k == "not5")
        strong5 = sum(1 for k, v in runs if k == "mult02" and v >= 8)
        weak = n - not5 - strong5
        if not5 and strong5:
            verdict = "MIXED"
        elif not5:
            verdict = "not 5"
        elif sum(v for k, v in runs if k == "mult02") >= 8:
            verdict = "5"
        else:
            verdict = "undetermined by data"
        rows.append(dict(exp=exp, arm=arm, fix01=fx, tag=tag, runs=n, runs_not_5=not5, runs_5_informative=strong5,
                         runs_ambiguous=weak, verdict=verdict, code_says=code_says(exp)))
    fields = list(rows[0])
    with (out / "eval_episodes_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # one line per experiment: how many groups / runs per verdict
    by_exp = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in rows:
        by_exp[r["exp"]][r["verdict"]][0] += 1
        by_exp[r["exp"]][r["verdict"]][1] += r["runs"]
    md = ["# Greedy evaluation episodes per experiment (audit)\n",
          "Generated by `experiments/audit_eval_episodes.py`. Verdict = what the scores say (see the script's "
          "docstring); `undetermined by data` = every run returned the same score at all checkpoints, so the "
          "script that launched the group is the only evidence.\n",
          "## By experiment (groups = experiment x arm x FIX-01 x tag; runs in brackets)\n",
          "| experiment | verdicts | code says |", "|---|---|---|"]
    for exp, v in by_exp.items():
        md.append(f"| `{exp}` | " + "; ".join(f"{k}: {g} groups ({r} runs)" for k, (g, r) in sorted(v.items()))
                  + f" | {code_says(exp)} |")
    md += ["", "## exp01, by arm\n", "| folder | arm | FIX-01 | tag | runs | not 5 | 5 (informative) | ambiguous | verdict |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["exp"].startswith("exp01"):
            md.append(f"| `{r['exp']}` | `{r['arm']}` | {r['fix01']} | {r['tag'] or '-'} | {r['runs']} | {r['runs_not_5']} | "
                      f"{r['runs_5_informative']} | {r['runs_ambiguous']} | {r['verdict']} |")
    (out / "eval_episodes_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(rows)} groups; wrote eval_episodes_audit.csv and .md to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
