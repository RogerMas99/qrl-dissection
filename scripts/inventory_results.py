"""What is actually in a results directory.

The question this answers is the one that comes before "can I reuse it?": which
cells finished, at what step budget, with how many seeds, and are their episode
CSVs still on disk.

    python scripts/inventory_results.py /content/drive/MyDrive/tfm_qrl
    python scripts/inventory_results.py ./results --verbose

Reads only. It never writes, moves or deletes anything, so it is safe to point at
a Drive folder you care about.

Reading the output
------------------
    cells      finished runs (one manifest each)
    seeds      distinct seeds - 3 is a coverage pass, 10 is a conclusion
    steps      step budgets found. More than one value in a directory usually
               means a smoke run and a real run share it, which the FIX-08 guard
               will refuse; move one of them.
    csv        episode CSVs still present. `0/6` means the manifests survived but
               the data did not - summary numbers only, no recomputable metrics.
    legacy     manifests missing the fields the reuse guard checks. Fix with
               scripts/migrate_manifests.py; see docs/REUSE.md.
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

GUARD_FIELDS = ("arm", "total_timesteps", "dqn_kwargs")


def scan(d: pathlib.Path) -> dict:
    manifests = sorted(d.glob("*.manifest.json"))
    info = {"dir": d, "cells": len(manifests), "seeds": set(), "steps": set(),
            "csv_present": 0, "csv_missing": [], "legacy": [], "arms": set(),
            "revisions": set(), "rows": []}
    for mp in manifests:
        try:
            m = json.loads(mp.read_text())
        except Exception:
            info["legacy"].append(mp.name + " (unreadable)")
            continue
        spec = m.get("spec", m)
        outcome = m.get("outcome", {})
        name = m.get("run_name") or m.get("name") or mp.stem

        seed = spec.get("seed", m.get("seed"))
        if seed is not None:
            info["seeds"].add(seed)
        steps = spec.get("total_timesteps", outcome.get("total_timesteps"))
        if steps:
            info["steps"].add(int(steps))
        if spec.get("arm"):
            info["arms"].add(spec["arm"])
        if m.get("git_revision"):
            info["revisions"].add(m["git_revision"])

        csv = outcome.get("episodes_csv")
        # Manifests store absolute paths from the machine that produced them, so
        # also look for the file where it would be relative to this directory.
        local = d / "runs" / f"{name}.csv"
        has_csv = bool(csv and pathlib.Path(csv).exists()) or local.exists()
        if has_csv:
            info["csv_present"] += 1
        else:
            info["csv_missing"].append(name)

        missing = [f for f in GUARD_FIELDS if f not in spec]
        if missing:
            info["legacy"].append(f"{name} (missing {', '.join(missing)})")

        info["rows"].append(dict(name=name, seed=seed, steps=steps,
                                 csv=has_csv, legacy=bool(missing)))
    return info


def report(info: dict, verbose: bool) -> None:
    d = info["dir"]
    if not info["cells"]:
        return
    seeds = sorted(info["seeds"])
    steps = sorted(info["steps"])
    print(f"\n{d}")
    print(f"  cells   {info['cells']}")
    print(f"  seeds   {len(seeds)}  {seeds}"
          + ("   <- coverage only, not a conclusion" if 0 < len(seeds) < 10 else ""))
    print(f"  steps   {steps}"
          + ("   <- MIXED BUDGETS in one directory; separate them" if len(steps) > 1 else ""))
    print(f"  csv     {info['csv_present']}/{info['cells']} present"
          + ("   <- data lost; only summary numbers survive" if info["csv_present"] == 0 else ""))
    if info["arms"]:
        print(f"  arms    {sorted(info['arms'])}")
    if info["revisions"]:
        print(f"  git     {sorted(info['revisions'])}")
    if info["legacy"]:
        print(f"  legacy  {len(info['legacy'])} manifest(s) predate the reuse guard")
        if verbose:
            for x in info["legacy"]:
                print(f"            {x}")
        print("          -> scripts/migrate_manifests.py, see docs/REUSE.md")
    if info["csv_missing"] and verbose:
        print("  missing CSVs:")
        for x in info["csv_missing"]:
            print(f"            {x}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="a results directory, or a parent of several")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    root = pathlib.Path(args.root).expanduser()
    if not root.exists():
        print(f"{root} does not exist.")
        print("In Colab, Drive paths only work after drive.mount('/content/drive').")
        return 1

    dirs = sorted({mp.parent for mp in root.rglob("*.manifest.json")})
    if not dirs:
        print(f"No manifests anywhere under {root}.")
        print("Either nothing has been run there, or the results are elsewhere -")
        print("check whether the notebook wrote to /content/... (ephemeral) rather")
        print("than /content/drive/MyDrive/... (persistent).")
        return 1

    print(f"{len(dirs)} results director{'y' if len(dirs)==1 else 'ies'} under {root}")
    totals = defaultdict(int)
    for d in dirs:
        info = scan(d)
        report(info, args.verbose)
        totals["cells"] += info["cells"]
        totals["csv"] += info["csv_present"]
        totals["legacy"] += len(info["legacy"])

    print(f"\n{'='*60}")
    print(f"TOTAL  {totals['cells']} cells, {totals['csv']} with episode CSVs, "
          f"{totals['legacy']} legacy manifests")
    if totals["legacy"]:
        print("\nLegacy manifests are still good runs - they simply predate the")
        print("fields the reuse guard checks. Migrate them once and they count")
        print("toward the robustness pass instead of being recomputed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
