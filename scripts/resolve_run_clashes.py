"""Decide which copy to keep when a run exists twice after a repair.

`repair_nested_runs.py` refuses to overwrite, so a name present both in `runs/`
and in `runs/runs/` is left in both places. That is the right default - it never
destroys anything - but it leaves a real ambiguity, and the wrong resolution is
silently wrong rather than loud.

The usual cause: the suite started a cell, wrote a partial CSV to the new
location, and was interrupted before the manifest was written. The manifest on
disk therefore describes the ORIGINAL complete run while the CSV at the top level
is a fragment of an abandoned one. Analysis would read the fragment and report it
as the finished cell.

This compares the two by their last logged `global_step` and by the budget the
manifest claims, and keeps the one that actually matches.

    python scripts/resolve_run_clashes.py <results-root>              # report only
    python scripts/resolve_run_clashes.py <results-root> --apply      # act

`--apply` never deletes: the rejected copy is renamed `*.rejected`.
"""
import argparse
import json
import pathlib
import shutil
import sys


def last_step(path: pathlib.Path):
    try:
        import pandas as pd
        df = pd.read_csv(path)
        col = "global_step" if "global_step" in df.columns else df.columns[0]
        return int(df[col].max()), len(df)
    except Exception:
        return None, None


def claimed_budget(cell_dir: pathlib.Path, name: str):
    mp = cell_dir.parent / f"{name}.manifest.json"
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text())
        return (m.get("spec", {}).get("total_timesteps")
                or m.get("outcome", {}).get("total_timesteps"))
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    root = pathlib.Path(args.root).expanduser()
    nested_dirs = [d for d in root.rglob("runs") if (d / "runs").is_dir()]
    if not nested_dirs:
        print(f"no leftover runs/runs/ under {root} - nothing to resolve")
        return 0

    swapped = kept = 0
    for runs_dir in sorted(nested_dirs):
        nested = runs_dir / "runs"
        for inner in sorted(nested.glob("*.csv")):
            outer = runs_dir / inner.name
            if not outer.exists():
                continue
            name = inner.stem.replace("_eval", "")
            budget = claimed_budget(runs_dir, name)

            o_step, o_rows = last_step(outer)
            i_step, i_rows = last_step(inner)
            print(f"\n{runs_dir.relative_to(root)}/{inner.name}")
            print(f"    top level : last step {o_step}, {o_rows} rows")
            print(f"    nested    : last step {i_step}, {i_rows} rows")
            print(f"    manifest claims {budget} steps")

            better = None
            if o_step is not None and i_step is not None:
                if budget:
                    # Whichever gets closer to the budget the manifest asserts.
                    better = "nested" if abs(i_step - budget) < abs(o_step - budget) else "top"
                else:
                    better = "nested" if i_step > o_step else "top"

            if better == "nested":
                print("    -> the NESTED copy matches the manifest; the top-level "
                      "one is a fragment of an abandoned run")
                if args.apply:
                    shutil.move(str(outer), str(outer) + ".rejected")
                    shutil.move(str(inner), str(outer))
                    print("       swapped (old top-level kept as *.rejected)")
                swapped += 1
            else:
                print("    -> keeping the top-level copy")
                if args.apply:
                    shutil.move(str(inner), str(inner) + ".rejected")
                kept += 1

    print(f"\n{swapped} to swap, {kept} to keep as-is")
    if not args.apply:
        print("report only. Re-run with --apply to act; nothing is ever deleted.")
    else:
        print("Confirm with: python scripts/inventory_results.py <root> -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
