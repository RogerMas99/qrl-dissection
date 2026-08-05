"""Undo `runs/runs/` nesting left by an earlier version of adopt_legacy_layout.py.

`shutil.move(dir, existing_dir)` places the source INSIDE the target instead of
merging. Adopting a legacy folder into a directory that already contained a
`runs/` therefore produced `runs/runs/`, and every episode CSV vanished from
where the analysis looks. The manifests were fine; only the data moved.

    python scripts/repair_nested_runs.py /content/drive/MyDrive/tfm_qrl/results --dry-run
    python scripts/repair_nested_runs.py /content/drive/MyDrive/tfm_qrl/results

Moves files up one level. Never overwrites: if a file of the same name already
sits at the destination it is reported and left alone, because the one already
in place came from a real run.
"""
import argparse
import pathlib
import shutil
import sys


def repair(runs_dir: pathlib.Path, dry_run: bool) -> tuple:
    nested = runs_dir / "runs"
    if not nested.is_dir():
        return 0, 0
    moved = clashed = 0
    for item in sorted(nested.iterdir()):
        target = runs_dir / item.name
        if target.exists():
            print(f"    CLASH {item.name} already exists above - leaving nested copy")
            clashed += 1
            continue
        print(f"    {'would move' if dry_run else 'moving'} {item.name} up one level")
        if not dry_run:
            shutil.move(str(item), str(target))
        moved += 1
    if not dry_run and not any(nested.iterdir()):
        nested.rmdir()
    return moved, clashed


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = pathlib.Path(args.root).expanduser()
    if not root.exists():
        print(f"{root} does not exist")
        return 1

    found = [d for d in root.rglob("runs") if (d / "runs").is_dir()]
    if not found:
        print(f"no nested runs/runs/ under {root} - nothing to repair")
        return 0

    total_moved = total_clash = 0
    for runs_dir in sorted(found):
        print(f"\n{runs_dir.relative_to(root)}")
        m, c = repair(runs_dir, args.dry_run)
        total_moved += m
        total_clash += c

    print(f"\n{total_moved} file(s) {'would be ' if args.dry_run else ''}moved, "
          f"{total_clash} clash(es) left alone")
    if args.dry_run:
        print("dry run: nothing written.")
    else:
        print("Confirm with: python scripts/inventory_results.py <root> -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
