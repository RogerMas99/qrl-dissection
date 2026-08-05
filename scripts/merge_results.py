"""Merge a results tree from another machine or Google account into this one.

Colab sessions in different accounts cannot see each other's Drive, so
`--claim` cooperative locking is not available: the work has to be partitioned by
hand and the outputs merged afterwards. This does the merging without the two
failure modes that make it dangerous by hand.

    python scripts/merge_results.py <source-root> <dest-root>            # report
    python scripts/merge_results.py <source-root> <dest-root> --apply

**It merges directories rather than nesting them.** `shutil.move(dir, dir)` puts
the source inside the target, which is how `runs/runs/` happens and how every
episode CSV disappears from where the analysis looks.

**It never overwrites a differing file.** A same-named cell in both trees means
two sessions ran the same work - either harmless duplication, or two different
runs sharing a name. Both are reported and left alone rather than resolved by
whichever copy happened to be second.

Identical files (same size and same last logged step) are treated as duplicates
and skipped quietly: that is the expected outcome when two partitions overlap
slightly.
"""
import argparse
import hashlib
import pathlib
import shutil
import sys


def digest(p: pathlib.Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def merge(src: pathlib.Path, dst: pathlib.Path, apply: bool, rel="") -> dict:
    stats = {"copied": 0, "duplicate": 0, "conflict": 0}
    dst.mkdir(parents=True, exist_ok=True) if apply else None
    for item in sorted(src.iterdir()):
        target = dst / item.name
        here = f"{rel}/{item.name}".lstrip("/")
        if item.is_dir():
            sub = merge(item, target, apply, here)
            for k in stats:
                stats[k] += sub[k]
            continue
        if not target.exists():
            print(f"    + {here}")
            if apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
            stats["copied"] += 1
        elif item.stat().st_size == target.stat().st_size and digest(item) == digest(target):
            stats["duplicate"] += 1
        else:
            print(f"    ! CONFLICT {here}: exists in both and differs - left alone")
            stats["conflict"] += 1
    return stats


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="results root downloaded from the other account")
    p.add_argument("dest", help="the results root you are consolidating into")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    src = pathlib.Path(args.source).expanduser()
    dst = pathlib.Path(args.dest).expanduser()
    if not src.exists():
        print(f"{src} does not exist")
        return 1
    if src.resolve() == dst.resolve():
        print("source and destination are the same directory")
        return 1

    print(f"merging {src}\n     -> {dst}\n")
    stats = merge(src, dst, args.apply)
    print(f"\n{stats['copied']} new file(s), {stats['duplicate']} identical "
          f"duplicate(s) skipped, {stats['conflict']} conflict(s) left alone")
    if stats["conflict"]:
        print("\nConflicts mean the same cell name exists in both trees with "
              "different content.\nInspect them before deciding: two sessions may "
              "have run the same seed, or\nthe same name may cover different specs. "
              "scripts/inventory_results.py helps.")
    if not args.apply:
        print("\nreport only. Re-run with --apply to copy.")
    else:
        print("\nConfirm with: python scripts/inventory_results.py <dest> -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
