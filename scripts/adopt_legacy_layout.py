"""Move results produced by the exploratory notebooks into the suite's layout.

The exploratory notebooks each wrote to `tfm_qrl/expNN`. `run_dqn_suite.py`
expects one root with a directory per experiment, named after the script. This
renames the former into the latter so the finished cells are found and skipped
instead of recomputed.

    python scripts/adopt_legacy_layout.py /content/drive/MyDrive/tfm_qrl --dry-run
    python scripts/adopt_legacy_layout.py /content/drive/MyDrive/tfm_qrl

Moves, never copies or deletes: if a target already exists the source is left
alone and reported, so nothing can be silently merged or overwritten.
"""
import argparse
import pathlib
import shutil
import sys

MAPPING = {
    "exp01": "results/exp01_dqn_cartpole_capacity",
    "exp02": "results/exp02_dqn_cartpole_output_reuse",
    "exp03": "results/exp03_dqn_cartpole_data_reuploading",
    "exp03b": "results/exp03b_dqn_cartpole_dr_unentangled",
    "exp04": "results/exp04_dqn_frozenlake_embeddings",
}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="the folder holding exp01/, exp02/, ...")
    p.add_argument("--only", nargs="+", choices=sorted(MAPPING),
                   help="adopt only these. Anything left out stays where it is, "
                        "which is the right choice for a folder you intend to "
                        "re-run rather than reuse.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = pathlib.Path(args.root).expanduser()
    if not root.exists():
        print(f"{root} does not exist. In Colab, mount Drive first.")
        return 1

    wanted = set(args.only) if args.only else set(MAPPING)
    left_alone = sorted(set(MAPPING) - wanted)
    moved = skipped = 0
    for old, new in MAPPING.items():
        if old not in wanted:
            continue
        src, dst = root / old, root / new
        if not src.exists():
            continue
        if not any(src.glob("*.manifest.json")):
            print(f"  skip {old}: no manifests inside")
            skipped += 1
            continue
        if dst.exists():
            print(f"  SKIP {old} -> {new}: target already exists, leaving both alone")
            skipped += 1
            continue
        print(f"  {'would move' if args.dry_run else 'moving'} {old} -> {new}"
              f"  ({len(list(src.glob('*.manifest.json')))} cells)")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        moved += 1

    for old in left_alone:
        if (root / old).exists():
            print(f"  leaving {old} where it is (not in --only)")

    if not moved and not skipped:
        print(f"nothing to adopt under {root} - already in the suite layout, or empty")
    elif args.dry_run:
        print("\ndry run: nothing moved.")
    else:
        print(f"\n{moved} moved. The manifests store ABSOLUTE paths to their CSVs,")
        print("which no longer resolve after a move - the analysis notebook and")
        print("inventory tool both fall back to <dir>/runs/<name>.csv, so this is")
        print("handled. Run the inventory to confirm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
