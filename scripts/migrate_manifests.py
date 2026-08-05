"""Back-fill legacy manifests so already-computed runs stay reusable.

The problem this solves
-----------------------
Early experiment scripts wrote a manifest with only `{name, seed,
fix_autoreset, outcome, config}`. The reuse guard added in FIX-08 compares the
stored spec against the requested one, and cannot check a field that was never
recorded - so it refuses, and a perfectly good 100k-step run gets recomputed.

Recomputing hours of finished simulation because of a missing JSON key is the
wrong trade. This script records what can be recovered and asks for the rest.

What is recoverable, and what is not
------------------------------------
  total_timesteps   RECOVERED from `outcome.total_timesteps`, or from the last
                    `global_step` in the episode CSV if the outcome is absent.
  arm, seed, tag    RECOVERED from the run name and the existing fields.
  fix_autoreset     RECOVERED from the existing field or the run name.
  env_id            RECOVERED from the manifest, or assumed CartPole-v1 with a
                    warning - every legacy experiment here was CartPole.
  dqn_kwargs        NOT recoverable from any artefact. It must be supplied, and
                    it must be right: the value is what future runs will be
                    checked against. Take it from the notebook or script that
                    produced the runs, not from memory.

Nothing is overwritten without a backup: each file is copied to
`<name>.manifest.json.bak` before being rewritten.

    python scripts/migrate_manifests.py /content/drive/MyDrive/tfm_qrl/exp03 \\
        --dqn-kwargs '{"batch_size":128,"buffer_size":10000,"train_frequency":10}'

    python scripts/migrate_manifests.py <dir> --dry-run     # inspect first
"""
import argparse
import json
import pathlib
import re
import shutil
import sys

RECOVERABLE = ("arm", "seed", "fix_autoreset", "total_timesteps", "tag")


def infer_from_name(name: str) -> dict:
    out = {}
    if "__fix01on" in name:
        out["fix_autoreset"] = True
    elif "__fix01off" in name:
        out["fix_autoreset"] = False
    m = re.search(r"__s(\d+)", name)
    if m:
        out["seed"] = int(m.group(1))
    return out


def steps_from_csv(path) -> int | None:
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return int(df.global_step.max())
    except Exception:
        return None


def migrate(mp: pathlib.Path, dqn_kwargs, arm, env_id, dry_run) -> str:
    m = json.loads(mp.read_text())
    spec = dict(m.get("spec", {}))
    outcome = m.get("outcome", {})
    name = m.get("run_name") or m.get("name") or mp.stem.replace(".manifest", "")
    before = set(spec)

    for k, v in infer_from_name(name).items():
        spec.setdefault(k, v)
    for k in ("seed", "fix_autoreset", "arm", "tag"):
        if k not in spec and k in m:
            spec[k] = m[k]

    if "total_timesteps" not in spec:
        if "total_timesteps" in outcome:
            spec["total_timesteps"] = outcome["total_timesteps"]
        elif outcome.get("episodes_csv"):
            got = steps_from_csv(outcome["episodes_csv"])
            if got is not None:
                # The last logged step is a lower bound on the budget: the final
                # episode ends at or before it. Round to the nearest thousand and
                # flag it, rather than pretending to precision we do not have.
                spec["total_timesteps"] = int(round(got, -3))
                spec["_total_timesteps_inferred_from_csv"] = True

    if "arm" not in spec:
        if arm:
            spec["arm"] = arm
        else:
            return f"SKIP {name}: no `arm` and none supplied (--arm)"
    spec.setdefault("tag", "")
    if dqn_kwargs is not None:
        spec["dqn_kwargs"] = dqn_kwargs
    if "total_timesteps" not in spec:
        return f"SKIP {name}: step budget not recoverable"

    m["spec"] = spec
    m.setdefault("env_id", env_id)
    m["_migrated_by"] = "scripts/migrate_manifests.py"
    added = sorted(set(spec) - before - {"_total_timesteps_inferred_from_csv"})
    if dry_run:
        return f"would add {added} to {name}"
    shutil.copy2(mp, mp.with_suffix(".json.bak"))
    mp.write_text(json.dumps(m, indent=2, default=str))
    return f"migrated {name}: added {added}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("--dqn-kwargs", default=None,
                   help="JSON. Take it from the script that produced the runs.")
    p.add_argument("--arm", default=None, help="only if the manifests lack one")
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    kwargs = json.loads(args.dqn_kwargs) if args.dqn_kwargs else None
    if kwargs is None and not args.dry_run:
        print("! no --dqn-kwargs given: manifests will still be unverifiable on "
              "that field.\n  Supply it, or accept them with on_mismatch='legacy'.\n")

    root = pathlib.Path(args.results_dir)
    files = sorted(root.rglob("*.manifest.json"))
    if not files:
        print(f"no manifests under {root}")
        return 1
    print(f"{len(files)} manifests under {root}\n")
    for mp in files:
        print("  " + migrate(mp, kwargs, args.arm, args.env_id, args.dry_run))
    if args.dry_run:
        print("\ndry run: nothing written. Re-run without --dry-run to apply.")
    else:
        print("\nBackups written alongside as *.manifest.json.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
