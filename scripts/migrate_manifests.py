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

    # The normal case: point it at the parent and let it recognise each
    # experiment. Hyper-parameters differ between them, so this is safer than
    # one blanket --dqn-kwargs.
    python scripts/migrate_manifests.py /content/drive/MyDrive/tfm_qrl --dry-run
    python scripts/migrate_manifests.py /content/drive/MyDrive/tfm_qrl

    # Only for a directory it does not recognise:
    python scripts/migrate_manifests.py <dir> \\
        --dqn-kwargs '{"batch_size":128,"buffer_size":10000,"train_frequency":10}'
"""
import argparse
import json
import pathlib
import re
import shutil
import sys


# Per-experiment defaults, read from the scripts that produced the runs. This
# table exists because `--dqn-kwargs` is the one field no artefact records, and
# whatever gets written becomes the value every future run is checked against -
# so hand-typing it per directory is a footgun, and typing ONE value for a whole
# Drive folder is worse: exp04 differs from the CartPole experiments on two of
# three keys.
#
# Source lines, so this can be re-verified rather than trusted:
#   exp01  experiments/exp01_dqn_cartpole_capacity.py, argparse defaults
#   exp02  experiments/exp02_dqn_cartpole_output_reuse.py :: kw
#   exp03  experiments/exp03_dqn_cartpole_data_reuploading.py :: kw
#   exp03b experiments/exp03b_dqn_cartpole_dr_unentangled.py :: DQN_KWARGS
#   exp04  experiments/exp04_dqn_frozenlake_embeddings.py :: DQN_KWARGS
CARTPOLE_KWARGS = {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}

KNOWN_KWARGS = {
    "exp01_dqn_cartpole_capacity": CARTPOLE_KWARGS,   # argparse defaults, tf=10
    "exp02_dqn_cartpole_output_reuse": CARTPOLE_KWARGS,
    "exp03_dqn_cartpole_data_reuploading": CARTPOLE_KWARGS,
    "exp03b_dqn_cartpole_dr_unentangled": CARTPOLE_KWARGS,
    "exp04_dqn_frozenlake_embeddings": {"batch_size": 128, "buffer_size": 50_000,
                                        "train_frequency": 1, "learning_starts": 1_000},
}
# The pre-adoption folder names, so migration works before or after
# scripts/adopt_legacy_layout.py has run.
for _short, _long in (("exp01", "exp01_dqn_cartpole_capacity"),
                      ("exp02", "exp02_dqn_cartpole_output_reuse"),
                      ("exp03", "exp03_dqn_cartpole_data_reuploading"),
                      ("exp03b", "exp03b_dqn_cartpole_dr_unentangled"),
                      ("exp04", "exp04_dqn_frozenlake_embeddings")):
    KNOWN_KWARGS[_short] = KNOWN_KWARGS[_long]


def kwargs_for(directory: pathlib.Path, override):
    """Explicit override wins; otherwise recognise the directory."""
    if override is not None:
        return override, "supplied"
    hit = KNOWN_KWARGS.get(directory.name)
    return (hit, f"recognised {directory.name}") if hit else (None, "unknown")


def arm_for(name: str, override):
    """Recover the arm from the run name.

    exp01 mixes several arms in one directory, so a single --arm would be wrong
    for most of its cells. Run names carry it: `paper_linear__fix01on__s1`.
    exp02/exp03 name theirs `hybrid_OR16__s2` / `hybrid_DR5__s1`, which are all
    variations on hybrid_fig4 - their guard checks seed, budget and kwargs, not
    the arm, so recording the base arm is accurate enough and never used to
    rebuild a config.
    """
    if override:
        return override
    for known in ("paper_linear", "oversized_mlp", "matched_classical",
                  "hybrid_fig4", "frozen_onehot_mlp", "frozen_matched_scalar",
                  "frozen_scalar_mlp_large"):
        if name.startswith(known):
            return known
    if name.startswith("hybrid_"):        # hybrid_OR16__s2, hybrid_DR5__s1
        return "hybrid_fig4"
    return None


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
        inferred = arm_for(name, arm)
        if inferred:
            spec["arm"] = inferred
        else:
            return f"SKIP {name}: arm not recoverable from the name; pass --arm"
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
                   help="JSON. Only needed for directories this script does not "
                        "recognise; it knows exp01-exp04's own values. An "
                        "override applies to EVERY directory, so use it on one "
                        "directory at a time.")
    p.add_argument("--arm", default=None,
                   help="only if it cannot be read from the run names")
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    override = json.loads(args.dqn_kwargs) if args.dqn_kwargs else None

    root = pathlib.Path(args.results_dir)
    files = sorted(root.rglob("*.manifest.json"))
    if not files:
        print(f"no manifests under {root}")
        return 1

    # Group by directory: hyper-parameters differ per experiment, so one blanket
    # value across a whole Drive folder would write the wrong thing into most of
    # it. Pointing this at the parent folder is the normal case and now safe.
    by_dir = {}
    for mp in files:
        by_dir.setdefault(mp.parent, []).append(mp)

    print(f"{len(files)} manifests in {len(by_dir)} director"
          f"{'y' if len(by_dir)==1 else 'ies'} under {root}\n")
    unknown_dirs = []
    for d, mps in sorted(by_dir.items()):
        kwargs, how = kwargs_for(d, override)
        print(f"{d.name}/  ({len(mps)} manifests, dqn_kwargs {how})")
        if kwargs is None:
            unknown_dirs.append(d.name)
            print("    ! unrecognised directory and no --dqn-kwargs: that field")
            print("      stays unverifiable. Pass --dqn-kwargs from the script")
            print("      that produced these runs.")
        for mp in mps:
            print("    " + migrate(mp, kwargs, args.arm, args.env_id, args.dry_run))
        print()
    if args.dry_run:
        print("\ndry run: nothing written. Re-run without --dry-run to apply.")
    else:
        print("\nBackups written alongside as *.manifest.json.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
