"""Experiment 10 - 100-seed extension on the simulators (docs/EXPERIMENT-10-X100.md).

SIMULATOR RESULTS - EXPLORATORY - NOT FOR THE FINAL WORK without the author's explicit approval
(docs/SIMULATED-RESULTS-POLICY.md). Arms, contrasts and reading rules are fixed in
docs/EXPERIMENT-10-X100.md, written before any cell ran.

Groups (priority order): P1 entanglement x Output Reuse (L=5), P2 depth, P3 depth x Output Reuse
(L=2), P4 FrozenLake Config B L=5. Seeds 1-100. Seeds that already exist for an arm with exactly the
same specification (arm, dqn_kwargs, total_timesteps, FIX-01 on, same tag in the source folder) are
reused from where they are; the others are run with tag "x100" into <results-root>/exp10_emulator_x100/.

    python experiments/exp10_emulator_x100.py --group P1 --results-root <results> --claim
    python experiments/exp10_emulator_x100.py --group P1 --results-root <results> --list
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import qrl_dissection
from qrl_dissection.core.configs import cartpole_replica_name
from qrl_dissection.core.obs_adapters import FROZEN_SCALAR_ID
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

# Protocols copied VERBATIM from experiments/exp09_dqn_emulator_sweep.py (itself copied from exp04 /
# exp01-08); tests/test_exp10_arms.py asserts they are equal.
PRIMARY_KWARGS: Dict[str, object] = {"batch_size": 128, "buffer_size": 50_000,
                                     "train_frequency": 1, "learning_starts": 1_000}
CARTPOLE_KWARGS: Dict[str, object] = {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}
SEEDS = list(range(1, 101))
TAG = "x100"
SUBDIR = "exp10_emulator_x100"
GROUPS = ["P1", "P2", "P3", "P4"]
RS = (4, 8, 16, 32)

# (source folder, tag in that folder) searched for runs with an identical specification
REUSE_SOURCES: List[Tuple[str, str]] = [("exp09_dqn_emulator_sweep", ""), ("exp01_dqn_cartpole_capacity", "init2pi")]


def group_arms(group: str) -> List[Tuple[str, Dict[str, object], str]]:
    """(arm, dqn_kwargs, env_id) for the group."""
    cp = "CartPole-v1"
    L5 = lambda kind, R, sfx="": cartpole_replica_name(kind, R, sfx)           # noqa: E731
    if group == "P1":
        names = ([L5("sv", 1)] + [L5("sv", R) for R in RS]
                 + [L5("sv", 1, "_noent")] + [L5("sv", R, "_noent") for R in RS]
                 + [L5("su2", 1)] + [L5("su2", R) for R in RS])
        return [(a, CARTPOLE_KWARGS, cp) for a in names]
    if group == "P2":
        names = [cartpole_replica_name("sv", 1, sfx, n_layers=L) for L in (1, 2) for sfx in ("", "_noent")]
        return [(a, CARTPOLE_KWARGS, cp) for a in names]
    if group == "P3":
        return [(cartpole_replica_name("sv", R, n_layers=2), CARTPOLE_KWARGS, cp) for R in RS]
    if group == "P4":
        return [(a, PRIMARY_KWARGS, FROZEN_SCALAR_ID)
                for a in ("sv_frozen_binary_4q_L5", "sv_frozen_binary_4q_noent_L5")]
    raise ValueError(group)


_index: Dict[Tuple[str, str], Dict[str, Dict[int, dict]]] = {}


def _source_index(root: pathlib.Path, folder: str):
    """{arm: {(seed, tag): spec dict}} for the manifests of one source folder whose run finished.
    Keyed by tag as well: the same arm and seed can exist under several tags (e.g. a learning-rate variant)."""
    if folder in _index:
        return _index[folder]
    out: Dict[str, Dict[Tuple[int, str], dict]] = {}
    for mp in (root / folder).glob("*.manifest.json"):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        sp, oc = m.get("spec"), m.get("outcome")
        if not sp or not oc or not oc.get("episodes_csv"):
            continue
        name = pathlib.Path(str(oc["episodes_csv"]).replace("\\", "/")).name
        if not ((root / folder / "runs" / name).exists() or (root / folder / name).exists()):
            continue
        out.setdefault(sp["arm"], {})[(sp["seed"], sp.get("tag", ""))] = sp
    _index[folder] = out
    return out


def reusable_seeds(root: pathlib.Path, arm: str, kwargs: Dict[str, object], steps: int) -> Dict[int, Tuple[str, str]]:
    """Seeds of `arm` that already exist elsewhere with exactly this specification."""
    found: Dict[int, Tuple[str, str]] = {}
    for folder, tag in REUSE_SOURCES:
        for (seed, sp_tag), sp in _source_index(root, folder).get(arm, {}).items():
            if (sp_tag == tag and sp.get("fix_autoreset") and sp.get("dqn_kwargs") == kwargs
                    and sp.get("total_timesteps") == steps and seed in SEEDS):
                found.setdefault(seed, (folder, tag))
    return found


def build_specs(group: str, steps: int, root: pathlib.Path = None, seeds: List[int] = None
                ) -> Tuple[List[RunSpec], Dict[str, Dict[int, Tuple[str, str]]]]:
    """Specs to run (seed-major order, so that all arms advance together) and the reused seeds."""
    sds = seeds or SEEDS
    reused: Dict[str, Dict[int, Tuple[str, str]]] = {}
    todo: Dict[str, List[int]] = {}
    arms = group_arms(group)
    for arm, kw, _env in arms:
        r = reusable_seeds(root, arm, kw, steps) if root is not None else {}
        reused[arm] = {s: v for s, v in r.items() if s in sds}
        todo[arm] = [s for s in sds if s not in r]
    specs = [RunSpec(arm=arm, seed=s, fix_autoreset=True, total_timesteps=steps, dqn_kwargs=dict(kw), tag=TAG)
             for s in sds for arm, kw, _env in arms if s in todo[arm]]
    return specs, reused


NOTICE = """# NOT FOR THE FINAL WORK

Everything in this folder was produced by SIMULATORS (agent types `su2` and `sv`).
It is experiment 10 (docs/EXPERIMENT-10-X100.md): a 100-seed extension.

By the author's rule (2026-10-21) simulator results are EXPLORATORY and must NOT enter the
final work - not in a table, a figure, a number in the text or an argument - unless the author
has approved it explicitly, for that specific use. The only approval so far is the validation
battery (2026-11-01); nothing in this folder is approved.

Policy: docs/SIMULATED-RESULTS-POLICY.md in the repository.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--results-root", required=True, help="folder that contains the experiment directories")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: 1-100")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--claim", action="store_true", help="cooperative locking: several workers per group")
    p.add_argument("--list", action="store_true", help="print the cells to run and exit")
    p.add_argument("--smoke", action="store_true", help="1 seed, 5k steps, _smoke dir, no reuse")
    args = p.parse_args()

    root = pathlib.Path(args.results_root).resolve()
    outdir = root / SUBDIR
    steps = 5_000 if args.smoke else args.steps
    seeds = [1] if args.smoke else args.seeds
    eval_every = max(1, steps // 2) if args.smoke else args.eval_every
    if args.smoke:
        outdir = outdir / "_smoke"
    specs, reused = build_specs(args.group, steps, None if args.smoke else root, seeds)
    n_reused = sum(len(v) for v in reused.values())
    if args.list:
        for sp in specs:
            print(sp.run_name)
        print(f"{len(specs)} cells to run, {n_reused} seeds reused from existing runs")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "NOT_FOR_THESIS.md").write_text(NOTICE, encoding="utf-8")
    if not args.smoke and args.seeds is None:      # a --seeds subset must not overwrite the full-range list
        rf = outdir / "reused_seeds.json"
        prev = json.loads(rf.read_text()) if rf.exists() else {}
        for arm, v in reused.items():
            prev[arm] = {str(s): list(src) for s, src in sorted(v.items())}
        rf.write_text(json.dumps(prev, indent=1, sort_keys=True))
    envs = {env for _a, _k, env in group_arms(args.group)}
    assert len(envs) == 1
    env_id = envs.pop()
    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    print(f"group {args.group}: {len(specs)} cells to run ({n_reused} seeds reused), steps={steps}, "
          f"env={env_id}, outdir={outdir}", flush=True)
    run_grid(specs, outdir, env_id=env_id,
             eval_cfg=GreedyEvalConfig(env_id=env_id, every_steps=eval_every, n_episodes=20),
             progress_bar=True, verbose=True, claim=args.claim, claim_ttl_hours=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
