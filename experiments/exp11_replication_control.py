"""Experiment 11 - classical control that replicates its input (docs/EXPERIMENT-11-REPLICATION-CONTROL.md).

Question: does replicating the input R times help a classical network that CAN learn? The two existing
controls cannot answer it (classical_OR{R} is linear and dies; classical_or_matched_r{R} replicates 4 times
whatever R is). The new arm `classical_or_matched_rep_r{R}` replicates R times, has a ReLU hidden layer, sees
the full observation and matches the MEASURED total of hybrid_OR{R}.

Real environment, classical arms only: no simulator is involved.

R = 4 is the same specification as the existing `classical_or_matched_r4` (exp02), so it is NOT re-run:
this script checks that identity and runs R in {8, 16, 32}. Results go to <results-root>/exp11_replication_control/
with tag "exp11". Protocol as exp02 (DQN, CartPole-v1, 100 000 steps, batch 128, buffer 10 000, train_frequency 10,
library defaults for the rest, FIX-01 on, seeds 1-10); the number of episodes of each greedy evaluation is a
required argument because exp02 used the library default (5), not 20.

    python experiments/exp11_replication_control.py --results-root <results> --eval-episodes 5 --claim
    python experiments/exp11_replication_control.py --results-root <results> --eval-episodes 5 --list
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List

import qrl_dissection
from qrl_dissection.core.capacity import build_agent_for, count_trainable
from qrl_dissection.core.configs import build_arm_config, hybrid_or_config
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

SUBDIR = "exp11_replication_control"
TAG = "exp11"
RS_NEW = (8, 16, 32)
SEEDS = list(range(1, 11))
KWARGS = {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}     # exp02, verbatim
ENV = "CartPole-v1"


def arm_name(R: int) -> str:
    return f"classical_or_matched_rep_r{R}"


def build_specs(steps: int, seeds: List[int] = None) -> List[RunSpec]:
    return [RunSpec(arm=arm_name(R), seed=s, fix_autoreset=True, total_timesteps=steps,
                    dqn_kwargs=dict(KWARGS), tag=TAG)
            for s in (seeds or SEEDS) for R in RS_NEW]


def _params(agent_type, cfg) -> int:
    return count_trainable(build_agent_for(agent_type, cfg, env_id=ENV, is_qnet=True))


def design_record() -> dict:
    """Widths and MEASURED parameter counts of every arm involved, and the R = 4 identity check."""
    rec = {}
    for R in (4, 8, 16, 32):
        hy = _params("hybrid", hybrid_or_config(R))
        ot, oc = build_arm_config(f"classical_or_matched_r{R}", env_id=ENV)
        nt, nc = build_arm_config(arm_name(R), env_id=ENV)
        rec[str(R)] = {
            "hybrid_OR_total": hy,
            "old_matched": {"width": oc["net_arch"][0], "n_repeats": oc["n_repeats"], "params": _params(ot, oc)},
            "new_rep": {"width": nc["net_arch"][0], "n_repeats": nc["n_repeats"], "params": _params(nt, nc)},
        }
    keys = ("reuse_indices", "n_repeats", "net_arch", "activation", "_budget")
    _, o4 = build_arm_config("classical_or_matched_r4", env_id=ENV)
    _, n4 = build_arm_config(arm_name(4), env_id=ENV)
    rec["r4_identical_to_classical_or_matched_r4"] = {k: o4[k] for k in keys} == {k: n4[k] for k in keys}
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True, help="folder that contains the experiment directories")
    p.add_argument("--eval-episodes", type=int, required=True,
                   help="episodes per greedy evaluation (exp02 used the library default, 5)")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: 1-10")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--claim", action="store_true", help="cooperative locking: several workers")
    p.add_argument("--list", action="store_true", help="print the cells to run and exit")
    p.add_argument("--smoke", action="store_true", help="1 seed, 5k steps, _smoke dir")
    args = p.parse_args()

    root = pathlib.Path(args.results_root).resolve()
    outdir = root / SUBDIR
    steps = 5_000 if args.smoke else args.steps
    seeds = [1] if args.smoke else args.seeds
    eval_every = max(1, steps // 2) if args.smoke else args.eval_every
    if args.smoke:
        outdir = outdir / "_smoke"
    specs = build_specs(steps, seeds)
    if args.list:
        for sp in specs:
            print(sp.run_name)
        print(f"{len(specs)} cells")
        return 0

    rec = design_record()
    if not rec["r4_identical_to_classical_or_matched_r4"]:
        print("R=4 is NOT identical to classical_or_matched_r4: it must be run as its own arm. Stopping.")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    rec["eval_episodes"] = args.eval_episodes
    (outdir / "design_widths_params.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    print(f"{len(specs)} cells, steps={steps}, eval every {eval_every} x {args.eval_episodes} episodes, outdir={outdir}",
          flush=True)
    run_grid(specs, outdir, env_id=ENV,
             eval_cfg=GreedyEvalConfig(env_id=ENV, every_steps=eval_every, n_episodes=args.eval_episodes),
             progress_bar=False, verbose=True, claim=args.claim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
