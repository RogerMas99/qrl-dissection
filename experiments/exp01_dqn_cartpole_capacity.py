"""
Experiment 01 - Capacity-matched classical control under DQN.

Question
--------
The paper's classical control arm is the hybrid arm minus the PQC, with matched
input dimensionality and parameter count. Under PPO both arms learn and the
comparison is clean. Under DQN the classical arm dies at ~9.7, below a random
policy, so any gap to the hybrid arm is uninterpretable: we would be comparing a
live agent with a dead one.

This experiment adds an arm in which the PQC is replaced by a CLASSICAL block of
the same measured parameter count, and crosses every arm with FIX-01 (the
autoreset correction) on and off.

Reading the outcomes
--------------------
matched_classical LEARNS  -> the death of paper_linear was capacity, not the
                             absence of a circuit. We also finally gain a second
                             live configuration in which FIX-01 is measurable.
matched_classical DIES    -> at equal parameter budget the classical block does
                             not reach where the PQC does. That is a positive
                             result for the circuit and the strongest available
                             claim about it under DQN.

Either way it is publishable. Run it before anything else.

Usage
-----
    python experiments/exp01_capacity_matched_control.py --outdir results/exp01
    python experiments/exp01_capacity_matched_control.py --ladder-only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import qrl_dissection
from qrl_dissection import ARMS, build_arm_config, capacity_ladder
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

CLASSICAL_ARMS = ["paper_linear", "matched_classical", "oversized_mlp"]
DEFAULT_SEEDS = [1, 2, 3]


def build_specs(args) -> list:
    specs = []
    for arm in args.arms:
        for fix in (False, True):
            for seed in args.seeds:
                specs.append(RunSpec(
                    arm=arm,
                    seed=seed,
                    fix_autoreset=fix,
                    total_timesteps=args.steps,
                    dqn_kwargs={
                        "batch_size": args.batch_size,
                        "buffer_size": args.buffer_size,
                        "train_frequency": args.train_frequency,
                    },
                    tag=args.tag,
                ))
    return specs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--claim", action="store_true",
                   help="cooperative locking across parallel sessions")
    p.add_argument("--outdir", default="results/exp01_dqn_cartpole_capacity")
    p.add_argument("--arms", nargs="+", default=CLASSICAL_ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=10_000)
    p.add_argument("--train-frequency", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--tag", default="")
    p.add_argument("--ladder-only", action="store_true",
                   help="print the parameter accounting and exit; trains nothing")
    args = p.parse_args()

    print(json.dumps(qrl_dissection.upstream_report(), indent=2))

    # Always print the capacity ladder first. If an arm does not build what you
    # think it builds, you want to know before spending compute.
    arms_for_ladder = {}
    for arm in set(args.arms) | {"hybrid_fig4"}:
        try:
            arms_for_ladder[arm] = build_arm_config(arm)
        except Exception as exc:
            print(f"  ! could not build {arm}: {exc}")
    print("\n=== capacity ladder ===")
    for row in capacity_ladder(arms_for_ladder):
        print(f"\n{row['arm']}  ({row['agent_type']})  "
              f"trainable params: {row['trainable_params']:,}")
        if "pqc_quantum" in row:
            print(f"    quantum: {row['pqc_quantum']:,}   "
                  f"classical head: {row['pqc_classical_head']:,}")
        print("   ", row["structure"].replace("\n", "\n    "))

    if args.ladder_only:
        return 0

    outdir = pathlib.Path(args.outdir)
    specs = build_specs(args)
    print(f"\n=== running {len(specs)} cells into {outdir} ===")
    results = run_grid(specs, outdir,
                       eval_cfg=GreedyEvalConfig(every_steps=args.eval_every, claim=args.claim))

    failures = [r for r in results if "error" in r]
    print(f"\ndone: {len(results) - len(failures)} ok, {len(failures)} failed")
    for r in failures:
        print("  FAILED", r["run_name"], r["error"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
