"""
Experiment 09 - Output Reuse x depth sweep on FrozenLake, on the SU(2) emulator.

EXPLORATORY EXTENSION - NOT PART OF THE THESIS CORE. It explores extra options the
core design (exp01-exp08) did not cover; results are reported separately and
labelled exploratory. See the banner in docs/EXPERIMENT-09.md.

Design and the pre-specified contrasts: docs/EXPERIMENT-09.md.

Why the emulator
----------------
`SU2SkolikEmulator` reproduces the UNENTANGLED skolik circuit exactly (NEW-05) at
a fraction of the cost: 3-8 minutes per 100k-step FrozenLake cell, against 7-25
HOURS for the real PennyLane circuit. It cannot represent an entangled circuit,
so this sweep says nothing about entanglement: that is phase 2, on the real
circuit, once this sweep shows where Output Reuse or depth matters.

Groups
------
F   Config A (1 qubit, scalar embedding): L in {1, 5, 10, 15} x R in {1, 4, 16}
G   Config B (4 qubits, binary embedding, ent=False): L in {1, 2, 3, 5} x R in {1, 4, 16}
    (at L=1 the entangled and unentangled circuits are the same circuit, FIX-07)
I   Protocol sensitivity: the 8 combinations of train_frequency {1, 10} x
    learning_starts {1k, 10k} x buffer_size {50k, 10k} on two L=5 arms, on
    SEPARATE seeds (101-103), so the primary seeds are never used to choose
    anything. Reported as a sensitivity analysis next to F and G, which run
    on exp04's own protocol so that they stay comparable with every existing
    FrozenLake result.

Every cell is written with the reuse guard's usual key. Group I carries the
protocol in `tag`. Run several workers of the same group with `--claim`: they
split the cells dynamically (cooperative locks, TTL 2 h).

Usage
-----
    python experiments/exp09_dqn_emulator_sweep.py --group F --results-root <dir> --claim
    python experiments/exp09_dqn_emulator_sweep.py --group I --results-root <dir> --claim
    python experiments/exp09_dqn_emulator_sweep.py --group G --smoke --results-root <tmp dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import qrl_dissection
from qrl_dissection.core.configs import (
    EXP09_CONFIG_A_DEPTHS,
    EXP09_CONFIG_B_DEPTHS,
    EXP09_FROZEN_REUSE,
    su2_frozen_arm_name,
)
from qrl_dissection.core.obs_adapters import FROZEN_SCALAR_ID
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

# exp04's own FrozenLake protocol, copied VERBATIM (also exp05 and
# experiments/rerun_controlled_arms.py): F and G must stay comparable with every
# FrozenLake result already on disk.
PRIMARY_KWARGS: Dict[str, object] = {
    "batch_size": 128,
    "buffer_size": 50_000,
    "train_frequency": 1,
    "learning_starts": 1_000,
}
PRIMARY_SEEDS = list(range(1, 11))
SENSITIVITY_SEEDS = [101, 102, 103]
SENSITIVITY_ARMS = ["su2_frozen_scalar_1q_L5", "su2_frozen_binary_4q_L5"]
SUBDIR = "exp09_dqn_emulator_sweep"


def _k(n: int) -> str:
    return f"{n // 1000}k"


def sensitivity_variants() -> List[Tuple[str, Dict[str, object]]]:
    out = []
    for tf, ls, buf in itertools.product((1, 10), (1_000, 10_000), (50_000, 10_000)):
        kw = dict(PRIMARY_KWARGS, train_frequency=tf, learning_starts=ls, buffer_size=buf)
        out.append((f"p_tf{tf}_ls{_k(ls)}_buf{_k(buf)}", kw))
    return out


def build_specs(group: str, steps: int, seeds: List[int] = None) -> List[RunSpec]:
    specs: List[RunSpec] = []
    if group == "F":
        arms = [su2_frozen_arm_name("scalar_1q", L, R)
                for L in EXP09_CONFIG_A_DEPTHS for R in EXP09_FROZEN_REUSE]
    elif group == "G":
        arms = [su2_frozen_arm_name("binary_4q", L, R)
                for L in EXP09_CONFIG_B_DEPTHS for R in EXP09_FROZEN_REUSE]
    elif group == "I":
        arms = None
    else:
        raise ValueError(group)

    if group in ("F", "G"):
        for arm in arms:
            for s in (seeds or PRIMARY_SEEDS):
                specs.append(RunSpec(arm=arm, seed=s, fix_autoreset=True, total_timesteps=steps,
                                     dqn_kwargs=dict(PRIMARY_KWARGS)))
    else:
        for tag, kw in sensitivity_variants():
            for arm in SENSITIVITY_ARMS:
                for s in (seeds or SENSITIVITY_SEEDS):
                    specs.append(RunSpec(arm=arm, seed=s, fix_autoreset=True,
                                         total_timesteps=steps, dqn_kwargs=dict(kw), tag=tag))
    return specs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=["F", "G", "I"], required=True)
    p.add_argument("--results-root", required=True,
                   help="folder that contains the experiment directories")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="default: 1-10 for F and G, 101-103 for I")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--claim", action="store_true",
                   help="cooperative locking: run several workers of one group")
    p.add_argument("--list", action="store_true", help="print the cells and exit")
    p.add_argument("--smoke", action="store_true", help="1 seed, 5k steps, _smoke dir")
    args = p.parse_args()

    outdir = pathlib.Path(args.results_root).resolve() / SUBDIR
    steps = 5_000 if args.smoke else args.steps
    seeds = [1] if args.smoke else args.seeds
    eval_every = max(1, steps // 2) if args.smoke else args.eval_every
    if args.smoke:
        outdir = outdir / "_smoke"

    specs = build_specs(args.group, steps, seeds)
    if args.list:
        for sp in specs:
            print(sp.run_name)
        print(f"{len(specs)} cells")
        return 0

    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    print(f"group {args.group}: {len(specs)} cells, steps={steps}, outdir={outdir}", flush=True)
    run_grid(specs, outdir, env_id=FROZEN_SCALAR_ID,
             eval_cfg=GreedyEvalConfig(env_id=FROZEN_SCALAR_ID, every_steps=eval_every,
                                       n_episodes=20),
             progress_bar=True, verbose=True, claim=args.claim, claim_ttl_hours=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
