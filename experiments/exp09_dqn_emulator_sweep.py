"""
Experiment 09 - Output Reuse x depth sweeps on FrozenLake and CartPole, on emulators.

EXPLORATORY EXTENSION - NOT PART OF THE THESIS CORE. It explores extra options the
core design (exp01-exp08) did not cover; results are reported separately and
labelled exploratory. See the banner in docs/EXPERIMENT-09.md. Nothing here replaces
a run planned for the core: where a real-circuit arm exists, the emulated one is a
replica under its own name.

Why emulators
-------------
`SU2SkolikEmulator` reproduces the UNENTANGLED skolik circuit exactly (NEW-05, init
matched in FIX-12); `SVSkolikLayer` (core/statevector.py, NEW-09) reproduces it ENTANGLED
too, as one batched tensor program instead of a per-sample PennyLane call. Per 100k-step
FrozenLake cell that is minutes against 7-25 hours.

Groups (wave 1)
---------------
F   Config A (1 qubit, scalar embedding), su2: L in {1, 5, 10, 15} x R in {1, 4, 16}
G   Config B (4 qubits, binary), su2 / ent=False: L in {1, 2, 3, 5} x R in {1, 4, 16}
I   Protocol sensitivity: train_frequency {1, 10} x learning_starts {1k, 10k} x
    buffer_size {50k, 10k} on two L=5 arms, SEPARATE seeds 101-103.

Groups (wave 2)
---------------
P   Config B ENTANGLED, sv: L in {1, 2, 3, 5} x R in {1, 4, 16}. Phase 2: Output Reuse
    x entanglement on FrozenLake. L=1 is the control (entangled == unentangled, FIX-07).
V   CartPole fidelity check, sv (entangled): `HYBRID_FIG4` with R in {1, 4, 8, 16, 32},
    10 seeds. Replicates exp01/exp02's real entangled arms under another name.
K   CartPole extras, su2: Output Reuse without entanglement (R = 4..32), the output
    scale (FIX-02) at R = 1 and 16, and learning rate {1e-3, 4e-3} at R = 1 and 16.
H   FrozenLake axes never varied, one at a time from the baseline, on the dead arms
    (Config A L5 / L10, Config B L5 unentangled): head16, init0.1, init0.5, trainable
    input scale (arm variants) and learning rate {1e-3, 4e-3} (tags). Seeds 1-5, so
    each cell is paired with the baseline of group F.

Groups (validation battery, docs/EXPERIMENT-09.md section 7)
------------------------------------------------------------
Q   sv, CartPole, depth L in {1, 2} x R in {1, 4, 8, 16, 32}: replicas of exp03 / exp08 real arms
R   su2 L in {1, 2} (replicas of exp03b) and sv with ent=False at L=5 (must match su2)
S   sv / su2 replicas of exp07's literal Skolik arms (4 and 8 qubits)
T   FrozenLake sv with ent=False (must match su2)

Every cell carries the reuse guard's usual key; groups I, K and H carry the variant in
`tag`. Run several workers of the same group with `--claim`: they split the cells
dynamically (cooperative locks, TTL 2 h).

Usage
-----
    python experiments/exp09_dqn_emulator_sweep.py --group P --results-root <dir> --claim
    python experiments/exp09_dqn_emulator_sweep.py --group V --smoke --results-root <tmp dir>
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
    EXP09_CARTPOLE_REUSE,
    EXP09_CONFIG_A_DEPTHS,
    EXP09_CONFIG_B_DEPTHS,
    EXP09_FROZEN_REUSE,
    EXP09_H_BASES,
    EXP09_H_VARIANTS,
    cartpole_replica_name,
    su2_frozen_arm_name,
    sv_frozen_arm_name,
)
from qrl_dissection.core.obs_adapters import FROZEN_SCALAR_ID
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

# exp04's own FrozenLake protocol, copied VERBATIM (also exp05 and
# experiments/rerun_controlled_arms.py): F, G, P and H must stay comparable with every
# FrozenLake result already on disk.
PRIMARY_KWARGS: Dict[str, object] = {
    "batch_size": 128,
    "buffer_size": 50_000,
    "train_frequency": 1,
    "learning_starts": 1_000,
}
# The CartPole experiments' protocol (exp01/exp02/exp03/exp07/exp08).
CARTPOLE_KWARGS: Dict[str, object] = {
    "batch_size": 128,
    "buffer_size": 10_000,
    "train_frequency": 10,
}
PRIMARY_SEEDS = list(range(1, 11))
SENSITIVITY_SEEDS = [101, 102, 103]
H_SEEDS = [1, 2, 3, 4, 5]
SENSITIVITY_ARMS = ["su2_frozen_scalar_1q_L5", "su2_frozen_binary_4q_L5"]
LR_VALUES = (1e-3, 4e-3)
SUBDIR = "exp09_dqn_emulator_sweep"
GROUPS = ["F", "G", "I", "P", "V", "K", "H", "Q", "R", "S", "T"]
CARTPOLE_GROUPS = {"V", "K", "Q", "R", "S"}


def group_env(group: str) -> str:
    return "CartPole-v1" if group in CARTPOLE_GROUPS else FROZEN_SCALAR_ID


def _k(n: int) -> str:
    return f"{n // 1000}k"


def sensitivity_variants() -> List[Tuple[str, Dict[str, object]]]:
    out = []
    for tf, ls, buf in itertools.product((1, 10), (1_000, 10_000), (50_000, 10_000)):
        kw = dict(PRIMARY_KWARGS, train_frequency=tf, learning_starts=ls, buffer_size=buf)
        out.append((f"p_tf{tf}_ls{_k(ls)}_buf{_k(buf)}", kw))
    return out


def _lr_tag(lr: float) -> str:
    return f"lr{lr:g}"


def build_specs(group: str, steps: int, seeds: List[int] = None) -> List[RunSpec]:
    specs: List[RunSpec] = []

    def add(arm, kw, sds, tag=""):
        for s in sds:
            specs.append(RunSpec(arm=arm, seed=s, fix_autoreset=True, total_timesteps=steps,
                                 dqn_kwargs=dict(kw), tag=tag))

    if group == "F":
        for L in EXP09_CONFIG_A_DEPTHS:
            for R in EXP09_FROZEN_REUSE:
                add(su2_frozen_arm_name("scalar_1q", L, R), PRIMARY_KWARGS, seeds or PRIMARY_SEEDS)
    elif group == "G":
        for L in EXP09_CONFIG_B_DEPTHS:
            for R in EXP09_FROZEN_REUSE:
                add(su2_frozen_arm_name("binary_4q", L, R), PRIMARY_KWARGS, seeds or PRIMARY_SEEDS)
    elif group == "I":
        for tag, kw in sensitivity_variants():
            for arm in SENSITIVITY_ARMS:
                add(arm, kw, seeds or SENSITIVITY_SEEDS, tag)
    elif group == "P":
        for L in EXP09_CONFIG_B_DEPTHS:
            for R in EXP09_FROZEN_REUSE:
                add(sv_frozen_arm_name(L, R), PRIMARY_KWARGS, seeds or PRIMARY_SEEDS)
    elif group == "V":
        for R in EXP09_CARTPOLE_REUSE:
            add(cartpole_replica_name("sv", R), CARTPOLE_KWARGS, seeds or PRIMARY_SEEDS)
    elif group == "K":
        sds = seeds or PRIMARY_SEEDS
        for R in (4, 8, 16, 32):                                    # OR without entanglement
            add(cartpole_replica_name("su2", R), CARTPOLE_KWARGS, sds)
        for R in (1, 16):                                           # FIX-02 output scale
            add(cartpole_replica_name("su2", R, "_outscale"), CARTPOLE_KWARGS, sds)
        for R in (1, 16):                                           # learning rate
            for lr in LR_VALUES:
                add(cartpole_replica_name("su2", R), dict(CARTPOLE_KWARGS, learning_rate=lr),
                    sds, _lr_tag(lr))
    elif group == "H":
        sds = seeds or H_SEEDS
        for base in EXP09_H_BASES:
            for vname in EXP09_H_VARIANTS:
                add(f"{base}__{vname}", PRIMARY_KWARGS, sds)
            for lr in LR_VALUES:
                add(base, dict(PRIMARY_KWARGS, learning_rate=lr), sds, _lr_tag(lr))
    elif group == "Q":          # battery: entangled CartPole depth x OR, vs exp03 / exp08 real arms
        for L in (1, 2):
            for R in EXP09_CARTPOLE_REUSE:
                add(cartpole_replica_name("sv", R, n_layers=L), CARTPOLE_KWARGS, seeds or PRIMARY_SEEDS)
    elif group == "R":          # battery: unentangled depth twins + sv(ent=False) vs su2
        sds = seeds or PRIMARY_SEEDS
        for L in (1, 2):
            add(cartpole_replica_name("su2", 1, n_layers=L), CARTPOLE_KWARGS, sds)
        add("sv_cartpole_L5_noent", CARTPOLE_KWARGS, sds)
    elif group == "S":          # battery: exp07's literal Skolik arms
        sds = seeds or PRIMARY_SEEDS
        for nq in (4, 8):
            for L in (1, 2, 5):
                add(f"sv_paper_skolik_{nq}q_L{L}", CARTPOLE_KWARGS, sds)
        for L in (2, 5):
            add(f"su2_paper_skolik_4q_L{L}_noent", CARTPOLE_KWARGS, sds)
    elif group == "T":          # battery: FrozenLake, sv(ent=False) vs su2
        for L in (1, 5):
            add(f"sv_frozen_binary_4q_noent_L{L}", PRIMARY_KWARGS, seeds or PRIMARY_SEEDS)
    else:
        raise ValueError(group)
    return specs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--results-root", required=True,
                   help="folder that contains the experiment directories")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="default per group: 1-10 (F, G, P, V, K), 101-103 (I), 1-5 (H)")
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

    env_id = group_env(args.group)
    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    print(f"group {args.group}: {len(specs)} cells, steps={steps}, env={env_id}, outdir={outdir}",
          flush=True)
    run_grid(specs, outdir, env_id=env_id,
             eval_cfg=GreedyEvalConfig(env_id=env_id, every_steps=eval_every, n_episodes=20),
             progress_bar=True, verbose=True, claim=args.claim, claim_ttl_hours=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
