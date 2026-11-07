"""Experiment 13, blocks E, F, G (addendum) - classical controls that contain Output Reuse, and a small MLP on bits.

  E  CartPole, additive Fourier ceiling: ceiling_dup (80 columns, 162 params; wires i and i+4 share a variable, so it is
     Output Reuse R = 2), ceiling_unique (40 columns, 82), ceiling_unique_lrx2 (40 columns, lr of the weight x 2),
     ceiling_unique_lrx2_collapsed (the dup model with its head collapsed, V0 = sum of the 2 blocks, lr x 2).
  F  CartPole, capacity-matched control of RQ2: matched_rep4_w7 (input x4, Linear(16->7), 135 params),
     matched_norep_w7_lrx4 (Linear(4->7), lr of the first-layer weight x 4, 51), matched_norep_w7_lrx4_collapsed
     (rep4 model collapsed, V0 = sum of the 4 blocks), matched_norep_w18 (128 params), matched_norep_w18_lrx4.
  G  FrozenLake 4x4 (deterministic), DQN: mlp_bits_w6 (58 params) and mlp_bits_w12 (112) on the 4 bits in {0, 1}.

Protocol: seeds 1-30, 100 000 steps, greedy evaluation every 10 000 steps with 20 episodes, FIX-01 on.
E, F: CartPole DQN kwargs of exp12 (batch 128, buffer 10 000, train_frequency 10).
G: FrozenLake DQN kwargs of exp04 (batch 128, buffer 50 000, train_frequency 1, learning_starts 1 000).
Own folder <results-root>/exp13_efg/, tag "exp13efg".

    python experiments/exp13_efg.py --results-root <results> --eval-episodes 20 --block EFG --shard 0 8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Tuple

import qrl_dissection
from qrl_dissection.core import FROZEN_SCALAR_ID, lr_efg
from qrl_dissection.core.capacity import build_agent_for, count_trainable
from qrl_dissection.core.configs import build_arm_config
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

SUBDIR = "exp13_efg"
TAG = "exp13efg"
SEEDS = list(range(1, 31))
CARTPOLE = "CartPole-v1"
KW_CARTPOLE = {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}
KW_FROZEN = {"batch_size": 128, "buffer_size": 50_000, "train_frequency": 1, "learning_starts": 1_000}
ENV_OF = {"E": CARTPOLE, "F": CARTPOLE, "G": FROZEN_SCALAR_ID}
KW_OF = {"E": KW_CARTPOLE, "F": KW_CARTPOLE, "G": KW_FROZEN}


def build_specs(steps: int, seeds: List[int], block: str) -> List[Tuple[str, RunSpec]]:
    return [(b, RunSpec(arm=a, seed=s, fix_autoreset=True, total_timesteps=steps, dqn_kwargs=dict(KW_OF[b]), tag=TAG))
            for b in block for a in lr_efg.BLOCK_ARMS[b] for s in seeds]


def design_record() -> dict:
    rec = {}
    for b, arms in lr_efg.BLOCK_ARMS.items():
        for n in arms:
            at, cfg = build_arm_config(n, env_id=ENV_OF[b])
            agent = build_agent_for(at, cfg, env_id=ENV_OF[b], is_qnet=True)
            rec[n] = {"block": b, "agent_type": at, "params": count_trainable(agent), "env": ENV_OF[b],
                      "efg_R": cfg.get("lr13_efg_R", 1), "collapse_R": cfg.get("lr13_collapse_R", 0),
                      "head_lr_mult": cfg.get("lr13_head_mult", 1.0), "dqn_kwargs": KW_OF[b], "simulated": False}
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True)
    p.add_argument("--eval-episodes", type=int, required=True, help="episodes per greedy evaluation (exp13: 20)")
    p.add_argument("--block", default="EFG", help="letters among E, F, G")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: 1-30")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 seed, 5k steps, _smoke dir")
    args = p.parse_args()

    root = pathlib.Path(args.results_root).resolve()
    outdir = root / SUBDIR
    steps = 5_000 if args.smoke else args.steps
    seeds = [1] if args.smoke else (args.seeds or SEEDS)
    eval_every = max(1, steps // 2) if args.smoke else args.eval_every
    if args.smoke:
        outdir = outdir / "_smoke"
    items = build_specs(steps, seeds, args.block)
    if args.shard:
        i, n = args.shard
        items = [it for k, it in enumerate(items) if k % n == i]
    if args.list:
        for _, sp in items:
            print(sp.run_name)
        print(f"{len(items)} cells")
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    rec = design_record()
    rec["_eval_episodes"] = args.eval_episodes
    dp = outdir / "design_params.json"
    if not dp.exists():
        dp.write_text(json.dumps(rec, indent=1))
    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    print(f"{len(items)} cells, steps={steps}, outdir={outdir}", flush=True)
    for b in args.block:
        specs = [sp for bb, sp in items if bb == b]
        if not specs:
            continue
        run_grid(specs, outdir, env_id=ENV_OF[b],
                 eval_cfg=GreedyEvalConfig(env_id=ENV_OF[b], every_steps=eval_every, n_episodes=args.eval_episodes),
                 progress_bar=False, verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
