"""Experiment 13 - float64 trace of a full training: reuse R against collapsed R (docs/EXPERIMENT-13-OR-VS-LR.md, P1).

For one agent (twin or hybrid-on-emulator), R and seed, trains the reuse-R arm and the collapsed-R arm (V0 = sum_r W_r0,
lr of head.weight x R) with the whole network in float64 and the full exp12 protocol (100 000 steps, no greedy
evaluation: only the training path is compared). Because upstream's exploration sampler is seeded by the arm hook, the
two runs differ only through the arithmetic, so they follow the same trajectory until a rounding difference flips a
greedy action.

Reports, per (agent, R, seed):
  first_diff_step     first environment step at which the action taken differs between the two runs (null = never);
                      with the exploration draws shared, it is the first greedy action that differs
  max_param_diff_to_first_diff   max over the traced steps (every 100) up to that step of
                      max |theta_collapsed - theta_reuse_effective|, theta_effective having V = sum_r W_r
  max_param_diff_final            same, at the last traced step (meaningless after the divergence)
  n_steps

    python experiments/exp13_trace64.py --out <file.json> --agent twin --R 8 32 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

import qrl_dissection  # noqa: F401
from qrl_dissection.core import lr_equivalence as L
from qrl_dissection.core.configs import build_arm_config
from qrl_dissection.dqn import GreedyEvalConfig, SafeDQN

ENV = "CartPole-v1"
KW = {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}
TRACE_EVERY = 100


def _run(arm: str, seed: int, steps: int, outdir: pathlib.Path):
    at, cfg = build_arm_config(arm, env_id=ENV)
    cfg = dict(cfg, lr13_float64=True, lr13_trace_every=TRACE_EVERY)
    holder = []
    pb = L.make_post_build(cfg)

    def pb2(dqn):
        h = pb(dqn)
        holder.append(h)
        return h

    SafeDQN(agent_type=at, agent_config=cfg, run_name=f"{arm}__s{seed}__t64", seed=seed, env_id=ENV,
            fix_autoreset=True, eval_cfg=GreedyEvalConfig(enabled=False), outdir=outdir, post_build=pb2,
            **KW).train(steps)
    return holder[0]


def compare(h_re, h_co) -> dict:
    a, b = h_re.actions, h_co.actions
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    steps = np.asarray(h_re.trace_steps)
    diffs = np.asarray([np.abs(x - y).max() for x, y in zip(h_re.trace_params, h_co.trace_params)])
    if first is None:
        upto = diffs
    else:
        upto = diffs[steps <= first]            # step i of the action list is trace step i + 1
    return {"first_diff_step": first,
            "max_param_diff_to_first_diff": float(upto.max()) if upto.size else None,
            "max_param_diff_final": float(diffs[-1]) if diffs.size else None,
            "n_steps": n}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", required=True, help="scratch folder for the run files")
    p.add_argument("--agent", choices=["twin", "hybrid"], required=True)
    p.add_argument("--R", nargs="+", type=int, default=[8, 32])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=100_000)
    args = p.parse_args()
    pre = L.PREFIX_TWIN if args.agent == "twin" else L.PREFIX_HYBRID
    out = pathlib.Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    work = pathlib.Path(args.workdir)
    for R in args.R:
        for s in args.seeds:
            key = f"{args.agent}_R{R}_s{s}"
            if key in res:
                continue
            t0 = time.time()
            h_re = _run(f"{pre}_reuse_r{R}", s, args.steps, work / "re")
            h_co = _run(f"{pre}_collapsed_r{R}", s, args.steps, work / "co")
            res[key] = dict(compare(h_re, h_co), agent=args.agent, R=R, seed=s, wall_s=round(time.time() - t0, 1))
            print(key, res[key], flush=True)
            out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
