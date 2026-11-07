"""Experiment 13, block D - sensitivity control of the float64 PPO trace.

The float64 traces of P7/P8 at R = 32 end up separated (first differing action at 2.7e4-5.1e4 steps) although the two nets are
algebraically equivalent. This control runs the SAME arm twice (ppo_reuse_noclip_r{R}, same seed), the second time with every
agent parameter shifted by a random 1e-14 before training. Same measures as exp13_trace64_ppo.py. If the perturbed copy
separates at a similar step and with similar parameter differences, the separation of the equivalent pairs is the amplification of
rounding error by the training dynamics, not a difference between the nets.

    python experiments/exp13_trace64_ppo_perturb.py --out <file.json> --workdir <dir> --R 8 32 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

import exp13_trace64_ppo as T  # noqa: E402  (same folder)

LP, e13 = T.LP, T.e13


def _run(arm_name, seed, steps, workdir, eps):
    orig_init = LP.LRPPO.train
    if eps:
        def train(self, *a, **k):
            g = torch.Generator().manual_seed(12345)
            with torch.no_grad():
                for p in self.agent.parameters():
                    p.add_(eps * torch.randn(p.shape, generator=g, dtype=p.dtype))
            return orig_init(self, *a, **k)
        LP.LRPPO.train = train
    try:
        return T._run(arm_name, seed, steps, workdir)
    finally:
        LP.LRPPO.train = orig_init


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--R", nargs="+", type=int, default=[8, 32])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eps", type=float, default=1e-14)
    args = p.parse_args()
    out = pathlib.Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    for R in args.R:
        for s in args.seeds:
            key = f"perturb_noclip_R{R}_s{s}"
            if key in res:
                continue
            t0 = time.time()
            name = f"ppo_reuse_noclip_r{R}"
            ra = _run(name, s, args.steps, pathlib.Path(args.workdir) / "a", 0.0)
            rb = _run(name, s, args.steps, pathlib.Path(args.workdir) / "b", args.eps)
            res[key] = dict(T.compare(ra, rb), pair="perturb_noclip", a=name, b=name + f"+{args.eps:g}", R=R, seed=s,
                            wall_s=round(time.time() - t0, 1))
            print(key, res[key], flush=True)
            out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
