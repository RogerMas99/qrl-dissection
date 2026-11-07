"""Experiment 13, block D - float64 trace of a full PPO training (docs/EXPERIMENT-13-OR-VS-LR.md, P7 and P8).

Pairs (full 100 000-step training, agent in float64, same seed):
  P8:  ppo_reuse_clip_r{R}    against  ppo_lrx_modclip_r{R}   (clipping 0.5; modified norm for the collapsed net)
  P7:  ppo_reuse_noclip_r{R}  against  ppo_lrx_noclip_r{R}    (no clipping)
  ref: ppo_reuse_clip_r{R}    against  ppo_lrx_clip_r{R}      (usual rule: expected NOT to follow)
PPO draws its actions from torch's seeded RNG, so with equal streams the two runs differ only through arithmetic. Per
pair: first rollout step at which the sampled action differs (null = never), max |theta_a - theta_b| over the effective
parameters (V = sum_r W_r) at the iteration ends up to that point, and the fraction of updates in which the clipping fired.

    python experiments/exp13_trace64_ppo.py --out <file.json> --workdir <dir> --R 8 32 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

import qrl_dissection  # noqa: F401
from qrl_dissection.ppo import lr_ppo as LP

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp13_ppo", HERE / "exp13_ppo.py")
e13 = importlib.util.module_from_spec(_spec)
sys.modules["exp13_ppo"] = e13
_spec.loader.exec_module(e13)

PAIRS = {"P8_clip_vs_modclip": ("ppo_reuse_clip_r{R}", "ppo_lrx_modclip_r{R}"),
         "P7_noclip": ("ppo_reuse_noclip_r{R}", "ppo_lrx_noclip_r{R}"),
         "ref_clip_vs_usualclip": ("ppo_reuse_clip_r{R}", "ppo_lrx_clip_r{R}")}


def _run(arm_name: str, seed: int, steps: int, workdir: pathlib.Path):
    torch.set_default_dtype(torch.float64)
    a = e13.ARMS[arm_name]
    workdir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        ppo = LP.LRPPO(env=LP.make_selected_env(), seed=seed, agent_type="mlp",
                       agent_config={"n_repeats": a.R, "net_arch": []}, run_name=f"{arm_name}__s{seed}__t64",
                       head_lr_mult=float(a.R) if a.collapse else 1.0, collapse=a.collapse, clip_mode=a.clip_mode,
                       clip_R=a.R, float64=True, **e13.PAPER_PPO_CONFIG)
        actions, trace = [], []
        orig = ppo.envs.step

        def _step(act, _o=orig):
            actions.append(np.asarray(act).reshape(-1).copy())
            return _o(act)

        ppo.envs.step = _step
        ppo.iter_hook = lambda p: trace.append((p.global_step, LP.effective_flat(p)))
        ppo.train(total_timesteps=steps)
    finally:
        os.chdir(cwd)
        torch.set_default_dtype(torch.float32)
    return actions, trace, ppo.clip_summary()


def compare(ra, rb) -> dict:
    (acts_a, tr_a, cs_a), (acts_b, tr_b, cs_b) = ra, rb
    n = min(len(acts_a), len(acts_b))
    first = next((i for i in range(n) if not np.array_equal(acts_a[i], acts_b[i])), None)
    diffs = [(s, float(np.abs(x - y).max())) for (s, x), (_, y) in zip(tr_a, tr_b)]
    upto = [d for s, d in diffs if first is None or s <= first + 1]
    return {"first_diff_step": None if first is None else first + 1,
            "max_param_diff_to_first_diff": max(upto) if upto else None,
            "max_param_diff_final": diffs[-1][1] if diffs else None, "n_steps": n,
            "clip_fraction_a": cs_a.get("clip_fraction"), "clip_fraction_b": cs_b.get("clip_fraction")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--R", nargs="+", type=int, default=[8, 32])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--pairs", nargs="+", default=list(PAIRS))
    args = p.parse_args()
    out = pathlib.Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    for pair in args.pairs:
        for R in args.R:
            for s in args.seeds:
                key = f"{pair}_R{R}_s{s}"
                if key in res:
                    continue
                t0 = time.time()
                na, nb = (n.format(R=R) for n in PAIRS[pair])
                ra = _run(na, s, args.steps, pathlib.Path(args.workdir) / "a")
                rb = _run(nb, s, args.steps, pathlib.Path(args.workdir) / "b")
                res[key] = dict(compare(ra, rb), pair=pair, a=na, b=nb, R=R, seed=s, wall_s=round(time.time() - t0, 1))
                print(key, res[key], flush=True)
                out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
