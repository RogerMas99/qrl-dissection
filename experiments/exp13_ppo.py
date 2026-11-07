"""Experiment 13, block D - Output Reuse against the learning rate under PPO (docs/EXPERIMENT-13-OR-VS-LR.md).

Agent: the reference study's classical control (observation reduced to indices [1, 2, 3], replicated R times, then one
linear layer for the actor and one for the critic; ``net_arch = []``), trained with SimplyQRL's PPO
(``qrl_dissection.ppo.lr_ppo.LRPPO``, which is the vendored PPO with a per-group learning rate, an optional or modified
gradient-norm clipping and a clipping diagnostic). Hyperparameters = ``PAPER_PPO_CONFIG`` of the reference repo
(src/experiments/post-pqc-inference.py): lr 2.5e-4 annealed, 1 env, 512 steps, gamma 0.99, GAE 0.95, 16 minibatches,
4 epochs, clip 0.2, ent 0.01, vf 0.5, max_grad_norm 0.5, 100 000 steps. No hyperparameter is tuned.

Arms (R in {1, 4, 8, 16, 32}; "lrx" = the same network without reuse, V^0 = sum_r W_r^0, lr of the weights that read the
replicated input x R; the bias and everything else keep the base lr; lrx at R = 1 is reuse at R = 1):
  ppo_reuse_clip_r{R}     reuse R, clipping 0.5 (the reference study's case)     R = 1, 4, 8, 16, 32
  ppo_reuse_noclip_r{R}   reuse R, no clipping                                   R = 1, 4, 8, 16, 32
  ppo_lrx_clip_r{R}       lr x R, clipping 0.5 with the usual norm               R = 4, 8, 16, 32
  ppo_lrx_noclip_r{R}     lr x R, no clipping                                    R = 4, 8, 16, 32
  ppo_lrx_modclip_r{R}    lr x R, clipping with the norm sqrt(R ||g_V||^2 + ...) R = 8, 32

Seeds 1-30. Own folder <results-root>/exp13_ppo/, tag "exp13". Greedy evaluation (argmax of the actor) every 10 000
steps with 20 episodes.

    python experiments/exp13_ppo.py --results-root <results> --shard 0 14
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import qrl_dissection
from qrl_dissection.ppo.lr_ppo import LRPPO, make_selected_env

SUBDIR = "exp13_ppo"
TAG = "exp13"
SEEDS = list(range(1, 31))
ENV = "CartPole-v1"
STEPS = 100_000
EVAL_EVERY, EVAL_EPISODES, DIAG_EVERY = 10_000, 20, 1_000

# Verbatim from the reference study's repo (github.com/javier-lazaro/qrl-dissection, src/experiments/*.py).
PAPER_PPO_CONFIG: Dict[str, Any] = {
    "learning_rate": 2.5e-4, "num_envs": 1, "num_steps": 512, "anneal_lr": True, "gamma": 0.99, "gae_lambda": 0.95,
    "num_minibatches": 16, "update_epochs": 4, "norm_adv": True, "clip_coef": 0.2, "clip_vloss": True,
    "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "target_kl": None,
}

RS_REUSE = (1, 4, 8, 16, 32)
RS_LRX = (4, 8, 16, 32)
RS_MOD = (8, 32)


@dataclass(frozen=True)
class PPOArm:
    name: str
    R: int
    collapse: bool          # True: lr x R arm (reuse-R init, head collapsed)
    clip_mode: str          # torch | none | modified


def arms() -> List[PPOArm]:
    out = [PPOArm(f"ppo_reuse_clip_r{R}", R, False, "torch") for R in RS_REUSE]
    out += [PPOArm(f"ppo_reuse_noclip_r{R}", R, False, "none") for R in RS_REUSE]
    out += [PPOArm(f"ppo_lrx_clip_r{R}", R, True, "torch") for R in RS_LRX]
    out += [PPOArm(f"ppo_lrx_noclip_r{R}", R, True, "none") for R in RS_LRX]
    out += [PPOArm(f"ppo_lrx_modclip_r{R}", R, True, "modified") for R in RS_MOD]
    return out


ARMS = {a.name: a for a in arms()}


@dataclass
class Spec:
    arm: str
    seed: int
    total_timesteps: int = STEPS
    tag: str = TAG
    ppo_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(PAPER_PPO_CONFIG))
    fix_autoreset: bool = False          # not applicable to PPO (no replay buffer); kept for the shared loaders

    @property
    def run_name(self) -> str:
        return f"{self.arm}__s{self.seed}__{self.tag}"


def build_ppo(arm: PPOArm, seed: int, run_name: str, eval_every: int = EVAL_EVERY, eval_episodes: int = EVAL_EPISODES,
              diag_every: int = DIAG_EVERY, ppo_kwargs: Dict[str, Any] = None) -> LRPPO:
    kw = dict(PAPER_PPO_CONFIG if ppo_kwargs is None else ppo_kwargs)
    return LRPPO(env=make_selected_env(ENV), seed=seed, agent_type="mlp",
                 agent_config={"n_repeats": arm.R, "net_arch": []}, run_name=run_name,
                 head_lr_mult=float(arm.R) if arm.collapse else 1.0, collapse=arm.collapse, clip_mode=arm.clip_mode,
                 clip_R=arm.R, eval_every=eval_every, eval_episodes=eval_episodes, diag_every=diag_every, **kw)


def run_cell(spec: Spec, outdir: pathlib.Path, force: bool = False) -> Dict[str, Any]:
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mpath = outdir / f"{spec.run_name}.manifest.json"
    if mpath.exists() and not force:
        stored = json.loads(mpath.read_text())
        if stored.get("spec", {}).get("total_timesteps") == spec.total_timesteps:
            return stored
    arm = ARMS[spec.arm]
    cwd = os.getcwd()
    t0 = time.time()
    try:
        os.chdir(outdir)
        ppo = build_ppo(arm, spec.seed, spec.run_name, ppo_kwargs=spec.ppo_kwargs)
        ppo.train(total_timesteps=spec.total_timesteps)
        paths = ppo.write_outputs(spec.run_name, outdir)
    finally:
        os.chdir(cwd)
    manifest = {
        "spec": asdict(spec),
        "run_name": spec.run_name,
        "outcome": {"run_name": spec.run_name, "seed": spec.seed, "total_timesteps": spec.total_timesteps,
                    "wall_seconds": round(time.time() - t0, 1), "probe": {},
                    "episodes_csv": str(outdir / "runs" / f"{spec.run_name}.csv"),
                    "eval_csv": paths["eval_csv"], "diag_csv": paths["diag_csv"], "clip_npy": paths["clip_npy"],
                    "clip": ppo.clip_summary()},
        "agent_type": "mlp",
        "agent_config": {"n_repeats": arm.R, "net_arch": [], "obs_indices": [1, 2, 3], "collapse": arm.collapse,
                         "head_lr_mult": float(arm.R) if arm.collapse else 1.0, "clip_mode": arm.clip_mode},
        "env_id": ENV,
        "git_revision": _git_revision(),
        "python": platform.python_version(),
    }
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _git_revision() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=pathlib.Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return ""


def build_specs(seeds: List[int], steps: int = STEPS) -> List[Spec]:
    return [Spec(arm=a.name, seed=s, total_timesteps=steps) for a in arms() for s in seeds]


def design_record() -> dict:
    rec = {"ppo_config": PAPER_PPO_CONFIG, "obs_indices": [1, 2, 3], "eval_every": EVAL_EVERY,
           "eval_episodes": EVAL_EPISODES, "arms": {}}
    for a in arms():
        ppo = build_ppo(a, 1, f"_design_{a.name}", eval_every=0, diag_every=0)
        rec["arms"][a.name] = {"R": a.R, "collapse": a.collapse, "clip_mode": a.clip_mode,
                               "params": sum(p.numel() for p in ppo.agent.parameters()),
                               "head_lr_mult": ppo.head_lr_mult}
        ppo.close()
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: 1-30")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--arms", nargs="+", default=None, help="restrict to these arm names")
    p.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 seed, 6000 steps, _smoke dir")
    args = p.parse_args()

    root = pathlib.Path(args.results_root).resolve()
    outdir = root / SUBDIR
    steps = 6_000 if args.smoke else args.steps
    seeds = [1] if args.smoke else (args.seeds or SEEDS)
    if args.smoke:
        outdir = outdir / "_smoke"
    specs = build_specs(seeds, steps)
    if args.arms:
        specs = [s for s in specs if s.arm in args.arms]
    if args.shard:
        i, n = args.shard
        specs = [s for k, s in enumerate(specs) if k % n == i]
    if args.list:
        for s in specs:
            print(s.run_name)
        print(f"{len(specs)} cells")
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    dp = outdir / "design_params.json"
    if not dp.exists():
        os.chdir(outdir)                       # the design builds PPO objects that open runs/*.csv
        dp.write_text(json.dumps(design_record(), indent=1))
        os.chdir(root)
    print(f"{len(specs)} cells, steps={steps}, outdir={outdir}", flush=True)
    for k, sp in enumerate(specs, 1):
        print(f"[{k}/{len(specs)}] {sp.run_name}", flush=True)
        try:
            m = run_cell(sp, outdir)
            c = m["outcome"]["clip"]
            print(f"    ok  {m['outcome']['wall_seconds']}s  clip_fraction {c.get('clip_fraction', float('nan')):.3f}",
                  flush=True)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
