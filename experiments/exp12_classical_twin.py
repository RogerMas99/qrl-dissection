"""Experiment 12 - classical twin of the hybrid agent (docs/EXPERIMENT-12-CLASSICAL-TWIN.md).

Question: is the benefit of Output Reuse (OR) specific to the features the circuit produces? The twin keeps the
head and the OR of hybrid_OR{R} exactly (same code, see core/classical_twin.py) and replaces only the circuit by a
small classical extractor with the same 8 outputs: Linear(4->6) -> ReLU -> Linear(6->8) [-> Tanh].

Two families, R in {1, 4, 8, 16, 32}, seeds 1-10: 2 x 5 x 10 = 100 cells.
    classical_twin_r{R}            bounded outputs (final Tanh, like the circuit's <Z_i>)
    classical_twin_unbounded_r{R}  no final Tanh (range control)

Real CartPole environment, classical arms only: no simulator is involved. Results go to
<results-root>/exp12_classical_twin/ with tag "exp12". Protocol as exp02 (DQN, CartPole-v1, 100 000 steps, batch
128, buffer 10 000, train_frequency 10, library defaults for the rest, FIX-01 on, seeds 1-10). The number of
episodes of each greedy evaluation is a required argument (the references hybrid_fig4 and hybrid_OR{R} used 5).

    python experiments/exp12_classical_twin.py --results-root <results> --eval-episodes 5 --seeds 1 2
    python experiments/exp12_classical_twin.py --results-root <results> --eval-episodes 5 --list
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List

import qrl_dissection
from qrl_dissection.core.capacity import build_agent_for, count_trainable
from qrl_dissection.core.configs import TWIN_REPEATS, build_arm_config, hybrid_or_config
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

SUBDIR = "exp12_classical_twin"
TAG = "exp12"
RS = tuple(TWIN_REPEATS)                      # (1, 4, 8, 16, 32)
FAMILIES = ("classical_twin_r", "classical_twin_unbounded_r")
SEEDS = list(range(1, 11))
KWARGS = {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}     # exp02, verbatim
ENV = "CartPole-v1"


def arm_name(family: str, R: int) -> str:
    return f"{family}{R}"


def build_specs(steps: int, seeds: List[int] = None) -> List[RunSpec]:
    return [RunSpec(arm=arm_name(f, R), seed=s, fix_autoreset=True, total_timesteps=steps,
                    dqn_kwargs=dict(KWARGS), tag=TAG)
            for s in (seeds or SEEDS) for f in FAMILIES for R in RS]


def _params(agent_type, cfg) -> int:
    return count_trainable(build_agent_for(agent_type, cfg, env_id=ENV, is_qnet=True))


def design_record() -> dict:
    """MEASURED parameters of every arm (twin, extractor, head) next to the hybrid they twin."""
    rec = {}
    for R in RS:
        hy = build_agent_for("hybrid", hybrid_or_config(R), env_id=ENV, is_qnet=True)
        hy_total = count_trainable(hy)
        hy_head = count_trainable(hy.network[1:])
        entry = {"hybrid_total": hy_total, "hybrid_head": hy_head, "hybrid_circuit": hy_total - hy_head}
        for fam in FAMILIES:
            at, cfg = build_arm_config(arm_name(fam, R), env_id=ENV)
            tw = build_agent_for(at, cfg, env_id=ENV, is_qnet=True)
            entry[fam.rstrip("r").rstrip("_")] = {
                "total": count_trainable(tw), "extractor": count_trainable(tw.extractor),
                "head": count_trainable(tw.head), "extra_vs_hybrid": count_trainable(tw) - hy_total,
                "head_equals_hybrid_head": count_trainable(tw.head) == hy_head}
        rec[str(R)] = entry
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True, help="folder that contains the experiment directories")
    p.add_argument("--eval-episodes", type=int, required=True,
                   help="episodes per greedy evaluation (the references hybrid_fig4 / hybrid_OR{R} used 5)")
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
    outdir.mkdir(parents=True, exist_ok=True)
    rec["eval_episodes"] = args.eval_episodes
    (outdir / "design_params.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    print(f"{len(specs)} cells, steps={steps}, eval every {eval_every} x {args.eval_episodes} episodes, outdir={outdir}",
          flush=True)
    run_grid(specs, outdir, env_id=ENV,
             eval_cfg=GreedyEvalConfig(env_id=ENV, every_steps=eval_every, n_episodes=args.eval_episodes),
             progress_bar=False, verbose=True, claim=args.claim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
