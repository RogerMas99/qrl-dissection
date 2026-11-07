"""Experiment 13 - Output Reuse against the learning rate of the head, DQN blocks (docs/EXPERIMENT-13-OR-VS-LR.md).

Question: is Output Reuse (R copies of the features before the linear head) equivalent, in this pipeline, to the
same network without reuse whose head matrix has its learning rate multiplied by R? (Proposition of thesis section
3.4.2; core/lr_equivalence.py.) Two agents:

  T  the classical twin of exp12 (bounded extractor 4->6->8->Tanh + the hybrid's head)       real env, classical arms
  H  the hybrid of Fig. 4 (Skolik, 8 qubits, L = 5, entangled) on the batched statevector EMULATOR (agent_type
     "sv"): SIMULATED results, flagged in every manifest (docs/SIMULATED-RESULTS-POLICY.md)

Blocks, per agent:  A equivalence (reuse R, collapsed R, gradient x R)    B sweep (fresh, lr of head.weight x k)
                    C global lr x k (exploratory)

Protocol of exp12 (DQN, CartPole-v1, 100 000 steps, batch 128, buffer 10 000, train_frequency 10, library defaults for
the rest, FIX-01 on), seeds 1-30 (1-10 are the seeds of exp12), greedy evaluation every 10 000 steps with 20
episodes. Own folder <results-root>/exp13_lr_equivalence/, tag "exp13".

    python experiments/exp13_lr_equivalence.py --results-root <results> --eval-episodes 20 --agent twin --block AB --shard 0 5
    python experiments/exp13_lr_equivalence.py --results-root <results> --eval-episodes 20 --list
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List

import qrl_dissection
from qrl_dissection.core import lr_equivalence as L
from qrl_dissection.core.capacity import build_agent_for, count_trainable
from qrl_dissection.core.configs import build_arm_config
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

SUBDIR = "exp13_lr_equivalence"
TAG = "exp13"
SEEDS = list(range(1, 31))
KWARGS = {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}        # exp12, verbatim
ENV = "CartPole-v1"
AGENTS = {"twin": L.PREFIX_TWIN, "hybrid": L.PREFIX_HYBRID}
BLOCKS = {"A": ("A_reuse", "A_collapsed", "A_grad"), "B": ("B_headlr",), "C": ("C_global",)}


def arms_for(agent: str, block: str) -> List[str]:
    """Arm names of an agent for a block selector made of letters, e.g. 'AB', 'C', 'ABC'."""
    names = L.arm_names(AGENTS[agent])
    out: List[str] = []
    for letter in block:
        for grp in BLOCKS[letter]:
            out += names[grp]
    return out


def build_specs(steps: int, seeds: List[int], agents=("twin", "hybrid"), block: str = "ABC") -> List[RunSpec]:
    return [RunSpec(arm=a, seed=s, fix_autoreset=True, total_timesteps=steps, dqn_kwargs=dict(KWARGS), tag=TAG)
            for ag in agents for a in arms_for(ag, block) for s in seeds]


def design_record() -> dict:
    """Measured parameter counts and learning-rate multipliers of every arm."""
    rec = {}
    for ag, pre in AGENTS.items():
        for grp, names in L.arm_names(pre).items():
            for n in names:
                at, cfg = build_arm_config(n, env_id=ENV)
                agent = build_agent_for(at, cfg, env_id=ENV, is_qnet=True)
                if cfg.get("lr13_collapse_R"):
                    L.collapse_head_(agent)
                rec[n] = {"agent": ag, "group": grp, "agent_type": at, "params": count_trainable(agent),
                          "reuse": cfg.get("reuse_repetitions", 1), "collapse_R": cfg.get("lr13_collapse_R", 0),
                          "head_lr_mult": cfg.get("lr13_head_mult", 1.0), "grad_mult": cfg.get("lr13_grad_mult", 1.0),
                          "global_lr_mult": cfg.get("lr13_global_mult", 1.0),
                          "simulated": at in ("su2", "sv")}
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True, help="folder that contains the experiment directories")
    p.add_argument("--eval-episodes", type=int, required=True, help="episodes per greedy evaluation (exp13: 20)")
    p.add_argument("--agent", choices=["twin", "hybrid", "both"], default="both")
    p.add_argument("--block", default="ABC", help="letters among A, B, C (default ABC)")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: 1-30")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), default=None,
                   help="run only cells whose position in the list is I mod N (deterministic split, no locks)")
    p.add_argument("--claim", action="store_true", help="cooperative locking instead of --shard")
    p.add_argument("--list", action="store_true", help="print the cells to run and exit")
    p.add_argument("--smoke", action="store_true", help="1 seed, 5k steps, _smoke dir")
    args = p.parse_args()

    root = pathlib.Path(args.results_root).resolve()
    outdir = root / SUBDIR
    steps = 5_000 if args.smoke else args.steps
    seeds = [1] if args.smoke else (args.seeds or SEEDS)
    eval_every = max(1, steps // 2) if args.smoke else args.eval_every
    if args.smoke:
        outdir = outdir / "_smoke"
    agents = ("twin", "hybrid") if args.agent == "both" else (args.agent,)
    specs = build_specs(steps, seeds, agents, args.block)
    if args.shard:
        i, n = args.shard
        specs = [sp for k, sp in enumerate(specs) if k % n == i]
    if args.list:
        for sp in specs:
            print(sp.run_name)
        print(f"{len(specs)} cells")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    rec = design_record()
    rec["_eval_episodes"] = args.eval_episodes
    dp = outdir / "design_params.json"
    if not dp.exists():                      # several workers: the first one writes it, the content is the same
        dp.write_text(json.dumps(rec, indent=1))
    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    print(f"{len(specs)} cells, steps={steps}, eval every {eval_every} x {args.eval_episodes} episodes, outdir={outdir}",
          flush=True)
    run_grid(specs, outdir, env_id=ENV,
             eval_cfg=GreedyEvalConfig(env_id=ENV, every_steps=eval_every, n_episodes=args.eval_episodes),
             progress_bar=False, verbose=True, claim=args.claim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
