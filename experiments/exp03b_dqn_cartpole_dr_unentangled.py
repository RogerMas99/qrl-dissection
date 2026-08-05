"""
Experiment 03b - Data Reuploading under DQN, with entanglement held OFF.

WHY THIS EXISTS
---------------
exp03 found the thesis's one clean positive: monotone DR transfer under DQN on
CartPole, greedy 15 -> 35 -> 199 at depths L = 1/2/5. FIX-07 then showed that
result is confounded.

On the `skolik` template the entangling CZ ring is the last operation of every
layer, and CZ is diagonal, so it commutes with the PauliZ readout: the FINAL
layer's ring can never affect the output. Depth L therefore carries only L-1
EFFECTIVE entangling blocks. exp03 ran with `ent=True`, so its three depths
supplied 0, 1 and 4 effective entangling blocks - the depth axis moved
entanglement at the same time.

That leaves two readings of the same curve, and exp03 cannot separate them:

    (a) reuploading the input more often helps DQN, as it helps PPO;
    (b) entanglement coming online helps, and depth is along for the ride.

This experiment reruns the identical grid with `ent=False` throughout, so every
depth has zero entangling blocks and only reuploading varies.

    H1  the monotone rise survives -> reading (a). exp03's claim stands, now
        properly isolated, and the thesis keeps its clean positive.
    H2  the rise flattens or disappears -> reading (b). exp03's headline is
        really about entanglement, and must be rewritten. Better to find that
        here than in a viva.

Cost is the same as exp03 and the only change is one boolean, which is the
cheapest insurance available for the repo's most load-bearing result.

Comparison target: the paper's own `Skolik_DR_L{2,5}_Unentangled` runs, now in
`data/paper_ppo_summary.csv` at 10 seeds. Note their L1 unentangled cell is
vacuous for the same FIX-07 reason - it is a copy of the entangled circuit - so
depth 1 has no meaningful published counterpart.

Usage
-----
    python experiments/exp03b_dqn_cartpole_dr_unentangled.py --smoke
    python experiments/exp03b_dqn_cartpole_dr_unentangled.py
    python experiments/exp03b_dqn_cartpole_dr_unentangled.py --seeds 1 2 3 4 5 6 7 8 9 10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

import qrl_dissection
from qrl_dissection.core.configs import DR_DEPTHS, hybrid_dr_config
from qrl_dissection.dqn import GreedyEvalConfig, SafeDQN
from qrl_dissection.dqn.runner import reuse_or_none

# Copied from exp03 verbatim, INCLUDING train_frequency=10. That value
# gradient-starves small networks (exp01), and on its own merits it would be 1 -
# but exp03's numbers were produced with 10, and this experiment exists solely to
# vary `ent`. Changing a second knob would destroy the contrast. If exp03 is ever
# re-run at train_frequency=1, re-run this too, together.
DQN_KWARGS: Dict[str, Any] = {
    "batch_size": 128,
    "buffer_size": 10_000,
    "train_frequency": 10,
}


def unentangled_dr_config(n_layers_q: int) -> Dict[str, Any]:
    """exp03's config with the entangling ring disabled at every layer."""
    cfg = hybrid_dr_config(n_layers_q)
    cfg["ent"] = False
    return cfg


def run_one(outdir, name, cfg, seed, steps, kw, every):
    """Mirror of exp03's runner, so the two grids are produced identically."""
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / f"{name}.manifest.json"
    # A same-named manifest is only a hit if it answers the same question. See
    # runner.reuse_or_none: a 5k smoke cell used to satisfy a 100k request.
    hit = reuse_or_none(manifest, {"seed": seed, "total_timesteps": steps,
                                   "dqn_kwargs": kw})
    if hit is not None:
        print(f"[skip] {name}")
        return hit
    print(f"[run ] {name}", flush=True)
    runner = SafeDQN(agent_type="hybrid", agent_config=cfg, run_name=name,
                     seed=seed, fix_autoreset=True,
                     eval_cfg=GreedyEvalConfig(env_id="CartPole-v1", every_steps=every),
                     outdir=outdir, **kw)
    out = runner.train(steps, progress_bar=False)
    m = {"name": name, "seed": seed, "total_timesteps": steps, "dqn_kwargs": kw, "fix_autoreset": True, "ent": False,
         "outcome": out.__dict__, "config": {k: str(v) for k, v in cfg.items()}}
    manifest.write_text(json.dumps(m, indent=2, default=str))
    print(f"       ok {out.wall_seconds}s  phantoms {100*out.probe['frac_poison']:.2f}%")
    return m


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/exp03b_dqn_cartpole_dr_unentangled")
    p.add_argument("--depths", nargs="+", type=int, default=DR_DEPTHS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--smoke", action="store_true", help="1 seed, 5k steps")
    args = p.parse_args()

    print(json.dumps(qrl_dissection.upstream_report(), indent=2))

    seeds: List[int] = [1] if args.smoke else args.seeds
    steps = 5_000 if args.smoke else args.steps
    every = min(args.eval_every, max(1, steps // 2))

    # Smoke runs go to their own directory. They use short budgets, so

    # sharing a directory with the real pass would leave stale cells that

    # the reuse guard then (correctly) refuses to accept.  _smoke_outdir

    outdir = pathlib.Path(args.outdir) / "_smoke" if args.smoke else pathlib.Path(args.outdir)
    print(f"{len(args.depths) * len(seeds)} cells: depths {args.depths} x seeds {seeds}")
    for depth in args.depths:
        cfg = unentangled_dr_config(depth)
        for seed in seeds:
            # Same naming scheme as exp03 plus a _noent suffix, so a joined
            # analysis can pair the cells by depth and seed without guessing.
            run_one(outdir, f"hybrid_DR{depth}_noent__s{seed}", cfg, seed,
                    steps, DQN_KWARGS, every)

    print("\nCompare against exp03 depth for depth. If the rise survives, exp03's")
    print("claim is about reuploading. If it flattens, exp03's claim was about")
    print("entanglement and RESULTS-LOG.md needs rewriting, not annotating.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
