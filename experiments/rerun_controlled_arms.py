"""
Controlled reruns of two arms whose first results were not like-for-like.

Both reruns are written ALONGSIDE the originals, distinguished by `RunSpec.tag`
(which is part of the reuse key and of the run name), so nothing is overwritten
or deleted: the first runs stay on disk for a later sweep.

group `ceilingA`  - `frozen_scalar_1q_fourier_ceiling_L{1,5,10,15}`   tag `flkw`
    The first runs came from notebooks/13_exp08_gaps_colab.ipynb, which passed
    the CARTPOLE DQN kwargs (buffer 10_000, train_frequency 10, learning_starts
    left at the 10_000 default) to a FrozenLake cell. exp04's circuits they are
    compared with use buffer 50_000 / train_frequency 1 / learning_starts 1_000.
    Evidence: the first `losses/td_loss` point is at step 10_100 for the
    ceiling and 1_100 for the circuit. The ceiling manifests nevertheless
    declare the FrozenLake values. This group retrains them with the real
    FrozenLake kwargs. See docs/CORRECTIONS.md#fix-13.

group `su2cp`     - `su2_cartpole_L5`                                   tag `init2pi`
    The first runs started the emulator's quantum weights at exactly zero while
    the real circuit starts at U[0, 2*pi). Retrained with the corrected
    initialisation. See docs/CORRECTIONS.md#fix-12.

Usage
-----
    python experiments/rerun_controlled_arms.py --group ceilingA --results-root <results dir>
    python experiments/rerun_controlled_arms.py --group su2cp    --results-root <results dir>
    python experiments/rerun_controlled_arms.py --group ceilingA --seeds 1 2 3 --smoke

Both groups are classical (no quantum simulator), roughly 70 s and 100 s per seed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

import qrl_dissection
from qrl_dissection.core.obs_adapters import FROZEN_SCALAR_ID
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

# Copied VERBATIM from experiments/exp04_dqn_frozenlake_embeddings.py (also what
# experiments/exp05_dqn_frozenlake_classical_ceiling.py uses).
FROZEN_KWARGS: Dict[str, object] = {
    "batch_size": 128,
    "buffer_size": 50_000,
    "train_frequency": 1,
    "learning_starts": 1_000,
}
# Copied from the CartPole experiments (exp01/exp02/exp03/exp07/exp08).
CARTPOLE_KWARGS: Dict[str, object] = {
    "batch_size": 128,
    "buffer_size": 10_000,
    "train_frequency": 10,
}

GROUPS = {
    "ceilingA": dict(
        arms=[f"frozen_scalar_1q_fourier_ceiling_L{L}" for L in (1, 5, 10, 15)],
        env_id=FROZEN_SCALAR_ID,
        subdir="exp04_dqn_frozenlake_embeddings",
        tag="flkw",
        dqn_kwargs=FROZEN_KWARGS,
    ),
    "su2cp": dict(
        arms=["su2_cartpole_L5"],
        env_id="CartPole-v1",
        subdir="exp01_dqn_cartpole_capacity",
        tag="init2pi",
        dqn_kwargs=CARTPOLE_KWARGS,
    ),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=sorted(GROUPS), required=True)
    p.add_argument("--results-root", required=True,
                   help="folder that contains the experiment directories "
                        "(exp04_..., exp01_...)")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--claim", action="store_true",
                   help="cooperative locking across parallel sessions")
    p.add_argument("--smoke", action="store_true", help="1 seed, short runs, _smoke dir")
    args = p.parse_args()

    g = GROUPS[args.group]
    outdir = pathlib.Path(args.results_root).resolve() / g["subdir"]
    steps = 5_000 if args.smoke else args.steps
    seeds: List[int] = [1] if args.smoke else args.seeds
    eval_every = max(1, steps // 2) if args.smoke else args.eval_every
    if args.smoke:
        outdir = outdir / "_smoke"

    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    specs = [RunSpec(arm=a, seed=s, fix_autoreset=True, total_timesteps=steps,
                     dqn_kwargs=dict(g["dqn_kwargs"]), tag=g["tag"])
             for a in g["arms"] for s in seeds]
    print(f"group {args.group}: {len(specs)} cells ({len(g['arms'])} arms x {len(seeds)} "
          f"seeds), tag={g['tag']!r}, dqn_kwargs={g['dqn_kwargs']}, outdir={outdir}")
    run_grid(specs, outdir, env_id=g["env_id"],
             eval_cfg=GreedyEvalConfig(env_id=g["env_id"], every_steps=eval_every,
                                       n_episodes=20),
             progress_bar=True, claim=args.claim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
