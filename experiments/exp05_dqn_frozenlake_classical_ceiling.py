"""
Experiment 05 - the additive Fourier ceiling on FrozenLake Config B, under DQN.

Full design: docs/EXPERIMENT-05.md. Short version:

exp04's Config B (`frozen_binary_4q_L{1,5}`, `ent=True`) already has a
capacity-matched classical control (`frozen_matched_scalar`, NEW-02). That
answers "does the circuit beat an equal-budget classical net on the same
information?" This experiment adds a SHARPER question: does the circuit beat
the exact function class an UNENTANGLED version of itself could already
express? On this specific embedding (`FrozenBasisToAngleTransformer`, each
bit mapped to {0, pi}), that class is known EXACTLY, not estimated
(docs/CORRECTIONS.md#new-06, verified against the real circuit in
tests/test_frozenlake_additive_ceiling.py): a 5-parameter-per-action affine
model on the four bits, independent of reuploading depth L.

Arms (docs/EXPERIMENT-05.md section 3):

    frozen_binary_4q_L1 / L5           entangled - the object of study
                                        (REUSED from exp04b, not retrained)
    frozen_binary_4q_noent_L1 / L5     unentangled circuit - what the ceiling
                                        bounds (registered, not yet run before
                                        this script)
    frozen_binary_4q_fourier_ceiling   the classical ceiling itself -
                                        linear_on_bits, 20 params, no L suffix
                                        (provably L-independent on this
                                        domain - see core/fourier_ceiling.py)
    frozen_matched_scalar              capacity-matched classical control
                                        (REUSED from exp04 stage 1, n=10
                                        already on disk)

WHY THIS SHARES exp04's OUTPUT DIRECTORY, not its own. Two of the five arms
above are reused, not retrained: `frozen_binary_4q_L{1,5}` (exp04b) and
`frozen_matched_scalar` (exp04 stage 1) already have completed manifests. The
cell-level reuse guard (`dqn/runner.py::reuse_or_none`) matches purely on
`(arm, seed, fix_autoreset, total_timesteps, dqn_kwargs, tag)` within ONE
output directory - it has no way to find a cell that exists in a different
directory. Pointing `--outdir` anywhere else would silently RETRAIN those two
arms from scratch instead of reusing them. `DQN_KWARGS` below is therefore
copied from exp04's script VERBATIM and must stay identical to it for the
same reason - a mismatch on any key make the guard refuse the "reused" cells
as a different question, loudly (which is the safe failure: it will not
mislabel a mismatched run as a hit).

Pre-declared hypotheses (docs/EXPERIMENT-05.md section 4), fixed before any
cell here runs:

    H1  SU2SkolikEmulator == the real noent circuit at 1e-6 (already checked,
        tests/test_su2_equivalence.py - not re-tested by this script).
    H2  noent ~= ceiling ~= chance at every depth; the entangled arm sits
        above both. The interesting outcome: entanglement (not depth, not
        the embedding) is what lets Config B escape the dead floor
        frozen_matched_scalar / frozen_scalar_1q already showed at low depth.
        If the entangled arm is ALSO at chance, the contrast is null in this
        environment - report it as a bound on the claim.
    H3  Not the primary question (CartPole is exp06's job) but worth
        watching: if noent or the ceiling ever beat the entangled arm, that
        is not evidence against entanglement mattering elsewhere.

Falsification (docs/EXPERIMENT-05.md section 5): if `frozen_binary_4q_noent`
does NOT come out affine in the bits, something has changed upstream -
re-run tests/test_frozenlake_additive_ceiling.py before trusting anything
here.

Every arm here trains ONLY at FIX-01 on (`fix_autoreset=True`) - stage 1
already measured the FIX-01 contrast on this environment; this experiment's
question is orthogonal to it.

Usage
-----
    python experiments/exp05_dqn_frozenlake_classical_ceiling.py --ladder-only
    python experiments/exp05_dqn_frozenlake_classical_ceiling.py --smoke
    python experiments/exp05_dqn_frozenlake_classical_ceiling.py
    python experiments/exp05_dqn_frozenlake_classical_ceiling.py --seeds 1 2 3 4 5 6 7 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

import numpy as np

import qrl_dissection
from qrl_dissection import analysis, build_arm_config
from qrl_dissection.core import capacity, stats
from qrl_dissection.core.obs_adapters import FROZEN_SCALAR_ID
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

# Copied VERBATIM from experiments/exp04_dqn_frozenlake_embeddings.py - see
# this file's own docstring for why it must stay identical, not just similar.
DQN_KWARGS: Dict[str, object] = {
    "batch_size": 128,
    "buffer_size": 50_000,
    "train_frequency": 1,
    "learning_starts": 1_000,
}

# The entangled pair is REUSED from exp04b; the noent pair and the ceiling
# are NEW cells this script actually trains.
ENTANGLED_ARMS = ["frozen_binary_4q_L1", "frozen_binary_4q_L5"]
NOENT_ARMS = ["frozen_binary_4q_noent_L1", "frozen_binary_4q_noent_L5"]
CEILING_ARM = "frozen_binary_4q_fourier_ceiling"
CONTROL_ARM = "frozen_matched_scalar"

ALL_ARMS = ENTANGLED_ARMS + NOENT_ARMS + [CEILING_ARM, CONTROL_ARM]


def eval_cfg_for(env_id: str, every: int) -> GreedyEvalConfig:
    """Mirrors exp04's own helper - the greedy hook builds its OWN env from
    cfg.env_id (defaults to CartPole), so passing it explicitly is mandatory."""
    return GreedyEvalConfig(env_id=env_id, every_steps=every, n_episodes=20)


def ladder() -> None:
    """Parameter accounting. Trains nothing; run before committing compute."""
    print("=== exp05 capacity ladder ===")
    for arm in ALL_ARMS:
        try:
            agent_type, cfg = build_arm_config(arm, env_id=FROZEN_SCALAR_ID)
            net = capacity.build_agent_for(agent_type, cfg, env_id=FROZEN_SCALAR_ID,
                                           is_qnet=True)
            n = capacity.count_trainable(net)
            print(f"  {arm:34s} {agent_type:17s} {n:6d}")
        except Exception as exc:
            print(f"  {arm:34s} {'?':17s}      ! {type(exc).__name__}: {exc}")
    print("\n  The ceiling's count is NOT meant to match the hybrid arms' - see")
    print("  docs/CORRECTIONS.md#new-06 'Open sizing question'. Reported for context.")


def summarise(outdir: pathlib.Path, window: int = 100) -> None:
    """Success rate (MA-100 max, biased), `final_performance` (unbiased) and
    greedy - all three, from the start, learning from RESULTS-LOG.md's exp04
    stage-2 update: `best_ma` alone overstates arms that are actually near
    chance once sustained performance is measured properly."""
    try:
        import pandas as pd
    except ImportError:
        return
    rows = []
    for arm in ALL_ARMS:
        best_ma, final_perf, greedy = [], [], []
        for mp in sorted(outdir.glob(f"{arm}__fix01on__s*.manifest.json")):
            m = json.loads(mp.read_text())
            if "error" in m or "spec" not in m:
                continue
            oc = m["outcome"]
            try:
                rew, _ = analysis.load_episodes(oc["episodes_csv"])
            except FileNotFoundError:
                continue
            best_ma.append(float(np.nanmax(analysis.moving_average(rew, window))))
            final_perf.append(stats.final_performance(rew, last_frac=0.1))
            if oc.get("eval_csv") and pathlib.Path(oc["eval_csv"]).exists():
                _, sc = analysis.load_eval(oc["eval_csv"])
                if len(sc):
                    greedy.append(float(max(sc)))
        if not best_ma:
            rows.append(dict(arm=arm, n=0, best_ma="-", final_performance="-", greedy="-"))
            continue
        bm_ci = stats.bootstrap_ci(best_ma)
        fp_ci = stats.bootstrap_ci(final_perf)
        gr_ci = stats.bootstrap_ci(greedy) if greedy else (float("nan"), float("nan"))
        rows.append(dict(
            arm=arm, n=len(best_ma),
            best_ma=f"{stats.iqm(best_ma):.3f} ({bm_ci[0]:.3f}, {bm_ci[1]:.3f})",
            final_performance=f"{stats.iqm(final_perf):.3f} ({fp_ci[0]:.3f}, {fp_ci[1]:.3f})",
            greedy=(f"{stats.iqm(greedy):.3f} ({gr_ci[0]:.3f}, {gr_ci[1]:.3f})"
                   if greedy else "-"),
        ))
    df = pd.DataFrame(rows)
    print("\n=== exp05: best_ma (max, biased) vs final_performance (unbiased) vs greedy ===")
    print(df.to_string(index=False))
    print("\nH2 reading: noent and the ceiling should sit near frozen_matched_scalar's")
    print("chance floor while the entangled arms sit above both, if entanglement is")
    print("what Config B needs. If the entangled arm is ALSO at chance, report the")
    print("contrast as null in this environment, not as inconclusive.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--claim", action="store_true",
                   help="cooperative locking across parallel sessions")
    p.add_argument("--outdir", default="results/exp04_dqn_frozenlake_embeddings",
                   help="SAME directory as exp04 by default - see this script's "
                        "docstring for why (reused cells must share it).")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="default: 3 (coverage). Robustness pass: 8-10.")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--ladder-only", action="store_true",
                   help="parameter accounting only; trains nothing")
    p.add_argument("--smoke", action="store_true", help="1 seed, short runs")
    args = p.parse_args()

    print(json.dumps(qrl_dissection.upstream_report(), indent=2))

    if args.ladder_only:
        ladder()
        return 0

    outdir = pathlib.Path(args.outdir) / "_smoke" if args.smoke else pathlib.Path(args.outdir)
    steps = 5_000 if args.smoke else args.steps
    seeds: List[int] = [1] if args.smoke else (args.seeds or [1, 2, 3])
    eval_every = max(1, steps // 2) if args.smoke else args.eval_every

    specs = [RunSpec(arm=arm, seed=s, fix_autoreset=True, total_timesteps=steps,
                     dqn_kwargs=DQN_KWARGS)
             for arm in ALL_ARMS for s in seeds]
    print(f"{len(specs)} cells ({len(ALL_ARMS)} arms x {len(seeds)} seeds). "
          f"{ENTANGLED_ARMS} and {CONTROL_ARM} are expected REUSE hits if exp04b "
          "and exp04 stage 1 already cover these seeds; only the noent pair and "
          "the ceiling are new compute here.")
    run_grid(specs, outdir, env_id=FROZEN_SCALAR_ID,
             eval_cfg=eval_cfg_for(FROZEN_SCALAR_ID, eval_every),
             claim=args.claim)

    summarise(outdir)
    print("\nROBUSTNESS: re-run at 8-10 seeds before writing this up as a")
    print("conclusion, same discipline as every other block sweep in this repo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
