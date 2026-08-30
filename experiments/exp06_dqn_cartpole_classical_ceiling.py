"""
Experiment 06 - the additive Fourier ceiling on the CartPole Skolik sweep,
under DQN.

Full design: docs/EXPERIMENT-06.md. Short version:

Same construction as exp05 (`core/fourier_ceiling.py::FourierAdditiveCeiling`,
guarded by `check_additive_embedding`), applied to `hybrid_fig4`'s Skolik
8-qubit circuit instead of FrozenLake's 4-qubit one. Priority 2, behind exp05:
CartPole is largely solvable by near-linear controllers already
(`paper_linear` at 26 parameters reliably dies, `matched_classical` at 135
barely clears it - docs/RESULTS-LOG.md, exp01), so the additive ceiling is
probably not the binding constraint here, unlike FrozenLake Config B where it
is the whole story. **Expect an inconclusive result and report it as such**
("the environment does not discriminate between these hypothesis classes"),
not as a null finding dressed up as informative.

Arms:

    hybrid_fig4              the reference Skolik 8q circuit, ent=True -
                              REUSED from exp01 (n=10 already on disk at
                              exactly this spec: 100k steps, these DQN_KWARGS)
    su2_cartpole_L5           the "noent counterpart" - NOT a second real-PQC
                              arm. hybrid_fig4 is 8 qubits at 100k steps, the
                              expensive regime in this repo (docs/REUSE.md:
                              "can run for hours"). NEW-05 already proves
                              SU2SkolikEmulator == real skolik(ent=False) to
                              1e-6 per call (docs/CORRECTIONS.md#new-05,
                              ~350x faster measured on this machine), so this
                              arm gets the noent comparison point for a small
                              fraction of the cost. See docs/EXPERIMENT-06.md
                              section 1 for the fuller argument, including
                              the real-PQC alternative (`paper_skolik_8q_L5`
                              with ent forced False) if a reviewer wants one.
    cartpole_fourier_ceiling_L5   the classical ceiling, sized to n_qubits=8,
                              n_layers=5, n_actions=2 - NOT capacity-matched
                              to hybrid_fig4 (162 vs 126 params; see
                              docs/CORRECTIONS.md#new-06 "Open sizing
                              question" - reported side by side, not equalised)

Pre-declared reading (H3, docs/CORRECTIONS.md#new-06): ceiling >= hybrid
`noent` (here, `su2_cartpole_L5`) within noise is the expected outcome. If
the ENTANGLED hybrid_fig4 beats the ceiling, do not call it expressivity -
the ceiling's coefficients are free real numbers where the circuit's are
constrained by unitarity, so the honest reading is inductive bias or an
optimisation effect, stated as such.

WHY THIS SHARES exp01's OUTPUT DIRECTORY, not its own. `hybrid_fig4` at this
exact spec (100k steps, `{"batch_size": 128, "buffer_size": 10000,
"train_frequency": 10}`, `fix_autoreset=True`) already has n=10 in
`results/exp01_dqn_cartpole_capacity` - ground truth read directly from that
directory's own manifest, not assumed. The cell-level reuse guard matches
only within ONE output directory, so `--outdir` must point there for that
arm to be reused rather than retrained (hours, for no new information).
`DQN_KWARGS` below is copied from that manifest's own `spec.dqn_kwargs`
verbatim and must stay identical for the same reason.

No new derivation is owed here - the Fourier-ceiling argument is
embedding-specific (angle_embedding, one feature per wire), not
environment-specific, and CartPole's Skolik circuit already uses that
embedding (the cycling path NEW-05/NEW-06 verify:
`skolik_8q_cartpole_L5`/`tests/test_fourier_ceiling_spectrum.py`, on a
continuous domain, ahead of exp05's two-point FrozenLake special case). What
is new is only the arm registration and this grid.

Usage
-----
    python experiments/exp06_dqn_cartpole_classical_ceiling.py --ladder-only
    python experiments/exp06_dqn_cartpole_classical_ceiling.py --smoke
    python experiments/exp06_dqn_cartpole_classical_ceiling.py
    python experiments/exp06_dqn_cartpole_classical_ceiling.py --seeds 1 2 3 4 5 6 7 8
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
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

# Copied VERBATIM from results/exp01_dqn_cartpole_capacity/hybrid_fig4__*
# .manifest.json's own spec.dqn_kwargs - see this file's docstring for why it
# must stay identical, not just similar, for hybrid_fig4 to be reused.
DQN_KWARGS: Dict[str, object] = {
    "batch_size": 128,
    "buffer_size": 10_000,
    "train_frequency": 10,
}

ALL_ARMS = ["hybrid_fig4", "su2_cartpole_L5", "cartpole_fourier_ceiling_L5"]
ENV_ID = "CartPole-v1"


def eval_cfg_for(every: int) -> GreedyEvalConfig:
    return GreedyEvalConfig(env_id=ENV_ID, every_steps=every, n_episodes=20)


def ladder() -> None:
    """Parameter accounting. Trains nothing; run before committing compute."""
    print("=== exp06 capacity ladder ===")
    for arm in ALL_ARMS:
        try:
            agent_type, cfg = build_arm_config(arm, env_id=ENV_ID)
            net = capacity.build_agent_for(agent_type, cfg, env_id=ENV_ID, is_qnet=True)
            n = capacity.count_trainable(net)
            print(f"  {arm:30s} {agent_type:17s} {n:6d}")
        except Exception as exc:
            print(f"  {arm:30s} {'?':17s}      ! {type(exc).__name__}: {exc}")
    print("\n  su2_cartpole_L5 vs hybrid_fig4 should match EXACTLY (same architecture,")
    print("  no quantum simulator underneath) - if it does not, something regressed.")
    print("  The ceiling's count is NOT meant to match either - reported for context.")


def summarise(outdir: pathlib.Path, window: int = 50) -> None:
    """best_ma (max, biased), final_performance (unbiased) and greedy - all
    three from the start, same discipline exp05 uses."""
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
            best_ma=f"{stats.iqm(best_ma):.1f} ({bm_ci[0]:.1f}, {bm_ci[1]:.1f})",
            final_performance=f"{stats.iqm(final_perf):.1f} ({fp_ci[0]:.1f}, {fp_ci[1]:.1f})",
            greedy=(f"{stats.iqm(greedy):.1f} ({gr_ci[0]:.1f}, {gr_ci[1]:.1f})"
                   if greedy else "-"),
        ))
    df = pd.DataFrame(rows)
    print("\n=== exp06: best_ma (max, biased) vs final_performance (unbiased) vs greedy ===")
    print(df.to_string(index=False))
    print("\nH3 reading: ceiling >= su2_cartpole_L5 within noise is expected. If")
    print("hybrid_fig4 (entangled) beats the ceiling, that is inductive bias or an")
    print("optimisation effect, NOT evidence of expressivity beyond the ceiling's")
    print("class - the ceiling's coefficients are unconstrained, the circuit's are")
    print("not. An inconclusive/overlapping result here is the PREDICTED outcome,")
    print("not a failure of the experiment - CartPole is not expected to")
    print("discriminate these hypothesis classes (see this script's docstring).")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--claim", action="store_true",
                   help="cooperative locking across parallel sessions")
    p.add_argument("--outdir", default="results/exp01_dqn_cartpole_capacity",
                   help="SAME directory as exp01 by default - see this "
                        "script's docstring for why (hybrid_fig4 must be "
                        "reused, not retrained).")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="default: 3 (coverage). Robustness pass: 8-10 - "
                        "hybrid_fig4 already has 10 on disk either way.")
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
          "hybrid_fig4 is expected to be a REUSE hit for seeds 1-10 (already on "
          "disk from exp01); su2_cartpole_L5 and cartpole_fourier_ceiling_L5 are "
          "the only new compute, and both are cheap.")
    run_grid(specs, outdir, env_id=ENV_ID, eval_cfg=eval_cfg_for(eval_every),
             claim=args.claim)

    summarise(outdir)
    print("\nExpect this comparison to be inconclusive (see docstring) - report it")
    print("as such rather than forcing a positive reading out of overlapping CIs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
