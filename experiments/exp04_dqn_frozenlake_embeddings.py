"""
Experiment 04 - Data Reuploading and observation embedding under DQN, on FrozenLake-v1.

ATTRIBUTION FIRST, because it decides what this experiment can claim. The
dissection paper (arXiv:2511.17112) is CartPole-ONLY. FrozenLake appears in the
SimplyQRL *library chapter*, Experiment 3: PPO, 500k steps, a SINGLE seed, no
classical control, no block-level conclusion. So the repo's usual question -
"does the paper's block finding transfer off-policy?" - cannot be asked here.
There is nothing dissected to transfer from.

What this experiment is for, then:

  (a) GENERALITY OF exp03. exp03 found strong monotone DR transfer under DQN on
      CartPole (greedy 15 -> 35 -> 199 for L = 1/2/5 at 100k). That is the
      thesis's one clean positive and it rests on a single environment. A DR
      sweep on a structurally different task - sparse reward, discrete state,
      short episodes - turns "DR transfers to DQN" into "DR transfers to DQN,
      and not only on CartPole". Failure to replicate bounds the exp03 claim,
      which is worth just as much.

  (b) FIX-01 WHERE IT IS MEASURABLE. Needs no paper at all. The phantom fraction
      is roughly 1 / mean_episode_length, so on CartPole it SHRINKS as the agent
      improves - under 1% once an arm learns, which is exactly why exp01 found
      no significant FIX-01 effect in any live arm. FrozenLake inverts that:
      episodes are 6-10 steps whether or not the agent is learning. MEASURED on
      this stack at 6k steps: 8.6% (FIX-01 off) and 9.9% (on), an order of
      magnitude above CartPole at convergence.

      The poison is also better aimed. The phantom transition is
      (terminal_state, arbitrary_action, r=0, s'=start, done=False). Under dense
      reward that is one bad sample among many good ones. On FrozenLake the
      reward is a single bit delivered only at the goal, and the phantom
      attaches to precisely the transition carrying it.

  (c) A CONTROLLED VERSION OF THE CHAPTER'S EXP 3. Config A and Config B, one
      seed each, no control. Re-running them with a capacity-matched classical
      control and 8-10 seeds is a contribution by itself, and it is cheap: 1 and
      4 qubits against CartPole's 8.

BLOCKER: FIX-05. Upstream cannot run a Discrete observation space under DQN at
all - see docs/CORRECTIONS.md#fix-05. Fixed by an environment adapter
(core/obs_adapters.py), which is why this script needs no changes to SafeDQN or the
runner: it passes a registered env id like any other experiment.

STAGING. Compute is spent only after the regime is shown alive - the failure
that cost exp01 two experiments. Stage 1 is classical and cheap; do not run
stage 2 until frozen_onehot_mlp reaches a success rate near 1.0.

Usage
-----
    python experiments/exp04_dqn_frozenlake_embeddings.py --ladder-only
    python experiments/exp04_dqn_frozenlake_embeddings.py --stage 1
    python experiments/exp04_dqn_frozenlake_embeddings.py --stage 2
    python experiments/exp04_dqn_frozenlake_embeddings.py --stage 2 --seeds 1 2 3 4 5 6 7 8
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
from qrl_dissection.core import capacity
from qrl_dissection.core.configs import (
    FROZEN_ARM_ENV,
    FROZEN_DR_A,
    FROZEN_DR_B,
    FROZEN_ONEHOT_ID,
    FROZEN_SCALAR_ID,
)
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid

# FrozenLake-specific hyper-parameters. Every deviation from the exp01/exp03
# CartPole defaults is recorded, because an unexplained hyper-parameter
# difference between environments is a confound a reviewer will find.
DQN_KWARGS: Dict[str, object] = {
    "batch_size": 128,
    # 50k rather than 10k: with one bit of reward, a buffer that forgets its
    # rare successful trajectories starves the only signal there is.
    "buffer_size": 50_000,
    # 1 rather than 10. exp01 established that train_frequency=10 gradient-
    # starves small networks enough to mask every other effect under study.
    "train_frequency": 1,
    # 1k rather than 10k: at these episode lengths 10k steps is a large fraction
    # of the run spent not learning.
    "learning_starts": 1_000,
}

STAGE1_ARMS = ["frozen_onehot_mlp", "frozen_scalar_mlp_large", "frozen_matched_scalar"]
LIVENESS_ARM = "frozen_onehot_mlp"

# WHAT THE SMOKE RUN ALREADY SHOWED (1 seed, 5k steps, this stack)
#
#   frozen_onehot_mlp        success 0.86 - 0.96   <- regime is alive, and fast
#   frozen_scalar_mlp_large  success 0.03 - 0.07   <- barely above random (0.015)
#   frozen_matched_scalar    success 0.04 - 0.07   <- same
#
# Read that before designing around H4. A classical network fed the raw state
# INDEX has almost nothing to work with: the index ordering is row-major and
# carries no usable metric, so a scalar-input MLP must fit a near-arbitrary
# 16-point function through one input. The one-hot arm solves the task in 5k
# steps; the scalar arms do not move.
#
# The consequence is uncomfortable and needs saying up front: the fair control
# may be DEAD, which is exactly the situation exp01 spent two experiments
# learning to detect. If frozen_matched_scalar stays at chance, H4 is
# unanswerable - a hybrid beating a dead control proves nothing - and the honest
# claim becomes one about the ENCODING rather than the circuit. Config A is a
# 1-qubit circuit on that same scalar input, so it may be dead for the same
# reason. In that case the informative comparison is Config B (binary basis)
# against the one-hot MLP, and it should be reported as such rather than dressed
# up as a circuit-vs-classical result.


def stage2_arms(dr_a: List[int], dr_b: List[int], ablation: bool) -> List[str]:
    arms = [f"frozen_scalar_1q_L{L}" for L in dr_a]
    arms += [f"frozen_binary_4q_L{L}" for L in dr_b]
    if ablation:
        arms += [f"frozen_binary_4q_noent_L{L}" for L in dr_b]
    return arms


def eval_cfg_for(env_id: str, every: int) -> GreedyEvalConfig:
    """Greedy eval builds its OWN environment from cfg.env_id, which defaults to
    CartPole. Passing it explicitly is mandatory here; getting it wrong would
    evaluate a FrozenLake agent on CartPole."""
    return GreedyEvalConfig(env_id=env_id, every_steps=every, n_episodes=20)


def ladder(arms: List[str]) -> None:
    """Parameter accounting. Trains nothing; run it before committing compute."""
    print("=== FrozenLake capacity ladder ===")
    rows = []
    for arm in arms:
        env_id = FROZEN_ARM_ENV[arm]
        try:
            agent_type, cfg = build_arm_config(arm, env_id=env_id)
            net = capacity.build_agent_for(agent_type, cfg, env_id=env_id, is_qnet=True)
            n = capacity.count_trainable(net)
            extra = ""
            if agent_type == "hybrid":
                b = capacity.pqc_parameter_budget(cfg, env_id=env_id)
                extra = f"  q={b['quantum']:<4d} head={b['classical_head']}"
            rows.append((arm, agent_type, n, extra))
        except Exception as exc:
            rows.append((arm, "?", -1, f"  ! {type(exc).__name__}: {exc}"))
    for arm, at, n, extra in rows:
        print(f"  {arm:30s} {at:8s} {n:6d}{extra}")

    hyb = dict((a, n) for a, _t, n, _e in rows if a == "frozen_scalar_1q_L5")
    ctl = dict((a, n) for a, _t, n, _e in rows if a == "frozen_matched_scalar")
    if hyb and ctl:
        h, c = hyb["frozen_scalar_1q_L5"], ctl["frozen_matched_scalar"]
        ok = "OK" if c >= h else "!! CONTROL IS SMALLER THAN THE HYBRID"
        print(f"\n  matched control {c} vs hybrid {h} params - {ok}")
    print("\n  Note the scale: the 1-qubit arms are 10-38 parameters, and their")
    print("  head is Linear(1, 4) - all four Q-values are affine in one")
    print("  expectation value. That is a structural limit of the chapter's")
    print("  Config A, not of our set-up, and it belongs in the write-up.")


def random_policy_baseline(episodes: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Success rate and episode length under a random policy.

    Two numbers needed before reading any curve: the success rate is the floor a
    learning curve must clear, and the mean episode length gives the predicted
    phantom fraction 1/len that H3 says the FIX-01 effect should track. Record
    both in RESULTS-LOG.md BEFORE running a grid, so the prediction pre-dates the
    measurement.
    """
    import gymnasium as gym

    env = gym.make(FROZEN_ONEHOT_ID)
    rng = np.random.default_rng(seed)
    successes, lengths, r = 0, [], 0.0
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(1 << 30)))
        done, n = False, 0
        while not done:
            _, r, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
            n += 1
            done = term or trunc
        successes += int(r > 0)
        lengths.append(n)
    env.close()
    mean_len = float(np.mean(lengths))
    return {"success_rate": successes / episodes,
            "mean_episode_length": mean_len,
            "predicted_phantom_fraction": 1.0 / mean_len}


def summarise(outdir: pathlib.Path, window: int = 100) -> None:
    """FrozenLake returns are 0/1, so a rolling mean IS the success rate.

    A 100-episode window rather than the repo default of 50: on a binary signal
    the 50-window estimator is noisy. `greedy_best` stays primary, as in exp01.
    """
    try:
        import pandas as pd
    except ImportError:
        return
    rows = []
    for mp in sorted(outdir.glob("*.manifest.json")):
        m = json.loads(mp.read_text())
        if "error" in m:
            continue
        oc, spec = m["outcome"], m["spec"]
        try:
            rew, _ = analysis.load_episodes(oc["episodes_csv"])
        except FileNotFoundError:
            continue
        best_sr = float(np.nanmax(analysis.moving_average(rew, window)))
        gb = float("nan")
        if oc.get("eval_csv") and pathlib.Path(oc["eval_csv"]).exists():
            _, sc = analysis.load_eval(oc["eval_csv"])
            gb = float(max(sc)) if len(sc) else float("nan")
        rows.append(dict(arm=spec["arm"], fix01=spec["fix_autoreset"], seed=spec["seed"],
                         best_sr=round(best_sr, 3), greedy_best=round(gb, 3),
                         phantom=round(oc["probe"]["frac_poison"], 4)))
    if not rows:
        print("(no completed runs yet)")
        return
    df = pd.DataFrame(rows)
    print("\n=== success rate (MA-%d) and greedy eval ===" % window)
    print(df.groupby(["arm", "fix01"]).agg(
        best_sr=("best_sr", "mean"), sd=("best_sr", "std"),
        greedy=("greedy_best", "mean"), phantom=("phantom", "mean"),
        n=("seed", "count")).round(3))

    if df.fix01.nunique() == 2:
        print("\n=== H3: FIX-01 delta where the phantom fraction is large ===")
        for arm in sorted(df.arm.unique()):
            a = df[(df.arm == arm) & (~df.fix01)]
            b = df[(df.arm == arm) & (df.fix01)]
            if len(a) and len(b):
                print(f"  {arm:26s} delta success {b.best_sr.mean() - a.best_sr.mean():+.3f}"
                      f"   phantoms {100 * df[df.arm == arm].phantom.mean():.1f}%")
        print("  CartPole reference (exp01): no significant FIX-01 effect in any live arm.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--claim", action="store_true",
                   help="cooperative locking across parallel sessions")
    p.add_argument("--outdir", default="results/exp04_dqn_frozenlake_embeddings")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="default: 5 seeds for stage 1, 3 for stage 2 (coverage)")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--dr-a", nargs="+", type=int, default=FROZEN_DR_A)
    p.add_argument("--dr-b", nargs="+", type=int, default=FROZEN_DR_B)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--with-ablation", action="store_true",
                   help="add the entanglement-off variants of Config B")
    p.add_argument("--ladder-only", action="store_true",
                   help="parameter accounting plus the random baseline; trains nothing")
    p.add_argument("--smoke", action="store_true", help="1 seed, short runs")
    args = p.parse_args()

    print(json.dumps(qrl_dissection.upstream_report(), indent=2))

    all_arms = STAGE1_ARMS + stage2_arms(args.dr_a, args.dr_b, True)
    if args.ladder_only:
        ladder(all_arms)
        print("\n=== random-policy baseline ===")
        base = random_policy_baseline()
        print(f"  success rate           {base['success_rate']:.4f}")
        print(f"  mean episode length    {base['mean_episode_length']:.2f}")
        print(f"  predicted phantom frac {base['predicted_phantom_fraction']:.4f}"
              "   (CartPole at convergence: < 0.01)")
        if base["success_rate"] > 0.05:
            print("  ! random policy already succeeds often; arms may not separate")
        return 0

    # Smoke runs go to their own directory. They use short budgets, so

    # sharing a directory with the real pass would leave stale cells that

    # the reuse guard then (correctly) refuses to accept.  _smoke_outdir

    outdir = pathlib.Path(args.outdir) / "_smoke" if args.smoke else pathlib.Path(args.outdir)
    steps = 5_000 if args.smoke else args.steps
    if args.smoke:
        args.eval_every = max(1, steps // 2)   # else the greedy hook never fires

    if args.stage == 1:
        seeds = args.seeds or ([1] if args.smoke else [1, 2, 3, 4, 5])
        # Two calls: the one-hot arm lives on a different registered id.
        for env_id, arms in ((FROZEN_ONEHOT_ID, [LIVENESS_ARM]),
                             (FROZEN_SCALAR_ID, [a for a in STAGE1_ARMS if a != LIVENESS_ARM])):
            specs = [RunSpec(arm=arm, seed=s, fix_autoreset=fix,
                             total_timesteps=steps, dqn_kwargs=DQN_KWARGS)
                     for arm in arms for fix in (False, True) for s in seeds]
            run_grid(specs, outdir, env_id=env_id,
                     eval_cfg=eval_cfg_for(env_id, args.eval_every, claim=args.claim))
        summarise(outdir)
        print("\nGATE: stage 2 is only interpretable if frozen_onehot_mlp reached a")
        print("success rate near 1.0. If it did not, repair the regime first.")
        return 0

    seeds = args.seeds or ([1] if args.smoke else [1, 2, 3])
    arms = stage2_arms(args.dr_a, args.dr_b, args.with_ablation)
    specs = [RunSpec(arm=arm, seed=s, fix_autoreset=True,
                     total_timesteps=steps, dqn_kwargs=DQN_KWARGS)
             for arm in arms for s in seeds]
    print(f"{len(specs)} hybrid cells. Measured throughput on a CPU runtime: "
          f"~25 steps/s at 4 qubits, ~50 at 1 qubit.")
    run_grid(specs, outdir, env_id=FROZEN_SCALAR_ID,
             eval_cfg=eval_cfg_for(FROZEN_SCALAR_ID, args.eval_every, claim=args.claim))
    summarise(outdir)
    print("\nROBUSTNESS (plan B): re-run at 8-10 seeds before writing any of this")
    print("up as a conclusion. At 1-4 qubits that is affordable here - it has not")
    print("been on any previous experiment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
