"""
Experiment 02 - Output Reuse (OR) under DQN. Paper block 1 (post-PQC inference).

The paper's first dissection block. OR replicates the PQC readout R times before
the linear head, enlarging the inference layer's input. The paper's finding
(under PPO): OR helps hybrid agents but not classical ones, so its benefit is a
genuine quantum-classical interaction, not a classical scaling heuristic.

This experiment asks whether that transfers to DQN. Same simple set-up as exp01
(8 qubits, 60k steps, 3 seeds) - a COVERAGE pass, not the statistically-robust
run. Every block sweep is scheduled for an 8-10 seed re-run later (plan "B" in
docs/ROADMAP.md).

Design (mirrors the paper): sweep R in {4, 8, 16, 32} on the hybrid arm, plus a
classical control that applies the same OR to the observation, to test whether
any OR benefit is quantum-dependent as the paper claims.

    python experiments/exp02_dqn_cartpole_output_reuse.py --outdir results/exp02
    python experiments/exp02_dqn_cartpole_output_reuse.py --smoke   # 1 seed, 2 R values

CAVEAT (learned in exp01): the hybrid barely learns under DQN in this regime
(greedy ~44, high variance). An OR effect measured on a weakly-learning agent may
be lost in noise - the same trap FIX-01 fell into. Read results with that in mind;
if the base hybrid is too weak, OR is not measurable and that itself is the finding.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import qrl_dissection
from qrl_dissection import analysis
from qrl_dissection.core.configs import (
    OR_REPEATS,
    hybrid_or_config,
    paper_classical_or_config,
)
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, SafeDQN, run_grid
from qrl_dissection.dqn.runner import claim_cell, release_cell, reuse_or_none

# Set from --claim. A list so run_one can read it without a global.
CLAIM = [False]


def run_one(outdir, name, agent_type, cfg, seed, fix_autoreset, steps, kw):
    manifest = outdir / f"{name}.manifest.json"
    # A same-named manifest is only a hit if it answers the same question. See
    # runner.reuse_or_none: a 5k smoke cell used to satisfy a 100k request.
    hit = reuse_or_none(manifest, {"seed": seed, "total_timesteps": steps,
                                   "dqn_kwargs": kw})
    if hit is not None:
        print(f"[skip] {name}")
        return hit
    if CLAIM[0] and not claim_cell(manifest):
        print(f"busy   {name} - another session holds this cell")
        return None
    print(f"[run ] {name}", flush=True)
    runner = SafeDQN(agent_type=agent_type, agent_config=cfg, run_name=name,
                     seed=seed, fix_autoreset=fix_autoreset,
                     eval_cfg=GreedyEvalConfig(every_steps=10_000), outdir=outdir, **kw)
    out = runner.train(steps, progress_bar=False)
    m = {"name": name, "agent_type": agent_type, "seed": seed,
         "fix_autoreset": fix_autoreset, "outcome": out.__dict__,
         "config": {k: str(v) for k, v in cfg.items()},
         # FIX: total_timesteps and dqn_kwargs were never written here, so a
         # manifest produced by this function was legacy the moment it was
         # created - a cell finished today could be rejected as unverifiable
         # tomorrow. Reproduced directly: hybrid_OR4__s4 completed successfully
         # overnight, then failed the reuse guard the next morning with exactly
         # that message. exp03 and exp03b never had this bug; only exp02 did.
         "total_timesteps": steps, "dqn_kwargs": kw}
    manifest.write_text(json.dumps(m, indent=2, default=str))
    if CLAIM[0]:
        release_cell(manifest)
    print(f"       ok {out.wall_seconds}s  phantoms {100*out.probe['frac_poison']:.2f}%")
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--claim", action="store_true",
                   help="cooperative locking across parallel sessions")
    p.add_argument("--outdir", default="results/exp02_dqn_cartpole_output_reuse")
    p.add_argument("--repeats", nargs="+", type=int, default=OR_REPEATS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=100_000)  # match DR (exp03)
    p.add_argument("--fix-autoreset", action="store_true", default=True)
    p.add_argument("--no-classical", action="store_true",
                   help="hybrid arm only. The classical control is what makes "
                        "the OR claim testable, so skip it only deliberately.")
    p.add_argument("--smoke", action="store_true", help="1 seed, R in {4,16} only")
    p.add_argument("--with-matched-control", action="store_true",
                   help="[FIX-09 follow-up] also run classical_or_matched_r{R}: "
                        "NEW-02's recipe (full observation, budget matched to "
                        "hybrid_OR{R}'s own total) applied to OR. classical_OR{R} "
                        "above is NOT capacity-matched - see docs/CORRECTIONS.md#fix-09 "
                        "- so this is the arm that can actually test whether OR's "
                        "benefit is quantum-specific. Runs through RunSpec/run_grid "
                        "(not run_one), so it gets correct spec.arm and git_revision "
                        "from the start, unlike the two loops above.")
    args = p.parse_args()
    CLAIM[0] = args.claim

    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    if args.smoke:
        args.seeds, args.repeats = [1], [4, 16]

    # Smoke runs go to their own directory. They use short budgets, so

    # sharing a directory with the real pass would leave stale cells that

    # the reuse guard then (correctly) refuses to accept.  _smoke_outdir

    outdir = pathlib.Path(args.outdir) / "_smoke" if args.smoke else pathlib.Path(args.outdir)
    kw = dict(batch_size=128, buffer_size=10_000, train_frequency=10)

    for R in args.repeats:
        cfg = hybrid_or_config(R)
        for seed in args.seeds:
            name = f"hybrid_OR{R}__s{seed}"
            run_one(outdir, name, "hybrid", cfg, seed, args.fix_autoreset, args.steps, kw)

    # THE CONTROL. Without it this experiment cannot test the claim it exists to
    # test. The paper's finding is not "OR helps" but "OR helps hybrid agents and
    # not classical ones" - a statement about the DIFFERENCE between two arms. A
    # hybrid-only sweep can show OR helping and still say nothing about whether
    # the help is quantum, because a classical net given the same repeated
    # observation may improve just as much. exp01 was nearly written up with a
    # broken control; this is the same lesson on the other axis.
    #
    # Cheap: these are 26-194 parameter classical nets with no PQC, so a cell is
    # minutes against hours for its hybrid counterpart. Config transcribed from
    # post-pqc-inference.py :: config_classical.
    if not args.no_classical:
        for R in args.repeats:
            cfg = paper_classical_or_config(R)
            for seed in args.seeds:
                name = f"classical_OR{R}__s{seed}"
                run_one(outdir, name, "classic", cfg, seed, args.fix_autoreset,
                        args.steps, kw)

    # [FIX-09 follow-up] THE CAPACITY-MATCHED CONTROL. classical_OR{R} above
    # keeps the paper's amputated, zero-hidden-layer design at every R while OR
    # grows the hybrid's own head with R (measured total 222/350/606/1118 for
    # R=4/8/16/32 vs classical_OR's 26/50/98/194 - a gap that widens with R,
    # not a fixed shortfall). Until this arm exists, exp02 can only say the
    # paper's own classical design dies under DQN here too - not whether OR's
    # benefit is quantum-specific. Runs through RunSpec/run_grid, which is why
    # its manifests need no migration and carry git_revision from the start.
    if args.with_matched_control:
        specs = [RunSpec(arm=f"classical_or_matched_r{R}", seed=seed,
                         fix_autoreset=args.fix_autoreset, total_timesteps=args.steps,
                         dqn_kwargs=kw)
                 for R in args.repeats for seed in args.seeds]
        run_grid(specs, outdir, env_id="CartPole-v1",
                eval_cfg=GreedyEvalConfig(every_steps=10_000), claim=CLAIM[0])

    print("\n=== summary (greedy_best by R) ===")
    try:
        import pandas as pd
        rows = []
        for mp in sorted(outdir.glob("*.manifest.json")):
            m = json.loads(mp.read_text())
            oc = m["outcome"]
            rew, step = analysis.load_episodes(oc["episodes_csv"])
            best = float(pd.Series(rew).rolling(50).mean().max())
            gb = float("nan")
            if oc.get("eval_csv") and pathlib.Path(oc["eval_csv"]).exists():
                _, sc = analysis.load_eval(oc["eval_csv"])
                gb = float(max(sc)) if len(sc) else float("nan")
            R = int(m["name"].split("OR")[1].split("__")[0])
            rows.append(dict(R=R, seed=m["seed"], best_ma50=round(best, 1),
                             greedy_best=round(gb, 1)))
        df = pd.DataFrame(rows)
        if len(df):
            print(df.groupby("R").agg(best=("best_ma50", "mean"),
                                      greedy=("greedy_best", "mean")).round(1))
    except Exception as exc:
        print("summary skipped:", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
