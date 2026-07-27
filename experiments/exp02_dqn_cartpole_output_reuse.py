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
from qrl_dissection.core.configs import OR_REPEATS, hybrid_or_config
from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, SafeDQN


def run_one(outdir, name, agent_type, cfg, seed, fix_autoreset, steps, kw):
    manifest = outdir / f"{name}.manifest.json"
    if manifest.exists():
        print(f"[skip] {name}")
        return json.loads(manifest.read_text())
    print(f"[run ] {name}", flush=True)
    runner = SafeDQN(agent_type=agent_type, agent_config=cfg, run_name=name,
                     seed=seed, fix_autoreset=fix_autoreset,
                     eval_cfg=GreedyEvalConfig(every_steps=10_000), outdir=outdir, **kw)
    out = runner.train(steps, progress_bar=False)
    m = {"name": name, "agent_type": agent_type, "seed": seed,
         "fix_autoreset": fix_autoreset, "outcome": out.__dict__,
         "config": {k: str(v) for k, v in cfg.items()}}
    manifest.write_text(json.dumps(m, indent=2, default=str))
    print(f"       ok {out.wall_seconds}s  phantoms {100*out.probe['frac_poison']:.2f}%")
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/exp02_dqn_cartpole_output_reuse")
    p.add_argument("--repeats", nargs="+", type=int, default=OR_REPEATS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--fix-autoreset", action="store_true", default=True)
    p.add_argument("--smoke", action="store_true", help="1 seed, R in {4,16} only")
    args = p.parse_args()

    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    if args.smoke:
        args.seeds, args.repeats = [1], [4, 16]

    outdir = pathlib.Path(args.outdir)
    kw = dict(batch_size=128, buffer_size=10_000, train_frequency=10)

    for R in args.repeats:
        cfg = hybrid_or_config(R)
        for seed in args.seeds:
            name = f"hybrid_OR{R}__s{seed}"
            run_one(outdir, name, "hybrid", cfg, seed, args.fix_autoreset, args.steps, kw)

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
