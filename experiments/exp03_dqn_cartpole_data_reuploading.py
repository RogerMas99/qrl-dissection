"""
Experiment 03 - Data Reuploading (DR) under DQN. Paper block 2 (embedding).

DR is circuit depth: repeating the encoding + variational layer L times. The
paper finds DR improves trainability and stability, with patterns consistent
with barren-plateau mitigation, but that the effect depends on the embedding
family. This experiment sweeps depth L on the Skolik template under DQN, holding
everything else at Fig. 4 values.

Same simple set-up as exp01 (8 qubits, 60k steps, 3 seeds) - a COVERAGE pass.
Scheduled for an 8-10 seed re-run later (plan "B" in docs/ROADMAP.md).

    python experiments/exp03_dqn_cartpole_data_reuploading.py --outdir results/exp03
    python experiments/exp03_dqn_cartpole_data_reuploading.py --smoke

CAVEAT (from exp01): the hybrid barely learns under DQN here. A DR effect on a
weakly-learning agent may be unmeasurable; if so, that is the finding. Also note
exp01's single-seed 103k run showed depth mattering (L=1 failed), so DR may be
the one block with a visible effect even in this regime - worth checking.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import qrl_dissection
from qrl_dissection import analysis
from qrl_dissection.core.configs import DR_DEPTHS, hybrid_dr_config
from qrl_dissection.dqn import GreedyEvalConfig, SafeDQN
from qrl_dissection.dqn.runner import reuse_or_none


def run_one(outdir, name, cfg, seed, fix_autoreset, steps, kw):
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
                     seed=seed, fix_autoreset=fix_autoreset,
                     eval_cfg=GreedyEvalConfig(every_steps=10_000), outdir=outdir, **kw)
    out = runner.train(steps, progress_bar=False)
    m = {"name": name, "seed": seed, "total_timesteps": steps, "dqn_kwargs": kw, "fix_autoreset": fix_autoreset,
         "outcome": out.__dict__, "config": {k: str(v) for k, v in cfg.items()}}
    manifest.write_text(json.dumps(m, indent=2, default=str))
    print(f"       ok {out.wall_seconds}s  phantoms {100*out.probe['frac_poison']:.2f}%")
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/exp03_dqn_cartpole_data_reuploading")
    p.add_argument("--depths", nargs="+", type=int, default=DR_DEPTHS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=100_000)  # DR needs more budget; matches exp02
    p.add_argument("--fix-autoreset", action="store_true", default=True)
    p.add_argument("--smoke", action="store_true", help="1 seed, L in {1,5}")
    args = p.parse_args()

    print(json.dumps(qrl_dissection.upstream_report(), indent=2))
    if args.smoke:
        args.seeds, args.depths = [1], [1, 5]

    # Smoke runs go to their own directory. They use short budgets, so

    # sharing a directory with the real pass would leave stale cells that

    # the reuse guard then (correctly) refuses to accept.  _smoke_outdir

    outdir = pathlib.Path(args.outdir) / "_smoke" if args.smoke else pathlib.Path(args.outdir)
    kw = dict(batch_size=128, buffer_size=10_000, train_frequency=10)

    for L in args.depths:
        cfg = hybrid_dr_config(L)
        for seed in args.seeds:
            name = f"hybrid_DR{L}__s{seed}"
            run_one(outdir, name, cfg, seed, args.fix_autoreset, args.steps, kw)

    print("\n=== summary (greedy_best by L) ===")
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
            L = int(m["name"].split("DR")[1].split("__")[0])
            rows.append(dict(L=L, seed=m["seed"], best_ma50=round(best, 1),
                             greedy_best=round(gb, 1)))
        df = pd.DataFrame(rows)
        if len(df):
            print(df.groupby("L").agg(best=("best_ma50", "mean"),
                                      greedy=("greedy_best", "mean")).round(1))
    except Exception as exc:
        print("summary skipped:", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
