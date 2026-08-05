"""Reduce the paper repo's TensorBoard logs to the two CSVs in data/.

The companion repository (javier-lazaro/qrl-dissection) ships ~140 MB of event
files: 3 blocks x 36 configurations x 10 seeds, 100k steps each. We do not vendor
that. This script turns it into 22 KB of CSV, summarised with OUR metric so the
comparison against our DQN runs is like-for-like on the statistic, if never on
the algorithm.

    pip install tbparse
    python scripts/extract_paper_baselines.py --results <path>/qrl-dissection/results

Re-run it if the upstream results change; the CSVs in data/ are derived
artefacts and should never be edited by hand.
"""
import argparse
import os
import pathlib
from glob import glob

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True,
                   help="path to the paper repo's results/ directory")
    p.add_argument("--outdir", default="data")
    p.add_argument("--window", type=int, default=50,
                   help="moving-average window; 50 matches core.analysis")
    args = p.parse_args()

    from tbparse import SummaryReader

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seeddir in sorted(glob(os.path.join(args.results, "*", "*", "Seed*"))):
        parts = seeddir.split(os.sep)
        block, config, seed = parts[-3], parts[-2], parts[-1]
        scalars = SummaryReader(seeddir).scalars
        ret_rows = scalars[scalars.tag == "charts/episodic_return"].sort_values("step")
        if ret_rows.empty:
            print(f"  ! no episodic_return in {seeddir}")
            continue
        returns = ret_rows.value.values
        ma = pd.Series(returns).rolling(args.window).mean()
        rows.append(dict(
            block=block, config=config, seed=int(seed[-2:]),
            n_episodes=len(returns), max_step=int(ret_rows.step.max()),
            best_ma50=round(float(np.nanmax(ma)), 2),
            final_ma50=round(float(ma.iloc[-1]), 2),
            mean_return=round(float(returns.mean()), 2)))

    per_seed = pd.DataFrame(rows).sort_values(["block", "config", "seed"])
    per_seed.to_csv(outdir / "paper_ppo_baselines.csv", index=False)

    summary = (per_seed.groupby(["block", "config"])
               .agg(n_seeds=("seed", "count"),
                    best_ma50_mean=("best_ma50", "mean"),
                    best_ma50_sd=("best_ma50", "std"),
                    final_ma50_mean=("final_ma50", "mean"),
                    final_ma50_sd=("final_ma50", "std"))
               .round(2).reset_index())
    summary.to_csv(outdir / "paper_ppo_summary.csv", index=False)

    print(f"{len(per_seed)} seed-runs, {len(summary)} configurations")
    print(f"wrote {outdir/'paper_ppo_baselines.csv'} and {outdir/'paper_ppo_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
