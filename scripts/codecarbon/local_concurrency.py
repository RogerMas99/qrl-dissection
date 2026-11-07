"""How many campaign runs shared this PC at the same time? (needed to choose the single/parallel calibration condition)

For every run whose manifest says it wrote its files on this PC (recorded_path_kind = windows_path) the interval is
[mtime of its episode CSV - wall_seconds, mtime of its episode CSV]. A sweep over all intervals gives, per run, the time-weighted mean
number of runs alive during it (itself included). Caveats: only runs with a manifest are counted (not tests, traces or analyses), and the
CSV modification time is the time of its last write on this PC.
    python scripts/codecarbon/local_concurrency.py --results-root <results> --manifests <p4_all_manifests.csv> --out <csv>
"""
import argparse
import pathlib

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--manifests", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = pathlib.Path(a.results_root)
    d = pd.read_csv(a.manifests)
    d = d[(~d.quarantined) & (d.recorded_path_kind == "windows_path")].copy()
    d["end"] = [(root / r.experiment / "runs" / f"{r.run_name}.csv").stat().st_mtime for r in d.itertuples()]
    d["start"] = d.end - d.wall_seconds
    ev = sorted([(s, 1) for s in d.start] + [(e, -1) for e in d.end])
    times, cur = np.array([t for t, _ in ev]), np.cumsum([x for _, x in ev])

    def mean_conc(s, e):
        i0 = np.searchsorted(times, s, side="right") - 1
        i1 = np.searchsorted(times, e, side="left")
        tt = np.concatenate([[s], times[max(i0 + 1, 0):i1], [e]])
        cc = np.concatenate([[cur[i0] if i0 >= 0 else 0], cur[max(i0 + 1, 0):i1]])[: len(tt) - 1]
        dt = np.diff(tt)
        return float((cc * dt).sum() / dt.sum()) if dt.sum() > 0 else 1.0

    d["mean_concurrency"] = [mean_conc(s, e) for s, e in zip(d.start, d.end)]
    d["hours"] = d.wall_seconds / 3600
    d[["experiment", "run_name", "agent_type", "env_id", "hours", "start", "end", "mean_concurrency"]].to_csv(a.out, index=False)
    print(len(d), "local runs;", round(d.hours.sum(), 1), "run-hours")


if __name__ == "__main__":
    main()
