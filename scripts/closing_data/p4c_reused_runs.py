"""Closing data, point 4: runs of exp10 that were REUSED from earlier folders instead of re-run
(results/exp10_emulator_x100/reused_seeds.json), with the wall hours of each source run.
    python scripts/closing_data/p4c_reused_runs.py --results-root <dir> --manifests-csv <p4_all_manifests.csv> --out <csv>
"""
import argparse, csv, json, pathlib
import pandas as pd


def main():
    p = argparse.ArgumentParser(); p.add_argument("--results-root", required=True); p.add_argument("--manifests-csv", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    root = pathlib.Path(a.results_root)
    reused = json.loads((root / "exp10_emulator_x100" / "reused_seeds.json").read_text())
    d = pd.read_csv(a.manifests_csv)
    d = d[~d.quarantined]
    d["tag"] = d["tag"].fillna("")
    rows = []
    for arm, seeds in reused.items():
        for seed, (src, tag) in seeds.items():
            m = d[(d.experiment == src) & (d.arm == arm) & (d.seed == int(seed)) & (d.tag == (tag or "")) & (d.steps == 100000)]
            rows.append(dict(arm=arm, seed=int(seed), source_experiment=src, source_tag=tag, n_source_manifests=len(m),
                             wall_hours=round(float(m.wall_seconds.iloc[0]) / 3600, 4) if len(m) == 1 else ""))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r["wall_hours"] != ""]
    print(len(rows), "reused runs;", len(ok), "with exactly one source manifest; hours saved =", round(sum(r["wall_hours"] for r in ok), 1))
    print("not matched:", [(r["arm"], r["seed"], r["n_source_manifests"]) for r in rows if r["wall_hours"] == ""][:10])


if __name__ == "__main__":
    main()
