"""Closing data, point 4: summaries of the manifest table (hours by experiment and class, totals, throughput).
    python scripts/closing_data/p4_summarize.py --manifests-csv <p4_all_manifests.csv> --out <dir>
Runs found under a *quarantine* folder are excluded from every figure (8 runs, listed separately).
"""
import argparse, pathlib
import pandas as pd


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifests-csv", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.manifests_csv); d["hours"] = d.wall_seconds / 3600
    q = d[d.quarantined]; d = d[~d.quarantined].copy()
    q.to_csv(out / "p4_excluded_quarantine_runs.csv", index=False)
    t = d.pivot_table(index="experiment", columns="class", values="hours", aggfunc="sum", fill_value=0)
    for c in ("real_circuit", "emulator", "classical"):
        if c not in t: t[c] = 0.0
    t = t[["real_circuit", "emulator", "classical"]]; t["total_hours"] = t.sum(axis=1); t["n_runs"] = d.groupby("experiment").size()
    t.loc["TOTAL"] = t.sum(); t.round(2).to_csv(out / "p4_hours_by_experiment_and_class.csv")
    # exp01..exp08 (the set the thesis' ~3,100 h refers to) vs everything
    first = d[d.experiment.str.match(r"exp0[1-8]")]
    s = pd.DataFrame({"set": ["exp01-exp08 (all manifests)", "exp09-exp13 (all manifests)", "everything"],
                      "hours": [first.hours.sum(), d.hours.sum() - first.hours.sum(), d.hours.sum()],
                      "runs": [len(first), len(d) - len(first), len(d)]}).round(1)
    s.to_csv(out / "p4_hours_cumulative_sets.csv", index=False)
    g = d.groupby(["class", "agent_type", "env_id", "recorded_path_kind"]).agg(
        n_runs=("hours", "size"), total_hours=("hours", "sum"), median_hours_per_run=("hours", "median"),
        median_steps_per_s=("steps", lambda x: float("nan")))
    d["sps"] = d.steps / d.wall_seconds
    g["median_steps_per_s"] = d.groupby(["class", "agent_type", "env_id", "recorded_path_kind"]).sps.median()
    g["p10_steps_per_s"] = d.groupby(["class", "agent_type", "env_id", "recorded_path_kind"]).sps.quantile(0.1)
    g["p90_steps_per_s"] = d.groupby(["class", "agent_type", "env_id", "recorded_path_kind"]).sps.quantile(0.9)
    g.round(2).to_csv(out / "p4_throughput_by_agent_type_and_recorded_path.csv")
    # the thesis' h/seed: hybrid without reuse (exp01, FIX-01 on) vs Output Reuse R=32 (exp02, FIX-01 on)
    from scipy import stats
    e1 = d[(d.experiment == "exp01_dqn_cartpole_capacity") & d.run_name.astype(str).str.startswith("hybrid_fig4__fix01on") & (d.steps == 100000)].copy()
    e2 = d[(d.experiment == "exp02_dqn_cartpole_output_reuse") & d.run_name.astype(str).str.startswith("hybrid_OR") & (d.steps == 100000)].copy()
    e1["arm_label"] = "hybrid_fig4 (no reuse)"; e2["arm_label"] = e2.run_name.astype(str).str.split("__").str[0]
    x = pd.concat([e1, e2])[["arm_label", "seed", "hours", "recorded_path_kind", "python", "git_revision"]].sort_values(["arm_label", "seed"])
    x["fast_cluster_lt7h"] = x.hours < 7.0
    x.round(3).to_csv(out / "p4_per_seed_hours_hybrid_noreuse_vs_OR.csv", index=False)
    summ = x.groupby("arm_label").agg(n=("hours", "size"), mean_h=("hours", "mean"), sd_h=("hours", "std"), min_h=("hours", "min"),
                                      max_h=("hours", "max"), n_below_7h=("fast_cluster_lt7h", "sum")).round(2)
    a_, b_ = x[x.arm_label == "hybrid_fig4 (no reuse)"].hours, x[x.arm_label == "hybrid_OR32"].hours
    tt = stats.ttest_ind(a_, b_, equal_var=False); mw = stats.mannwhitneyu(a_, b_)
    summ.loc["Welch t-test no-reuse vs OR32 (p)"] = [tt.pvalue] + [float("nan")] * 5
    summ.loc["Mann-Whitney U (p)"] = [mw.pvalue] + [float("nan")] * 5
    summ.to_csv(out / "p4_hybrid_hours_per_seed_summary.csv")
    print(summ.to_string())
    print(t.round(1).to_string()); print(s.to_string())


if __name__ == "__main__":
    main()
