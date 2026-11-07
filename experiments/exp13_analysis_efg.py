"""Experiment 13, blocks E, F, G - analysis (addendum of the design document).

    python experiments/exp13_analysis_efg.py --results-root <results> --out <dir>

Reads results/exp13_efg (tag exp13efg), the float64 traces results/exp13_traces/traces_efg.json and, as references, the
10 seeds of the REAL circuits already on disk (real PennyLane runs, not the emulator):
  CartPole   hybrid_fig4 (entangled, 126 params, exp01), hybrid_fig4_noent (exp08), cartpole_fourier_ceiling_L5 (exp01)
             and matched_classical (exp01)
  FrozenLake frozen_binary_4q_L1, frozen_binary_4q_L5 (entangled, exp04), their noent versions and frozen_matched_scalar
References are labelled "real_<arm>"; their per-seed values come from their own manifests with the same metric
definitions (final = mean of the last 10 % of training episodes; greedy = mean of the last 3 greedy checkpoints).
Writes per_seed_exp13_efg.csv, summary_exp13_efg.csv, contrasts_exp13_efg.csv, diag_exp13_efg.csv,
traces_exp13_efg.csv and tables_exp13_efg.md. Works on partial data. Criteria as in the design document.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp13_analysis as a13  # noqa: E402
from exp13_efg import SUBDIR, TAG  # noqa: E402
from qrl_dissection.core import lr_efg, stats  # noqa: E402

a10 = a13.a10
BLOCK_OF = {a: b for b, arms in lr_efg.BLOCK_ARMS.items() for a in arms}
REFS = {  # label -> (folder, arm)
    "real_hybrid_fig4": ("exp01_dqn_cartpole_capacity", "hybrid_fig4"),
    "real_hybrid_fig4_noent": ("exp08_dqn_cartpole_frozenlake_followups", "hybrid_fig4_noent"),
    "real_ceiling_L5": ("exp01_dqn_cartpole_capacity", "cartpole_fourier_ceiling_L5"),
    "real_matched_classical": ("exp01_dqn_cartpole_capacity", "matched_classical"),
    "real_frozen_binary_4q_L1": ("exp04_dqn_frozenlake_embeddings", "frozen_binary_4q_L1"),
    "real_frozen_binary_4q_L5": ("exp04_dqn_frozenlake_embeddings", "frozen_binary_4q_L5"),
    "real_frozen_binary_4q_noent_L1": ("exp04_dqn_frozenlake_embeddings", "frozen_binary_4q_noent_L1"),
    "real_frozen_binary_4q_noent_L5": ("exp04_dqn_frozenlake_embeddings", "frozen_binary_4q_noent_L5"),
    "real_frozen_matched_scalar": ("exp04_dqn_frozenlake_embeddings", "frozen_matched_scalar"),
}


def collect(root, out_dir):
    rows, diags = [], []
    d = root / SUBDIR
    for mp in sorted(d.glob("*.manifest.json")) if d.exists() else []:
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        sp, oc = m.get("spec"), m.get("outcome")
        if not sp or not oc or sp.get("tag") != TAG or sp.get("total_timesteps", 0) < 100_000:
            continue
        r = a10.run_record(m, d, "exp13efg", sp["arm"])
        if not r:
            continue
        r["row"].update(block=BLOCK_OF.get(sp["arm"], "?"), source="exp13", simulated=False)
        rows.append(r["row"])
        dc = a10.local(oc.get("diag_csv") or (oc.get("extra") or {}).get("diag_csv"), d)
        if dc is not None:
            df = pd.read_csv(dc)
            df.insert(0, "seed", sp["seed"])
            df.insert(0, "arm", sp["arm"])
            diags.append(df)
    for label, (folder, arm) in REFS.items():
        idx = a10.index_folder(root, folder)
        for (a, seed, _tag), m in sorted(idx.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if a != arm or not m["spec"].get("fix_autoreset", True) or m["spec"]["total_timesteps"] < 100_000:
                continue
            r = a10.run_record(m, root / folder, "ref", label)
            if r:
                sim = bool(m.get("provenance", {}).get("simulated", False))
                assert not sim, f"{label} seed {seed} is flagged simulated"
                r["row"].update(block="ref", source=folder, simulated=False)
                rows.append(r["row"])
    per = pd.DataFrame(rows).sort_values(["block", "arm", "seed"]).reset_index(drop=True)
    per.to_csv(out_dir / "per_seed_exp13_efg.csv", index=False)
    if diags:
        pd.concat(diags).to_csv(out_dir / "diag_exp13_efg.csv", index=False)
    return per


def design_params(root):
    p = root / SUBDIR / "design_params.json"
    return json.loads(p.read_text()) if p.exists() else {}


def summary(per, dp, out_dir):
    rows = []
    for arm, g in per.groupby("arm"):
        f, gr, ma = g.final.dropna().tolist(), g.greedy.dropna().tolist(), g.best_ma50.dropna().tolist()
        ci = lambda x: stats.bootstrap_ci(x) if len(x) > 1 else (np.nan, np.nan)      # noqa: E731
        (flo, fhi), (glo, ghi), (mlo, mhi) = ci(f), ci(gr), ci(ma)
        blk = g.block.iloc[0]
        fi, gi = stats.iqm(f), (stats.iqm(gr) if gr else np.nan)
        row = dict(block=blk, arm=arm, n=len(g), params=dp.get(arm, {}).get("params"),
                   final_perf_iqm=fi, final_ci_lo=flo, final_ci_hi=fhi, greedy_final_iqm=gi, greedy_ci_lo=glo,
                   greedy_ci_hi=ghi, best_ma50_iqm=stats.iqm(ma) if ma else np.nan, best_ma50_ci_lo=mlo,
                   best_ma50_ci_hi=mhi)
        if blk == "G" or arm.startswith("real_frozen"):
            # FrozenLake is deterministic with 0/1 returns: success rate = final_perf; a seed is solved when the greedy
            # policy reached the goal in all of its last 3 checkpoints (greedy_final == 1).
            row["n_solved_seeds"] = int((g.greedy.dropna() >= 0.999).sum())
            row["does_not_learn"] = np.nan
        else:
            row["n_solved_seeds"] = np.nan
            row["does_not_learn"] = bool(fi <= 12 and gi <= 12)
        row["real_circuit_reference"] = blk == "ref"
        rows.append(row)
    s = pd.DataFrame(rows).round(4)
    s.to_csv(out_dir / "summary_exp13_efg.csv", index=False)
    return s


def make_contrasts(per, out_dir):
    C = lambda fam, pred, a, b, note="": a13.contrast(per, fam, pred, a, b, note)          # noqa: E731
    rows = []
    # ---- E
    rows += [
        C("E_equivalence", "E1: lrx2 (fresh init) = dup", "ceiling_unique_lrx2", "ceiling_dup"),
        C("E_equivalence", "E1c: lrx2 collapsed (V0 = sum of the blocks) = dup", "ceiling_unique_lrx2_collapsed", "ceiling_dup"),
        C("E_equivalence", "E1d: fresh init = collapsed init, both lr x2", "ceiling_unique_lrx2", "ceiling_unique_lrx2_collapsed"),
        C("E_effect", "E2: no prediction (effect of the x2): unique vs dup", "ceiling_unique", "ceiling_dup"),
        C("E_effect", "E2b: unique vs unique lrx2", "ceiling_unique", "ceiling_unique_lrx2"),
        C("E_consistency", "E0: dup rerun vs the ceiling of exp01 (other pipeline: seeded sampler)", "ceiling_dup",
          "real_ceiling_L5", "10 seeds of the original run"),
    ]
    for arm in lr_efg.E_ARMS:
        for ref, lab in (("real_hybrid_fig4_noent", "no entanglement"), ("real_hybrid_fig4", "entangled")):
            rows.append(C("E_vs_real_hybrid", f"P({arm} > real hybrid {lab})", arm, ref, "REAL circuit, 10 seeds"))
    # ---- F
    quad = ["matched_rep4_w7", "matched_norep_w7_lrx4", "matched_norep_w18", "matched_norep_w18_lrx4"]
    rows += [
        C("F_equivalence", "F1: norep w7 lrx4 (fresh init) = rep4 w7", "matched_norep_w7_lrx4", "matched_rep4_w7"),
        C("F_equivalence", "F1c: norep w7 lrx4 collapsed = rep4 w7", "matched_norep_w7_lrx4_collapsed", "matched_rep4_w7"),
        C("F_equivalence", "F1d: fresh = collapsed init, both lr x4", "matched_norep_w7_lrx4", "matched_norep_w7_lrx4_collapsed"),
        C("F_consistency", "F0: rep4 rerun vs matched_classical of exp01 (other pipeline: seeded sampler)",
          "matched_rep4_w7", "real_matched_classical", "10 seeds of the original run"),
    ]
    for i, a in enumerate(quad):
        for b in quad[i + 1:]:
            if (a, b) not in (("matched_rep4_w7", "matched_norep_w7_lrx4"),):
                rows.append(C("F_four_arms", f"no prediction: {a} vs {b}", a, b))
    for arm in lr_efg.F_ARMS:
        rows.append(C("F_vs_real_hybrid", f"P(real hybrid without reuse > {arm})", "real_hybrid_fig4", arm,
                      "REAL circuit (126 params), 10 seeds"))
        rows.append(C("F_vs_real_hybrid_noent", f"P(real hybrid noent > {arm})", "real_hybrid_fig4_noent", arm,
                      "REAL circuit, 10 seeds"))
    # ---- G
    for arm in lr_efg.G_ARMS:
        for L in (1, 5):
            rows.append(C("G_vs_real_circuit", f"P({arm} > binary entangled circuit L={L})", arm,
                          f"real_frozen_binary_4q_L{L}", "REAL circuit, 10 seeds"))
    rows.append(C("G_pair", "no prediction: w12 vs w6", "mlp_bits_w12", "mlp_bits_w6"))
    df = pd.DataFrame(rows)
    df.insert(0, "block", df.family.str.split("_").str[0])
    df = df.drop(columns=["agent"], errors="ignore")
    df.round(4).to_csv(out_dir / "contrasts_exp13_efg.csv", index=False)
    return df


def traces(root, out_dir):
    p = root / "exp13_traces" / "traces_efg.json"
    if not p.exists():
        return pd.DataFrame()
    t = pd.DataFrame(json.loads(p.read_text())).T.reset_index(drop=True)
    t.to_csv(out_dir / "traces_exp13_efg.csv", index=False)
    return t


def md(df, cols, fmt=None):
    if df is None or len(df) == 0:
        return "(no data yet)\n"
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.values) + "\n"


def tables(per, s, c, tr, out_dir):
    out = ["# Experiment 13, blocks E, F, G - tables\n",
           "> All arms are classical on the real environment; the reference rows (`real_*`) are the REAL circuits "
           "(10 seeds, other pipeline: upstream's exploration sampler was not seeded there).\n"]
    for blk, title in (("E", "E - Fourier ceiling (CartPole)"), ("F", "F - capacity-matched control (CartPole)"),
                       ("G", "G - MLP on the 4 bits (FrozenLake)")):
        arms = list(lr_efg.BLOCK_ARMS[blk]) + [r for r in s.arm if r.startswith("real_") and
                                                   (("frozen" in r) == (blk == "G"))]
        ss = s[s.arm.isin(arms)].set_index("arm").reindex([a for a in arms if a in set(s.arm)]).reset_index()
        cols = ["arm", "params", "n", "final_perf_iqm", "final_ci_lo", "final_ci_hi", "greedy_final_iqm", "greedy_ci_lo",
                "greedy_ci_hi", "n_solved_seeds" if blk == "G" else "does_not_learn"]
        out += [f"\n## {title}\n", md(ss, cols)]
        cc = c[c.block == blk]
        out += [f"\n### Contrasts {blk}\n", md(cc, ["family", "prediction", "n_a", "n_b", "iqm_final_a", "iqm_final_b",
                                                   "p_final", "iqm_greedy_a", "iqm_greedy_b", "p_greedy", "status"])]
    out += ["\n## float64 traces (E, F)\n", md(tr, ["pair", "seed", "first_diff_step", "max_param_diff_to_first_diff",
                                                   "max_q_diff_final_2000_states", "greedy_disagreement_final",
                                                   "n_steps"])]
    (out_dir / "tables_exp13_efg.md").write_text("\n".join(out), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root, out = pathlib.Path(args.results_root), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per = collect(root, out)
    s = summary(per, design_params(root), out)
    c = make_contrasts(per, out)
    tr = traces(root, out)
    tables(per, s, c, tr, out)
    print(s[["block", "arm", "n", "params", "final_perf_iqm", "greedy_final_iqm"]].to_string(index=False))
    print(f"\n{len(c)} contrasts -> {out}")


if __name__ == "__main__":
    main()
