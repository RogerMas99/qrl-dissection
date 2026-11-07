"""Experiment 13 analysis - Output Reuse against the learning rate of the head (docs/EXPERIMENT-13-OR-VS-LR.md).

    python experiments/exp13_analysis.py --results-root <results> --out <dir>

Reads the manifests of results/exp13_lr_equivalence (agents T = classical twin, H = hybrid on the EMULATOR, simulated),
results/exp13_ppo (block D) and results/exp13_traces (float64 traces); writes per_seed_exp13.csv,
curves_exp13_train_ma50.csv, curves_exp13_eval_greedy.csv, diag_exp13_dqn.csv, diag_exp13_ppo.csv, summary_exp13.csv,
contrasts_exp13.csv, shift_exp13.csv, clip_exp13.csv and tables_exp13.md. Works on whatever cells exist (partial runs).
The criteria are those of the design document: equivalence = P(a>b) in [0.35, 0.65] in both metrics and overlapping 95 % CIs
of the IQMs; strong = P >= 0.95 in both metrics (the project's verdicts for the rest).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re

import numpy as np
import pandas as pd

from qrl_dissection.core import stats

HERE = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a10 = _load("exp10_analysis")
a11 = _load("exp11_analysis")
e13 = _load("exp13_lr_equivalence")
p13 = _load("exp13_ppo")

DQN_DIR, PPO_DIR, TRACE_DIR = e13.SUBDIR, p13.SUBDIR, "exp13_traces"
P, verdict, ci_str = a11.P, a11.verdict, a11.ci_str
RS = (4, 8, 16, 32)
KS = (2, 4, 8, 16, 32, 64, 128)
AG = {"T": "classical_twin", "H": "hybrid_emu"}
NOTE_SIM = ("> Agent H runs on the batched statevector EMULATOR: simulated results (docs/SIMULATED-RESULTS-POLICY.md; "
            "approval of 2026-11-06 recorded there). Agent T and block D are classical arms on the real environment.\n")


def agent_of(arm):
    if arm.startswith("ppo_"):
        return "PPO"
    return "T" if arm.startswith("classical_twin") else "H"


# ---------------------------------------------------------------------------------------------------- collection
def collect_folder(root, folder, out_rows, out_tr, out_ev, out_diag, out_clip):
    d = root / folder
    if not d.exists():
        return
    for mp in sorted(d.glob("*.manifest.json")):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        sp, oc = m.get("spec"), m.get("outcome")
        if not sp or not oc or sp.get("tag") != "exp13" or sp.get("total_timesteps", 0) < 100_000:
            continue
        arm = sp["arm"]
        r = a10.run_record(m, d, "exp13", arm)
        if not r:
            continue
        row = r["row"]
        row.update(agent=agent_of(arm), simulated=bool(m.get("provenance", {}).get("simulated", False)))
        if "clip" in oc:
            row.update(clip_fraction=oc["clip"].get("clip_fraction"), median_factor=oc["clip"].get("median_factor"),
                       median_norm=oc["clip"].get("median_norm"))
        out_rows.append(row)
        key = [arm, sp["seed"]]
        out_tr += [key + [s, v] for s, v in r["train"]]
        out_ev += [key + [s, v] for s, v in r["evals"]]
        dc = a10.local(oc.get("diag_csv") or (oc.get("extra") or {}).get("diag_csv"), d)
        if dc is not None:
            df = pd.read_csv(dc)
            df.insert(0, "seed", sp["seed"])
            df.insert(0, "arm", arm)
            out_diag.append(df)


def collect(root, out_dir):
    rows, tr, ev, dq, dp, cl = [], [], [], [], [], []
    collect_folder(root, DQN_DIR, rows, tr, ev, dq, cl)
    n_dqn = len(rows)
    collect_folder(root, PPO_DIR, rows, tr, ev, dp, cl)
    per = pd.DataFrame(rows).sort_values(["agent", "arm", "seed"]).reset_index(drop=True)
    per.to_csv(out_dir / "per_seed_exp13.csv", index=False)
    pd.DataFrame(tr, columns=["arm", "seed", "global_step", "train_ma50"]).to_csv(
        out_dir / "curves_exp13_train_ma50.csv", index=False)
    pd.DataFrame(ev, columns=["arm", "seed", "global_step", "greedy_return"]).to_csv(
        out_dir / "curves_exp13_eval_greedy.csv", index=False)
    if dq:
        pd.concat(dq).to_csv(out_dir / "diag_exp13_dqn.csv", index=False)
    if dp:
        pd.concat(dp).to_csv(out_dir / "diag_exp13_ppo.csv", index=False)
    return per, (pd.concat(dq) if dq else pd.DataFrame()), (pd.concat(dp) if dp else pd.DataFrame())


def design_params(root):
    p = root / DQN_DIR / "design_params.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------------------------------------------ summary
def summary(per, dp, out_dir):
    rows = []
    for arm, g in per.groupby("arm"):
        f, gr, ma = g.final.dropna().tolist(), g.greedy.dropna().tolist(), g.best_ma50.dropna().tolist()
        flo, fhi = stats.bootstrap_ci(f) if len(f) > 1 else (np.nan, np.nan)
        glo, ghi = stats.bootstrap_ci(gr) if len(gr) > 1 else (np.nan, np.nan)
        mlo, mhi = stats.bootstrap_ci(ma) if len(ma) > 1 else (np.nan, np.nan)
        fi, gi = stats.iqm(f), (stats.iqm(gr) if gr else np.nan)
        rows.append(dict(agent=g.agent.iloc[0], arm=arm, n=len(g), params=dp.get(arm, {}).get("params"),
                         final_perf_iqm=fi, final_ci_lo=flo, final_ci_hi=fhi,
                         greedy_final_iqm=gi, greedy_ci_lo=glo, greedy_ci_hi=ghi,
                         best_ma50_iqm=stats.iqm(ma) if ma else np.nan, best_ma50_ci_lo=mlo, best_ma50_ci_hi=mhi,
                         does_not_learn=bool(fi <= 12 and gi <= 12), simulated=bool(g.simulated.iloc[0])))
    s = pd.DataFrame(rows).round(4)
    s.to_csv(out_dir / "summary_exp13.csv", index=False)
    return s


# ---------------------------------------------------------------------------------------------------- contrasts
def _iqm_ci(per, arm, col):
    x = per[per.arm == arm][col].dropna().tolist()
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    lo, hi = stats.bootstrap_ci(x)
    return stats.iqm(x), lo, hi


def contrast(per, family, pred, a, b, note=""):
    pf, na, nb = P(per, a, b, "final")
    pg, _, _ = P(per, a, b, "greedy")
    row = dict(agent=agent_of(a), family=family, prediction=pred, a=a, b=b, n_a=na, n_b=nb, note=note)
    if na == 0 or nb == 0:
        return row
    fa, fb = _iqm_ci(per, a, "final"), _iqm_ci(per, b, "final")
    ga, gb = _iqm_ci(per, a, "greedy"), _iqm_ci(per, b, "greedy")
    overlap = lambda x, y: bool(x[1] <= y[2] and y[1] <= x[2])          # noqa: E731
    in_band = lambda p: 0.35 <= p <= 0.65                                   # noqa: E731
    ov_f, ov_g = overlap(fa, fb), overlap(ga, gb)
    equiv = in_band(pf) and in_band(pg) and ov_f and ov_g
    v = verdict(pf, pg)
    if equiv:
        status = "equivalent"
    elif v in ("strong", "strong (reverse)") or v.startswith("indicates"):
        status = f"not equivalent ({v})"
    elif v == "mixed":
        status = "not equivalent (mixed)"
    else:
        status = "inconclusive"
    row.update(iqm_final_a=fa[0], iqm_final_b=fb[0], p_final=pf, iqm_greedy_a=ga[0], iqm_greedy_b=gb[0], p_greedy=pg,
               ci_overlap_final=ov_f, ci_overlap_greedy=ov_g, verdict=v, status=status)
    return row


def best_arm(per, arms, col="final"):
    cand = [(stats.iqm(per[per.arm == a][col].dropna().tolist()), a) for a in arms if (per.arm == a).any()]
    return max(cand)[1] if cand else None


def dqn_contrasts(per, pre):
    rows = []
    for R in RS:
        rows.append(contrast(per, "A", "P1/P5", f"{pre}_collapsed_r{R}", f"{pre}_reuse_r{R}"))
        rows.append(contrast(per, "A", "P2/P5", f"{pre}_headlr_x{R}", f"{pre}_reuse_r{R}"))
    for R in (8, 32):
        rows.append(contrast(per, "A", "P3/P5", f"{pre}_gradx{R}", f"{pre}_reuse_r1"))
    for R in RS:
        rows.append(contrast(per, "ref", "effect of OR", f"{pre}_reuse_r{R}", f"{pre}_reuse_r1"))
    for k in KS:
        rows.append(contrast(per, "B", "P4", f"{pre}_headlr_x{k}", f"{pre}_reuse_r1"))
    bf = best_arm(per, [f"{pre}_headlr_x{k}" for k in KS])
    br = best_arm(per, [f"{pre}_reuse_r{R}" for R in RS])
    if bf and br:
        rows.append(contrast(per, "B", "P4", bf, br, note=f"best fresh ({bf}) vs best reuse ({br}), chosen by IQM final_perf"))
    for k in e13.L.KS_GLOBAL:
        rows.append(contrast(per, "C", "P6", f"{pre}_globallr_x{e13.L.kname(k)}", f"{pre}_reuse_r1"))
    return rows


def ppo_contrasts(per):
    rows = []
    for R in RS:
        rows.append(contrast(per, "D", "P7 (clip)", f"ppo_lrx_clip_r{R}", f"ppo_reuse_clip_r{R}"))
        rows.append(contrast(per, "D", "P7 (no clip)", f"ppo_lrx_noclip_r{R}", f"ppo_reuse_noclip_r{R}"))
    for R in (8, 32):
        rows.append(contrast(per, "D", "P8", f"ppo_lrx_modclip_r{R}", f"ppo_reuse_clip_r{R}"))
        rows.append(contrast(per, "D", "P8 (usual rule)", f"ppo_lrx_clip_r{R}", f"ppo_lrx_modclip_r{R}",
                             note="usual vs modified clipping on the same lr x R network"))
    for R in (1,) + RS:
        rows.append(contrast(per, "D", "clip vs no clip (reuse)", f"ppo_reuse_clip_r{R}", f"ppo_reuse_noclip_r{R}"))
    for R in RS:
        rows.append(contrast(per, "D", "effect of OR (clip)", f"ppo_reuse_clip_r{R}", "ppo_reuse_clip_r1"))
        rows.append(contrast(per, "D", "effect of OR (no clip)", f"ppo_reuse_noclip_r{R}", "ppo_reuse_noclip_r1"))
    return rows


def shift_ci(x, y, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = [stats.iqm(rng.choice(x, x.size)) - stats.iqm(rng.choice(y, y.size)) for _ in range(n_boot)]
    return stats.iqm(x) - stats.iqm(y), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def shifts(per, pre):
    rows = []
    for R in RS:
        a, b = f"{pre}_reuse_r{R}", f"{pre}_headlr_x{R}"
        row = dict(agent=agent_of(a), R=R, reuse=a, fresh=b)
        row.update({f"shift_{c}{suf}": float("nan") for c in ("final", "greedy") for suf in ("", "_lo", "_hi")})
        for col in ("final", "greedy"):
            x, y = per[per.arm == a][col].dropna().tolist(), per[per.arm == b][col].dropna().tolist()
            if len(x) > 1 and len(y) > 1:
                d, lo, hi = shift_ci(x, y)
                row.update({f"shift_{col}": d, f"shift_{col}_lo": lo, f"shift_{col}_hi": hi})
        rows.append(row)
    return rows


def clip_table(per):
    q = per[per.agent == "PPO"]
    if q.empty:
        return pd.DataFrame()
    g = q.groupby("arm").agg(n=("seed", "count"), clip_fraction_mean=("clip_fraction", "mean"),
                             clip_fraction_min=("clip_fraction", "min"), median_factor=("median_factor", "median"),
                             median_norm=("median_norm", "median")).reset_index()
    return g.round(4)


# ------------------------------------------------------------------------------------------------- scale tracking
def scale_table(dq, arms, steps=(10_000, 30_000, 60_000, 100_000)):
    if dq.empty:
        return []
    out = []
    for arm in arms:
        g = dq[dq.arm == arm]
        if g.empty:
            continue
        row = {"arm": arm}
        for s in steps:
            h = g[g.step == s]
            row[f"V@{s // 1000}k"] = h.v_eff_fro.median() if len(h) else np.nan
            row[f"|Q|@{s // 1000}k"] = h.mean_abs_q.median() if len(h) else np.nan
        out.append(row)
    return out


def traces(root):
    d = root / TRACE_DIR
    rows = []
    if not d.exists():
        return rows
    for f in sorted(d.glob("trace64_*.json")):
        for k, v in json.loads(f.read_text()).items():
            rows.append(dict(key=k, **v))
    return rows


# ------------------------------------------------------------------------------------------------------ markdown
def md_table(df, cols=None, fmt=None):
    if df is None or len(df) == 0:
        return "(no data)\n"
    cols = cols or list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else (fmt or {}).get(c, "{:.3g}").format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def summary_table(per, s, arms):
    out = ["| arm | n | IQM final_perf (95 % CI) | IQM greedy_final (95 % CI) | learns |", "|---|---|---|---|---|"]
    for a in arms:
        g = per[per.arm == a]
        if g.empty:
            continue
        r = s[s.arm == a].iloc[0]
        out.append(f"| {a} | {len(g)} | {ci_str(g.final.dropna().tolist())} | {ci_str(g.greedy.dropna().tolist())} | "
                   f"{'no' if r.does_not_learn else 'yes'} |")
    return "\n".join(out) + "\n"


def contrast_md(rows):
    out = ["| prediction | a vs b | n | IQM final a / b | P final | IQM greedy a / b | P greedy | CI overlap (f/g) | status |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "p_final" not in r:
            out.append(f"| {r['prediction']} | {r['a']} vs {r['b']} | {r['n_a']}/{r['n_b']} | (not available) | | | | | |")
            continue
        out.append(f"| {r['prediction']} | {r['a']} vs {r['b']}{' (' + r['note'] + ')' if r.get('note') else ''} | "
                   f"{r['n_a']}/{r['n_b']} | {r['iqm_final_a']:.1f} / {r['iqm_final_b']:.1f} | {r['p_final']:.2f} | "
                   f"{r['iqm_greedy_a']:.1f} / {r['iqm_greedy_b']:.1f} | {r['p_greedy']:.2f} | "
                   f"{'yes' if r['ci_overlap_final'] else 'no'} / {'yes' if r['ci_overlap_greedy'] else 'no'} | {r['status']} |")
    return "\n".join(out) + "\n"


def status_counts(rows, pred_prefix):
    sel = [r for r in rows if r["prediction"].startswith(pred_prefix) and "status" in r]
    if not sel:
        return "no data"
    c = {}
    for r in sel:
        c[r["status"]] = c.get(r["status"], 0) + 1
    return f"{len(sel)} pairs: " + ", ".join(f"{v} {k}" for k, v in sorted(c.items()))


def sweep_md(per, pre):
    out = ["| k or R | IQM final: fresh x k | IQM final: reuse R | IQM greedy: fresh x k | IQM greedy: reuse R |",
           "|---|---|---|---|---|"]
    for n in (1, 2, 4, 8, 16, 32, 64, 128):
        fr = f"{pre}_headlr_x{n}" if n > 1 else f"{pre}_reuse_r1"
        ru = f"{pre}_reuse_r{n}" if n in (1, 4, 8, 16, 32) else None
        cell = lambda arm, col: (ci_str(per[per.arm == arm][col].dropna().tolist()) if arm and (per.arm == arm).any() else "-")  # noqa
        out.append(f"| {n} | {cell(fr, 'final')} | {cell(ru, 'final')} | {cell(fr, 'greedy')} | {cell(ru, 'greedy')} |")
    return "\n".join(out) + "\n"


def tables(per, s, dp, dqn_diag, ppo_diag, crows, srows, ctab, tr, root):
    md = ["# Experiment 13 - Output Reuse against the learning rate of the head\n", NOTE_SIM,
          "Design, predictions and criteria: docs/EXPERIMENT-13-OR-VS-LR.md (written before any cell ran). No "
          "multiple-comparison correction. Equivalence = P(a>b) in [0.35, 0.65] in both metrics and overlapping 95 % CIs "
          "of both IQMs. 'Strong' = P >= 0.95 in both metrics.\n",
          "## Cells per arm\n", "| agent | arms | cells |", "|---|---|---|"]
    for ag in ("T", "H", "PPO"):
        g = per[per.agent == ag]
        md.append(f"| {ag} | {g.arm.nunique()} | {len(g)} |")
    md.append("")
    for ag, pre in (("T", AG["T"]), ("H", AG["H"])):
        if (per.agent == ag).any():
            names = e13.L.arm_names(pre)
            md += [f"## Agent {ag}: IQM per arm" + (" (EMULATOR, simulated)" if ag == "H" else ""),
                   "### Block A\n", summary_table(per, s, names["A_reuse"] + names["A_collapsed"] + names["A_grad"]),
                   "### Block B (fresh, lr of head.weight x k)\n", summary_table(per, s, names["B_headlr"]),
                   "### Block C (global lr x k, exploratory)\n", summary_table(per, s, names["C_global"])]
            rows = [r for r in crows if r["agent"] == ag]
            md += [f"## Agent {ag}: contrasts\n", contrast_md(rows),
                   f"P1 collapsed vs reuse: {status_counts(rows, 'P1')}  ",
                   f"P2 fresh x R vs reuse R: {status_counts(rows, 'P2')}  ",
                   f"P3 gradient x R vs R = 1: {status_counts(rows, 'P3')}\n",
                   "### Performance against k (fresh) and against R (reuse) - P4\n", sweep_md(per, pre),
                   "### Shift of the IQM, reuse R minus fresh x R, with 95 % bootstrap CI\n",
                   md_table(pd.DataFrame([r for r in srows if r["agent"] == ag]),
                            ["R", "shift_final", "shift_final_lo", "shift_final_hi", "shift_greedy", "shift_greedy_lo",
                             "shift_greedy_hi"], {c: "{:.1f}" for c in ("shift_final", "shift_final_lo", "shift_final_hi",
                                                                       "shift_greedy", "shift_greedy_lo", "shift_greedy_hi")}),
                   "### Scale trajectory (median over seeds of ||V_eff||_F and mean |Q|, from the 1 000-step diagnostics)\n",
                   md_table(pd.DataFrame(scale_table(dqn_diag, [f"{pre}_reuse_r{R}" for R in (1,) + RS] +
                                                     [f"{pre}_collapsed_r{R}" for R in RS] +
                                                     [f"{pre}_headlr_x{R}" for R in RS]))), ""]
    if (per.agent == "PPO").any():
        names = [a.name for a in p13.arms()]
        md += ["## Block D: PPO (classical control of the reference study)\n",
               "Metrics: final_perf, greedy_final and best_ma50 (maximum of the 50-episode moving average, the "
               "reference study's metric, which gives 264-382 for its classical control at R = 4-32).\n",
               "| arm | n | IQM final_perf | IQM greedy_final | IQM best_ma50 (95 % CI) |", "|---|---|---|---|---|"]
        for a in names:
            g = per[per.arm == a]
            if len(g):
                md.append(f"| {a} | {len(g)} | {ci_str(g.final.dropna().tolist())} | {ci_str(g.greedy.dropna().tolist())} | "
                          f"{ci_str(g.best_ma50.dropna().tolist())} |")
        rows = [r for r in crows if r["agent"] == "PPO"]
        md += ["", "### Contrasts\n", contrast_md(rows),
               f"P7 no clipping: {status_counts([r for r in rows if r['prediction'] == 'P7 (no clip)'], 'P7')}  ",
               f"P7 with clipping 0.5: {status_counts([r for r in rows if r['prediction'] == 'P7 (clip)'], 'P7')}  ",
               f"P8 modified rule: {status_counts([r for r in rows if r['prediction'] == 'P8'], 'P8')}\n",
               "### Clipping diagnostic (fraction of the updates in which the clipping fired, mean over seeds; "
               "median factor min(1, 0.5/(norm + 1e-6)))\n", md_table(ctab),
               "### Scale trajectory of the actor / critic V (median over seeds)\n"]
        if len(ppo_diag):
            sc = []
            for arm in names:
                g = ppo_diag[ppo_diag.arm == arm]
                if g.empty:
                    continue
                row = {"arm": arm}
                for st in (20_000, 60_000, 100_000):
                    h = g[(g.step >= st) & (g.step < st + 1_000)]
                    row[f"V_actor@{st // 1000}k"] = h.v_eff_fro_actor.median() if len(h) else np.nan
                    row[f"V_critic@{st // 1000}k"] = h.v_eff_fro_critic.median() if len(h) else np.nan
                sc.append(row)
            md.append(md_table(pd.DataFrame(sc)))
    md += ["## Float64 traces (P1, P5, P7, P8)\n",
           "first_diff_step = first environment step at which the action taken differs between the two runs (null = never "
           "in 100 000 steps); max_param_diff = max over the traced steps up to that point of the maximum difference of the "
           "effective parameters (V = sum_r W_r).\n",
           md_table(pd.DataFrame(tr), [c for c in ("key", "first_diff_step", "max_param_diff_to_first_diff",
                                                   "max_param_diff_final", "n_steps", "clip_fraction_a", "wall_s")
                                       if tr and c in tr[0]], {"max_param_diff_to_first_diff": "{:.2e}",
                                                               "max_param_diff_final": "{:.2e}"})]
    (root / "tables_exp13.md").write_text("\n".join(md), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root, out = pathlib.Path(args.results_root).resolve(), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per, dq, dp_ = collect(root, out)
    dp = design_params(root)
    s = summary(per, dp, out)
    crows = []
    for pre in AG.values():
        crows += dqn_contrasts(per, pre)
    crows += ppo_contrasts(per)
    pd.DataFrame(crows).to_csv(out / "contrasts_exp13.csv", index=False)
    srows = shifts(per, AG["T"]) + shifts(per, AG["H"])
    pd.DataFrame(srows).to_csv(out / "shift_exp13.csv", index=False)
    ctab = clip_table(per)
    ctab.to_csv(out / "clip_exp13.csv", index=False)
    tr = traces(root)
    pd.DataFrame(tr).to_csv(out / "traces_exp13.csv", index=False)
    tables(per, s, dp, dq, dp_, crows, srows, ctab, tr, out)
    print(f"{len(per)} runs, {len(s)} arms; wrote {out}")


if __name__ == "__main__":
    main()
