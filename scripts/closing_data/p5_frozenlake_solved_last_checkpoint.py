"""Closing data, point 5: FrozenLake seeds "solved" with the criterion of the thesis' table 4.4: the greedy policy
reaches the goal at the LAST evaluation checkpoint (100 000 steps). Also shown: the criterion of the exp13 dossier
(greedy_final >= 0.999, i.e. the mean over the last 3 checkpoints is 1), for comparison.

Reads <exp>/runs/<run>_eval.csv (columns global_step, greedy_return) through the manifests (arm, seed, FIX-01 flag, tag).
    python scripts/closing_data/p5_frozenlake_solved_last_checkpoint.py --results-root <dir> --out <csv>
"""
import argparse, csv, json, pathlib
import pandas as pd

GROUPS = [  # (label, experiment folder, arm, tag, fix_autoreset, where)
    ("mlp_bits_w6 (exp13)", "exp13_efg", "mlp_bits_w6", "exp13efg", True),
    ("mlp_bits_w12 (exp13)", "exp13_efg", "mlp_bits_w12", "exp13efg", True),
    ("frozen_binary_4q_L1 (entangled binary circuit, exp04)", "exp04_dqn_frozenlake_embeddings", "frozen_binary_4q_L1", "", True),
    ("frozen_binary_4q_L5 (entangled binary circuit, exp04)", "exp04_dqn_frozenlake_embeddings", "frozen_binary_4q_L5", "", True),
]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--results-root", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); root = pathlib.Path(a.results_root)
    rows = []
    for label, exp, arm, tag, fix in GROUPS:
        for f in sorted((root / exp).glob("*.manifest.json")):
            m = json.loads(f.read_text()); s = m.get("spec", {})
            if s.get("arm") != arm or (s.get("tag") or "") != tag or bool(s.get("fix_autoreset")) != fix or s.get("total_timesteps") != 100000:
                continue
            ev = root / exp / "runs" / pathlib.Path(str(m["outcome"]["eval_csv"]).replace("\\", "/")).name
            e = pd.read_csv(ev)
            last = e.sort_values("global_step").iloc[-1]
            tail3 = e.sort_values("global_step").greedy_return.tail(3)
            rows.append(dict(group=label, arm=arm, seed=s["seed"], last_checkpoint_step=int(last.global_step), last_greedy_return=float(last.greedy_return),
                             solved_last_checkpoint=bool(last.greedy_return >= 0.999), greedy_final_last3=float(tail3.mean()),
                             solved_exp13_dossier_criterion=bool(tail3.mean() >= 0.999), n_eval_rows=len(e)))
    df = pd.DataFrame(rows)
    df.to_csv(a.out.replace(".csv", "_per_seed.csv"), index=False)
    s = df.groupby("group").agg(n_seeds=("seed", "size"), solved_last_checkpoint=("solved_last_checkpoint", "sum"),
                                solved_exp13_dossier_criterion=("solved_exp13_dossier_criterion", "sum"), last_step_min=("last_checkpoint_step", "min"), last_step_max=("last_checkpoint_step", "max"))
    s.to_csv(a.out); print(s.to_string())


if __name__ == "__main__":
    main()
