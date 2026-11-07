"""Closing data, point 1: compare the battery report regenerated now with the original (2026-10-21) table kept in
docs/EXPERIMENT-09-BATTERY.md. Output: one row per pair with n_real old/new, d old/new, informative old/new, and the
three verdicts side by side.   python scripts/closing_data/p1_compare_battery.py --new <report.md> --old docs/EXPERIMENT-09-BATTERY.md --out <csv>
"""
import argparse
import csv
import re


def parse_pairs(path, stop_at=None):
    rows, in_t = {}, False
    for line in open(path, encoding="utf-8"):
        if stop_at and line.startswith(stop_at):
            break
        if line.startswith("| kind | env | simulator arm | real arm | n sim / real"):
            in_t = True
            continue
        if in_t and line.startswith("|---"):
            continue
        if in_t and line.startswith("|"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            rows[(c[2], c[3])] = dict(kind=c[0], env=c[1], n=c[4], iqm=c[5], P=c[6], d=c[8], informative=c[9], note=c[10] if len(c) > 10 else "")
        elif in_t and not line.startswith("|"):
            in_t = False
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    old = parse_pairs(a.old, stop_at="## Re-evaluation")
    new = parse_pairs(a.new)
    keys = list(dict.fromkeys(list(old) + list(new)))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "simulator_arm", "real_arm", "n_sim_real_old", "n_sim_real_new", "d_old", "d_new",
                    "informative_old", "informative_new", "informative_changed", "P_sim_gt_real_old", "P_sim_gt_real_new", "note"])
        for k in keys:
            o, n = old.get(k, {}), new.get(k, {})
            ch = "" if not (o and n) else ("changed" if o["informative"] != n["informative"] else "")
            w.writerow([(n or o)["kind"], k[0], k[1], o.get("n", ""), n.get("n", ""), o.get("d", ""), n.get("d", ""),
                        o.get("informative", ""), n.get("informative", ""), ch, o.get("P", ""), n.get("P", ""), (n or o).get("note", "")])
    ch = [k for k in keys if old.get(k) and new.get(k) and old[k]["informative"] != new[k]["informative"]]
    print(len(keys), "pairs; informative flag changed:", ch)


if __name__ == "__main__":
    main()
