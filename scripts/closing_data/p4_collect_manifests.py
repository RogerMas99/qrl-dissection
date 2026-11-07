"""Closing data, point 4: one row per finished run found under <results-root>, from the manifests.

recorded_path_kind = where the run wrote its CSV according to the manifest ("/content/..." = Colab runtime with Drive
mounted; "C:\...Mi unidad..." = this PC with Drive for desktop). It is a hint about the machine, not a hardware record.
Columns: experiment folder, run_name, arm, agent_type, class (real_circuit / emulator / classical), seed, steps,
wall_seconds, python, git_revision, env_id, tag, has_wall, quarantined.
  real_circuit = agent_type 'hybrid' (the PennyLane circuit); emulator = 'su2' or 'sv'; classical = everything else.

    python scripts/closing_data/p4_collect_manifests.py --results-root <dir> --out <csv>
"""
import argparse
import csv
import json
import pathlib

EMU = {"su2", "sv"}


def klass(agent_type):
    a = str(agent_type).lower()
    return "emulator" if a in EMU else ("real_circuit" if a == "hybrid" else "classical")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = pathlib.Path(a.results_root)
    rows, bad = [], []
    for f in sorted(root.rglob("*.manifest.json")):
        rel = f.relative_to(root).parts
        try:
            m = json.loads(f.read_text())
        except Exception as e:                       # unreadable manifest: report, do not guess
            bad.append((str(f), repr(e)))
            continue
        o = m.get("outcome") or {}
        s = m.get("spec") or {}
        path = str(o.get("episodes_csv", ""))
        where = ("colab_path" if path.startswith("/content/") else
                 "windows_path" if (":\\" in path or "Mi unidad" in path) else "unknown_path" if path else "no_path")
        w = o.get("wall_seconds")
        at = m.get("agent_type", s.get("agent_type", ""))
        inferred = False
        if not at:                                   # 76 old manifests (exp02/03/03b) carry no agent_type
            nm = str(m.get("run_name", m.get("name", "")))
            if nm.startswith("hybrid"):              # every one of them is named hybrid_* -> the PennyLane circuit
                at, inferred = "hybrid", True
        rows.append({
            "experiment": rel[0], "subpath": "/".join(rel[1:-1]), "run_name": m.get("run_name", m.get("name", "")),
            "arm": s.get("arm", ""), "agent_type": at, "agent_type_inferred_from_run_name": inferred, "class": klass(at),
            "seed": s.get("seed", o.get("seed", "")), "steps": o.get("total_timesteps", s.get("total_timesteps", "")),
            "wall_seconds": w if w is not None else "", "python": m.get("python", ""),
            "git_revision": m.get("git_revision", ""), "env_id": m.get("env_id", ""), "tag": s.get("tag", ""),
            "has_wall": w is not None, "recorded_path_kind": where, "reuse_repetitions": (m.get("agent_config") or m.get("config") or {}).get("reuse_repetitions", ""), "quarantined": any("quarantine" in x.lower() for x in rel),
        })
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    print(len(rows), "manifests;", len(bad), "unreadable")
    for b in bad[:20]:
        print("UNREADABLE", b)


if __name__ == "__main__":
    main()
