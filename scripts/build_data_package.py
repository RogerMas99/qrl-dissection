"""Build the minimal data package (manifests + per-run CSV, no TensorBoard, no traces, no weights) from <results-root>.

Two archives, so that the simulator results (docs/SIMULATED-RESULTS-POLICY.md) can be published or withheld independently:
  <out>/qrl-dissection-data-core.zip         real PennyLane circuit and classical arms (every manifest whose agent_type is not su2/sv)
  <out>/qrl-dissection-data-simulators.zip   runs of the SU(2) emulator and the batched statevector simulator (agent_type su2 / sv)
Layout inside each archive mirrors results/:  <experiment>/<run>.manifest.json  and  <experiment>/runs/<run>*.csv
(episodes, greedy evaluation, diagnostics). Excluded: *.npy traces, *.pt weights, TensorBoard event files, *.bak, _probe and
_quarantine_* folders and the OtherDriveAccounts copies (already merged into results/).
Also written: MANIFEST_INDEX.csv inside each archive (one row per run: experiment, run name, agent_type, class, seed, files, sha256 of the
manifest). Legacy manifests without agent_type are classed by their run name (hybrid_* = real circuit).

    python scripts/build_data_package.py --results-root <results> --out <dir>
"""
import argparse
import csv
import hashlib
import json
import pathlib
import zipfile

SIM = {"su2", "sv"}
EXTRA_JSON = {"design_params.json", "reused_seeds.json"}


def agent_type(m, name):
    at = m.get("agent_type") or (m.get("spec") or {}).get("agent_type") or ""
    if not at and str(name).startswith("hybrid"):
        at = "hybrid"
    return at


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root, out = pathlib.Path(a.results_root), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zips = {k: zipfile.ZipFile(out / f"qrl-dissection-data-{k}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) for k in ("core", "simulators")}
    idx = {"core": [], "simulators": []}
    skipped = []
    for exp in sorted(p_ for p_ in root.iterdir() if p_.is_dir()):
        if exp.name.startswith("_") or "quarantine" in exp.name.lower():
            continue
        if exp.name == "exp13_traces":
            for f in sorted(exp.glob("*.json")):
                zips["core"].write(f, f"{exp.name}/{f.name}")        # float64 trace summaries (classical and emulator checks, small)
            continue
        for mp in sorted(exp.glob("*.manifest.json")):                # *.bak and sub-folders (_probe, _quarantine_*) are not matched
            m = json.loads(mp.read_text())
            o = m.get("outcome") or {}
            name = o.get("run_name") or m.get("run_name") or m.get("name") or mp.name.replace(".manifest.json", "")
            at = agent_type(m, name)
            grp = "simulators" if at in SIM else "core"
            z = zips[grp]
            z.write(mp, f"{exp.name}/{mp.name}")
            files = [mp.name]
            names = [v for src in (o, o.get("extra") or {}) for k, v in src.items() if k.endswith("_csv") and isinstance(v, str)]
            names = list(dict.fromkeys(names))     # episodes, eval, diagnostics (DQN: outcome.extra.diag_csv; PPO: outcome.diag_csv)
            for n in names:
                if not n:
                    continue
                base = pathlib.Path(str(n).replace("\\", "/")).name
                src = exp / "runs" / base
                if src.exists():
                    z.write(src, f"{exp.name}/runs/{base}")
                    files.append(f"runs/{base}")
                else:
                    skipped.append((exp.name, mp.name, base))
            idx[grp].append([exp.name, name, at, (m.get("spec") or {}).get("seed", o.get("seed", "")), ";".join(files),
                             hashlib.sha256(mp.read_bytes()).hexdigest()])
        for f in sorted(exp.iterdir()):
            if f.is_file() and f.name in EXTRA_JSON:
                zips["simulators" if exp.name in ("exp09_dqn_emulator_sweep", "exp10_emulator_x100") else "core"].write(f, f"{exp.name}/{f.name}")
            if f.is_file() and f.name == "NOT_FOR_THESIS.md":
                zips["simulators"].write(f, f"{exp.name}/{f.name}")
    for k, z in zips.items():
        rows = idx[k]
        tmp = out / f"_index_{k}.csv"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["experiment", "run_name", "agent_type", "seed", "files", "manifest_sha256"])
            w.writerows(rows)
        z.write(tmp, "MANIFEST_INDEX.csv")
        z.close()
        tmp.unlink()
        print(k, len(rows), "runs;", round((out / f"qrl-dissection-data-{k}.zip").stat().st_size / 2**20, 1), "MiB")
    if skipped:
        print("referenced CSV not found:", len(skipped), skipped[:5])


if __name__ == "__main__":
    main()
