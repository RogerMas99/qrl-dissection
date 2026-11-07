"""Closing data, point 3: effective DQN/PPO settings read from the manifests (spec.dqn_kwargs / ppo_kwargs), per
experiment folder and environment, with the number of runs that carry each distinct setting. Library defaults are read
from the vendored src/simplyqrl signatures (not typed by hand).

    python scripts/closing_data/p3_effective_hparams.py --results-root <dir> --out <dir>
"""
import argparse
import collections
import csv
import inspect
import json
import pathlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from simplyqrl.dqn import DQN
    from simplyqrl.ppo import PPO
    for name, cls in (("dqn", DQN), ("ppo", PPO)):
        sig = inspect.signature(cls.__init__)
        with open(out / f"p3_library_defaults_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["parameter", "library_default"])
            for k, v in sig.parameters.items():
                if k not in ("self", "env", "env_id", "agent_type", "agent_config", "run_name", "capture_video", "verbose"):
                    w.writerow([k, v.default])

    root = pathlib.Path(a.results_root)
    cnt = collections.Counter()
    for f in root.rglob("*.manifest.json"):
        if any("quarantine" in x.lower() for x in f.relative_to(root).parts):
            continue
        m = json.loads(f.read_text())
        s = m.get("spec") or {}
        kw = s.get("dqn_kwargs") or s.get("ppo_kwargs") or {}
        key = (f.relative_to(root).parts[0], m.get("env_id", ""), "ppo" if "ppo_kwargs" in s else "dqn",
               json.dumps(kw, sort_keys=True), s.get("total_timesteps", ""), s.get("fix_autoreset", ""))
        cnt[key] += 1
    with open(out / "p3_effective_kwargs_by_experiment.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "env_id", "algo", "kwargs_in_spec (rest = library default)", "total_timesteps", "fix_autoreset", "n_runs"])
        for k, n in sorted(cnt.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            w.writerow(list(k) + [n])
    print(len(cnt), "distinct settings")


if __name__ == "__main__":
    main()
