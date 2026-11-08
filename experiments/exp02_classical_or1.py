"""
Added arm (outside the registered plan): classical reference control at R = 1 under DQN.

Completes the R axis of the exp02 classical control (R = 4, 8, 16, 32 already exist) with the
no-reuse point: Linear(3 -> 2), 8 parameters, fed with [x_dot, theta, theta_dot]. Cell name:
classical_OR1__s{seed}.

The vendored simplyqrl applies `reuse_indices` only when n_repeats > 1 (src/simplyqrl/agents.py), so
paper_classical_or_config(1) would give Linear(4 -> 2), 10 parameters. The components [1, 2, 3] are
therefore selected in the ENVIRONMENT (SelectObs / make_selected_env, the mechanism of exp13 PPO, from
qrl_dissection.ppo.lr_ppo) and the agent config is {"n_repeats": 1, "net_arch": []}. simplyqrl is not modified;
SafeDQN only gained an optional env_factory (default None = unchanged behaviour).

Everything else is the classical loop of exp02 (agent_type "classic", 100k steps, fix_autoreset=True,
batch_size=128, buffer_size=10_000, train_frequency=10, greedy evaluation every 10k steps, 5 episodes, same
evaluation seeds). Only this arm is run and the cells go into their own folder: no existing cell is touched and
the default behaviour of exp02 is unchanged.

    python experiments/exp02_classical_or1.py --results-root <results> run          # seeds 1..10
    python experiments/exp02_classical_or1.py --results-root <results> repro        # classical_OR4 seed 1 again
    python experiments/exp02_classical_or1.py --results-root <results> tables --out <dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

import qrl_dissection  # noqa: F401  (applies compat patches)
from qrl_dissection.core import analysis, stats
from qrl_dissection.core.configs import paper_classical_or_config
from qrl_dissection.dqn import GreedyEvalConfig, SafeDQN
from qrl_dissection.dqn.runner import reuse_or_none
from qrl_dissection.ppo.lr_ppo import REUSE_INDICES, make_selected_env

HERE = pathlib.Path(__file__).resolve().parent
OUT_NAME = "exp02_classical_OR1"           # new folder under <results-root>
STORED_NAME = "exp02_dqn_cartpole_output_reuse"
STEPS = 100_000
KW = dict(batch_size=128, buffer_size=10_000, train_frequency=10)   # same as exp02


def load_exp02():
    spec = importlib.util.spec_from_file_location("exp02_mod", HERE / "exp02_dqn_cartpole_output_reuse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def local(p, d):
    if not p:
        return None
    name = pathlib.Path(str(p).replace("\\", "/")).name
    for q in (d / "runs" / name, d / name):
        if q.exists():
            return q
    return None


def metrics_of(manifest_path, d):
    """Same definitions as scripts/build_dossier_v3.py (final = last 10 % of training returns,
    greedy = mean of the last 3 greedy checkpoints, best/final of the 50-episode moving average)."""
    m = json.loads(pathlib.Path(manifest_path).read_text())
    oc = m["outcome"]
    sp = m.get("spec") or dict(seed=m["seed"], total_timesteps=m["total_timesteps"], fix_autoreset=m["fix_autoreset"], tag="")
    rew, steps = analysis.load_episodes(local(oc["episodes_csv"], d))
    ma50 = analysis.moving_average(rew, 50)
    es, sc = analysis.load_eval(local(oc["eval_csv"], d))
    rn = oc.get("run_name") or m["name"]
    arm = re.sub(r"__fix01(on|off)$", "", re.sub(r"__s\d+(__\w+)?$", "", rn))
    row = dict(exp="exp02", arm=arm, fix="on" if sp["fix_autoreset"] else "off", tag=sp.get("tag", ""),
               steps=sp["total_timesteps"], seed=sp["seed"], final=stats.final_performance(rew, 0.1),
               greedy=stats.greedy_final(sc), best_ma50=float(np.nanmax(ma50)), final_ma50=float(ma50[-1]),
               wall_h=(oc.get("wall_seconds") or 0) / 3600.0, phantom=(oc.get("probe") or {}).get("frac_poison"))
    ma = np.array([rew[max(0, i - 49): i + 1].mean() for i in range(len(rew))])
    tr = []
    for edge in range(1000, int(sp["total_timesteps"]) + 1, 1000):
        idx = np.searchsorted(steps, edge, side="right") - 1
        if idx >= 0:
            tr.append(("exp02", arm, row["fix"], row["tag"], row["seed"], edge, round(float(ma[idx]), 3)))
    ev = [("exp02", arm, row["fix"], row["tag"], row["seed"], int(s_), round(float(v_), 3)) for s_, v_ in zip(es, sc)]
    return row, tr, ev, rew, sc


OR1_CFG = {"n_repeats": 1, "net_arch": []}
OR1_NOTE = ("Added arm outside the registered plan. The components [1, 2, 3] (x_dot, theta, theta_dot) are selected IN THE "
            "ENVIRONMENT (SelectObs wrapper, make_selected_env as in exp13 PPO), not by the agent: the vendored simplyqrl "
            "applies reuse_indices only when n_repeats > 1, so paper_classical_or_config(1) would give Linear(4->2). "
            "Agent: Linear(3->2), 8 parameters. Greedy evaluation uses the same selecting wrapper.")


def run_one_or1(outdir, seed):
    """Same flow as exp02.run_one (reuse guard, SafeDQN, manifest layout), with the selecting environment."""
    name = f"classical_OR1__s{seed}"
    manifest = outdir / f"{name}.manifest.json"
    hit = reuse_or_none(manifest, {"seed": seed, "total_timesteps": STEPS, "dqn_kwargs": KW})
    if hit is not None:
        print(f"[skip] {name}")
        return hit
    print(f"[run ] {name}", flush=True)
    factory = make_selected_env("CartPole-v1", REUSE_INDICES)
    runner = SafeDQN(agent_type="classic", agent_config=OR1_CFG, run_name=name, seed=seed, fix_autoreset=True,
                     eval_cfg=GreedyEvalConfig(every_steps=10_000, env_factory=factory), outdir=outdir,
                     env_factory=factory, **KW)
    out = runner.train(STEPS, progress_bar=False)
    m = {"name": name, "agent_type": "classic", "seed": seed, "fix_autoreset": True, "outcome": out.__dict__,
         "config": {k: str(v) for k, v in OR1_CFG.items()}, "total_timesteps": STEPS, "dqn_kwargs": KW,
         "env_id": "CartPole-v1", "obs_selection": {"where": "environment (SelectObs)", "indices": list(REUSE_INDICES)},
         "note": OR1_NOTE}
    manifest.write_text(json.dumps(m, indent=2, default=str))
    print(f"       ok {out.wall_seconds}s  phantoms {100*out.probe['frac_poison']:.2f}%")
    return m


def cmd_run(a):
    outdir = pathlib.Path(a.results_root) / OUT_NAME
    for seed in a.seeds:
        run_one_or1(outdir, seed)


def cmd_check(a):
    """Before launching: (a) parameter count of the agent actually built, (b) what the network receives."""
    import os
    import tempfile

    cwd = os.getcwd()
    td = tempfile.TemporaryDirectory()
    os.chdir(td.name)           # upstream writes ./runs/<run_name> relative to the cwd
    try:
        return _check_inner()
    finally:
        os.chdir(cwd)
        td.cleanup()


def _check_inner():
    import gymnasium as gym
    from simplyqrl.dqn import DQN
    from qrl_dissection.core.capacity import count_trainable
    factory = make_selected_env("CartPole-v1", REUSE_INDICES)
    raw = gym.make("CartPole-v1")
    res = {}
    wrapped = factory()
    o_w, _ = wrapped.reset(seed=123)
    o_r, _ = raw.reset(seed=123)
    same = True
    for k in range(50):
        same &= bool(np.array_equal(o_w, o_r[1:]))
        o_w, _, t1, tr1, _ = wrapped.step(k % 2)
        o_r, _, t2, tr2, _ = raw.step(k % 2)
        if t1 or tr1:
            break
    res["wrapper_obs_equals_raw_1_3_over_an_episode"] = same
    dqn = DQN(env=factory, agent_type="classic", agent_config=OR1_CFG, run_name="check_or1", seed=1,
              learning_starts=50, **KW)
    res["single_observation_space_shape"] = list(dqn.envs.single_observation_space.shape)
    res["dqn_first_obs_shape"] = list(np.asarray(dqn.obs).shape)
    r0, _ = gym.make("CartPole-v1").reset(seed=1)
    res["dqn_first_obs"] = np.asarray(dqn.obs)[0].tolist()
    res["raw_first_obs_same_seed"] = r0.tolist()
    res["first_obs_equals_raw_1_3"] = bool(np.allclose(np.asarray(dqn.obs)[0], r0[1:]))
    res["trainable_params"] = count_trainable(dqn.q_network)
    res["layers"] = [f"{type(l).__name__}{tuple(l.weight.shape) if hasattr(l, 'weight') else ''}" for l in dqn.q_network.network]
    seen = set()
    dqn.q_network.register_forward_pre_hook(lambda m, args: seen.add(tuple(args[0].shape[1:])))
    dqn.train(total_timesteps=400, progress_bar=False)
    res["network_input_shapes_seen_in_training"] = [list(s) for s in sorted(seen)]
    res["greedy_eval_obs_shape"] = list(np.asarray(GreedyEvalConfig(env_factory=factory).env_factory().reset(seed=1)[0]).shape)
    print(json.dumps(res, indent=2))
    ok = (res["trainable_params"] == 8 and res["single_observation_space_shape"] == [3]
          and res["wrapper_obs_equals_raw_1_3_over_an_episode"] and res["first_obs_equals_raw_1_3"]
          and res["network_input_shapes_seen_in_training"] == [[3]] and res["greedy_eval_obs_shape"] == [3])
    print("CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1


def cmd_repro(a):
    """Re-run classical_OR4 seed 1 into a separate folder and compare with the stored cell (read only)."""
    exp02 = load_exp02()
    root = pathlib.Path(a.results_root)
    outdir = root / OUT_NAME / "_repro_classical_OR4"
    exp02.run_one(outdir, "classical_OR4__s1", "classic", paper_classical_or_config(4), 1, True, STEPS, KW)
    new, *_n = metrics_of(outdir / "classical_OR4__s1.manifest.json", outdir)
    old_d = root / STORED_NAME
    old, *_o = metrics_of(old_d / "classical_OR4__s1.manifest.json", old_d)
    rn, sn = _n[2], _n[3]
    ro, so = _o[2], _o[3]
    res = {
        "stored": {k: old[k] for k in ("final", "greedy", "best_ma50", "final_ma50")},
        "rerun": {k: new[k] for k in ("final", "greedy", "best_ma50", "final_ma50")},
        "episodes_identical": bool(len(rn) == len(ro) and np.array_equal(rn, ro)),
        "eval_identical": bool(np.array_equal(sn, so)),
        "n_episodes": [int(len(ro)), int(len(rn))],
        "final_greedy_equal": bool(new["final"] == old["final"] and new["greedy"] == old["greedy"]),
    }
    (outdir / "repro_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0 if res["final_greedy_equal"] else 1


def cmd_tables(a):
    root = pathlib.Path(a.results_root)
    d = root / OUT_NAME
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, trs, evs = [], [], []
    for mp in sorted(d.glob("classical_OR1__s*.manifest.json"), key=lambda p: int(re.search(r"__s(\d+)", p.name).group(1))):
        r, tr, ev, *_ = metrics_of(mp, d)
        rows.append(r); trs += tr; evs += ev
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed_classical_OR1.csv", index=False)
    cols = ["exp", "arm", "fix", "tag", "seed", "global_step"]
    pd.DataFrame(trs, columns=cols + ["train_ma50"]).to_csv(out / "curves_classical_OR1_train_ma50.csv", index=False)
    pd.DataFrame(evs, columns=cols + ["greedy_return"]).to_csv(out / "curves_classical_OR1_eval_greedy.csv", index=False)
    f, g = df.final.tolist(), df.greedy.tolist()
    summ = dict(n=len(df), final_iqm=stats.iqm(f), final_ci=stats.bootstrap_ci(f), greedy_iqm=stats.iqm(g),
                greedy_ci=stats.bootstrap_ci(g), wall_h_total=float(df.wall_h.sum()), wall_s_each=(df.wall_h * 3600).round(1).tolist())
    (out / "summary_classical_OR1.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=2, default=float))


def cmd_validate(a):
    """Check that metrics_of() reproduces the rows of the delivered per_seed_metrics.csv for the stored classical_OR4/8 cells."""
    root = pathlib.Path(a.results_root)
    d = root / STORED_NAME
    ref = pd.read_csv(a.reference)
    bad = 0
    n = 0
    for arm in ("classical_OR4", "classical_OR8"):
        for seed in range(1, 11):
            r, *_ = metrics_of(d / f"{arm}__s{seed}.manifest.json", d)
            q = ref[(ref.exp == "exp02") & (ref.arm == arm) & (ref.seed == seed) & (ref.fix == "on")].iloc[0]
            n += 1
            for k in ("final", "greedy", "best_ma50", "final_ma50", "wall_h", "phantom"):
                if not np.isclose(r[k], q[k], rtol=0, atol=1e-9):
                    bad += 1
                    print("DIFF", arm, seed, k, r[k], q[k])
    print(f"validated {n} stored cells, {bad} differing values")
    return 0 if bad == 0 else 1


def cmd_params(a):
    from qrl_dissection.core.capacity import build_agent_for, count_trainable
    for R in (1, 4):
        ag = build_agent_for("classic", paper_classical_or_config(R), env_id="CartPole-v1", is_qnet=True)
        print(f"R={R}: trainable parameters = {count_trainable(ag)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True)
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    sp.add_parser("repro")
    t = sp.add_parser("tables"); t.add_argument("--out", required=True)
    v = sp.add_parser("validate"); v.add_argument("--reference", required=True)
    sp.add_parser("params")
    sp.add_parser("check")
    a = p.parse_args()
    return {"run": cmd_run, "repro": cmd_repro, "tables": cmd_tables, "validate": cmd_validate, "params": cmd_params,
            "check": cmd_check}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
