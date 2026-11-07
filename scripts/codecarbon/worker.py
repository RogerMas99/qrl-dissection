"""Calibration worker: runs cells of ONE kind in a loop (consecutive seeds) until it is killed by the orchestrator.

Kinds (configuration identical to the campaign; only the stopping rule differs):
  real_cartpole  hybrid_fig4 (real PennyLane circuit, 8 qubits, L=5, entangled)   CartPole, exp01 kwargs
  real_frozen    frozen_binary_4q_L5 (real circuit, 4 qubits, entangled)           FrozenLake, exp04 kwargs
  sv_cartpole    hybrid_emu_reuse_r1 (batched statevector simulator, as in exp13 H) CartPole, exp13 kwargs
  su2_cartpole   su2_cartpole_L5 (SU(2) emulator, hybrid without entanglement)      CartPole, exp01 kwargs
  classic_dqn    classical_twin_reuse_r8 (classical twin of exp13 T)                CartPole, exp13 kwargs
  ppo            ppo_reuse_clip_r4 (reference classical control of exp13 D)         CartPole, PAPER_PPO_CONFIG

    python scripts/codecarbon/worker.py --kind sv_cartpole --seed0 9001 --outdir <dir>
Outputs go to <outdir> (never to a campaign folder); tag 'calib'.
"""
import argparse
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


KW_CARTPOLE = {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}
KW_FROZEN = {"batch_size": 128, "buffer_size": 50_000, "train_frequency": 1, "learning_starts": 1_000}
DQN_KINDS = {  # kind: (arm, env, kwargs, eval episodes)
    "real_cartpole": ("hybrid_fig4", "CartPole-v1", KW_CARTPOLE, 5),
    "real_frozen": ("frozen_binary_4q_L5", None, KW_FROZEN, 20),
    "sv_cartpole": ("hybrid_emu_reuse_r1", "CartPole-v1", KW_CARTPOLE, 20),
    "su2_cartpole": ("su2_cartpole_L5", "CartPole-v1", KW_CARTPOLE, 20),
    "classic_dqn": ("classical_twin_reuse_r8", "CartPole-v1", KW_CARTPOLE, 20),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", required=True, choices=list(DQN_KINDS) + ["ppo"])
    p.add_argument("--seed0", type=int, required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--steps", type=int, default=100_000)
    a = p.parse_args()
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    seed = a.seed0
    if a.kind == "ppo":
        e = load("exp13_ppo")
        while True:
            e.run_cell(e.Spec(arm="ppo_reuse_clip_r4", seed=seed, total_timesteps=a.steps, tag="calib"), out)
            seed += 1000
    from qrl_dissection.core import FROZEN_SCALAR_ID
    from qrl_dissection.dqn import GreedyEvalConfig, RunSpec, run_grid
    arm, env, kw, n_eval = DQN_KINDS[a.kind]
    env = env or FROZEN_SCALAR_ID
    if a.kind == "sv_cartpole" or a.kind == "classic_dqn":
        load("exp13_lr_equivalence")          # registers nothing by itself; arms are registered in core/configs.py on import
    while True:
        spec = RunSpec(arm=arm, seed=seed, fix_autoreset=True, total_timesteps=a.steps, dqn_kwargs=dict(kw), tag="calib")
        run_grid([spec], out, env_id=env, eval_cfg=GreedyEvalConfig(env_id=env, every_steps=10_000, n_episodes=n_eval),
                 progress_bar=False, verbose=False)
        seed += 1000


if __name__ == "__main__":
    main()
