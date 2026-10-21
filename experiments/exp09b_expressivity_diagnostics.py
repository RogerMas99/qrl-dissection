"""
Experiment 09b - expressivity and gradient diagnostics, with no reinforcement learning.

EXPLORATORY EXTENSION - NOT PART OF THE THESIS CORE (docs/EXPERIMENT-09.md).

Question
--------
On FrozenLake, a classical additive Fourier ceiling reaches ~0.95 success at L >= 10 while
the circuit with the same embedding and depth stays at zero (FIX-13). Two different
explanations are possible and this script separates them without any RL:

  (a) EXPRESSIVITY: the model cannot represent the optimal action values Q*.
  (b) OPTIMISATION / RL: it can represent them, but DQN does not find them.

Part 1 - supervised fit of Q*
  Q* of the deterministic 4x4 FrozenLake is computed exactly by value iteration
  (gamma = 0.99). Every model is the same Q-network class the RL runs use (built with
  `build_arm_config` / `build_agent_for`), trained by full-batch Adam with MSE on the 10
  non-terminal states. Reported: final MSE, the share of non-terminal states whose
  argmax is an optimal action, and whether the greedy policy of the fitted network
  reaches the goal from the start. If a circuit fits Q* well, its failure under DQN is not
  an expressivity limit.

Part 2 - gradient variance at initialisation
  The variance of d<Z_0>/d theta over random initialisations, for the unentangled
  (SU(2)) and entangled (statevector) layer, as a function of qubits and depth. The
  evidence that docs/EXPERIMENT-04.md asks for before anyone says "barren plateau".

Usage
-----
    python experiments/exp09b_expressivity_diagnostics.py --out <dir>
    python experiments/exp09b_expressivity_diagnostics.py --out <dir> --quick
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch

from qrl_dissection.core.capacity import build_agent_for, count_trainable
from qrl_dissection.core.configs import build_arm_config
from qrl_dissection.core.obs_adapters import FROZEN_ONEHOT_ID, FROZEN_SCALAR_ID

GAMMA = 0.99


# ----------------------------------------------------------------------------- Q*
def optimal_q():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    P, desc = env.unwrapped.P, env.unwrapped.desc.astype(str).reshape(-1)
    n_s, n_a = 16, 4
    q = np.zeros((n_s, n_a))
    v = np.zeros(n_s)
    for _ in range(2000):
        for s in range(n_s):
            for a in range(n_a):
                q[s, a] = sum(pr * (r + (0.0 if d else GAMMA * v[s2])) for pr, s2, r, d in P[s][a])
        v_new = q.max(1)
        if np.abs(v_new - v).max() < 1e-13:
            v = v_new
            break
        v = v_new
    terminal = np.array([c in ("H", "G") for c in desc])
    env.close()
    return q, terminal, P


def greedy_reaches_goal(qfn, P, terminal) -> bool:
    s = 0
    for _ in range(100):
        a = int(np.argmax(qfn(s)))
        _, s2, r, done = P[s][a][0]
        if done:
            return r > 0
        s = s2
    return False


# ------------------------------------------------------------------------ part 1
def fit_arm(arm: str, seed: int, steps: int, lr: float, q_star, terminal, P):
    onehot = arm == "frozen_onehot_mlp"
    env_id = FROZEN_ONEHOT_ID if onehot else FROZEN_SCALAR_ID
    torch.manual_seed(seed)
    agent_type, cfg = build_arm_config(arm, env_id=env_id)
    net = build_agent_for(agent_type, cfg, env_id=env_id, is_qnet=True)
    n_params = count_trainable(net)
    states = torch.arange(16)
    x_all = torch.eye(16) if onehot else states.float().unsqueeze(1)
    keep = torch.as_tensor(~terminal)
    x, y = x_all[keep], torch.as_tensor(q_star, dtype=torch.float32)[keep]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_all = net(x_all).numpy()
    mse = float(((pred_all[~terminal] - q_star[~terminal]) ** 2).mean())
    best = q_star.max(1, keepdims=True)
    optimal = np.isclose(q_star, best, atol=1e-9)
    acc = float(np.mean([optimal[s, int(np.argmax(pred_all[s]))] for s in np.where(~terminal)[0]]))
    reach = greedy_reaches_goal(lambda s: pred_all[s], P, terminal)
    return dict(arm=arm, seed=seed, lr=lr, params=n_params, mse=mse, acc=acc, reach=bool(reach))


def part1(out: pathlib.Path, quick: bool):
    q_star, terminal, P = optimal_q()
    steps = 1500 if quick else 4000
    seeds = [1, 2, 3] if quick else [1, 2, 3, 4, 5]
    arms = (
        [f"su2_frozen_scalar_1q_L{L}" for L in (1, 5, 10, 15)]
        + [f"frozen_scalar_1q_fourier_ceiling_L{L}" for L in (5, 10, 15)]
        + [f"su2_frozen_binary_4q_L{L}" for L in (1, 2, 3, 5)]
        + [f"sv_frozen_binary_4q_L{L}" for L in (1, 2, 3, 5)]
        + ["frozen_binary_4q_fourier_ceiling", "frozen_matched_scalar",
           "frozen_scalar_mlp_large", "frozen_onehot_mlp"]
    )
    rows: List[Dict] = []
    for arm in arms:
        for lr in (1e-2, 5e-2):
            t0 = time.time()
            for sd in seeds:
                rows.append(fit_arm(arm, sd, steps, lr, q_star, terminal, P))
            print(f"  fitted {arm:40s} lr={lr:g} ({time.time() - t0:.0f}s)", flush=True)
    # real PennyLane entangled circuit, 3 seeds, one lr: cross-check of the statevector layer
    if not quick:
        for arm in ("frozen_binary_4q_L2", "frozen_binary_4q_L5"):
            t0 = time.time()
            for sd in (1, 2, 3):
                rows.append(fit_arm(arm, sd, 1500, 1e-2, q_star, terminal, P))
            print(f"  fitted (real PennyLane) {arm} ({time.time() - t0:.0f}s)", flush=True)
    (out / "part1_rows.json").write_text(json.dumps(rows, indent=1))
    return rows


def summarise_part1(rows) -> str:
    out = ["| arm | params | lr | n | median MSE | argmax optimal (mean) | greedy reaches goal |",
           "|---|---|---|---|---|---|---|"]
    keys = []
    for r in rows:
        k = (r["arm"], r["lr"])
        if k not in keys:
            keys.append(k)
    for arm, lr in keys:
        rs = [r for r in rows if r["arm"] == arm and r["lr"] == lr]
        out.append(f"| `{arm}` | {rs[0]['params']} | {lr:g} | {len(rs)} | "
                   f"{np.median([r['mse'] for r in rs]):.2e} | {np.mean([r['acc'] for r in rs]):.2f} | "
                   f"{sum(r['reach'] for r in rs)} / {len(rs)} |")
    return "\n".join(out)


# ------------------------------------------------------------------------ part 2
def part2(out: pathlib.Path, quick: bool):
    from qrl_dissection.core.statevector import SVSkolikLayer
    from qrl_dissection.core.su2_emulator import SU2SkolikEmulator

    n_inits = 60 if quick else 200
    qubits = (2, 4, 6) if quick else (2, 4, 6, 8, 10)
    depths = (1, 2, 5) if quick else (1, 2, 5, 10, 15)
    rows = []
    for nq in qubits:
        for L in depths:
            for kind in ("unentangled (SU2)", "entangled (statevector)"):
                if kind.startswith("entangled") and L == 1:
                    continue                      # FIX-07: identical to unentangled at L = 1
                g0, gnorm = [], []
                for sd in range(n_inits):
                    torch.manual_seed(1000 + sd)
                    layer = (SVSkolikLayer(nq, L, ent=True) if kind.startswith("entangled")
                             else SU2SkolikEmulator(nq, L))
                    x = torch.rand(16, nq) * np.pi
                    layer(x)[:, 0].mean().backward()
                    g = layer.weights.grad
                    g0.append(float(g[0, 0, 0]))
                    gnorm.append(float(g.norm()))
                rows.append(dict(qubits=nq, depth=L, kind=kind, var_g0=float(np.var(g0)),
                                 mean_gnorm=float(np.mean(gnorm))))
        print(f"  gradients done for {nq} qubits", flush=True)
    (out / "part2_rows.json").write_text(json.dumps(rows, indent=1))
    return rows


def summarise_part2(rows) -> str:
    out = ["| qubits | depth | kind | Var[dC/dtheta_0] | mean ||grad|| |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['qubits']} | {r['depth']} | {r['kind']} | {r['var_g0']:.2e} | {r['mean_gnorm']:.3f} |")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    print("part 1: supervised fit of Q*", flush=True)
    r1 = part1(out, args.quick)
    print("part 2: gradient variance at initialisation", flush=True)
    r2 = part2(out, args.quick)
    text = ("## Part 1 - supervised fit of the optimal action values\n\n" + summarise_part1(r1)
            + "\n\n## Part 2 - gradient variance at initialisation\n\n" + summarise_part2(r2) + "\n")
    (out / "diagnostics.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
