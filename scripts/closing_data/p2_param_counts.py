"""Closing data, point 2: measured trainable-parameter counts (build the agent, count requires_grad parameters).

(a) the 24 configurations of the dissection paper transcribed in core/configs.py (paper_skolik_*, paper_salinas_*,
    paper_hsiao_or_*, paper_classical_or_*, *_noent, *_ent);
(b) the FrozenLake SU(2)-emulator arms and the linear ceiling of the binary circuit.

    python scripts/closing_data/p2_param_counts.py --out <dir>
"""
import argparse
import csv
import pathlib

import qrl_dissection  # noqa: F401  (applies compat patches)
from qrl_dissection.core import configs as C
from qrl_dissection.core.capacity import build_agent_for, count_trainable


def row(arm, env_id):
    at, cfg = C.build_arm_config(arm, env_id=env_id)
    agent = build_agent_for(at, cfg, env_id=env_id, is_qnet=True)
    tf = cfg.get("transform_fn")
    tf = tf if isinstance(tf, str) or tf is None else getattr(tf, "__name__", type(tf).__name__)
    return {
        "arm": arm, "agent_type": at, "template": cfg.get("circ_type", ""),
        "n_qubits": cfg.get("n_qubits", ""), "n_layers_q": cfg.get("n_layers_q", ""),
        "R": cfg.get("reuse_repetitions", cfg.get("n_repeats", "")),
        "entangled": cfg.get("ent", ""), "input_transform": tf if tf is not None else "",
        "emb_indices_or_reuse_indices": str(cfg.get("emb_indices", cfg.get("reuse_indices", ""))),
        "net_arch": str(cfg.get("net_arch", "")),
        "trainable_params_measured": count_trainable(agent), "env_id": env_id,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paper = [k for k in C.PAPER_ARM_TO_RESULTS]
    rows_a = [row(k, "CartPole-v1") for k in paper]
    su2_frozen = [k for k, (t, _) in C.ARMS.items() if t == "su2" and k.startswith("su2_frozen")]
    rows_b = [row(k, C.FROZEN_SCALAR_ID) for k in su2_frozen]
    rows_b += [row("frozen_binary_4q_fourier_ceiling", C.FROZEN_SCALAR_ID)]
    for name, rows in (("p2a_paper_configs_params.csv", rows_a), ("p2b_frozenlake_su2_and_linear_ceiling_params.csv", rows_b)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(name, len(rows))
    for r in rows_a + rows_b:
        print(r["arm"], r["trainable_params_measured"], r["n_qubits"], r["n_layers_q"], r["R"], r["entangled"])


if __name__ == "__main__":
    main()
