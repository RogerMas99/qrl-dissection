"""[exp09] The emulator FrozenLake arms with Output Reuse, and the sweep's cell lists.

Cheap structural checks only - the numbers come from the sweep itself. What must
not silently drift: that every arm builds, that R > 1 really widens the head's
input by R (and R == 1 does not), that no entangled arm sneaks into the
emulator family, and that the three groups have the sizes docs/EXPERIMENT-09.md
quotes.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")

from qrl_dissection.core.capacity import build_agent_for, count_trainable  # noqa: E402
from qrl_dissection.core.configs import (  # noqa: E402
    ARMS,
    EXP09_CONFIG_A_DEPTHS,
    EXP09_CONFIG_B_DEPTHS,
    EXP09_FROZEN_REUSE,
    build_arm_config,
    su2_frozen_arm_name,
)
from qrl_dissection.core.obs_adapters import FROZEN_SCALAR_ID  # noqa: E402

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "exp09_dqn_emulator_sweep.py"


def _script():
    spec = importlib.util.spec_from_file_location("exp09_sweep", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arms():
    out = []
    for R in EXP09_FROZEN_REUSE:
        out += [("scalar_1q", L, R) for L in EXP09_CONFIG_A_DEPTHS]
        out += [("binary_4q", L, R) for L in EXP09_CONFIG_B_DEPTHS]
    return out


@pytest.mark.parametrize("config,L,R", _arms())
def test_every_arm_builds_and_widens_the_head_by_R(config, L, R):
    name = su2_frozen_arm_name(config, L, R)
    assert name in ARMS
    agent_type, cfg = build_arm_config(name, env_id=FROZEN_SCALAR_ID)
    assert agent_type == "su2"
    assert not cfg.get("ent", False), "the emulator family is unentangled by construction"
    agent = build_agent_for(agent_type, cfg, env_id=FROZEN_SCALAR_ID, is_qnet=True)
    n_qubits = 1 if config == "scalar_1q" else 4
    head = [m for m in agent.network if isinstance(m, torch.nn.Linear)][-1]
    assert head.in_features == n_qubits * R
    assert count_trainable(agent) == 2 * L * n_qubits + (n_qubits * R + 1) * 4


def test_group_sizes_match_the_design_doc():
    mod = _script()
    assert len(mod.build_specs("F", 100_000)) == 12 * 10
    assert len(mod.build_specs("G", 100_000)) == 12 * 10
    assert len(mod.build_specs("I", 100_000)) == 8 * 2 * 3


def test_primary_protocol_is_exp04s_and_sensitivity_uses_separate_seeds():
    mod = _script()
    assert mod.PRIMARY_KWARGS == {"batch_size": 128, "buffer_size": 50_000,
                                  "train_frequency": 1, "learning_starts": 1_000}
    assert not set(mod.SENSITIVITY_SEEDS) & set(mod.PRIMARY_SEEDS)
    tags = [t for t, _ in mod.sensitivity_variants()]
    assert len(set(tags)) == 8 and "p_tf1_ls1k_buf50k" in tags
    # the sensitivity grid contains the primary protocol itself
    assert any(kw == mod.PRIMARY_KWARGS for _, kw in mod.sensitivity_variants())


def test_run_names_are_unique():
    mod = _script()
    for g in ("F", "G", "I"):
        names = [s.run_name for s in mod.build_specs(g, 100_000)]
        assert len(names) == len(set(names))


# --------------------------- wave 2 (statevector simulator and extra axes) -----------
def test_wave2_group_sizes_and_environments():
    mod = _script()
    assert len(mod.build_specs("P", 100_000)) == 12 * 10
    assert len(mod.build_specs("V", 100_000)) == 5 * 10
    assert len(mod.build_specs("K", 100_000)) == (4 + 2 + 2 * 2) * 10
    assert len(mod.build_specs("H", 100_000)) == 3 * (4 + 2) * 5
    assert mod.group_env("V") == mod.group_env("K") == "CartPole-v1"
    assert mod.group_env("P") == mod.group_env("H") == FROZEN_SCALAR_ID
    for g in mod.GROUPS:
        names = [s.run_name for s in mod.build_specs(g, 100_000)]
        assert len(names) == len(set(names)), g


def test_wave2_arms_build_with_the_expected_agent_type_and_entanglement():
    for name, kind, ent in (("sv_frozen_binary_4q_L5", "sv", True),
                            ("sv_cartpole_L5_OR16", "sv", True),
                            ("su2_cartpole_L5_OR8", "su2", False)):
        agent_type, cfg = build_arm_config(
            name, env_id="CartPole-v1" if "cartpole" in name else FROZEN_SCALAR_ID)
        assert agent_type == kind and bool(cfg["ent"]) is ent
    # the replicas have the same parameter count as the real arms they stand in for
    ref = {"sv_cartpole_L5_OR16": 606, "su2_cartpole_L5_OR8": 350, "sv_cartpole_L5": 126}
    for name, n_params in ref.items():
        at, cfg = build_arm_config(name, env_id="CartPole-v1")
        agent = build_agent_for(at, cfg, env_id="CartPole-v1", is_qnet=True)
        assert count_trainable(agent) == n_params, name


def test_exploratory_variants_change_only_what_they_say():
    at, base = build_arm_config("su2_frozen_scalar_1q_L5", env_id=FROZEN_SCALAR_ID)
    for suffix, key in (("head16", "net_arch"), ("init0.1", "q_init_std"), ("tis", "trainable_input_scale")):
        _, cfg = build_arm_config(f"su2_frozen_scalar_1q_L5__{suffix}", env_id=FROZEN_SCALAR_ID)
        assert key in cfg and cfg["n_layers_q"] == base["n_layers_q"] and cfg["n_qubits"] == base["n_qubits"]


# --------------------------- validation battery (docs/EXPERIMENT-09.md section 7) ------
def test_battery_groups_and_parameter_parity_with_the_real_arms():
    mod = _script()
    assert len(mod.build_specs("Q", 100_000)) == 2 * 5 * 10
    assert len(mod.build_specs("R", 100_000)) == 3 * 10
    assert len(mod.build_specs("S", 100_000)) == 8 * 10
    assert len(mod.build_specs("T", 100_000)) == 2 * 10
    for g in "QRS":
        assert mod.group_env(g) == "CartPole-v1"
    assert mod.group_env("T") == FROZEN_SCALAR_ID
    pairs = {"sv_paper_skolik_4q_L1": "paper_skolik_4q_L1", "sv_paper_skolik_4q_L5": "paper_skolik_4q_L5",
             "sv_paper_skolik_8q_L2": "paper_skolik_8q_L2", "su2_paper_skolik_4q_L2_noent": "paper_skolik_4q_L2_noent",
             "sv_cartpole_L2_OR8": "hybrid_DR2_OR8", "sv_cartpole_L1_OR32": "hybrid_DR1_OR32"}
    for sim, real in pairs.items():
        a1, c1 = build_arm_config(sim, env_id="CartPole-v1")
        n_sim = count_trainable(build_agent_for(a1, c1, env_id="CartPole-v1", is_qnet=True))
        a2, c2 = build_arm_config(real, env_id="CartPole-v1")
        n_real = count_trainable(build_agent_for(a2, c2, env_id="CartPole-v1", is_qnet=True))
        assert n_sim == n_real, (sim, n_sim, real, n_real)
