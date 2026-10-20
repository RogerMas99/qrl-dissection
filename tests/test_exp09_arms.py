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
