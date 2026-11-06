"""[exp11] The replication control: classical_or_matched_rep_r{R}.

What must not silently drift: that the arm replicates R times (not 4), has a ReLU hidden layer, sees the full
observation, matches the hybrid's MEASURED total; that R = 4 is exactly classical_or_matched_r4 (so it is not
re-run); that the old arm's name prefix cannot swallow the new one; and that the runner uses exp02's protocol.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from qrl_dissection.core.capacity import build_agent_for, count_trainable  # noqa: E402
from qrl_dissection.core.configs import (MATCHED_REUSE_INDICES_FAIR, build_arm_config,  # noqa: E402
                                         hybrid_or_config)

EXP = pathlib.Path(__file__).resolve().parents[1] / "experiments"
spec = importlib.util.spec_from_file_location("exp11_replication_control", EXP / "exp11_replication_control.py")
e11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e11)

CP = "CartPole-v1"
HYBRID_TOTAL = {4: 222, 8: 350, 16: 606, 32: 1118}          # measured, quoted in docs/CORRECTIONS.md#fix-09


def _params(agent_type, cfg):
    return count_trainable(build_agent_for(agent_type, cfg, env_id=CP, is_qnet=True))


@pytest.mark.parametrize("R", [4, 8, 16, 32])
def test_replicates_r_times_with_relu_hidden_layer_and_full_observation(R):
    at, cfg = build_arm_config(f"classical_or_matched_rep_r{R}", env_id=CP)
    assert at == "classic"
    assert cfg["n_repeats"] == R
    assert cfg["reuse_indices"] == MATCHED_REUSE_INDICES_FAIR == [0, 1, 2, 3]
    assert cfg["activation"] is nn.ReLU
    assert len(cfg["net_arch"]) == 1 and cfg["net_arch"][0] >= 1
    assert cfg["_matched_to"] == f"hybrid_or_r{R}"


@pytest.mark.parametrize("R", [4, 8, 16, 32])
def test_budget_is_the_hybrids_measured_total_and_params_are_close(R):
    at, cfg = build_arm_config(f"classical_or_matched_rep_r{R}", env_id=CP)
    assert cfg["_budget"] == cfg["_hybrid_total"] == HYBRID_TOTAL[R]
    assert _params("hybrid", hybrid_or_config(R)) == HYBRID_TOTAL[R]
    measured = _params(at, cfg)
    assert measured == cfg["_spent_params"]
    assert abs(measured - HYBRID_TOTAL[R]) <= 0.07 * HYBRID_TOTAL[R]      # closest width, +/- a few units


def test_widths_are_the_ones_the_design_document_quotes():
    widths = {R: build_arm_config(f"classical_or_matched_rep_r{R}", env_id=CP)[1]["net_arch"][0] for R in HYBRID_TOTAL}
    assert widths == {4: 12, 8: 10, 16: 9, 32: 9}


@pytest.mark.parametrize("R", [8, 16, 32])
def test_replication_leaves_less_budget_for_the_hidden_layer_than_the_fixed_4_control(R):
    old = build_arm_config(f"classical_or_matched_r{R}", env_id=CP)[1]
    new = build_arm_config(f"classical_or_matched_rep_r{R}", env_id=CP)[1]
    assert old["n_repeats"] == 4 and new["n_repeats"] == R
    assert new["net_arch"][0] < old["net_arch"][0]


def test_r4_is_exactly_the_existing_matched_r4_arm():
    keys = ("reuse_indices", "n_repeats", "net_arch", "activation", "_budget")
    old = build_arm_config("classical_or_matched_r4", env_id=CP)[1]
    new = build_arm_config("classical_or_matched_rep_r4", env_id=CP)[1]
    assert {k: old[k] for k in keys} == {k: new[k] for k in keys}
    assert e11.design_record()["r4_identical_to_classical_or_matched_r4"] is True


def test_old_prefix_does_not_swallow_the_new_name_and_garbage_is_rejected():
    for R in (8, 16, 32):
        cfg = build_arm_config(f"classical_or_matched_rep_r{R}", env_id=CP)[1]
        assert cfg["n_repeats"] == R                                     # not 4: it did not go to the old branch
    with pytest.raises(ValueError):
        build_arm_config("classical_or_matched_rep_rXX", env_id=CP)
    with pytest.raises(ValueError):
        build_arm_config("classical_or_matched_repXX", env_id=CP)


def test_runner_protocol_is_exp02s_and_runs_only_r8_16_32():
    assert e11.KWARGS == {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}
    specs = e11.build_specs(100_000)
    assert len(specs) == 3 * 10
    assert {s.arm for s in specs} == {"classical_or_matched_rep_r8", "classical_or_matched_rep_r16",
                                      "classical_or_matched_rep_r32"}
    assert {s.seed for s in specs} == set(range(1, 11))
    assert all(s.fix_autoreset and s.total_timesteps == 100_000 and s.tag == "exp11" for s in specs)
    assert all(s.dqn_kwargs == e11.KWARGS for s in specs)
