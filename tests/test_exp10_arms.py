"""[exp10] The 100-seed simulator extension: arm registration, twins, protocol and reuse rules.

Structural checks only; the numbers come from the sweep. What must not silently drift: that the
sv ent=False arms are exact twins of the su2 arms (same configuration, same parameter count), that
the protocols equal exp09's, that the cell lists have the sizes docs/EXPERIMENT-10-X100.md quotes,
and that seed reuse looks at the tag (a learning-rate variant must not shadow the plain arm).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

from qrl_dissection.core.capacity import build_agent_for, count_trainable  # noqa: E402
from qrl_dissection.core.configs import ARMS, build_arm_config  # noqa: E402

EXP = pathlib.Path(__file__).resolve().parents[1] / "experiments"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e10 = _load("exp10_emulator_x100")
e09 = _load("exp09_dqn_emulator_sweep")


def _params(arm):
    at, cfg = build_arm_config(arm, env_id="CartPole-v1")
    return count_trainable(build_agent_for(at, cfg, env_id="CartPole-v1", is_qnet=True))


def test_protocols_equal_exp09():
    assert e10.CARTPOLE_KWARGS == e09.CARTPOLE_KWARGS
    assert e10.PRIMARY_KWARGS == e09.PRIMARY_KWARGS


@pytest.mark.parametrize("R", [4, 8, 16, 32])
def test_sv_noent_or_arm_is_the_exact_twin_of_the_su2_arm(R):
    sv, su = f"sv_cartpole_L5_OR{R}_noent", f"su2_cartpole_L5_OR{R}"
    assert ARMS[sv][0] == "sv" and ARMS[su][0] == "su2"
    assert ARMS[sv][1] == ARMS[su][1]
    assert _params(sv) == _params(su)


@pytest.mark.parametrize("L", [1, 2])
def test_depth_noent_sv_arm_is_the_twin_of_the_su2_arm(L):
    sv, su = f"sv_cartpole_L{L}_noent", f"su2_cartpole_L{L}"
    assert ARMS[sv][0] == "sv" and ARMS[su][0] == "su2"
    assert ARMS[sv][1] == ARMS[su][1]
    assert ARMS[sv][1]["ent"] is False
    assert _params(sv) == _params(su)


def test_group_sizes_without_reuse():
    n = {g: len(e10.build_specs(g, 100_000, root=None)[0]) for g in e10.GROUPS}
    assert n == {"P1": 15 * 100, "P2": 4 * 100, "P3": 4 * 100, "P4": 2 * 100}


def test_specs_are_tagged_seed_major_and_use_the_right_protocol():
    specs, _ = e10.build_specs("P1", 100_000, root=None)
    assert {s.tag for s in specs} == {"x100"}
    assert all(s.fix_autoreset and s.total_timesteps == 100_000 for s in specs)
    assert all(s.dqn_kwargs == e10.CARTPOLE_KWARGS for s in specs)
    assert [s.seed for s in specs[:15]] == [1] * 15           # seed-major: all arms advance together
    specs4, _ = e10.build_specs("P4", 100_000, root=None)
    assert all(s.dqn_kwargs == e10.PRIMARY_KWARGS for s in specs4)


def test_every_arm_of_every_group_is_registered_as_a_simulator():
    for g in e10.GROUPS:
        for arm, _kw, _env in e10.group_arms(g):
            assert arm in ARMS and ARMS[arm][0] in ("sv", "su2"), arm


def _fake_manifest(folder: pathlib.Path, arm, seed, tag, kwargs, steps=100_000):
    (folder / "runs").mkdir(parents=True, exist_ok=True)
    stem = f"{arm}__fix01on__s{seed}" + (f"__{tag}" if tag else "")
    (folder / "runs" / f"{stem}.csv").write_text("episode_n,ep_reward,global_step\n")
    (folder / f"{stem}.manifest.json").write_text(json.dumps({
        "spec": {"arm": arm, "seed": seed, "fix_autoreset": True, "total_timesteps": steps,
                 "dqn_kwargs": kwargs, "tag": tag},
        "outcome": {"episodes_csv": f"/content/drive/{stem}.csv"}}))


def test_reuse_requires_identical_spec_and_looks_at_the_tag(tmp_path):
    e10._index.clear()
    d = tmp_path / "exp09_dqn_emulator_sweep"
    kw = e10.CARTPOLE_KWARGS
    for s in (1, 2, 3):
        _fake_manifest(d, "sv_cartpole_L5", s, "", kw)
    _fake_manifest(d, "sv_cartpole_L5", 1, "lr0.001", dict(kw, learning_rate=1e-3))   # must not shadow seed 1
    _fake_manifest(d, "sv_cartpole_L5", 4, "", dict(kw, train_frequency=1))           # different spec
    _fake_manifest(d, "sv_cartpole_L5", 5, "", kw, steps=60_000)                      # different steps
    got = e10.reusable_seeds(tmp_path, "sv_cartpole_L5", kw, 100_000)
    assert sorted(got) == [1, 2, 3]
    assert all(v == ("exp09_dqn_emulator_sweep", "") for v in got.values())
    e10._index.clear()
