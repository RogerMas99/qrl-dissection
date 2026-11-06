"""[exp12] The classical twin of the hybrid: classical_twin_r{R} and classical_twin_unbounded_r{R}.

What must not silently drift: that the head and the Output Reuse of the twin ARE the hybrid's (same classes, same
shapes, same initial weights under one seed), that the bounded extractor stays in [-1, 1] and the unbounded one does
not, that the parameter counts are the hybrid's + 6, that no PennyLane layer is left inside the twin, that the arm
names cannot capture each other or an existing arm, and that the runner uses exp02's protocol.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from torch import nn  # noqa: E402

import qrl_dissection  # noqa: E402,F401  (applies compat.py patches on import)
from qrl_dissection.core.capacity import build_agent_for, count_trainable  # noqa: E402
from qrl_dissection.core.classical_twin import ClassicalTwinAgent  # noqa: E402
from qrl_dissection.core.configs import ARMS, build_arm_config, hybrid_or_config  # noqa: E402
from qrl_dissection.core.provenance import is_simulated_arm  # noqa: E402
from simplyqrl.agents import HybridAgent, OutputReuse  # noqa: E402

EXP = pathlib.Path(__file__).resolve().parents[1] / "experiments"
spec = importlib.util.spec_from_file_location("exp12_classical_twin", EXP / "exp12_classical_twin.py")
e12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e12)

CP = "CartPole-v1"
RS = [1, 4, 8, 16, 32]
HYBRID_TOTAL = {1: 126, 4: 222, 8: 350, 16: 606, 32: 1118}        # measured
TWIN_TOTAL = {1: 132, 4: 228, 8: 356, 16: 612, 32: 1124}          # hybrid + 6
FAMS = ("classical_twin_r", "classical_twin_unbounded_r")


def _twin(fam, R, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    at, cfg = build_arm_config(f"{fam}{R}", env_id=CP)
    return build_agent_for(at, cfg, env_id=CP, is_qnet=True)


def _hybrid(R, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return build_agent_for("hybrid", hybrid_or_config(R), env_id=CP, is_qnet=True)


def _signature(seq):
    return [(type(m).__name__, tuple(tuple(p.shape) for p in m.parameters()),
             getattr(m, "n_repeats", None)) for m in seq]


@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("R", RS)
def test_head_and_output_reuse_are_structurally_the_hybrids(fam, R):
    tw, hy = _twin(fam, R), _hybrid(R)
    assert isinstance(tw, ClassicalTwinAgent) and isinstance(tw, HybridAgent)
    assert _signature(tw.head) == _signature(hy.network[1:])
    names = [type(m).__name__ for m in tw.head]
    assert names == (["OutputReuse"] if R > 1 else []) + ["Linear", "Identity", "Linear"]
    if R > 1:
        assert isinstance(tw.head[0], OutputReuse) and tw.head[0].n_repeats == R
        assert tw.head[1].in_features == 8 * R
    else:
        assert tw.head[0].in_features == 8
    assert (tw.head[-3].out_features, tw.head[-1].out_features) == (4, 2)


@pytest.mark.parametrize("R", [1, 4, 32])
def test_head_starts_with_the_hybrids_weights_under_one_seed(R):
    tw, hy = _twin("classical_twin_r", R, seed=7), _hybrid(R, seed=7)
    for a, b in zip(tw.head.parameters(), hy.network[1:].parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("fam", FAMS)
def test_extractor_shape_and_layers(fam):
    tw = _twin(fam, 4)
    kinds = [type(m).__name__ for m in tw.extractor]
    assert kinds == ["Linear", "ReLU", "Linear"] + (["Tanh"] if fam == "classical_twin_r" else [])
    assert (tw.extractor[0].in_features, tw.extractor[0].out_features) == (4, 6)
    assert (tw.extractor[2].in_features, tw.extractor[2].out_features) == (6, 8)


def test_bounded_extractor_is_in_minus_one_one_and_unbounded_is_not():
    torch.manual_seed(0)
    x = torch.randn(512, 4) * 50.0                                    # far outside CartPole's range on purpose
    bounded, unbounded = _twin("classical_twin_r", 4, seed=0), _twin("classical_twin_unbounded_r", 4, seed=0)
    with torch.no_grad():
        zb, zu = bounded.extractor(x), unbounded.extractor(x)
    assert zb.shape == zu.shape == (512, 8)
    assert float(zb.min()) >= -1.0 and float(zb.max()) <= 1.0
    assert float(zu.abs().max()) > 1.0
    with torch.no_grad():                                             # also on CartPole-like observations
        assert float(bounded.extractor(torch.randn(64, 4)).abs().max()) <= 1.0


@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("R", RS)
def test_parameter_counts_are_the_hybrids_plus_six(fam, R):
    tw, hy = _twin(fam, R), _hybrid(R)
    assert count_trainable(hy) == HYBRID_TOTAL[R]
    assert count_trainable(tw) == TWIN_TOTAL[R] == count_trainable(hy) + 6
    assert count_trainable(tw.extractor) == 86 == (4 * 6 + 6) + (6 * 8 + 8)
    assert count_trainable(tw.head) == count_trainable(hy.network[1:]) == 32 * R + 4 + 10


def test_no_quantum_layer_is_left_and_no_parameter_is_registered_twice():
    tw = _twin("classical_twin_r", 8)
    assert not hasattr(tw, "qnet_q_layer")
    assert not any("q_layer" in n or "weights" == n.split(".")[-1] for n, _ in tw.named_parameters())
    keys = list(tw.state_dict())
    assert len(keys) == len(set(keys)) == 8                            # 4 Linear layers x (weight, bias)
    assert sum(p.numel() for p in tw.parameters()) == count_trainable(tw)


def test_forward_shape_and_gradients_reach_extractor_and_head():
    tw = _twin("classical_twin_r", 4, seed=1)
    out = tw(torch.randn(5, 4))
    assert out.shape == (5, 2)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in tw.parameters())
    assert any(float(p.grad.abs().sum()) > 0 for p in tw.extractor.parameters())


def test_only_dqn_and_only_hybrid_or_configs_are_accepted():
    at, cfg = build_arm_config("classical_twin_r4", env_id=CP)
    with pytest.raises(ValueError, match="is_qnet=True"):
        build_agent_for(at, cfg, env_id=CP, is_qnet=False)
    with pytest.raises(ValueError, match="skolik"):
        ClassicalTwinAgent(4, 2, {"net_arch": [4], "activation": nn.Identity, "n_qubits": 8}, is_qnet=True)


def test_registry_names_flags_and_no_prefix_capture():
    for R in RS:
        at, cfg = build_arm_config(f"classical_twin_r{R}", env_id=CP)
        au, cu = build_arm_config(f"classical_twin_unbounded_r{R}", env_id=CP)
        assert at == au == "classical_twin"
        assert cfg["twin_bounded"] is True and cu["twin_bounded"] is False
        assert cfg["twin_hidden"] == cu["twin_hidden"] == 6
        assert cfg["reuse_repetitions"] == cu["reuse_repetitions"] == R
        # the head is defined by the hybrid_OR{R} config: nothing but the two twin keys differs
        assert {k: v for k, v in cfg.items() if not k.startswith("twin_")} == hybrid_or_config(R)
    with pytest.raises(KeyError):
        build_arm_config("classical_twin_r3", env_id=CP)
    with pytest.raises(KeyError):
        build_arm_config("classical_twin_unbounded_r", env_id=CP)
    # existing arms untouched and not captured by the new names
    assert build_arm_config("classical_or_matched_r8", env_id=CP)[1]["n_repeats"] == 4
    assert build_arm_config("classical_or_matched_rep_r8", env_id=CP)[1]["n_repeats"] == 8
    assert ARMS["hybrid_fig4"][0] == "hybrid" and ARMS["hybrid_or_r8"][0] == "hybrid"


def test_twin_is_not_flagged_as_a_simulator():
    assert not any(is_simulated_arm(f"{f}{R}") for f in FAMS for R in RS)


def test_runner_protocol_is_exp02s_and_has_100_cells():
    assert e12.KWARGS == {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10}
    specs = e12.build_specs(100_000)
    assert len(specs) == 2 * 5 * 10 == 100
    assert {s.arm for s in specs} == {f"{f}{R}" for f in FAMS for R in RS}
    assert {s.seed for s in specs} == set(range(1, 11))
    assert all(s.fix_autoreset and s.total_timesteps == 100_000 and s.tag == "exp12" for s in specs)
    assert all(s.dqn_kwargs == e12.KWARGS for s in specs)
    assert len({s.run_name for s in specs}) == 100


def test_design_record_matches_the_documented_numbers():
    rec = e12.design_record()
    for R in RS:
        r = rec[str(R)]
        assert r["hybrid_total"] == HYBRID_TOTAL[R]
        for fam in ("classical_twin", "classical_twin_unbounded"):
            assert r[fam]["total"] == TWIN_TOTAL[R] and r[fam]["extra_vs_hybrid"] == 6
            assert r[fam]["extractor"] == 86 and r[fam]["head_equals_hybrid_head"] is True
