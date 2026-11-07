"""[exp13 blocks E, F, G] classical controls that contain Output Reuse (core/lr_efg.py).

What must not silently drift: parameter counts of every arm (162/82/82/162, 135/51/135/128/128, 58/112); that the dup
ceiling really has identical columns for wires i and i+4 (so it is Output Reuse R = 2); that the collapsed model has the
same function as the replicated one at step 0 and, in float64 with the head weight at lr x R, follows it along a real
DQN run; that the parameter groups carry the expected learning rates; that the arms equal the registered reference arms
(cartpole_fourier_ceiling_L5, matched_classical); and that the bits input equals the state's binary representation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

import qrl_dissection  # noqa: E402,F401
from qrl_dissection.core import FROZEN_SCALAR_ID, lr_efg  # noqa: E402
from qrl_dissection.core import lr_equivalence as L1  # noqa: E402
from qrl_dissection.core.capacity import build_agent_for, count_trainable  # noqa: E402
from qrl_dissection.core.configs import ARMS, build_arm_config  # noqa: E402
from qrl_dissection.core.provenance import is_simulated_arm  # noqa: E402
from qrl_dissection.dqn import GreedyEvalConfig, SafeDQN  # noqa: E402

CP = "CartPole-v1"
KW = dict(buffer_size=2000, batch_size=32, learning_starts=200, train_frequency=4, target_network_frequency=100)
BASE_LR = 2.5e-4
PARAMS = {"ceiling_dup": 162, "ceiling_unique": 82, "ceiling_unique_lrx2": 82, "ceiling_unique_lrx2_collapsed": 162,
          "matched_rep4_w7": 135, "matched_norep_w7_lrx4": 51, "matched_norep_w7_lrx4_collapsed": 135,
          "matched_norep_w18": 128, "matched_norep_w18_lrx4": 128, "mlp_bits_w6": 58, "mlp_bits_w12": 112}


@pytest.fixture(autouse=True)
def _default_dtype_restored():
    d = torch.get_default_dtype()
    yield
    torch.set_default_dtype(d)


def _env(arm):
    return FROZEN_SCALAR_ID if arm.startswith("mlp") else CP


def _agent(arm, seed=0, f64=False):
    torch.manual_seed(seed)
    at, cfg = build_arm_config(arm, env_id=_env(arm))
    a = build_agent_for(at, cfg, env_id=_env(arm), is_qnet=True)
    return L1.to_float64_(a) if f64 else a


def _train(arm, tmp_path, seed=1, steps=1500, f64=True, trace=100):
    at, cfg = build_arm_config(arm, env_id=CP)
    cfg = dict(cfg)
    if f64:
        cfg["lr13_float64"] = True
    if trace:
        cfg["lr13_trace_every"] = trace
    holder = []
    pb = L1.make_post_build(cfg)

    def pb2(dqn):
        h = pb(dqn)
        holder.append((dqn, h))
        return h

    SafeDQN(agent_type=at, agent_config=cfg, run_name=f"t_{arm}_{seed}", seed=seed, env_id=CP, fix_autoreset=True,
            eval_cfg=GreedyEvalConfig(enabled=False), outdir=tmp_path, post_build=pb2, **KW).train(steps)
    return holder[0]


# ------------------------------------------------------------------------------------------------------- registry
@pytest.mark.parametrize("arm,n", sorted(PARAMS.items()))
def test_parameter_counts_and_not_simulated(arm, n):
    assert count_trainable(_agent(arm)) == n
    assert not is_simulated_arm(arm)


def test_dup_ceiling_is_the_registered_ceiling():
    ref_t, ref = ARMS["cartpole_fourier_ceiling_L5"]
    t, cfg = ARMS["ceiling_dup"]
    assert t == ref_t
    assert {k: cfg[k] for k in ref} == ref


def test_rep4_w7_is_matched_classical():
    _, ref = build_arm_config("matched_classical", env_id=CP)
    _, cfg = build_arm_config("matched_rep4_w7", env_id=CP)
    for k in ("reuse_indices", "n_repeats", "net_arch", "activation"):
        assert cfg[k] == ref[k], k


def test_collapsed_variants_share_the_model_of_their_replicated_twin():
    assert ARMS["ceiling_unique_lrx2_collapsed"][1]["n_qubits"] == ARMS["ceiling_dup"][1]["n_qubits"] == 8
    assert ARMS["ceiling_unique"][1]["n_qubits"] == 4
    a, b = ARMS["matched_norep_w7_lrx4_collapsed"][1], ARMS["matched_rep4_w7"][1]
    assert all(a[k] == b[k] for k in ("reuse_indices", "n_repeats", "net_arch"))
    assert ARMS["matched_norep_w7_lrx4"][1]["n_repeats"] == 1 and ARMS["matched_norep_w18"][1]["net_arch"] == [18]


# ------------------------------------------------------------- the dup ceiling is Output Reuse R = 2 on 40 columns
def test_dup_ceiling_has_identical_columns_for_wires_i_and_i_plus_4():
    ag = _agent("ceiling_dup")
    x = torch.randn(64, 4) * 2
    feats = []
    z = ag._embed(x)
    k = torch.arange(1, ag.n_layers + 1, dtype=z.dtype)
    ang = z.unsqueeze(-1) * k.view(1, 1, -1)
    f = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).reshape(64, 8, 10)
    assert torch.equal(f[:, :4], f[:, 4:])
    assert ag.head.weight.shape == (2, 80)


def test_dup_output_equals_unique_output_with_summed_weights():
    dup, uni = _agent("ceiling_dup", f64=True), _agent("ceiling_unique", f64=True)
    with torch.no_grad():
        w = dup.head.weight
        uni.head.weight.copy_(w[:, :40] + w[:, 40:])
        uni.head.bias.copy_(dup.head.bias)
    x = torch.randn(32, 4, dtype=torch.float64)
    assert (dup(x) - uni(x)).abs().max() < 1e-12


# ---------------------------------------------------------------- step 0: the collapsed model is the same function
@pytest.mark.parametrize("rep,collapsed,R", [("ceiling_dup", "ceiling_unique_lrx2_collapsed", 2),
                                             ("matched_rep4_w7", "matched_norep_w7_lrx4_collapsed", 4)])
def test_step0_collapse_equals_replicated(rep, collapsed, R):
    a = _agent(rep, seed=3, f64=True)
    b = _agent(collapsed, seed=3, f64=True)                          # same seed -> same weights before the surgery
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    lr_efg.collapse_(b, R)
    assert count_trainable(b) == {2: 82, 4: 51}[R]
    x = torch.randn(64, 4, dtype=torch.float64)
    assert (a(x) - b(x)).abs().max() < 1e-12
    assert torch.allclose(lr_efg.effective_v(a, R), lr_efg.effective_v(b, 1), atol=1e-14)
    assert torch.equal(lr_efg.head_linear(a).bias, lr_efg.head_linear(b).bias)


def test_collapse_keeps_the_torch_rng_stream():
    b = _agent("matched_norep_w7_lrx4_collapsed", seed=3)
    state = torch.get_rng_state().clone()
    lr_efg.collapse_(b, 4)
    assert torch.equal(state, torch.get_rng_state())


# ------------------------------------------------------------------------------------------- parameter groups
@pytest.mark.parametrize("arm,mult", [("ceiling_dup", 1.0), ("ceiling_unique", 1.0), ("ceiling_unique_lrx2", 2.0),
                                      ("ceiling_unique_lrx2_collapsed", 2.0), ("matched_rep4_w7", 1.0),
                                      ("matched_norep_w7_lrx4", 4.0), ("matched_norep_w7_lrx4_collapsed", 4.0),
                                      ("matched_norep_w18", 1.0), ("matched_norep_w18_lrx4", 4.0)])
def test_param_groups(arm, mult, tmp_path):
    dqn, _ = _train(arm, tmp_path, steps=30, f64=False, trace=0)
    hw = lr_efg.head_linear(dqn.q_network).weight
    flat = [p for g in dqn.optimizer.param_groups for p in g["params"]]
    assert len(flat) == len(list(dqn.q_network.parameters())) == len({id(p) for p in flat})
    for g in dqn.optimizer.param_groups:
        holds = any(p is hw for p in g["params"])
        assert g["lr"] == pytest.approx(BASE_LR * (mult if holds else 1.0), rel=1e-12)
        assert g["weight_decay"] == 0
    if mult != 1.0:
        assert [len(g["params"]) for g in dqn.optimizer.param_groups][0] == 1
        assert not any(p is lr_efg.head_linear(dqn.q_network).bias for p in dqn.optimizer.param_groups[0]["params"])


# ---------------------------------------------------------- real runs: the collapsed arm follows the replicated one
@pytest.mark.parametrize("rep,collapsed", [("ceiling_dup", "ceiling_unique_lrx2_collapsed"),
                                           ("matched_rep4_w7", "matched_norep_w7_lrx4_collapsed")])
def test_float64_run_collapsed_follows_replicated(rep, collapsed, tmp_path):
    d_re, h_re = _train(rep, tmp_path / "re", steps=1500)
    d_co, h_co = _train(collapsed, tmp_path / "co", steps=1500)
    assert h_re.actions == h_co.actions
    assert len(h_re.trace_params) == len(h_co.trace_params) >= 10
    diffs = [np.abs(a - b).max() for a, b in zip(h_re.trace_params, h_co.trace_params)]
    assert max(diffs) < 1e-9, max(diffs)


def test_diagnostics_are_recorded_and_the_unique_arm_is_not_the_collapsed_one(tmp_path):
    d_u, h_u = _train("ceiling_unique_lrx2", tmp_path / "u", steps=1500, f64=False, trace=0)
    d_c, h_c = _train("ceiling_unique_lrx2_collapsed", tmp_path / "c", steps=1500, f64=False, trace=0)
    assert [r[0] for r in h_u.rows] == [1000]
    assert h_u.rows[0][1] > 0 and h_u.rows[0][2] > 0
    # fresh init and summed init are different draws: the two arms are not the same run
    assert not math.isclose(h_u.rows[0][1], h_c.rows[0][1], rel_tol=1e-9)


def test_same_seed_same_run(tmp_path):
    _, a = _train("matched_norep_w7_lrx4", tmp_path / "a", steps=700, trace=100)
    _, b = _train("matched_norep_w7_lrx4", tmp_path / "b", steps=700, trace=100)
    assert a.actions == b.actions
    for x, y in zip(a.trace_params, b.trace_params):
        np.testing.assert_array_equal(x, y)


# ------------------------------------------------------------------------------------------------------ block G
def test_state_bits_equal_the_binary_representation_of_the_state():
    from simplyqrl.transformations import FrozenBasisToAngleTransformer
    tr = FrozenBasisToAngleTransformer("4x4")
    s = torch.arange(16, dtype=torch.float32).unsqueeze(1)
    bits = lr_efg.StateBits()(s)
    assert bits.shape == (16, 4)
    assert torch.equal(bits, tr(s) / math.pi)
    assert set(bits.unique().tolist()) == {0.0, 1.0}
    assert torch.equal(lr_efg.StateBits()(torch.tensor([5.0])), torch.tensor([[1.0, 0.0, 1.0, 0.0]]))


def test_bits_mlp_forward_and_shapes():
    for arm, w in (("mlp_bits_w6", 6), ("mlp_bits_w12", 12)):
        a = _agent(arm)
        out = a(torch.arange(16, dtype=torch.float32).unsqueeze(1))
        assert out.shape == (16, 4)
        assert a.network[1].weight.shape == (w, 4) and a.network[3].weight.shape == (4, w)


def test_bits_mlp_runs_under_dqn_on_frozenlake(tmp_path):
    at, cfg = build_arm_config("mlp_bits_w6", env_id=FROZEN_SCALAR_ID)
    SafeDQN(agent_type=at, agent_config=cfg, run_name="t_bits", seed=1, env_id=FROZEN_SCALAR_ID, fix_autoreset=True,
            eval_cfg=GreedyEvalConfig(enabled=False), outdir=tmp_path, post_build=L1.make_post_build(cfg),
            **dict(KW, buffer_size=2000, train_frequency=1)).train(600)


def test_post_build_of_old_arms_is_unchanged_by_the_delegation():
    assert L1.make_post_build(build_arm_config("classical_twin_reuse_r4", env_id=CP)[1]) is not None
    assert L1.make_post_build(build_arm_config("classical_twin_reuse_r4", env_id=CP)[1]).__name__ == "post_build"
    assert L1.make_post_build(build_arm_config("matched_classical", env_id=CP)[1]) is None
