"""The `paper_*` arms must replicate the dissection paper's own configurations.

These arms exist so a cell-for-cell comparison against `data/paper_ppo_*.csv` is
possible. That only works if the configs really are theirs, so the values are
pinned here against the published scripts rather than trusted to stay put.

Equally important: this file asserts that adding them changed NOTHING about the
arms exp01, exp02 and exp03 ran on. That is the whole reason they were added
instead of edited in place.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("pennylane")

from qrl_dissection.core.configs import (  # noqa: E402
    ARMS,
    PAPER_ARM_TO_RESULTS,
    build_arm_config,
)


def test_existing_arms_are_untouched():
    """exp01/02/03 provenance. If this fails, previous results are invalidated."""
    from qrl_dissection.core import capacity

    expected = {"paper_linear": 26, "hybrid_fig4": 126,
                "oversized_mlp": 10934, "matched_classical": 135}
    for arm, n in expected.items():
        agent_type, cfg = build_arm_config(arm)
        got = capacity.count_trainable(capacity.build_agent_for(agent_type, cfg))
        assert got == n, f"{arm}: {got} params, expected {n} - provenance broken"


def test_hybrid_fig4_is_not_the_papers_config():
    """Pins the divergence rather than letting it be forgotten.

    HYBRID_FIG4 is the library chapter's DQN example: 8 qubits with an
    inference net. The paper's Skolik arm has no inference net and uses 4 qubits
    as its base. Both are legitimate; conflating them is not.
    """
    _t, fig4 = build_arm_config("hybrid_fig4")
    _t, paper = build_arm_config("paper_skolik_8q_L5")
    assert fig4["net_arch"] == [4] and paper.get("net_arch", []) == []
    assert fig4["n_qubits"] == 8 and paper["n_qubits"] == 8
    assert fig4 != paper


@pytest.mark.parametrize("arm", sorted(PAPER_ARM_TO_RESULTS))
def test_every_paper_arm_builds(arm):
    from qrl_dissection.core import capacity

    agent_type, cfg = build_arm_config(arm)
    net = capacity.build_agent_for(agent_type, cfg)
    assert capacity.count_trainable(net) > 0


@pytest.mark.parametrize("arm", sorted(PAPER_ARM_TO_RESULTS))
def test_every_paper_arm_maps_to_real_results(arm):
    from qrl_dissection.core.baselines import DATA_DIR, paper_results_for_arm

    if not (DATA_DIR / "paper_ppo_summary.csv").exists():
        pytest.skip("paper baselines not extracted")
    row = paper_results_for_arm(arm)
    assert row["n_seeds"] == 10


def test_paper_config_values_match_the_published_scripts():
    """Spot-checks against src/experiments/*.py in the companion repository."""
    _t, sk = build_arm_config("paper_skolik_4q_L5")
    assert sk["circ_type"] == "skolik" and sk["n_qubits"] == 4
    assert sk["n_layers_q"] == 5 and sk["ent"] is True

    _t, sa = build_arm_config("paper_salinas_1q_L5")
    assert sa["circ_type"] == "dr" and sa["n_qubits"] == 1
    assert sa["emb_indices"] == [1, 2, 3]

    _t, hs = build_arm_config("paper_hsiao_or_r16")
    assert hs["circ_type"] == "hsiao" and hs["n_qubits"] == 4
    assert hs["reuse_repetitions"] == 16 and hs["ent"] is False
    assert hs["n_layers_q"] == 1


def test_no_vacuous_skolik_l1_entanglement_arm():
    """FIX-07: at depth 1 the ent flag cannot change the output, so shipping an
    L1 entangled/unentangled pair would invite someone to run a cell that cannot
    produce a difference."""
    assert "paper_skolik_4q_L1_noent" not in ARMS


def test_paper_linear_is_the_or_arm_at_r4():
    """Consistency: PAPER_LINEAR is post-pqc-inference.py's classical config with
    n_repeats=4, so it must be identical to the r4 member of the OR sweep."""
    _t, a = build_arm_config("paper_linear")
    _t, b = build_arm_config("paper_classical_or_r4")
    assert a == b
