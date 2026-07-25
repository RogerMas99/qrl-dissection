"""FIX-02 / FIX-03: upstream patches are applied and guarded.

Requires torch + simplyqrl to be installed.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("simplyqrl")

import qrl_dissection  # noqa: E402  (import applies the patches)
from qrl_dissection.core.configs import HYBRID_FIG4, PAPER_LINEAR  # noqa: E402


def test_report_lists_both_patches():
    applied = qrl_dissection.upstream_report()["patches_applied"]
    assert "FIX-02" in applied
    assert "FIX-03" in applied


def test_classic_alias_resolves():
    """The paper's own experiment script passes agent_type='classic', which
    upstream build_agent cannot resolve (returns None)."""
    from qrl_dissection.core.capacity import build_agent_for
    agent = build_agent_for("classic", PAPER_LINEAR, is_qnet=True)
    assert agent is not None


def test_unknown_agent_type_raises_instead_of_returning_none():
    from simplyqrl.agents import build_agent
    with pytest.raises(ValueError):
        build_agent("does-not-exist", (4,), 2, {}, is_qnet=True)


def test_paper_linear_head_is_actually_linear():
    """3 reused indices x 4 repeats -> Linear(12, 2), no hidden layers."""
    from qrl_dissection.core.capacity import build_agent_for, count_trainable
    agent = build_agent_for("classic", PAPER_LINEAR, is_qnet=True)
    n = count_trainable(agent)
    assert n < 100, f"expected a tiny linear head, got {n} parameters"


def test_output_scaling_actually_reaches_the_model():
    """FIX-02: with use_output_scaling the module must contain an OutputScale."""
    from simplyqrl.agents import OutputScale
    from qrl_dissection.core.capacity import build_agent_for
    cfg = dict(HYBRID_FIG4)
    cfg["use_output_scaling"] = True
    agent = build_agent_for("hybrid", cfg, is_qnet=True)
    assert any(isinstance(m, OutputScale) for m in agent.modules()), (
        "OutputScale missing: FIX-02 did not take effect"
    )
