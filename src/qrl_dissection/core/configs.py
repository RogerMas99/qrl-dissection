"""
Experiment arms.

Three arms, three distinct questions. They are not interchangeable and none
substitutes for another.

    paper_linear      Faithful replication of the paper's classical control
                      (section 3.4A / 3.3): no hidden layers, observation
                      reduced to indices [1,2,3] - cart position discarded, as
                      in Hsiao et al. Its behaviour under DQN is a RESULT, not
                      a configuration to be fixed.

    matched_classical [NEW-02] Same design, but the PQC is replaced by a
                      classical block of equal parameter count. This is the arm
                      that makes "does the circuit contribute anything under
                      DQN?" answerable.

    hybrid            The object of study.

    oversized_mlp     Labelled control only. ~11k parameters, ~90x the PQC
                      budget. It answers "is the environment/algorithm setup
                      sound at all?" and nothing about the circuit. Never
                      compare it to the hybrid arm.
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

# Paper's classical control: post-pqc-inference.py :: config_classical
PAPER_LINEAR: Dict[str, Any] = {
    "reuse_indices": [1, 2, 3],   # cart position discarded, as in the original paper
    "n_repeats": 4,
    "net_arch": [],               # no hidden layers: input > reuse > output
}

# Fig. 4 of the SimplyQRL chapter: examples/cartpole_dqn.py
HYBRID_FIG4: Dict[str, Any] = {
    "circ_type": "skolik",
    "n_qubits": 8,
    "n_layers_q": 5,
    "ent": True,
    "net_arch": [4],
    "activation": nn.Identity,
}

# Labelled control. Upstream default for is_qnet agents (CleanRL's CartPole net).
OVERSIZED_MLP: Dict[str, Any] = {
    "net_arch": [120, 84],
    "activation": nn.ReLU,
}

ARMS = {
    "paper_linear": ("classic", PAPER_LINEAR),
    "hybrid_fig4": ("hybrid", HYBRID_FIG4),
    "oversized_mlp": ("classic", OVERSIZED_MLP),
}


# ---------------------------------------------------------------------------
# Environments. The second axis the study grows along (the first is algorithm).
#
# The paper uses CartPole-v1 only. As experiments extend to Acrobot,
# LunarLander, etc., register them here with the observation preprocessing each
# one needs, so that experiment scripts refer to environments by name and the
# preprocessing stays in one place rather than scattered across scripts.
# ---------------------------------------------------------------------------
ENVIRONMENTS: Dict[str, Dict[str, Any]] = {
    "cartpole": {
        "env_id": "CartPole-v1",
        "obs_dim": 4,
        "n_actions": 2,
        # index set the paper's classical arm keeps (cart position discarded)
        "paper_reuse_indices": [1, 2, 3],
    },
    # "acrobot":     {"env_id": "Acrobot-v1",     "obs_dim": 6, "n_actions": 3, ...},
    # "lunarlander": {"env_id": "LunarLander-v3", "obs_dim": 8, "n_actions": 4, ...},
}


def get_environment(name: str) -> Dict[str, Any]:
    if name not in ENVIRONMENTS:
        raise KeyError(f"unknown environment {name!r}. Known: {sorted(ENVIRONMENTS)}")
    return dict(ENVIRONMENTS[name])


def build_arm_config(
    arm: str,
    hybrid_reference: str = "hybrid_fig4",
    activation: type = nn.ReLU,
    env_id: str = "CartPole-v1",
    **overrides: Any,
):
    """Return (agent_type, agent_config) for an arm name.

    `matched_classical` is computed, not hard-coded: the hidden width is derived
    from the reference hybrid arm's measured PQC parameter budget, so it stays
    correct if the circuit changes.
    """
    from .capacity import match_hidden_width, pqc_parameter_budget

    if arm == "matched_classical":
        ref_type, ref_cfg = ARMS[hybrid_reference]
        if ref_type != "hybrid":
            raise ValueError("hybrid_reference must name a hybrid arm")
        budget = pqc_parameter_budget(ref_cfg, env_id=env_id)["quantum"]
        indices = PAPER_LINEAR["reuse_indices"]
        in_dim = len(indices)
        width, spent = match_hidden_width(budget, in_dim=in_dim, out_dim=2)
        cfg = {
            "reuse_indices": list(indices),
            "n_repeats": PAPER_LINEAR["n_repeats"],
            "net_arch": [width],
            "activation": activation,
            "_matched_to": hybrid_reference,
            "_pqc_budget": budget,
            "_spent_params": spent,
        }
        cfg.update(overrides)
        return "classic", cfg

    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}. Known: {sorted(ARMS) + ['matched_classical']}")
    agent_type, cfg = ARMS[arm]
    cfg = dict(cfg)
    cfg.update(overrides)
    return agent_type, cfg
