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

# Input policy for the capacity-matched control. The paper's classical arm keeps
# only indices [1,2,3] (cart position discarded, following Hsiao et al.). That is
# right for `paper_linear`, a faithful replication - but WRONG for a fair control
# against the hybrid, which sees all four observations. Cart position is one of
# the two termination conditions of CartPole (|x| > 2.4), so an arm that cannot
# see it may fail from blindness rather than from being classical. The fair
# control therefore uses the full observation. See docs/CORRECTIONS.md#new-02.
MATCHED_REUSE_INDICES_FAIR = [0, 1, 2, 3]   # full observation, matches the hybrid
MATCHED_REUSE_INDICES_PAPER = [1, 2, 3]     # paper's amputated set (for ablation)

ARMS = {
    "paper_linear": ("classic", PAPER_LINEAR),
    "hybrid_fig4": ("hybrid", HYBRID_FIG4),
    "oversized_mlp": ("classic", OVERSIZED_MLP),
}


# ---------------------------------------------------------------------------
# Block-sweep configs: OR and DR (paper blocks 1 and 2).
#
# exp01 studied block 3 (ansatz / entanglement). These two build the sweeps for
# the other two blocks, as minimal variations on HYBRID_FIG4 so that only the
# studied knob moves. Same simple set-up as exp01 (8 qubits, 60k steps): a first
# pass for COVERAGE, not the final statistically-robust run. See the standing
# note in docs/ROADMAP.md - every block sweep must later be re-run at 8-10 seeds
# (plan "B").
#
# OR (Output Reuse, block 1): replicate the PQC readout R times before the
# linear head. Knob: reuse_repetitions. Paper sweeps R in {4, 8, 16, 32}.
# DR (Data Reuploading, block 2): circuit depth. Knob: n_layers_q. The paper
# contrasts embedding philosophies; here we sweep depth on the Skolik template,
# holding everything else at the Fig. 4 values.
# ---------------------------------------------------------------------------
def hybrid_or_config(reuse_repetitions: int) -> Dict[str, Any]:
    """HYBRID_FIG4 plus Output Reuse of factor `reuse_repetitions`."""
    cfg = dict(HYBRID_FIG4)
    cfg["reuse_repetitions"] = int(reuse_repetitions)
    return cfg


def hybrid_dr_config(n_layers_q: int) -> Dict[str, Any]:
    """HYBRID_FIG4 with circuit depth (Data Reuploading) set to `n_layers_q`."""
    cfg = dict(HYBRID_FIG4)
    cfg["n_layers_q"] = int(n_layers_q)
    return cfg


# Paper's sweep points, kept here so experiment scripts refer to them by name.
OR_REPEATS = [4, 8, 16, 32]     # Output Reuse factors R
DR_DEPTHS = [1, 2, 5]           # Data Reuploading depths L (Fig. 4 uses 5)


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
    match_to: str = "total",
    observation: str = "full",
    **overrides: Any,
):
    """Return (agent_type, agent_config) for an arm name.

    `matched_classical` is computed, not hard-coded: the hidden width is derived
    from the reference hybrid arm's measured parameter budget, so it stays
    correct if the circuit changes.

    match_to
    --------
    "total"   (default) match the FULL hybrid parameter count (PQC + classical
              head). This is the honest control: same total budget as the model
              it is meant to rival, so any gap is attributable to the circuit
              rather than to a parameter-count advantage. The hybrid arm keeps
              the paper's exact circuit unchanged - only this classical control
              is resized.
    "quantum" match only the PQC weight count. Leaves the classical arm with
              MORE total parameters than the hybrid (it adds its own head on
              top), which biases the comparison toward the classical arm.
              Kept for completeness; not the default.

    observation
    -----------
    "full"  (default) the matched control sees all four observations, exactly
            like the hybrid. This is the fair control: same information, same
            budget, only the circuit differs.
    "paper" reproduce the paper's amputated input (cart position discarded).
            Use ONLY for the ablation that measures how much the amputation
            itself costs - not for the main circuit-vs-classical comparison.
    """
    from .capacity import match_hidden_width, pqc_parameter_budget

    if arm == "matched_classical":
        ref_type, ref_cfg = ARMS[hybrid_reference]
        if ref_type != "hybrid":
            raise ValueError("hybrid_reference must name a hybrid arm")
        budgets = pqc_parameter_budget(ref_cfg, env_id=env_id)
        if match_to == "total":
            budget = budgets["total"]
        elif match_to == "quantum":
            budget = budgets["quantum"]
        else:
            raise ValueError(f"match_to must be 'total' or 'quantum', got {match_to!r}")
        if observation == "full":
            indices = MATCHED_REUSE_INDICES_FAIR
        elif observation == "paper":
            indices = MATCHED_REUSE_INDICES_PAPER
        else:
            raise ValueError(f"observation must be 'full' or 'paper', got {observation!r}")
        n_repeats = PAPER_LINEAR["n_repeats"]
        # SelectiveOutputReuse replicates the (reduced) observation n_repeats
        # times before the network, so the first Linear sees len(indices) *
        # n_repeats features, not len(indices). Sizing against the wrong in_dim
        # made the matched arm ~2.5x too big. See docs/CORRECTIONS.md#new-02.
        in_dim = len(indices) * n_repeats
        width, spent = match_hidden_width(budget, in_dim=in_dim, out_dim=2)
        cfg = {
            "reuse_indices": list(indices),
            "n_repeats": n_repeats,
            "net_arch": [width],
            "activation": activation,
            "_matched_to": hybrid_reference,
            "_match_target": match_to,
            "_observation": observation,
            "_budget": budget,
            "_hybrid_total": budgets["total"],
            "_hybrid_quantum": budgets["quantum"],
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
