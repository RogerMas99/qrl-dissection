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

FrozenLake arms (exp04) are registered further down, in their own section, with
the same three roles: the chapter's two hybrid configurations, a
capacity-matched classical control (`frozen_matched_scalar`), and a labelled
liveness guard (`frozen_onehot_mlp`).
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .obs_adapters import FROZEN_ONEHOT_ID, FROZEN_SCALAR_ID, register_environments

# Make the adapted FrozenLake ids exist as soon as the package does, so that
# SafeDQN(env_id=...) and run_grid(env_id=...) work with no runner changes.
# Cheap: registration touches only gymnasium, no torch, no simplyqrl.
register_environments()

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
# The dissection paper's OWN configurations, verbatim.
#
# WHY THESE EXIST ALONGSIDE HYBRID_FIG4, AND WHY NOTHING IS BEING REPLACED
# ------------------------------------------------------------------------
# HYBRID_FIG4 comes from the SimplyQRL *library chapter*'s DQN example
# (examples/cartpole_dqn.py). The dissection paper's own scripts use different
# values. Both are legitimate; they answer different questions.
#
#                       HYBRID_FIG4        paper (pcq-embeddings.py)
#     n_qubits          8                  4  (8 only as "augmented tests")
#     net_arch          [4]                not set  -> [] (no inference net)
#     activation        nn.Identity        not set
#     circ_type/ent     skolik / True      skolik / True      (same)
#     n_layers_q        5                  5                  (same)
#
# exp01, exp02 and exp03 all ran on HYBRID_FIG4 and their results stand. They
# answer: "does the block effect survive the move to DQN, on the configuration
# the library itself ships for DQN?" That is a real question with a real answer.
#
# What they cannot answer is the narrower one: "do OUR numbers line up with
# THEIR numbers, cell for cell?" - because the cells are not the same cell. The
# arms below close that gap without touching anything already run. Adding an arm
# invalidates no previous result; only editing HYBRID_FIG4 would, which is
# exactly why it is left alone.
#
# Each arm names the directory in the paper's results/ tree that it replicates,
# so `core.baselines.compare_to_paper` can be pointed at the right row without
# guessing. Sources: src/experiments/{pcq-embeddings,post-pqc-inference,
# pqc-entanglement}.py in javier-lazaro/qrl-dissection.
#
# CAVEAT that travels with every Skolik row here: docs/CORRECTIONS.md#fix-07.
# On that template the final layer's CZ ring cannot affect a PauliZ readout, so
# depth L carries only L-1 effective entangling blocks and a depth sweep also
# sweeps entanglement.
# ---------------------------------------------------------------------------

def paper_skolik_config(n_qubits: int, n_layers_q: int, ent: bool = True) -> Dict[str, Any]:
    """pcq-embeddings.py :: config_skolik. Note: no net_arch, no activation."""
    return {
        "circ_type": "skolik",
        "n_qubits": int(n_qubits),
        "n_layers_q": int(n_layers_q),
        "ent": bool(ent),
        "transform_fn": "cartpole_norm",
    }


def paper_salinas_config(n_qubits: int, n_layers_q: int) -> Dict[str, Any]:
    """pcq-embeddings.py :: config_uqc. The UQC embedding has only three
    rotations (Rz, Ry, Rz), so the observation is reduced to indices [1,2,3] -
    the same amputation NEW-02 flags for the classical control, but here it is
    forced by the circuit rather than chosen."""
    return {
        "circ_type": "dr",
        "n_qubits": int(n_qubits),
        "emb_indices": [1, 2, 3],
        "n_layers_q": int(n_layers_q),
        "ent": True,                  # no effect at 1 qubit; the paper says so too
        "transform_fn": "cartpole_norm",
    }


def paper_hsiao_or_config(reuse_repetitions: int, ent: bool = False) -> Dict[str, Any]:
    """post-pqc-inference.py :: config_hybrid. Hsiao's default is ent=False."""
    return {
        "circ_type": "hsiao",
        "n_qubits": 4,
        "emb_indices": [1, 2, 3],
        "reuse_repetitions": int(reuse_repetitions),
        "n_layers_q": 1,
        "ent": bool(ent),
        "transform_fn": "arctan",
    }


def paper_classical_or_config(n_repeats: int) -> Dict[str, Any]:
    """post-pqc-inference.py :: config_classical. Identical in shape to
    PAPER_LINEAR, which is that same config at n_repeats=4."""
    return {"reuse_indices": [1, 2, 3], "n_repeats": int(n_repeats), "net_arch": []}


PAPER_DEPTHS = [1, 2, 5]        # their DR sweep
PAPER_REUSE = [4, 8, 16, 32]    # their OR sweep

# arm name -> directory name under the paper's results/ tree
PAPER_ARM_TO_RESULTS: Dict[str, str] = {}

for _nq in (4, 8):
    for _L in PAPER_DEPTHS:
        _a = f"paper_skolik_{_nq}q_L{_L}"
        ARMS[_a] = ("hybrid", paper_skolik_config(_nq, _L))
        PAPER_ARM_TO_RESULTS[_a] = f"Skolik_{_nq}Q_L{_L}"
for _nq in (1, 2):
    for _L in PAPER_DEPTHS:
        _a = f"paper_salinas_{_nq}q_L{_L}"
        ARMS[_a] = ("hybrid", paper_salinas_config(_nq, _L))
        PAPER_ARM_TO_RESULTS[_a] = f"Salinas_{_nq}Q_L{_L}"
for _R in PAPER_REUSE:
    _a = f"paper_hsiao_or_r{_R}"
    ARMS[_a] = ("hybrid", paper_hsiao_or_config(_R))
    PAPER_ARM_TO_RESULTS[_a] = f"Quantum_r{_R}"
    _c = f"paper_classical_or_r{_R}"
    ARMS[_c] = ("classic", paper_classical_or_config(_R))
    PAPER_ARM_TO_RESULTS[_c] = f"Classical_r{_R}"
# Entanglement block. The L1 Skolik pair is deliberately absent: FIX-07 makes it
# a comparison of a circuit against itself, so shipping it as an arm would invite
# someone to run a cell that cannot produce a difference.
for _L in (2, 5):
    _a = f"paper_skolik_{4}q_L{_L}_noent"
    ARMS[_a] = ("hybrid", paper_skolik_config(4, _L, ent=False))
    PAPER_ARM_TO_RESULTS[_a] = f"Skolik_DR_L{_L}_Unentangled"
for _R in (4, 16):
    _a = f"paper_hsiao_or_r{_R}_ent"
    ARMS[_a] = ("hybrid", paper_hsiao_or_config(_R, ent=True))
    PAPER_ARM_TO_RESULTS[_a] = f"Hsiao_OR_r{_R}_Entangled"


# ---------------------------------------------------------------------------
# FrozenLake arms (exp04). Block 2: observation embedding and Data Reuploading.
#
# ATTRIBUTION. These configurations come from the SimplyQRL *library chapter*,
# Experiment 3 - NOT from the dissection paper (arXiv:2511.17112), which is
# CartPole-only. The chapter's FrozenLake run is a single-seed illustrative
# demonstration with no classical control and no block-level conclusion. It is a
# sanity anchor, not a baseline to transfer from. See docs/EXPERIMENT-04.md.
#
#     Config A - scalar-to-phase   n_qubits = 1, n_layers_q in {1, 5, 10, 15}
#     Config B - binary-basis      n_qubits = 4, n_layers_q in {1, 5}
#
# Both use `circ_type="skolik"`, whose template re-uploads the input once per
# layer - so `n_layers_q` IS the Data Reuploading depth and this sweep is a DR
# sweep, directly comparable to exp03 on CartPole.
#
# `ent` is forced False at one qubit: the circular CZ would be CZ(wires=[0, 0]),
# a control equal to its target. Config A is therefore *necessarily*
# unentangled, which means Config A vs Config B confounds embedding with
# entanglement. The `noent` variants of Config B exist to separate them; the
# chapter does not flag the confound.
#
# `transform_fn` is stored as a MARKER STRING here and resolved to a real object
# in build_arm_config. That keeps this module importable without simplyqrl (the
# light tests depend on it) and keeps ARMS printable and serialisable.
# ---------------------------------------------------------------------------
FROZEN_GRID = "4x4"

FROZEN_TRANSFORMS = {
    "frozen_scalar": "FrozenNormalizationTransformer",   # state -> one angle
    "frozen_binary": "FrozenBasisToAngleTransformer",    # state -> 4 bits -> {0, pi}
    "cartpole_norm": "CartPoleNormalizationTransformer", # paper's Skolik/Salinas arms
    "arctan": "ArctanTransformer",                       # paper's Hsiao arms
}


def _resolve_transform(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a `transform_fn` marker string into the upstream transformer object.

    No-op for configs that already hold a callable, so CartPole arms and any
    user-supplied transform pass through untouched.
    """
    marker = cfg.get("transform_fn")
    if not isinstance(marker, str):
        return cfg
    if marker not in FROZEN_TRANSFORMS:
        raise KeyError(
            f"unknown transform marker {marker!r}. Known: {sorted(FROZEN_TRANSFORMS)}"
        )
    import simplyqrl.transformations as _t
    klass = getattr(_t, FROZEN_TRANSFORMS[marker])
    # The Frozen* transformers need the grid size; the CartPole ones take no
    # argument. NOTE: the chapter's text writes the Frozen ones with no argument
    # either, which does not run. docs/CORRECTIONS.md#fix-06.
    cfg["transform_fn"] = klass(FROZEN_GRID) if marker.startswith("frozen_") else klass()
    return cfg


def frozen_scalar_config(n_layers_q: int) -> Dict[str, Any]:
    """Chapter Config A: one qubit, discrete state mapped to a single angle."""
    return {
        "circ_type": "skolik",
        "n_qubits": 1,
        "n_layers_q": int(n_layers_q),
        "ent": False,                 # forced: CZ(0,0) is invalid at one qubit
        "transform_fn": "frozen_scalar",
        "net_arch": [],               # chapter comments the inference net out
    }


def frozen_binary_config(n_layers_q: int, ent: bool = True) -> Dict[str, Any]:
    """Chapter Config B: four qubits, state as a binary basis vector."""
    return {
        "circ_type": "skolik",
        "n_qubits": 4,
        "n_layers_q": int(n_layers_q),
        "ent": bool(ent),
        "transform_fn": "frozen_binary",
        "net_arch": [],
    }


# Liveness guard. One-hot state, deliberately oversized. NOT a fair comparison
# and not meant to be: if this arm does not solve FrozenLake the regime is dead
# and no hybrid number measured in it is interpretable. Run it before stage 2.
FROZEN_ONEHOT_MLP: Dict[str, Any] = {"net_arch": [64, 64], "activation": nn.ReLU}

# Separates "the scalar encoding is the problem" from "the budget is".
FROZEN_SCALAR_MLP_LARGE: Dict[str, Any] = {"net_arch": [64, 64], "activation": nn.ReLU}

# The chapter's own sweep points, named so experiment scripts cannot drift from
# the published configuration by accident.
FROZEN_DR_A = [1, 5, 10, 15]      # Config A depths
FROZEN_DR_B = [1, 5]              # Config B depths

for _L in FROZEN_DR_A:
    ARMS[f"frozen_scalar_1q_L{_L}"] = ("hybrid", frozen_scalar_config(_L))
for _L in FROZEN_DR_B:
    ARMS[f"frozen_binary_4q_L{_L}"] = ("hybrid", frozen_binary_config(_L))
    ARMS[f"frozen_binary_4q_noent_L{_L}"] = ("hybrid", frozen_binary_config(_L, ent=False))
ARMS["frozen_onehot_mlp"] = ("classic", FROZEN_ONEHOT_MLP)
ARMS["frozen_scalar_mlp_large"] = ("classic", FROZEN_SCALAR_MLP_LARGE)

# Which registered env id each arm expects. Arms see the scalar observation
# unless they are the one-hot reference.
FROZEN_ARM_ENV = {a: (FROZEN_ONEHOT_ID if a == "frozen_onehot_mlp" else FROZEN_SCALAR_ID)
                  for a in ARMS if a.startswith("frozen_")}
FROZEN_ARM_ENV["frozen_matched_scalar"] = FROZEN_SCALAR_ID   # computed, not in ARMS

# Reference hybrid the matched classical control is sized against.
FROZEN_MATCHED_REFERENCE = "frozen_scalar_1q_L5"


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
    "frozenlake": {
        # The adapted id, not "FrozenLake-v1": upstream's DQN cannot consume a
        # Discrete observation space at all. See docs/CORRECTIONS.md#fix-05.
        "env_id": FROZEN_SCALAR_ID,
        "raw_env_id": "FrozenLake-v1",
        "env_kwargs": {"map_name": "4x4", "is_slippery": False},
        "obs_dim": 1,
        "n_actions": 4,
        "onehot_env_id": FROZEN_ONEHOT_ID,   # classical reference arm only
        "max_episode_steps": 100,
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

    if arm == "frozen_matched_scalar":
        # [exp04] The FrozenLake counterpart of matched_classical, kept as its
        # own branch because the geometry differs on every axis: the control
        # sees a 1-D scalar rather than a reused 4-vector, and FrozenLake has
        # four actions rather than two. Same principle, though - the width is
        # derived from the reference hybrid's measured budget, never hard-coded,
        # so it stays correct if the circuit changes.
        ref = hybrid_reference if hybrid_reference.startswith("frozen_") \
            else FROZEN_MATCHED_REFERENCE
        ref_type, ref_cfg = ARMS[ref]
        if ref_type != "hybrid":
            raise ValueError("hybrid_reference must name a hybrid arm")
        budgets = pqc_parameter_budget(_resolve_transform(dict(ref_cfg)),
                                       env_id=FROZEN_SCALAR_ID)
        budget = budgets["total"] if match_to == "total" else budgets["quantum"]
        # in_dim 1: the scalar state. out_dim 4: FrozenLake's action space.
        width, spent = match_hidden_width(budget, in_dim=1, out_dim=4)
        # match_hidden_width returns the CLOSEST width, which at these tiny
        # budgets can land under the hybrid (18 params -> width 2 -> 16). Round
        # up instead: a control that loses while carrying fewer parameters
        # proves nothing, so the conservative direction is to give it slightly
        # more. Costs at most one extra hidden unit.
        while spent < budget and width < 4096:
            width += 1
            spent = 1 * width + width + width * 4 + 4
        cfg = {
            "net_arch": [width],
            "activation": activation,
            "_matched_to": ref,
            "_match_target": match_to,
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
    # No-op unless the arm carries a FrozenLake transform marker.
    return agent_type, _resolve_transform(cfg)
