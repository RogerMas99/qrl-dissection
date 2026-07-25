"""Algorithm-agnostic building blocks shared by every experiment.

Nothing here knows about DQN or PPO. If a component starts needing to, it does
not belong in core.
"""

from .capacity import (
    build_agent_for,
    capacity_ladder,
    count_trainable,
    match_hidden_width,
    pqc_parameter_budget,
)
from .compat import apply_upstream_patches, upstream_report
from .configs import ARMS, ENVIRONMENTS, build_arm_config

__all__ = [
    "apply_upstream_patches", "upstream_report",
    "build_agent_for", "capacity_ladder", "count_trainable",
    "match_hidden_width", "pqc_parameter_budget",
    "ARMS", "ENVIRONMENTS", "build_arm_config",
]
