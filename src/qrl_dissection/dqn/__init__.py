"""Off-policy (DQN) specifics.

Import explicitly: `from qrl_dissection.dqn import SafeDQN, run_grid`. This
subpackage carries FIX-01 (autoreset) and the autoreset probe, neither of which
is meaningful on-policy.
"""

from .runner import RunSpec, run_arm, run_grid
from .safe import AutoresetProbe, GreedyEvalConfig, SafeDQN

__all__ = [
    "SafeDQN", "AutoresetProbe", "GreedyEvalConfig",
    "RunSpec", "run_arm", "run_grid",
]
