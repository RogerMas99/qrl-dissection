"""
qrl_dissection - Cross-algorithm extension of the QRL component dissection.

Built on top of SimplyQRL (Lazaro, Vazquez & Garcia Bringas, 2025). The paper
evaluates three pipeline blocks under a fixed PPO/CartPole protocol; this
package asks how those conclusions behave as the RL algorithm and the
environment change.

Layout
------
    core/   algorithm-agnostic: SimplyQRL corrections (FIX-02, FIX-03),
            parameter accounting and capacity matching, experiment arms,
            result loading and plotting.
    dqn/    off-policy specifics: FIX-01 (autoreset), the probe, greedy eval.
    ppo/    on-policy specifics (see ppo/README.md; being built out).

Importing this top-level package applies ONLY the algorithm-agnostic upstream
corrections (core.compat) and exposes core. The dqn and ppo subpackages are
imported explicitly - `from qrl_dissection.dqn import SafeDQN` - so an on-policy
experiment never pulls in off-policy machinery, and vice versa.

Correction / addition registry: docs/CORRECTIONS.md
    FIX-01  autoreset NEXT_STEP phantom transition   -> dqn/safe.py
    FIX-02  OutputScale no-op in the is_qnet branch   -> core/compat.py
    FIX-03  agent_type="classic" unresolvable         -> core/compat.py
    FIX-04  broken dependency pins / jax in Colab      -> scripts/verify_env.py
    FIX-05  Discrete observation spaces unusable (DQN) -> core/obs_adapters.py
    FIX-06  chapter's transformer signature (doc only) -> core/configs.py
    NEW-01  autoreset instrumentation                  -> dqn/safe.py
    NEW-02  capacity-matched classical control arm     -> core/capacity.py
    NEW-03  greedy evaluation hook                      -> dqn/safe.py
    NEW-04  resumable run orchestration                 -> dqn/runner.py
"""

__version__ = "0.1.0"

from .core.compat import apply_upstream_patches, upstream_report

# FIX-02 and FIX-03 patch SimplyQRL itself, so they belong at package import
# regardless of which algorithm you use. FIX-01 lives in dqn/ because it only
# concerns the off-policy replay buffer.
apply_upstream_patches()

from .core import analysis  # noqa: E402
from .core.capacity import capacity_ladder, count_trainable, match_hidden_width  # noqa: E402
from .core.configs import ARMS, build_arm_config  # noqa: E402
from .core.obs_adapters import FROZEN_ONEHOT_ID, FROZEN_SCALAR_ID, register_environments  # noqa: E402

__all__ = [
    "analysis",
    "ARMS", "build_arm_config",
    "register_environments", "FROZEN_SCALAR_ID", "FROZEN_ONEHOT_ID",
    "capacity_ladder", "count_trainable", "match_hidden_width",
    "upstream_report",
]
