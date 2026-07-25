"""On-policy (PPO) specifics. Under construction - see ppo/README.md.

Deliberately empty for now. When populated it will mirror dqn/: a SafePPO
orchestration wrapper and a runner, reusing core/ for everything algorithm-
agnostic (capacity matching, arms, analysis). It will NOT carry FIX-01: PPO has
no replay buffer, so the autoreset phantom is a bounded GAE boundary effect
rather than cumulative poisoning (see docs/CORRECTIONS.md#fix-01).
"""

__all__ = []
