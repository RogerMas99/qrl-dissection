# ppo/ - on-policy specifics (under construction)

Empty by design. When populated, this subpackage mirrors `dqn/`:

- `safe.py`  - a `SafePPO` orchestration wrapper around `simplyqrl.ppo.PPO`,
  plus a greedy-eval hook analogous to the one in `dqn/safe.py`.
- `runner.py` - `RunSpec` / `run_grid` for PPO, reusing `core/` manifests.

**What it will NOT contain: FIX-01.** PPO has no replay buffer, so the autoreset
phantom transition is a bounded point bias in GAE at the episode boundary, not
cumulative poisoning. See `docs/CORRECTIONS.md#fix-01`.

**What it reuses from `core/`:** everything algorithm-agnostic - capacity
accounting and matching (`capacity.py`), the experiment arms and environment
registry (`configs.py`), and result loading/plotting (`analysis.py`). If a PPO
experiment needs a new arm or environment, it is registered in `core/configs.py`
so both algorithms share it.

The paper's own results are PPO, so this is also where cross-checks against the
published numbers will live.
