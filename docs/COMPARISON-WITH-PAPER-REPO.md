# This repository vs. the paper's companion repository

Written because the divergence is worth being deliberate about rather than
drifting into. The short answer: **the differences are the contribution, except
for one that was an accident and is now fixed.**

## What they ship, and what we ship

| | `javier-lazaro/qrl-dissection` | this repository |
|---|---|---|
| SimplyQRL | vendored at `src/simplyqrl/` | **vendored at `src/simplyqrl/`, byte-identical** |
| experiment code | 3 flat scripts, one per paper section | a package (`qrl_dissection/`) plus per-experiment scripts |
| configuration | literals inline in each script | `core/configs.py :: ARMS`, one registry |
| algorithm | PPO | DQN (PPO subpackage is scaffolding) |
| environment | CartPole-v1 | CartPole-v1 and FrozenLake-v1 |
| seeds | 10 | 3 (coverage); 8–10 pending, see ROADMAP plan B |
| logging | TensorBoard | CSV plus per-run manifests |
| corrections | none | seven, registered in `CORRECTIONS.md` |
| their results | `results/`, 140 MB of event files | `data/paper_ppo_*.csv`, extracted, 430 KB |

## The differences that are intentional

**Vendoring: now the same as theirs.** We used to pin SimplyQRL as a git
dependency at `b534cc9`. It is now vendored at `src/simplyqrl/`, which is
byte-identical to upstream *and* byte-identical to their copy — verified, and
guarded by `tests/test_vendored_integrity.py`. Diff the two repositories
directly if you want to.

This does not make us a fork. Nothing in `src/simplyqrl/` is edited. Every
correction lives in `qrl_dissection/` and is applied at runtime by
`core/compat.py`, with guards that fail loudly if upstream shifts underneath.
That separation *is* the audit trail: a reader must be able to see what upstream
does and what we do to it as two separable things. The integrity test exists
because the rule was broken within minutes of being written — a bulk rename
matched an import inside the vendored code and nothing failed, because an
installed copy was shadowing the vendored one.

**Package instead of flat scripts.** Their three scripts define configs inline
and are not meant to be composed. Ours has to sweep arms across seeds, block
settings and now environments, with resumable runs and capacity accounting. A
registry is not gold-plating here; without it, arms drift between scripts and
the fair-control discipline cannot be enforced in one place.

**Corrections.** FIX-01 to FIX-07 are the reason this repository exists. They
cannot be expressed in their layout, since it has no place to put a correction
that is not an edit to the library.

**A second environment and a second algorithm.** The thesis question.

## The difference that was an accident

**`hybrid_fig4` is not the paper's configuration, and exp01/02/03 ran on it.**

`HYBRID_FIG4` came from the *library chapter*'s DQN example
(`examples/cartpole_dqn.py`). The dissection paper's own `pcq-embeddings.py`
uses different values:

| | `hybrid_fig4` | paper's Skolik arm |
|---|---|---|
| `n_qubits` | 8 | 4 (8 only as "augmented tests") |
| `net_arch` | `[4]` | not set → `[]`, no inference net |
| `activation` | `nn.Identity` | not set |
| `circ_type`, `ent`, `n_layers_q` | skolik, True, 5 | skolik, True, 5 |

This is the one real problem, and it is narrower than it looks. exp01, exp02 and
exp03 remain valid: they answer *"does the block effect survive the move to DQN,
on the configuration the library ships for DQN?"* — a real question with a real
answer. What they cannot answer is *"do our numbers line up with theirs, cell
for cell?"*, because the cells are not the same cell.

**The fix adds, it does not replace.** `core/configs.py` now registers 24
`paper_*` arms transcribed from their three scripts, each mapped to the
directory in their `results/` tree it replicates, so
`baselines.paper_results_for_arm()` finds the right 10-seed row without anyone
having to remember the correspondence. `HYBRID_FIG4` is untouched, and
`tests/test_paper_arms.py` asserts the four original arms still count 26 / 126 /
10934 / 135 parameters. Nothing already run is invalidated; editing
`HYBRID_FIG4` in place would have invalidated all of it, which is exactly why it
was left alone.

Nothing needs re-running for the existing conclusions to stand. The new arms are
there for when a cell-for-cell comparison is wanted.

## Their coverage vs ours

Their published grid, and where we stand on it under DQN:

| block | their configurations (10 seeds, PPO) | ours (DQN) |
|---|---|---|
| OR | `Quantum_r{4,8,16,32}`, `Classical_r{4,8,16,32}` | exp02, on `hybrid_fig4` — running |
| DR | `Skolik_{4,8}Q_L{1,2,5}`, `Salinas_{1,2}Q_L{1,2,5}` | exp03 covers Skolik 8Q only |
| Ansatz | `Hsiao_{OR,DR}_*_{Ent,Unent}`, `Skolik_DR_L{1,2,5}_*` | exp01, capacity-focused, no ent sweep |

Two gaps are now visible that were not before:

- **The Salinas/UQC template has never been run under DQN.** Their DR block
  contrasts two embedding philosophies; ours sweeps depth on one of them. So
  exp03 tests DR *depth* transfer, not the embedding contrast the paper actually
  drew its conclusion from.
- **`Hsiao_OR_r{4,16}_{Ent,Unent}` has no counterpart here.** exp01 studied
  capacity, not entanglement on/off.

These are gaps in *breadth*. The recommendation stands that they wait: no
experiment in this repository has had its 8–10 seed robustness pass, and their
own spreads (see `PAPER-BASELINES.md`) show that at 10 seeds several published
contrasts still sit inside one standard deviation. Breadth without seeds would
produce a wide, shallow thesis with the same weakness the original has.

## What we take from them

- `src/simplyqrl/` — vendored, unmodified, byte-identical to their copy.
- `data/paper_ppo_*.csv` — their 360 logged runs, reduced to 430 KB and
  summarised with our metric, including full learning curves at 500-step
  resolution so their figures can be redrawn and ours overlaid.
- `core/configs.py :: paper_*` arms — their configurations, transcribed.
- `simplyqrl.envs.make_vec_env` — used, as upstream intends. Our
  `core/obs_adapters.py` was renamed from `envs.py` precisely so the two are
  never confused: theirs builds vector environments, ours changes how an
  observation is presented.
