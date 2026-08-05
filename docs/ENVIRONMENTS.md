# Which stack, and why

Three environments are in play. They are not interchangeable, and the difference
between two of them is the subject of a correction.

| | stack A — **ours** | stack B — the paper's published pins | stack C — modern |
|---|---|---|---|
| source | SimplyQRL's own June-2025 lock | `requirements.txt` in `javier-lazaro/qrl-dissection` | not chosen yet |
| python | ≥ 3.10 | ≥ 3.12 (declared) | — |
| gymnasium | 1.1.1 | 1.2.1 | latest |
| pennylane | 0.41.1 | ≥ 0.36 (README) | latest |
| autoray | 0.7.1 | **0.8.0** | latest |
| status | works | **cannot import PennyLane** | untested |

## Stack A is the default, and stack B is why FIX-04 exists

`autoray==0.8.0` removes the attribute PennyLane 0.41 imports at module load, so
`import pennylane` raises before any project code runs. The published artefact
cannot be installed as specified. We therefore follow SimplyQRL's own lock —
the environment that actually produced the published results — rather than the
companion repository's export. Details in `CORRECTIONS.md#fix-04`.

## The library code is the same; only the environment differs

Worth stating plainly, because it is easy to assume otherwise: the SimplyQRL
vendored in the paper's repository is **byte-identical** to upstream `b534cc9`,
and to the copy vendored here at `src/simplyqrl/`. Verified by `diff -r`, and
`b534cc9` is upstream's current HEAD (last commit 31 Oct 2025), so there is no
newer version to diverge from either.

So "their library" and "SimplyQRL" are the same code. What differs is what gets
installed around it.

## Upgrading gymnasium does not fix FIX-01

Checked directly rather than assumed, because it decides whether the
modernisation branch would make the autoreset correction unnecessary:

```
gymnasium 1.1.1  SyncVectorEnv default: autoreset_mode = AutoresetMode.NEXT_STEP
gymnasium 1.2.1  SyncVectorEnv default: autoreset_mode = AutoresetMode.NEXT_STEP
```

Both default to `NEXT_STEP`, which is the behaviour that produces the phantom
transitions FIX-01 removes. This is an intentional gymnasium design decision, not
a bug that a later release repairs — so moving to stack B or C changes nothing
about FIX-01, and the correction stays necessary at every version.

## On PPO and FrozenLake

No separate stack is needed, and this is worth writing down because it has come
up more than once.

The FrozenLake material lives in the SimplyQRL *library chapter*, not the
dissection paper, and it is PPO. PPO is unaffected by both FIX-01 (it does not
use a replay buffer, so there are no phantom transitions to poison) and FIX-05
(it reshapes observations to `(num_envs, -1)` explicitly and stores them as
floats). A PPO FrozenLake run to sanity-check against the chapter's Fig. 6 would
use **stack A**, exactly like everything else here.

What would differ is not the stack but the correction set: a PPO experiment needs
neither FIX-01 nor FIX-05, and `qrl_dissection/ppo/` is still scaffolding.

## When to move to stack C

`docs/ROADMAP.md` carries the standing commitment: modernise on a branch when the
work stops comparing against published numbers. Two constraints bind that
decision:

- Cross-experiment comparisons must share a stack. exp04's headline is FIX-01's
  effect on FrozenLake measured *against* exp01's null on CartPole; changing
  gymnasium between them would make the claim unfalsifiable.
- Upgrading buys less than it looks like it does. It fixes FIX-04 — which is an
  installation problem we have already routed around — and nothing else on the
  list.

## Reproducing the check

```bash
pip download gymnasium==1.2.1 -d /tmp/g --no-deps
python - <<'PY'
import zipfile, glob, re
src = zipfile.ZipFile(glob.glob('/tmp/g/gymnasium-1.2.1*.whl')[0]) \
        .read('gymnasium/vector/sync_vector_env.py').decode()
print(re.search(r'autoreset_mode[^\n]*=[^\n]*', src).group(0).strip())
PY
```
