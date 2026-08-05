# The paper's own results, at 10 seeds

## What we now have

The dissection paper's companion repository ships the raw TensorBoard event
files behind every published figure: **3 blocks, 36 configurations, 10 seeds,
360 runs, 100k steps each**. `scripts/extract_paper_baselines.py` reduces that
~140 MB of binary logs to two CSVs in `data/`, summarised with **our** metric
(`best_ma50`, the same rolling window `core.analysis.summarise_run` applies to
our DQN runs).

This changes the standing of every transfer claim in the repo. Until now, "the
paper reports X" meant a number read off a figure. It now means the authors'
logged returns, with standard deviations.

```python
from qrl_dissection.core import baselines
baselines.compare_to_paper(our_best=199.0, paper_config_name="Skolik_8Q_L5")
# 'Skolik_8Q_L5: paper (PPO, n=10) 199.3 +/- 60.1 | ours (DQN) 199.0 | delta -0.3 - within spread'
```

## Read the spread first

| block | configuration | best_ma50 (mean ± sd, n=10) |
|---|---|---|
| OR | Quantum_r4 / r8 / r16 / r32 | 116.5 ± 28.9 / 230.0 ± 88.1 / 354.8 ± 124.3 / 430.5 ± 115.6 |
| OR | Classical_r4 / r8 / r16 / r32 | 263.9 ± 18.4 / 320.0 ± 36.3 / 381.7 ± 37.3 / 330.2 ± 63.4 |
| DR | Skolik_4Q_L1 / L2 / L5 | 28.2 ± 1.3 / 74.5 ± 50.9 / 175.4 ± 68.4 |
| DR | Skolik_8Q_L1 / L2 / L5 | 32.2 ± 2.6 / 144.0 ± 37.9 / 199.3 ± 60.1 |
| DR | Salinas_1Q_L1 / L2 / L5 | 35.4 ± 9.6 / 53.6 ± 9.4 / 95.5 ± 62.7 |
| DR | Salinas_2Q_L1 / L2 / L5 | 48.6 ± 17.5 / 51.0 ± 10.6 / 107.1 ± 47.4 |
| Ent | Hsiao_OR_r4 Ent / Unent | 31.8 ± 1.9 / 116.5 ± 28.9 |
| Ent | Hsiao_OR_r16 Ent / Unent | 35.4 ± 2.4 / 354.8 ± 124.3 |
| Ent | Skolik_DR_L5 Ent / Unent | 177.8 ± 66.4 / 147.2 ± 49.9 |

Two things jump out of the spread column.

**The quantum OR arms are enormously variable.** `Quantum_r32` is 430.5 ± 115.6
against `Classical_r16` at 381.7 ± 37.3. The quantum arm's advantage is roughly
one third of its own standard deviation. At 10 seeds that separation is not
established, and any statement that the PQC readout beats the classical one at
matched reuse should say so.

**The Skolik entanglement contrast is inside noise.** At L5, entangled 177.8 ±
66.4 against unentangled 147.2 ± 49.9. The difference is well under a pooled
standard deviation, and at L2 the ordering reverses (74.5 vs 176.6). Our exp01
found no clear circuit advantage under DQN; the authors' own data does not
establish a clear entanglement advantage under PPO either.

**This is the argument for plan B.** If the original work needs 10 seeds and
still cannot separate several of its arms, a 3-seed coverage pass here settles
nothing. Every conclusion in `RESULTS-LOG.md` stays provisional until its
robustness pass exists.

## What is not independent

Some configurations are the same runs under two labels. Legitimate - the default
`ent` value of each template makes one block's runs serve as the other block's
control arm - but it means the blocks are not independent evidence, and a reader
comparing across them should know. `baselines.paper_duplicate_groups()` lists
them; verified by checksum:

| label | same files as | why |
|---|---|---|
| `Hsiao_OR_r4_Unentangled` | `OR/Quantum_r4` | Hsiao's default is `ent=False` |
| `Hsiao_OR_r16_Unentangled` | `OR/Quantum_r16` | same |
| `Skolik_DR_L1_Entangled` | `DR/Skolik_4Q_L1` | Skolik's default is `ent=True` |
| `Skolik_DR_L2_Entangled` | `DR/Skolik_4Q_L2` | same |

And one that is *not* file duplication but is still identical data:
`Skolik_DR_L1_Unentangled` matches its entangled counterpart on all ten seeds
because at depth 1 the flag cannot do anything. See `CORRECTIONS.md#fix-07`.

## What this says about exp03

exp03 found greedy 15 -> 35 -> 199 for L = 1/2/5 under DQN. Against the paper's
own Skolik_8Q curve:

| depth | paper (PPO, n=10) | ours (DQN, n=3) | verdict |
|---|---|---|---|
| 1 | 32.2 ± 2.6 | 15 | outside spread - and below the random baseline of ~22, i.e. a degenerate policy |
| 2 | 144.0 ± 37.9 | 35 | outside spread |
| 5 | 199.3 ± 60.1 | 199 | within spread - indistinguishable |

Both curves are monotone in depth, so the headline "DR transfers to DQN" holds.
But the **shape** does not transfer. Under PPO most of the gain has arrived by
L2 (32 -> 144); under DQN almost nothing has happened by L2 (15 -> 35) and the
gain arrives late. The accurate claim is narrower and more interesting than the
one exp03 currently supports: *DR helps DQN too, but DQN needs more depth to
extract the same benefit.*

Carry the FIX-07 caveat with it. On the Skolik template, depths 1/2/5 supply
0/1/4 effective entangling blocks, so both curves confound depth with
entanglement. The `ent=False` rerun that separates them is now on the roadmap.

## Reproducing the extraction

```bash
pip install tbparse
git clone https://github.com/javier-lazaro/qrl-dissection.git /tmp/paper-repo
python scripts/extract_paper_baselines.py --results /tmp/paper-repo/results
```

The CSVs in `data/` are derived artefacts. Regenerate them; never hand-edit.
