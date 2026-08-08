# Literature: where this work sits

## The genealogy, and why it changes the framing

The obvious story is "a 2025 paper dissected QRL under PPO, and this thesis
repeats it under DQN". That is true and it undersells the result. The actual
lineage:

```
Skolik, Jerbi & Dunjko (2022)   DQN + PQC, CartPole AND FrozenLake,
                                with architectural ablation studies
        |
        v
SimplyQRL chapter (2025)        a library packaging those circuit templates,
                                with illustrative demos (Exp 3 = FrozenLake, PPO)
        |
        v
Dissection paper (2025)         dissects the blocks systematically - but under
                                PPO, and on CartPole only
        |
        v
this thesis                     returns them to DQN, and finds out why that was
                                not a formality
```

**The dissection paper moved the algorithm axis away from Skolik and did not
say so.** Skolik et al. had already performed ablations over architectural
choices under deep Q-learning, on both CartPole and FrozenLake, in 2022. The
2025 dissection is more systematic on the block decomposition, but it is
on-policy and single-environment.

So this work is not "extending a recent paper". It is **closing a loop**: taking
a block-level analysis that was developed on-policy back to the off-policy
setting the templates originally came from - and discovering that the library's
off-policy path had not been maintained for the journey. FIX-01 (autoreset
phantom transitions) and FIX-05 (Discrete observation spaces) are both instances
of the same finding, in different subsystems, with the commented-out CartPole-era
reshape in `ppo.py` as direct evidence that the on-policy path was hardened and
the off-policy one was not.

That is a better introduction than "we re-ran their experiments".

### What each source can and cannot support

| source | algorithm | environments | seeds | can it anchor a transfer claim? |
|---|---|---|---|---|
| Skolik et al. 2022 | **DQN** | CartPole, FrozenLake | ablations | **yes** - same algorithm as ours |
| SimplyQRL chapter | PPO (+1 DQN curve) | CartPole, FrozenLake | 1 (Exp 3) | no - illustrative, uncontrolled |
| Dissection paper | PPO | CartPole | 10 | yes, for the algorithm-transfer question |

Keep the last two apart in the write-up. Conflating them was corrected once
already; see `EXPERIMENT-04.md` section 0.

Note the consequence for exp04: FrozenLake **does** have a published DQN
reference, in Skolik et al. Its FrozenLake sweep is `n_layers ∈ {5, 10, 15}`,
which overlaps our Config A. exp04 is therefore a partial replication of a
foundational result with the additions the original lacks - a capacity-matched
classical control, multiple seeds, and FIX-05.

## Reading list, in priority order

Marked **[checked]** where the citation was verified against the published
record while writing this; the rest are from established memory and should be
confirmed before they enter a bibliography.

### 1 — required

**Skolik, Jerbi & Dunjko (2022), "Quantum agents in the Gym: a variational
quantum algorithm for deep Q-learning".** *Quantum* 6, 720. arXiv:2103.15084
**[checked]**
The direct ancestor. PQC + deep Q-learning on CartPole and FrozenLake, with
ablation studies over architectural choices. Read it against exp01/exp03/exp03b
and state explicitly what this thesis adds.

**Franz et al. (2022), "Uncovering instabilities in variational-quantum deep
Q-networks".** *J. Franklin Inst.* arXiv:2202.05195 **[checked]**
VQ-DQN policies diverge, and this affects the reproducibility of established
results from classical simulation. The correction registry in `CORRECTIONS.md`
has a precedent: this is a known failure mode of the off-policy branch of the
field, not an anecdote. Cite it to place the contribution in an existing line.

### 2 — statistics

**Agarwal et al. (2021), "Deep Reinforcement Learning at the Edge of the
Statistical Precipice".** NeurIPS, Outstanding Paper. arXiv:2108.13264
**[checked]**
See `docs/STATISTICS.md`. Changes what we report and, more importantly, exposes
a bias in the metric this repository was using.

**Henderson et al. (2018), "Deep Reinforcement Learning that Matters".** AAAI.
Seeds, implementation details and hyper-parameters dominating algorithmic
claims. The bibliographic backing for plan B.

**Engstrom et al. (2020), "Implementation Matters in Deep RL: A Case Study on
PPO and TRPO".** ICLR.
Gains attributed to PPO over TRPO traced to implementation details rather than
the algorithm. The framing for FIX-01 through FIX-08.

### 3 — per block

**Hsiao et al. (2022), "Unentangled quantum reinforcement learning agents in the
OpenAI Gym".**
Source of the `hsiao` template. Argues a circuit **without entangling gates**
suffices. Directly relevant to FIX-07 and exp03b, and it points the opposite way
to the intuition those experiments test.

**Pérez-Salinas et al. (2020), "Data re-uploading for a universal quantum
classifier".** *Quantum* 4, 226.
Origin of data reuploading and of the `dr`/Salinas template. Needed to argue why
depth should matter at all.

**Jerbi et al. (2021), "Parametrized quantum policies for reinforcement
learning".** NeurIPS.
The on-policy side, where the dissection paper comes from.

### 4 — context

**"A Survey on Quantum Reinforcement Learning".** arXiv:2211.03464 **[checked]**
For situating the field and finding references this list misses.

**McClean et al. (2018), "Barren plateaus in quantum neural network training
landscapes".** *Nature Communications*.
Read this **before** invoking barren plateaus for a curve that falls with depth.
It sets out what evidence the claim requires - gradient norms, not a downward
trend.

## Gaps worth noting in the write-up

- The Salinas/UQC template has never been run under DQN here. The dissection
  paper's DR block contrasts two embedding philosophies; exp03 sweeps depth on
  one of them. So exp03 tests DR *depth* transfer, not the embedding contrast
  the paper drew its conclusion from.
- `Hsiao_OR_r{4,16}_{Ent,Unent}` has no counterpart here - exp01 studied
  capacity, not entanglement on/off.

Both are gaps in breadth. `ROADMAP.md` argues they should wait behind the
robustness pass; recording them here means the write-up can state the scope
honestly rather than leaving a reader to notice.
