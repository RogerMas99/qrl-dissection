# Statistics: what we report, and one thing we were reporting wrongly

Following Agarwal et al., *Deep Reinforcement Learning at the Edge of the
Statistical Precipice* (NeurIPS 2021, Outstanding Paper, arXiv:2108.13264).
Implemented in `src/qrl_dissection/core/stats.py`, pinned by `tests/test_stats.py`,
and wired into `notebooks/20_dqn_results.ipynb` section 2b.

Implemented by hand rather than by depending on `rliable`, so the pinned
environment stays untouched. The estimators are short; the point is to get the
reporting right, not to add a package.

---

## The part that matters: our metric was biased

`best_ma50` and `greedy_best` are **maxima over training**. A maximum over a
noisy curve is positively biased, and **the bias grows with the variance of the
curve** - so a noisier arm scores higher for nothing.

Measured, with two arms of identical true performance:

```
true mean 200, sd 120 (noisy)   ->  max over 500 evaluations = 567
true mean 200, sd  35 (stable)  ->  max over 500 evaluations = 306
```

An invented 85% advantage, from variance alone. Pinned as
`test_max_over_training_rewards_noise`.

**This is not academic here.** The dissection paper's own OR block, now available
at 10 seeds in `data/paper_ppo_summary.csv`, has quantum arms at ±115 and
classical arms at ±37. The quantum side is roughly three times noisier, so a
max-over-training protocol systematically flatters it. Agarwal et al. identify
exactly this class of non-standard evaluation protocol as an explanation for
apparent gains elsewhere in the literature (their Figure 5).

Every headline number in `RESULTS-LOG.md` was produced with this statistic,
including exp03's 15 → 35 → 199. **Recompute them before writing any of it up.**

### The replacement

`stats.final_performance(rewards, last_frac=0.1)` - the mean return over the
final 10% of episodes. Every run reaches the end of training on equal terms, so
there is no variance premium.

It needs **no retraining**: every episode is already in the CSVs. This is the
clearest case of the analysis-level reuse described in `REUSE.md`.

Notebook 20 reports both side by side, deliberately. If a result survives the
change of statistic it is stronger for having been checked; if it does not, that
is a finding, and a cheaper one to make now than during a defence.

---

## What we report instead of mean ± sd

**IQM** (`stats.iqm`) - the mean of the middle 50% of runs. Keeps the median's
robustness against a diverging run, with far less uncertainty. VQ-DQN policies
diverge regularly (Franz et al. 2022), so the outlier case is the common one
here, not the pathological one.

**Percentile bootstrap CIs** (`stats.bootstrap_ci`) - better justified at small
N than a sample standard deviation, which is not an interval estimate at all.

**Probability of improvement** (`stats.probability_of_improvement`) - the honest
phrasing of a comparison at small N:

```
P(an exp03 run > an exp03b run) = 0.62   (n = 10 vs 10)
```

Weaker than "exp03 is better", more accurate, and it does not pretend two means
are separated when the runs overlap.

---

## Why the target is 10 seeds, precisely

Agarwal et al. validate that percentile CIs give good coverage from about
**N = 10 runs**, and warn that **at N = 3 bootstrap CIs underestimate the true
95% interval** - they are too narrow, so they overstate confidence.

That is the statistical argument for the plan-B commitment in `ROADMAP.md`, and
it is a better one than "more seeds is nicer": a three-seed interval is not
merely wide, it is *wrong in a direction that flatters the result*.

Two practical consequences:

- **A three-seed pass is coverage, not a conclusion.** Already the standing
  commitment; now it has a citation.
- **Report N beside every interval.** `stats.summarise_scores` returns it and
  the notebook warns when N < 10.

Note also that IQM below four runs degrades to the plain mean, because trimming
25% from each side of three values discards everything. At n = 3 there is no
robust estimator to be had.

---

## The limitation we do not have a way around

Agarwal et al.'s **stratified** bootstrap aggregates across tasks: Atari 100k
gives 26 tasks × 3 runs = 78 samples for uncertainty estimation. This project has
CartPole and FrozenLake - **2 tasks × 10 runs = 20**, and the scores are not
commensurable (CartPole returns run 0–500, FrozenLake success 0–1), so
aggregating would require a normalisation chosen by us.

`stats.stratified_bootstrap_ci` exists and is tested, but the honest default here
is **per-task interval estimates**. The aggregation machinery buys much less at
two tasks than it does at twenty-six, and inventing a normalisation to use it
anyway would add an arbitrary choice for no gain.

This is worth one sentence in the methodology chapter: the tooling is applied at
the level the data supports, not at the level the paper demonstrates.

---

## Checklist before a number enters the memoria

- [ ] computed with `final_performance`, not a maximum over training
- [ ] reported as IQM with a percentile bootstrap CI, not mean ± sd
- [ ] N stated next to the interval
- [ ] comparisons phrased as probability of improvement where the runs overlap
- [ ] per-task, unless a normalisation across tasks has been argued for
- [ ] the same statistic used for our numbers and for the paper's
      (`data/paper_ppo_*.csv` holds the per-seed scores, so this is possible)
