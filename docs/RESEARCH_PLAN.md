# OR-TS v2 — research plan (pre-registration)

Every simulation in this repository must appear here **before** it is run.
One section per hypothesis. Fill in "Result" only after the run exists under
`results/raw/`, and link the run id. Never rewrite a prediction after seeing the
outcome; add a dated amendment instead.

---

## H1 — Contrast memory beats absolute-rate memory under a drifting baseline

- **Config**: `sim/configs/drifting_baseline.json`
- **Claim**: when a common per-round shock moves every arm's logit while the
  relative arm effects are fixed, OR-TS accumulates less cumulative regret than
  Full-TS and Beta-Bernoulli TS.
- **Prediction**: mean final cumulative regret, OR-TS < Full-TS < Beta-TS, with
  the OR-TS/Beta-TS gap larger than 2 standard errors.
- **Falsified if**: OR-TS is not better than Beta-TS by more than 2 SE, or is
  worse than Full-TS.
- **Fixed in advance**: `n_reps = 20`, `n_rounds = 40`, `seed = 20260822`.
- **Result**: **Supported.** Run `20260831T120314Z_drifting_baseline_a25f7c5`:
  final mean cumulative regret OR-TS 1220.9 (se 62.8), Full-TS 7360.3
  (se 1979.6), Beta-TS 8104.7 (se 1422.7). Ordering as predicted; the
  OR-TS/Beta-TS gap is 4.8 SE (registered threshold 2). The Full-TS/Beta-TS
  gap is within noise, which the prediction did not require to be
  significant. Notably OR-TS under drift matches its own no-drift regret
  (1160.4), so the drift's whole cost falls on the level-memorizing
  policies.

## H1-control — The advantage comes from the drift, not from the estimator

- **Config**: `sim/configs/no_drift_control.json` (`sigma_drift = 0`)
- **Prediction**: with no common drift the three policies are broadly
  comparable; any OR-TS advantage is much smaller than under H1.
- **Falsified if**: OR-TS shows a comparable advantage with the drift switched
  off — that would mean the mechanism claimed in the paper is not what is
  driving the result, and the claim must be rewritten.
- **Result**: **Supported.** Run `20260831T120342Z_no_drift_control_a25f7c5`:
  OR-TS 1160.4 (se 44.3), Beta-TS 1174.4 (se 25.5), Full-TS 1210.2
  (se 50.8) — parity at 0.3 and 0.7 SE. The OR-TS advantage collapses from
  6883.8 under drift to 13.9 without it: the mechanism, not the estimator,
  drives H1.

## H2 — Laplace error does not accumulate across cycles at platform batch sizes

Motivated by an objection worth taking seriously: OR-TS carries a *Gaussian*
contrast state, so each cycle's approximate posterior becomes the next cycle's
prior. Beta-TS is conjugate and has no such error. Nobody has checked whether
the approximation error compounds over a long run.

- **Config**: `sim/configs/laplace_fidelity_common.json` (baseline 3%),
  `sim/configs/laplace_fidelity_rare.json` (baseline 0.5%)
- **Design**: `K = 2`, so the contrast `beta` is scalar and the exact marginal
  contrast recursion
  `q_t(beta) ∝ q_{t-1}(beta) ∫ L_t(beta, alpha) pi(alpha) d alpha`
  can be evaluated on a grid. That grid filter is the gold standard; the
  comparator is the shipped `LogisticBandit(odds_ratios_only=True)`. Both
  filters receive **identical data** under fixed 50/50 allocation, so any
  divergence is attributable to the approximation and not to different data.
  Both start from the same weakly informative prior, `beta ~ N(0, 2^2)` with a
  flat intercept. Batch sizes are swept within one run.
- **Claim**: at platform batch sizes the Gaussian state is accurate enough that
  the decision it produces is indistinguishable from the exact one, and the
  error does not grow with the cycle index.
- **Prediction**: at `n_per_round = 100000`, (a) the median absolute error in
  the winner probability `P(beta > 0)` at the final cycle is below `0.01`, and
  (b) median `KL(exact || laplace)` at cycle 39 is no more than `3x` its value
  at cycle 4 — error is bounded, not accumulating.
- **Falsified if**: at `n_per_round = 100000` the median KL grows by more than
  an order of magnitude from cycle 4 to cycle 39, or the final median
  `|Delta P(win)| > 0.01`. Either outcome means the objection lands and the
  manuscript must say so.
- **Also reported, with no prediction attached**: the batch size at which the
  final median `|Delta P(win)|` first exceeds `0.05`, under both the common and
  the rare baseline. This is exploratory — it is where OR-TS is expected to be
  *worse* than Beta-TS, and it is reported whichever way it comes out.
- **Fixed in advance**: `n_reps = 10`, `n_rounds = 40`, `seed = 20260822`,
  `batch_sizes = [100000, 10000, 1000, 100]`, beta grid `4001` points,
  alpha grid `801` points.
- **Scope**: this tests filter fidelity under a fixed allocation. It does not
  test whether approximation error feeds back through adaptive allocation;
  that needs a separate design and is not claimed here.
- **Result**: **H2 supported.** Runs
  `20260822T134851Z_laplace_fidelity_common_a21b6b1` (3% baseline) and
  `20260822T135132Z_laplace_fidelity_rare_a21b6b1` (0.5% baseline).

  Common baseline (3%):

| batch | med KL cyc 4 | med KL cyc 39 | growth | med \|ΔP(win)\| final | max \|ΔP(win)\| final |
|---|---|---|---|---|---|
| 100,000 | 1.59e-07 | 7.97e-08 | 0.50 | 1.11e-16 | 7.77e-16 |
| 10,000 | 1.14e-06 | 3.65e-07 | 0.32 | 3.26e-04 | 5.16e-03 |
| 1,000 | 2.46e-05 | 1.97e-06 | 0.08 | 6.26e-03 | 8.00e-03 |
| 100 | 4.25e-03 | 2.32e-03 | 0.54 | 2.47e-02 | 5.22e-02 |

  Rare baseline (0.5%):

| batch | med KL cyc 4 | med KL cyc 39 | growth | med \|ΔP(win)\| final | max \|ΔP(win)\| final |
|---|---|---|---|---|---|
| 100,000 | 9.46e-07 | 3.26e-07 | 0.35 | 5.56e-05 | 7.41e-04 |
| 10,000 | 4.38e-05 | 8.47e-06 | 0.19 | 4.91e-03 | 1.20e-02 |
| 1,000 | 3.22e-03 | 1.02e-03 | 0.32 | 1.35e-02 | 6.46e-02 |
| 100 | 2.44e-02 | 1.63e-02 | 0.67 | 5.28e-02 | 1.36e-01 |

  Both predictions hold at `n = 100000`: final median `|ΔP(win)|` is
  1.1e-16 (common) and
  5.6e-05 (rare), both far below
  `0.01`; the KL growth ratio is
  0.50 and
  0.35, both below `1`.

  **The KL growth ratio is below 1 at every batch size tested, in both runs.**
  The approximation error does not accumulate across cycles; it shrinks as
  evidence accumulates. This is the opposite of what the objection predicts,
  and it is the main result.

  Two honest qualifications. First, at `n = 100000` the winner probability
  saturates at 1 within a few cycles, so `|ΔP(win)|` there is at machine
  epsilon and is not an informative measure — KL is. Second, the measured
  Laplace error at `n = 100000` is at or below the gold standard's own
  resolution (see H2-grid), so the correct statement is that it is smaller
  than this design can resolve, not that it is exactly the number printed.

  Exploratory, no prediction attached: final median `|ΔP(win)|` never exceeds
  `0.05` at any batch size tested under the common baseline (largest value
  2.5e-02 at `n = 100`), but under the
  rare baseline it reaches 5.3e-02 at
  `n = 100`, with a worst repetition of
  1.4e-01. The Gaussian state is
  materially wrong in the small-batch, rare-event corner. That corner is where
  Beta-TS is expected to be the safer choice, and the manuscript should say so
  rather than leave it to a reviewer.

## H2-grid — The gold standard is converged

- **Config**: `sim/configs/laplace_fidelity_gridcheck.json`
- **Claim**: the grid filter is a valid gold standard, i.e. its own
  discretisation error is far below the Laplace error it is used to measure.
- **Prediction**: doubling both grids (`8001 x 1601`) changes the reported
  final `P(beta > 0)` of the *exact* filter by less than `1e-4`.
- **Falsified if**: the change exceeds `1e-4`, in which case H2's numbers are
  grid-limited and must be recomputed before they are used.
- **Result**: **H2-grid supported, as worded.** Run
  `20260822T135253Z_laplace_fidelity_gridcheck_61007e6`, compared against the
  H2 common run over the 3 trajectories that saw
  identical data (120 cells, `n = 100000`).
  Doubling both grids changes the exact filter's **final** `P(beta > 0)` by at
  most `1.11e-16`, far below the `1e-4` threshold.

  Reported alongside, because the prediction did not cover it: the maximum
  difference over *all* cycles is `5.36e-04`, and it is
  concentrated almost entirely in cycle 0, where the posterior is widest. It
  decays to `3e-8` by cycle 10 and to machine epsilon by cycle 30 — the
  recursion is contractive, so early grid error is forgotten rather than
  carried. The consequence for H2 is stated in its Result: the cycle-0 numbers
  carry roughly `5e-4` of grid uncertainty, the final-cycle numbers do not.


## H3 — On public logged data, the common level moves and the arm contrasts do not

The working model `logit(p_{i,t}) = alpha_t + beta_i` is assumed throughout the
paper, and the only empirical support for it is the 2020 advertising replay,
whose data are proprietary and cannot be redistributed. Nobody has checked the
assumption on data a reader can download. This hypothesis does that.

- **Data**: Open Bandit Dataset (Saito et al. 2020), ZOZOTOWN, CC BY 4.0.
  **ALL campaign, `random` policy bucket only**: 1,374,327 impressions, 80
  items, 7 days, CTR 0.35%. The random bucket is used because items are
  assigned uniformly at random, so an item's exposure is not confounded with
  the logging policy. The Bernoulli TS bucket is deliberately excluded: there
  exposure is a function of the policy's belief, which is exactly the
  confounding this test must avoid.
- **Design**: batch = calendar day (7 batches); arms = all 80 items; position
  enters as a fixed effect. Counts are aggregated to (day, item, position) and
  two nested logistic models are fitted:
  - `M1: logit p = alpha_t + beta_i + gamma_pos`  — contrast fixed across batches
  - `M2: logit p = alpha_t + beta_i + delta_{i,t} + gamma_pos` — contrast free per batch
  The nested-model test is used rather than per-cell empirical contrasts
  because it pools across sparse cells instead of relying on each one.
- **Claim**: the arm contrasts are stable across batches while the common level
  is not, so M2 buys nothing over M1.
- **Prediction**: (a) the likelihood-ratio statistic for M2 against M1 does not
  exceed its degrees of freedom by more than two standard deviations of the
  corresponding chi-square; and (b) the standard deviation of the fitted
  `alpha_t` across the 7 days exceeds the excess — beyond-sampling — standard
  deviation of the per-batch contrasts by a factor of at least 3.
- **Falsified if**: the interaction is significant at this resolution, or the
  alpha-to-excess-beta SD ratio is below 3. Either outcome means the
  fixed-contrast specification does not describe this dataset, and the
  assumptions section of the paper must say so rather than assume otherwise.
- **Power, stated before running**: this bucket is thin. Per item-day there are
  about 2,450 impressions and about 8.6 clicks, so a *per-cell* empirical
  contrast carries a standard error near 0.5. A null result here is therefore
  weak evidence, not strong evidence, and must be reported as such. The
  powered version of this test needs the Yahoo R6A random bucket (about 36M
  events over 7 days, uniformly random article selection, restricted to the F1
  story position), which requires a Webscope agreement; H3 is run on ZOZO
  first because it is downloadable today and exercises the whole path.
- **Known reason the prediction could fail**: item-specific drift. Fashion
  inventory turns over and an item's appeal is not constant across a week. If
  the contrasts do move, the honest reading is that this domain needs the decay
  control or a dynamic `beta_t`, both of which the paper already specifies.
- **Fixed in advance**: campaign `ALL`, policy `random`, batch = calendar day,
  all 80 items with no filtering by volume, position as a fixed effect.
- **Result**: **Falsified, both criteria.** Run
  `20260830T084633Z_obd_contrast_stability_cd40101`. (a) LR = 921.2 on
  df = 474, z = +14.5 — the day-by-item interaction is overwhelmingly
  significant. (b) `sd(alpha_t) = 0.184` against an excess contrast SD of
  `0.321`: ratio 0.57, on the wrong side of 3 and below 1 — on this dataset
  the contrasts move almost twice as much as the common level. A ridge-free
  check on empirical per-cell log-odds contrasts reproduces the excess
  variance (0.127), so the verdict is not an artifact of the penalised fit.
  The named failure reason is what happened: item-specific drift, including
  a structured mean trend of +0.046/day in the contrasts. Consequence for
  the paper: ZOZO is reported as the case where the fixed-contrast
  specification fails and the decay / dynamic-`beta_t` extensions are the
  appropriate reading; it is not usable as support for the basic
  specification. The shared-shock claim still lacks a public powered test
  (R6A pending).



## H4 — Across 78 real A/B experiments, does the level move more than the contrast?

H3 asked this on one fashion-recommendation log and was falsified: on ZOZO the
contrasts moved about twice as much as the common level. H3 cannot be rewritten,
so this is a separate hypothesis on different data. It is the version the paper
actually needs, because here the arms *are* experiment variants rather than a
stand-in for them, and because 78 experiments turn the question into a
distribution rather than a single verdict.

- **Data**: ASOS Digital Experiments Dataset (Liu, Cardoso et al., NeurIPS 2021
  Datasets & Benchmarks; https://osf.io/64jsb/), single CSV, sha256
  `aa0cc3f8304fac545f232ec0bc363373f932919f6e72e28908958bc10651ce22`.
  78 online controlled experiments, 2--5 variants, 4 metrics, 12-hourly
  checkpoints **cumulative from experiment start**, aggregated at group level:
  `count_c, count_t, mean_c, mean_t, variance_c, variance_t`.
- **What has been inspected before registering**: schema, metric types, series
  lengths and sample sizes only. `metric_id = 1` is exactly binary
  (`variance == mean(1-mean)` to machine precision) and is the metric used
  here. Nothing about alpha or beta has been computed.
- **Design**: consecutive checkpoints are differenced to per-period counts.
  For each series (experiment x treatment variant) and period t:
  `alpha_t = logit(p_control,t)` and `beta_t = logit(p_treat,t) - logit(p_control,t)`,
  matching the paper's reference parameterisation with control as reference.
  Sampling variance of `beta_t` by the delta method,
  `1/s_c + 1/(n_c - s_c) + 1/s_t + 1/(n_t - s_t)`. Excess variance is
  `max(0, Var_t(beta_t) - mean_t sampling variance)`. The statistic is
  `R = sd(alpha_t) / sqrt(excess variance of beta)`.
- **Fixed in advance**: metric 1 only; periods dropped when the differenced
  `n <= 0` or successes fall outside `(0, n)`; series kept only with at least
  10 usable periods; no other filtering; every retained series reported.
- **Claim**: in real A/B experiments the common level is the volatile part and
  the treatment contrast is comparatively stable, so `R` is well above 1.
- **Prediction**: median `R` across retained series is at least 3, and `R > 1`
  in at least 70% of them.
- **Falsified if**: median `R` is below 3, or fewer than 70% of series have
  `R > 1`. On ZOZO the same statistic was 0.57; a value near or below 1 here
  would mean the fixed-contrast specification is not the right default for
  online experiments either, and Sections 6 and 8 must be rewritten to say so.
- **Secondary, registered now so it cannot be chosen later**: the identical
  computation on the linear scale, with `beta_t = p_treat,t - p_control,t` and
  its own delta-method sampling variance. Reported alongside the logit result
  whichever way it comes out. This is the empirical form of "why odds ratios?"
  and the answer is allowed to be "neither scale is more stable".
- **Result**: **Supported on both primary criteria; the secondary is
  unanswerable from this data.** Run
  `20260830T090754Z_asos_level_vs_contrast_9a7c7e9`, 86 series over 78
  experiments, 5,516 periods. Excess `sd(alpha) = 0.341` against excess
  `sd(beta) = 0.0125`; **median R = 25.1** (registered threshold 3),
  quartiles 10.6 / 19.2 / 37.3; `R > 1` in **100%** of series (registered
  70%), `R > 3` in 97.7%. Descriptive, not registered: lag-1
  autocorrelation is 0.481 for `alpha` and -0.033 for `beta` — the level
  carries memory between periods, the contrast is a constant plus sampling
  noise. The scale comparison returns nothing usable (logit 25.12 vs linear
  25.17, logit larger in 56% of series) because the effects are tiny:
  median `|beta_logit| = 0.0032`, an odds ratio of 1.003. That outcome was
  admitted in advance. **Limitation**: this shows small contrasts do not
  drift while the level does; it does not show that large contrasts are
  stable, since there are almost none here. H3 (ZOZO, R = 0.57, large item
  contrasts that did drift) is the counterweight and the two must be
  reported together.



## H5 — When the contract fails, does OR-TS lose to Beta-TS, or merely tie it?

H3 established that arm-specific drift is real in at least one public log,
and Section 4 wants to argue an insurance asymmetry: the contract holding is
a disaster for Beta-TS, while the contract failing costs OR-TS little. The
second half of that sentence has never been checked, and it is not obvious.
With `lambda = 0` OR-TS keeps accumulating precision on stale contrasts, so
under contrast drift it could in principle be *worse* than Beta-TS, whose
absolute-rate memory is equally stale but less confident. This hypothesis
tests that cell of the matrix before the paper is allowed to claim it.

- **Config**: `sim/configs/arm_drift_robustness.json` (new experiment
  `sim/experiments/arm_drift.py`; the pre-registered H1 configs must not be
  edited, and the H1 experiment has no arm-specific term).
- **Design**: identical to H1's loop — 10 arms, 40 rounds, 100{,}000 trials
  per round, baseline 3\%, arm effects `k * 0.05` on the logit scale,
  probability-matching allocation, regret against the round's realised best
  arm — except the disturbance. Instead of a common shock,each round adds an
  independent per-arm random walk step `N(0, sigma_arm^2)` to each arm's
  own logit, so the *contrasts* drift and the common level is flat. Two
  severities are swept in one run: `sigma_arm = 0.02`, matched to the ZOZO
  estimate (the H3 run measured a mean contrast trend of about +0.046/day
  with day-level batches; 0.02 per cycle is the same order), and
  `sigma_arm = 0.06`, roughly the H1 common-shock energy split across arms.
  Policies: Beta-TS, Full-TS, OR-TS with `lambda = 0`, and OR-TS with
  `lambda = 0.1` (the paper's own prescription for this regime, effective
  memory about 10 batches).
- **Fixed in advance**: `n_reps = 20`, `n_rounds = 40`, `seed = 20260831`;
  final-round mean cumulative regret with standard errors over repetitions
  is the reported quantity, aggregated over all repetitions as always.
- **Claim**: contrast drift hurts every policy that remembers contrasts,
  and it hurts them all about equally; the insurance premium of OR-TS is
  small even where its assumption is wrong.
- **Prediction**: at both severities, mean final cumulative regret of
  OR-TS(`lambda = 0`) is within 2 SE of Beta-TS's, or better. Secondary,
  stated for honesty: OR-TS(`lambda = 0.1`) is expected to beat
  OR-TS(`lambda = 0`) at the higher severity — that is what Section 5 now
  prescribes — and if it does not, Section 5's guidance overstates decay.
- **Falsified if**: OR-TS(`lambda = 0`) is worse than Beta-TS by more than
  2 SE at either severity. In that case the insurance framing is wrong and
  Section 4 must instead report the measured cost of running OR-TS outside
  its contract, with the decay setting as the mitigation.
- **Result**: **Supported as registered.** Run
  `20260831T120441Z_arm_drift_robustness_d11f158`. At `sigma_arm = 0.02`
  OR-TS(0) 4091.1 (se 830.0) vs Beta-TS 3859.3 (se 785.8) — 0.2 SE apart.
  At `0.06` OR-TS(0) 26219.8 (se 6570.3) vs Beta-TS 18247.8 (se 4051.4) —
  1.0 SE, within the registered bound, but the point estimate leans 44%
  against OR-TS(0) and the run cannot resolve a 1-SE gap; the paper must
  say "ties within noise at the calibrated severity, point estimate favors
  Beta-TS at triple severity" rather than claim parity outright. Secondary
  prediction supported strongly: OR-TS(0.1) beats OR-TS(0) at 0.06 by
  2.7 SE (7936.7 vs 26219.8) — and also beats Beta-TS there (2.5 SE), so
  under contrast drift the best policy in this run is OR-TS configured as
  Section 5 prescribes, not the conjugate default.


## H6 — Selective forgetting versus indiscriminate forgetting

H5 compared OR-TS with decay against a Beta-TS with no forgetting device at
all, which confounds two things: the contrast-state representation and the
mere possession of a forgetting knob. Discounted Thompson sampling (Raj and
Kalyani 2017, already cited in Section 8) is the standard knob for the
conjugate state, and Section 9's future-work list has admitted this missing
baseline since the manuscript was drafted. This hypothesis deconfounds it.

- **Config**: `sim/configs/forgetting_comparison.json` (new experiment
  `sim/experiments/forgetting_comparison.py`; H1 and H5 configs are already
  run and must not be edited).
- **Design**: the H1/H5 loop unchanged — 10 arms, 40 rounds, 100{,}000
  trials per round, baseline 3\%, effects `k * 0.05`, probability
  matching. Three disturbances in one run: common shock sd 0.30 (H1's),
  per-arm random walk sd 0.02, per-arm sd 0.06 (H5's). Four policies:
  Beta-TS; discounted Beta-TS shrinking both counters toward the uniform
  prior by the factor `1 - d` each cycle with `d = 0.1`, matched to OR-TS's
  decay so the two forgetting devices have the same geometric memory;
  OR-TS `lambda = 0`; OR-TS `lambda = 0.1`.
- **Fixed in advance**: `n_reps = 20`, `n_rounds = 40`, `seed = 20260901`;
  the reported quantity is final-round mean cumulative regret with standard
  errors over repetitions.
- **Claim**: forgetting helps any policy under drift, but *selective*
  forgetting (drop the level, keep the contrasts) beats *indiscriminate*
  forgetting (discount everything) exactly where the state separation
  holds, and only there.
- **Prediction**: (a) under the common shock, discounted Beta-TS improves
  on plain Beta-TS yet still trails OR-TS(`lambda = 0`) by more than 2 SE,
  because discounting throws away contrast evidence that never went stale;
  (b) under per-arm drift at both severities, discounted Beta-TS and
  OR-TS(`lambda = 0.1`) are within 2 SE of each other.
- **Falsified if**: (a) fails — discounted Beta-TS matches OR-TS under the
  common shock. That would mean H1's advantage is mostly a forgetting
  effect rather than a representation effect, and the paper's central
  claim must be cut down accordingly, stated as such in Sections 4 and 9.
  If (b) fails with discounted Beta-TS clearly ahead, Section 5's decay
  prescription is inferior to plain discounting in the drift regime and
  must say so.
- **Result**: **Supported on both registered predictions.** Run
  `20260831T122242Z_forgetting_comparison_e613b0c`. (a) Under the common
  shock, discounted Beta-TS at 6461.5 (se 974.6) trails OR-TS(0) at 1246.4
  (se 53.5) by 5.3 SE — the forgetting knob does not rescue the conjugate
  state. The subordinate guess that discounting would at least improve on
  plain Beta-TS (6317.1) was wrong (−0.1 SE): the registered shock is
  redrawn each batch, so there is no drift rate for a discount to track,
  and OR-TS wins by not carrying the level rather than by forgetting it
  better. (b) Under per-arm drift, discounted Beta-TS and OR-TS(0.1) are
  within 2 SE at both severities (−0.1 SE at 0.02; 0.9 SE at 0.06).
  Descriptive extras: decay's premium where the separation holds is large
  (OR-TS(0.1) 2475.6 vs OR-TS(0) 1246.4, 9.2 SE), and H5's 1.0-SE lean
  against OR-TS(0) at arm 0.06 reverses sign in this seed (−0.3 SE), so the
  paper must call that comparison indistinguishable rather than leaning on
  either run.


## H7 — The insurance claim on real traffic: an OBD replay

H5 and H6 priced the policies in synthetic environments; the advertising
replay in the manuscript is real but proprietary. This run closes the loop
with a policy evaluation on public real traffic, in the regime where H3
showed the state separation *fails* — so it tests the half of the insurance
argument that matters there: that running OR-TS where its assumption is
wrong costs nothing detectable.

- **Config**: `sim/configs/obd_replay.json` (new experiment
  `sim/experiments/obd_replay.py`).
- **Data**: the committed aggregate `sim/data/obd_random_all_daily.csv`
  (sha pinned in the config), ALL campaign, uniformly-random bucket.
  Because the logging policy is uniform random, the day-by-item CTRs it
  yields are unbiased estimates of each item's rate on each day; pooled
  over the three positions they define the replay environment, exactly the
  construction the 2020 advertising study used. The environment is treated
  as fixed; replication randomness is binomial outcome noise and policy
  sampling only.
- **Design**: 7 daily batches, 80 arms, each day's total traffic equal to
  the actual random-bucket volume that day. Day 1 is allocated uniformly;
  thereafter each policy allocates by its own winner probabilities.
  Policies: Beta-TS, discounted Beta-TS (d = 0.1), OR-TS (lambda = 0),
  OR-TS (lambda = 0.1). Reported quantity: total expected clicks over the
  week (allocation times true environment rate, so outcome noise does not
  enter the metric), and cumulative regret against each day's realised
  best item.
- **Fixed in advance**: `n_reps = 20`, `seed = 20260902`, no filtering of
  items, position pooled.
- **Power, stated before running**: seven batches, six adaptive decisions,
  80 arms at a 0.35% CTR is a short, hard horizon. The honest expectation
  is that no policy separates; that is also what the claim predicts, so a
  null here is consistent but weak, and is to be reported as such.
- **Claim**: on real traffic from an inventory-like arm set, the choice of
  state representation is inconsequential at this horizon.
- **Prediction**: every pairwise difference in total expected clicks among
  the four policies is within 2 SE across repetitions.
- **Falsified if**: any policy separates from another by more than 2 SE.
  Either direction is informative and gets reported: OR-TS(0) losing would
  be the first real-data measurement of the cost of the wrong assumption;
  OR-TS winning would be evidence the contrast state helps even where the
  separation fails.
- **Result**: **Falsified as worded; the substance survives.** Run
  `20260831T123938Z_obd_replay_9a33b2b`. One of six pairs crosses 2 SE:
  plain Beta-TS beats *discounted* Beta-TS by 380.9 clicks (+2.1 SE) —
  forgetting 10%/day of ~8 clicks per item-day costs real information on a
  seven-batch horizon. Every OR-TS comparison is within noise, and the one
  the insurance argument turns on, OR-TS(0) vs Beta-TS, is +0.1 SE (34.8
  clicks in ~7,000). Six pairwise tests carried no registered multiplicity
  correction, so a lone 2.1-SE excursion is unremarkable, but the verdict
  stands as worded. The manuscript reports the falsification first, then
  the reading: on real inventory traffic the representation is free, and
  the only measurable mistake at this horizon was forgetting too fast, in
  either family.

---

## H8 — Class-1 replay on real experiments: what a carried level costs on ASOS

*Registered 2026-09-04. Not yet run.* H7 replayed the policies where the
state separation fails (class 2). H4 showed that real A/B experiments live
in class 1 — the level moves, the contrasts do not — but no policy has been
run on that data. This closes the class-1 cell of the Section 4 taxonomy
with real traffic.

- **Config**: `sim/configs/asos_replay.json` (new experiment
  `sim/experiments/asos_replay.py`), to be written after this entry.
- **Data**: the same ASOS CSV as H4 (sha pinned there), metric 1, cumulative
  checkpoints differenced to per-period counts exactly as in H4, same
  series filter (at least ten usable periods). Because allocation in an
  A/B test is fixed and randomised, each period's per-variant rates are
  unbiased estimates of that period's environment; they define the replay
  environment, as the OBD random bucket did in H7. The environment is
  treated as fixed; replication randomness is binomial outcome noise and
  policy sampling only.
- **Design**: one replay per experiment, arms = control plus its variants
  (2–5 arms), one batch per twelve-hourly period, each period's total
  traffic equal to the actual total across variants that period. Period 1
  is allocated uniformly; thereafter each policy allocates by its own
  winner probabilities. Policies: Beta-TS, discounted Beta-TS (d = 0.1),
  OR-TS (lambda = 0), OR-TS (lambda = 0.1).
- **Two reported quantities, both fixed now**:
  1. *Regret*: cumulative expected regret against each period's realised
     best arm, summed over the experiment, then averaged across
     experiments (each experiment weighted equally).
  2. *Level-tracking error*: after each period t, the absolute error, on
     the logit scale, between the policy's belief about the control arm's
     rate and the control arm's realised rate in period t+1. For Beta-TS
     the belief is the posterior mean of its cumulative Beta state; for
     OR-TS it is the fresh intercept fitted on period t, i.e. a one-period
     forecast. Reported as the median over periods, then over experiments.
- **Fixed in advance**: `n_reps = 20`, `seed = 20260904`, no filtering of
  experiments beyond H4's, equal weighting across experiments.
- **Power, stated before running**: H4 measured a median absolute contrast
  of 0.003 logit. Arms are therefore nearly interchangeable and the regret
  of any policy is close to zero; quantity 1 is expected to be a null and
  is registered so that it cannot be dropped if it is one. The level, by
  contrast, moved with a median excess sd of 0.34 and positive
  autocorrelation, so quantity 2 is where a difference can show.
- **Claim**: on real class-1 traffic the representation is free in regret
  terms and the carried level is measurably stale.
- **Prediction**: (1) every pairwise difference in mean regret among the
  four policies is within 2 SE; (2) Beta-TS's median level-tracking error
  exceeds OR-TS(0)'s in at least 70% of experiments, and the median ratio
  across experiments is at least 2.
- **Falsified if**: (1) any policy separates in regret by more than 2 SE —
  OR-TS losing would be the first real-data cost of the fresh intercept on
  the data where its bet holds; or (2) Beta-TS's tracking error exceeds
  OR-TS's in fewer than 70% of experiments or the median ratio is below 2,
  which would mean the level moves too little between periods for a
  cumulative state to be caught out, and the paper's "must unlearn"
  language must be softened.
- **Not registered, and not to be added later**: any semi-synthetic
  variant that injects a larger contrast into ASOS traffic. If such a
  study is wanted it gets its own entry before it is run.
- **Result**: **Prediction 1 falsified as worded and the metric that
  falsified it does not support the reverse claim; prediction 2 supported.**
  Run `20260907T083417Z_asos_replay_ded2091` (71 experiments, 2-4 arms,
  4,379 twelve-hourly batches, 20 repetitions).

  *Regret.* Five of six pairwise differences exceed 2 SE, all with OR-TS
  ahead: Beta-TS minus OR-TS(0) is +1,182.5 events per experiment (10.3 SE),
  Beta-TS minus OR-TS(0.1) +1,010.1 (8.8 SE), discounted Beta-TS minus
  OR-TS(0) +1,489.7 (15.3 SE), discounted Beta-TS minus OR-TS(0.1) +1,317.4
  (17.7 SE), OR-TS(0.1) minus OR-TS(0) +172.4. Only the two Beta variants tie
  (-307.2, 2.0 SE). The registered prediction of a null is therefore
  falsified, and OR-TS is on the favourable side of it.

  That direction is not a finding. The registered aggregation is the
  equally-weighted mean of per-experiment regret *totals* in raw event units,
  and one experiment carries it: `81761c` contributes 81,858 of the 83,958
  total difference between Beta-TS and OR-TS(0), or 97.5%. The median
  per-experiment difference is -18.7, in Beta-TS's favour, and Beta-TS is the
  worse policy in 33 of 71 experiments, fewer than half. Nothing here
  supports a general regret advantage for either family; the registered
  statistic is dominated by scale, which the entry did not anticipate and
  which is being reported rather than replaced.

  For the record and not as a result: `81761c` has the smallest mean contrast
  of the 71 experiments (0.0000 on the logit scale, rank 71 of 71), 3 arms,
  113 periods and a median 91,277 trials per period. Genuinely tied arms plus
  a moving level plus large batches is the configuration in which a carried
  level can lock a policy onto an arm chosen by level noise and pay for it in
  event units. That mechanism is the manuscript's thesis, but at n = 1 it is
  an observation, not evidence. Descriptive floors for the same environments:
  a fixed uniform allocation averages 12,491.8 events of regret and the
  hindsight-best fixed arm 7,882.5, against 9,370.0 for OR-TS(0) and 10,552.5
  for Beta-TS.

  *Level tracking.* Supported on both criteria. Beta-TS's median tracking
  error exceeds OR-TS(0)'s in 88.7% of the 71 experiments (registered
  threshold 70%) and the median ratio across experiments is 2.78 (threshold
  2). Median absolute one-period forecast error of the control arm's log
  odds: Beta-TS 0.364, discounted Beta-TS 0.259, OR-TS(0) 0.150, OR-TS(0.1)
  0.149. The carried level is measurably stale on real class-1 traffic, and
  discounting recovers about half of the gap.

  *Note on execution.* Each repetition draws from its own spawned child
  generator and completed repetitions are cached under `tmp/`, because the
  machine that ran this could not hold a single process for the full
  duration. The cached set is what an uninterrupted run would have produced;
  a repetition's stream does not depend on how many preceded it.
  `results/raw/20260907T081859Z_asos_replay_4fa7d1b/` is an earlier attempt
  that was killed before writing results, kept with a NOTES.md as raw
  directories are append-only.

## H9 — The premium at the origin

Section 3 of the manuscript claims that when nothing moves the costs of
every policy are "fixed and small" and that decay "only discards good
evidence". The H6 run priced decay under three disturbances but never with
none, and the H1-control run predates the two forgetting variants, so the
undisturbed row of the regret table has two blanks and the claim is
unmeasured. This run fills the blanks with the H6 loop and no disturbance.

- **Config**: `sim/configs/forgetting_comparison_origin.json` — the H6
  experiment `sim/experiments/forgetting_comparison.py` unchanged, with
  `disturbances = [["common", 0.0]]` (a zero-sd common shock is no shock),
  the same four policies, the same `forget = 0.1`, seed 20260902, 20 reps,
  40 rounds, 100,000 trials per round, ten arms, baseline 3%, effects
  k·0.05 on the logit scale. Nothing else edited.
- **Claim**: with nothing moving, forgetting can only discard good
  evidence, so each forgetting variant pays a premium over its
  unbounded-memory counterpart, and that premium is small next to the
  common-shock gap of Table 3 (about 5,000 regret).
- **Prediction**: OR-TS(λ=0.1) above OR-TS(0), and discounted Beta-TS above
  Beta-TS, each by more than 2 se; each premium below 1,000 regret.
- **Falsified if**: either premium is within 2 se of zero (decay is free at
  the origin and the "insurance premium" framing overstates its cost), or
  either premium exceeds 1,000 (decay is not cheap at the origin, and the
  manuscript's "fixed and small" must be withdrawn).
- **Secondary, descriptive**: Beta-TS and OR-TS(0) should agree with the
  H1-control run within one se; reported either way.
- **Use in the manuscript**: the undisturbed row of the regret table and the
  first panel of the price-of-each-bet figure come from this run; the
  Section 3 sentence on the origin quotes the premium.
- **Result**: **Half supported, half falsified, and reported as such.** Run
  `20260904T215054Z_forgetting_comparison_origin_8e22c5f`. Final regret
  (mean ± se, 20 reps): Beta-TS 1,157 ± 35; discounted Beta-TS 2,161 ± 42;
  OR-TS(0) 1,152 ± 50; OR-TS(0.1) 2,226 ± 58. Both premiums are far above
  2 se (Beta family 1,005, 18 se; OR-TS family 1,074, 14 se), so forgetting
  is not free at the origin. But both exceed the registered 1,000 line
  (by 0.5% and 7%): decay with nothing moving roughly **doubles** the
  undisturbed regret, and the manuscript's "fixed and small" is withdrawn in
  favour of "fixed, about the size of the undisturbed regret itself, a fifth
  of the common-shock gap". Secondary: Beta-TS and OR-TS(0) agree with
  H1-control within 0.4 se and 0.1 se. Descriptive end-state: the best arm
  holds 99.7% / 99.6% of final traffic without decay and 93.4% / 92.1% with it.


## H10 — Both move: the level and the contrasts together

Figure 5's per-arm panels hold the level fixed by construction, and a
reader can conclude from them that discounted Beta-TS is as good as OR-TS
with decay wherever the contrasts move. No real dataset in Section 4.1 has
a fixed level: on the Open Bandit log the contrasts moved and the level
kept moving too. This run supplies the missing environment.

- **Config**: `sim/configs/forgetting_comparison_both.json` — the H6
  experiment with a new disturbance kind `both`: the per-arm random walk at
  step sd 0.06 (H6's stronger severity) and, on top of it, a common shock of
  sd 0.30 redrawn every cycle (H6's shock), via the new parameter
  `both_common_sigma`. Same four policies, `forget = 0.1`, seed 20260903,
  20 reps, otherwise the H6 loop. The experiment file gains the `both`
  branch only; the H6 and H9 configs are untouched and their behaviour is
  unchanged.
- **Claim**: when both coordinates move, a state that carries no level and
  forgets its contrasts pays only the second cost, whereas a state that
  forgets everything still chases the level.
- **Prediction**: OR-TS(λ=0.1) beats discounted Beta-TS by more than 2 se,
  and OR-TS(0) beats Beta-TS by more than 2 se. Descriptively, OR-TS(0.1)
  should land near its per-arm-0.06 regret (about 8,000), since the level
  costs it nothing.
- **Falsified if**: either comparison is within 2 se or reversed. That
  would mean forgetting everything is as good as forgetting selectively
  even when the level moves, and the insurance argument would lose its
  second half.
- **Use in the manuscript**: a fifth panel of Figure 5 and a fifth row of
  the regret table; Section 4.2 gains the sentence that the fixed-level
  panels correspond to no real dataset.
- **Result**: **Supported on both criteria.** Run
  `20260904T222823Z_forgetting_comparison_both_a34ce49`. Final regret
  (mean ± se, 20 reps): Beta-TS 35,386 ± 5,608; discounted Beta-TS
  28,895 ± 3,968; OR-TS(0) 21,342 ± 3,364; OR-TS(0.1) 8,183 ± 905.
  OR-TS(0.1) beats discounted Beta-TS by 20,713 (5.1 se) and OR-TS(0) beats
  Beta-TS by 14,044 (2.1 se). As predicted descriptively, OR-TS(0.1) lands
  at its per-arm-0.06 regret (7,990 in H6): the level costs it nothing.
  End-state: the best arm holds 60% of final traffic under OR-TS(0.1)
  against 30% under Beta-TS and 43% under discounted Beta-TS.


## H11 — Where Beta-TS's tolerance for a moving level ends

Section 4.1 measures how much the level moves; Sections 3 and 4.2 say
Beta-TS pays for it. Neither says how much movement Beta-TS can absorb, or
why. A first-order argument gives a boundary and a mechanism. Each Beta-TS
arm posterior is a cumulative average, so a common logit shift Δ in one
batch distorts the apparent contrast between two arms by about
Δ·(w_lag − w_lead), where w is the share of an arm's cumulative exposures
that the latest batch contributes. Under balanced allocation the two
shares are equal and the distortion cancels to first order; under the
imbalance a bandit creates, w_lead is small and w_lag approaches one, so
the distortion approaches Δ itself and the ordering flips once Δ is of the
order of the contrast. The boundary is therefore "level shift per batch ×
allocation imbalance ≲ contrast", which for a bandit reduces to
τ_α ≲ |β|. This run tests both halves: the boundary and the mechanism.

- **Config**: `sim/configs/level_tolerance.json` — the H6 experiment with
  the common shock swept over sd ∈ {0, 0.02, 0.05, 0.1, 0.3} (the effect
  gap between adjacent arms is 0.05, so the sweep brackets it), run twice:
  under the usual adaptive allocation and under fixed 1/K allocation every
  round (new parameter `allocations`; the default path of the experiment
  is unchanged and reproduces H6/H9/H10 bit for bit). The four H6 policies,
  `forget = 0.1`, seed 20260907, 20 reps, 40 rounds, 100,000 trials per
  round, ten arms, baseline 3%. Two outcomes per policy and cell: the share
  of the final cycle's traffic on the true best arm (adaptive only, as in
  H9/H10), and the policy's own posterior probability after the final
  update that the true best arm is best (`win_prob_best`, the
  identification metric that is meaningful under fixed allocation).
- **Claim**: Beta-TS's exposure to a moving level is the product of the
  movement and the allocation imbalance, and becomes material when the
  per-batch shift reaches the contrast scale.
- **Prediction**, adaptive allocation, final best-arm share, OR-TS(0)
  minus Beta-TS: within 2 se at sd 0 and 0.02 (below the contrast scale),
  and beyond 2 se in OR-TS's favour at sd 0.1 and 0.3 (above it). The
  crossing is expected between 0.02 and 0.1; sd 0.05 is reported either way
  and is not a criterion.
- **Prediction**, fixed allocation, final `win_prob_best`, OR-TS(0) minus
  Beta-TS: within 2 se at every sd, including 0.3. With equal cumulative
  exposures the shift cancels to first order, so Beta-TS should identify
  the best arm as well as OR-TS however far the level moves.
- **Falsified if**: the adaptive difference exceeds 2 se already at sd
  0.02 (Beta-TS breaks well below the contrast scale and the boundary is
  too generous), or fails to exceed 2 se at sd 0.1 (the boundary is too
  strict), or the fixed-allocation difference exceeds 2 se at any sd
  (imbalance is not the mechanism; the level hurts Beta-TS's inference
  directly).
- **Secondary, descriptive**: OR-TS(0)'s final share and `win_prob_best`
  are flat across the sweep; the two discounted variants are reported for
  completeness and carry no prediction.
- **Use in the manuscript**: if supported, Section 3's "up the level axis"
  gets the boundary in one sentence and Section 4.1 reads ASOS and Open
  Bandit against it (ASOS level 0.34 against contrasts of 0.003; Open
  Bandit level 0.18 against contrasts of order 0.3, where Section 4.3
  found Beta-TS and OR-TS tied); if falsified, the result is reported in
  Section 4.2 and the boundary is not stated.
- **Result**: **Mechanism supported; boundary half supported, half
  falsified, and reported as such.** Run
  `20260907T141704Z_level_tolerance_875fd5c_flat` (flat initial and new
  contrasts, the registered method; an earlier run of the same config,
  `20260907T123710Z_level_tolerance_bc130fa`, used the proper N(0, 2^2)
  contrast prior that commit 2cbd190 had introduced, reached the same
  verdict, and is kept with a supersession note. The working tree held
  uncommitted manuscript edits at launch, preserved in
  `source_snapshot.json`; the simulation source is that of 875fd5c).
  Adaptive allocation, final best-arm share, OR-TS(0) minus Beta-TS: sd 0
  −0.002 (1.5 se), sd 0.02 +0.003 (1.1 se), sd 0.05 −0.002 (1.8 se), sd 0.1
  +0.076 (1.4 se), sd 0.3 +0.394 (3.5 se). Parity holds through sd 0.05 as
  predicted and the collapse at 0.3 is as predicted, but the difference at
  sd 0.1 is **not** beyond 2 se: Beta-TS's mean share is 0.921 ± 0.055
  against 0.997 ± 0.001, two runs in twenty end below a majority, and its
  regret is 2,068 ± 443 against 1,181 ± 42 (75% higher). The registered
  boundary of one effect gap was too strict; at this batch size the break
  lies between two and six gaps. Fixed allocation, final `win_prob_best`,
  OR-TS(0) minus Beta-TS: within 1.3 se at every sd, including 0.3 (0.994 ±
  0.006 against 1.000). Secondary: OR-TS(0)'s share is 0.994--0.997 across
  the sweep and its `win_prob_best` 0.995--1.000; the discounted variants
  sit at 0.89--0.95 (adaptive) as in H9 and are not hurt at 0.1; at 0.3
  discounted Beta-TS falls to 0.63 while OR-TS with decay holds 0.94.
  Manuscript: the mechanism sentence goes into Section 3 with the measured
  tolerance stated as a range, Section 4.2 reports the sweep, Supplement F
  carries the figure.

---

## Amendments

**2026-08-30 — R6A is not obtainable at present.** H3's power paragraph names
the Yahoo R6A random bucket as the powered version of this test. As of today
the Webscope portal is dead at the infrastructure level: `webscope.sandbox.yahoo.com`
resolves through `rc.yahoo.com` to an AWS Global Accelerator that accepts TCP
but completes no TLS handshake, on both of its IPs, and the current
yahooinc.com research pages mention no dataset program. The historical process
(request from an accredited-university address plus a signed Data Sharing
Agreement) has no working endpoint to receive it. Unofficial mirrors of R6A
exist but the Webscope licence prohibits redistribution, so they are not a
provenance a paper can cite. If a powered replication is wanted before the
portal returns, the candidate with the same essential property — a logged
uniformly-random exposure component, freely downloadable — is KuaiRand
(Gao et al. 2022, kuairand.com); using it would require a new pre-registered
entry, not a rewrite of H3.


_Date, what changed, and why. Amendments are additive; nothing above is edited
once a run has been recorded against it._

**2026-08-22 — H2 verdict flag was initially computed against the wrong
quantity.** `scripts/analyze.py` first evaluated H2-grid using the maximum
`|ΔP(win)|` over *all* cycles (`5.36e-04`) and reported the prediction as
falsified. The pre-registered wording is about the **final** `P(beta > 0)`, for
which the value is `1.11e-16`. The script was corrected to evaluate exactly the
quantity that was pre-registered, and it now prints both numbers so the
all-cycle maximum is not hidden. No prediction was rewritten; only the code that
checks it. The all-cycle figure is disclosed in the H2-grid Result.

**2026-08-22 — an earlier trajectory-matching bug in the same check.** The grid
comparison first matched cells whose counts agreed in the current cycle only.
Because the filters are recursive, a cell is comparable only when its whole
trajectory saw the same data; single-cycle counts collide by chance at
`n = 100`, which admitted 4 unrelated cells and produced an implausible
`5.5e-01` difference. Fixed to require full-trajectory agreement, which leaves
the 3 genuinely shared trajectories at `n = 100000`.

---

## Open methodological debt

- `src/orts` samples from numpy's legacy global RNG inside `win_prop()`.
  Experiments bridge this by seeding the global RNG from the run's
  `Generator`, which makes runs reproducible but couples all three policies to
  one global stream. Threading a `Generator` through `LogisticBandit` and
  `TSPar` would be cleaner; it changes library behaviour, so ask before doing it.
- `sim/simulate.py` (the 2020 helpers) has a latent bug: `run_all` calls
  `add_random_effect(np.array(p_list), ...)`, which needs a mapping, not an
  array. It is kept for provenance and is not on the v2 path.

## 2026-09-06 — implementation-correction reruns (before execution)

Authorized by the request to fix the review findings and prepare en6/kor6 and
an independent TMLR manuscript. Existing hypotheses, seeds, configurations,
repetition counts and horizons are unchanged. The suffix `en6fix` distinguishes
these implementation-correction runs. Run H1 and H1-control, then both H2
configurations and H2-grid with the original inputs. Report both supporting and
contrary findings. No new superiority prediction is introduced.

Corrections fixed before these runs: stable binomial log likelihood with matching
gradient/curvature; checked optimizer convergence; proper N(0, 2^2) priors for
previously uninitialized/new contrasts; flat-intercept all-zero/all-one batch
skip in both the library and grid comparator; half-cell integration at beta=0.
The new proper contrast initialization corrects the previous unspecified flat
initial contrast fit; it is disclosed as an implementation change, not a tuned
parameter. Decay remains zero in all configurations. The source snapshot records
the exact revised code alongside each run, even with an uncommitted worktree.

The old small-batch fidelity measurements mix improper-integral truncation and
implementation error with approximation error. Keep all old results; do not
reuse their event-count thresholds as validated operating rules. The original
H2-grid comparison only matches the first three common-baseline large-batch
trajectories because repetition counts alter subsequent RNG consumption; retain
and disclose that limitation rather than changing the design after inspection.

New broad baseline comparisons and adaptive-feedback fidelity checks remain
unperformed. The manuscript will limit its claims accordingly. Results will be
recorded by a dated additive entry once the runs finish.

## 2026-09-06 — implementation-correction reruns (results)

All five preregistered `en6fix` runs completed and were analysed without
changing their configs, seeds, repetition counts, or horizons.

- Under the drifting baseline, mean final cumulative regret was 1,186.92 for
  OR-TS, 5,647.69 for full logistic TS, and 6,774.79 for independent Beta TS.
  The paired comparator-minus-OR-TS differences were 4,460.77 (SE 1,408.71)
  and 5,587.87 (SE 1,533.42), respectively.
- In the no-drift control, mean final cumulative regret was 1,186.76 for OR-TS,
  1,149.45 for full logistic TS, and 1,141.56 for Beta TS. Thus the large
  drifting-baseline advantage disappeared; no equivalence margin was
  preregistered, so this is not an equivalence test.
- At batch size 100,000, median final winner-probability error was at numerical
  resolution in the common regime and 8.73e-7 in the rare regime. The largest
  rare-regime final error was 3.03e-5. Endpoint skips occurred only at smaller
  batch sizes: 24 in the common batch-100 condition, and 5 and 234 in the rare
  batch-1,000 and batch-100 conditions. Skipped batches were excluded from the
  fidelity comparison by design.
- The matched grid check covered three large-batch trajectories and 120 cells.
  Maximum final-cycle winner-probability discrepancy was 1.11e-16 and the
  maximum over all cycles was 1.33e-6. This does not extend the claim to small
  batches or adaptive allocation.

Run ids are `20260906T131732Z_drifting_baseline_ec730ae_en6fix`,
`20260906T131831Z_no_drift_control_ec730ae_en6fix`,
`20260906T131733Z_laplace_fidelity_common_ec730ae_en6fix`,
`20260906T131832Z_laplace_fidelity_rare_ec730ae_en6fix`, and
`20260906T131919Z_laplace_fidelity_gridcheck_ec730ae_en6fix`.

## 2026-09-07 — flat initial and new contrasts restored (before execution)

Commit 2cbd190 (2026-09-06) gave every initial and newly introduced
contrast a proper N(0, 2^2) prior inside the library and the manuscript was
edited to say so. That is not the registered method: Algorithm 1 and
Supplement A specify flat, zero-precision initial and new contrasts, with a
proper prior only as an explicit option for an arm whose first batch has
only events or only non-events. Commit 875fd5c restores the flat default
(`LogisticBandit(new_contrast_prior_sd=None)`), raises instead of fitting a
large finite value when a flat new contrast is separated, allows decay = 1
as a restart, and keeps the stable likelihood, checked optimizer and
endpoint skip of 2cbd190. The runs that depended on the proper prior are
rerun with unchanged configs, seeds, repetitions and horizons: H1
(`drifting_baseline`), H1-control (`no_drift_control`) and H11
(`level_tolerance`), tagged `flat`. The fidelity runs (H2) construct their
two-arm state explicitly and never introduce a new contrast, so they are
unaffected and are not rerun. Runs made before 2cbd190 (H5–H10, the OBD
replay) already used flat new contrasts, with the older clipped likelihood;
they stand.

## 2026-09-07 — flat initial and new contrasts restored (results)

- H1, `20260907T141525Z_drifting_baseline_875fd5c_flat`: OR-TS 1,170.13 ±
  52.83, Full-TS 7,337.45 ± 2,103.28, Beta-TS 5,897.76 ± 1,177.54. Ordering
  OR-TS < Beta-TS < Full-TS; the OR-TS/Beta-TS gap is 4,728 (4.0 se), the
  Full-TS/Beta-TS ordering is within noise (0.6 se), as in the original run.
- H1-control, `20260907T141614Z_no_drift_control_875fd5c_flat`: OR-TS
  1,192.93 ± 34.82, Full-TS 1,127.48 ± 23.41, Beta-TS 1,158.30 ± 26.50;
  parity within 1.6 se.
- H11, `20260907T141704Z_level_tolerance_875fd5c_flat`: verdict unchanged
  from the proper-prior run; numbers recorded under H11 above.


## H12 — single-position Open Bandit redesign (2026-09-10, before run)

Motivation: the H3 item-plus-position regression is not saturated over
item-position cells, and H7 pools positions. Neither directly describes a
single-slot binary bandit. The user authorized choosing a replacement design.
We choose position 1 by its label, without comparing outcomes across positions.
This is a follow-up to known pooled results, not an independent replication.

Data: the same checksum-pinned ALL/random aggregate as H3/H7. Retain only
position 1, all calendar days and every item; do not select items by clicks.
No position adjustment, weighting, item grouping, or smoothing across days.
Every item must have positive exposures on every day; otherwise stop and
report the structural failure. The target is the marginal click probability
of an item at position 1 under the logged recommendation context. Other
slots, repeated users, and changing audience composition remain limitations;
this is not a joint slate policy evaluation.

H12a (diagnostic): use the H3 daily binomial logistic fit with item 0 as
reference, unpenalized day level, and independent N(0,2^2) item-effect
regularization. With one position, the likelihood has K free probabilities
per day (saturated); regularization still shrinks estimates. Predict that
pooled excess contrast sd exceeds daily level sd (R < 1). Falsification:
R >= 1, including zero estimated excess contrast variation. This threshold
is descriptive, not a calibrated significance test. Report all estimates,
including both fixed/free-contrast likelihood fits, but do not interpret
the penalized-fit likelihood difference as a calibrated chi-square test.
Reuse all seven days; seed 20260910; no stochastic repetitions required.

H12b (measured-environment simulation): use the unadjusted position-1
clicks/exposures for each item-day as fixed Bernoulli environment rates and
the corresponding observed daily traffic. All seven days, all items,
20 repetitions, seed 20260910; Beta-TS, discounted Beta-TS, OR-TS with
lambda=0, OR-TS with lambda=0.1; gamma=1 throughout. Because single-slot
counts can include zero-click item-days, explicitly use a proper N(0,2^2)
initial contrast prior for both OR-TS variants before inspecting results;
Beta-TS keeps Beta(1,1). Report this prior difference. No tuning after run.
Predict no pairwise final-regret difference exceeds twice its independent
Monte Carlo SE, following H7's short-horizon prediction. Any pair exceeding
that threshold falsifies it. This is a descriptive separation criterion,
not an equivalence test or a multiplicity-adjusted inference. Report all
pairs; scoring uses each day's best item and aggregates all repetitions.
Sampling uncertainty in the measured environment is not included.

Run the unchanged H1 no-drift control as a pipeline check. It does not
validate the sparse single-slot approximation or establish H12 equivalence.
Preserve H3/H7 and report their design differences in the supplement.

### H12 — results (2026-09-10, after run)

- **H12a**, run `20260909T215447Z_obd_position1_stability_7e61726`
  (79 item contrasts against item 0, 7 days, 458,005 displays, 1,622
  clicks, 2.90 clicks per item-day): daily level sd 0.236, pooled excess
  contrast sd 0.347, R = 0.679. **Prediction (R < 1) met.** Descriptive
  likelihood difference between fixed- and free-contrast fits 771.6 on
  474 df (z = 9.7, not interpreted as a calibrated test); mean item trend
  +0.024 per day (t = 1.40). Exposure balance at position 1: within-day
  CV 0.105, chi-square per df 9.7, total max/min 1.24, day-to-day share
  correlation 0.00, share-vs-click-rate correlation 0.01.
- **H12b**, run `20260909T215454Z_obd_position1_replay_7e61726` (20
  repetitions, seed 20260910, OR-TS variants with the pre-declared proper
  N(0,2^2) initial contrast prior, Beta-TS with Beta(1,1)): final
  cumulative regret Beta-TS 5,858.6 ± 46.7, discounted Beta-TS 5,851.4 ±
  39.7, OR-TS(0) 5,765.9 ± 112.3, OR-TS(0.1) 5,809.6 ± 101.6. All six
  pairwise differences within 2 independent SE; the largest is OR-TS(0)
  minus Beta-TS, −92.7 clicks (0.8 SE). **Prediction (no pair separates)
  met.** Regret is scored against each day's observed best item, which at
  three clicks per item-day is a noisy extreme, so the level of regret is
  inflated and only differences are read. Final share on the day's best
  item 0.03–0.16; Beta-TS ends 20/20 repetitions with it below a majority,
  OR-TS(0) 19/20, OR-TS(0.1) 16/20. The H7 separation (plain Beta-TS ahead
  of discounted Beta-TS by 2.1 SE at ~8 clicks per item-day) does not
  recur at position-1 sparsity (difference 7 clicks, 0.1 SE).
- **Pipeline check**, run
  `20260909T215455Z_no_drift_control_7e61726_position1_check`: Beta-TS
  1,158.3 ± 26.5, Full-TS 1,127.5 ± 23.4, OR-TS 1,192.9 ± 34.8, identical
  to the H1 control reported in Supplement F.
- **Manuscript**: Section 4.1 and Figure 4 now use H12a; Section 4.2's
  measured environment uses H12b; H3/H7 are preserved in Supplement E with
  their run ids and design differences. The contrast-only simulation
  severities (H5/H10) are unchanged and are described as prespecified
  stress scenarios rather than a calibration to H12.

## H13 — symmetric proper-prior implementation check (2026-09-12, before run)

Motivation: the fixed-reference independent Gaussian option used in some
historical runs privileges the reference arm.  The new OR-TS default instead
starts from exchangeable latent arm effects,
`z_i iid Normal(0, tau^2)`, and stores the induced reference contrasts
`beta_i = z_i - z_K`.  This first matched rerun checks that adopting that
prior preserves the high-traffic H1 mechanism.  It is an implementation and
high-traffic policy check, not validation of the sparse-event, ASOS, Open
Bandit, new-arm, or Laplace-fidelity claims listed in the 2026-09-12 handoff.

- **Configs**: `sim/configs/symmetric_prior_drifting_baseline.json` and
  `sim/configs/symmetric_prior_no_drift_control.json`.
- **Prior fixed before outcomes**: pairwise contrast standard deviation `2`,
  hence arm-effect standard deviation `tau = sqrt(2)`.  For `K` initial arms,
  `mu_0 = 0` and
  `Sigma_0 = tau^2 (I + 11')`, equivalently
  `S_0 = tau^-2 (I - 11'/K)`.  The prior is applied once at initialization;
  later batches carry the resulting joint contrast posterior.  The intercept
  remains fresh and flat in every batch.  This scale matches only the
  marginal contrast variance of the historical independent `N(0, 2^2)`
  option; the joint priors are not the same.
- **Policies**: (1) symmetric-prior OR-TS with `lambda = 0`, `gamma = 1`;
  (2) legacy-flat OR-TS with the same settings; (3) historical flat-prior
  Full-TS; and (4) Beta-TS with independent `Beta(1,1)` arm priors.  The last
  three are controls and retain their prior behavior.  All policies see the
  same environment sequence within a repetition and independently sampled
  outcomes conditional on their allocations.
- **Environment and fixed design**: copy H1 exactly: ten arms, forty batches,
  100,000 trials per batch, baseline rate 0.03, adjacent logit effect 0.05,
  20 repetitions, seed 20260822.  The common-shock condition uses
  `sigma_drift = 0.30`; the mandatory control uses `sigma_drift = 0`.
- **Endpoint and transition semantics**: an individual arm may have zero or
  complete events under the proper contrast prior when the whole batch has
  both outcomes.  A whole-batch all-zero or all-one outcome is skipped
  atomically, without decay or state mutation.  `lambda = 1` retains the
  legacy meaning of an attempted flat-prior restart; it is not used in these
  runs.  A newly constructed bandit is the explicit administrative reset and
  receives the selected initialization prior.
- **Reported metrics**: mean, standard deviation, and standard error of final
  cumulative regret; paired within-repetition final-regret differences;
  final best-arm allocation share; and every policy/run, including null or
  adverse results.
- **Predictions**: under the common shock, symmetric OR-TS has less final
  cumulative regret than both Beta-TS and Full-TS by more than two paired
  standard errors.  Its absolute mean paired difference from legacy-flat
  OR-TS is at most 10% of legacy-flat OR-TS mean regret.  In the no-drift
  control, its absolute mean difference from each of legacy-flat OR-TS and
  Beta-TS is at most 10% of that comparator's mean regret.  These fixed 10%
  margins express practical similarity; a two-SE non-rejection is not used
  as an equivalence claim.
- **Falsified if**: either common-shock comparator gap fails the registered
  two-paired-SE threshold, or any registered 10% practical margin is crossed.
  A failure is reported as such; no seed, scale, horizon, repetitions, or
  margin will be changed after inspecting the runs.
- **Result** (2026-09-12; runs
  `20260911T221857Z_symmetric_prior_drifting_baseline_d0714c3` and
  `20260911T221944Z_symmetric_prior_no_drift_control_d0714c3`): **supported
  on every registered criterion**; verdicts are recorded in each run's
  `results/processed/<run_id>/summary.json` under `h13_registered_verdict`.
  An earlier attempt, `20260911T221537Z_symmetric_prior_drifting_baseline_d0714c3`,
  crashed before producing any output when the fitted-precision symmetry
  validator rejected machine-precision rounding noise; it is retained with a
  NOTES.md and no outcome of it existed or was inspected.
  - Common shock (`sigma_drift = 0.30`), final cumulative regret mean (se):
    symmetric OR-TS 1310.6 (68.9); legacy-flat OR-TS 1308.0 (47.6); Beta-TS
    7741.0 (1517.1); Full-TS 7255.8 (2025.9). Paired against symmetric:
    Beta-TS +6430.4 (paired se 1510.4, 4.3 se) and Full-TS +5945.1 (paired
    se 2027.2, 2.9 se), both beyond the registered two-se threshold;
    |symmetric - flat| = 2.7 against the registered margin 130.8.
  - No-drift control: symmetric 1174.4 (37.3); flat 1155.1 (31.4); Beta-TS
    1147.1 (30.7); Full-TS 1214.2 (36.1). Symmetric is nominally *worse*
    than flat by 19.3 and than Beta-TS by 27.3; both sit inside the
    registered 10% practical margins (115.5 and 114.7), which is the
    registered similarity reading, not a superiority claim.
  - Scope as registered: this validates the implementation and the
    high-traffic H1 mechanism under the new default prior. Sparse-event,
    ASOS, Open Bandit, new-arm augmentation, and Laplace-fidelity claims
    remain unvalidated for the new prior.

## Known limitation carried into H13-H19 (recorded 2026-09-12, before any outcome was inspected)

`LogisticBandit.get_par` re-references a state by mapping its covariance
linearly. When the stored precision has an exactly zero intercept row and
column — the flat-intercept contrast state — the transform mixes the
intercept coordinate into the returned rows, so the result no longer has an
exact zero row: `_covariance` then takes the `inv` branch rather than
`pinv`, and the last row and column of the returned state describe the new
reference arm's log-odds as if it were known rather than flat. In the
`odds_ratios_only=True` path `update` re-zeroes the intercept row before
fitting, so only that path's contrast block is carried forward; the
`odds_ratios_only=False` path would hand the row to the fitter as an
informative prior on the level. No configuration in H13-H19 reaches it: the
only full-rank caller starts from an empty state whose first fitted
precision is validated positive definite. Recorded here rather than fixed
while runs are in flight, so that the library source behind every run in
this batch stays identical across machines.

**Fixed 2026-09-12, after every run in the batch had completed.**
`get_par` now re-references the contrast block on its own whenever the
stored intercept precision is exactly zero, and returns the intercept still
flat. This is exact rather than a repair: with arm logits
`theta_i = alpha + beta_i`, the new contrasts `beta'_j = theta_j - theta_s
= beta_j - beta_s` do not involve `alpha`, so they transform by the
contrast rows of the same map, and `alpha' = theta_s = alpha + beta_s` is
flat whenever `alpha` is. The flat test is exact equality, not a tolerance:
a small but nonzero intercept precision is information and is still
transformed by the full linear map, as a proper Full-TS state is.

No recorded result changes. The corrupted row was confined to the
intercept: a pseudo-inverse of the old return still recovered the correct
contrast marginal, so `win_prop` drew from the same distribution before and
after, and `update` under `odds_ratios_only=True` re-zeroed the intercept
row anyway. The only behavioural change is the `odds_ratios_only=False`
path under a reference change, which no configuration in this batch
exercises. Byte-identity of the re-run results was checked directly against
the recorded runs rather than argued. Regression tests are in
`tests/test_flat_intercept_rereference.py`, including reference invariance
of winner probabilities and the near-zero-is-not-zero boundary.

## H14 — sparse events: the motivation for the proper prior (2026-09-12, before run)

The symmetric prior was adopted because the flat contrast prior has no
finite fit for an arm whose first batch has only events or only non-events.
This run tests that motivation where it lives: per-batch counts small enough
that individual-arm endpoints are routine. It also quantifies the cost of
the historical behavior, in which such a batch must be skipped.

- **Configs**: `sim/configs/sparse_events_drifting.json` (sigma_drift 0.30)
  and `sim/configs/sparse_events_no_drift.json` (the mandatory control).
- **Design**: the H13 experiment and policies at sparse traffic — ten arms,
  60 batches, **400 trials per batch**, baseline 0.03, adjacent logit effect
  0.05, 30 repetitions, seed 20260912, `skip_failed_updates` on. At 400
  trials, roughly 40 per arm under uniform allocation, an arm shows only
  non-events in a batch with probability about 0.30, so the flat prior's
  failure mode is common; a whole-batch endpoint (no events anywhere) has
  probability about 5e-6 and should essentially never occur.
- **Skip semantics, fixed before the run**: a batch the library cannot use
  is dropped atomically, through either of its two channels — a raised
  error (a separated newcome arm under the flat prior) or a silent
  no-change return (a whole-batch endpoint under a flat intercept). In
  both cases the posterior, and hence the next allocation, are exactly
  what they were before the batch; a policy that has never fitted
  allocates uniformly; skipped batches are not replayed. `fit_status`
  records fit/skipped per policy and batch, detected as "the state
  changed", not as "no exception was raised".
- **Predictions**: (1) mechanism — symmetric OR-TS fits at least 99% of
  batches whose pooled outcome is mixed (both events and non-events
  present); the legacy-flat OR-TS fit rate is reported and expected to be
  far below 1 early on. (2) decision quality — symmetric OR-TS's final
  cumulative regret is not worse than Beta-TS's and not worse than
  legacy-flat OR-TS's by more than two paired standard errors, in both
  conditions. Directional gains at these counts are *not* claimed; sparse
  batches may be too noisy for the common-shock advantage to clear two SE,
  and the result is reported either way.
- **Falsified if**: symmetric OR-TS fails to fit a mixed-outcome batch
  (rate below 0.99), or is worse than Beta-TS or legacy-flat OR-TS by more
  than two paired SE in either condition.
- **Result** (2026-09-12; runs
  `20260911T224315Z_sparse_events_drifting_d870ba6` and
  `20260911T224453Z_sparse_events_no_drift_d870ba6`): **supported on both
  registered criteria, in both conditions.**
  - Mechanism. Symmetric OR-TS fitted **100.0%** of batches in both
    conditions — every batch, not merely every mixed-outcome batch. Legacy
    flat OR-TS fitted 90.9% under the shock and 83.1% in the control,
    skipping the rest because an arm entered separated; Full-TS fitted
    88.4% and 82.6%. Beta-TS never skips. At 400 trials over ten arms the
    registered whole-batch endpoint never occurred, so the mixed-outcome
    fit rate equals the overall rate for every policy.
  - Decision quality, final cumulative regret, mean (se): common shock —
    symmetric 124.7 (5.8), Beta-TS 125.1 (4.9), legacy-flat 137.1 (4.4),
    Full-TS 145.7 (4.2); control — symmetric 116.0 (3.7), Beta-TS 120.1
    (3.6), legacy-flat 138.2 (4.7), Full-TS 143.4 (4.9). Paired against
    symmetric: Beta-TS +0.5 (se 7.2) and legacy-flat +12.5 (se 7.5) under
    the shock; Beta-TS +4.1 (se 5.5) and legacy-flat +22.2 (se 6.0) in the
    control. Symmetric is not worse than either comparator by two paired
    SE anywhere, as registered.
  - Unregistered and therefore exploratory: in the control, symmetric is
    *better* than legacy-flat by 3.7 paired SE. The registration
    explicitly declined to claim a directional gain at these counts, so
    this is reported as an observation, not as a confirmed effect, and no
    criterion was changed after seeing it. The plausible mechanism is the
    skipped batches themselves — a skipping policy discards the batch's
    information and keeps allocating from a staler state — but this run
    was not designed to separate that from the shrinkage difference.

## H15 — Laplace fidelity under the shipped symmetric default (2026-09-12, before run)

H2 measured the Gaussian filter against the exact marginal filter using an
explicitly constructed independent contrast prior. The library default is
now the symmetric proper prior, whose K=2 contrast marginal is the same
N(0, 4); the construction path, the intercept prior (truly flat rather than
precision 1e-6), and the endpoint semantics (whole-batch endpoints skipped
atomically) differ. This run checks that the shipped default is the same
filter where it should be, and no worse where it is allowed to differ.

- **Config**: `sim/configs/laplace_fidelity_symmetric.json` — the H2 rare
  design (baseline 0.005, batch sizes 100000/10000/1000/100, 40 rounds, 10
  repetitions, sigma_drift 0.30), seed 20260912, both prior modes fed
  identical data within a repetition.
- **Predictions**: (1) identity — at the two largest batch sizes, where a
  whole-batch endpoint has negligible probability, the two modes' Laplace
  mean and SD agree within 1e-3 at every round; (2) fidelity — the
  symmetric mode's median KL and median winner-probability error over
  fitted rows are at most 1.5x the explicit mode's at every batch size;
  (3) endpoint accounting — symmetric-mode skipped batches are counted and
  reported (at batch size 100 and baseline 0.005 most batches are
  whole-batch endpoints; both filter sides skip them together).
- **Falsified if**: any identity discrepancy exceeds 1e-3, or any fidelity
  median ratio exceeds 1.5.
- **Result** (2026-09-12; run
  `20260911T224447Z_laplace_fidelity_symmetric_d870ba6`): **supported on
  every registered criterion.**
  - Identity. Over all 400 paired (repetition, round) cells at each of the
    two largest batch sizes, the largest absolute discrepancy between the
    two constructions was 5.6e-08 in the contrast mean and 6.8e-10 in its
    SD at n=100000, and 2.6e-08 / 1.1e-09 at n=10000 — four to five orders
    of magnitude inside the registered 1e-3. The registered premise held:
    no whole-batch endpoint occurred at either size.
  - Fidelity. Every registered ratio is within 0.03% of 1.0 (largest
    1.0002, smallest 0.9991), against the registered 1.5x allowance. The
    median KL over fitted rows is 3.6e-07, 6.7e-06, 1.0e-03 and 1.4e-02 at
    n = 100000, 10000, 1000 and 100 respectively, identical to four
    significant figures across the two constructions.
  - Endpoint accounting, as registered: both filter sides skipped the same
    batches — 0, 0, 3 and 226 of 400 at the four batch sizes. At n=100 and
    baseline 0.005 a whole-batch endpoint is the common case, and the
    grid filter and the shipped filter skip it together.
  - Reading: on this benchmark the shipped default is not merely close to
    the historical construction, it is the same filter. H2's fidelity
    conclusions therefore carry over to the new default unchanged.

## H16 — level-tolerance audit under the symmetric default (2026-09-12, before run)

H11's boundary and mechanism were established with flat-prior OR-TS. If the
symmetric prior is the default, the H11 pattern must be shown for it, not
assumed.

- **Config**: `sim/configs/level_tolerance_symmetric_audit.json` — the H11
  design and seed policy verbatim (ten arms, 40 rounds, 100000 trials,
  baseline 0.03, effect 0.05, common shock sd in {0, 0.02, 0.05, 0.1, 0.3},
  adaptive and fixed allocation, forget 0.1, 20 repetitions) with fresh
  seed 20260912 and two added policies: `orts_l0_sym` and `orts_decay_sym`.
  Historical policies keep their flat priors.
- **Predictions**: (1) equivalence — in every (disturbance, allocation)
  cell, the paired mean final-cumulative-regret difference between each
  symmetric twin and its flat counterpart is within 10% of the flat
  counterpart's mean; (2) the H11 pattern reproduces for the symmetric
  variant against Beta-TS: adaptive final best-arm share within 2 paired SE
  at sd 0 and 0.02 and beyond +2 SE in the symmetric variant's favour at
  0.1 and 0.3 (0.05 reported either way, not a criterion), and fixed-
  allocation final `win_prob_best` within 2 SE at every sd.
- **Falsified if**: any cell crosses its 10% margin, or the pattern fails
  as registered for H11.
- **Result** (2026-09-12; run
  `20260911T224800Z_level_tolerance_symmetric_audit_d870ba6`):
  **equivalence supported; the registered H11 pattern criterion is
  falsified — but not by the prior.**
  - Equivalence (prediction 1): supported in all ten cells for both twins.
    The largest adaptive gap is -125.7 (se 68.1) for `orts_l0_sym` minus
    `orts_l0` at sd 0, against a 127.3 margin; every fixed-allocation cell
    is identical to the last digit (difference exactly 0.0), because with
    1/K allocation every round the two variants see the same counts and
    the contrast prior does not enter the allocation at all.
  - Pattern (prediction 2): falsified. The fixed-allocation half held
    exactly as registered — the symmetric variant's final `win_prob_best`
    is within two paired SE of Beta-TS's at every sd including 0.3. The
    adaptive half did not: at sd 0.1 the symmetric variant is ahead by
    only 0.0025 (se 0.0038), 0.7 SE, where the registration required more
    than two; and at sd 0 it is *behind* by 0.0039 (se 0.0017), 2.3 SE,
    where the registration required agreement within two. Only sd 0.3
    behaved as registered (+0.458, se 0.109).
  - **The falsification does not implicate the symmetric prior.** The same
    run's historical flat policy — the one H11 itself used — fails in
    exactly the same cells: `orts_l0` minus Beta-TS is -0.0026 (se 0.0010)
    at sd 0, +0.0030 (se 0.0034) at sd 0.1, and +0.459 (se 0.109) at sd
    0.3. Both variants cross between 0.1 and 0.3 here, where H11 under
    seed 20260907 placed the crossing between 0.02 and 0.1. What this run
    falsifies is the reproducibility of H11's boundary location at 20
    repetitions, not the equivalence of the two priors. The sd 0.1 cell is
    evidently under-powered at this repetition count: its paired SE is of
    the same order as the effect H11 reported there.
  - Registered consequence: H16 is recorded as **not supported**, because
    the conjunction was registered. No threshold, seed or repetition count
    was changed after seeing this, and H11's own entry is left as it
    stands; the cross-check above is reported alongside it.

## H17 — ASOS class-1 replay audit under the symmetric default (2026-09-12, before run)

- **Config**: `sim/configs/asos_replay_symmetric_audit.json` — the H8
  design verbatim (same data, checksum, filter, forget 0.1, 20 repetitions)
  with fresh seed 20260912 and the two symmetric twins added.
- **Predictions**: the paired mean experiment-regret difference between
  each symmetric twin and its flat counterpart is within 10% of the flat
  counterpart's mean. Level-tracking error for the symmetric variant is
  reported descriptively (H8's level-tracking claim was about Beta-TS
  versus OR-TS, not about the prior).
- **Falsified if**: either twin crosses its 10% margin.
- **Result** (2026-09-12; run
  `20260911T225934Z_asos_replay_symmetric_audit_d870ba6`): **supported on
  both registered margins.** Correction (2026-09-12): two sessions invoked
  this config concurrently. Both runs completed — `224714Z` at
  23:29:48Z and `225934Z` at 23:30:54Z, each with 20 repetitions over 71
  experiments — and their `results.csv` files are byte-identical
  (sha256 `82d85c65…`), as they must be under one seed and the shared
  repetition cache. The NOTES.md inside `224714Z` was written while that
  run was still executing and wrongly reports it as aborted; it is left
  untouched because it is part of that run's manifest, and this entry
  is the corrected record. `225934Z` remains the run cited below.
  - Mean experiment regret, mean (se): `orts_l0` 9404.2 (47.9),
    `orts_l0_sym` 9463.2 (76.6), `orts_decay_sym` 9491.7 (28.2),
    `orts_decay` 9519.1 (22.3), `beta_ts` 10502.6 (132.4),
    `beta_ts_discount` 10669.7 (169.4).
  - Registered comparisons, paired within repetition: |`orts_l0_sym` minus
    `orts_l0`| = 59.0 (se 100.2) against a 940.4 margin; |`orts_decay_sym`
    minus `orts_decay`| = 27.5 (se 39.0) against a 951.9 margin. Both are
    inside by more than an order of magnitude and neither is resolved from
    zero. On real class-1 traffic the two priors are, for practical
    purposes, the same policy.
  - Level tracking, descriptive as registered: the symmetric variant's
    median absolute level error is 0.1489 against the flat variant's
    0.1489, a ratio of 0.9998 — the prior does not touch this quantity,
    as expected, since the intercept is refitted flat every batch under
    both. Beta-TS is at 0.3451 and discounted Beta-TS at 0.2606, so H8's
    level-tracking gap is reproduced under the new default.
  - Beside the registered criteria, and matching H8 rather than testing it
    anew: every Beta-TS variant separates from every OR-TS variant at two
    SE on regret here, and Beta-TS's per-experiment level error exceeds
    OR-TS(0)'s in 85.9% of the 71 experiments with a median ratio of 2.82.
    H8's prediction 1 (all pairs within 2 SE) does not hold at this seed
    and repetition count; that is a statement about H8, recorded here
    without amending H8's own entry.

## H17-posthoc — Where the ASOS regret aggregate comes from: allocation and per-experiment click gain (2026-09-14, exploratory, registered after the outcomes were seen)

*Not a hypothesis test.* This entry re-analyses the existing H17 run
`20260911T225934Z_asos_replay_symmetric_audit_d870ba6` without a new
simulation. It is registered after a scratch analysis of the same raw
file had been inspected, so no prediction is made and no verdict is
issued; the quantities below are descriptive and are reported in full,
including whichever direction they take.

- **Motivation**: the registered regret aggregate (H8, H17) favours
  OR-TS but is dominated by one experiment and the per-experiment median
  runs the other way, so Section 4.3 currently reads as "no broad
  advantage". Two questions that the registered quantities do not answer:
  (a) how large the per-experiment differences are in the units a service
  reads, expected clicks relative to Beta-TS; and (b) whether Beta-TS's
  allocation is suboptimal even where its regret is small, i.e. whether
  the level-contamination mechanism of Section 3 is visible in where
  Beta-TS sends traffic.
- **Quantities**, computed by `scripts/analyze.py` from `results.csv`
  and written to `results/processed/<run>/allocation_by_experiment.csv`
  and the `posthoc_allocation` block of `summary.json`:
  1. Per experiment and policy, the relative expected-event gain against
     Beta-TS, `(events_policy - events_beta) / events_beta`, from the
     per-repetition sums of `expected_events`, averaged over the 20
     repetitions; and the paired standard error over repetitions of the
     final-regret difference `beta_ts - policy`.
  2. For the two-arm experiments only (the raw file records the control
     allocation, which determines the whole allocation only when there is
     one variant): the long-run best arm, defined as the arm with the
     higher pooled event rate over the replayed periods (the same
     periods and filter as `sim/experiments/asos_replay._environment`);
     each policy's traffic share on that arm, traffic-weighted over the
     periods and in the final period, averaged over repetitions; and the
     number of experiments in which the final-period share is below one
     half.
- **Headline fields**: for every policy, the median, mean, minimum and
  maximum of the per-experiment gain, the number of experiments with
  positive gain and with |gain| below 0.1%; for the two-arm subset, the
  mean share on the long-run best arm and the below-majority count.
- **Fixed in advance of the analyzer change**: no policy, repetition,
  experiment or period is dropped beyond the two-arm restriction stated
  above, which is a data limitation, not a choice after inspection.
- **Result** (2026-09-14, `analyze.py` re-run on the H17 run; block
  `posthoc_allocation` of its `summary.json`, rows in
  `allocation_by_experiment.csv`): 57 of the 71 experiments have two arms.
  - Expected-event gain against Beta-TS, per experiment: `orts_l0_sym`
    median -0.000%, mean +0.034%, positive in 33 of 71, |gain| below 0.1%
    in 57 of 71, minimum -0.61%, maximum +1.49%; `orts_decay_sym` median
    +0.003%, positive in 37, within 0.1% in 53, range -0.33% to +1.62%;
    `beta_ts_discount` median +0.001%, positive in 42, within 0.1% in 62,
    range -0.22% to +0.32%. Paired over repetitions, `orts_l0_sym` is worse
    than Beta-TS beyond 2 se in 17 experiments and better in 14; the other
    40 are within 2 se.
  - Share of traffic on the pooled-best arm, two-arm subset, final period
    averaged over repetitions: `beta_ts` 0.511 (below one half in 29 of
    57), `beta_ts_discount` 0.556 (22), `orts_decay_sym` 0.673 (13),
    `orts_decay` 0.689 (12), `orts_l0` 0.766 (6), `orts_l0_sym` 0.774 (4).
    Traffic-weighted over all periods the ordering is the same
    (`beta_ts` 0.513, `orts_l0_sym` 0.707).
  - Reading, descriptive: at contrasts near the resolution of the data the
    per-experiment click differences are negligible for every policy, but
    Beta-TS's allocation is only loosely tied to the pooled-best arm,
    whereas OR-TS's is not below a majority in more than four experiments.
    The flat and symmetric OR-TS variants agree, so the prior is not the
    driver.

## H18 — Open Bandit position-1 replay audit under the symmetric default (2026-09-12, before run)

- **Config**: `sim/configs/obd_position1_symmetric_audit.json` — the H12b
  measured-environment design verbatim (same aggregate and checksum,
  position 1, forget 0.1, 20 repetitions; historical OR-TS entries keep
  their independent N(0, 2^2) new-contrast prior) with fresh seed 20260912
  and the two symmetric twins added.
- **Predictions**: the paired mean final-cumulative-regret difference
  between each symmetric twin and its historical counterpart is within 10%
  of the counterpart's mean; Beta-TS and discounted Beta-TS differences
  against the symmetric variant are reported without a prediction.
- **Falsified if**: either twin crosses its 10% margin.
- **Result** (2026-09-12; run
  `20260911T224324Z_obd_position1_symmetric_audit_d870ba6`): **supported
  on both registered margins.**
  - Final cumulative regret, mean (se): `orts_decay_sym` 5615.3 (109.0),
    `orts_l0_sym` 5662.5 (119.1), `orts_l0` 5722.6 (109.4), `beta_ts_discount`
    5809.5 (55.0), `orts_decay` 5858.3 (80.2), `beta_ts` 5866.3 (49.2).
  - Registered comparisons, paired within repetition: `orts_l0_sym` minus
    `orts_l0` is -60.1 (se 170.0) against a 572.3 margin; `orts_decay_sym`
    minus `orts_decay` is -243.0 (se 146.5) against a 585.8 margin. Both
    well inside, and both nominally in the symmetric variants' favour
    without being resolved from zero.
  - Reported without a registered prediction: Beta-TS minus `orts_l0_sym`
    is +203.7 (se 117.9) and discounted Beta-TS minus `orts_l0_sym` is
    +147.0 (se 126.5) — nominally favouring the symmetric variant, neither
    resolved at two SE. This remains the dataset where the contrast bet
    fails (H12a found item contrasts moving more than the level), so no
    advantage is claimed here.

## H19 — new-arm arrival under the symmetric augmentation (2026-09-12, before run)

The symmetric prior defines how a newly arriving arm joins the joint
contrast state. The unit tests establish the algebra (existing marginals
and reference invariance are preserved); this experiment asks what the
choice does to decisions.

- **Configs**: `sim/configs/new_arm_drifting.json` (sigma_drift 0.30) and
  `sim/configs/new_arm_no_drift.json` (the mandatory control); new
  experiment `sim/experiments/new_arm.py`.
- **Design**: five arms from round 0; the true best arm (largest effect,
  5 x 0.05 on the logit scale) arrives at round 20 of 60; 100000 trials
  per round, baseline 0.03, 20 repetitions, seed 20260912. Policies:
  symmetric OR-TS (joint augmentation), legacy-flat OR-TS (flat new
  contrast), legacy-independent OR-TS (N(0, 2^2) new contrast), Beta-TS
  (fresh Beta(1,1) on arrival). Registered entry rule for every policy: in
  the arrival round, allocation is uniform over the active arms; other
  rounds allocate by probability matching on the policy's own posterior
  (round 0 excepted — see the amendment below).
- **Endpoint**: cumulative regret accumulated from the arrival round to the
  end (`cum_regret_post`), paired within repetition.
- **Predictions**: under the common shock, symmetric OR-TS beats Beta-TS
  post-arrival by more than two paired SE, and is within 10% of both the
  flat and the independent variants (the augmentation neither helps nor
  hurts materially at high traffic; its value is coherence, not regret).
  In the control, symmetric OR-TS is within 10% of every comparator.
- **Falsified if**: the Beta-TS gap fails the two-SE threshold under the
  shock, or any registered 10% margin is crossed in either condition.
- **Amendment (2026-09-12, before any outcome was inspected)**: the entry
  rule as first written omitted round 0. The experiment has always
  allocated uniformly there as well — `sim/experiments/new_arm.py` line 127
  and its module docstring say so, and probability matching is not defined
  at round 0: a bandit with no fitted posterior returns an empty winner
  distribution. The text is corrected to match the code and the docstring.
  No prediction, endpoint, margin, threshold, seed, or config value is
  changed; the rule is identical across all four policies; and the
  registered endpoint accumulates only from the arrival round onward, so
  round 0 never enters it. Recorded as an amendment rather than a silent
  edit because the runs already existed when the discrepancy was found —
  their outcomes had not been read.
- **Result** (2026-09-12; runs
  `20260911T224624Z_new_arm_drifting_d870ba6` and
  `20260911T224712Z_new_arm_no_drift_d870ba6`): **falsified in both
  conditions, as registered.** Post-arrival cumulative regret, paired
  within repetition.
  - Common shock. The contrast-carrying advantage survived the arm change
    decisively: Beta-TS minus symmetric OR-TS is **+7054.9 (se 1277.1),
    5.5 paired SE**, far beyond the registered two. Post-arrival means:
    symmetric 580.4, legacy-flat 573.2, independent 648.0, Beta-TS 7635.3.
    But one registered practical margin was crossed: symmetric minus
    independent is -67.7 against a 64.8 margin — outside by 4% of the
    comparator's mean, in the symmetric variant's *favour*, and well
    inside one SE of zero (se 71.7). The conjunction therefore fails.
  - No-drift control. Two margins crossed. Symmetric minus legacy-flat is
    +76.1 (se 40.5) against a 57.2 margin; symmetric minus Beta-TS is
    **+201.8 (se 38.2), 5.3 SE**, against a 44.7 margin — Beta-TS is
    decisively better post-arrival when there is no level to relearn.
    Post-arrival means: Beta-TS 446.7, legacy-flat 572.3, independent
    605.6, symmetric 648.4. Symmetric minus independent, +42.8 (se 40.9)
    against a 60.6 margin, was the only registered comparison inside.
  - Reading, stated no more strongly than the design supports. The
    registered claim had two halves and they fail differently. The
    *directional* half held under the shock and is the substantive
    finding: a mid-run arrival does not cost OR-TS its advantage when the
    common level moves. The *equivalence* half failed, but every crossing
    among the three OR-TS augmentations is within about one SE of zero, so
    what the run shows is that 20 repetitions cannot resolve a 10% margin
    among them, not that the augmentations differ. The exception is the
    control's Beta-TS comparison, which is resolved and adverse: with a
    static level, a newly arrived best arm is learned faster from
    independent absolute rates than from a shrunk contrast against a
    freshly refitted intercept. That is the shrinkage cost the handoff
    warned about, now measured; it is the same direction as H13's control
    and about seven times larger, which is what the arm change adds.
  - No seed, horizon, repetition count or margin was changed after these
    outcomes were read. The falsification stands as registered.

## H19b — an extreme newcomer under the proper prior (2026-09-12, before run)

H19's control measured the shrinkage cost of the symmetric augmentation on
a newcomer whose true effect (0.25 logits) is small against the prior scale:
Beta-TS learned it faster by 5.3 paired SE when no level had to be relearned.
This run pushes on that cost where it should be largest — a genuinely
extreme newcomer at sparse traffic — and asks whether inflating the
newcomer's prior variance buys it back, the question raised when the cost
was first predicted. With one batch of information $I$ and prior variance
$V$ the posterior mean is pulled by $I/(I+1/V)$, so the cost is a first-few-
batches effect whose size depends on $V$ and $I$ together.

- **Configs**: `sim/configs/new_arm_extreme_drifting.json` (sigma_drift
  0.30) and `sim/configs/new_arm_extreme_no_drift.json` (the mandatory
  control), experiment `new_arm` with the new `arrival_effect_logit` and
  `extra_policies` parameters (defaults reproduce H19).
- **Design**: as H19 — five arms from round 0, arrival at round 20 of 60,
  baseline 0.03, registered uniform entry rule — with the arriving arm's
  true effect fixed at **2.0 logits** (odds ratio about 7.4 over the
  reference), **2,000 trials per round** so the first post-arrival batch
  carries limited information, 30 repetitions, seed 20260912.
- **Fixed sensitivity grid, chosen before outcomes**: symmetric OR-TS at
  arm-effect SD sqrt(2) (`orts`, the default) and at 2 sqrt(2) (`orts_wide`,
  the newcomer's prior variance quadrupled), against legacy-independent
  OR-TS (N(0, 2^2) new contrast), legacy-flat OR-TS, and Beta-TS. All five
  are reported.
- **Endpoint**: post-arrival cumulative regret, paired within repetition;
  descriptively, each policy's mean traffic share on the newcomer over the
  first five post-arrival batches (how fast it is recognized as best).
- **Predictions**: (1) under the common shock both symmetric variants beat
  Beta-TS post-arrival by more than two paired SE — the level-relearning
  advantage survives even an extreme arrival; (2) in both conditions the
  wide variant's paired gap to legacy-independent OR-TS is smaller in
  absolute value than the default's — widening the newcomer's prior does
  move the symmetric augmentation toward the weaker-prior option. The 10%
  margins against legacy-independent are reported for both variants but,
  after H19, are not registered as criteria: 30 repetitions are not
  expected to resolve them.
- **Falsified if**: either symmetric variant fails the two-SE Beta-TS
  threshold under the shock, or the wide variant is not closer to
  legacy-independent than the default in either condition.
- **Result** (2026-09-12; runs
  `20260911T234125Z_new_arm_extreme_drifting_5698c53` and
  `20260911T234256Z_new_arm_extreme_no_drift_5698c53`): **falsified as
  registered — and the design did not reach the regime it was meant to
  probe.** Post-arrival cumulative regret is identical across all five
  policies to fourteen significant figures in the control (254.877) and
  differs by at most 0.15 (se 0.15) under the shock; every policy's
  newcomer share over the first five post-arrival batches is 0.8333, i.e.
  the uniform arrival round followed by four rounds of essentially 100%
  allocation to the newcomer. Prediction (1) fails because Beta-TS and both
  symmetric variants are tied (paired difference 0.0, se 0.0 in the
  control); prediction (2) holds under the shock (wide closer to
  independent, 0.026 against 0.126) and is a 0-versus-0 tie in the control,
  so it fails as registered.
  - Reading. A +2.0-logit newcomer converts at about 18.6% against
    incumbents at 3–3.7%; with about 333 trials in its first batch its
    contrast information is $I\approx 50$ against a prior precision of at
    most 0.42, so every prior keeps $I/(I+1/V)>0.99$ of the MLE and all
    five policies identify it from a single batch. The registration
    quoted the shrinkage factor but chose parameters at which it is
    indistinguishable from 1. The falsification stands as registered; no
    seed, horizon, repetition count or margin was changed after these
    outcomes were read. H19c below is a new design whose parameters are
    derived from that factor before any outcome is inspected.


## H19c — the shrinkage regime, derived before the run (2026-09-12, before run)

H19b showed that at 333 trials per arm a +2.0-logit newcomer is identified
from one batch under any prior. This design is chosen so that the first
post-arrival batch leaves the prior visible. With $n$ trials on the newcomer
and on the reference in that batch, the contrast's information is
$I=(1/I_{\rm new}+1/I_{\rm ref})^{-1}$ with $I_i=n\,p_i(1-p_i)$, and the
posterior mean keeps the fraction $w=I/(I+1/V)$ of the MLE, where $V$ is the
newcomer's prior variance. At **30 trials per arm** (180 per round over
six arms), baseline 0.03 and a newcomer effect of **1.0 logits**
($p_{\rm new}\approx0.078$): $I\approx0.62$, and
$w\approx0.60$ for the default symmetric prior ($V=2.40$),
$0.71$ for the independent $N(0,2^2)$ option, $0.86$ for the
wide symmetric prior ($V=9.6$), and 1 for the flat prior when its fit
exists. These numbers were computed from the formula, not from any run.

- **Configs**: `sim/configs/new_arm_sparse_drifting.json` (sigma_drift 0.30)
  and `sim/configs/new_arm_sparse_no_drift.json` (the mandatory control);
  experiment `new_arm` with `skip_failed_updates` on. At 30 trials and 3%
  an incumbent shows no events in a batch with probability about 0.40, so
  the flat prior's first-fit failure is common here as in H14; skipped
  batches are dropped atomically and recorded in `fit_status`; a policy
  with no fitted state allocates uniformly, and an arm its state does not
  yet cover (its arrival batch was skipped) is reserved $1/K'$ with the
  covered arms scaled into the remainder — the routing rule of the
  manuscript's Supplement~C, applied identically to every policy.
- **Design**: as H19b otherwise — five arms from round 0, arrival at round
  20 of 60, registered uniform entry rule, 30 repetitions, seed 20260912;
  the same five policies (`orts`, `orts_wide`, `orts_indep`, `orts_flat`,
  `beta_ts`).
- **Endpoints**: post-arrival cumulative regret, and the newcomer's mean
  traffic share over the first five post-arrival batches, both paired
  within repetition.
- **Predictions**: (1) in both conditions the wide variant's paired gap to
  legacy-independent OR-TS on post-arrival regret is smaller in absolute
  value than the default's; (2) in both conditions the wide variant's
  first-five newcomer share exceeds the default's by more than two paired
  SE — the prior's pull is now visible in allocation; (3) in the control,
  Beta-TS beats the default symmetric variant on post-arrival regret by
  more than two paired SE — the shrinkage cost H19 measured at high traffic
  is present at this sparsity too. The shock condition's Beta-TS
  comparison is reported without a prediction: level relearning and
  newcomer shrinkage pull in opposite directions there and the design does
  not fix which dominates. Fit rates are reported for every policy.
- **Falsified if**: (1) or (2) fails in either condition, or (3) fails.
- **Result** (2026-09-12; runs
  `20260911T234902Z_new_arm_sparse_drifting_e96b461` and
  `20260911T235037Z_new_arm_sparse_no_drift_e96b461`): **falsified as
  registered in both conditions.** The design reached its regime this time:
  fit rates 99.9% (default), 99.8–100% (wide, independent), 93–94% (flat),
  100% (Beta-TS); first-five-batch newcomer shares spread from 0.58 to 0.77.
  - Prediction (1), wide closer to independent on post-arrival regret: held
    in the control (|+3.98| vs |+6.26|) and failed under the shock (|−6.04|
    vs |−4.40|, the wide variant *further* from independent because it beat
    it by more). Prediction (2), wide's first-five newcomer share above the
    default's by two paired SE: failed in both — +0.027 (se 0.041) in the
    control, −0.006 (se 0.037) under the shock. Prediction (3), Beta-TS
    ahead of the default post-arrival in the control: held, 10.3 against
    28.6, 8.2 paired SE.
  - Post-arrival regret, mean, control / shock: Beta-TS 10.3 / 13.3;
    independent 22.4 / 30.7; flat 25.1 / 24.4; wide 26.3 / 24.6; default
    28.6 / 26.3. First-five newcomer share, control / shock: Beta-TS
    0.770 / 0.732; independent 0.681 / 0.584; flat 0.644 / 0.672; wide
    0.633 / 0.640; default 0.606 / 0.646.
  - Reading, no stronger than the design supports. Quadrupling the
    newcomer's prior variance did not measurably speed its recognition or
    lower post-arrival regret: the answer to "would a wider prior fix the
    cost" is no, at least not at 30 trials per arm. And at this sparsity
    Beta-TS beat every OR-TS variant post-arrival under the shock as well
    (13.3 against 24–31, 4.7 SE against the default) — the level-relearning
    advantage that survived an arrival at 100,000 trials (H19) does not
    survive one at 30, where a fresh intercept refit from roughly five
    events per batch and a Laplace approximation at n≈30 (H15: median KL
    1.4e-2 at n=100) cost more than a moving level does. Which of newcomer
    shrinkage and sparse refitting drives the gap is not separable in this
    design. No seed, horizon, repetition count or margin was changed after
    these outcomes were read.

## H20 — learning the decay from the data: windowed Laplace evidence over a λ grid (2026-09-12, before run)

Every OR-TS run so far fixes the decay λ in advance, and H9 priced that
choice: with nothing moving, λ = 0.1 roughly doubles the undisturbed regret
(2,226 against 1,152), while with the contrasts moving (H10, "both") the same
λ = 0.1 cuts regret from 21,342 to 8,183. A practitioner who cannot say ex
ante whether the contrasts will move has no fixed λ that is right in both
places. This hypothesis asks whether a data-driven λ removes the choice at an
acceptable price. The rule is fixed here, before any run, and is not tuned
afterwards.

- **Rule** (`src/orts/adaptive.py`, `AdaptiveDecayBandit`, a pure library
  object built on the shipped symmetric-prior `LogisticBandit`; fixed arm
  sets only). One contrast filter per grid decay λ ∈ Λ = {0, 0.05, 0.1, 0.3}
  absorbs every batch in parallel; each filter is byte for byte the
  fixed-decay OR-TS state it would be on its own. Before batch t filter λ
  carries N(μ_λ, S_λ⁻¹) after the usual re-referencing; it fits the batch
  with the tempered prior (1−λ)S_λ and the batch is scored by its Laplace
  log-evidence under that filter,

      E_t(λ) = ℓ(θ̂_λ) − ½(1−λ)(β̂_λ−μ_λ)ᵀS_λ(β̂_λ−μ_λ) + ½ log det((1−λ)S_λ) − ½ log det H_λ,

  where ℓ is the binomial log-likelihood at the mode, β̂_λ the fitted
  contrasts, and H_λ the fitted precision (contrast block plus intercept).
  Only constants shared by every filter are dropped — the binomial
  coefficients, the (2π) powers and the flat intercept's improper constant;
  the prior normalizer log det S_λ is kept because the filters carry
  different states. The filter used for allocation after batch t is
  argmax_λ Σ_{s≤t} E_s(λ), the cumulative one-step prequential score, ties
  to the smaller λ; the first batch is scored from the initial symmetric
  prior like any other. A batch the library cannot use (all events or none,
  a silent no-op) is declined by every filter alike and records nothing.
  Selection and memory are separate: a filter that is not allocating still
  absorbs the batch with its own decay, so a wrong pick costs one batch of
  allocation and never discards accumulated precision. The aggressiveness
  γ is not learned. This is discount-factor selection by predictive
  likelihood in the sense of West and Harrison; Supplement G's mapping
  between λ and an innovation variance is what makes it a type-II maximum
  likelihood over the memory length rather than a heuristic.
- **Amendment before any run (2026-09-12).** The rule as first written here
  kept a *single* state, tempered it by each grid λ, scored the four fits
  with the same evidence, and chose by a five-batch window. Unit tests on
  toy data — not on any registered configuration — showed two defects
  before a run existed: (i) with one shared state the per-batch score is
  flat in λ whenever the carried state is sharp relative to the batch (a
  correct λ = 0 beats λ = 0.3 by about 0.002 nats per batch against noise
  of order 0.06), so the window selection was a coin flip in exactly the
  stationary regime the rule must protect; (ii) a wrong pick under a single
  state permanently discards (1−λ) of the accumulated precision. The filter
  bank removes both: the score compares genuinely different memories, and
  memory is never touched by selection. Fixing (i) also exposed that the
  prior normalizer ½ log det S must be kept once the candidates differ in
  S. The window was dropped for the cumulative score, the parameter-free
  textbook form. Grid, configs, endpoints, predictions, margins and
  consequences below are unchanged from the first registration.
- **Config**: `sim/configs/adaptive_decay_sweep.json`, experiment
  `forgetting_comparison`, seed 20260913 (new; the H6/H9/H10 seeds are not
  reused because the added policies consume draws and would change the
  historical rows anyway). Five disturbance cells in one run, taken
  unchanged from the registered H6, H9 and H10 designs: common shock sd
  0.30; per-arm random walk sd 0.02; per-arm sd 0.06; both (per-arm 0.06
  with common 0.30); no disturbance (common sd 0). 10 arms, 40 rounds,
  100,000 trials per round, baseline 0.03, effects k·0.05, probability
  matching, 20 repetitions (the adaptive policy carries four filters, so it
  costs four fits per batch). Policies: the four historical
  (`beta_ts`, `beta_ts_discount`, `orts_l0`, `orts_decay`, flat prior) plus
  the symmetric-prior fixed grid `orts_l0_sym` (λ=0), `orts_l005_sym`
  (0.05), `orts_decay_sym` (0.1), `orts_l03_sym` (0.3), and the adaptive
  policy `orts_adapt_sym`. The decay of the filter selected for allocation
  after each batch is recorded in a `decay_selected` column (present only when the adaptive policy runs, so
  every historical config's row stream is unchanged).
- **Endpoint**: final-round cumulative regret per cell, paired within
  repetition. "Best fixed" in a cell is the fixed-grid symmetric policy with
  the lowest mean final regret in that cell — an oracle no practitioner has.
- **Predictions**. (1) *No loss anywhere*: in every one of the five cells,
  adaptive minus best-fixed is at most max(10% of the best-fixed mean,
  2 paired SE). (2) *Cheap insurance*: in the no-disturbance cell, the
  adaptive policy's paired premium over `orts_l0_sym` is less than half of
  `orts_decay_sym`'s paired premium over `orts_l0_sym` — the rule pays less
  than half of H9's fixed-forgetting price when nothing moves. (3) *Earns
  its keep*: in the "both" cell the adaptive policy beats `orts_l0_sym` by
  more than 2 paired SE — it learns to forget where H10 showed forgetting
  pays. Descriptive, no prediction: the mean selected λ per cell and its
  path over rounds; adaptive against the flat-prior `orts_decay` and against
  `beta_ts_discount`.
- **Falsified if**: (1) fails in any cell, or (2) fails, or (3) fails. A
  harder failure is also named: adaptive worse than the *worst* fixed-grid
  symmetric policy in any cell by more than 2 paired SE, which would mean
  the evidence rule is actively misled by the data rather than merely
  imprecise.
- **Consequence, fixed in advance**. If (1), (2) and (3) all hold, the
  manuscript (en7/kor7 lineage) presents learned λ as the default method —
  Section 5.1 and Algorithm 2 gain the selection step, the Section 4.1
  sentence about fixed settings is adjusted, and the Discussion's "learn λ
  online" future-work item is retired — while the practitioner edition keeps
  its simpler fixed-λ rule and points to this section. If (1) holds but (2)
  or (3) fails, learned λ is presented as an option with its measured cost,
  not as the default. If (1) fails, the fixed-λ prescriptions stand and the
  result is reported in Supplement F as a negative finding. No grid, window,
  seed, horizon, repetition count or margin will be changed after the
  outcome is read.
- **Result** (2026-09-12; run
  `20260912T021318Z_adaptive_decay_sweep_bde342b`, commit `bde342b`, clean
  tree; two earlier directories, `20260912T021105Z_..._ce9541f` aborted by a
  wrapper defect before any output and `20260912T021241Z_..._9cf55a2`
  stopped by hand a minute in, are kept with NOTES.md): **not supported as
  registered.** Predictions (2) and (3) held; prediction (1) failed in three
  of five cells. The harder failure named above did not occur.
  - (1) *No loss anywhere*: **failed** at `common:0.0`, `arm:0.06` and
    `both:0.06`; held at `common:0.3` and `arm:0.02`. Per cell below.
  - (2) *Cheap insurance*: **held.** At the origin the adaptive premium over
    `orts_l0_sym` is 184.1 (paired se
    78.0) against
    932.4 for `orts_decay_sym` — a
    fifth of the fixed-forgetting price, below the registered half.
  - (3) *Earns its keep*: **held.** In `both:0.06` the adaptive policy beats
    `orts_l0_sym` by 10,451.7 (se
    3,411.9,
    3.1 se).
  - Never worse than the worst fixed grid decay in any cell (nearest
    -964.9 at se
    698.2, `arm:0.02`).
  - Final cumulative regret by cell (mean over 20 reps; paired se where stated):
  - `common:0.0`: adaptive 1,429.2; fixed 0/0.05/0.1/0.3 = 1,245.1/1,620.9/2,177.5/4,599.7; best fixed `orts_l0_sym`, adaptive − best 184.1 (se 78.0) against margin 156.0: **outside**; adaptive − worst fixed -3,170.5 (se 116.8); mean selected λ 0.002; flat `orts_decay` − adaptive 959.9; `beta_ts_discount` − adaptive 734.2.
  - `common:0.3`: adaptive 1,295.0; fixed 0/0.05/0.1/0.3 = 1,268.1/1,591.2/2,313.1/4,790.7; best fixed `orts_l0_sym`, adaptive − best 26.9 (se 62.6) against margin 126.8: within; adaptive − worst fixed -3,495.7 (se 136.3); mean selected λ 0.001; flat `orts_decay` − adaptive 1,071.0; `beta_ts_discount` − adaptive 6,605.2.
  - `arm:0.02`: adaptive 3,850.4; fixed 0/0.05/0.1/0.3 = 3,871.3/2,967.4/2,937.7/4,815.3; best fixed `orts_decay_sym`, adaptive − best 912.8 (se 618.4) against margin 1,236.9: within; adaptive − worst fixed -964.9 (se 698.2); mean selected λ 0.020; flat `orts_decay` − adaptive -906.9; `beta_ts_discount` − adaptive -937.5.
  - `arm:0.06`: adaptive 7,364.1; fixed 0/0.05/0.1/0.3 = 19,944.3/13,193.7/7,732.7/6,502.1; best fixed `orts_l03_sym`, adaptive − best 862.0 (se 362.2) against margin 724.4: **outside**; adaptive − worst fixed -12,580.2 (se 4,781.8); mean selected λ 0.205; flat `orts_decay` − adaptive -20.2; `beta_ts_discount` − adaptive 1,715.6.
  - `both:0.06`: adaptive 15,116.6; fixed 0/0.05/0.1/0.3 = 25,568.3/13,177.1/9,012.3/7,793.7; best fixed `orts_l03_sym`, adaptive − best 7,322.9 (se 2,525.8) against margin 5,051.5: **outside**; adaptive − worst fixed -10,451.7 (se 3,411.9); mean selected λ 0.161; flat `orts_decay` − adaptive -6,984.3; `beta_ts_discount` − adaptive 1,613.2.
  - Mechanism, as far as the design shows: the mean selected decay leaves
    zero only from about round 3–5 and plateaus near 0.22–0.24 in the two
    drift cells (per-round paths in `summary.json`), so the first drifting
    batches are allocated from the full-memory filter; in the stationary
    cells the score is essentially always at zero (mean 0.001–0.002) and the
    15% origin premium over `orts_l0_sym` is paid in the early rounds before
    the cumulative score separates. Whether a shorter scoring window trades
    one for the other is a new hypothesis, not run.
  - **Consequence, as registered**: (1) failed, so the fixed-λ prescriptions
    stand and the learned decay is reported in Supplement F as a negative
    finding on its registered criterion, with the descriptive picture; it
    ships as an implemented option (`AdaptiveDecayBandit`), not the default.
    The practitioner edition's fixed rule is unchanged. No grid, window,
    seed, horizon, repetition count or margin was changed after these
    outcomes were read.

## H21 — learned decay on the Open Bandit position-1 environment (2026-09-12, before run)

H20 tested the learned decay (`AdaptiveDecayBandit`, cumulative Laplace
evidence over λ ∈ {0, 0.05, 0.1, 0.3}) only on the synthetic cells, where it
failed its registered price in the stationary cells. The Open Bandit
position-1 environment (H12b, H18) is the one measured environment in which
the contrasts actually move, so it is where a learned decay could earn its
keep; it is also very sparse (about three clicks per item-day, seven
batches), so the evidence scores may never separate the filters. This run
asks which of the two happens. Rule, grid, and window are those of H20,
unchanged.

- **Config**: `sim/configs/obd_position1_adaptive_decay.json` — the H18
  design verbatim (same aggregate and checksum, position 1, forget 0.1,
  20 repetitions, historical OR-TS entries with their independent N(0, 2^2)
  new-contrast prior, the two symmetric twins) plus `orts_adapt_sym`
  (symmetric prior, grid {0, 0.05, 0.1, 0.3}, cumulative window), fresh
  seed 20260914. `sim/experiments/obd_replay.py` gains the policy and a
  `decay_selected` column that exists only when it runs; historical row
  streams are unchanged.
- **Prediction (1)**: the learned decay's final cumulative regret is within
  max(10% of the better fixed mean, 2 paired SE) of the better of
  `orts_l0_sym` and `orts_decay_sym`, paired within repetition. This is the
  H20 criterion (1) on one cell.
- **Prediction (2), descriptive**: the mean selected λ per day is reported.
  The honest expectation is that at this sparsity the cumulative score
  barely leaves zero (mean selected λ below 0.05 on every day); this is an
  expectation, not a criterion.
- **Falsified if**: the learned decay is worse than the better fixed
  symmetric variant by more than the margin in (1). Beta-TS comparisons are
  reported without a prediction, as in H18.
- **Fixed in advance**: `n_reps = 20`, `seed = 20260914`, grid, window, and
  margin as stated; nothing is changed after the outcome is read.

- **Result** (2026-09-12; run
  `20260912T054821Z_obd_position1_adaptive_decay_561fb93`, commit `561fb93`;
  the tree was dirty only in files the run does not read, the other
  session's uncommitted figure PNG/PDFs and `en4.tex`, and
  `source_snapshot.json` preserves the exact source): **falsified as
  registered, on both counts.**
  - Final cumulative regret, mean (se): `orts_l0_sym` 5,396.9 (87.9),
    `orts_l0` 5,620.8 (94.9), `beta_ts` 5,814.4 (56.8), `orts_decay_sym`
    5,823.8 (57.6), `beta_ts_discount` 5,831.5 (40.9), `orts_decay` 5,891.5
    (70.9), `orts_adapt_sym` 5,953.5 (60.7).
  - Prediction (1): the better fixed symmetric variant was `orts_l0_sym`;
    adaptive minus it, paired within repetition, is +556.6 (se 76.9) against
    a margin of max(10% × 5,396.9, 2 × 76.9) = 539.7. Outside the margin, so
    (1) fails; against `orts_decay_sym` the learned decay is +129.7 (se
    90.1), against Beta-TS +139.1 (se 71.1).
  - Prediction (2), the descriptive expectation, was wrong in the opposite
    direction: the mean selected λ is 0.000, 0.035, 0.19, 0.29, 0.29, 0.29,
    0.30 across the seven days. Far from staying at zero, the cumulative
    Laplace evidence moved to the strongest forgetting in the grid by the
    fourth day and stayed there.
  - Reading: at about three clicks per item-day the tempered prior of the
    λ = 0.3 filter is the widest and fits each sparse batch best by the
    evidence score, but allocating from it discards the little that six
    batches can accumulate; the full-memory filter, which the score never
    preferred, ends far ahead. Seed-to-seed variation is also large here:
    the same `orts_l0_sym` was 5,662.5 in H18 (seed 20260912) and 5,396.9
    now, a 2 se move, so single-run margins on this environment are fragile
    either way.
  - **Consequence, as registered**: nothing changes in the manuscripts'
    prescriptions; the learned decay is reported in Supplement F as a second
    negative finding, now on the one measured environment where the
    contrasts move. No grid, window, seed, horizon, repetition count or
    margin was changed after the outcome was read.

## H22 — three decay-selection rules on the H20 filter bank (2026-09-12, before run)

H20 (synthetic) and H21 (Open Bandit) tested one selection rule, the
argmax of cumulative Laplace evidence over λ ∈ {0, 0.05, 0.1, 0.3}, and it
failed its registered price in both: in the stationary cells it paid a 15%
premium, and on the sparse measured environment the evidence preferred the
widest tempered prior and drove λ to 0.3 by the fourth day. This
hypothesis keeps the bank and the grid and changes only how the decay is
chosen. Three rules, fixed here before any run (`src/orts/adaptive.py`):

- **(a) `orts_bma_sym`, mixture.** Bayesian model averaging over the grid
  with a uniform prior: filter weights ∝ exp(cumulative evidence), and the
  allocation is the weighted average of the filters' winner probabilities.
  No argmax, so no lurch; with little evidence the rule hedges across the
  grid.
- **(b) `orts_reward_sym`, retrospective allocation reward.** Each filter is
  scored by the expected events its own allocation, formed at the previous
  boundary, would have earned at the next batch's observed per-arm rates
  (c/n over arms with exposure), cumulated; allocation from the argmax. The
  objective is allocation quality, not predictive fit. The first batch
  scores nothing.
- **(c) `orts_cont_sym`, continuous decay.** One decay λ ∈ [0, 0.5]
  re-estimated at every boundary by replaying the whole batch history from
  the template state and maximizing cumulative Laplace evidence plus the
  log density of a Beta(1, 4) prior on λ ((4−1)·log(1−λ)), by bounded
  scalar search with at most 10 evidence evaluations. The allocating state
  is the replayed filter at the chosen λ, byte for byte the fixed-decay
  state it would have produced. The prior is the regularizer H21 showed to
  be missing; with it the decay must be earned by evidence.

H20's rule (`orts_adapt_sym`) runs alongside with the same seed for a
like-for-like comparison. Historical row streams are unchanged; the
`decay_selected` column now carries each rule's decay (for the mixture,
its mean decay).

- **Configs**: `sim/configs/adaptive_rules_sweep.json` (the five H20 cells,
  seed 20260915, 20 repetitions, the four fixed symmetric grid twins and the
  four rules) and `sim/configs/obd_position1_adaptive_rules.json` (the H21
  design, seed 20260916, the four rules added).
- **Predictions, per rule, on the synthetic cells (H20's criteria)**: (1)
  within max(10% of the best fixed mean, 2 paired SE) of the best fixed grid
  decay in every cell; (2) with nothing moving, the paired premium over λ =
  0 is less than half of fixed λ = 0.1's; (3) where both level and contrasts
  move, ahead of λ = 0 by more than 2 paired SE.
- **Prediction, per rule, on Open Bandit position 1 (H21's criterion)**:
  within max(10%, 2 paired SE) of the better of `orts_l0_sym` and
  `orts_decay_sym`, paired within repetition; mean selected decay per day
  reported.
- **Expectations written down before running**: (a) passes (2) and comes
  closest to passing (1) everywhere but trails fixed 0.3 in the drift cells;
  (b) does best in the drift cells but is the noisiest on Open Bandit, where
  seven batches give its score little to work with; (c) sits between (a)
  and (b) and stays near zero on Open Bandit because of the prior. If the
  three rules split this way, the manuscript gains a when-to-use-which
  sentence; if all three fail (1) somewhere, learned decay is reported as
  failing on three rules and stays an option.
- **Falsified if**, per rule: (1) fails in any synthetic cell, or the Open
  Bandit margin is exceeded. Reported either way, per rule and per cell.
- **Fixed in advance**: seeds, repetitions, horizons, grid, prior Beta(1, 4),
  range [0, 0.5], 10 evaluations, margins; nothing changes after the
  outcomes are read. Runtime note: (c) replays the history at every
  boundary, so the synthetic run is expected to take about an hour.

- **Result, Open Bandit position 1** (2026-09-12; run
  `20260912T122357Z_obd_position1_adaptive_rules_ebbe1c1`, commit `ebbe1c1`,
  seed 20260916; a duplicate launch `20260912T123043Z_..._ebbe1c1` was
  stopped by hand and carries a note; the completed run's `NOTES.md` was
  written prematurely while it was still running and is corrected by
  `NOTES-correction.md`): **all four rules inside the margin; the reward
  rule is the only one that stays put.** Best fixed `orts_l0_sym` 5,520.4
  (se 69.7); margin max(10%, 2 paired SE) = 552.0. Paired differences from
  it: (a) mixture +389.7 (se 91.4), (b) reward +21.1 (se 126.1), (c)
  continuous +434.9 (se 132.9), H20's argmax rule +323.0 (se 95.5). Against
  Beta-TS, paired: (b) −302.5 (se 114.9, 2.6 SE), the others within 1.2 SE
  of zero. Mean selected decay by day: (a) 0.04 → 0.29, (c) 0.00 → 0.39 on
  day four then 0.26–0.30, H20's rule 0.00 → 0.29, (b) 0.00–0.10 throughout.
  Every evidence-based rule, the Beta(1, 4) prior included, still drifts to
  the strongest forgetting on this sparse log; the reward-based rule does
  not, because its score is the allocation's own return. The H21 rule that
  failed its margin at seed 20260914 (557 against 540) passes it here (323
  against 552): the margin on this environment is seed-fragile, as H21
  noted, so the Open Bandit verdict is read as "no rule detectably worse
  than the best fixed decay, and (b) at parity with it", not more.

- **Result, synthetic cells** (2026-09-12; run
  `20260912T122357Z_adaptive_rules_sweep_ebbe1c1`, commit `ebbe1c1`, clean
  tree apart from the other session's figure/en4 files the run does not
  read; 1 h 40 min wall clock, the continuous rule's replay dominating):
  **no rule passes criterion (1) in every cell; (2) holds for the
  continuous rule and H20's, (3) for all four.** Fixed symmetric grid, final
  cumulative regret: none 1,225/1,598/2,263/4,696 (λ = 0/0.05/0.1/0.3);
  common 0.3: 1,218/1,633/2,279/4,692; arm 0.02: 4,918/2,807/2,894/4,650;
  arm 0.06: 23,990/12,439/7,337/7,092; both: 26,565/12,012/9,701/7,043.
  Per rule, adaptive minus best fixed (paired se; margin), cells in the
  same order:
  - (a) mixture: +563 (89; 178) NO, +686 (65; 130) NO, +517 (240; 480) NO,
    +673 (439; 879) OK, +371 (370; 739) OK. Stationary premium over λ = 0
    563 = 46% (cheap insurance fails); in the drift cells it nearly matches
    the best fixed decay; mean selected decay 0.04 → 0.24.
  - (b) reward: +699 (226; 451) NO, +566 (189; 377) NO, +1,231 (791; 1,582)
    OK, +5,631 (2,732; 5,463) NO, +6,841 (2,536; 5,071) NO. Noisiest rule
    (stationary se 226 against 64–89 for the others) and worst under drift;
    premium 57%.
  - (c) continuous: +41 (64; 128) OK, +217 (130; 261) OK, +1,049 (472; 944)
    NO, +4,675 (1,844; 3,689) NO, +3,737 (1,320; 2,640) NO. Stationary
    premium 41 = 3% (cheap insurance holds); ahead of λ = 0 by 15,785 (se
    3,703) in the both cell but 3,737 behind fixed 0.3 there; mean selected
    decay stays at 0.005–0.13, the Beta(1, 4) prior holding it down.
  - H20's rule rerun at this seed: +56 (83; 165) OK, +240 (74; 148) NO,
    +1,501 (537; 1,074) NO, +4,374 (2,101; 4,203) NO, +12,234 (4,100; 8,199)
    NO; premium 5%.
  - The pre-run expectations held in part: (a) is the drift specialist and
    pays for it when nothing moves, (c) sits between; (b) was expected to do
    best under drift and did worst, its retrospective score being too noisy
    at forty batches to track a random walk.
- **Verdict of H22**: falsified per rule on criterion (1). Learned decay
  stays an implemented option, not a default, now on four selection rules
  and two environments. Two regularities are recorded for Supplement F: a
  prior on the decay (rule c) buys the stationary case for about 3% and
  captures most, not all, of the drift gain; and on the sparse measured
  environment every evidence-based rule drifts to the strongest forgetting
  while the reward-based rule does not. No design parameter was changed
  after the outcomes were read.

## H23 — closing the batch by event count on Open Bandit position 1 (2026-09-13, before run)

Sections 5 and 7 now say that decay is a tool for where a batch brings
enough evidence to replace what it discards, and that the batch length is
the operator's to set. On the Open Bandit position-1 log every policy
updates daily on about three clicks per item and none finds the best item
in a week (H12b, H18, H21, H22). This run asks the operational question
that follows: if the batch is closed by an event-count trigger instead of
the calendar day, so that updates are fewer and each carries more events,
does OR-TS lose anything?

- **Design** (`sim/experiments/obd_replay.py`, `batch_cells`): the H18
  environment and policies (Beta-TS, discounted Beta-TS, OR-TS with the
  historical N(0, 2^2) new-contrast prior with and without decay 0.1, and
  the two symmetric twins), seed 20260917, 20 repetitions. Within a cell
  each policy holds the allocation it formed at its last update, pools the
  days' exposures and events, and updates when the trigger is met or the
  week ends; regret is still scored daily against each day's best item.
  Cells: daily (the H18 loop restated in this path), pooled events >= 300
  (about two days at the log's volume), pooled events >= 600 (about three
  days), and every item >= 1 event (the first-fit-safe rule). The trigger
  reads the policy's own observed events, as a platform would. Fewer
  updates also mean fewer decay steps per week for the decay variants;
  that is part of the design, not corrected for.
- **Config**: `sim/configs/obd_position1_batch_trigger.json`.
- **Prediction (1)**: for each OR-TS variant and each triggered cell, final
  cumulative regret is within max(10% of the daily mean, 2 paired SE) of
  the daily cell, paired within repetition. Fewer updates do not cost.
- **Prediction (2), descriptive**: updates per week fall to about 3-4
  (>= 300), 2-3 (>= 600) and 2-4 (every item >= 1); reported.
- **Expectation written down before running**: no cell separates from
  daily beyond noise for any policy; at three clicks per item-day the
  week is too short for the trigger to change what is learned, so this is
  a check that the operational rule is safe, not a claimed gain. Beta-TS
  is reported without a prediction.
- **Falsified if**: any OR-TS variant in any triggered cell exceeds the
  margin. Reported either way.
- **Fixed in advance**: seed, repetitions, cells, margins; nothing changes
  after the outcome is read.

- **Result** (2026-09-13; run
  `20260913T011906Z_obd_position1_batch_trigger_4f711dc`, commit `4f711dc`;
  the tree was dirty only in the other session's figure PNG/PDFs and
  `en4.tex`, which the run does not read): **falsified as registered.**
  Prediction (1) fails for both full-memory OR-TS variants in the
  every-item-≥1 cell; the two-day trigger is free; the three-day trigger is
  inside the margin but costs the full-memory variants.
  - Updates per week: daily 7.0; ≥ 300 events 4.6–5.2; ≥ 600 events
    3.0–3.4; every item ≥ 1 event 1.6–1.7 (in effect one mid-week update
    plus the week-end one; the sparsest items need most of the week to
    reach one event).
  - Final cumulative regret, paired against daily within repetition
    (mean, se; margin max(10%, 2 SE)):
    - `orts_l0_sym` (daily 5,525.4): ≥ 300 −52.3 (126.1); ≥ 600 +359.4
      (113.5); every-item +708.6 (134.9) against 552.5, **outside**.
    - `orts_l0` (daily 5,631.8): ≥ 300 +34.4 (154.5); ≥ 600 +153.0
      (146.3); every-item +687.1 (123.9) against 563.2, **outside**.
    - `orts_decay_sym` (daily 5,836.0): ≥ 300 −246.5 (97.9, 2.5 SE
      better); ≥ 600 +11.4 (104.4); every-item +395.6 (62.7), inside.
    - `orts_decay` (daily 6,000.1): ≥ 300 −287.1 (83.5, 3.4 SE better);
      ≥ 600 +78.1 (81.8); every-item +333.8 (68.5), inside.
    - Beta-TS (no prediction): ≥ 300 +61.1 (47.4); ≥ 600 +294.1 (60.2);
      every-item +428.1 (56.1). Discounted Beta-TS: +30.6, +209.4, +328.4.
  - Reading. Pooling to about two days costs nothing and, for the decay
    variants, helps by 2.5–3.4 SE, because fewer updates mean fewer decay
    steps per week; pooling to three days costs the full-memory variants
    about 6%, and one or two updates a week costs everyone 6–13%. On this
    log the daily best item moves (Section 4.2 measured the contrasts
    moving more than the level), so an allocation held for several days
    trails the environment by more than the extra precision returns. The
    every-item rule is the wrong trigger here: it waits for the rarest
    item, which at three clicks per item-day means waiting most of the
    week. The operational rule that survives is "pool to the volume of
    about two days, not longer, and not until the rarest arm reports".
  - **Consequence, as registered**: (1) failed, so the manuscripts do not
    recommend event-count batches as such; Section 5's batch-length
    paragraph and Supplement F report the measured trade-off. No cell,
    threshold, seed or margin was changed after the outcome was read.


## Figure 3 click-recording replication (2026-09-14, before reruns)

User-requested descriptive alternative to the existing traffic-share figure:
plot absolute cumulative observed clicks, with cumulative expected clicks
recorded separately to distinguish sampling noise from allocation value.
This is a replication of H6/H9/H10, not independent confirmatory evidence.
Keep every original seed, horizon (40), repetition count (20), traffic count,
disturbance and policy; retain H6's two contrast-only controls as well as its
common shock. Use copies named figure3_clicks_h6/h9/h10.json with only an
output-recording flag added. Also run the mandatory no_drift_control.json.
Do not alter existing configs, runs, estimands, or the manuscript figure.

Prediction: enabling recording consumes no random draws and leaves all legacy
columns identical to an uninstrumented run with the same current code and
seed. Compare legacy columns to the original runs as a historical replication
audit, reporting any differences rather than replacing the old results.
Falsification of the instrumentation claim: any changed legacy value in the
same-code paired recording check. Arithmetic checks: observed cumulative
clicks equal cumulative recorded clicks; cumulative expected clicks plus
cumulative regret equal the cumulative oracle reward, within numerical error.
Show all repetitions over the full horizon; means and standard errors are
computed after accumulation within repetition. No new superiority hypothesis
or stopping rule is introduced; the new plot is for editorial comparison.

## H24 — continuous learning through traffic rollout and a level step (2026-09-14, before runs)

**Question.** Can the contrast state continue learning across a rollout that
changes traffic volume and the common success level, without resetting the
posterior? This is a synthetic operational stress test motivated by the
user's 2020 experience, not a reconstruction of those proprietary observations.

**Design fixed before outcomes.** Six configs in sim/configs/rollout_*.json:
none, traffic_only, level_up, level_down, both_up, both_down. Ten fixed arms,
logit effects k*0.05, baseline probability 0.03 for arm 0; 40 batches, 20
repetitions, seed 20260914 in every cell. Batches 1–20 have 10,000 trials;
from batch 21, traffic is either unchanged or increases tenfold to 100,000.
The common logit either stays fixed or takes a permanent +1/-1 step at that
boundary (odds multiplied by exp(+1)/exp(-1)); no other environmental drift.
The values specify a large stress scenario, not empirical calibration or a
searched threshold. Policy states and allocation rules are not reset or given
special exploration at rollout. Initial allocation is uniform; subsequent
allocation is ordinary probability matching with gamma=1. Arm contrasts and
ranking remain constant even when the level changes.

**Policies.** Beta(1,1) TS; discounted Beta-TS with lambda=0.1; OR-TS with
lambda=0; OR-TS with lambda=0.1. Both OR-TS policies explicitly use the current
symmetric proper prior (tau=sqrt(2)), rather than the legacy flat-prior lineage
of the earlier Figure 3. This is a new registered design, not a replacement
of H6/H9/H10. The same discount factor is applied per batch in both families.
Each repetition and policy has a reward RNG and a seeded legacy posterior
RNG derived from the passed Generator. The same seed in each config ensures
identical pre-rollout histories; no environmental draws depend on allocations.
Run the mandatory sim/configs/no_drift_control.json as an additional audit;
the factorial none and traffic_only cells are the matched controls.

**Primary prediction and falsification.** In both both_up and both_down,
Beta-TS minus no-decay OR-TS cumulative expected regret over batches 21–40
is positive by more than two paired SE. The primary conjunction is falsified
if either direction fails; report both, including flat or reversed results.
This is a prespecified descriptive two-SE criterion, not a multiplicity-adjusted
familywise significance claim. No equivalence prediction for the controls.

**Secondary, reported without directional thresholds.** Post-rollout regret
per trial (also shown per 1,000 trials), post-rollout mean best-arm traffic
share, final best-arm share, actual and expected cumulative clicks, and all
four policies' cumulative regret. Within each traffic setting, compare each
level-step cell against its no-step counterpart, paired by repetition; then
report the interaction [(Beta-OR loss-rate gap with traffic+step) minus
(Beta-OR gap with traffic only)] minus [(Beta-OR gap with step only) minus
(Beta-OR gap with neither)]. This separates a rollout interaction from the
mechanical multiplication of regret by traffic volume. Do not interpret it
as guaranteed positive; rapid replacement of stale counts may help Beta-TS.

**Checks and reporting.** All repetitions and all 40 batches; means +/-1 SE
computed after within-repetition accumulation. Preserve failed runs; do not
skip failed fits or tune after outcomes. Verify pre-rollout equality across
cells and invariant contrasts, traffic schedule, count conservation, reward
accounting and deterministic replay in tests. In each row preserve the
allocation counts and successes, actual/expected reward, regret and level.
Show all six conditions in both regret and best-arm-share figures, with a
marked rollout boundary; any panel-specific y scales start at zero and are
labelled. No manuscript replacement until the user decides after review.


## H24b — ASOS-scale rollout sensitivity (2026-09-14, before additional runs)

The user requested the ASOS excess-level-SD estimate as an empirical scale
anchor after the H24 +/-1-logit runs completed. H24's primary conjunction was
supported; it is retained, not tuned or replaced. This follow-up reuses the
registered six-cell design, seed, repetitions, traffic and rollout timing,
policies and endpoints. Only the permanent level-step magnitude changes to
0.34104577451497575, the full-precision `headline.median_excess_sd_alpha_logit`
from `results/processed/20260830T090754Z_asos_level_vs_contrast_9a7c7e9/summary.json`. Four new configs are
rollout_asos_level_up/down and rollout_asos_both_up/down; unchanged none and
traffic_only controls are reused from H24, not counted as new evidence.
The mandatory no_drift_control audit is the unchanged H24 control run.

The ASOS statistic is across-period excess dispersion of the level. It is
NOT an estimated shock innovation SD, a fitted jump distribution, or an
ASOS rollout observation. Taking a single permanent step equal to one such
SD is a scale-anchored sensitivity design. Baseline, gaps and traffic schedule
remain synthetic; no ASOS causal rollout claim is made.

Prediction: the H24 direction survives at this smaller scale, with Beta-TS
minus no-decay OR-TS post-rollout expected regret above two paired SE in BOTH
both_up and both_down. Falsified if either direction fails. All other outcomes
and interaction estimates remain descriptive. Report H24 and H24b together,
including every flat or reversed result. Identical seeds make the studies
correlated; these are sensitivity comparisons, not independent replications.
No further magnitude, seed, horizon or repetition tuning after these outcomes.


### H24/H24b result record (2026-09-14; after all prespecified runs)

H24 primary conjunction supported; H24b primary conjunction falsified.
Full raw IDs, all cells/policies, exposure-adjusted effects and interactions:
`docs/reviews/rollout_h24_20260914.md`, backed by
`results/processed/rollout_h24_20260914/summary.json` and
`results/processed/rollout_h24b_asos_20260914/summary.json`.
H24 rollout+up: Beta minus OR post regret 3983.875 (paired SE 1569.401);
rollout+down: 797.581 (87.918). H24b rollout+up: 1469.768 (806.051), below
the two-SE threshold; rollout+down: 764.888 (112.433), above it. Thus the
ASOS-scale upward direction does not pass, even though its mean is positive.
No seeds, repetitions, horizon, thresholds or configs were changed after
outcomes. All six cells have identical pre-rollout histories within each
suite. Controls are reused in H24b, not counted as independent evidence.


## H25 — ASOS replay with per-arm allocation recorded and a measurement-error environment (2026-09-15, before runs)

Motivation. The ASOS replay (H8, H17) treats each period's measured
per-variant rates as the truth. Two consequences were found in the
H17-posthoc re-analysis: (i) the raw file records only the control arm's
allocation, so the allocation diagnostic could be computed for the 57
two-arm experiments only; (ii) the measured rates carry sampling noise that
differs by arm because the original tests gave arms different traffic, so
the period's "best arm" and the regret against it are partly noise, the
paired repetition se covers policy noise only, and a lightly sampled arm
whose measured rate jumps between periods can attract a within-batch
policy (experiment 54a85a: variant 0's contrast sampling sd 0.094 against
0.023 for the other variants; OR-TS(0) locked onto it in 16 of 20
repetitions and lost 0.61% of expected events). This entry fixes (i) by
recording every arm's allocation and addresses (ii) with an environment that
models the measurement error and propagates it across repetitions.

Two runs, one experiment module (`sim/experiments/asos_replay.py`, extended
after this entry is written):

- **H25a**, config `sim/configs/asos_replay_measured_alloc.json`: the H17
  design verbatim (data, checksum, filter, forget 0.1, 20 repetitions, seed
  20260912, the six policies) with the environment unchanged
  (`environment: "measured"`) and two columns added to every row,
  `n_alloc_all` (every arm's allocation, pipe-joined in arm order) and
  `rates_all` (the period's environment rates in the same order).
  Registered check: on the pre-existing columns the run must reproduce
  H17's `results.csv` exactly (same seed, no additional random draws);
  a difference falsifies the implementation, not a hypothesis.
- **H25b**, config `sim/configs/asos_replay_shrunk.json`: the same design
  with `environment: "shrunk"` and seed 20260915.

The shrunk environment, fixed before any run. For each experiment, with
$y_{i,t}$ the measured log odds of arm $i$ in period $t$ and
$s_{i,t}^2 = 1/\text{events} + 1/\text{non-events}$ its sampling variance:

1. Fit the weighted two-way additive model $y_{i,t}\approx\alpha_t+\beta_i$
   with weights $1/s_{i,t}^2$ and $\beta_{\text{control}}=0$, by
   alternating weighted means for 100 iterations from $\beta=0$.
2. Residual $r_{i,t}=y_{i,t}-\alpha_t-\beta_i$. Per arm,
   $\tau_i^2=\max\{0,\ \overline{r_{i,\cdot}^2}-\overline{s_{i,\cdot}^2}\}$
   with plain (unweighted) means over the arm's periods; no cap and no
   sharing across arms. This is the excess variance of Section 4.2 applied
   to the residual.
3. Shrinkage $k_{i,t}=\tau_i^2/(\tau_i^2+s_{i,t}^2)$ (zero when
   $\tau_i^2=0$); posterior mean $m_{i,t}=\alpha_t+\beta_i+k_{i,t}r_{i,t}$,
   posterior variance $v_{i,t}=k_{i,t}s_{i,t}^2$. Uncertainty in
   $\alpha_t$, $\beta_i$ and $\tau_i^2$ is ignored, and this is a
   limitation stated here.
4. In each repetition, before any policy runs, draw
   $\theta_{i,t}\sim N(m_{i,t},v_{i,t})$ independently for every
   $(i,t)$ from the repetition's own generator, and use
   $\text{expit}(\theta_{i,t})$ as that repetition's environment: outcomes
   are binomial in it, the period's best arm and the regret are taken in
   it, and the level-tracking target is its next-period control rate.
   Period volumes are unchanged.

The level $\alpha_t$ is free in every period and $\tau_i^2$ is estimated,
not set to zero, so the environment allows contrasts to move; the additive
decomposition is nevertheless the paper's own, which is why H25b is a
sensitivity analysis beside the registered H8/H17 replay, not a
replacement for it.

Analysis (`scripts/analyze.py`, `asos_replay` branch, extended after this
entry): the registered H8 quantities as before; the H17-posthoc quantities
now for all 71 experiments using `n_alloc_all` and `rates_all`, with the
better arm of an experiment defined as the arm with the highest
volume-weighted mean of that repetition's environment rates over the
replayed periods; and the count of experiments whose paired
Beta-TS-minus-OR-TS(0) final-regret difference exceeds two repetition se in
absolute value.

Predictions for H25b, all 71 experiments retained (structural, not a
prediction):

- **P1**: the number of experiments separated beyond two paired se between
  `beta_ts` and `orts_l0_sym` is at most 10 (H17: 31, of which 17 with
  OR-TS worse and 14 with Beta-TS worse).
- **P2**: every Beta-TS variant keeps a higher mean experiment regret than
  every OR-TS variant, as in H17 (direction only; no se threshold).
- **P3**: `orts_l0_sym`'s expected-event gain against `beta_ts` is within
  ±0.1% in at least 50 of the 71 experiments (H17: 57).
- **P4** (reported, not a pass/fail criterion): experiment 54a85a's
  `orts_l0_sym` gain moves from −0.61% to within ±0.1%.

Falsified if P1, P2 or P3 fails. Whatever happens, both runs, every policy
and every experiment are reported; no seed, repetition count, shrinkage
rule, cap or threshold is changed after the outcomes. The allocation shares
on all 71 experiments from H25a replace the two-arm subset in the
manuscript's Section 4.3 whether or not they look better.


### H25 result record (2026-09-15; after both runs, nothing changed afterwards)

Runs `20260915T134105Z_asos_replay_measured_alloc_05a47e1` (H25a) and
`20260915T134109Z_asos_replay_shrunk_05a47e1` (H25b); verdicts and all
per-policy fields in each run's `results/processed/<run>/summary.json`
(`posthoc_allocation`, and `h25b_registered_verdict` for H25b), rows in
`allocation_by_experiment.csv`.

- **H25a reproduction check: passed.** The pre-existing columns are
  identical to H17's `results.csv`, row for row. With every arm's
  allocation recorded, the final-period share on the better arm over all 71
  experiments is 0.526 for Beta-TS (below one half in 34) and 0.732 for
  OR-TS(0, symmetric) (below one half in 11); traffic-weighted over all
  periods 0.525 and 0.656. The two-arm subset had given 0.511 (29 of 57)
  and 0.774 (4 of 57): the 14 three- and four-arm experiments add 5
  below-majority endings for Beta-TS and 7 for OR-TS, so OR-TS's advantage
  on this diagnostic is smaller than the two-arm subset suggested and the
  manuscript now cites the all-experiment numbers.
- **H25b, P1 falsified.** 22 experiments are separated beyond two paired
  se between `beta_ts` and `orts_l0_sym` (threshold 10; H17 31): 6 with
  OR-TS worse and 16 with Beta-TS worse (H17: 17 and 14). The posterior
  draws add environment variability to the repetition se, but at these
  volumes the policies are close to deterministic and the repetition se
  stays small, so separation counts remain above the registered bound.
  The direction of the separations changed even though their number did
  not fall enough.
- **P2 met.** Mean experiment regret: `beta_ts` 4848.2, `beta_ts_discount`
  4885.7, `orts_decay_sym` 3692.2, `orts_decay` 3667.6, `orts_l0` 3394.3,
  `orts_l0_sym` 3316.7; every Beta-TS variant above every OR-TS variant.
  All regrets are about half their measured-environment values because
  the shrinkage removes the sampling noise from the per-period best arm.
- **P3 met.** `orts_l0_sym` within ±0.1% of Beta-TS in 58 of 71
  experiments (threshold 50; measured 57).
- **P4 not met.** 54a85a: `orts_l0_sym` gain −0.24% (measured −0.61%),
  paired z −1.5 (measured −5.6). The loss halves and stops being
  resolved from zero, but does not enter ±0.1%.
- **Beside the registered predictions**, reported in full: `orts_l0_sym`
  is ahead of Beta-TS in 51 of 71 experiments (measured 33; binomial sign
  test p = 0.0003 against 0.64), median gain +0.014% (measured −0.000%),
  mean +0.060% (measured +0.034%), range −0.24% to +1.75%. Beta-TS ends
  below a majority on the better arm in 27 of 71, OR-TS(0) in 11 (final
  period), 0.555 against 0.751 on average. 162a38, the largest H17
  separation in Beta-TS's favour (z −12.4), becomes +0.003% at z 0.4.
- **Reading.** The measurement-error environment removes most of the
  noise-driven separations in Beta-TS's favour and leaves the ones in
  OR-TS's favour, so the per-experiment sign, uninformative on the
  measured environment, favours OR-TS here; the magnitudes stay within
  0.1% for most experiments as before. The verdict on H25b as registered
  is falsified through P1, and the manuscript reports it as a sensitivity
  analysis with that outcome stated. The registered shrinkage is the
  moment-method normal approximation of the binomial GLMM
  logit p = alpha_t + beta_i + eps_it, eps_it ~ N(0, tau_i^2); at ASOS's
  event counts (hundreds to tens of thousands per arm-period) the
  approximation is not distinguishable from the full likelihood fit, and
  the two differ only in how tau_i^2 is estimated.


## H26 — Open Bandit position-1 replay on a measurement-error environment (2026-09-16, before run)

Motivation. The position-1 replay (H12b, H18) treats each day's measured
item click rate as the truth. At about 2 clicks per item-day (median; 97 of
the 560 item-days have none) the measured daily rate is mostly sampling
noise, so the day's "best item" and the regret against it are largely
noise, as H12b already noted. H25b addressed the same problem on ASOS; this
entry applies the same construction here, where it matters more.

- **Config**: `sim/configs/obd_position1_shrunk.json` — the H18 design
  verbatim (same aggregate and checksum, position 1, forget 0.1, 20
  repetitions, historical OR-TS entries with the independent N(0, 2^2)
  new-contrast prior, the two symmetric twins) with
  `environment: "shrunk"` and seed 20260916. `environment: "measured"`
  (the default) leaves every earlier row stream byte-identical.
- **The shrunk environment, fixed before the run.** With $c_{i,t}$ clicks
  in $n_{i,t}$ impressions of item $i$ on day $t$:
  $y_{i,t}=\log\{(c_{i,t}+\tfrac12)/(n_{i,t}-c_{i,t}+\tfrac12)\}$ and
  $s_{i,t}^2=1/(c_{i,t}+\tfrac12)+1/(n_{i,t}-c_{i,t}+\tfrac12)$ (the
  half-count correction is needed because zero-click item-days exist;
  this is the one departure from the H25b rule). Fit $y\approx\alpha_t+\beta_i$
  by alternating weighted means with weights $1/s^2$, item 0 as reference,
  100 iterations. Residual $r$. One shared
  $\tau^2=\max\{0,\ \overline{r^2}-\overline{s^2}\}$ over all item-days
  (seven days per item are too few for a per-item estimate; the second
  departure from H25b, stated). $k_{i,t}=\tau^2/(\tau^2+s_{i,t}^2)$,
  $m=\alpha_t+\beta_i+k\,r$, $v=k\,s^2$. Each repetition draws
  $\theta_{i,t}\sim N(m_{i,t},v_{i,t})$ independently before any policy
  runs and uses $\mathrm{expit}(\theta)$ as its environment: outcomes,
  the day's best item and the regret are all taken in it. Impressions per
  day are unchanged. Uncertainty in $\alpha$, $\beta$ and $\tau^2$ is
  ignored.
- **Recorded**: the existing columns plus `rates_all` (that repetition's
  environment rates, pipe-joined in item order), written only when the
  environment is shrunk.
- **Analysis** (`scripts/analyze.py`): the existing headline, paired
  differences and all-pairs table; plus $\tau^2$, the median shrinkage
  factor, and for each repetition the fraction of days on which the drawn
  best item is the item with the highest fitted $\beta_i$.
- **Predictions**:
  - **P1**: paired within repetition, Beta-TS minus `orts_l0_sym` final
    cumulative regret stays within two paired se, as on the measured
    environment (H18: +203.7, se 117.9). This is the dataset where the
    contrast bet fails (H12a) and clicks are sparse, so no OR-TS advantage
    is predicted; a separation in either direction is reported as a
    finding about what the measured-environment noise concealed.
  - **P2**: each decay twin (`orts_decay_sym` against `orts_l0_sym`,
    `beta_ts_discount` against `beta_ts`) stays within two paired se of
    its no-decay counterpart.
  - **P3**: every policy's mean final cumulative regret is at least 30%
    below its H18 value, because the shrinkage removes most of the
    sampling noise from the daily best item.
- **Falsified if** P1, P2 or P3 fails. Every policy and comparison is
  reported either way; no seed, repetition count, correction, shrinkage
  rule or threshold is changed after the outcome.


### H26 result record (2026-09-16; after the run, nothing changed afterwards)

Run `20260915T142538Z_obd_position1_shrunk_990f885`; verdict in its
`results/processed/<run>/summary.json` under `h26_registered_verdict`.
Environment: shared excess variance 0.074 (sd 0.27 on the logit), median
shrinkage factor 0.16; the drawn best item is the item with the highest
fitted effect on 23.6% of days, so the week remains Contrast-varying under
the shared excess variance.

- **P1 met.** Beta-TS minus `orts_l0_sym` +95.9, paired se 81.4 (1.2 se).
  As on the measured environment (H18: +203.7, se 117.9), no pair between
  the families is resolved; no OR-TS advantage is claimed here.
- **P2 falsified.** `orts_decay_sym` minus `orts_l0_sym` +303.0, paired se
  88.1 (3.4 se); `beta_ts_discount` minus `beta_ts` +166.3, paired se 76.5
  (2.2 se). On the measured environment forgetting had cost nothing
  detectable, which H12b called a weak null; with the sampling noise taken
  out of the daily best item, forgetting costs both families.
- **P3 met.** Mean final cumulative regret against H18: `beta_ts` 3388.1
  vs 5866.3 (−42%), `beta_ts_discount` 3554.3 vs 5809.5 (−39%),
  `orts_l0` 3303.7 vs 5722.6 (−42%), `orts_decay` 3633.9 vs 5858.3
  (−38%), `orts_l0_sym` 3292.2 vs 5662.5 (−42%), `orts_decay_sym` 3595.2
  vs 5615.3 (−36%); every reduction above the registered 30%.
- Final share of the day's best item stays low for every policy (0.10 to
  0.23), and every policy ends below a majority in 16 to 19 of the 20
  repetitions: with 80 items, a 0.35% click rate and about three clicks
  per item-day, one week does not identify the best item for any of them.
- The analyzer was also fixed in passing (commit 54e44bb): the cumulative
  click summary added for H24 assumed click-recording columns and crashed
  on replay rows; the guard changes no number.


## H27 — Open Bandit replay pooled over the three slots, measured and measurement-error environments (2026-09-16, before runs)

Motivation. The position-1 replays (H12b, H18, H26) use one slot and
therefore about three clicks per item-day. In the random bucket the three
slots have the same click rate (1622, 1590 and 1556 clicks on 458k
impressions each; chi-square 1.4 on 2 df, p = 0.50, computed before this
entry), so pooling the slots triples the events per item-day (median 7
clicks, 5 zero-click item-days of 560) without a position level to absorb.
Items do interact with position (per-item 2x3 tests pooled over days:
chi-square 327 on 160 df, chi-square/df 2.0), so the pooled target is not
the position-1 rate but the item's slot-averaged rate, the quantity a
service faces when it chooses which items to show but not where. H7 used
this pooled environment with the flat prior and four policies; this entry
reruns it with the H18 policy set and adds the measurement-error
environment of H26.

- **H27a**, config `sim/configs/obd_pooled_measured.json`: the H7 design
  (same aggregate and checksum, all positions, forget 0.1, 20 repetitions)
  with the H18 policy set (historical OR-TS with the independent N(0, 2^2)
  new-contrast prior, the two symmetric twins, Beta-TS and its discounted
  variant), `environment: "measured"`, seed 20260916. The reference for P3.
- **H27b**, config `sim/configs/obd_pooled_shrunk.json`: the same with
  `environment: "shrunk"`, seed 20260917; the H26 rule verbatim
  (half-count logits, weighted level-plus-item fit, one shared excess
  variance, one posterior draw per repetition).
- **Analysis** (`scripts/analyze.py`): the existing headline and pairwise
  tables; for the shrunk run the environment diagnostics of H26 and the
  verdict below, with H27a as the reference; the position main-effect and
  item-by-position interaction tests recorded from the aggregate.
- **Predictions** (H27b):
  - **P1**: Beta-TS minus `orts_l0_sym` within two paired se. No OR-TS
    advantage is predicted on this Contrast-varying log.
  - **P2**: forgetting costs both families: `orts_decay_sym` minus
    `orts_l0_sym` and `beta_ts_discount` minus `beta_ts` both positive and
    beyond two paired se, as found in H26 at one third of the events.
  - **P3**: every policy's mean final cumulative regret at least 30% below
    its H27a value.
- **Falsified if** P1, P2 or P3 fails. Everything is reported either way;
  nothing is changed after the outcomes.


### H27 result record (2026-09-16; after both runs, nothing changed afterwards)

Runs `20260915T213759Z_obd_pooled_measured_6069ff8` (H27a) and
`20260915T213954Z_obd_pooled_shrunk_6069ff8` (H27b); the H27b verdict is in
its `summary.json` under `h27_registered_verdict`, the position tests under
`environment` of both. Pre-check as recorded: position main effect
chi-square 1.40 on 2 df (p 0.50); item-by-position interaction 327.0 on
160 df.

- **H27a** (measured): mean final regret `orts_decay` 8802.2 (se 196.8),
  `orts_decay_sym` 8874.0 (203.6), `orts_l0_sym` 8935.2 (228.0), `orts_l0`
  8962.0 (225.2), `beta_ts_discount` 9095.0 (132.1), `beta_ts` 9104.7
  (155.4); no family pair resolved, as in H7.
- **H27b, P1 met.** Beta-TS minus `orts_l0_sym` −99.5, paired se 238.8.
- **P2 falsified.** `orts_decay_sym` minus `orts_l0_sym` −76.5 (se 217.1);
  `beta_ts_discount` minus `beta_ts` +324.0 (se 195.0, 1.7 se). Neither
  forgetting variant is resolved from its twin; the H26 finding at one
  slot does not carry to the pooled environment at this repetition count.
- **P3 falsified.** Reductions against H27a: `beta_ts` 23%,
  `beta_ts_discount` 19%, `orts_l0` 22%, `orts_decay` 15%, `orts_l0_sym`
  20%, `orts_decay_sym` 21%; all below the registered 30%.
- **Why the environment moved less than at one slot.** The shared excess
  variance is 0.114 (position 1: 0.074) and the median shrinkage factor
  0.46 (0.16): with three times the impressions per cell the sampling
  variance is a third, so the measurements are trusted more, and the
  slot-averaged item contrasts genuinely move more day to day than the
  position-1 contrasts, consistent with the item-by-position interaction.
  The drawn best item is the top fitted item on 18% of days (position 1:
  24%). Every policy still ends below a majority on the day's best item in
  17 to 20 of the 20 repetitions.
- **Reading.** Pooling the slots triples the events but does not change
  the conclusions: no advantage for either family on this Contrast-varying
  log, and the cost of forgetting seen at one slot is not resolved here.
  Both falsified predictions are reported as such.
