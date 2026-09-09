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
- **Result**: pending.

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
