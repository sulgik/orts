# OR-TS against GrowthBook's bandit engine

`gbbench.py` runs GrowthBook's own bandit code (`gbstats`, `BanditsSimple`) next to OR-TS on a
binomial metric, with and without a shift common to every arm. `calibration.py` checks the
variance `gbstats` reports for its period-weighted mean.

GrowthBook's bandit reduces each variation's per-period data in SQL before `gbstats` sees it. That
reduction is rebuilt in Python here from `packages/back-end/src/integrations/sql/ctes/bandit-statistics-cte.ts`;
the docstring of `gbbench.py` gives the formulas and the statistic path a binomial metric takes.
It is a reading of the query, so please check it against the real one.

## Setup

PyPI's `gbstats` (0.8.0) predates bandits, so install it from GrowthBook's source. It pins a pandas
that does not build on Python 3.13; 3.11 works.

```bash
git clone --filter=blob:none --sparse https://github.com/growthbook/growthbook.git
git -C growthbook checkout 48658ccf4dd1f2a5792f2135db7ad3b6b3c0f0fa   # main, 2026-09-17
git -C growthbook sparse-checkout set packages/stats
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e growthbook/packages/stats "orts>=2.3"
python gbbench.py --reps 20
python calibration.py
```

## Results

Five arms, log-odds contrasts 0 to 0.20, 3% base rate, 40 periods of 20,000 users, a 1% floor for
every policy. GrowthBook runs with the settings the product passes to the stats engine
(`weight_by_period=True`, `top_two=False`, `min_variation_weight=0.01`). Cumulative regret, lower
is better; each policy sees the same 20 environment draws.

| | common shifts (sd 0.3) | no shifts |
|---|---|---|
| OR-TS | 420.7 | 360.4 |
| GrowthBook as shipped (binomial, period-weighted) | 565.4 | 548.8 |
| GrowthBook unweighted (pooled) | 798.2 | 400.0 |
| GrowthBook period-weighted, SQL variance (diagnostic) | 642.0 | 621.6 |

GrowthBook as shipped minus OR-TS: +144.7 ± 32.6 (4.4 SE, OR-TS lower in 18/20) with shifts, and
+188.4 ± 28.5 (6.6 SE, 19/20) without. Pooled and OR-TS are close without shifts (+39.6 ± 21.8,
1.8 SE), which is the check that the harness is sound.

Period weighting does its job under shifts (565 against 798 pooled). It costs efficiency otherwise
(549 against 400): each period's mean is weighted by the period's share of all users, not by how
many users that variation got in it, so a variation's thin periods count as much as its thick
ones. Using the SQL's variance instead does not recover this, so the cost is the estimator rather
than its calibration.

The calibration is off as well. On the allocation GrowthBook's bandit produced in a no-shift run,
the period-weighted mean's actual variance is 2.0–2.3 times the reported one for the trailing
variations and 1.2 times for the leader, because the binomial path reports `sum_squares = sum`.

## Contextual bandits

`ctxbench.py` does the same for contexts. GrowthBook's contextual engine is TypeScript
(`packages/stats-ts/src/contextualBanditWeights.ts`): it takes the decision metric pooled over the
whole run per (context, variation), fits a greedy regression tree over the context attributes, and
runs Gaussian Thompson sampling inside each leaf. `gb_contextual_driver.ts` wraps that function so
the benchmark can call the engine itself over stdin. It is compared with
`orts.ContextualLogisticBandit`, one logistic model over arm-by-cell contrasts with a fresh
intercept per cell and period.

```bash
git -C growthbook checkout b70f8c77d   # main, 2026-10-01; bandits.py is unchanged since 48658cc
git -C growthbook sparse-checkout set packages/stats packages/stats-ts packages/shared
npm install esbuild @stdlib/stats@0.1.1
GB=$PWD/growthbook/packages
npx esbuild gb_contextual_driver.ts --bundle --platform=node --format=cjs --outfile=driver.cjs \
  '--external:@stdlib/*' --alias:stats-ts=$GB/stats-ts --alias:shared/constants=$GB/shared/src/constants.ts \
  --alias:shared/experiments=$GB/shared/src/experiments/contextual-bandit-condition.ts \
  --alias:shared/types=$GB/shared/types --alias:shared=$GB/shared/src
pip install -e growthbook/packages/stats -e ../..
python ctxbench.py --driver driver.cjs --reps 20
```

Four arms with main log-odds contrasts 0 to 0.15 and two attributes, region (3 levels) and device
(2 levels), so six cells holding 35% down to 6% of traffic. A cell's contrasts are the main ones
plus a region effect and a device effect per arm, drawn once per run with sd 0.2, or all zero when
the best arm is the same everywhere. That structure is additive in the attributes, which is what a
tree splitting one attribute at a time is built for. Cells differ in base rate (3%, sd 0.3 on the
log-odds), 40 periods of 20,000 users, a 1% floor for every policy, `maxLeaves` 8. Regret is
against each cell's own best arm; each policy sees the same 20 environment draws.

| | best arm differs by cell | one best arm everywhere | differs, common shifts (sd 0.3) | one best arm, common shifts |
|---|---|---|---|---|
| Contextual OR-TS | 819.7 | 336.9 | 820.6 | 364.6 |
| GrowthBook contextual (as shipped) | 1221.8 | 526.1 | 1453.3 | 768.2 |
| GrowthBook non-contextual (as shipped) | 3296.6 | 401.8 | 3468.7 | 415.7 |
| OR-TS, context ignored | 3221.8 | 295.7 | 3384.1 | 364.9 |
| OR-TS, independent per cell | 798.7 | 728.0 | 846.6 | 691.3 |
| Contextual OR-TS, `interaction_sd` 0.02 | 2022.3 | 337.8 | 1987.8 | 295.5 |
| Contextual OR-TS, `interaction_sd` 0.25 | 807.2 | 692.7 | 847.5 | 668.1 |

GrowthBook contextual minus contextual OR-TS: +402.0 ± 93.9 (4.3 SE, higher in 15/20) where the
best arm differs, +189.1 ± 57.1 (3.3 SE, 17/20) where it does not, and with common shifts
+632.8 ± 115.9 (5.5 SE, 20/20) and +403.6 ± 116.6 (3.5 SE, 19/20).

What the rows say:

- **No fixed amount of pooling works in both worlds.** An independent OR-TS per cell is as good as
  anything when the best arm differs (799) and wastes the data when it does not (728 against 296
  for ignoring the context). `interaction_sd` 0.02 is the reverse. Estimating it by marginal
  likelihood after each batch lands on the better end both times (820 and 337).
- **GrowthBook's tree splits on the outcome level, not on the contrast.** Where one arm is best
  everywhere it still ends with 4.0 leaves, because the cells differ in base rate, and each leaf
  then learns the same ranking from a fraction of the data. With the base-rate spread set to zero
  it keeps one leaf and does well: 226 against 361 for contextual OR-TS over 10 runs. The logistic
  model gives each cell its own intercept, so a base-rate difference costs it nothing and only a
  difference in contrasts separates cells.
- **The contextual engine has no period weighting.** Its input is pooled over the run
  (`getBanditDates` returns nothing for a contextual bandit, so the period reduction is skipped), so a common shift biases the pooled
  rates as it does for the unweighted non-contextual bandit above. Its regret rises from 1222 to
  1453 and from 526 to 768 under shifts; contextual OR-TS stays at 820 and goes from 337 to 365.

## Limits

One environment per benchmark, not a sweep; binomial metrics only; and the non-contextual SQL
reduction is reconstructed, not run. In the contextual benchmark the cells are the full cross of
the attribute levels, six of them. With many attributes that cross is too fine to use directly,
and the cells would have to come from somewhere, such as the leaves of a tree like GrowthBook's.
