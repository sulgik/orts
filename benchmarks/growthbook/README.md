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

## Limits

One environment, not a sweep; binomial metrics only; and the SQL reduction is reconstructed, not
run.
