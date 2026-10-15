# Contrast-splitting tree and per-attribute components: pre-registration

Written and committed on 2026-10-15, before any of the runs below. The results go in
`README.md` and are judged against the criteria here, as written. Anything not listed here is
exploratory and reported as such. Before writing this, the script was only run on environment
seed 100 for two periods, to check that every policy runs; those numbers are not reported.

## Why

In the pre-registered robustness runs (`PREREG_contextual_robustness.md`), H3 failed in all four
L24 scenarios: OR-TS refitted on GrowthBook's tree leaves was no better than GrowthBook itself.
The explanation offered there was that GrowthBook's tree splits on how much a segment converts,
not on which arm wins there, so its leaves merge cells whose best arms differ. That explanation
was not tested. Two things could be at fault: the splitting criterion, or partitioning 24 cells
into eight leaves at all. This experiment separates them by keeping the partition (same input,
same leaf cap, same leaf minimum) and changing only the criterion.

The same scenarios also give the first test of the contextual model with one variance component
per attribute (`ContextualLogisticBandit(..., attributes=[...])`), added after those runs.

## Design

Script: `ctxuplift.py`, which imports the environment, layouts, GrowthBook engine wrapper and
`run` from `ctxrobust.py` unchanged. GrowthBook's contextual engine is
`computeContextualBanditWeights` from `packages/stats-ts` at commit `b70f8c77d`, the same bundle
as the robustness runs (`driver.cjs`, sha256 `8e1f25b7…6a53b`), `maxLeaves` 8,
`minUsersPerLeaf` 100.

Scenarios: Part B of the robustness runs, unchanged. Layout L24 (region 4 levels, device 2,
plan 3), heterogeneity additive and residual, common-shift sd `sigma` 0 and 0.3; four scenarios.
4 arms, 40 periods of 20,000 users, 1% floor, 20 paired runs on environment seeds 0 to 19.
The measure is cumulative regret against each cell's own best arm.

**The contrast tree.** Input: the run's pooled (cell, arm) counts so far, the same input
GrowthBook's tree gets. A leaf's candidate splits are, for each attribute, every bipartition of
the attribute's levels present in the leaf, with at least 100 users on each side. A split's score
is the likelihood-ratio statistic for the arm-by-side interaction in the leaf's (side, arm) table:
twice the log-likelihood of a separate rate per (side, arm) minus that of
`logit p = alpha[side] + theta[arm]`. A base-rate difference between the sides therefore scores
nothing; only a difference in the arms' log-odds contrasts does. The tree grows best-first,
always splitting the leaf whose best split scores highest, until it has eight leaves or no
admissible split remains. It is rebuilt from scratch every period.

**Policies.**

- **Contextual OR-TS**: `ContextualLogisticBandit` over the full 24-cell cross,
  `interaction_sd="auto"`, as in the robustness runs.
- **Contextual OR-TS, per-attribute components**: the same over the full cross with
  `attributes=["region", "device", "plan"]`, every component `"auto"`.
- **GrowthBook contextual**: the engine, as shipped.
- **Hybrid, GrowthBook leaves**: the robustness runs' hybrid, unchanged.
- **Hybrid, contrast-tree leaves**: the same hybrid with the contrast tree's leaves in place of
  GrowthBook's: OR-TS is refitted every period from the whole history with the current leaves as
  cells (`allocate_cells_from_rows`), and every cell gets its leaf's allocation.

## Hypotheses and failure criteria

A difference "holds" when the mean paired difference in regret is more than two standard errors
in the stated direction.

- **H5 (the criterion).** In all four scenarios the contrast-tree hybrid has lower regret than
  GrowthBook contextual. *Fails* if any one of the four does not hold.
- **H6.** In all four scenarios the contrast-tree hybrid has lower regret than the hybrid on
  GrowthBook's leaves. *Fails* if any one does not hold.
- **H7 (the partition).** In both residual scenarios, contextual OR-TS over the full cross has
  lower regret than the contrast-tree hybrid: when the contrasts do not decompose by attribute,
  eight leaves must merge cells with different best arms whatever the criterion. *Fails* if
  either does not hold.
- **H8.** In both additive scenarios, the per-attribute model has lower regret than contextual
  OR-TS with one exchangeable component. *Fails* if either does not hold.
- **H9.** In both residual scenarios, the per-attribute model's mean regret is no more than 5%
  above contextual OR-TS's: with no attribute structure to find, the extra components should cost
  little. *Fails* if either is more than 5% above.

**How the outcome will be read.** If H5 and H6 hold, H3 failed because of GrowthBook's splitting
criterion. If they fail, the eight-leaf partition is the bottleneck whatever the criterion, and
the README's conclusion that cells should come from the attributes stands without qualification.

**Exploratory, no prediction:** the contrast-tree hybrid against contextual OR-TS in the additive
scenarios; the per-attribute model against GrowthBook; both trees' leaf counts; whether the
re-run GrowthBook and GrowthBook-leaf hybrid columns match the robustness runs' numbers.
