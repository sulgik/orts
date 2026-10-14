# Contextual robustness: pre-registration

Written and committed on 2026-10-14, before any of the runs below. The results go in
`README.md` and are judged against the criteria here, as written. Anything not listed here is
exploratory and reported as such.

## Why

The contextual benchmark in `ctxbench.py` has two weak points.

1. **One kind of heterogeneity.** A cell's contrast there is a main effect plus a region effect
   plus a device effect. That additive structure is what a tree splitting one attribute at a time
   is built for, but nobody has checked whether the comparison changes when cells differ in ways
   that do not decompose by attribute.
2. **Six cells.** The cells are the full cross of two attributes. With more attributes that cross
   gets thin, and in production the cells would more likely come from GrowthBook's own tree. The
   comparison with the tree's leaves used as cells has not been run.

## Design

Script: `ctxrobust.py`. GrowthBook's contextual engine is `computeContextualBanditWeights` from
`packages/stats-ts`, bundled from commit `b70f8c77d` as in `ctxbench.py`, `maxLeaves` 8,
`minUsersPerLeaf` 100, fed the run's pooled (context, variation) counts as the product does.

Common to every scenario: 4 arms with main log-odds contrasts `0, 0.05, 0.10, 0.15`; base log-odds
`logit(0.03)` plus a cell offset drawn once per run with sd 0.3; a common shift redrawn every
period with sd `sigma`; 40 periods of 20,000 users split across cells by their traffic shares; a
1% floor for every policy; 20 paired runs (environment seeds 0 to 19, the same draws for every
policy). The measure is cumulative regret against each cell's own best arm.

**Layouts.**

- **L6**: region (3 levels, shares 0.5, 0.3, 0.2) by device (2 levels, 0.7, 0.3); 6 cells.
- **L24**: region (4 levels, 0.4, 0.3, 0.2, 0.1) by device (0.7, 0.3) by plan (3 levels,
  0.6, 0.3, 0.1); 24 cells, the smallest holding 0.3% of traffic, about 60 users a period.

**How the cells' contrasts differ**, on top of the main contrasts, drawn once per run:

- **additive**: each attribute level gets an effect per arm, sd 0.2.
- **residual**: each (arm, cell) gets its own effect, sd `0.2 * sqrt(m)` for `m` attributes, so
  the total spread matches the additive case and none of it decomposes by attribute.
- **checkerboard** (L6 only): with `s = +1` when region index plus device index is even and `-1`
  otherwise, arm 0 gets `+0.2 s` and arm 3 gets `-0.2 s`. The best arm alternates between arm 0
  and arm 3 in a pattern no single attribute separates.

**Scenarios.** Part A: L6 with additive, residual and checkerboard, each at `sigma` 0 and 0.3 (six
scenarios). Part B: L24 with additive and residual, each at `sigma` 0 and 0.3 (four scenarios).

**Policies.**

- **Contextual OR-TS**: `ContextualLogisticBandit` over the full cross, `interaction_sd="auto"`.
- **GrowthBook contextual**: the engine above, as shipped.
- **Hybrid**: after each period GrowthBook's engine is run as usual, and each cell is mapped to the
  leaf it currently falls in. OR-TS is then refitted from the whole history with the leaves as
  cells (`allocate_cells_from_rows`), and every cell gets its leaf's allocation. The tree may
  change between periods, so nothing is carried; the state is re-derived each time.
- **OR-TS, one per cell**: an independent `LogisticBandit` in each cell.
- **OR-TS, context ignored**: one `LogisticBandit` over pooled counts.

## Hypotheses and failure criteria

A difference "holds" when the mean paired difference in regret is more than two standard errors
in the stated direction.

- **H1.** In all four non-additive Part A scenarios (residual and checkerboard, at both `sigma`),
  contextual OR-TS has lower regret than GrowthBook contextual. *Fails* if any one of the four
  does not hold.
- **H2.** Contextual OR-TS does not depend on additivity: its mean regret in Part A residual is
  within 20% of its mean regret in Part A additive, at each `sigma`. The model treats cells as
  exchangeable and uses no attribute structure, so it should not care. *Fails* if either `sigma`
  is outside 20%.
- **H3.** In all four Part B scenarios, the hybrid has lower regret than GrowthBook contextual.
  *Fails* if any one does not hold.
- **H4.** In all four Part B scenarios, contextual OR-TS over the full 24-cell cross has lower
  regret than GrowthBook contextual, even though some cells are thin. *Fails* if any one does not
  hold.

**Exploratory, no prediction:** the hybrid against contextual OR-TS over the full cross; whether
the checkerboard hurts GrowthBook's tree more than OR-TS; the tree's final leaf counts; the
per-cell and context-ignored baselines.
