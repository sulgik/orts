# Changelog

## 2.3.0 (2026-09)

Caught up with the 2026 manuscript's symmetric-default revision, which makes
the proper contrast prior the paper's basic specification and states it as
what this reference implementation does.

**The default contrast prior changed.** `LogisticBandit()` now starts from
the symmetric proper prior of Supplement A — latent arm effects
`N(0, tau^2)` with Algorithm 1's `tau = sqrt(2)`, so every pairwise log-odds
contrast has prior sd 2, every arm prior winner probability `1/K`, and
neither depends on which arm is the reference. Allocations from a first
batch will differ from 2.2's, most visibly on sparse or separated batches,
which the default now fits instead of skipping. To reproduce 2.2 exactly,
pass `contrast_prior="flat"`.

- `contrast_prior` selects `"symmetric"` (default), `"flat"` (the historical
  zero-precision option) or `"independent"`; `arm_effect_prior_sd` is `tau`
  and `new_contrast_prior_sd` the independent option's scale. `init_scale`
  and `new_arm_scale` still work and map onto these, with a
  `DeprecationWarning`.
- An arm that joins after the first fit now enters through Supplement B's
  symmetric augmentation: it is drawn from the same `N(0, tau^2)` population
  as the incumbents and carries the retained latent centre at `tau^2 / K`,
  leaving the incumbents' pairwise posteriors and the state's reference
  invariance untouched. The paper no longer treats an informative new-arm
  prior as a separate option. `orts.priors` holds the construction.
- Under `contrast_prior="flat"` (or `decay=1`, which discards the carried
  evidence and re-enters the same condition) an arm with no carried evidence
  and only events, or only non-events, now raises `RuntimeError`: that fit
  has no finite mode, and 2.2 skipped the batch silently. The batch-level
  skip — no events or no non-events at all — still returns `False`.
- Paper cross-references throughout the package, examples, notebook and
  README updated to the current manuscript: decay is Section 5.1, the
  diagnostics Section 4.2, changing arm sets Section 6.1, the warm start
  Supplement D, and the stopping quantities, floors and `implied_decay`
  Supplement G.
- `docs/RESEARCH_PLAN.md` re-synced with the research repository: the
  pre-registration record now runs H1-H27, including the symmetric-default
  registrations (H13-H19c) that the paper's supplements cite.
- Separate experiment groups in one bandit (Supplement B). A batch that
  shares no arm with the state used to fail with a shape error; it now starts
  a group of its own -- a second state, with no covariance to the first,
  because no comparison links them. `groups()` lists them and `known_arms()`
  reports every arm held.
- The paper's bridge case is unaffected and needs nothing new: a batch that
  shares an arm with the state stays inside that group, and the newcomer
  joins by augmentation, which is how the joint posterior comes to carry a
  comparison never run directly.
- A batch serving arms of two separate groups raises `ValueError`. Joining
  states that were initialized independently, each with a contrast prior
  centred on its own arm set, is not a construction the paper gives, so the
  batch is refused rather than fitted on an invented one.
- A query may span groups: Supplement B's new-arm rule, read for a group
  rather than a single arm, gives each group traffic in proportion to its
  size and lets it allocate inside itself by its own winner probabilities.
  An arm with no posterior is a group of one, which is the rule exactly as
  the paper states it, so a query over one group and some brand-new arms
  behaves as before. Because "best" is defined only among arms a batch has
  compared, `p_best` and `expected_loss` are `nan` throughout a query that
  spans groups, so a stopping rule cannot read a ranking across them.
  `query()` with no arms allocates over every arm held. The single-state
  attributes (`mu`, `sigma_inv`, `action_list`, `contrasts()`, ...) report
  the primary group, which is the whole state in the usual one-group case.

## 2.2.0 (2026-09)

The zero-count rules of the paper's Algorithm 1 and Supplements A and C.

- First-fit check: under the flat contrast prior a first fit requires every
  arm to have both events and non-events; a separated first batch forms no
  state and `update` returns `False`. Until a fit has completed, `query`
  returns the uniform start-up allocation (`p_best` and `expected_loss` are
  `nan`) and `bandit.fitted` is `False`.
- `LogisticBandit(init_scale=tau)`: the symmetric proper initialization,
  `S_0 = tau^-2 (I - 11'/K)` on the contrasts, which fits a first batch
  with zero or complete cells.
- `LogisticBandit(new_arm_scale=s)`: a proper `N(0, s^2)` contrast prior for
  an arm that joins after the first fit.
- The batch-level skip (no events or no non-events) is unchanged.

## 2.1.0 (2026-09)

- `LogisticBandit.query(arms, ...)` is the action: name the arms that will
  be live in the next batch, in any order and independently of the state's
  arm set, and receive an `Allocation` with `shares`, `p_best`,
  `expected_loss` and `leader`, all from one set of draws. `win_prop()`
  returns `query(...).shares`; `expected_loss()` returns
  `query(...).expected_loss`.

## 2.0.0 (2026-09)

The repository is now the reference implementation of *Odds-Ratio Thompson
Sampling: A Specification and Design Guide for Contrast-Based Multi-Armed
Bandits* (Kim, 2026), which supersedes the 2020 preprint's code.

**Package.** Everything lives in the `orts` package (`from orts import
LogisticBandit, TSPar, DiscountedTSPar, diagnostics`). The root modules
`logisticbandit`, `ts` and `utils` remain as deprecated shims.

**LogisticBandit.**
- The state is kept in one canonical arm order (first seen, reference
  last; `reference=` picks the reference), so batches may name arms in any
  order or subset without re-expressing the state; a batch that does not
  expose the reference is fitted against an observed arm and mapped back.
  `contrasts()`, `level`, `set_reference()` and `drop()` read or re-base it.
- `update` returns `True`/`False` and skips a batch with no events or no
  non-events, whose posterior is improper under the flat intercept prior.
- `win_prop` gains `floor` (allocation floors after the power map) and an
  optional `rng` for reproducible draws; `contrast_draws` exposes step A1.
- `expected_loss` (expected loss of committing to an arm, in log-odds) and
  `contrast_sd` support stopping and dropping rules.
- `implied_decay` maps a measured excess contrast sd to the decay it implies.
- `LogisticBandit.from_beta_posteriors` warm-starts from an incumbent
  Beta-Bernoulli service.
- `remove_not_observed` now folds absent arms away through the reference
  transformation instead of dropping them from the fit silently.

**Baselines.** `DiscountedTSPar`: Beta-Bernoulli with geometric count
discounting, the forgetting baseline whose memory matches OR-TS's decay.

**Diagnostics.** `orts.diagnostics`: per-batch level and contrasts with
delta-method sampling variances, excess variance, the level-versus-contrast
ratio `R`, lag-one autocorrelation, sampling bands, implied decay.

**Removed.** `LinearBandit` (per-arm Gaussian Thompson sampling, unrelated to
the contrast state), the benchmark, coverage and type-check reports, and the
long-form tutorial. The 2020 synthetic runner and its archived outputs moved
to `archive/2020/`.

## 0.1.0 (2020)

Code accompanying Kim and Kim (2020), arXiv:2003.01905.
