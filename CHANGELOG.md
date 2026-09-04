# Changelog

## 2.0.0 (2026-09)

The repository is now the reference implementation of *Odds-Ratio Thompson
Sampling: A Specification and Design Guide for Contrast-Based Multi-Armed
Bandits* (Kim, 2026), which supersedes the 2020 preprint's code.

**Package.** Everything lives in the `orts` package (`from orts import
LogisticBandit, TSPar, DiscountedTSPar, diagnostics`). The root modules
`logisticbandit`, `ts` and `utils` remain as deprecated shims.

**LogisticBandit.**
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
