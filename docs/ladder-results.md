# Ladder results (E1)

GPU-vs-CPU throughput on this box as models get bigger and stiffer. Produced by
`tools/bench_ladder.py`; raw rows in `~/pf-runs/ladder-*.jsonl`. Sampler at the shipped
defaults (2,000 draws, 2,000 ChEES warmup, `JAX_COMPILATION_CACHE_DIR` set); CPU arm is
`pm.sample(draws=1000, tune=1000, target_accept=0.8)` with one BLAS thread per chain, so
"chains = cores" is literal.

## Measured cells

Bulk ESS summed over variables per wall second, best of the seeds, warm unless noted.

| model | d | CPU best | GPU best | GPU / CPU | what it shows |
|---|---|---|---|---|---|
| GEV Port Pirie | 3 | 6,733 (32 ch) | 7,856 (32 ch) | **1.17x** | tiny model, GPU barely ahead |
| 8 Schools | 10 | 37,494 (32 ch) | 45,614 (32 ch) / 134,898 (128 ch) | **1.21x / 3.6x** | funnel, cheap gradients |
| logistic 1000x25 | 25 | 10,131 (32 ch) | 24,238 (32 ch) / 30,719 (128 ch) | **2.4x / 3.0x** | correlated design matrix, 0 divergences both arms |
| radon noncentered | 175 | **511,606** (32 ch) | 295,400 (32 ch) | **0.58x** | CPU wins outright |
| radon centered | 175 | 48,909 (32 ch), R-hat 1.08 | 273, R-hat 4.89 | **0.006x** | GPU sampler collapses |

Draw-covariance condition numbers (sampled variables only, so no deterministic collinearity):
5.25 (8 Schools), ~10.2 (GEV), **426** (logistic), **6,660** (radon centered),
**25,116** (radon noncentered).

## Three conclusions

1. **The GPU advantage is not monotone in parameter count.** It went 1.17x -> 1.21x -> 3.0x
   going from d=3 to d=25, then reversed at d=175. What actually predicts it here is the ratio of
   per-gradient cost to per-gradient launch overhead: radon noncentered is a wide but *cheap and
   well-behaved* posterior whose 919-row likelihood vectorises well on 32 CPU cores, while the
   logistic model needs 56-86 leapfrog steps per draw, which is where 32 vmapped chains pay off.
   Chain-count scaling on the GPU is still near-linear (128 chains gave 2.7x of 32 chains on
   logistic and 3.0x on 8 Schools), so the GPU's headroom is in chains, not in per-chain speed.

2. **Our sampler fails on the centered funnel, and this is a quality failure, not a speed one.**
   R-hat 4.89 with the trajectory frozen at 4.8 leapfrog steps and 3.6 divergences per 1,000
   draws. ChEES shrank the trajectory instead of taming the geometry. CPU NUTS is poor on the same
   model too (R-hat 1.04-1.18, 29-99 divergences per 1,000) but it is not broken. This reproduces
   the published pattern exactly: Seyboldt et al. excluded every centered model for
   non-convergence, and Dance et al. report ChEES returning `nan` ESS on a funnel target. We now
   have the same failure on our own box, which makes it testable rather than folklore.

3. **Gate G1 is satisfied by measurement.** The threshold was condition number >= 50 or >= 1
   divergence per 1,000 draws on a ladder model. The logistic model alone is at 426, and radon is
   at 6,660 and 25,116, two to three orders of magnitude above the two shipped examples. So the
   metric work (E4) is justified by our own numbers rather than by the Fisher paper's 4x, which we
   showed is a small-d, funnel-free result. The centered radon funnel is the acceptance target for
   E4: a metric that gets R-hat below 1.05 there is worth shipping; a metric that only speeds up
   the already-fine noncentered model is not.

## Measurement caveats that matter

- `cold` means "first compile of that graph in this process", not "cold XLA cache". A multi-model
  run reuses compilation across models that share a flattened dimension, so the first radon job
  reported 155 s wall for a cell that later measured 47 s. Cells that quote cold numbers must run
  one model per process.
- 128 chains at d=175 does not fit the 6 GB RTX A2000 (allocation failure at 699 MiB in a
  `dot_general`). The chain sweep stops where the device stops; do not extrapolate past it.
- Every radon cell here predates the RNG stage-key fix (`a9d1a3e`), so re-runs will not reproduce
  these exact realisations, only the same distribution.
- Padding ratio (`max/mean` leapfrog steps across chains per draw) was **1.0** in every GPU cell,
  at 32 and at 128 chains. None of these results is warp divergence; the ChEES lockstep design is
  doing its job.
