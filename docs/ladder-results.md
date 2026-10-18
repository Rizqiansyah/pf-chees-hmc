# Ladder results (E1)

GPU-vs-CPU throughput on this box as models get bigger and stiffer. Produced by
`tools/bench_ladder.py`; raw rows in `~/pf-runs/ladder-*.jsonl`. Sampler at the shipped
defaults (2,000 draws, 2,000 ChEES warmup, `JAX_COMPILATION_CACHE_DIR` set); CPU arm is
`pm.sample(draws=1000, tune=1000, target_accept=0.8)` with one BLAS thread per chain, so
"chains = cores" is literal.

## Measured cells

115 cells, no errors, median over 4 warm seeds per configuration. CPU arm is `pm.sample` with
`chains = cores`; "best CPU" is its 32-chain configuration. `cov_cond` is the draw-covariance
condition number over sampled variables only.

| model | d | cov_cond | best CPU ESS/s | GPU 32 ch | GPU 128 ch | GPU 4 ch |
|---|---|---|---|---|---|---|
| GEV Port Pirie | 3 | 23.2 | 6,786 | 6,702 (**0.99x**) | 18,526 (**2.73x**) | 462 (0.07x) |
| 8 Schools | 10 | 19.8 | 39,213 | 41,568 (**1.06x**) | 133,824 (**3.41x**) | 6,198 (0.16x) |
| logistic 1000x25 | 25 | 357.8 | 10,852 | 24,944 (**2.30x**) | 41,134 (**3.79x**) | 1,699 (0.16x) |
| radon centered | 175 | 2,207 | 53,642 | 262 (**0.005x**) | does not fit | 58 (0.001x) |
| radon noncentered | 175 | 8,837 | 463,298 | 271,585 (**0.59x**) | does not fit | 17,570 (0.04x) |

Worst-variable throughput and divergence rates (GPU at 32 chains): 8 Schools 2,987 ESS/s and
0 divergences per 1,000 draws versus CPU 2,062 and 1.344; logistic 521 and 0 versus 281 and 0;
GEV 504 and 0.328 versus 1,482 and 11.5; radon noncentered 126 and 0 versus 204 and 1.312.

## Four conclusions

1. **The GPU advantage tracks steps-per-draw, not parameter count.** It went 0.99x -> 1.06x ->
   2.30x from GEV to 8 Schools to logistic, then reversed at d=175: radon noncentered is wide but
   *cheap and well-behaved*, its 919-row likelihood vectorises well across 32 CPU cores, and the
   GPU loses by 1.7x. The logistic model needs 56-86 leapfrog steps per draw, and long shared
   trajectories are exactly what 32 vmapped chains are good at. Chain scaling on the GPU stays
   strong (128 chains is 3.2x of 32 chains on 8 Schools, 2.8x on GEV, 1.65x on logistic), so the
   GPU's headroom is chains, not per-chain speed. Note the GPU cannot hold 128 chains at d=175.

2. **Our sampler fails on the centered funnel, and this is a quality failure, not a speed one.**
   At 32 chains: max R-hat 4.33, 262 ESS/s, trajectory frozen at ~4.8 leapfrog steps. ChEES shrank
   the trajectory instead of taming the geometry. CPU NUTS is poor on the same model too (R-hat
   1.11-1.18, 50-54 divergences per 1,000 draws) but it is not broken. This reproduces the
   published pattern exactly: Seyboldt et al. excluded every centered model for non-convergence,
   and Dance et al. report ChEES returning `nan` ESS on a funnel target. We now have the same
   failure on our own box, which makes it testable rather than folklore.

3. **Four chains is not a usable GPU configuration, which retroactively justifies the default
   change.** At 4 chains the GPU is 6x to 25x slower than the best CPU on every model and its
   max R-hat degrades to 3.59 (radon centered), 1.21 (radon noncentered) and 1.13 (GEV). ChEES
   estimates its trajectory length from the ensemble, so with 4 chains it has almost nothing to
   estimate from. The shipped default used to be 4; that setting was not merely slow, it produced
   unconverged output.

4. **Gate G1 is satisfied by measurement.** The threshold was condition number >= 50 or >= 1
   divergence per 1,000 draws on a ladder model. The logistic model alone is at 358, and radon is
   at 2,207 and 8,837, one to three orders of magnitude above the two shipped examples at 20-23.
   So the metric work (E4) is justified by our own numbers rather than by the Fisher paper's 4x,
   which we showed is a small-d, funnel-free result. The centered radon funnel is the acceptance
   target for E4: a metric that gets R-hat below 1.05 there is worth shipping; a metric that only
   speeds up the already-fine noncentered model is not.

## Measurement caveats that matter

- `cold` means "first compile of that graph in this process", not "cold XLA cache". A multi-model
  run reuses compilation across models that share a flattened dimension, so the first radon job
  reported 155 s wall for a cell that later measured 47 s. The table above is warm-only for that
  reason; cells that quote cold numbers must run one model per process.
- 128 chains at d=175 does not fit the 6 GB RTX A2000 (allocation failure at 699 MiB in a
  `dot_general`). The chain sweep stops where the device stops; do not extrapolate past it.
- The grid above ran after the RNG stage-key fix (`a9d1a3e`), so it is the current realisation.
  The headline numbers in `docs/gpu-upgrade-plan.md` section 2b (45,614 and 134,898 ESS/s on
  8 Schools) predate that fix; they are the same distribution but not the same draws, and this
  table supersedes them (41,568 and 133,824).
- Padding ratio (`max/mean` leapfrog steps across chains per draw) was **1.0** in every GPU cell,
  at 4, 32 and 128 chains. None of these results is warp divergence; the ChEES lockstep design is
  doing its job.

## Metric A/B (E4)

Same harness, 32 GPU chains, 2,000 draws, 2,000 warmup, warm. `mc` is the shipped metric
(inverse of Pathfinder's empirical covariance + ridge). `lbfgs_lowrank` converts Pathfinder's own
L-BFGS factors, which are already computed and thrown away, so it costs nothing.
`fisher_lowrank` is Algorithm 1 of arXiv:2603.18845 via
`blackjax.adaptation.metric_estimators.fisher_score_low_rank` over 4,096 draws.

| model | mc | lbfgs_lowrank | fisher_lowrank | steps mc -> fisher |
|---|---|---|---|---|
| 8 Schools | 41,568 | — | **50,626** (1.22x) | 21 -> 8 |
| logistic 1000x25 | 24,944 | — | **48,986** (1.96x) | 60 -> 9 |
| radon noncentered | **271,585** | 155,190 | 51,971 | 37 -> 514 |
| radon centered | 266, R-hat 3.79 | 447, R-hat 3.55 | **9,356, R-hat 2.10** (rank 40) | 4 -> 314 |

Eigen-spectra of the metrics themselves explain the failure (geometric mean / condition number of
the inverse mass matrix, d=175):

| metric | geo_mean | cond | p95 | max |
|---|---|---|---|---|
| mc | 148 | 74 | 213 | 2,025 |
| lbfgs_lowrank | 0.0069 | 72 | 0.0085 | 0.036 |
| fisher_lowrank | 0.069 | 2,366 | **1.000** | **1.000** |

Fisher's tail is pinned at exactly 1.0 by the construction's clamp, and its units are 2,000x below
the posterior's own scale. One step size has to serve both.

**Rescaling to the empirical covariance's geometric mean was tried and rejected.** It made every
model worse: 8 Schools 50,626 -> 13,628, logistic 48,986 -> 15,044, radon noncentered 51,971 ->
53,188 with steps still at 416, and radon centered R-hat 2.10 -> 5.69. The "wrong" units were
carrying part of the benefit, which is the honest read of why: with the metric, the step size and
the trajectory length all adapted jointly, isolating one of them is not a free operation. The code
for it is not in the repo.

Raising `cutoff` from 2 to 10 (keep fewer eigen-directions) helped radon noncentered modestly --
514 -> 359 steps, 51,971 -> 60,442 ESS/s -- and did not close the gap to `mc`.

**Verdict.** E4 ships as an option, not a default. `imm_source="fisher_lowrank"` is worth 1.2-2.0x
on models whose current metric is a poor fit, and loses 5x on models where it is already good, and
the discriminator is realised trajectory length (a jump from tens to hundreds of steps is the
symptom), which no per-draw diagnostic shows until it is too late. The centered funnel is not a
metric problem either: rank 10/40, warmup 2,000/10,000, cutoff 2/10 and scaling all leave R-hat
between 2.10 and 5.69, while non-centering the same model reaches 1.005. That is a parameterisation
fix in the user's model, not a sampler fix.
