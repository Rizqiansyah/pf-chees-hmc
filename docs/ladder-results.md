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

## Equal-draw-budget comparison (time to the same number of samples)

ESS per second rewards running more chains for longer. The operationally fair question is: same
total post-warmup draws, varied chains, **who finishes first** — and are those draws worth
anything. Configuration: chains x draws-per-chain pairs holding the total at 4,000 (a first look at
a model) and 40,000 (a publication run). GPU at its shipped 2,000-step ChEES warmup; CPU NUTS with
PyMC's default `tune=1000`, `target_accept=0.8`, one BLAS thread per chain, `cores = chains`.
**One seed per cell** — this is a scoping measurement, not a shipped claim. `ESS1` is the bulk ESS of
the single worst variable. Reproduce with `tools/bench_ladder.py` + `tools/budget_summary.py`.

### 8 Schools (d=10)

| total draws | chains | GPU cold s | GPU warm s | CPU s | GPU/CPU | R-hat G | R-hat C | ESS1 G | ESS1 C |
|---|---|---|---|---|---|---|---|---|---|
| 4,000 | 4x1,000 | 17.6 | 8.8 | **2.4** | 3.61x | 1.0061 | 1.0022 | 1,718 | 2,652 |
| 4,000 | 8x500 | 18.1 | 8.7 | **2.6** | 3.34x | 1.0055 | 1.0030 | 1,938 | 2,383 |
| 4,000 | 16x250 | 18.8 | 9.2 | **4.6** | 2.03x | 1.0117 | 1.0073 | 2,160 | 2,248 |
| 4,000 | 32x125 | 37.0 | 10.3 | **8.5** | 1.21x | 1.0169 | 1.0143 | 2,308 | 2,549 |
| 40,000 | 4x10,000 | 18.8 | 17.6 | **8.6** | 2.05x | 1.0004 | 1.0002 | 16,043 | 21,232 |
| 40,000 | 8x5,000 | 14.9 | 13.8 | **6.2** | 2.25x | 1.0008 | 1.0004 | 19,888 | 21,174 |
| 40,000 | 16x2,500 | 13.4 | 12.2 | **8.7** | 1.41x | 1.0008 | 1.0008 | 22,094 | 22,793 |
| 40,000 | 32x1,250 | 13.1 | 12.4 | 12.6 | **0.98x** | 1.0013 | 1.0016 | 22,717 | 23,726 |

### Radon non-centered (d=175)

| total draws | chains | GPU cold s | GPU warm s | CPU s | GPU/CPU | R-hat G | R-hat C | ESS1 G | ESS1 C |
|---|---|---|---|---|---|---|---|---|---|
| 4,000 | 4x1,000 | 39.6 | 21.5 | **10.5** | 2.05x | 1.0250 | 1.0081 | 172 | 587 |
| 4,000 | 8x500 | 34.6 | 19.0 | **8.6** | 2.20x | 1.0331 | 1.0119 | 228 | 656 |
| 4,000 | 16x250 | 44.2 | 24.7 | **11.2** | 2.20x | 1.0430 | 1.0144 | 296 | 594 |
| 4,000 | 32x125 | 78.5 | 30.3 | **16.2** | 1.87x | 1.0834 | 1.0606 | 333 | 390 |
| 40,000 | 4x10,000 | 70.4 | 75.2 | **35.8** | 2.10x | 1.0015 | 1.0006 | 2,029 | 6,487 |
| 40,000 | 8x5,000 | 43.7 | 37.6 | **24.3** | 1.55x | 1.0165 | 1.0008 | 564 | 6,094 |
| 40,000 | 16x2,500 | 45.6 | 43.7 | **24.1** | 1.81x | 1.0067 | 1.0020 | 3,889 | 6,119 |
| 40,000 | 32x1,250 | 43.6 | 43.1 | **26.7** | 1.62x | 1.0058 | 1.0030 | 4,009 | 6,209 |

GPU at 128 chains does not fit radon on 6 GB (allocation failure at 699 MiB); 8 Schools at 128 chains
takes 17.3 s for 4,096 draws versus 8.5 s for the CPU at 32x125.

### What this says

1. **At equal sample counts the CPU wins this box today.** 15 of 16 cells. The one tie is 8 Schools
   at 40,000 draws with 32 chains (12.4 s vs 12.6 s). The earlier GPU wins in ESS/s were real but are
   a throughput statement about long runs with many chains; they do not survive fixing the sample
   count, because the GPU pays two costs the CPU does not: a 2,000-step lockstep warmup and a fresh
   JAX trace for every new chain count (~15-19 s).
2. **The gap is fixed cost, not sampling speed.** GPU warm time at 8 Schools 40,000 draws (12.2-13.8 s)
   is within 1.5-2.3x of CPU even though the sampling itself is faster; subtract ~6 s of setup and
   ~11 s of warmup and the remaining sampling time is below the CPU's. At 4,000 draws the fixed cost
   is 75-90% of the run, which is why the ratio is worst there.
3. **Short chains cost quality on both arms, and the GPU more.** At 4,000 draws GPU max R-hat rises
   monotonically with chains (1.006 -> 1.017 on 8 Schools, 1.025 -> 1.083 on radon); CPU rises less.
   Worst-variable ESS at 4,000 draws is below the 400 target on radon in every GPU cell (172-333).
   A 4,000-draw budget is simply not enough for d=175, sliced any way.
4. **Warmup 1,000 instead of 2,000 is a real lever only on long chains.** Radon at 40,000 draws:
   33.2-41.1 s with R-hat 1.004-1.009 at 16 and 32 chains. At 4,000 draws it is a disaster:
   R-hat 2.16 at 4 chains. (Different seed from the 2,000-step rows, so read the direction, not the
   decimal.) This is the strongest argument for a stopping criterion rather than a shorter constant.
5. **CPU divergences are the counterweight.** CPU NUTS shows 1.3-2.3 divergences per 1,000 on
   8 Schools and ~1 on radon, against 0-0.05 for the GPU. On wall time today the CPU wins at these
   budgets; the GPU's case is divergence-free output plus throughput once the run is long enough for
   the fixed cost to amortise.

Next work item this points at, in order: kill the warmup fixed cost (criterion-based stop, then a
tuned-hyperparameter cache for the second run of the same model), and stop re-tracing per chain
count. Both are generic across models; neither touches the kernel.

## Measurement-integrity note

Two probe bugs were caught during this work and both are worth knowing before trusting any timing
here. (1) A scan carry of the form `x - 0.0 * g` is dead-code-eliminated by XLA, so a probe that
"consumes" a gradient that way times an empty loop -- this voided the first two versions of the
per-gradient probe. Any such probe needs a carry that changes the returned value, plus a
gradient-ignoring floor control row. (2) `CUDA_VISIBLE_DEVICES=''` does not make a `pixi run`
execute on the CPU, because `pixi.toml` exports `CUDA_VISIBLE_DEVICES = "0"` in `[activation.env]`
and activation wins. The correct form is `pixi run env CUDA_VISIBLE_DEVICES= python ...`. Numbers
labelled CPU in the parallel lane notes at `/home/akbar/pf-research/` may therefore be GPU numbers.

## Per-gradient cost of the real graphs, chains varied

Valid version: the scan carry consumes the gradient (accumulated as a sum of squares) and every row
carries a logp-only floor control. The first two versions of this probe are void — `x - 0.0 * g` in a
carry is dead-code-eliminated by XLA, so they timed an empty loop. Microseconds per batched gradient,
float64, inside a 20-iteration `lax.scan` (dispatch excluded), GPU.

| model | 1 chain | 4 | 8 | 32 | 128 | floor at 32 |
|---|---|---|---|---|---|---|
| 8 Schools | 24.9 | 29.9 | 19.7 | **23.0** | 25.3 | 19.1 |
| logistic 1000x25 | 29.4 | 86.0 | 86.9 | **116.3** | 394.6 | 111.2 |
| radon non-centered | 44.6 | 91.0 | 119.9 | **138.7** | 268.8 | 41.6 |

1. **Chains are effectively free on small models.** 8 Schools costs the same 23-25 us whether 1, 32 or
   128 chains run. This is why ESS/s scales almost linearly with chains, and it also closes the
   chain-batching idea from the sampler side: there is no duplicated per-chain work left to remove.
2. **Float32 buys nothing on the graphs we ship.** f64 vs f32 at 32 chains: 8 Schools 23.0 vs 22.5,
   radon 138.7 vs 141.4, logistic 116.3 vs 126.7. The 3.3x-20x float32 win in the synthetic probe was
   a dense 919x175 matmul pinned at the fp64 ALU limit; real PyMC graphs are not that. **The parked
   float32 item should stay parked** — measured, not assumed.
3. **The remaining cost is ours, not the model's.** 8 Schools production runs at ~2.9 ms per
   transition at 32 chains (12.4 s for 3,250 lockstep steps at the 40,000-draw budget) while 21
   leapfrog steps of the real logp cost ~480 us. That is ~83% overhead. Radon: 11.4 ms per transition
   against 37 x 139 us = 5.1 ms of model, ~55% overhead. Attribution is the next measurement.

## Where the wall time actually lives (stage timers, both models)

`~/pf-o1-stages.py` monkeypatches each `PfCheesHMC` stage with a wall timer that blocks on the GPU
result before stopping. Two runs per model in one process; **read the warm run** — the first carries
compilation. One correction on first reading: `finite_init` runs *inside* `_run_pathfinder`, so the
stages overlap and the residual column goes negative. The numbers below have that double count
removed (pathfinder shown net of finite_init).

8 Schools, 32 chains, 1,250 draws, 2,000 warmup, warm total 11.86 s:

| stage | s | share |
|---|---|---|
| PyMC adapt_diag start | 0.37 | 3% |
| finite-start repair | 1.75 | 15% |
| Pathfinder (net) | 1.72 | 15% |
| ε0 tuner | 1.03 | 9% |
| ChEES warmup, 2,000 steps | 4.09 | 35% |
| sampling, 1,250 draws | 3.11 | 26% |

Radon non-centered, same shape, warm total 43.03 s: start 0.75 (2%), finite repair 4.24 (10%),
Pathfinder net 3.23 (7%), tuner 1.56 (4%), **warmup 18.39 (43%)**, **sampling 14.99 (35%)**.

Cold runs are dominated by compilation instead: Pathfinder stage 25.18 s of 39.28 s on 8 Schools,
43.11 s of 91.32 s on radon.

1. **Warmup is the biggest single stage, but not because its steps are expensive.** 4.09 s / 2,000
   steps = 2.05 ms per warmup step on 8 Schools, versus 3.11 s / 1,250 = 2.49 ms per sampling step.
   Warmup ≈ 1.6x the count-adjusted sampling cost, i.e. roughly the same step, just 2,000 of them.
   An earlier isolated probe reported 8.6 ms per warmup step; that probe used an identity metric and
   no Pathfinder initialisation, so its trajectories were longer than production's. Production is the
   authority here.
2. **~80% of every step is not the model.** Kernel-only scan at L=21 on the real 8 Schools logp is
   443 us; carrying the full HMCInfo makes it 491 us (+48 us). Production pays ~2,000-2,500 us per
   step. So the overhead is machinery around the kernel, and the info carry is not it.
3. **The finite-start repair is now the third-largest fixed cost** (1.75 s at 8 Schools, 4.24 s at
   radon, ~10-15% of a warm run). It checks log-density and gradient per chain with 32 separate
   eager calls, which is Python dispatch, not compute — the same dispatch cost that misled the
   roofline probe. One vmapped check should collapse it.

## What limits a step on the GPU (roofline probe)

`tools/probe_gpu_roofline.py`, run as `X64=1 pixi run python tools/probe_gpu_roofline.py` and again
with `X64=0`. It measures this card's ceilings in the dtype under test (a 4096^3 matmul for
arithmetic, a 64M-element reduction for bytes), then times a 32-chain batched `value_and_grad` of a
linear-Gaussian log-density at shapes that matter. RTX A2000, measured 2026-10-18.

Ceilings measured: float64 **195.6 GFLOP/s**, float32 **14,771 GFLOP/s** (75x). The reduction-based
bandwidth numbers are not trustworthy (float64 measured 218 GB/s against float32's 186, which is
physically backwards), so conclusions below rest on dtype ratios, not on percentage-of-bandwidth.

Per 32-chain gradient, microseconds:

| shape (n observations, d params) | float64 | float32 | f64/f32 |
|---|---|---|---|
| 1, 175 (control: parameters, no data) | 155.3 | 140.5 | 1.1x |
| 8, 10 (8 Schools) | 153.8 | 130.5 | 1.2x |
| 919, 175 (radon) | 377.7 | 115.9 | **3.3x** |
| 9,190, 175 | 2,196.7 | 316.1 | **7.0x** |
| 91,900, 175 | 18,862.4 | 936.6 | **20.1x** |

1. **Small models sit on a dtype-independent latency floor of ~130-155 us per batched gradient.**
   8 Schools and a 175-parameter control with one observation cost the same. Nothing data- or
   arithmetic-shaped is happening there; it is launch and dependency-chain cost, and it is the reason
   2,000 warmup steps dominate a short run.
2. **Above the floor the cost is dtype-dependent and grows with n.** A bandwidth limit would give a
   ratio near 2 (float64 moves 2x the bytes); measured 3.3x at n=919 and 20x at n=91,900, where the
   fp64:f32 arithmetic-rate ratio is 75x. So at data-heavy shapes the float64 ALU path, not DRAM
   bandwidth, is what costs.
3. **Retracted from the first pass of this analysis:** the claim that radon is "50% of the bandwidth
   ceiling" and that chain cost is linear in chain count at the radon shape. Both came from a byte
   model that assumed every chain re-reads the design matrix, and from a bandwidth ceiling
   measurement that is itself unreliable. The dtype ratios above are the evidence that survives.
4. **Practical consequence, and it is the largest lever measured here:** running the model in
   float32 makes each gradient 3.3x cheaper at radon's shape and 20x at 91,900 observations, while
   costing nothing at 8 Schools' shape. Before any of it ships: the sampler's numerics must be
   re-validated in float32 (draw-covariance condition numbers reach 8,837 on radon, so metric
   conditioning is the risk), and the existing "one fp64 rerun" gate becomes the comparison, not an
   afterthought.

## After the batched start check (same config, warm runs)

| model | finite-start repair before | after | warm total before | after |
|---|---|---|---|---|
| 8 Schools | 1.75 s (15%) | **0.24 s (2%)** | 11.86 s | 12.03 s |
| radon non-centered | 4.24 s (10%) | **0.39 s (1%)** | 43.03 s | **39.66 s** |

Radon gains 3.4 s end to end, which is the stage saving. On 8 Schools the total did not move,
because the sampling stage went 3.11 s -> 4.52 s in the same pair of runs; two seeds cannot separate
that from noise, and the fix also stops jittering already-usable starts, so realised trajectories
change. The stage-level saving is measured on both models; the 8 Schools end-to-end claim is left
open until 5 seeds.

What is left after the fix, warm, on radon: **ChEES warmup 19.11 s (48%)**, sampling 14.97 s (38%),
Pathfinder 3.39 s (8.5%), PyMC start 0.73 s, tuner 1.57 s. Warmup is now the whole game: 2,000
steps at ~2.05 ms each, of which the model accounts for ~0.3 ms at L=21. Two exits, and they are
independent: do fewer warmup steps (a stop rule) or make each step cheaper (the kernel fork, which
the MEADS lane prices at one leapfrog per step instead of L).
