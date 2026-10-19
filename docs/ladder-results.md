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

## Chain-start dispersion feeds the ensemble tuner, but only at short warmup

`tools/probe_init_policy.py`, radon non-centered, 32 chains, 500 draws, 2 seeds per cell, one shared
Pathfinder fit and metric per (warmup, policy) so only the start *pick* varies. `spread` is the mean
per-dimension standard deviation of the 32 chosen starts. Worst R-hat across seeds; min bulk ESS
across variables.

| warmup | policy | worst R-hat | min bulk ESS | mean steps | spread |
|---|---|---|---|---|---|
| 250 | same_point | 1.0379 | 13,037 | 7.0 | 0.0000 |
| 250 | iid (today) | 1.0323 | 13,303 | 7.5 | 0.0827 |
| 250 | stratified | 1.0210 | 14,111 | 9.0 | 0.0828 |
| 250 | overdispersed 2x | **1.0166** | **15,154** | 11.9 | 0.1654 |
| 500 | same_point | 1.0128 | 14,855 | 15.4 | 0.0000 |
| 500 | iid | 1.0099 | 15,495 | 17.2 | 0.0827 |
| 500 | stratified | 1.0084 | 16,236 | 21.6 | 0.0828 |
| 500 | overdispersed | **1.0061** | 16,628 | 27.4 | 0.1654 |
| 1000 | same_point | 1.0053 | 17,061 | 37.2 | 0.0000 |
| 1000 | iid | 1.0055 | 17,024 | 37.4 | 0.0827 |
| 1000 | stratified | 1.0067 | 16,961 | 37.9 | 0.0828 |
| 1000 | overdispersed | 1.0064 | 16,932 | 37.9 | 0.1654 |
| 2000 | same_point | 1.0054 | 16,909 | 37.6 | 0.0000 |
| 2000 | iid | 1.0071 | 16,930 | 37.5 | 0.0827 |
| 2000 | stratified | 1.0060 | 16,682 | 37.5 | 0.0828 |

(Overdispersed at warmup 2,000 was cut by the job's own time cap; 30 of 32 cells.)

1. **Dispersion is not neutral — but it is temporary.** At warmup 250 and 500 the ranking is strictly
   monotone in dispersion on both R-hat and ESS. By warmup 1,000 every policy is within 0.0014 of
   every other and mean steps have converged to ~37.5, i.e. the ensemble forgets where it started.
2. **Stratified beats iid at exactly the point where it is free.** Same spread as iid (0.083), no
   extra steps at 1,000+, +6% min bulk ESS at 250, and worst R-hat 1.0210 vs 1.0323. It is a
   different *pick* from the same pool, so the cost is a sort and a strided index.
3. **Over-dispersion buys quality with compute.** Best R-hat at 250 and 500 (1.0166, 1.0061) but 1.6x
   the mean steps at 500 (27.4 vs 17.2). Not a default; a knob for people chasing a marginal R-hat.
4. **PS-IR with replacement is ruled out on evidence, not taste** — Pathfinder's own paper reports it
   collapsing to 4 distinct draws out of 100 on `mcycle_gp-accel_gp`. If a weighted resampling variant
   is ever tried, it must be systematic/without-replacement.
5. **Coupling that changes the order of work:** the init policy only matters when warmup is short, and
   a stop rule's whole purpose is to make warmup short. So the stop rule must be validated *with*
   stratified starts in place, not after. Deciding them separately would produce a stop rule measured
   in a regime that no longer exists.

## Warmup stop rule: costs more than it saves (E5)

Fresh process per cell, seed 43, 32 chains, 1,250 draws, warmup cap 2,000, blocks of 250,
gate = split-R-hat < 1.01 over the block plus step-size quietness. `wall_s` is the whole run.

| model | base wall | rule wall | delta | warmup used | max R-hat (base / rule) | min bulk ESS (base / rule) |
|---|---|---|---|---|---|---|
| 8 Schools | 39.21 s | 45.93 s | **+6.7 s** | 500 | 1.0018 / 1.0018 | 21,827 / 19,360 |
| GEV | 53.72 s | 60.13 s | +6.4 s | 2,000 | 1.0074 / 1.0378 | 5,189 / 926 |
| radon non-centered | 87.92 s | 95.51 s | +7.6 s | 2,000 | 1.0061 / 1.0053 | 4,257 / 4,098 |
| radon centered | 70.23 s | 79.23 s | +9.0 s | 2,000 | 4.2197 / 4.6323 | 34 / 34 |

Read the table as three separate facts:

1. **Quality is not the problem.** Where the rule fired (8 Schools, 500 of 2,000 steps) posterior means
   moved 0.0100 sd against the same-seed 2,000-step reference; two legitimate 2,000-step runs at
   different seeds move 0.0108 sd. Stopping early was statistically free.
2. **The criterion is conservative, not weak.** It fired on one model of four. On GEV and on both
   radons the warmup window R-hat never dropped below 1.01, so they ran the full 2,000 steps -- which
   is also the correct behaviour on radon centered, the funnel this rule was required to refuse.
3. **The blocked path costs 6.7-9.0 s that the scan does not contain.** 1,500 fewer warmup steps on
   8 Schools is worth about 3.0 s at the measured 2.05 ms per step, so the rule loses on every model,
   including the three it never stopped on.

Cost history, because the first number was misleading: calling `blackjax.diagnostics.rhat` per
parameter compiled per shape and cost ~8 s per block (warmup read 19.17 s for 500 steps against
4.85 s for 2,000 steps). Concatenating all chains, draws and dimensions into one call removed that.
The residual ~7-9 s is block-boundary machinery -- a second scan artifact per block shape plus a host
synchronisation at every boundary -- and was not worth chasing further, because the fix is to move the
loop inside the trace (`blackjax.eca.while_with_info` with the diagnostic as the condition), which is a
rewrite rather than a tuning of this design.

Verdict: **not a default, and not worth the rewrite at this budget.** The `stop_rule` kwarg stays for
anyone who wants the block-level instrumentation; with `stop_rule=None` the warmup is the single scan
it always was, with the same key stream.

What the finding does keep: the *fixed* cost of warmup is not the number of steps. Warmup is 43-48% of
a radon run, and cutting its steps in half is worth ~10 s while the machinery around a stepped-down
warmup costs about the same. The lever that survives is warmup that does less work per step, not fewer
steps -- which is what the MEADS/GHMC one-leapfrog-per-step probe is for.

## MEADS warmup + GHMC sampling: cheaper per gradient, fails on quality (E5)

32 chains, 1,250 draws, same Pathfinder starts and the same log-density in every arm. `grad_per_ess`
is total gradient evaluations divided by the bulk ESS of the *worst* variable, the E5 acceptance
measure. `wall_s` is NOT comparable across arms: the ChEES row times the whole `sample_pf_chees_hmc`
call including Pathfinder, the MEADS rows time warmup + sampling only. The verdict does not depend on
wall time -- it is decided by R-hat and ESS.

| model | arm | max R-hat | min bulk ESS | gradients | grad / ESS | mean shift vs ChEES |
|---|---|---|---|---|---|---|
| 8 Schools | ChEES + dynamic HMC | 1.0020 | 22,652 | 68,754 | 3.0 | -- |
| 8 Schools | MEADS 500 + GHMC | **1.0331** | 2,775 | 1,750 | 0.6 | 0.224 sd |
| 8 Schools | MEADS 2,000 + GHMC | **1.0308** | 3,390 | 3,250 | 1.0 | 0.176 sd |
| radon non-centered | ChEES + dynamic HMC | 1.0075 | 4,102 | 123,752 | 30.2 | -- |
| radon non-centered | MEADS 500 + GHMC | **343.9** | 32 | 1,750 | 54.7 | 31.9 sd, eps = 0.0 |
| radon non-centered | MEADS 2,000 + GHMC | **4.4e7** | 32 | 3,250 | 101.6 | 31.9 sd, eps = NaN |

Verdict: **not shippable.**

- On 8 Schools the gradient economy is real -- 4.8x fewer gradients per effective sample -- and it is
  bought with an unconverged fit: R-hat 1.031-1.033 against the 1.01 gate, 8x less ESS at equal draws,
  and posterior means 0.18-0.22 sd away from the ChEES reference (the noise floor between two
  legitimate ChEES runs is 0.011 sd, so that gap is 20x noise). More warmup does not fix it: 2,000
  MEADS steps is no better than 500.
- On radon d=175 the adaptation degenerates: step size 0.0 at 500 steps and NaN at 2,000, with R-hat
  reported as 4.4e7. One leapfrog per step is cheap only when the step is finite.

Why the 20-86x warmup-gradient saving from the source audit did not survive contact: it counted
gradient *evaluations*, and GHMC spends one per step while ChEES spends L = 21-38. But ChEES's long
trajectory is what buys decorrelation per draw -- 22,652 ESS from 40,000 draws versus 2,775 -- so the
per-evaluation saving is repaid in evaluations spent fighting autocorrelation, and on these two models
the trade is negative.

Two things worth keeping from the probe:

1. The seam works. MEADS warmup -> `blackjax.mcmc.ghmc.as_top_level_api` -> vmapped `step` in a scan
   reproduces a target with variances [4.0, 0.25] as [4.38, 0.27] at max R-hat 1.006 (CPU, 32 chains).
   `momentum_inverse_scale` goes in unchanged -- do not transform it. Note the argument-order trap:
   `init(position, key)` but `step(key, state)`; the docstring at ghmc.py:264 shows both backwards.
2. MEADS needs `num_chains % num_folds == 0` (ValueError otherwise) and `num_chains // num_folds >= 2`,
   which silently excludes 32-chain defaults with 4 folds only at small chain counts, and a guard is
   mandatory because the n_per_fold == 1 case returns NaN without raising.

E5 status: the warmup-gradient lever is closed negative, along with the stopping rule. What remains
measured and open is the metric refresh during warmup (the lane showed it is a value change, not a
state rebuild) and the 5-seed end-to-end confirmation of the batched finite-start check.

### Five-seed confirmation of the batched start check (8 Schools, 32 chains, 1,250 draws, 2,000 warmup)

The previous note reported the repair cost dropping 1.75 s -> 0.24 s but no end-to-end gain, because
the sampling stage read 3.11 s before and 4.52 s after across two seeds. Five warm seeds say the
sampling stage was the unstable measurement, not the change:

| seed | total | finite-start check | ChEES warmup | sampling | max R-hat | min bulk ESS |
|---|---|---|---|---|---|---|
| 43 | 10.42 s | 0.23 s | 4.10 s | 3.09 s | 1.0020 | 22,652 |
| 44 | 10.64 s | 0.25 s | 4.12 s | 3.17 s | 1.0017 | 22,567 |
| 45 | 10.57 s | 0.26 s | 4.27 s | 3.09 s | 1.0016 | 21,997 |
| 46 | 10.63 s | 0.25 s | 4.15 s | 3.11 s | 1.0020 | 21,240 |
| 47 | 10.85 s | 0.26 s | 4.43 s | 3.12 s | 1.0020 | 22,086 |
| **median** | **10.63 s** | **0.25 s** | **4.15 s** | **3.11 s** | **1.0020** | **22,086** |

Against the pre-change warm reference of 11.86 s with a 1.75 s repair: **1.23 s faster (10%)**, and
the per-seed spread on the total is 0.43 s (4%), so the gain is larger than seed-to-seed variation.
Convergence readouts are flat across all five seeds.

Method note: the ESS in this table is arviz's per-chain quantity under this script's own convention and
is only comparable within it. Cross-model ESS/s claims use `tools/bench_ladder.py`, which is what the
grid numbers elsewhere in this file come from.

## Chains at fixed total draws: parallelism buys throughput, not latency (E5)

Prompt for this section: if a GPU has thousands of lanes, why don't many chains beat a few long
chains at the same number of draws? Measured on the RTX A2000, 32 cores on the other arm, warm rows
(compile excluded), `c/step = wall / (warmup + draws_per_chain)`.

Fixed total draws = 40,000, warmup 2,000:

| model | chains | draws/chain | wall | c/step | max R-hat | ESS/s worst var |
|---|---|---|---|---|---|---|
| 8 Schools | 32 | 1,250 | **10.3 s** | 3.17 ms | 1.0013 | 2,185 |
| 8 Schools | 128 | 313 | 11.1 s | 4.79 ms | 1.0060 | 2,118 |
| 8 Schools | 512 | 78 | 17.7 s | 8.51 ms | 1.0229 | 1,387 |
| 8 Schools | 1,024 | 39 | 27.9 s | 13.69 ms | **1.0404** | 940 |
| logistic 1000x25 | 32 | 1,250 | **34.3 s** | 10.54 ms | 1.0022 | 355 |
| logistic 1000x25 | 64 | 625 | 46.8 s | 17.84 ms | 1.0024 | **525** |
| logistic 1000x25 | 512 | 78 | 181.6 s | 87.39 ms | 1.0164 | 146 |
| radon non-centered | 32 | 1,250 | **38.6 s** | 11.89 ms | 1.0069 | 105 |
| radon non-centered | 64 | 625 | 38.6 s | 14.70 ms | 1.0124 | 110 |
| radon non-centered | 128 / 256 | — | **OOM** | — | — | — |

**What you hold fixed is the whole answer.** At fixed *draws per chain* the GPU's parallelism pays
comfortably -- 128 chains is 3.2x the 32-chain throughput on 8 Schools and 3.8x on logistic (grid
above). At fixed *total draws* it loses monotonically, because wall is

    wall(C) = (W + D/C) x c(C),   W = 2,000 warmup steps

and the two terms move in opposite directions: `D/C` shrinks with chains while `c(C)` grows. At
1,024 chains there are only 1.6x fewer sequential steps than at 32, while each step costs 4.3x more.
ChEES warmup is a chain of 2,000 *dependent* adaptation updates; parallelism lives inside a step, so
adding chains cannot remove one.

Quality moves the same way -- max R-hat 1.0013 -> 1.0404 and worst-variable ESS/s 2,185 -> 940 going
32 -> 1,024 chains. Short chains do not substitute for long ones.

### Warmup length versus chain count (fixed 10,000 total draws, warm rows)

| warmup | 8 Schools C=32 | C=128 | C=512 |
|---|---|---|---|
| 100 | 6.5 s / R-hat 1.0408 | 8.2 s / 1.0787 | 13.7 s / 1.1148 |
| 250 | 6.7 s / 1.0088 | 8.2 s / 1.0218 | 14.1 s / 1.1118 |
| 500 | 7.1 s / 1.0063 | 8.6 s / 1.0213 | 14.3 s / 1.0822 |
| 1,000 | 7.8 s / 1.0067 | 9.3 s / 1.0215 | 15.5 s / 1.0559 |
| 2,000 | 9.1 s / 1.0073 | 10.8 s / 1.0245 | 17.3 s / 1.0938 |

More warmup does not repair what more chains broke: at 512 chains (20 draws per chain) R-hat stays
1.08-1.11 at every warmup length. A Pathfinder start needs draws-per-chain to burn off, not warmup.

Radon at C=32, the same grid: R-hat 2.2397 (W=100), 1.6646 (250), 1.4540 (500), 1.0698 (1,000),
1.0381 (2,000) -- monotone, and still above the 1.01 bar at the shipped default for a 10,000-draw
budget at d=175.

### Warmup 500 versus 2,000, five seeds -- why there is no smaller default

| model | warmup | wall | max R-hat | ESS/s worst var | realised steps/draw |
|---|---|---|---|---|---|
| 8 Schools (d=8) | 500 | **8.52 s** | 1.0020 | **2,260** | 19.8 |
| 8 Schools (d=8) | 2,000 | 10.66 s | 1.0020 | 2,098 | 21.4 |
| logistic (d=25) | 500 | 12.48 s | **1.0433** | **63** | **15.4** |
| logistic (d=25) | 2,000 | 35.58 s | 1.0020 | 389 | 59.9 |

A dimension-scaled warmup looked tempting after 8 Schools (20% faster for free) and was killed by its
own control: logistic is d=25 and at warmup 500 it lost 6x ESS with R-hat 1.043, because ChEES had
only grown the trajectory to 15.4 steps where the geometry needs ~60. **Required warmup tracks
realised trajectory length, which is not knowable before warmup runs.** No default change ships from
this. It is also the strongest argument in the project for a criterion-based stop rule -- and that
rule currently measures cost-negative (+6.7-9.0 s of block-boundary overhead), so the remaining
honest option is to move the loop inside the trace, not to pick a smaller constant.

### Two claims from earlier in this file that these runs refuted

1. "128 chains does not fit because every chain re-reads the data" -- the OOM ceiling is not
   explained by the draw pool: radon OOMs at 128 chains even with warmup 100, where the stacked
   warmup output is tiny. Measured and unrefuted fact: **radon's ceiling on this card is 64 chains at
   any warmup or draw count.** Where the memory actually goes (Pathfinder pool, step-size tuner, or
   warmup state) is NOT ESTABLISHED -- the earlier explanation was wrong twice and should not be
   repeated.
2. "`c(C)` is roughly flat, so chains are nearly free." True for the 32-chain batched gradient alone
   (23.0 -> 25.3 us from 32 to 128 chains) and false for the full transition, which is what the wall
   is made of: c/step on 8 Schools rises 3.17 -> 13.69 ms from 32 to 1,024 chains.
