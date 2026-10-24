# F2k SUMMARY: float32 (`PYTENSOR_FLAGS=floatX=float32`), measured and fixed, no default change

Task: fast-tier, one round. It stopped once at the time limit and was resumed.
- Branch `f2-k`: 5 commits on 62c38a3, head 9392c38. Merged into `f2` with `--no-ff` (58b5c93).
- Task's suite: `596 passed, 63 warnings in 755.67s (0:12:35)`. It was 584; this adds 12 tests in
  `tests/test_float32.py` (about 47 s), with child runner `tests/float32_run.py`.
- Lead suite on merged `f2`: `596 passed, 63 warnings in 752.89s (0:12:32)`, rc 0 (58b5c93; load
  0.8 → 2.7). The real `~/.cache/pfcheeshmc` does not exist after it.

## Result
- **float32 never ran because of the example notebooks, not because of a defect in this package, BlackJAX or
  optax.** Their first cell runs `jax.config.update("jax_enable_x64", True)`, and `bench_ladder.load_notebook_model`
  runs that cell.
  - PyTensor had already set x64 off from floatX. The notebook turned it back on for the whole process.
  - The 8 Schools data are float64 arrays, which PyMC leaves float64. So the log-density came out float64 at a
    float32 position, and the L-BFGS line search's `lax.cond` refused the mismatch ("cond branches must have equal
    output types"). The leaf that differs is the L-BFGS `value`.
  - Every model built after the first notebook model failed the same way. On 62c38a3, `logit_1000x25` alone under
    float32 runs.
- **The fix is in the tools.** `tools/r1_bias.build_model` switches x64 back off after the build when floatX is not
  float64.
- **The sampler now warns about the contradiction.** `sample_pf_chees_hmc` raises a `RuntimeWarning` when floatX is
  narrower than float64 and x64 is on (`sampler.py:46`). The warning names the cause and the fix. It never fires
  at float64.
  - The task tried a log-density cast to the position's dtype instead, and removed it. With x64 on, the HMC loops
    draw momentum in float64, so the cast gave only half-support.
- **Audit fixes, each dtype-generic and bit-identical at float64:**
  - the Halton probe asks for `int`, not `int64` (`sampler.py:192`);
  - the start-repair jitter and restart noise take the position's dtype (`sampler.py:436, 454`);
  - JAX's "Explicitly requested dtype float64 ..." warning is dropped inside the lifted function while x64 is off
    (`lifted.py:148-166`). It comes from PyTensor's own `Sum` accumulator cast (`.astype(acc_dtype)`), and its
    advice would undo the float32 run.
- **`erfcx` needs no change.** At float32 the guard band is `lifted._band(float32)` = (9.185, 9.429). JAX's own
  `erfcx` is wrong (0 or NaN) exactly in [9.19455, 9.41928]. On a 5.6M-point float32 grid from -20 to 1e5, the
  guarded function has no point off by more than 1e-5 (`results/f2/probes/erfcx_float32_band.py`).
- **Variants:** 14 sampler variants × 4 models run under float32 with finite float32 draws and no dtype warnings
  (`results/f2/runs/F2k/f32_variants.txt`). They cover `pf_closed`, the low-rank and precision metrics, the iid pick,
  `start_scale`, `pm_chains`, the four refresh estimators, `iterate raw` and `warmup_stop`.

## Lead review
- **The float64 path is unchanged.** Every library change reduces to the old code when x64 is on.
  - The `RuntimeWarning` and the warning filter only act when floatX and x64 disagree, or when x64 is off.
  - The pick change (deviation 4) changes nothing when every pool row is finite.
  - Proof: a child-process test compares draws with `git archive 62c38a3` (8schools and gev, both `init_source`
    values): every `posterior/*` and `sample_stats/*` array is equal, with the same dtypes.
- **Deviation 4 is accepted as a bug fix, not a default change.** The pick sorted the pool by log-density and took
  evenly spaced ranks from the lowest, so a pool row outside the support (`-inf`) became chain 0's start. Every
  proposal from there is rejected, so the chain sat at `-inf` for the whole run. That can happen at float64 too,
  wherever a Pathfinder pool holds such a row. Now non-finite rows are ranked last and left out of the ranks.
  - The pick now runs in NumPy on the host. It costs one device-to-host copy of the scored subset (a few thousand
    floats), and the eager `argsort`, `arange` and floor-divide programs it compiled before are gone. The compile
    tests pass.
- **GEV under float32 fails the R1 gate on the CPU** (2 of 3 seeds) because its step size collapses (below). This is
  a finding about float32 on this model, not a regression: float64 is untouched. Nothing in the sampler was
  changed for it, as the plan says ("report only").
- **Small, not fixed:** `build_model` imports `jax` and `pytensor` again inside the function, although the module
  already imports them. This is harmless.

## DEVIATIONS (accepted)
1. **No upstream defect, so no upstream repro.** The cause and fix are above.
2. **`floatX` goes in the rows' `kwargs` JSON, not as a top-level key.** The top-level keys are fixed by
   `tools/_common.py`, which is outside the task's file list. `warm_stages.py` adds `"floatX"` to each per-call dict
   and prints `floatX` and the x64 state in its header.
3. **Files beyond the list:** `results/f2/runs/F2k/*` (CPU rows) and `tests/float32_run.py`. The tools did not need
   b² scored in float64: `flatten_run` and `score_ensemble` already cast to float64 NumPy.
4. **The stratified pick skips non-finite pool rows** (`sampler.py:127`), with the review above. Before the fix,
   float32 GEV at C=8 and C=256 had a chain stuck at `-inf`; after it, there are 0 non-finite chains.
5. **A new `RuntimeWarning`** when floatX is narrower than float64 and x64 is on (`sampler.py:46`).

## Tests (`tests/test_float32.py`, 12, about 47 s; child runner `tests/float32_run.py`)
- **float32 runs:** each ladder model in its own float32 child, both `init_source` values, 8 chains × 10 draws.
  - The run finishes, the draws are finite, and every posterior array is float32.
  - `jax_enable_x64` is False in the child, and no "Explicitly requested dtype" warning is raised.
  - The child appends to `PYTENSOR_FLAGS` rather than replacing it.
- **The leaf regression:** after `build_model`, x64 is off for 8schools and gev.
- **The warning:** x64 on under float32 gives exactly one warning naming the fix.
- **float64 bit-identity** against `git archive 62c38a3`, as above. It also asserts that `build_model` leaves x64
  on at float64.
- **In-process unit tests:**
  - the Halton probe uses no int64 when x64 is off;
  - the start repair keeps the position's dtype;
  - the stratified pick, including a `-inf` row;
  - erfcx at float32 from -9 to 1e5.
- **Must-fail:** the module run against a `git archive 62c38a3` tree gives `9 failed, 1 passed (erfcx), 2 skipped`
  (the identity tests need git). The failures include:
  - `x64_built is True`;
  - `TypeError: cond branches must have equal output types but they differ.` for both init sources;
  - 44–63 "Explicitly requested dtype float64" warnings;
  - `dtype('float64') == jnp.float32` (start repair);
  - `[10, 70, 60] == [50, 30, 60]` (the pick).

## CPU (task): float32 against float64, same shape and seeds
C=1024, D=10, seeds 1–3, `--repeat 1`, `OMP_NUM_THREADS=8`. Rows are in `results/f2/runs/F2k/r1_float{32,64}__*.jsonl`.
The command for one model and floatX (drop `floatX=float32,` for float64):
```
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=8
env -u JAX_PLATFORM_NAME PYTENSOR_FLAGS="floatX=float32,blas__ldflags=-llapack -lblas -lcblas -lm" pixi run --manifest-path "$MAIN/pixi.toml" \
  env PYTHONPATH="$WT" python tools/r1_bias.py --models M --chains 1024 --draws 10 --warmups W --seeds 1,2,3 --repeat 1 \
  --arm f2k:float32 --wp F2 --step F2k --out results/f2/runs/F2k/r1_float32__M.jsonl
```

| model (W) | floatX | b²_max, seeds 1 / 2 / 3 | MC floor | G-R1 | divergences per 1k | frozen ε | L |
|---|---|---|---|---|---|---|---|
| 8schools (25) | f32 | 0.0010 / 0.0003 / 0.0008 | 0.0019 / 0.0015 / 0.0016 | pass ×3 | 0 / 0 / 0 | 0.23 / 0.33 / 0.26 | 24 / 23 / 28 |
| | f64 | 0.0007 / 0.0007 / 0.0018 | 0.0016 / 0.0016 / 0.0015 | pass ×3 | 0 / 0 / 0 | 0.26 / 0.26 / 0.32 | 28 / 28 / 23 |
| gev (25) | f32 | **0.0112 / 0.4703 / 0.0068** | 0.0021 / 0.0059 / 0.0017 | **fail / fail / pass** | 0.29 / 0 / 0.29 | **6e-5 / 7e-3 / 9e-6** | 307 / 4 / 449 |
| | f64 | 0.0014 / 0.0003 / 0.0002 | 0.0011 / 0.0014 / 0.0011 | pass ×3 | 0 / 0.20 / 0 | 0.54 / 0.52 / 0.51 | 11 / 14 / 14 |
| logit_1000x25 (50) | f32 | 0.0009 / 0.0065 / 0.0007 | 0.0026 / 0.0023 / 0.0023 | pass ×3 | 0 / 0 / 0 | 0.52 / 0.46 / 0.48 | 10 / 11 / 10 |
| | f64 | 0.0039 / 0.0044 / 0.0011 | 0.0024 / 0.0024 / 0.0025 | pass ×3 | 0 / 0 / 0 | 0.52 / 0.57 / 0.58 | 10 / 9 / 9 |
| radon_noncentered (300) | f32 | 0.0024 / 0.0026 / 0.0031 | 0.0037 / 0.0045 / 0.0043 | pass ×3 | 0 / 0 / 0 | 0.83 / 0.84 / 0.84 | 21 / 19 / 15 |
| | f64 | 0.0033 / 0.0024 / 0.0041 | 0.0032 / 0.0037 / 0.0045 | pass ×3 | 0 / 0 / 0 | 0.85 / 0.84 / 0.85 | 21 / 21 / 16 |

- **8schools, logit and radon:** float32 matches float64 within seed noise.
- **GEV:** under float32 the step size collapses in all three seeds, and split R-hat is 68–123.
  - Longer warmup does not help: W=100 and W=400 give ε from 1.4e-5 to 0.35 and R-hat up to 88
    (`r1_float32__gev_longer_warmup.jsonl`). Those b² pass only because the Pathfinder starts are already near the
    posterior.
  - Measured:
    - In the warmup trace, the harmonic-mean acceptance ChEES adapts on is 0 at 7 of the first 25 steps, and ε
      falls from 3.9 to 9e-6. float64 hits the same zeros and recovers.
    - The float32 gradient of PyMC's `GenExtreme` density has a median relative error of 2.7e-4 in the smallest-|ξ|
      quintile, with a maximum of 18; elsewhere it is about 1e-5 (`f32_logp_grad_error.py`).
    - At reference-posterior points with the reference metric, float32 HMC has chains at acceptance 0–1e-6 that are
      not flagged divergent, and the harmonic mean is 0.28 at ε 0.05. float64's minimum is 0.72 at ε 0.2
      (`f32_hmc_acceptance.py`).
  - UNVERIFIED: that the gradient noise near ξ = 0 is the whole cause.

Warm CPU wall per call (`tools/warm_stages.py TREE M 1024 10 W --calls 3`; no compile on calls 2 and 3; host load
1–9; informative only):

| model | f32, calls 2 / 3 | f64, calls 2 / 3 |
|---|---|---|
| 8schools | 0.31 / 0.31 s | 0.36 / 0.36 s |
| gev | 10.0 / 21.5 s (collapsed ε, long trajectories) | 1.1 / 0.9 s |
| logit_1000x25 | 1.49 / 1.53 s | 5.45 / 5.20 s |
| radon_noncentered | 29.5 / 23.4 s | 39.1 / 26.9 s (noisy) |

## GPU (lead)
2026-10-24 16:40–16:51, f2 9697f6c (the merged F2k tree). The host load was 2–5, from other users, and only our
jobs ran on the GPU. Drivers: `results/f2/drivers/f2k_gpu.sh` (warm stages, float64 then float32 per model, in one
session) and `run_step.sh` with `F2k_gate.cmds`, `F2k_seeds_f32.cmds` and `F2k_seeds_f64.cmds`. Rows:
`results/f2/runs/F2k/gpu_*.json`, `results/f2/runs/F2k__float32.jsonl` and `results/f2/runs/F2k_seeds__f{32,64}.jsonl`.
All tables are in `gpu.md` (from `f2k_table.py`). Every job printed the floatX it ran at, and every float32 job had
`jax_enable_x64: False`.

**Warm call on the GPU** (C=1024, D=10, same data, cache off, mean of calls 2–3, 0 compiles on both):

| model (W) | float64 | float32 | ratio | call 1, f64 → f32 |
|---|---|---|---|---|
| 8schools (25) | 0.21 s | 0.16 s | 1.3× | 16.0 → 14.4 s |
| gev (25) | 0.47 s | 0.75 s | **0.6×** (the collapsed ε) | 26.6 → 27.0 s |
| logit_1000x25 (50) | 1.39 s | **0.18 s** | **7.7×** | 31.5 → 19.4 s |
| radon_noncentered (300) | 6.42 s | **2.06 s** | **3.1×** | 43.7 → 44.0 s |

The speed-up is in the warmup and sampling stages: logit 1.07 → 0.07 s of warmup, radon 5.59 → 1.72 s. Pathfinder
and init are already small. GEV's warm init is 0.026 → 0.162 s under float32.

**R1 gate under float32 on the GPU** (seeds 1–3, 2 reps; the float64 rows are F2f's, from the same shapes):

| model (W) | b²_max seeds 1–3, f64 (F2f) → f32 | G-R1 f32 | ε f32 | L f32 | max R-hat f32 | warm wall f64 → f32 |
|---|---|---|---|---|---|---|
| 8schools (25) | 0.0007 0.0007 0.0018 → **0.0142** 0.0005 0.0008 | **2/3** | 0.27 0.32 0.26 | **15** 23 28 | **2.40** 1.44 1.48 | 0.38 → 0.30 s |
| gev (25) | 0.0014 0.0003 0.0002 → 0.0068 0.0023 0.0072 | 3/3 (not sampling, below) | **2.8e-5 7.4e-4 8.9e-3** | 112 16 3 | **112 92 57** | 0.63 → 0.58 s |
| logit_1000x25 (50) | 0.0039 0.0045 0.0011 → 0.0008 0.0056 0.0007 | 3/3 | 0.52 0.48 0.47 | 10 10 11 | 2.04 1.96 1.93 | 1.79 → **0.50 s** |
| radon_noncentered (300) | 0.0032 0.0025 0.0031 → 0.0039 0.0029 0.0032 | 3/3 | 0.86 0.86 0.85 | 19 18 15 | 1.72 1.69 1.85 | 11.15 → **5.58 s** |

The warm wall here includes the harness's scoring, about 4 s on radon (F2f).

- **logit and radon:** float32 passes on every seed, with ε and L equal to float64's, and runs 3–8× faster warm.
- **GEV:** the CPU collapse repeats on the GPU. ε falls to 3e-5–9e-3 and R-hat is 57–112. The b² values pass
  only because the chains barely move from Pathfinder starts that are already close. Count it as a fail.
- **8schools:** seed 1 fails, and its repeat is identical. The run took a short-trajectory path: L 15, 610
  warmup divergences against about 1,350 on the other seeds, and the bias is in `mu`. On the CPU, float32 seed 1
  passed (L 24). To tell float32 from W=25's own fragility, seeds 4–13 were run at both floatXs:

  | 8schools, W=25, GPU | seeds | fail | the failing signature |
  |---|---|---|---|
  | float32 | 1–13 | **2** (seeds 1 and 4: b² 0.0142, 0.0191) | L 14–15, warmup divergences 610–631, R-hat 2.4, bias in `mu` |
  | float64 | 1–13 | 0 | never seen; the nearest is seed 12 (L 22, 1,082 warmup divergences, R-hat 1.80, b² 0.0009) |

  Two failures in 13 against none in 13 is not significant by itself (Fisher's exact test, p ≈ 0.48). But both
  float32 failures share one signature that float64 never reached, so this is treated as a float32 effect at
  W=25 until shown otherwise. UNVERIFIED: whether a longer warmup removes it.

**Against CPU NUTS** (F2f's per-call figures: logit 9.68 s, radon 18.3 s, 32 chains at their passing configs).
Measured warm calls on logit and radon would move F2f's GPU-vs-CPU ratio from 6.8× to about 54×, and from 2.5–3.8×
to about 9×. This compares calls from different sessions; this session's float64 rows match F2f's (1.39 against
1.43 s; 6.42 against 4.8–7.4 s).

## UNVERIFIED
- Whether a longer warmup removes 8schools' float32 failure signature on the GPU.
- The GEV mechanism beyond the measured gradient error and acceptance zeros.
- The other `tools/*.py` (`wp*`, `bench_ladder`) under float32: not run.
