# F2b-2 SUMMARY — warmup and sampling loops as host-driven chunk programs

Task: fast-tier, one round (interrupted once and resumed).
- Branch `f2-b2`: commits 34e85d1, ea4a65b, 4326292 on c35fbba. Merged into `f2` with `--no-ff`.
- Task's suite: `361 passed, 41 warnings in 445.40s (0:07:25)`.
- Files: `pfcheeshmc/compiled.py`, `pfcheeshmc/utils.py`, `pfcheeshmc/sampler.py`,
  `tests/test_compile_once.py`, `tests/test_progressbar.py` (docstring); new
  `tests/test_chunked_loops.py`, `tests/known_posterior_run.py`, `tools/chunk_overhead.py`.

## Result
A warm second call compiles **0** programs on all four ladder models. c35fbba compiled 2, and `main`
compiled 8. Each of these also compiles 0:
- a new seed, data or priors;
- W 20→37 and 20→5;
- D 10→23 and 10→3;
- `progressbar` on↔off;
- the refresh with another W_A;
- superchains;
- the stop rule.

`jax.debug.callback` is gone from every loop program (grep of the jaxprs).

## Design
- **Programs:** `compiled.warmup_init`, `warmup_chunk` and `sampling_chunk` are module-level jits.
  - **Static:** `fn`, `names` and the settings (Adam lr as a float, target_accept, decay_rate,
    jitter_amount, max_leapfrog_steps, iterate, single_step_iters, chunk, stack_positions, has_t0).
  - **Dynamic:** the values, the carry, the metric, ε, t0, key, start, n and traj_len_adj.
- **Chunks:** each chunk is a `lax.scan` of `chunk` steps. A step is skipped with `lax.cond(j < n)` on
  a scalar outside the per-chain vmap, so skipped steps cost no model evaluation.
- **Host loop:** `run_chunks` keeps one chunk in flight, and `HostStack` gathers per-step outputs into
  NumPy, so post-processing never device-slices with a W- or D-dependent bound.
- **Chunk sizes:** WARMUP_CHUNK = SAMPLE_CHUNK = TRANSFORM_CHUNK = 50. The value→RV transform runs over
  padded 50-draw chunks.
- **Halton:** the bit depth is fixed at 30.
- **Progress bars:** updated from the host between chunks.

## DEVIATIONS (all accepted)
1. **S_d is the fixed constant 50, not `min(D, 1000)`.** With `min(D, 1000)`, D would be a program
   shape and D=10→23 would compile; the plan's own goal rules that out. One program serves every D, and
   device memory per call is bounded by 50 draws. A run with D<50 copies back 50 rows: radon's warm
   transform went 0.05→0.13 s.
2. **The default path's key stream is unchanged.** Under `jax_threefry_partitionable` (JAX 0.11.2's
   default), `split(key, N)[i] == fold_in(key, i)` for every N and i tried. So per-step
   `fold_in(stage_key, global_index)` keeps the old stream, using public API only; a test pins this.
   - Draws vs c35fbba: bit-identical on the known-posterior model.
   - Ladder models differ by ≤ 1.9e-10 relative, with discrete statistics exactly equal.
3. **The stop-rule and refresh paths are new realisations.** Their blocks were keyed by a block-length
   split. They are checked statistically:

   | path | max \|z\| of the mean vs the analytic posterior | max \|z\| of the variance |
   |---|---|---|
   | refresh, old | 2.56 | 2.25 |
   | refresh, new | 3.03 | 1.69 |

   KS of new vs old: min p = 0.218 (bound 1e-3).
4. **The refresh pool `min(W_A, 20)` is an estimator shape.** W_A ≥ 20 changes nothing compiled; W_A < 20
   compiles once per distinct pool size. The default W_A is 50.
5. **API changes:**
   - `chees_warmup_with_progress` gained `positions_last=`, and takes `optim` as an Adam lr float or as
     an optax object (the latter recompiles per call);
   - `update_every` and `max_sampling_steps` are accepted but ignored;
   - the helper types moved to `compiled.py` and are re-exported;
   - `chees_parameters` gained `traj_len_adj` and `jitter_amount`.
6. **Bar behaviour:**
   - the warmup bar's divergence total is now exact; the old bar added only every 25th step;
   - updates come at 50-step granularity.
7. **Pre-existing defect, kept and flagged:** with a stop rule and several blocks, `WarmupInfo`
   holds only the last block, so gradient counts under-count (see the ledger).

Also found: the old Halton depth `ceil(log2(W+1000))` gave jitter exactly 0 (a 0-leapfrog draw) at
`i+1 == 2^bits`. The fixed 30-bit depth gives a positive value there; everywhere else the values are
bit-identical.

## Tests
- **`test_compile_once.py`** (69 tests): every compile-0 case above. Also:
  - multi-chunk regimes, using monkeypatched chunk sizes;
  - a shape change compiles exactly one each of `warmup_init`, `warmup_chunk`, `sampling_chunk` and
    `transform_draws`, then 0;
  - equivalence with c35fbba.
- **`test_chunked_loops.py`** (30 tests):
  - fold_in == split;
  - Halton depth;
  - determinism;
  - draws bit-identical across chunk sizes (50,100,100), (7,3,4), (1,1,1) and (50,7,23) on the
    default, refresh and stop-rule paths;
  - per-step `num_integration_steps` recomputed independently from Halton × L;
  - attrs totals;
  - known posterior (conjugate Normal-Normal, d=10, C=256, W=200, D=50) vs analytic and vs c35fbba;
  - progress bars;
  - no callbacks;
  - `run_chunks` order, `HostStack`, and the padded transform.
- **Mutations M1–M10** each fail the intended test. Examples:
  - a non-conditional skip, a chunk-local key, or a chunk-local Halton index each break chunk
    invariance;
  - a static `n` recompiles;
  - device slicing gives 5 `dynamic_slice` compiles;
  - an optax object recompiles;
  - a callback is detected in the jaxpr.

## Measurements (task, CPU, C=1024, D=10, W=25, load average 0.1–2.5; `tools/warm_stages.py`)
Warm call: the mean of calls 2 and 3. Cells are seconds / compiles.

| model | tree | wall | init | pathfinder | eps0 | warmup | sampling | transform |
|---|---|---|---|---|---|---|---|---|
| 8schools | c35fbba | 4.39/2 | 0.48 | 0.03 | 0.03 | 1.94/1 | 1.72/1 | 0.02 |
| 8schools | new | **0.99/0** | 0.47 | 0.03 | 0.03 | 0.20 | 0.07 | 0.02 |
| gev | c35fbba | 4.59/2 | 1.24 | 0.03 | 0.04 | 1.60/1 | 1.19/1 | 0.07 |
| gev | new | **2.56/0** | 1.24 | 0.03 | 0.04 | 0.56 | 0.20 | 0.06 |
| logit_1000x25 | c35fbba | 5.21/2 | 0.40 | 0.09 | 0.04 | 2.92/1 | 1.60/1 | 0.02 |
| logit_1000x25 | new | **3.72/0** | 0.39 | 0.10 | 0.04 | 2.13 | 0.90 | 0.02 |
| radon_noncentered | c35fbba | 7.05/2 | 0.65 | 1.08 | 0.12 | 3.13/1 | 1.74/1 | 0.05 |
| radon_noncentered | new | **5.00/0** | 0.64 | 1.08 | 0.11 | 1.98 | 0.76 | 0.13 |

- **Cold first call:** unchanged (13–20 s). The same programs compile once, and now they are reused.
- **Chunk overhead:** C=1024 is flat for chunk lengths 5–500. At C=32 a chunk costs about 0.05–0.2 ms
  overlapped and about 0.3 ms serial. A 50-step chunk is under 1.5% overhead at C=32.

## UNVERIFIED
- GPU: per-chunk overhead, and whether XLA:GPU reads the `cond` predicate back to the host per step.
  To be checked in F2f.
- float32.
- Quiet-host timings.
- W_A < 20.
- Memory at C=1024, d=175.
- `tools/t5_step_cost.py` still labels the host-bar cost "callback".
- `tools/r1_bias.nominal_length` uses the old Halton depth: same values for i < W, but not re-run.
