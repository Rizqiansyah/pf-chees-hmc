# C7 SUMMARY (example notebooks)

## DEVIATIONS
1. **Finding in the library, not fixed (out of scope): an integer-valued float prior location recompiles once per new value.**
   Found while writing `New data every call.ipynb`. With a `pm.Data` regression and `b = pm.Normal("b", m, 3.0)`, calls with
   `m = 0.0, 1.0, -1.0, 2.0, 0.0, 1.0, 2.0, 3.0` (same data shapes, chains 1024 or 256) compile 0 programs except at the first
   sighting of 1.0, 2.0 and 3.0 (8 XLA programs, about 4 s each: `pathfinder_fit`, `warmup_init`, `warmup_chunk`, `search_step_sizes`,
   `score_pool`, `score_positions`, `sampling_chunk`, `usable_mask`). Values 4.0, -1.0, 1.5, 0.37, 1.21, -0.8, 2.3, 0.9 compile nothing.
   So README "new prior values compile nothing" does not hold for every value. Likely PyTensor's constant autocast
   or a rewrite on small integral values (`_widen_model_inputs` / the `lifted.py` memo); not investigated. Repro:
   `/tmp/-1002/scr/dbg2.py "[1.0,2.0,3.0,4.0]"` (scratch, not committed): model `a~N(0,5); b~N(m,3); s~HalfN(2); obs~N(a+b*x,s)`,
   `x, y` as `pm.Data`, N=200, 256 chains, 10 draws, W=20, printing the count of `/jax/core/compile/backend_compile_duration` events.
   The notebook uses non-round prior values (0.3, 1.2, -0.7, 2.4; scales 3.1, 2.4, 4.2, 3.7), so it shows 0 programs on calls 2 to 4.
2. R-hat at 10 draws per chain is inflated even in float64 (8 Schools 1.46, GEV 1.89 max). The brief and README say "R-hat far above 1.01";
   the notebooks therefore say to compare with a float64 run (float32 GEV: 85.7 vs 1.89). README text left alone (only the `examples/` edit allowed).
3. The float32 notebook's float64 timing baseline is `benchmarks/warm_stages.py` (same graph from `benchmarks/models.py`) while the float32 side
   is built inline. The logistic data differ (inline: AR(1)-correlated predictors, sparse beta, own seed), so the logistic ratio is rough. Stated in the notebook.
4. The GEV float64 reference for the step size comes from a child process running the same inline model code (`python -c`), not from `benchmarks/`
   (allowed: it is not a `benchmarks/` import). `float32.ipynb` also writes nothing to disk except `~/.cache/pf-ladder/radon_mn.json` (needs network on first use, like `benchmarks/models.py`).
5. Every notebook has a `%matplotlib inline` line in its setup cell: in this env the kernel backend was Agg and no plot was stored otherwise.

## Spec items
- Flag cell (`USE_FLOAT32`, first code cell, loud markdown above it with opt-in, restart, where it helps, GEV failure, the three checks, README section link by name): DONE in all four. Default False; `float32.ipynb` True.
- `8 Schools example.ipynb`: DONE (model, sample, summary, trace plot of 8 chains, prints `frozen_step_size`, mean `num_integration_steps`, max R-hat). Executed with True in a scratch copy (see below) and committed at False.
- `GEV Port Pirie example.ipynb`: DONE (data, model, prior and posterior predictive, NUTS comparison via `pm.sample(draws=1000, chains=4, tune=1000, target_accept=0.95)` printed side by side). Plots capped: trace of 8 chains, posterior predictive from 100 chains, dpi 72. Executed size 0.77 MB (was 1.7 MB).
- `New data every call.ipynb`: DONE (`pm.Data` x and y, 4 datasets and priors, rebuild per call, per-call wall time and XLA programs, posterior vs true slope table and histogram, disk cache section).
- `float32.ipynb`: DONE (see below). No GEV fix attempted.
- README: two sentences changed (lines 38, 125) to name the four notebooks. Nothing else.
- No notebook test added. Committed with outputs cleared.

## Per notebook (CPU, `OMP_NUM_THREADS=4`, load 1 to 8 from other jobs, XDG_CACHE_HOME empty before the first run)
Defaults: 1,024 chains, 10 draws, device-agnostic (prints `jax.devices()`, floatX, x64).

1. **8 Schools** (cells: flag md, flag, intro, setup, model, sample, checks md, checks, trace plot). W=25. 17 s cold cache, 8.7 s with the cache warm.
   Prints: frozen step size 0.2819, mean `num_integration_steps` 13.7, max R-hat 1.46. Scratch copy with `USE_FLOAT32 = True`: floatX float32, x64 False,
   step size 0.2558, max R-hat 1.47, 18 s, no errors.
2. **GEV Port Pirie** (flag md and cell, intro, setup, data and hist, model, prior predictive, sample + summary, trace, posterior predictive + dist comparison, NUTS md, NUTS + comparison table, NUTS trace). W=25. 24 s cold, 20 s warm.
   Prints: step size 0.520; sample: mu 3.872, sigma 0.204, xi -0.029, z_p 4.318; NUTS: 3.872, 0.205, -0.029, 4.320 (5 divergences after tuning in NUTS, 2 s).
3. **New data every call** (flag md and cell, explanation md, setup, model function + program counter, loop and table, histogram, disk cache md). W=50. 11 s cold, 8.8 s warm.
   Call 1: 5.07 s, 103 programs (2.6 s when the disk cache was warm: the events count loads as well); calls 2 to 4: 0.25 / 0.23 / 0.24 s, 0 programs.
   Slopes: true -2 / 0.5 / 1.5 / 3 -> posterior mean -2.034 / 0.474 / 1.502 / 3.018 (sd 0.03 to 0.04).
4. **float32** (flag md and cell True, intro, setup with `assert floatX == float32`, part 1: inline logistic and radon, timing table; part 2: GEV, three signs table). 270 s.
   Part 1 (warm calls 2 and 3 mean, CPU): logistic float64 4.26 s, float32 1.26 s, ratio 3.37 (calls: f32 7.9 / 1.3 / 1.2, f64 10.7 / 4.4 / 4.2);
   radon float64 28.4 s, float32 22.2 s, ratio 1.28 (f32 35.9 / 24.6 / 19.8, f64 42.9 / 33.7 / 23.2). The notebook says the ratio is not a GPU prediction.
   Part 2 (GEV, W=25, seed 42): float64 child: step 0.520, max R-hat 1.89, mean steps 7.1; float32: step 1.7e-5, max R-hat 85.7, mean steps 312.8.
   The text says plainly that `b2_max` can pass while the sampler does not move.

## GPU command list (no edit needed; run from the repo root, `examples/` as cwd for the notebook paths)
```
cd examples
for n in "8 Schools example" "GEV Port Pirie example" "New data every call" "float32"; do
  ../benchmarks/gpu_run.sh pixi run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$n.ipynb"
done
```
(`pixi run` already puts `pfcheeshmc` on the path when the package is installed in the env; if not, prefix `env PYTHONPATH=<repo>`, as I did on the CPU.)
Do not set `JAX_PLATFORMS=cpu` or `CUDA_VISIBLE_DEVICES=""`.

Device-dependent items: the device line, all wall times, the float32 ratios and the float32 GEV numbers (step size, R-hat, steps; README measured 3e-5 to 9e-3 and R-hat 57 to 112 on the GPU),
and the 8 Schools / float32 samples. `float32.ipynb` spawns child processes that inherit the environment (GPU too): run it with the GPU free; its radon part takes about 4.5 min on this CPU.
`float32.ipynb` runs `benchmarks/warm_stages.py` which turns the disk cache off, and `New data every call` call 1 depends on the disk cache state (check `xla_programs`).
Execution order dependence: restart the kernel before changing `USE_FLOAT32`.

## Suite
`542 passed, 48 warnings in 460.39s (0:07:40)` (load average at start 3.24, at end 13.32 from other jobs on the shared host). Command: the TASK-RULES pytest line.

## UNVERIFIED
- Nothing was run on a GPU; the GPU numbers in the notebook text are the README's.
- The cause of deviation 1 is unexamined.
- `float32.ipynb` needs network on first use for the radon data (cached in `~/.cache/pf-ladder`; present on this host).
