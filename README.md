# pf-chees-hmc
A Pathfinder-ChEES-HMC sampler using BlackJAX backend, compatible with PyMC model.

# ChEES-HMC vs. NUTS (or Why Use ChEES-HMC?)
The No-U-Turn sampler (NUTS) is a common and ubiquotous general purpose posterior sampling method that is the default if you use a PyMC model with continuous priors and likelihood.
It's a variant of the Hamiltonian Monte Carlo (HMC), with a main advantage that it avoids the need to tune the step size ($\epsilon$) and trajectory length ($L$) parameters in HMC, which is notoriously difficult and time consuming to do.
However, NUTS require substantial overhead in its computation, and it is not suitable for GPU acceleration.

ChEES-HMC, proposed in [1], tunes the HMC step size ($\epsilon$) and trajectory length ($L$) automatically during the warm up phase. *ChEES* is the **Ch**ange in the **E**stimator of the **E**xpected **S**quare criterion that the method maximises (section 2.2 and Algorithm 1 of [1]). Note that ChEES adapts $\epsilon$ and $L$ only — it does not adapt the mass matrix, which is why this implementation takes that from Pathfinder. That statement is about the part of BlackJAX this package calls: it uses BlackJAX's private `chees_adaptation.base()`, which adapts step size and trajectory length and holds the metric fixed. BlackJAX's public `chees_adaptation(...)` builder has an optional `mass_matrix_estimation="diagonal"` (a Welford estimator that engages over the latter part of warmup) which this package does not use.
HMC itself is suitable for GPU acceleration, and has less computational overhead compared to NUTS.
This allows posterior sampling efficiently using GPU(s).

NUTS is usually run over a small number of long chains, usually with the number of chains equal to the number of CPU cores available.
In contrast, ChEES-HMC can be run over a lot of short chains simultaneously in a GPU (or multiple GPUs).
This means that ChEES-HMC has the potential to sample much faster as the chains are shorter (for the same amount of effective sample size compared to NUTS).

# Implementation
The Pathfinder-ChEES-HMC sampler follows these steps:
1. Initialize the starting point using `adapt_diag` from `pymc` (the default initialization for NUTS in PyMC)
2. Use the Pathfinder algorithm (with some jitter) to find a better initial starting point and mass matrix.
3. Perform short HMC runs to tune the $\epsilon$ parameter.
4. Perform ChEES adaptation to tune the $\epsilon$, $L$, initial starting point, and other HMC parameters.
5. Run a Dynamic HMC using the tuned parameters from ChEES (Step 4) and mass matrix from the Pathfinder algorithm (Step 2)

# Installation

Requires Python 3.11+ (the floor set by BlackJAX 1.7). The repository ships a [pixi](https://pixi.sh) workspace that pins the GPU stack:

```
pixi install          # solve and create ./.pixi/envs/default
pixi run gpu-check    # must print: [CudaDevice(id=0)]
pixi run vram         # GPU name and memory
pixi run python       # or: pixi run jupyter lab
```

With pip instead, install the package and then a CUDA-enabled JAX:

```
pip install .
pip install -U "jax[cuda13]"      # GPU on Linux
```

# Versions and pins

- **jax / jaxlib >= 0.9, `cuda13` extra.** Needs driver >= 580 and an SM 7.5+ GPU. CUDA 12 wheels also work (driver >= 525, SM 5.2+) but are on the documented deprecation path. No system CUDA toolkit is required; the runtime ships inside the wheels.
- **blackjax >= 1.7.0.** This sampler calls BlackJAX internals (`chees_adaptation.base`, `dynamic_hmc.build_kernel`, `dynamic_hmc.halton_sequence`, `mcmc.dynamic_hmc.as_top_level_api`). BlackJAX 1.7 changed the ChEES `update()` signature (it now requires the diagonal inverse mass matrix) and replaced the kernel's `trajectory_length_adjusted` kwarg with `integration_steps_params`.
- **numpy is held below 2.5.** NumPy 2.5 removed `np.row_stack` and `np.trapz`; numba <= 0.66 registers overloads for them at import time, and numba >= 0.67 (which guards them) is outside PyTensor's `numba<=0.66` cap. Without this pin the resolver silently picks the pre-release `numba 0.63.0b1` and PyMC fails to import. See the comments in `pixi.toml`.

# Testing

```
pixi run python tests/test_inference_data.py
```

A few seconds, CPU only, no MCMC. It pins the `sample_stats` contract in `# Output` below.

The whole suite runs in one process (`pytest tests`, about 9.5 minutes on CPU). `tests/conftest.py` releases the compiled programs after each test module (a process may hold at most `vm.max_map_count`, 65,530 by default, memory mappings, and every compiled XLA program takes dozens) and switches the default compilation cache off for the session (see "Keeping the compiled programs between Python processes").

# Use
See the `examples/` folder for some examples. In short
```
# Define your PyMC model here
with pymc.Model() as model:
  # Define prior ...
  # Define likelihood
  # Call the sampler
  trace = sample_pf_chees_hmc(draws=2000, chains=128, random_seed=42)

#Postprocess the trace with arviz, as usual
```

Pass `random_seed`: if it is omitted the seed defaults to the current date, so the run is not reproducible.

**Chain count is the cheapest performance lever.** On an RTX A2000 the wall time per transition is flat from 32 to 256 chains (+2%), while ESS grows linearly with the number of chains, so more chains are nearly free until memory runs out. Measured on the 8 Schools example (2000 draws, 2,000 ChEES warmup, warm median of 5 seeds): 45,614 ESS/s at 32 chains and 134,898 ESS/s at 128. Sample histories are kept as `chains × draws × dim`, so raise `chains` with the model dimension in mind. **pf-r1 caveat:** the ESS/s values predate the RNG stage-key fix (superseded by 41,568 / 133,824 in `docs/ladder-results.md`, "Measurement caveats that matter") and the GPU arm ran on the shipped metric, which fails the R̂ ≤ 1.01 bar on GEV (LEDGER `c2993cf`; `a03dc0f`); the flat-wall fact is metric-independent throughput, the ESS/s and chain-scaling statements are re-measurement pending.

**How that compares to `pm.sample` on the same machine.** Best CPU configuration we could build on a 32-core box was 32 chains of NUTS: 37,494 ESS/s on 8 Schools and 6,733 on the GEV example. So the GPU is about 3.6x and 2.6x at 128 chains, and only ~1.2x at 32. For a small one-shot job CPU still finishes sooner (9.5 s versus 15.0 s), and its cold start is far lower — the GPU advantage is samples per second, not time-to-first-answer on tiny models. **pf-r1 caveat:** every GPU number in this comparison is on the shipped metric arm, which fails the project's own R̂ ≤ 1.01 bar on GEV (LEDGER `c2993cf`; orientation retraction `a03dc0f`); the direction of the device recommendation may move per model once the corrected arm lands, and the equal-draw walls are re-measurement pending.

**Compile once per model structure.** A call compiles its XLA programs the first time it meets a model structure, set of array shapes, chain count and sampler settings, and nothing after that in the same Python process. A new seed, new data values, new prior values, a longer or shorter warmup (`chees_kwargs={"draws": ...}`) or another number of draws, all at the same shapes, compile nothing: data, priors, the metric, the step size and the trajectory length are runtime arguments of the compiled programs, not constants inside them (`tests/test_compile_once.py` counts 0 XLA compiles and 0 PyTensor function compiles on a second call). Rebuilding the model object for each dataset is fine: a rebuilt model is recognised as the same structure, which costs 0.05-0.14 s per call on the four example models instead of 0.16-0.54 s (`tools/lift_cost.py`). A new array shape (more observations, more groups), another chain count or another model structure compiles once. Measured on CPU (JAX CPU backend, 32-core host), 1,024 chains, 10 draws, 25 warmup, with `pixi run python tools/warm_stages.py . MODEL 1024 10 25 --calls 3`: the first call in a process, then the mean of calls 2-3 with a new seed each:

| model | first call | calls 2-3 |
|---|---|---|
| 8 Schools | 13.5 s | 0.39 s |
| GEV | 17.8 s | 1.0 s |
| logistic 1000x25 | 11.9 s | 3.2 s |
| radon non-centered | 16.9 s | 4.6 s |

Most of the first call is XLA compilation, and most of the later calls is the sampler's own work; radon's later calls depend on the trajectory length ChEES settles on for that seed (3.6-5.5 s). GPU figures for the same shapes are pending the F2f measurement. Quote first-call or later-call timings, not both interchangeably.

**Optional metric upgrade:** `pathfinder_kwargs={"imm_source": "fisher_lowrank"}` builds the metric from Pathfinder draws and their gradients (BlackJAX's Fisher score, low-rank) instead of the draw covariance. It is not a default, because it is not uniformly better — measured at 32 GPU chains, 2,000 draws: **1.22×** on 8 Schools, **1.96×** on logistic 1000x25, and **0.19×** on radon non-centered, with R-hat unchanged on the healthy models. The tell is `num_integration_steps` in `sample_stats`: if the realised trajectory length jumps from tens of steps to hundreds, the metric is wrong for the model and you are paying for it. `lbfgs_lowrank` reuses the L-BFGS factors Pathfinder already fitted at no extra cost. **pf-r1 caveat:** that A/B compared arms that never shared a metric orientation — `mc` feeds M⁻¹ while the low-rank arms feed ≈Σ into the same kernel slot (LEDGER `9f892b2`) — from a baseline on the orientation the 2×2 fixture shows failing (LEDGER `a03dc0f`, confirmed on GEV at `c2993cf`); the ratios and the step-jump tell are re-measurement pending on one explicit convention. The option itself stands; the numbers do not.

**Turn the progress bars off in scripts:** `sample_pf_chees_hmc(..., progressbar=False)` silences the bars, which is what you want in scripts and logs. The bars are drawn by the host between chunks of the compiled warmup and sampling loops, not from inside them, so on or off the compiled programs are the same and the flag does not change compile cost (`tests/test_compile_once.py`, `tests/test_progressbar.py`).

**Warmup length: 2,000 is enough for these models; 10,000 was not.** Measured on both examples (2000 draws, 32 chains, warm median of 5 seeds, RTX A2000):

| model | warmup | wall | ESS/s | max R-hat | divs /1k draws | tail-ESS |
|---|---|---|---|---|---|---|
| 8 Schools | 1,000 | 10.4 s | 48,014 | 1.0011 | 0.008 | 30,227 |
| 8 Schools | 2,000 | 12.1 s | 39,624 | 1.0012 | 0.024 | 31,743 |
| 8 Schools | 10,000 | 22.4 s | 21,082 | 1.0014 | 0.008 | 32,436 |
| GEV | 1,000 | 12.1 s | 1,469 | **1.0680** | 0.039 | **583** |
| GEV | 2,000 | 16.2 s | 6,486 | 1.0052 | 0.125 | 8,163 |
| GEV | 10,000 | 50.9 s | 3,672 | 1.0009 | 0.438 | 24,048 |

**pf-r1 caveat on this table and the heading's conclusion:** the quality columns (R̂, tail-ESS) and the walls were measured on the shipped metric arm, which fails the R̂ ≤ 1.01 bar on GEV itself (LEDGER `c2993cf`; orientation retraction `a03dc0f`), and the GEV rows additionally carry a coordinate permutation. The numbers stand as measured; the heading's "2,000 is enough; 10,000 was not" and the floor guidance below are re-measurement pending, with the warmup floor to be re-derived by the bias gate (b²_max < 0.01 against the CPU NUTS reference) rather than split-R̂. The 2,000-versus-10,000 posterior comparison (0.49 / 0.70 sigma, sds within 0.6%) is a same-metric A/B and stands.

Do not go as low as 1,000: on GEV the frozen trajectory length collapsed to 6.2 leapfrog steps, R-hat went to 1.068 and tail-ESS to 583 — ChEES had not learned the trajectory length (pf-r1: CONFOUNDED with the mechanism now stated — that GEV cell ran on the failing metric, LEDGER `c2993cf`; `a03dc0f`, and the frozen-L collapse is adapter-dependent, plan §0.3; re-measurement pending). At 2,000 the posterior is indistinguishable from 10,000: worst per-variable difference 0.49 sigma (8 Schools) and 0.70 sigma (GEV), posterior sds within 0.6%. More chains also means less needed warmup, since ChEES pools statistics across all chains — **retracted (pf-r1)**: the plan's superseded-beliefs list (A2) removes this claim, and the warmup × chains grid it leaned on measures the short-chain split-R̂ artifact (plan §2b) on the failing metric arm (LEDGER `c2993cf`; `a03dc0f`). The replacement, pending, is the bias-based W: plan §2e reaches b²_max < 0.01 at W=100 on 8 Schools through the public API at 1024 × 10 chains.

Stage-specific options are forwarded as dicts: `pathfinder_kwargs`, `step_size_tune_kwargs`, `chees_kwargs`, `dynamic_hmc_kwargs`.

For use with GPU sampling, see also the `examples/` folder.

## Keeping the compiled programs between Python processes (on by default)

Compile-once holds inside one Python process. The first call in a new process compiles every program again, 10.6 s of a 13.5 s first call for 8 Schools at 1,024 chains on CPU. JAX can keep the compiled programs in a directory and read them back in the next process, and since F2g `sample_pf_chees_hmc` does so **by default**, the way PyTensor keeps its compiled C code in a per-user directory (`~/.pytensor`):

- **Where:** `${XDG_CACHE_HOME:-~/.cache}/pfcheeshmc/jax` (`pfcheeshmc.default_compilation_cache_dir()` returns it; a relative or empty `XDG_CACHE_HOME` is ignored, as the XDG convention says). It is created owner-only (mode 0700). If it exists already and belongs to someone else or can be written by group or others, or cannot be created, the cache stays off and a `RuntimeWarning` says why; a failure to cache never fails a sampler call.
- **When:** at the start of each `sample_pf_chees_hmc` call, once per process. **Importing `pfcheeshmc` changes no JAX setting** (`tests/test_compilation_cache.py` diffs `jax.config`), and the class API (`PfCheesHMC(...).sample`) does not switch it on.
- **The global effect:** JAX's compilation cache is process-wide. The call sets `jax_compilation_cache_dir` to the directory above and the two thresholds below, so **after the first sampler call every other JAX program compiled in the process, yours or another library's, is cached there too** (the thresholds make that include the very small ones), and is read back by the next process that uses the directory. Programs compiled before the first sampler call are not cached (they have compiled already); JAX takes the directory at the next compile, whatever was compiled before (tested).
- **Your own choice always wins.** If a JAX cache directory is set already (`JAX_COMPILATION_CACHE_DIR`, `jax.config.update("jax_compilation_cache_dir", ...)` or `jax.experimental.compilation_cache.set_cache_dir`), or `jax_enable_compilation_cache` is off, **no JAX setting is touched**: your directory, thresholds and size limit stay as they are, and our directory is not created. The default moves a threshold only if it is still at JAX's own default, and never sets `jax_compilation_cache_max_size`. (With JAX's default thresholds only 3 of 134 programs of a first call are kept, so a directory set without the thresholds below helps little; see the next paragraph.)
- **Turn it off:** set the environment variable `PFCHEESHMC_COMPILATION_CACHE=0` (`false`, `no` and `off` also count, in any case) before the sampler call: in the shell, or in a script with `os.environ["PFCHEESHMC_COMPILATION_CACHE"] = "0"` anywhere before the first call. It is read at the call, not at import. Nothing is created and no JAX setting is touched. `tests/conftest.py` sets it for the whole test session, so the tests are hermetic and write nothing into your home directory.
- **Choose the directory yourself:** `pfcheeshmc.enable_compilation_cache("/some/writable/dir")`, at any time, whatever the variable says. It replaces the default if a call has switched that on already. It sets three JAX settings and nothing else: `jax_compilation_cache_dir`, `jax_persistent_cache_min_compile_time_secs = 0` and `jax_persistent_cache_min_entry_size_bytes = -1` (no minimum size, and no platform override of it). JAX's own environment variables do exactly the same without code: `JAX_COMPILATION_CACHE_DIR=/some/writable/dir JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1`. The directory alone is not enough: JAX writes a program only if it took at least 1 s to compile, and nearly all of this sampler's programs take less (8 Schools at 1,024 chains: 3 of 134 programs reach 1 s, so the directory alone gives 8.2 s for the warm first call against 3.4 s with all three settings, `JAX_COMPILATION_CACHE_DIR` alone on `tools/warm_stages.py . 8schools 1024 10 25 --calls 1`).

Measured on CPU, one first call in a fresh process, 1,024 chains, 10 draws, 25 warmup (`tools/warm_stages.py . MODEL 1024 10 25 --calls 1 --cache-dir DIR`, run three times: without `--cache-dir`, then twice with a new empty `DIR`; `warm_stages` turns the default cache off so that its first call is cold, see its docstring):

| model | no cache | cache cold (writes it) | cache warm | cache size |
|---|---|---|---|---|
| 8 Schools | 13.5 s | 14.0 s | 3.4 s | 1.5 MB, 133 files |
| radon non-centered | 16.9 s | 17.6 s | 9.5 s | 1.8 MB, 138 files |

XLA compile time in the warm run is 0.6 s (8 Schools) and 1.0 s (radon) instead of 10.6 s and 8.6 s. Radon's warm call is dominated by running the warmup and the sampler (4.0 s and 1.7 s for this seed), not by compiling. @@DEFAULT_MEASURED@@ GPU figures are pending the F2f measurement; no GPU number is claimed here.

What it covers, and what it does not:

- **Covered:** the XLA compilation of every program of a sampler call (the warmup and sampling loops, the Pathfinder fit, the step-size search, the log-density, the transform). The cached programs hold no data and no prior values, so a **new dataset or new priors at the same shapes use the cache of the old ones**; a new shape, chain count or model structure writes new entries. A fresh-process test pins this (`tests/test_compilation_cache.py`): the second process rebuilds the model with new data and priors, and every one of its compile requests but one (see *Order* below) is a cache hit.
- **Not covered:** JAX still traces each function and lowers it to StableHLO in every new process (about 2 s for 8 Schools), and PyMC and PyTensor still build their graphs (0.05-0.14 s per call). The warm 8 Schools call above is 3.4 s for these reasons and for running the sampler.
- **Size:** the default directory has **no size limit**. It grows by about 1.5-4.6 MB per distinct model structure, set of array shapes and chain count (133-138 files each on the CPU examples; 2.9-4.6 MB per model on the GPU, F2f), not per dataset. JAX can bound a cache (`jax_compilation_cache_max_size`, bytes; least recently used entries go first, and it evicts correctly on JAX 0.11.2), but it scans every file in the directory at every write: 33 microseconds per file already there, measured, so writing the 135 programs of one new model costs 4.5 s per 1,000 files in the directory (22 s at 5,000, which is about 37 models) on top of the compile. A limit large enough to be useful would slow the first call of every new model, so none is set for you; set JAX's if you want one (we leave it alone). To see the size and to clear: `du -sh "$(python -c 'import pfcheeshmc; print(pfcheeshmc.default_compilation_cache_dir())')"`, then delete the directory. On a GPU, JAX also keeps its XLA autotune results in a subdirectory of the same directory.
- **Safe to delete:** the directory only holds copies. Delete it at any time (or after upgrading JAX, a GPU driver or the device count: entries are keyed by the lowered program, the jaxlib version, the backend, the devices and the XLA flags, so a change misses and compiles again rather than misbehaving). If a process was killed while writing an entry, JAX warns `Error reading persistent compilation cache entry` on every later use of that entry and compiles instead; delete the directory (or that file) then. Several processes can share the directory; an entry that fails to read is compiled instead, with that warning.
- **Trust:** a JAX compilation cache is trusted code: whoever can write to the directory can make your process run code. The default directory is created owner-only for that reason, and is not used if others can write to it. Do not point `enable_compilation_cache` or `JAX_COMPILATION_CACHE_DIR` at a directory shared with users you do not trust.
- **Order:** with the default, nothing to arrange. A program compiled while the packages are imported (a 0.04 s `jit(log)`) comes before any call can set the directory, so it is not cached; JAX's environment variables, set before the process starts, cache it too.
- **Measuring:** a "cold" first call that reads this cache is a warm one. `tools/warm_stages.py`, `tools/r1_bias.py` and `tools/bench_ladder.py` set `PFCHEESHMC_COMPILATION_CACHE=0` unless it is in the environment already, and `tools/warm_stages.py` prints which cache it ran with; if `JAX_COMPILATION_CACHE_DIR` is in your environment, JAX's cache is used by everything, including them.

# Output

`sample_pf_chees_hmc` returns an `arviz.InferenceData` with:

- `posterior` in **problem space**, including `pm.Deterministic` variables;
- `sample_stats` with exactly `lp`, `diverging`, `acceptance_rate`, `is_accepted`, `energy`, `num_integration_steps`. `lp` is the logdensity of the *sampled* state, matching what PyMC's own JAX samplers emit.

# Common Issues
1. If the sampler is not converging, try increasing the number of ChEES draws using the argument
```
sample_pf_chees_hmc(chees_kwargs={"draws": 2_000})
```
2. On a shared GPU, JAX pre-allocates most of the device memory at import. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (the pixi env already does) so other users' jobs are not starved.
3. Divergence counts can differ by one between identical runs at the same seed. XLA does not guarantee bitwise-identical reductions, so a near-threshold divergence may flip; this is not a sampler change.


# References
[1] Matthew Hoffman, Alexey Radul, Pavel Sountsov "An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo", in Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, PMLR 130:3907-3915, 2021. Available: https://proceedings.mlr.press/v130/hoffman21a.html
