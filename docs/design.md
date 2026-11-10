# Design notes for maintainers

What the code does today and why, in the places where a change is easy to get wrong without any test failing loudly.
Each paragraph names the function that carries the rule. Nothing here is history.

The pipeline, in `pfcheeshmc/sampler.py:PfCheesHMC.sample`: re-read the model's values (`_refresh_values`), Pathfinder and
the metric (`_run_pathfinder`), the step-size search (`_run_step_size_tune`), ChEES warmup (`_run_chees`), the frozen
kernel's draws (`_run_dynamic_hmc`), and the transform back to problem space (`_transform_in_draw_chunks`).

## 1. Rule C: what is lifted out of the log-density, and when

`pfcheeshmc/lifted.py:lift_graph` turns a PyMC graph (`lift_logp` for `model.logp()`, `lift_initial_point` for the start)
into `fn(inputs, values)`, a pure JAX function whose second argument holds the model's data and prior values. PyMC's own
`get_jaxified_graph` turns every shared variable into a constant, and PyTensor's JAX rewrite then folds and fuses constants,
so a data or prior value ends up inside the XLA program and a new value recompiles everything.

**What is lifted** (`_candidates`, `_build`):
- the constants that are direct inputs of a model RV node (the distribution parameters), whatever their dtype;
- observed-data constants;
- every non-scalar constant (data arrays, index arrays, constant matrices);
- every non-RNG shared variable (`pm.Data`).

**Only value uses are lifted.** `_first_bad_client` looks through `DimShuffle` at every client of a candidate and keeps it
static unless all of them are in `_VALUE_OPS` (`Elemwise`, `CAReduce`, `Dot`, `Blockwise`, advanced indexing, `Output`,
`Shape`, `Shape_i`). A shape, a size, an axis or a basic-index position has to be a compile-time constant for JAX. The kept
ones are reported in `LiftedGraph.kept_static`.

**Before PyTensor's JAX rewrite.** `_build` replaces the candidates by symbolic inputs first and only then runs
`mode.JAX.optimizer.rewrite`. After the rewrite the prior values have been folded into `Composite` ops and cannot be
recovered.

**Formula constants stay static.** The 0.5, `log(2 pi)` and 1 that PyMC's logp derivations contain are not candidates.
Lifting them would hide them from PyTensor's value-dependent rewrites (`x**2 -> sqr`, `log(1 + x) -> log1p`), which keep the
density numerically stable; `tests/test_lifted.py` checks that `log1p` and `sqr` survive.

**Widening, so that the program does not depend on the value.** PyTensor autocasts a Python scalar to the smallest exact
dtype (0 and 10 become int8, 1000 int16, 2.5 float32, 2.7 float64). Without a fix, the dtype of a lifted prior, and so the
program, would depend on its value. `_widen_model_inputs` (rule `model_inputs`) first widens scalar prior constants: a float
narrower than floatX to floatX, an integer parameter of a float RV to floatX, an integer parameter of an integer RV to int64,
never narrowing. This also repairs an int8 overflow in PyMC 5.28.5 (`pm.Uniform("u", -100, 100)` has a NaN logp; see
`docs/upstream-drafts.md`).

**Fallback chain** (`RULES`, `lift_graph`): `model_inputs`, `model_inputs_native_dtype` (no widening; used when widening
fails, for example `pm.LKJCholeskyCov`), `arrays_and_shared` (only arrays and shared variables), `none` (PyMC's behaviour,
every constant compiled in). A failing rule raises a `RuntimeWarning` naming the rule and the exception; the last rule's
failure propagates. A graph that contains a shared RNG is refused (`_build`, same refusal as `pymc.sampling.jax`).

**A shared variable kept static** is baked in as its value at build time. `LiftedGraph.is_stale` compares it with the value
now, and `PfCheesHMC._refresh_values` lifts again if it changed, or if a lifted shared variable changed shape or dtype.

## 2. Compile once

A jitted function is cached by the identity of the Python function, its static arguments and the shapes of its dynamic
ones. Two habits defeat that on every call: a jitted function built inside a method, and per-run values closed over
(JAX compiles them in as constants). The layout that avoids both:

- **One program per model structure.** `lift_graph` keys a function by the sha256 of its lowered StableHLO at the abstract
  shapes (`_key`). The text holds structure, static constants, shapes and dtypes, not the lifted values. `_REGISTRY` maps the
  key to the first function built, so structurally equal models share one function object and with it JAX's in-memory caches.
  `_key` lowers with `keep_unused=True`: an argument a function does not read is still part of its calling convention.
- **Per-run values are arguments.** `pfcheeshmc/compiled.py` defines every jitted site once, at module level. `fn` and the
  value-variable `names` are static (`_STATIC`); the lifted `values`, RNG keys, positions, the metric, the step size and
  the trajectory length are dynamic. `LogDensity` bundles `fn`, `names` and `values`. A second call with a new seed, new data
  or new prior values at the same shapes compiles nothing (`tests/test_compile_once.py` counts XLA compiles and PyTensor
  functions).
- **Warmup length is a runtime trip count.** `compiled.warmup_chunk` is one program of `WARMUP_CHUNK` (50) steps with a
  dynamic active count `n`; `utils.py:chees_warmup_with_progress` calls it `ceil(W / 50)` times through
  `compiled.run_chunks`, the last call with a smaller `n`. `_chunk_scan` skips step `j >= n` with a scalar `lax.cond`
  outside the vmap, so a skipped step evaluates no model. The per-step key is `fold_in(stage_key, global_step_index)`,
  independent of W and of the chunk size. `HALTON_BITS` is a fixed 30 so that W is not a constant of the program
  (it used to be `ceil(log2(W + 1000))`).
- **The draws are the same kind of program.** `compiled.sampling_chunk` has `SAMPLE_CHUNK` (50) steps and a dynamic count,
  driven by `sampler.py:_run_dynamic_hmc`; `compiled.transform_draws` runs in chunks of `TRANSFORM_CHUNK` (50) draws
  with the last chunk padded (`_transform_in_draw_chunks`). A different D is a different number of calls of the same
  programs, not a new program. The chunk sizes are module constants that tests set small to exercise the multi-chunk path.
- **The host overlaps chunks.** `run_chunks` dispatches chunk k+1 before converting chunk k's outputs, so the transfer and the
  progress bar overlap the next chunk's compute. A progress bar is drawn between chunks, never inside a program, so it does
  not change the programs.

New array shapes, another chain count, another model structure or another sampler setting that is static (the entries of
`_ADAPTER`, `chunk`) compile once more.

## 3. The lift memo and the disk cache

**The lift memo** (`lifted.py:_MEMO`, `_recall`, `_signature`). Rebuilding a model around new data gives the same graph up to
the lifted values. `_build` summarises the lifted graph, before PyTensor's JAX rewrite, as `_signature` (ops by type and
props, wiring, types, static constants by value) and files the rewritten and funcified function under it with the context
(`_context`: whether `jax_enable_x64` is on). A candidate with an equal signature is used only if PyTensor's
`equal_computations` agrees, so a hash collision costs a rebuild and never a wrong function. The keys at different argument
shapes are kept per memo (`_spec_ids`). Values, sources, kept-static entries and static snapshots are always read from the
current model. `_REGISTRY` and `_MEMO` are plain module dictionaries, unbounded; they grow with the number of distinct
model structures in a process.

**The disk cache** (`pfcheeshmc/compilation_cache.py`). `sample_pf_chees_hmc` calls `ensure_default_compilation_cache` at its
start (`sampler.py:sample_pf_chees_hmc`). It sets `jax_compilation_cache_dir` to `default_compilation_cache_dir()`
(`$XDG_CACHE_HOME/pfcheeshmc/jax`, else `~/.cache/pfcheeshmc/jax`) and, only if they are still at JAX's defaults, the two
thresholds to 0 s and -1 bytes (otherwise only the few programs that take over 1 s are written). Rules to keep:
- it never overrides a user's choice: a JAX cache directory already set, or `jax_enable_compilation_cache` off, means no JAX
  setting is touched;
- `PFCHEESHMC_COMPILATION_CACHE=0` (`false`, `no`, `off`) opts out; it is read at the call, not at import, and importing
  the package changes no JAX setting;
- the directory is created mode 0700, and is refused with a `RuntimeWarning` if it belongs to someone else or is group- or
  world-writable (a compilation cache is trusted code); a failure to cache never fails a call;
- no size limit is set: with a limit JAX scans the whole directory at every write;
- `enable_compilation_cache(path)` is the explicit form and replaces a default already in use.

The cached programs hold no data and no prior values, so a new dataset at the same shapes reads the old one's entries.

## 4. Start repair

A start must have a finite log-density and a finite gradient. GEV Port Pirie at `model.initial_point()` (`xi = 0`) has a finite
logp and a NaN gradient: `GenExtreme.logp` evaluates both branches of its `switch`, and the unselected one is `inf * 0`.

- `PfCheesHMC._usable_mask` (via `compiled.usable_mask`) is one jitted, vmapped check of logp and gradient for all chains.
- `_ensure_finite_inits` reads the mask once on the host, returns the arrays untouched when every row passes, and repairs
  only the failing rows, row c under `split(key, chains)[c]`, on a host copy.
- `_find_finite_start` repairs one row in order: a tiny relative jitter (1e-8), up to 16 halvings toward the origin of the
  unconstrained space, then up to 64 random restarts around the origin with a radius growing by 1.5 from 0.05. It decides
  "usable" with `_usable_mask` on a one-row stack, the same compiled program, so no log-density is evaluated op by op. If that
  check cannot be jitted it falls back to an eager one. If a row cannot be repaired, `_ensure_finite_inits` tries again from
  the origin; a second failure raises `RuntimeError`.

## 5. The metric convention

The array that blackjax's kernel calls `inverse_mass_matrix` is the matrix `A` in the kinetic energy `1/2 p.(A p)`
(`blackjax/mcmc/metrics.py`), so the momentum has covariance `A^-1` and the leapfrog step is `q'' = -A grad U`. For a target
with Hessian `H` the squared frequencies are `eig(A H)`. With `A = H^-1 = Cov(q)` every coordinate oscillates at unit
frequency and one dimensionless step size serves all of them. Therefore **every metric source hands the kernel a posterior
covariance, as it is, never its inverse** (`_run_pathfinder`, the comment at "Orientation"). The sources are `mc` (the
empirical covariance of the Pathfinder pool plus a ridge), `pf_closed`, `lbfgs_lowrank` and `fisher_lowrank`; the low-rank
ones are expanded to a dense covariance `D (I + U (lam - 1) U^T) D` before the same slot.

**Coordinate order.** Two orders exist in the pipeline:
- Pathfinder is fitted on `tuple(position[n] for n in self.rvs)`, so its draws, covariance and L-BFGS factors are in
  `model.value_vars` order;
- the HMC state is a dict, `ravel_pytree` sorts dict keys, so the kernel reads the metric in sorted-name order.

When they differ (GEV: `[mu, sigma_log__, xi_interval__]` against `[mu, xi_interval__, sigma_log__]`) a metric built in tuple
order is applied to the wrong coordinates and nothing raises. `_kernel_permutation` computes the permutation (it accepts only
Pathfinder's tuple and raises on a dict), `_metric_to_kernel_order` applies it to the expanded dense matrix, and
`_run_pathfinder` does that at one choke point, after the covariance is final. Everything downstream (the step-size search,
the ChEES whitening diagonal, the kernel) ravels the dict state. Anything that builds or reads a metric must flatten through
`_kernel_flat`. `tests/test_metric_orientation.py` pins the convention and the permutation.

## 6. The `erfcx` registration and its guard band

PyTensor sends `Erfcx` (used by the truncated normal and by other logp's) to TensorFlow Probability in its JAX backend, which
has no release for this JAX. `lifted.py` registers JAX's own `jax.scipy.special.erfcx` for it (`jax_funcify_Erfcx`, on import;
it also applies inside `Composite`, whose inner graph is funcified through the same dispatch).

JAX's `erfcx` returns exactly 0 in a band just below the point where it switches to its asymptotic branch, because
`exp(x**2) * erfc(x)` is evaluated with `erfc(x)` already flushed to zero (`docs/upstream-drafts.md`). The band is
`[erfcinv(finfo.tiny), sqrt(log(finfo.max))]`: about [26.543, 26.642] in float64 and [9.1946, 9.4193] in float32. Its
gradient is wrong there too.

`_erfcx_guarded` replaces the band with the asymptotic series (`_erfcx_series`, nine terms; three leave a relative error of
5e-9). `_band(dtype)` derives the edges from the dtype and widens each by 0.1%; float16 and bfloat16 share the float32 band
because JAX upcasts them. The `jnp.where` pattern is safe: JAX's `erfcx` is fed 0 inside the band and the series is fed the
band's centre outside it, so neither branch can produce a NaN value or gradient for the other. JAX 0.12.0.1 is right in the band (value and gradient; 0.11.2 is not), so the guard only matters for the older JAX
versions `pyproject.toml` still allows; on a fixed JAX the series and JAX's function agree there. Remove the guard and
register JAX's function directly once the floor is `jax>=0.12` (`tests/test_lifted.py` fails without the guard on older JAX).

## 7. float32

`PYTENSOR_FLAGS=floatX=float32` before PyTensor is imported makes PyTensor's JAX backend set `jax_enable_x64 = False`.
`sampler.py:_warn_if_x64_contradicts_floatx` warns when floatX is narrower than float64 and x64 is on, because the
log-density would then be float64 at a float32 position. `lifted.py:_quiet_truncation` drops JAX's "Explicitly requested dtype
float64" warning while x64 is off, since its advice would undo the float32 run.
