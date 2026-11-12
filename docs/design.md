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

## 8. Tried and not adopted (0.4)

- **SNAPER in place of ChEES** (Sountsov & Hoffman 2021): the trajectory-length gradient of the ESJD of the squared
  projection on the leading principal component (of the chains, whitened by the dense metric), per unit length, in
  BlackJAX's adapter. On an upper-bounded GEV regression (1,024 chains, seeds 1-5) the tuned step count L varied as
  much between seeds as under ChEES (4.2x in both on one case), and the ESS per gradient changed by +7% to +13% on the
  GEV cases and -8% to -12% on two benchmark models, inside the seed-to-seed range (about 3x). Not added. (Corrected
  after 0.4's first draft, which said this range is not explained by the tuned length: that compared step counts, not
  the trajectory length T = L x step size. The step size and T do explain it, and SNAPER does not change why T varies;
  section 9.)
- **An R-hat implied by the ESS.** sqrt((n - 1)/n + m/ESS) matches the observed R-hat whether chains mix slowly or
  disagree, because ArviZ's ESS uses the between-chain variance as R-hat does; it carries no information beyond R-hat. A
  within-chain-only autocorrelation estimate is biased low on short chains (centring each chain on its own mean).
  The docs give the check that separates the cases instead (R-hat at half the draws; [R-hat](rhat-many-chains.md)).
- **Leapfrog unrolling under delayed rejection before the ghost loop**: unrolling multiplied each of the 15 integration
  loops of a 4-stage program (compile 98 s -> 286 s at two steps per pass on a GPU); with the ghost loop (one loop per
  stage) the cost is per stage.
- **MALT in the sampling phase** (Riou-Durand & Vogrinc; Algorithm 2: partial momentum refreshes between leapfrog steps,
  one accept per trajectory), at ChEES's tuned length, from the same warmed-up state as HMC: friction 0.5 gave 1.17-1.22x
  HMC's ESS per gradient on one GEV case and 0.75-0.87x on another, with twice the divergences; friction 2 gave
  0.4-0.7x. Not added; it also cannot be combined with delayed rejection as built (a deterministic proposal is assumed).
  Tuning the length for MALT itself (adaptive MALT) was not tried. Both prototypes: branch `p7-prototypes`.

## 9. The ChEES warmup: where run-to-run spread comes from

Measured for 0.4 on an upper-bounded GEV regression: 1,024 chains, warmup 200 with metric updates at 0.25, 0.5 and 0.75,
target acceptance 0.8, delayed rejection 8 x 4, float32, one GPU. Every run was its own process, with the disk cache off
and XLA's autotuning on, so each run is a fresh draw of the run-to-run variation a new user sees
([GPU notes](gpu-notes.md#runs-at-the-same-seed-can-differ-on-the-gpu)). The user-facing summary and the measured effect
of the two settings are in [Choosing settings](choosing-settings.md#run-to-run-spread-a-step-size-floor-and-a-fixed-trajectory-length-opt-in-experimental).

**What carries the spread.** On a common grid of step size and T, sampled from each run's warmed-up state, the two
settings explained 87-100% of the variance of log ESS per gradient between runs; the warmed-up state changed it by at
most 1.05-1.12x, and sampling noise by about 1%.

**Step size.** `chees_adaptation.base` (BlackJAX 1.7.1) feeds dual averaging the harmonic mean, over the non-divergent
chains, of the acceptance probabilities. A chain whose proposal is rejected with probability near 1 without crossing the
divergence threshold (acceptances of 1e-12 to 1e-44 were seen) puts that mean near 0. Dual averaging then cuts the step
size about tenfold and climbs back: a sawtooth, and the frozen step size is its phase when the warmup ends (2.3x between
runs). `acceptance_floor` replaces the statistic by the harmonic mean of max(p, f). `compiled._adapter` builds it from
two calls of BlackJAX's update, so BlackJAX's code is not copied: the step-size fields come from the call fed max(p, f),
the trajectory fields from the call fed p (p also weights ChEES's gradient), and T is clipped at the new step size as
BlackJAX clips it. `tests/test_warmup_options.py` pins this against BlackJAX bit for bit. Floors tried: 0.005, 0.01 and
0.03. 0.01 was the smallest that halved the step size's spread where the sawtooth occurred and kept the stuck chains
within noise; 0.03 left 8 chains stuck where the defaults left none.

**Trajectory length.** BlackJAX starts T at the initial step size e0, as Algorithm 1 of the ChEES paper does. e0 is the
geometric median of 32 doubling/halving searches (Hoffman & Gelman 2014, Algorithm 4), so it moves between runs by
factors of 2. Dual averaging's first step sets the step size to 10 e0 exp(-(target - h1) / (0.05 x 11)), h1 the first
step's statistic (2.3-2.5 e0 when h1 is near 0, at target 0.8), and T is clipped up to it. BlackJAX's update then applies
Adam's step to the moving average of log T (weight step^-0.5), which is also the T the kernel runs, so each step moves
log T by step^-0.5 times Adam's update and T stays near its start. With e0 forced to 0.16, 0.64 and 2.56 the slope of log
frozen T on log e0 was 0.44-0.52, and runs at the same e0 gave nearly the same T. The ChEES paper's Adam settings
(0.025, beta1 0, beta2 0.95) did not change this. Algorithm 1's update (Adam on log T itself) kept the dependence on one
configuration and on another drove T to 8-19 and halved ESS per gradient. `trajectory_length` holds T and its moving
average at the given value instead.

**ChEES's criterion.** Sampled from warmed-up states at 8 values of T (about a quarter to four times the best), the ESS
per gradient at each criterion's maximiser, relative to the best T, over three configurations:

| criterion | ESS per gradient at its maximiser / best |
|---|---|
| ChEES (BlackJAX's) | 0.26-0.36 (its maximiser above the largest T tried) |
| ChEES per gradient | 0.35-0.49 |
| ChEES / sqrt(L) (Wang et al. 2013) | 0.66-0.99 |
| SNAPER (Sountsov & Hoffman) | 0.35-0.70 |
| slowest coordinate: (1 - rho)/(1 + rho) / L | 0.94-0.98 |
| the same over the chains' principal directions | 0.93-0.98 |

rho is the cross-chain correlation of a whitened coordinate before and after one transition, so (1 - rho)/(1 + rho) is
the effective draws per transition of a lag-1 autoregression.

**Adapting T by the slowest-coordinate criterion: tried, not adopted.**

- A probe after the warmup (5 values of T around the warmup's, 8 draws each, a quadratic fit in log T): its T varied
  between runs by a standard deviation of log 0.11-0.24, at 4-15% more gradients. From a start far below the best it
  cannot reach it.
- A finite-difference search inside the warmup (three chain groups at T e^-d, T and T e^d; Kiefer-Wolfowitz steps on the
  log of the criterion summed over blocks of 16 steps): it froze T low in single runs (0.72 in one run, 2.2-2.8 in three,
  against 4.1-4.2 in the rest). A first version took the log of each step's criterion and drove T up whatever T was: the
  chains of a step share the Halton jitter, and at a small jitter every group runs a short trajectory, where the criterion
  rises with T.
- One fixed T for all three configurations (the geometric mean of their ESS-per-gradient optima) beat both, and is the
  `trajectory_length` setting.

The two settings reproduce the prototypes these measurements used: one configuration, run both ways with XLA's
autotuning off, gave the same frozen step size and trajectory length, the same step-size and trajectory-length traces,
gradient counts and posterior summaries, bit for bit. The warmup trace's harmonic acceptance differs by design: with
`acceptance_floor` it records the floored statistic, which is what the step size is adapted on.
