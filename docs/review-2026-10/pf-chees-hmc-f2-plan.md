# F2 — compile once; data and priors as runtime inputs (plan)

Status:
- Scoped 2026-10-21.
- Owner's go 2026-10-22.
- F2-2 revised 2026-10-22 after probe 6; see "Decisions" below.

Branch: `f2` from `main`. Never push. Merge to `main` only on the owner's say.

## Goal
A sampler call compiles nothing new when only these change:
- the seed;
- data values;
- prior values;
- warmup length W;
- the number of draws D.

It compiles once per (model structure, array shapes, chain count, sampler settings). Two other
per-call costs that hit every model are removed as well. The improvements must be universal, not
tuned to the benchmark models.

## Evidence (probes in `results/f2/probes/`)
- **A warm call recompiles 8 XLA programs.** 8schools at C=1,024, host loaded: XLA compile is
  15.9–17.4 s of a 19.9–21.9 s call. The programs are 3 loops, the ε search, Pathfinder's L-BFGS and
  3 helpers.
- **Cause 1:** jitted closures are rebuilt on every call.
- **Cause 2:** data, metric, step size and trajectory length are compile-time constants. PyMC turns
  data into constants in `pymc/sampling/jax.py:116`.
- **PyTensor's JAX rewrite folds and fuses constants.** Prior values and formula constants end up
  inside `Composite` scalar ops, so lifting constants **after** the rewrite misses them
  (`ops_probe.py`, `hoist_probe5.py`). Lifting has to happen **before** the rewrite.
- **Lifting every constant before the rewrite (rule A, `hoist_probe4.py`)** blocks value-dependent
  rewrites. All four models lose `x**2 → sqr` and fall back to a general `pow` with a traced exponent.
  `log(1+x) → log1p` and similar stability rewrites would be lost the same way.
- **Lifting only model inputs before the rewrite (rule C, `hoist_probe6.py`, CPU, 4 models):**
  - logp is equal to today's;
  - rebuilding the logp with every lifted value perturbed (floats ×1.01+0.003, integers +1) gives
    **byte-identical** lowered HLO, so no value leaks into the program;
  - `Sqr`, `Log1p` and `Softplus` survive;
  - logit's HLO shrinks from 421,654 to 7,544 characters;
  - `vmap(value_and_grad)` at C=1,024 on CPU is equal or faster than today: 0.161→0.156,
    2.90→2.69, 14.5→13.1 and 17.9→16.9 ms;
  - gev needs a JAX `erfcx` (F2-1), because its prior normaliser becomes parameter-dependent.
- **BlackJAX:** the dynamic-HMC kernel already takes step size, metric and trajectory-length params at
  call time (`blackjax/mcmc/dynamic_hmc.py:101-111`).

## Work packages
- Each WP is one fast-tier task. Tasks run sequentially, one at a time.
- The lead reviews each task against its checklist.
- The suite runs after each merge into `f2`.
- Task rules are in `results/f2/TASK-RULES.md`.

### F2a — log-density with model inputs as runtime inputs
- **New module `pfcheeshmc/lifted.py`.** One function builds a lifted JAX function for any
  (inputs, outputs) of a PyMC model. It is used for:
  - the logp, at `sampler.py:185`;
  - the value→RV transform, at `sampler.py:1362`.
- **What is lifted: rule C, applied BEFORE PyTensor's JAX rewrite (F2-2).**
- **Erfcx:** register JAX's own implementation with a guard band (F2-1).
- **Fallback chain:** rule C → arrays and shared variables only → today's constants. A fallback is
  taken when jaxify or the abstract lowering raises, with a warning that names the exception.
- **Cache key:** sha256 of the lowered StableHLO text of the lifted function at abstract
  (`ShapeDtypeStruct`) inputs.
  - Lifted programs contain no data or prior values, so the text is exactly the structure plus the
    static constants plus the input shapes and dtypes. It is also what JAX's own caches key on.
  - No custom graph serialisation is needed.
- **API returned to the sampler:**
  - a pure function `fn(value_vars_tuple, lifted_values_tuple)`;
  - the current lifted values, as a tuple of `jnp` arrays;
  - the key;
  - diagnostics: counts lifted and kept static, and the client op of each kept constant.
- **A module-level registry maps key → the first `fn` built for it.** Later models with the same key
  reuse that function object. Equal HLO means an equal computation, so this is exact, and it is what
  lets JAX's in-memory caches hit in F2b.
- **Sampler wiring (minimal; behaviour unchanged):** `self._ld_fn(seq)` closes over the lifted values,
  as today. F2b threads the values through as arguments.
- **Tests:**
  - logp equality (≤ 1e-12 relative) on the 4 models, and on the extra models below;
  - lowered HLO identical when the model is rebuilt with new data and prior values at the same
    shapes (fresh `pm.Model`s, not just perturbed inputs);
  - equal keys across two such builds;
  - a different key on a shape change;
  - stability rewrites survive: a `log(1 + x)` model keeps `Log1p`, and an `x**2` model keeps `Sqr`;
  - the fallback path, forced;
  - `erfcx` against `scipy.special.erfcx` on [−20, 1e5], including the guard band;
  - a truncated-Normal model whose `mu` is a free parameter. This crashes today without TFP, and must
    work after F2a.
- **Extra models (universality, not benchmark models):**
  - `pm.Data` used for observed data;
  - integer-written priors such as `pm.Normal("mu", 0, 10)`;
  - an `MvNormal` with a constant covariance matrix;
  - a `pm.Potential` that uses a data array;
  - a Poisson GLM with integer observations.

### F2b — sampler builds once, per-run values as arguments
- **Cache** the jitted functions on (F2a key, C, `max_leapfrog_steps`, `target_accept`, `iterate`,
  dtype, metric kind, progress bar on/off, and the other static settings the code branches on).
- **Pass these as arguments:**
  - RNG keys;
  - the metric;
  - step size and trajectory length;
  - the F2a lifted values.
- **Sites:**
  - the warmup loop, `utils.chees_warmup_with_progress`;
  - the two-phase refresh path;
  - the sampling loop, `sampler.py:1181-1353`;
  - the ε search, `:892-958`;
  - the helpers at `:458`, `:928`, `:1174` and `utils.py:200`;
  - the Pathfinder fit and draws, `:640` and `:662`;
  - the value→RV transform.
- **Mechanism.** Jitted functions are defined once at module level, with two kinds of arguments:
  - **static:** the lifted `fn` (from F2a's registry, so an equal key gives the same object) and the
    settings;
  - **dynamic:** everything per-run.

  JAX's own jit cache then does the caching. No hand-written cache dict, and no closures rebuilt per
  call.
- **Two tasks (re-split 2026-10-23):**
  - **F2b-1: infrastructure plus every non-loop site.**
    - Sites: the helpers, the ε search, the Pathfinder fit, draws and scoring, and the value→RV
      transform.
    - Adds a compile-counting test helper (`jax.monitoring` backend-compile events).
    - Done when a second call with a new seed, data or priors compiles nothing outside the warmup
      and sampling loops.
  - **F2b-2: the warmup and sampling loops, plus F2-3.**
    - The loops run as host-driven chunks of jitted steps with dynamic trip counts.
    - The progress bar is updated from Python between chunks, so `jax.debug.callback` leaves the
      compiled programs. Today a cached program would call the first call's bar, and a callback blocks
      JAX's on-disk cache (F2e).
    - Draws go into a buffer of static size `min(D, 1000)`, filled with a dynamic count. One program
      serves every D at or above the cap; the remainder needs no extra program.
    - Also covers the two-phase refresh (fixed 20-step position buffer) and the stop-rule path.
    - Done when W and D change nothing compiled.
- **Tests:**
  - zero XLA backend compiles on a second call with a new seed, new data or new priors (same shapes),
    a larger W, or a larger D. Count them with `jax.monitoring` compile events, not log handlers;
  - exactly one compile set after a shape change;
  - draws bit-identical to `main` for a fixed seed where the algorithm is unchanged. Where it cannot
    be, say why and run the bias gate;
  - the R1 bias gate on an M2a slice of the 4 models, compared with `main`;
  - the refresh and superchain paths;
  - the full suite.

### F2c — cap the start-candidate scoring (scales with d; universal)
- **Today:** the stratified start pick scores `min(pool, 64·C)` candidates. Under the `mc` pool
  `max(4C, 4d²)` that is up to 4d² log-density evaluations: 122,500 (6.4 s) at d=175, and 1,000,000
  at d=500.
- **Change:** score at most k·C candidates (k fixed, about 4), whatever d is. The metric's pool is
  unchanged.
- Tests:
  - start-dispersion ratio (REPORT run-1 §3j method);
  - the R1 bias gate at the M2a smallest-passing W, compared with `main`.

### F2d — initial point without per-call PyTensor compiles (fixed cost per call; universal)
- **Today:**
  - `_pymc_adapt_diag_start` runs PyMC's init through PyTensor's C backend on every call
    (0.5–1.6 s);
  - `model.initial_point()` compiles a PyTensor function (`:202` and elsewhere).
- **Change:**
  - build the initial-point graph once per model through F2a's lifting, in JAX;
  - apply PyMC's own jitter semantics;
  - check finiteness with the F2a log-density.
- Tests: the start distribution matches PyMC's (statistical test), and the Pathfinder fit is unchanged.

### F2e — opt-in on-disk compile cache (made on by default in F2g, 2026-10-23)
- Once keys are structure-only, compiled programs are reused across Python sessions.
- It is a config switch, documented and off by default.

### F2f — measurement (lead; needs a quiet host)
- **Back to back on the same host:**
  - `main` (checkout) at the M4 shape (C=1,024, D=10, each model at its smallest passing W), cold and
    warm, per stage;
  - `f2` at the same shape:
    - first call in a new process, with and without the on-disk cache;
    - second call with a new seed, new data or new priors;
    - per stage.
- **CPU NUTS first call** (fresh process, PyTensor cache cold vs warm), for the fair comparison when
  data or priors change every run.
- **Output:** R2-T4 redone for two workflows: "new data/priors every run" and "repeat runs".

### F2g — after F2f (owner's go 2026-10-23: "Yes to all")
Three items from the F2f report (`results/f2/F2f/REPORT.md`, Findings), done in one task.
1. **Jitted start-repair check (universal).**
   - **Today:** `_find_finite_start` (`sampler.py:350`) decides "usable" with an eager `self.ld_fn(pos)`
     and `jax.grad(self.ld_fn)(pos)`.
   - **Who pays:** any model whose shared start has a non-finite log-density or gradient. GEV's start is
     at xi = 0, where the gradient is NaN by design.
   - **Cost:** 156 one-off primitive compiles on a cold call (about 9 s on the GPU), and op-by-op
     dispatch on every warm call (0.21 of GEV's 0.55 s).
   - **Change:** decide it with `compiled.usable_mask` on a one-row stack (`self._logdensity()`, so an
     instance override of `ld_fn` is still honoured). The decision and the returned point are
     unchanged, so the draws are too.
2. **`tests/conftest.py` teardown.** The one-process suite came within 4.5k of `vm.max_map_count` at
   f6108ac. Add the F2e module-scoped teardown (`gc.collect(); jax.clear_caches()`) for every module.
3. **On-disk compilation cache on by default** (supersedes F2e's "off by default").
   - **Default directory:** under the user's cache directory, the analogue of PyTensor's compiledir,
     which is also on by default.
   - **Never override a user's own JAX cache setting.**
   - **Opt-out:** one documented switch.
   - **No import-time side effects.** The README states the global effect on JAX settings.
   - **Tests run with it off,** except the tests of the default itself, which run in fresh
     subprocesses.

## Projection to verify in F2f (from the M4 stage regression; quiet host)

| model | warm call today (quiet, M4) | after F2b (projected) | CPU NUTS (M5) |
|---|---|---|---|
| 8schools | 6.3 s | about 1.3 s | 3.5 s |
| gev | 10.5 s | about 3.5 s | 5.4 s |
| logit_1000x25 | 8.4 s | about 2.2 s | 9.3 s |
| radon_noncentered | 24.9 s | about 17 s; about 11 s with F2c | 19.1 s |

## Decisions (resolved with the owner, 2026-10-21; F2-2 revised 2026-10-22)

### F2-1: `erfcx`
- Register PyTensor's `Erfcx` JAX dispatch to JAX's own `jax.scipy.special.erfcx`.
  - That function exists in jax 0.11.2 and matches SciPy to the last digit, except at
    26.55 ≤ x ≤ 26.63, where it returns 0 (a JAX bug).
  - Guard that band with the asymptotic series 1/(x√π)·(1 − 1/(2x²) + 3/(4x⁴)).
- Test it against `scipy.special.erfcx` from −20 to 1e5.
- File the JAX bug and drop the guard when it is fixed. Offer the registration upstream to PyTensor.
- Rejected options:
  - TFP: only nightly builds support this JAX (`tfp-nightly 0.26.0.dev20261020` verified), and it is a
    heavy dependency for one function;
  - our own `erfcx`: more code to maintain than wrapping JAX's.
- Side effect: this also fixes today's JAX path for any model with a parameter-dependent `log(erfc(·))`,
  for example a truncated or censored Normal. The same PyTensor rewrite emits `erfcx` there.

### F2-2: which constants become runtime inputs (rule C; revised 2026-10-22)
The owner's intent is unchanged: changing data or prior values must not recompile, whatever their
type. The 2026-10-21 wording ("every float constant") also lifted the formula constants of PyMC's
logp derivations, such as the 0.5, the log 2π and the 1 in `log(1 + x)`. Probe 6 shows that blocks
PyTensor's stability and specialisation rewrites. Those constants never change between runs.

- **Lifted, i.e. the model's inputs:**
  - every constant that is a direct input of a random-variable node (distribution parameters; for
    example the `0` and `10` of `pm.Normal("mu", 0, 10)`, whatever their dtype);
  - every observed-data constant;
  - every non-scalar constant anywhere in the graph (data arrays, index arrays, constant matrices);
  - every shared variable (`pm.Data`), replaced directly rather than through
    `_replace_shared_variables`.
- **Applied only to value uses.**
  - A candidate is lifted when every client, looking through `DimShuffle`, is one of: `Elemwise`,
    `CAReduce`, `Dot`, `Blockwise`, or `AdvancedSubtensor`/`AdvancedSubtensor1`.
  - Otherwise it stays static, for example a shape, size, axis or basic-index position. JAX needs
    those at compile time.
  - Kept constants are reported in the diagnostics.
- **Kept static:** formula constants, and scalar constants written inside user expressions rather
  than passed to a distribution. Changing one of those recompiles once, which is correct.
- **Applied before** `mode.JAX.optimizer.rewrite`, so prior values cannot be folded or fused into
  `Composite` ops.

### F2-3: loop lengths (no user knob)
- **Warmup: W becomes a runtime trip count**, so any W runs with no recompile and no wasted steps. The
  warmup depends on W only incidentally:
  - Halton bit depth (`utils.py:168`): fix it at 32 bits, which gives identical values;
  - the progress-bar end check (`utils.py:278`).
- The per-step history the warmup stacks is used only for:
  - sums (gradients, divergences), which become running sums;
  - the refresh's last `min(W_A, 20)` positions, which become a fixed 20-step buffer.
- The stop-rule path already runs in blocks; its block length stays static.
- **Draws: one program per distinct `min(D, 1000)`.** The buffer has that static size and is filled
  with a dynamic count, so D=10 compiles once, and D=2,500 and D=5,000 both reuse the 1,000-draw
  program with no remainder program (refined 2026-10-23). The cap only bounds per-call device memory.
