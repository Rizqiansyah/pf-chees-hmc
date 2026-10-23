# F2b-1 SUMMARY — infrastructure plus every non-loop site

Task: fast-tier, two rounds (round 1 was interrupted and resumed).
- Branch `f2-b1`: commits 4ac47a3, 3bd5976, dae25fa, 6594c6f (round 1) and f1ca01d, 581fe0d
  (round 2).
- Merged into `f2` with `--no-ff`.
- Task's final suite (581fe0d): `315 passed, 13 warnings in 391.38s (0:06:31)`.

## Design
`pfcheeshmc/compiled.py` defines one module-level `jax.jit` per site.
- **Static arguments:** the lifted `fn` (from F2a's registry, so structurally equal models pass the
  same object) and the value-variable `names`.
- **Dynamic arguments:** the lifted values, keys, positions, metric and step sizes.
- JAX's jit cache does all the caching; there is no hand-written cache dict.
- `LogDensity(fn, names, values)` is the handle, and it is also a plain callable for the loops, which
  F2b-2 converts.

## DEVIATIONS (all accepted)
1. **Draws are not bit-identical to 1eda92c.**
   - The Pathfinder fit, pool draw, scoring and transform used to run eagerly, op by op. Each is now
     one fused XLA program.
   - The log-density's data and priors are arguments, so XLA can no longer fold them as constants.
   - Worst relative differences, end to end (C=8, W=20, D=10, seed 3):

     | model | worst relative difference |
     |---|---|
     | 8schools | 1.3e-11 |
     | gev | 1.2e-10 |
     | logit | 5.4e-13 |
     | radon | 1.2e-12 |

   - Discrete statistics (`num_integration_steps`, `diverging`, `is_accepted`) are exactly equal.
   - float32 was not examined: the package runs float64, because PyTensor's dispatch enables x64.
2. **Files outside the brief:**
   - `ensemble_metric.py`: `_estimate` is split into a module-level jitted `_raw_covariance` and the
     host PD check. The refresh path recompiled 2 scans per call; results are unchanged.
   - `tools/warm_stages.py` (new), the measurement tool.
   - Test spies on `bj.vi.pathfinder.approximate`/`sample` moved to `compiled.pathfinder_fit`/`_draw`.
     A spy that stores a result inside the jit would leak a tracer.
3. **`LiftedGraph` gained `sources`, `current_values()`, `static_shared` and `is_stale()`.** `sample()`
   calls `_refresh_values()`:
   - a `pm.set_data` at the same shape is seen with 0 compiles;
   - a shape change lifts again;
   - a change to a kept-static (baked-in) shared variable lifts again. That was round 2: the
     lead found that such a value would otherwise go silently stale.

## Sites (all DONE)
| site | where |
|---|---|
| `usable_mask` | `sampler.py:493` |
| `is_feasible` | `:959` |
| ε search (`fn` static; `values`, `imm` dynamic) | `:967` |
| `pool_gradients` (also `fisher_lowrank`) | `:1203`, `:818` |
| `batch_init` (falls back to the old jit for a plain callable) | `utils.py:201-206` |
| Pathfinder fit, one program; BlackJAX defaults as trace constants, as before | `:672` |
| Pathfinder pool draw (`num_samples` static) | `:693` |
| start scoring | `:660`, `:713` |
| value→RV transform | `:1417` via `_lift_value_to_rv:1382` |
| `_find_path` probe step, replaced by `jax.eval_shape` (it compiled an eager `while` per call) | `:1235` |
| ensemble estimators | `ensemble_metric.py:64-107` |
| a replaced `ld_fn`/`_ld_fn` is honoured, memoised per replacement | `sampler.py:208` |

## Tests
`tests/test_compile_once.py`: 52 tests, about 2 minutes. `tests/compile_counter.py` counts
`/jax/core/compile/backend_compile_duration` events with `fun_name` and unregisters in `finally`.
- **Per site:** 6 sites × (new seed, new data, new priors, all three) give 0 compiles.
- **Whole `sample()`:** a second call compiles exactly `['jit(scan)', 'jit(scan)']`, the two loops.
  This holds with `progressbar=False` and with `True`. The refresh path compiles 3.
- **Shape change:** changing C or J compiles once; repeating it compiles 0; changing back compiles 0.
- **Paths:** `imm_source` (`pf_closed`, `lbfgs_lowrank`, `fisher_lowrank`), superchains and the 4
  estimators compile 0.
- **`pm.set_data`:**
  - same shape: seen, with 0 compiles;
  - shape change: lifted again;
  - kept-static shared variable (`theta[idx]`): lifted again, the log-density equals a fresh build,
    and a whole `sample()` follows the new index.
- **Equivalence with 1eda92c** (subprocess on a `git archive`): the 4 ladder models, a refresh case,
  and `pf_closed` with superchains. Continuous arrays agree within `1e-7·(1+|x|)`, discrete arrays
  exactly, and the config keys and non-floats are equal.
- **Mutations M1–M11** each fail the intended test. M11 (drop `is_stale()`) reproduces the silent
  wrong answer: after `set_data(idx=3)` the sampler still concentrates on `theta[2]`.
- **Not covered by a mutation:** the transform site. An eager transform did not recompile either,
  because of the primitive caches.

## Measurements (task, CPU, C=1024, D=10, W=25, load average about 2; `tools/warm_stages.py`)
Warm call: the mean of calls 2 and 3, each with a new seed. Cells are seconds / XLA compiles.

| model | tree | wall | init | pathfinder | eps0 | warmup | sampling |
|---|---|---|---|---|---|---|---|
| 8schools | old | 6.07/8 | 0.59/1 | 0.80/1 | 0.53/2 | 2.04/2 | 1.93/2 |
| 8schools | new | 4.27/2 | 0.46/0 | 0.03/0 | 0.03/0 | 1.92/1 | 1.63/1 |
| gev | old | 9.18/8 | 1.62/1 | 2.20/1 | 0.98/2 | 2.01/2 | 1.85/2 |
| gev | new | 4.60/2 | 1.24/0 | 0.03/0 | 0.04/0 | 1.61/1 | 1.19/1 |
| logit_1000x25 | old | 7.55/8 | 0.45/1 | 1.36/1 | 0.58/2 | 3.05/2 | 1.95/2 |
| logit_1000x25 | new | 5.16/2 | 0.38/0 | 0.09/0 | 0.05/0 | 2.90/1 | 1.59/1 |
| radon_noncentered | old | 16.81/8 | 1.05/1 | 8.94/1 | 0.86/2 | 3.36/2 | 2.23/2 |
| radon_noncentered | new | 7.12/2 | 0.65/0 | 1.14/0 | 0.11/0 | 3.13/1 | 1.74/1 |

Cold first call in a fresh process (old → new, s / compiles):
- 8schools: 25.9/505 → 14.3/152;
- gev: 31.4/678 → 19.4/283;
- logit: 25.6/502 → 13.0/130;
- radon: 39.7/656 → 18.9/158.

The remaining warmup and sampling time is dominated by the 2 loop compiles (F2b-2). "Init" is PyMC's
`init_nuts` on the C backend (F2d).

## UNVERIFIED
- Quiet-host and GPU timings.
- float32.
- The stop-rule path beyond one `block=10` probe.
- `set_data` on a shape-feeding shared variable: it goes through the same re-lift path, but is
  untested.

Tooling note: `JAX_PLATFORM_NAME=gpu` leaks from the example notebooks into `os.environ` of any process
that builds a ladder model. The equivalence subprocesses drop it.
