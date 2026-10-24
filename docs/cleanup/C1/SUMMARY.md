# C1 SUMMARY: the four ladder models in Python, no notebook loading

Branch `cleanup-c1`, from `cleanup` at e087a2f. Model: fast-tier.

## DEVIATIONS
1. **`logit_10000x50` and `radon_centered` are not carried.** No test and no kept tool uses either (`grep -rn "logit_10000\|radon_centered" tests tools README.md`
   finds only the two old `model_specs()` entries). `benchmarks.models.MODELS` is the four ladder names, so `bench_ladder.model_specs()` and the
   `make_reference.py` model choices shrink from six to four. `build_radon(False)` and `build_logistic(10000, 50)` are still callable from
   `benchmarks/models.py` (both builders moved unchanged and keep their parameters).
2. **Files outside the C1 list were edited, each for a reason:**
   - `tools/stage_timers.py` and `tools/probe_init_policy.py` called `bl.load_notebook_model` / `bl.build_radon` directly: now `bl.build_model(...)`;
   - `tools/wp1_rebaseline.py`: a comment plus an `assert` that guarded against the deleted `_tiny_idata` stub (it named `load_notebook_model`, which the
     acceptance grep must not find). The `assert` in its repeat loop is kept; it is true and harmless;
   - `tests/test_metric_orientation.py`: one line, `assert ...__name__ != "_tiny_idata"`, the same guard;
   - `tests/test_r1_bias.py`: one comment.
3. **`REAL_SAMPLE_PF` and `REAL_PM_SAMPLE` stay in `tools/bench_ladder.py` as plain aliases** (nine tools import them, C4 deletes most). Nothing stubs them now; the comment says so.
   `r1_bias.run_cell`'s per-cell "is the sampler a stub?" check is deleted with the stubs.
4. **Left alone, now stale in a comment only:** `tools/make_reference.py` (docstring lines 26-27: "its notebook loader stubs pm.sample"),
   `tools/t3_cpu_nuts.py` (line 53 comment), `tools/t5_step_cost.py` (line 38 comment). C4 rewrites or deletes them. Five `env.pop("JAX_PLATFORM_NAME", None)` lines in
   test_widen, test_pm_chains, test_start_repair, test_compile_once, test_chunked_loops and test_float32's `_env` are defensive and harmless; C3 removes those tests or runners.
5. **A side effect that went with the loader:** the first `r1_bias.build_model` in a process imported `bench_ladder`, which set `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
   `NUMEXPR_NUM_THREADS=1` in `os.environ` (and the notebook cell set `JAX_PLATFORM_NAME=gpu`), and the pytest process's child runners inherited them. Nothing sets them now. The suite
   compares trees in children that share one environment, and it passes, so no test depended on them. It could change the timing of a child, not its bits.

No other deviation. Tools that no longer import: **none** (static check of every `bl.X` / `r1_bias.X` / `from ... import` in `tools/**/*.py` and `tests/*.py` against the new module
names: no missing name; `py_compile` of all of `tools`, `tests`, `benchmarks`; `r1_bias.py --help`, `make_reference.py --help` and an import of `bench_ladder` ran).

## Items
| spec item | status | where |
|---|---|---|
| `benchmarks/__init__.py`, `benchmarks/models.py` with `build_model(name)` and `MODELS` | DONE | `benchmarks/models.py:150` (`MODELS`), `:153` (`build_model`; unknown name raises `KeyError`) |
| 8schools and gev transcribed from the notebooks' model cells; nothing else run | DONE | `benchmarks/models.py:30` (`build_8schools`), `:46` (`build_gev`; `pymc_extras` imported in the function, as the cell did). The notebook's `pm.sample_prior_predictive(1000)` calls were stubs in the loader and are not part of the model |
| `pdb_data` + cache and the logistic builder moved unchanged; no vendored data | DONE | `benchmarks/models.py:81`, `:96` (`build_radon`), `:129` (`build_logistic`) |
| `build_model` touches no JAX setting | DONE, tested | `tests/test_benchmark_models.py:147`, `tests/test_float32.py:138` |
| `logit_10000x50`, `radon_centered` | SKIPPED (unused), see DEVIATIONS 1 | |
| Proof of sameness (value vars, free/observed/deterministic, `initial_point`, `compile_logp`/`compile_dlogp` at 100 points, `equal_computations`, draws C=8 D=10 W=25 seeds 1, 2, float64 and float32) | DONE, all hold | `tests/benchmark_models_run.py` (child), `tests/test_benchmark_models.py:106` and `:119`; skips when git or the tag is unavailable (`:73`) |
| `r1_bias.build_model` thin; snapshot/restore and F2k's x64 switch deleted | DONE | `tools/r1_bias.py:117`; `sys.path` for `benchmarks/` at `:77` |
| `bench_ladder.model_specs()` from `benchmarks.models`; `load_notebook_model`, stubbing, `_tiny_idata`, `SKIP_PREFIX`, `build_radon`, `build_logistic`, `pdb_data` deleted | DONE | `tools/bench_ladder.py:97` (`model_specs`), `:52-58` |
| Tests and child runners switched; runners work for the tag and this tree | DONE | in-process: test_lifted, test_lift_memo, test_widen, test_step_size_batched, test_initial_points, test_pf_closed_form (`from benchmarks.models import build_model`). Runners (`try: from benchmarks.models ... except ModuleNotFoundError: from r1_bias import build_model`): `compile_equiv_run.py:36`, `float32_run.py:51`, `pm_chains_run.py:25`, `start_repair_run.py:43`, `widen_run.py:43` |
| `tests/test_bench_ladder_stub_restore.py` deleted | DONE | `git rm`; its 2 tests are the only ones gone |
| F2k "x64 is off after `build_model`" becomes "never changes `jax_enable_x64`, at either floatX" | DONE | `tests/test_float32.py:138` (4 models, float32), `:158` (float64: `x64_start` and `x64_built` both True), `tests/test_benchmark_models.py:147` (4 models x 2 floatX x before/after the sampler import) |
| Notebooks, `pfcheeshmc/`, untouched tools | DONE (untouched) | `git diff --stat cleanup..cleanup-c1 -- examples pfcheeshmc` is empty |

## Tests
Added `tests/test_benchmark_models.py` (23 tests) with the child `tests/benchmark_models_run.py` (modes `structure`, `draws`, `settings`). The old tree is `git archive archive/pre-cleanup` extracted
to a temp dir. About 80 s with 4 children at a time.
- `test_the_python_model_is_the_notebook_model[model x floatX]` (8): in one process, `tools/r1_bias.build_model` of the tag (executes the notebooks) and `benchmarks.models.build_model`. Checks:
  value vars (name, shape, dtype, order), free/observed/deterministic/potentials/basic/unobserved RV names, `named_vars` order, dims and coords, `initial_point` (names, dtypes, bits),
  `equal_computations` on the `logp` graph (whole and term by term), on `dlogp` and on the deterministics, and `compile_logp`/`compile_dlogp` bit-equal at 100 points.
- `test_the_draws_are_bit_identical_to_the_notebook_models[model x floatX]` (8): two fresh processes (tag, this tree), a fresh build per call, seeds 1 and 2, C=8 D=10 `chees_kwargs={"draws": 25}`,
  `keep_untransformed=True`: all `posterior/*` and `sample_stats/*` arrays, their dtypes and the resolved config.
- `test_build_model_never_changes_a_jax_option_or_an_environment_variable` (4): every `jax_*` value of `jax.config` (116 options) and `os.environ`, before and after building the four models, at both floatX, before and after importing `pfcheeshmc`.
- `test_the_check_can_fail_the_notebook_loader_changed_them`: the premise, on the tag.
- `test_the_ladder_is_the_four_models`, `test_build_model_replaces_no_sampling_entry_point`.
- `tests/test_float32.py`: `test_the_two_notebook_models_leave_x64_off_under_float32` became `test_build_model_never_changes_x64_under_float32[model]` (4), and the float64 identity test asserts x64 is on before and after.

**Tests that must fail on the old code or on a wrong model:**
- Old code, `settings` child on the tag (`python tests/benchmark_models_run.py settings TAG OUT.json`, PYTHONPATH=TAG): `changed = {"jax:jax_enable_x64": ["False", "True"], "env:JAX_PLATFORM_NAME": [null, "gpu"], "env:OPENBLAS_NUM_THREADS": [null, "1"], "env:XLA_FLAGS": [null, "--xla_force_host_platform_device_count=100"], ...}`
  at float64; at float32 the same without the x64 line. (The `XLA_FLAGS` change is `pymc/sampling/jax.py:64` running at import; `pfcheeshmc/lifted.py:59` imports it, and the old `build_model` imported `bench_ladder`, which imports `pfcheeshmc`. The new `build_model` imports neither; a process that imports `pfcheeshmc` sets it as before.)
  `test_the_check_can_fail_the_notebook_loader_changed_them` asserts exactly this.
- Mutations of `benchmarks/models.py` in a scratch copy, `pytest tests/test_benchmark_models.py -k "float64 and <model>"`:
  - 8schools `sigma[0]` 15.0 -> 15.1: `AssertionError: {'value_vars': True, ...} assert ['logp_graph'...'dlogp_graph'] == []` and
    `AssertionError: 8schools/seed1/config ... - _size": 0.755304965476942, "imm_or / + _size": 0.815315231857937, "imm_or` (the first posterior array equal check is after the config; the config differs through the tuned step size).
  - 8schools `tau` created before `mu`: `{'value_vars': False, 'value_vars_n': 3, 'free_RVs': False, ...}` (9 checks fail) and the draws test fails.
  - logit prior `pm.Normal("b", sigma=1.01, ...)`: `assert ['logp_graph'...'dlogp_graph'] == []` and `AssertionError: logit_1000x25/seed1/config ... - size": 0.5384792865518725 / + size": 0.5654271262114754`.

## Measurements
All on CPU (`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=4`), from the checkout:
`pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH=$WT python -m pytest $WT/tests/test_benchmark_models.py -q --rootdir $WT -p no:cacheprovider` -> `23 passed in 82.33s (0:01:22)`.
- Structure, float64 and float32 (`tests/benchmark_models_run.py structure TAG WT OUT.json`, run with `PYTENSOR_FLAGS=...,floatX=float32` for the second): every check true for all four models,
  `logp_graph` and `dlogp_graph` by `equal_computations` included (so the strongest comparison, graph equality, holds); `compile_logp`/`compile_dlogp` bit-equal at 100/100 points, of which finite: 100, 93 (gev), 100, 100.
- Draws, float64 and float32 (`... draws TREE OUT.npz`, tag against this tree): 102 arrays per floatX (8schools 22, gev 26, logit_1000x25 16, radon_noncentered 38; seeds 1 and 2 together),
  **0 differences** in values, dtypes or config, at float64 and at float32. Posterior arrays finite; float32 posteriors are float32; x64 is on at float64 and off at float32 in both trees.
- Settings: 116 `jax_*` options and all environment variables unchanged by `build_model`, in the four combinations.
- Test ids against the tag (`pytest --collect-only -q`): 596 -> 620. Gone: `test_bench_ladder_stub_restore.py::test_loaded_model_has_observed_rvs`, `::test_notebook_load_restores_sampling_entry_points`,
  and `test_float32.py::test_the_two_notebook_models_leave_x64_off_under_float32` (replaced by 4 parametrized tests). Added: the 23 above.
- Acceptance grep `grep -rn "load_notebook_model\|ipynb" tools tests benchmarks`: **no output**.

## Full suite (CPU, the head before this file's commit)
On commit 6d89797 (the code of the final head; the head adds only this file):

`pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH=$WT python -m pytest $WT/tests -q --rootdir $WT -p no:cacheprovider`

`620 passed, 63 warnings in 828.14s (0:13:48)`

(An earlier run on e72e968, before the last comment and test-id edits, gave `620 passed, 63 warnings in 891.41s (0:14:51)`.)

## UNVERIFIED
- Nothing GPU was run (not asked). `radon_noncentered` read the posteriordb data already cached in `~/.cache/pf-ladder`; a cold download was not exercised (the function is unchanged).
- The runners' `except ModuleNotFoundError` fallback is exercised by the git-archive proofs of test_compile_once, test_chunked_loops, test_pm_chains, test_start_repair, test_widen and test_float32 (old trees at their BASE_SHA) and by the new test's `draws:old` child; the suite had no skips, so none of them was skipped here. A checkout without those commits or the tag skips them cleanly, as before.
