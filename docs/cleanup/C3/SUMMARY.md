# C3 (tests) SUMMARY

## DEVIATIONS
1. **The "about 5 min" goal is not met on this host: 8 min 35 s (515 s) against 13 min 24 s (804 s) before (-36%).** The host load was 36-38
   (112 cores, shared) before, during and after both runs, and `OMP_NUM_THREADS=4`; the suite is compile-bound and
   compile time scales with the load. What is left is spread thin: the largest module is `test_golden.py` (54 s), then
   `test_compile_once`, `test_pf_closed_form`, `test_chunked_loops`, `test_compilation_cache` (42 s, one 30 s child), `test_lift_memo`,
   `test_lifted` (37-52 s each); no test is over 17 s. A run on a quiet host is UNVERIFIED (I could not get one).
2. The old-code proofs also covered a few things that the golden files do not, by choice (all were proofs against old
   commits, not behaviour): the `compile_equiv` cases at C=1,024 (`test_compile_once` section 4, tolerance-based for the cases that
   changed at F2c), the notebook-vs-Python `logp`/`dlogp` graph equality of `test_benchmark_models` (the draws of the four models
   are pinned bit for bit instead), the widened-graph digests against 44a0556 (`test_widen`; the `clone_replace` equality tests
   stay), the compile counts "no program the old code did not compile" (`test_pm_chains`; the cold counts of the new code stay in
   `test_start_repair`), and the F2 head KS comparison of `test_chunked_loops` (the analytic check stays).
3. No package change: `git diff cleanup -- pfcheeshmc benchmarks pixi.toml pixi.lock pyproject.toml` is empty.

## Spec items
1. **Golden draws: DONE.** `tests/make_golden.py` (writes `tests/data/golden_<case>.npz`; `--tree`, `--out-dir`, case names),
   `tests/test_golden.py`, 18 files, 660 KB, 448 arrays. Cases (C=8, D=10, W=20; every posterior and sample_stats array incl. the unconstrained
   ones, the numeric attrs `frozen_step_size`/`frozen_L_nominal`/gradient counts/`warmup_steps_used`, and the config; compared with `np.array_equal`,
   dtype and shape; the failure message says how to regenerate and that a regeneration needs a reason in the commit):
   the 4 ladder models; GEV with `pm_chains=3` (1 < pm_chains < chains), `pm_chains=C`, `init_source=adapt_diag` (GEV's start is on xi=0, so each runs the start repair);
   `start_repair` (the repaired stacks of GEV rows on the singularity, the backtrack, the restart and the origin search of the fence and ball models, and the
   starts a `pm_chains=3` Pathfinder stage hands the chains); `8schools_rebuilt` (second call on a rebuilt model: lift memo) and `shaped_literals`
   (float32 and int8 literals widened); `8schools_chunked` (`WARMUP_CHUNK/SAMPLE_CHUNK/TRANSFORM_CHUNK` = 7/3/4: several warmup, sampling and transform chunks);
   `8schools_warmup_stop_fires`; `8schools_imm_source` (mc, lbfgs_lowrank, fisher_lowrank, pf_closed); `8schools_pf_pool_cap`; `float32_<model>` for the four
   ladder models, both init sources (the children `test_float32.py` runs anyway, via `float32_run.py`; the float32 goldens are compared there, `test_float32.py::test_the_float32_draws_are_those_of_the_golden_file`).
   - **Evidence against the tag: all 448 arrays equal.** `git archive archive/pre-cleanup` into a scratch dir, then
     `PYTHONPATH=<tag tree> python tests/make_golden.py --tree <tag tree> --out-dir <dir>` (the tag side builds models with its own `tools/r1_bias.build_model`, picked by
     `os.path.isdir(TREE/benchmarks)`; only kept options are passed) and compared file by file with `tests/data` (config compared after dropping the 14 keys of the options C2 removed, which the tag still records). Result, 18 of 18 files `MATCH`, `total arrays 448 differing 0`. That comparison is not a test.
   - **Mutation:** `pfcheeshmc/sampler.py:969` `kwargs.setdefault("adam_lr", 0.05)` -> `0.051` (reverted): `test_golden.py -k gev` -> 4 failed, e.g.
     `AssertionError: 12 of 18 arrays of gev_pm_chains_C differ from tests/data/golden_gev_pm_chains_C.npz: ['attrs/frozen_step_size', ..., 'posterior/z_p', 'posterior/μ', 'posterior/ξ']. If the change is meant to alter the draws, regenerate with ...`.
2. **Git-archive proofs dropped: DONE.** Removed: `test_compile_once` section 4 (equivalence), `test_chunked_loops` known-posterior-vs-F2-head, `test_pm_chains` (draws, cold compiles vs old),
   `test_start_repair` (old-vs-new arrays, draws, the premise asserts of old compile counts), `test_widen` (graphs and draws vs 44a0556), `test_float32` float64 identity,
   `test_api_prune` tag proof, `test_benchmark_models` (structure, draws, settings of the tag). Children removed: `api_prune_run`, `benchmark_models_run`, `compile_equiv_run`,
   `known_posterior_run` (its `analytic()`/model now in `test_chunked_loops.py`), `pm_chains_run`, `widen_run`. Kept: `start_repair_run.py` (new tree only; cold compile counts), `float32_run.py`, the two `compilation_cache_*_run.py`.
   `without_removed` and `REMOVED_CONFIG_KEYS` are gone (no importer left). **No test imports from another test module**: shared helpers are in `tests/_helpers.py` (extra models, literal models, `point`, `pymc_logp`, `mixing_report`) and
   `conftest.py` (the `odd_order_model`/`anisotropic_model` fixtures); `grep -n "^from test_\|^import test_" tests/*.py` is empty. (`test_float32` and `test_golden` import `make_golden`, a script, not a test module.)
3. **Tool tests: DONE.** `test_bias`, `test_nuts_cpu`, `test_reference_npz` test `benchmarks/` code that stays and total 11 s: untouched. `test_benchmark_models.py` is now 3 tests (the ladder is the four models; `build_model` changes no `jax_*` option or env variable, in-process; no entry point replaced).
4. **Merged: DONE.** `test_acceptance_display` -> `test_progressbar` (the number the bars show), `test_init_pick` -> `test_start_scoring` (the start pick). Nothing else merged (same-unit modules are already one).
5. **Kept:** compile-count tests, the compilation-cache tests, float32 behaviour, statistical checks, output contract, `test_conftest_release`: all pass unchanged. Shrunk/dropped as redundant with another test: `test_chunked_loops` chunk-size invariance of "stop rule that never fires" (same as `test_draws_do_not_depend_on_the_stop_rule_block`); `test_lift_memo` memo-vs-no-memo draws on `logit_1000x25` (kept on 8schools; the graph-generic claim, also pinned by the ladder rebuild tests there).
6. **Durations:** below.

## Coverage diff (required)
Tool: a throwaway `sys.monitoring` LINE tool (scratch dir, not committed; id 4, restricted to files under `<tree>/pfcheeshmc/`, `DISABLE` after the first hit of a line, dumped at exit per pid, files unioned),
installed through `usercustomize.py` on a `PYTHONUSERBASE` so that **child processes are covered too** (the children set their own `PYTHONPATH`, which would drop a `sitecustomize`).
Run on the suite of `216bae3` (a clone with `.git`, so the git-archive tests ran, not skipped: 573 passed) and on this branch's suite (542 passed).
Executable lines in `pfcheeshmc/` (from `co_lines`): 1,804.

| | lines hit | dump files (pids) |
|---|---|---|
| before (216bae3) | 1,738 | 90 |
| after (this branch) | 1,738 | 31 |

**Lines covered before and not after: 0. Gained: 0** (the two sets are identical; `diff` printed `lost 0 gained 0`). The 66 lines never hit by either suite are the same (not listed).
Caveat: Python trace-time execution only (the bodies of jitted functions run when traced, so they are counted; what XLA does is not); the tool is not `coverage.py` and its executable-line set is `co_lines`, so a line count is not comparable to a `coverage.py` percentage.

## Timings (`--durations=0`, per-module sums of setup+call+teardown; the host is shared and loaded)
Before: `216bae3`, clone with `.git`, `573 passed, 56 warnings in 804.31s (0:13:24)`, load `34.30 36.76 37.48` after, `32.53 35.90 36.87` before.
After (this branch, plain command): load before `36.00 37.29 37.36`, after `35.67 36.27 36.99`.

| module | before s | after s |
|---|---|---|
| test_compile_once.py | 87.4 | 52.5 |
| test_benchmark_models.py | 69.2 | 0.1 |
| test_pm_chains.py | 67.5 | 15.6 |
| test_api_prune.py | 54.5 | 0.1 |
| test_widen.py | 54.5 | 9.9 |
| test_chunked_loops.py | 53.1 | 43.0 |
| test_pf_closed_form.py | 46.4 | 43.8 |
| test_lift_memo.py | 46.3 | 39.7 |
| test_compilation_cache.py | 42.7 | 41.7 |
| test_lifted.py | 41.1 | 37.7 |
| test_float32.py | 40.6 | 21.4 |
| test_start_repair.py | 38.4 | 9.0 |
| test_metric_orientation.py | 25.5 | 22.2 |
| test_initial_points.py | 23.8 | 23.4 |
| test_step_size_batched.py | 19.5 | 18.2 |
| test_wp5_capacity.py | 15.5 | 15.6 |
| test_warmup_stop.py | 15.5 | 14.0 |
| test_start_scoring.py (+ test_init_pick 6.3) | 13.0 | 15.0 |
| test_bias.py | 10.1 | 10.5 |
| test_chees_vendored.py | 7.5 | 8.6 |
| test_progressbar.py (+ test_acceptance_display 0.6) | 7.1 | 5.2 |
| test_rng_stage_keys / test_statistical_correctness / test_finite_start / test_step_size_tune / small | 13.8 | 13.5 |
| test_golden.py (new) | 0 | 53.6 |
| sum of durations | 800.2 | 514.3 |

Commands: `pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 PYTHONPATH=<tree> python -m pytest <tree>/tests -q --rootdir <tree> -p no:cacheprovider --durations=0`
(the coverage runs add `PYTHONUSERBASE=<scratch>/ub COV_PREFIX=<tree>/pfcheeshmc/ COV_OUT=<dir>`; they gave `542 passed, 48 warnings in 544.48s (0:09:04)` here, load 31-40).

## Final suite line (the finishing command, plain, copied from its output)
`542 passed, 48 warnings in 515.28s (0:08:35)`

Tests: 573 -> 542 (-31 net: the archive proofs and the redundancies above out, 28 golden/float32-golden tests in).

## Notes for the lead
- `docs/cleanup/TASK-RULES.md:33` still names `git show archive/pre-cleanup:tests/widen_run.py` as the F2 pattern; `tests/widen_run.py` is gone from the branch (the tag still has it). C2/C4 SUMMARYs refer to the removed files (history, left as is).
- `tests/make_golden.py --tree` is the only place the tag's `tools/r1_bias` builder is still named (as a path chosen by `os.path.isdir(TREE/benchmarks)`), for regenerating or re-checking the files against the tag.

## UNVERIFIED
- The ~5 min goal on a quiet host (see DEVIATION 1).
- The golden files are exact on this CPU/JAX/XLA build; another CPU or XLA version may change a float bit (the old proofs compared two runs of one host, so they could not see that). If it does, the failure message says what to do.
