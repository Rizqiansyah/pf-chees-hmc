# C2 (API prune) SUMMARY

`pfcheeshmc/` at 432aaac is byte-identical to the tag's, so "line N" in the Q3 table are lines of the tag (b34e66e).

## DEVIATIONS
None from the Q3 list. Consequences and neighbours of the removal:
1. **`idata.attrs["config"]` loses keys:** `imm_orientation`, `imm_order`, `pf_start_scale`, `init_pick`, `superchains`, `iterate`, `metric_refresh`, and the 6 `metric_refresh_*` keys. The 9 `metric_refresh_*` attrs (written only when a refresh ran) are gone too. No array changes. Tests that compare a config with an older tree now drop those keys (`test_api_prune.without_removed`): test_api_prune, test_benchmark_models, test_compile_once, test_start_repair, test_pm_chains, test_widen, test_float32.
2. **Every stage rejects every key it does not read**, not only the removed ones. `_run_pathfinder`, `_run_step_size_tune`, `_run_chees`, `_run_dynamic_hmc` and `PfCheesHMC.sample(**kwargs)` used `kwargs.get`. This surfaced dead keys: `pathfinder_kwargs={"num_paths": 2}` in `tests/test_rng_stage_keys.py` and `tests/test_lifted.py` (never read; removed from both), and in a probe `_run_step_size_tune(..., eps_probes=...)` (the option is `eps0_probes`).
3. **Dead helpers removed beyond the nine names** (each lost its only caller):
   - `ensemble_metric.py`, `_run_chees_two_phase`, `_pool_gradients`, `_concat_warmup_info`, `_scale_starts`, `_pathfinder_mean`.
   - `center=` of `_find_finite_start` / `_ensure_finite_inits`.
   - `chees.base(iterate=, t0=)`, `compiled.warmup_init(t0, has_t0)`, `compiled.HostStack(keep_last=)`.
   - `chees_warmup_with_progress(iterate=, t0=, stack_positions=, positions_last=)`, and the `positions` field of `utils.WarmupInfo`.
   - `compiled.warmup_chunk(stack_positions=)` stays, because `warmup_stop` uses it. `sample_pf_chees_hmc`'s signature is unchanged.
4. **`chees.py` no longer has its stated reason for being vendored** (the `iterate` option). I rewrote that docstring section: it now differs from upstream `base` only by `ChEESTrace`, the `optim=None` convenience and `single_step_iters`. Un-vendoring is a C4/lead call (it touches the trace the progress bar reads), so I did not attempt it.
5. **Found, not removed** (not on the Q3 list, not made dead by this work): `single_step_iters` (`chees.py:150`; reachable only from a direct `chees_warmup_with_progress` call), `PfCheesHMC._kernel_flat` (one test only), `PfCheesHMC.last_covariance` (never read by the sampler).
6. Other test files changed because they compare a config or pass a removed argument:
   - `test_start_repair.py` + `start_repair_run.py`: the `last_resort` case used `center=`; it now forbids backtracks and restarts, which reaches the same fallback and the same answer.
   - `test_t3_cpu.py`: the `score_ensemble` signature.
   - `test_rng_stage_keys.py`, `test_lifted.py`: `num_paths`.
7. `README.md` and `examples/*.ipynb`: no line names a removed option (the README says "orientation" only in prose caveats and names `imm_source`, which stays). Nothing deleted. `tools/bench_ladder.py` passes only kept options. Nothing to remove.

## Q3 verification (tag's `pfcheeshmc/sampler.py` unless named)
| option | defined | default | gates | off default path? | entangled? |
|---|---|---|---|---|---|
| `imm_orientation` | :966 | "covariance" | "precision" -> `inv(covariance)` (:967-968) | yes (default is the surviving branch) | no; one value left, parameter gone |
| `imm_order` | :991 | "kernel" | "pathfinder" skipped `_metric_to_kernel_order` (:992-995) | yes | no; reorder unconditional |
| `metric_refresh` + `_ridge _shrink _rank _warmup _pool _reseed` | :1075, :1090-1092, :1157, :1167, :1204-1206, :1222 | "none" | `_run_chees_two_phase` (:1139-1269), `_pool_gradients` (:1271), `_concat_warmup_info` (:150), `ensemble_metric.py` (231 lines), 9 attrs (:1633-1638), `t0` / `positions_last` / `stack_positions` plumbing | yes (`refreshing` False by default) | `stack_positions` also serves `warmup_stop` (kept in `compiled.warmup_chunk`); `score_chunk` and `compiled.pool_gradients` (also `fisher_lowrank`) stay |
| `superchains` | :827 | None | `repeat(pick_global[:K], C//K)`; nested R-hat in `tools/r1_bias.py` `score_ensemble`; `--superchains`, `--nested-limit`; row columns in `tools/_common.py` | yes | no |
| `iterate` | `chees.py:162`, `utils.py:110`, `compiled.py:413`, `sampler.py:1087,1122,1178,1241` | "ma" | "raw" -> `use_raw` (`chees.py:192`, :339-366) | yes | no |
| `init_pick` | :797 | "stratified" | "iid" -> `randint(key_pick)` | yes | no; parameter gone, 5-way key split kept (`key_pick` -> `_`) |
| `pf_start_scale` | :863 | 1.0 | `!= 1.0` -> `_scale_starts` (:339-358) | yes | no |
| `pf_center` | :717 | None | `center=` of `_find_finite_start` (backtrack and restart target; None = zeros) | yes | no; the kept repair knobs are independent |

## Items
| # | item | status | where |
|---|---|---|---|
| 1 | verify Q3 against the code | DONE | table above |
| 2 | remove options and code paths | DONE | `sampler.py` (`_run_pathfinder` :608-892, `_run_chees` :955, `sample` :1292, `_config_record` :1371), `chees.py` (`base` :142, `init` :369), `utils.py` (:16, :80), `compiled.py` (:413, :457, :587); `ensemble_metric.py` deleted |
| 3 | unknown or removed option raises a named `TypeError` | DONE | `sampler.py:57` `_check_options`, called at :637, :938, :964, :1027 and :1302-1308 (`sample`, before any stage runs); sets :41-54 |
| 4 | bit-identity proof + must-fail | DONE | `tests/test_api_prune.py`, `tests/api_prune_run.py` |
| 5 | tests and tools | DONE | `tools/r1_bias.py`, `tools/_common.py`, `tools/t3_cpu_nuts.py` edited; the rest below |
| 6 | compile-once tests pass | DONE | `tests/test_compile_once.py`, 51 tests |
| 7 | no default changed | DONE | proof in item 4 |

TypeError as the user sees it: `TypeError: pathfinder_kwargs got an unknown option 'imm_orientation' (an option that was removed raises as well). The options are: chains, draws, fisher_cutoff, fisher_draws, fisher_gamma, imm_source, init_source, max_rank, pf_jitter_eps, pf_max_backtracks, pf_max_restarts, pf_pool_cap, pf_restart_radius, pm_chains, pm_init, pm_seed, pm_tune, ridge, score_chunk, seed`

## Lines of `pfcheeshmc/` (`wc -l pfcheeshmc/*.py`)
4,302 before, 3,709 after (-593): sampler.py 1,805 -> 1,510; ensemble_metric.py 231 -> deleted; chees.py 441 -> 408; utils.py 321 -> 292; compiled.py 636 -> 631; lifted.py, compilation_cache.py, __init__.py unchanged.

## The proof
`tests/test_api_prune.py::test_the_draws_are_bit_identical_to_the_tag[default|kept]`. Four children at once (2 groups x 2 trees), each a fresh process (`PFCHEESHMC_COMPILATION_CACHE=0`, CPU, x64) running `tests/api_prune_run.py TREE benchmarks/models.py GROUP OUT.npz`. The tag tree is `git archive archive/pre-cleanup` extracted to a temp dir. Both runs build models with this tree's `benchmarks/models.py`, so only `pfcheeshmc/` differs. 16 chains, 5 draws, 20 warmup draws.
- `default`: 4 ladder models x seeds 1, 2 (8 runs, 142 arrays).
- `kept` (22 runs, 356 arrays):
  - every `imm_source` (`mc`, `lbfgs_lowrank`, `fisher_lowrank`, `pf_closed`) and the Fisher knobs;
  - `pm_chains` 4 (shared point, and jittered starts with `pm_init`/`pm_tune`/`pm_seed`) and C (jittered), and 4 with `lbfgs_lowrank`;
  - `init_source="adapt_diag"` alone and with `pm_chains=4`;
  - `pf_pool_cap` 64 and 0, `score_chunk`, `ridge`, `eps0_probes`, the ChEES knobs (`adam_lr`, `target_accept`, `max_leapfrog_steps`, warmup seed);
  - `warmup_stop` not firing and firing, the dynamic-HMC seed;
  - GEV with the 4 start-repair knobs, and GEV with `pm_chains=C`.
- Compared: every posterior and sample_stats array (dtype, shape, bits), every numeric attr (`frozen_step_size`, `frozen_L_nominal`, `warmup_steps_used`, both gradient counts), and the config less the removed keys. Result: 0 differences.
- Guards against a vacuous pass: `test_the_tag_records_the_options_this_tree_removed` (the tag's config has the removed keys at their defaults, this tree's has none), and `test_the_cases_are_not_all_the_same_draw` (seeds, sources, `pm_chains`, a firing stop rule give different, finite draws; the stop rule stops at step 10).

### Must-fail (two mutations of `sampler.py`, each reverted). Command: `pytest tests/test_api_prune.py -k bit_identical` (~58 s)
1. `jax.random.split(key, 5)` -> `split(key, 4)` (the careless way to delete `key_pick`). Only the `adapt_diag` + `pm_chains<chains` jitter draws from the moved key, and only `kept` has that case:
```
FAILED tests/test_api_prune.py::test_the_draws_are_bit_identical_to_the_tag[kept]
E  AssertionError: 18 of 356 differ: ['8schools/init_source=adapt_diag+pm_chains=4/attrs/frozen_step_size', '8schools/init_source=adapt_diag+pm_chains=4/config', '8schools/init_source=adapt_diag+pm_chains=4/posterior/mu', ...]
1 failed, 1 passed, 23 deselected   (the default group passed)
```
2. `ridge = kwargs.get("ridge", 1e-6)` -> `2e-6`:
```
E  AssertionError: 113 of 142 differ: ['8schools/seed1/attrs/frozen_L_nominal', '8schools/seed1/attrs/frozen_step_size', '8schools/seed1/attrs/sampling_grad_evals_per_chain', ...]
E  AssertionError: 178 of 356 differ: ['8schools/chees_knobs/attrs/frozen_step_size', '8schools/chees_knobs/config', ...]
2 failed, 23 deselected
```

## Tests
Collected 620 -> 587 (`pytest --collect-only -q`, ids compared with `comm`: 71 gone, 38 new).
- **Added 38:**
  - `test_api_prune.py` 25: 14 removed options, 4 stages with typos, `sample(**kwargs)`, public function, removed modules and helpers gone, bit identity x2, 2 guards.
  - 13 renamed or re-parametrised in place: `test_chees_vendored` x2, `test_chunked_loops::...host_stack...`, `test_compile_once[8schools+pf_closed]`, `test_init_pick` x1, `test_metric_orientation` x4, `test_pf_closed_form` x3, `test_r1_bias::test_none_flags_add_no_key`.
- **Removed 71**, all exercising only a removed option or a deleted module:
  - `test_ensemble_metric.py` 18 (whole module); `test_nested_rhat.py` 5 (whole module; tests `arviz_stats.rhat_nested`, the superchains diagnostic).
  - `test_compile_once.py` 12: refresh and superchain second-call cases, 4 estimator cases, the `+refresh` and `+superchains` equivalence cases.
  - `test_metric_orientation.py` 11: the precision half of the source x orientation grid (6 ids), unknown-orientation error, two precision / `imm_order` failure tests, one duplicate default-mixing test.
  - `test_pf_closed_form.py` 10: `pf_start_scale` (4), `iterate` and refresh config (2), precision halves of G-ORDER (4).
  - `test_chees_vendored.py` 4; `test_chunked_loops.py` 5 (refresh cases and the `keep_last` half of the HostStack test); `test_r1_bias.py` 4 (2 superchains start tests, the GEV probe `--start-scale` test, the refresh-warmup flag test); `test_init_pick.py` 2 (iid pick, bad pick name).
- **Edited, none removed:** `test_r1_bias`, `test_t3_cpu`, `test_start_repair` (+ runner), `test_benchmark_models`, `test_pm_chains`, `test_widen`, `test_float32`, `test_rng_stage_keys`, `test_lifted`, `test_compile_once`, `test_chees_vendored`, `test_chunked_loops`, `test_metric_orientation`, `test_pf_closed_form`, `test_init_pick`.

## Tools not repaired (C4 deletes the probes; they pass a removed option, flag or signature and now fail)
- `tools/gev_divergence_probe.py` (`imm_orientation`, `pf_start_scale`, `--orientation`, `--start-scale`)
- `tools/gev_g3b.py` (`imm_orientation`)
- `tools/t5_step_cost.py` (`imm_orientation`, `iterate`, `stack_positions`)
- `tools/wp1_rebaseline.py` (`imm_orientation` via `IMM_ORIENT`)
- `tools/wp2_lgrowth.py` (`iterate`, `t0`)
- `tools/wp3_selftest.sh` (`--init-pick`, `--superchains`)
- `tools/probes/e2e.py` (`init_pick`), `tools/probes/tgrowth.py` (raw-iterate `base`, `t0`)
- `tools/probe_init_policy.py:125` (`eps_probes`: a key nothing read, now a TypeError)
- `tools/wp3_chop_check.py` computes nested R-hat straight from `arviz_stats`; it does not call the sampler and still runs.

`tools/r1_bias.py` lost: `--init-pick`, `--orientation`, `--iterate`, `--start-scale`, `--metric-refresh*`, `--superchains`, `--nested-limit`; nested R-hat in `score_ensemble(unc, ref, chains, draws)` and `pool_seed_scores(unc_by_seed, ref, draws)`; `refresh_warmup_for`; the refresh and superchain parts of `chees_kwargs_from`, `expected_config` and the row. The row schema name `pf-r2/v1` is unchanged. `tools/_common.py` lost the `nested_rhat_max` and `superchains` columns and the `arviz_stats_has_rhat_nested` env field. `tools/t3_cpu_nuts.score_run` lost `nested_limit`.

## Full suite
Command: `pixi run --manifest-path $MAIN/pixi.toml env PYTHONPATH=$WT python -m pytest $WT/tests -q --rootdir $WT -p no:cacheprovider`, run on head b7ce5cd.
`587 passed, 53 warnings in 847.97s (0:14:07)`

## UNVERIFIED
- GPU: not run (rule). The compiled programs changed only by dropping unused static and dynamic arguments (`iterate`, `has_t0`, `t0`); the CPU proof covers the draws, not the GPU's.
- float32: the new proof is CPU float64 only. `test_float32.py` (float64 identity against an older tree plus float32 cases) passes in the suite.
- The probe tools listed above: not run; they fail by construction.

Written by the task (fast-tier); the write step refused the file for the subtask, so the lead committed the task's text verbatim.

## Integrator (fast-tier)
Integrator suite on merged `cleanup` (9c5fe20): `587 passed, 53 warnings in 846.23s (0:14:06)`, rc 0, load 3.03 → 3.93 (1-minute average).

Step-1 verdicts (cleanup-c2 at 0c18874):
1. Scope: OK. 29 files; `pfcheeshmc/`, tests, and `tools/{r1_bias,_common,t3_cpu_nuts}.py` (nested R-hat and superchains parts); every deviation is listed above.
2. No behaviour change: OK. `tests/test_api_prune.py` run by the integrator: 25 passed in 58 s. No default value changed; the `sample_pf_chees_hmc` signature and `__init__.py` are unchanged. The tag child runs the tag's `pfcheeshmc/` (`PYTHONPATH` and `sys.path.insert(0, tree)`; checked by import: the tag tree has `_run_chees_two_phase`, this tree has not).
3. Acceptance: OK. Q3 list checked against the tag's code (defaults are the surviving branches; the 14 removed keys are exactly the Q3 list; the Kept list is intact). `test_compile_once.py` + `test_rng_stage_keys.py`: 67 passed. The accepted-key sets match what each stage reads; kept callers (notebooks, README, `bench_ladder`, `r1_bias`, `warm_stages`) pass no rejected key. Deviation 3 helpers have no remaining caller.
4. Must-fail: OK. Mutation 1 (`split(key, 5)` → `split(key, 4)`) reproduced: `[kept]` fails ("18 of 356 differ"), `[default]` passes; reverted. The rejection tests fail against the tag's code.
5. Dangling references: OK (only the probes C4 deletes, tests that assert the names are gone, `tools/wp3_chop_check.py`, and README/`chees.py` prose).
6. Code quality: OK. No new unused imports; no ladder-model special-casing.
7. SUMMARY: accurate. Finding 1 (corrected): item 1 of the deviations lists `test_compile_once` among the tests that use `without_removed`; it does not import it (the importers are `test_api_prune`, `test_benchmark_models`, `test_pm_chains`, `test_float32`, `test_start_repair`, `test_widen`).
- Finding 2 (carried to C3): five test modules (`test_benchmark_models`, `test_pm_chains`, `test_float32`, `test_start_repair`, `test_widen`) import `test_api_prune.without_removed`, and `test_api_prune` is itself a git-archive proof. Keep the helper or move it when C3 drops the git-archive proofs.
- Finding 3 (UNVERIFIED): float32 and the GPU programs. The new proof is CPU float64 only.
No finding blocked the merge.
