# R2-WP3a SUMMARY (task R2-WP3a; branch r2/wp3)

## DEVIATIONS
none. (Floor band: no conflict -- `r1_bias.bias` is `max_f(mcse_f^2/var_ref_f) * 2 ln(2d)`, so for d=5 it is `2 ln(10)`; measured
ratio 0.93-0.99, see Test 1. Nothing was needed from `tools/r1_bias.py` beyond what exists: `build_model`, `load_reference`,
`score_ensemble`, `g_r1`. `r1_bias.py` is untouched on this branch.)

## Item status
| plan item | status | where |
|---|---|---|
| 4. `tools/t3_cpu_nuts.py` | DONE (smoke below; all four models' value-var names checked) | `tools/t3_cpu_nuts.py` (`sample_kwargs :64`, `score_run :114`, `grads_per_chain :126`, `run_cell :146`) |
| 5. `summarise_rows.py smallest-pass` | DONE | `tools/summarise_rows.py` (`smallest_pass :160`, `format_smallest_pass :225`, `smallest_pass_main :251`, dispatch `:274`) |
| Test 1. floor regression, via `score_ensemble` and via the T3 scorer, + mutation | DONE | `tests/test_t3_cpu.py:51`, `:61` |
| smallest-pass tests (synthetic rows) | DONE | `tests/test_summarise_rows.py` (12 tests) |
| Review checklist: T3 driver calls `score_ensemble` | DONE | `t3_cpu_nuts.py:116` (`rb.score_ensemble`), `:117` (`rb.g_r1`); spy test `test_t3_cpu.py:69` |
| Review checklist: `include_transformed` on, value-var names scored | DONE | `t3_cpu_nuts.py:70`, `:92-111`; names from `model.value_vars` in `main` |
| Review checklist: gradients include tuning | DONE | `t3_cpu_nuts.py:126-141` (raises if `warmup_sample_stats` absent); test `test_t3_cpu.py:107` |
| Review checklist: floor test fails if the square is removed | DONE (shown below) | |
| Plan items 1, 2, 3, 6 and tests 2, 3; "no harness default differs" checklist item | OUT OF SCOPE (task 3b) | `r1_bias.py` not edited |

## Item 4 -- t3_cpu_nuts.py
- Sampling: `bl.REAL_PM_SAMPLE` (`bl` = `tools/bench_ladder.py`, `pm.sample` captured before the notebook stubs);
  `sample_kwargs()` fixes chains=32, cores=32, target_accept=0.8, discard_tuned_samples=False,
  idata_kwargs={"include_transformed": True}.
- No scorer of its own: `score_run()` calls `r1_bias.score_ensemble` and `r1_bias.g_r1`; names from
  `model.value_vars`; reference from `r1_bias.load_reference`.
- Gradients: `grads_per_chain()` = (sum n_steps over warmup_sample_stats + sample_stats) / chains, and raises
  if `warmup_sample_stats` is absent.
- Row = pf-r1/v1 fields (via `_common.new_row`) + `schema "pf-r2/v1"`, `step "M5"`, `pooled false`, `effective null`,
  `tune`, `draws`, `target_accept`, `cores`, `core_seconds`, `grads_per_chain_total`, `g_r1`, `loadavg_start/end`.
  b2/floor/argmax use the r1_bias names `b2_max` / `b2_floor` / `b2_argmax`.
- Observation for the lead: C*D = 32*draws, so draws=50 gives `g_r1 = "not eligible (C*D < 2000)"` for every
  seed (g_r1 unchanged, as instructed). M5 cells with draws=50 can never pass.
- Observation: `os.fork()` JAX-multithreading RuntimeWarning is printed by pm.sample (r1_bias imports jax). Same as
  bench_ladder's CPU arm; chains do not use JAX.

### Smoke
Command (from `~/pf/r2-wp3`, CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1, load avg 1.26 at start):
`pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$PWD" python tools/t3_cpu_nuts.py --models 8schools --tune 200 --draws 50 --seeds 1 --out ~/pf-runs/r2-wp3a-smoke/8schools_t200_d50_s1.jsonl`

Row (`~/pf-runs/r2-wp3a-smoke/8schools_t200_d50_s1.jsonl`): model 8schools, chains 32, tune 200, draws 50, seed 1,
wall_s 3.34, core_seconds 12.0, grads_per_chain_warmup 2131.97, grads_per_chain_sampling 372.94,
grads_per_chain_total 2504.91, divs_per_1k 0.0, warmup_divs 99, d 10, b2_max 0.012107 (argmax theta[1] x^2),
b2_floor 0.014886, g_r1 "not eligible (C*D < 2000)", target_accept 0.8, scored_vars [mu, tau_log__, theta],
git_sha 4e07b45, schema pf-r2/v1, step M5, cpu_contended false, loadavg_start [1.26,1.30,0.93],
loadavg_end [1.26,1.30,0.93] (kernel average, 3 s run).

## Test 1 -- floor regression (`tests/test_t3_cpu.py`)
- Tests: `test_floor_is_squared_through_score_ensemble[0,1,2]` (`:51`), `test_floor_is_squared_through_the_t3_scorer[0,1,2]`
  (`:61`, from an InferenceData through `t3.score_idata`), `test_t3_scorer_calls_score_ensemble_and_g_r1_not_its_own_scorer`
  (spy on `rb.score_ensemble` / `rb.g_r1`), plus driver pieces: sampler settings, value-var names from the posterior
  (tuned draws not read; missing name -> KeyError mentioning include_transformed), gradients include tuning and are exact,
  and a real 2-chain toy `run_cell` that scores `s_log__`.
- Band: +-30% of `2 ln(10)/(C*D)` on iid N(0,1), (C,D,d) = (256, 40, 5), reference 80,000 iid rows. No deviation: the
  formula in `r1_bias.bias` is `max_f(mcse_f^2/var_ref_f) * 2 ln(2d)`, and with d=5 that is exactly `2 ln(10)`.
  Measured ratio over 10 seeds (probe, same instrument): 0.926-0.989 (floor 4.16e-4 - 4.45e-4 vs expected 4.497e-4).
- The unsquared T3 value is `2 ln(10)/sqrt(C*D)` = 0.0455, i.e. sqrt(C*D) = 101x the correct floor (the plan's "about
  sqrt of that" is loose; either way it is far outside the band, and the test asserts that).
- Mutation (removes both squares in `r1_bias.bias`, i.e. the run-1 `MCSE/sd` floor; reverted with `git checkout`, `r1_bias.py`
  is unmodified in the branch):
  `sed -i 's|... ** 2 / den|... / np.sqrt(den)|' tools/r1_bias.py` (both lines), then
  `pytest tests/test_t3_cpu.py` gives `6 failed, 5 passed`:
  `E assert 0.04410858822391871 <= (1.3 * np.float64(0.0004497236509753996))` (3 x score_ensemble, 3 x t3 scorer; the other
  5 tests do not depend on the floor).
- Unmutated: `11 passed in 5.22s`.

## Item 5 -- `summarise_rows.py smallest-pass`
Usage: `python tools/summarise_rows.py smallest-pass PATH_OR_GLOB... [--arm A] [--models a,b] [--seeds 1,2,3] [--max-w N]
[--mode auto|r1|m5] [--warm-only] [--cells]`. The plain `summarise_rows.py PATH [model ...] [--cold]` mode is unchanged
(re-run on `results/runs/WP3__smoke.jsonl`).
- Verdict is recomputed with `r1_bias.g_r1(b2_max, b2_floor, chains*draws)` for every row kind (lazy import); a stored `g_r1`
  that disagrees is counted in the `[audit]` lines, not trusted.
- R1 mode: grid = `warmup`; smallest W with `pass` on EVERY expected seed (seeds seen for that model/arm, or `--seeds`); a W
  missing a seed is `incomplete`, `not resolvable` / `not eligible` are not passes. Prints median `grads_per_chain_total`
  (falls back to warmup+sampling), per-seed b2/floor, pooled b2/floor (diagnostic) at that W, or `none <= max W N`.
- Pooled rows (`pooled: true` or `seed: "pooled"`) never enter the every-seed test and are not seeds.
- M5 mode (auto-detected from `step == "M5"` / `tune` without `warmup`): grid = (tune, draws), cost = tune+draws, ties by
  median `wall_s` then tune; prints median `wall_s`, `core_seconds`, `grads_per_chain_total`, per-seed b2/floor, or
  `none <= max cost N`.
- Choices to confirm: duplicate cells (same model/arm/W/seed, e.g. cold and warm reps) -> last row in file order wins (counted in
  the audit; `--warm-only` drops cold rows first); error/VOID/no-b2 rows are excluded and counted.
- Real-file check: `smallest-pass results/runs/T1__r1_sweep.jsonl --arm int:shipped` gives 8schools W=100 (grads/chain 1320),
  gev W=100 (526), logit_1000x25 and radon_noncentered `none <= max W 200`.

### Other models, value-var coverage (4 chains, tune 40, draws 20, throw-away check, not a measurement)
`~/pf-runs/r2-wp3a-smoke/names_check_c4.jsonl`: gev scored `[μ, ξ_interval__, σ_log__]` (d=3); logit_1000x25 `[b]` (d=25);
radon_noncentered `[alpha_raw, beta_raw, mu_alpha, mu_beta, sigma_alpha_log__, sigma_beta_log__, sigma_y_log__]` (d=175);
8schools `[mu, tau_log__, theta]` (d=10). No errors; the transformed coordinates run 1 dropped are all scored.

## Full suite (forced CPU)
`pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider`
(CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1), at ec81834:

    148 passed in 176.64s (0:02:56)

(= 125 baseline + 11 in `test_t3_cpu.py` + 12 in `test_summarise_rows.py`.)

## UNVERIFIED
- The full M5 grid (32 chains, tune 200-1000, draws 50-400) was not run; only the 8schools 32-chain smoke above.
- `core_seconds` = delta of `RUSAGE_SELF + RUSAGE_CHILDREN` CPU time across the `pm.sample` call. In the smoke (12.0 core-s for
  3.34 s wall) the children are clearly counted, but it was not cross-checked against `perf`/`time` at long runs.
- `cold` is written `false` for every NUTS row. The pytensor C-module cache is on disk, so a never-compiled model's first row can
  carry a one-off compile in `wall_s`; run each model once untimed before timing (also in the module docstring).
- `loadavg_start`/`loadavg_end` are the kernel 1/5/15-min averages; on runs of a few seconds they are identical.
- pm.sample prints a JAX-multithreading `os.fork()` RuntimeWarning (r1_bias imports jax). The chains do not use JAX, the smoke
  completed, and `bench_ladder`'s CPU arm behaves the same.
- The `XLA_FLAGS` in this shell (`--xla_force_host_platform_device_count=100`) is copied into rows' `xla_flags`; it does not
  affect NUTS.
