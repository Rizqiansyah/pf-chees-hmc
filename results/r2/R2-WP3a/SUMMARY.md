# R2-WP3a SUMMARY (task R2-WP3a; branch r2/wp3) -- IN PROGRESS

## DEVIATIONS
none so far.

## Item status
| plan item | status | where |
|---|---|---|
| 4. `tools/t3_cpu_nuts.py` | DONE (smoke below) | `tools/t3_cpu_nuts.py` |
| 5. `summarise_rows.py smallest-pass` | PENDING (code written, untested, not committed) | `tools/summarise_rows.py` |
| Test 1. floor regression (score_ensemble + t3 scorer) + mutation | DONE | `tests/test_t3_cpu.py:50` (score_ensemble), `:62` (t3 scorer) |
| Review-checklist items (T3 driver, floor) | PENDING | |

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
- Tests: `test_floor_is_squared_through_score_ensemble[0,1,2]` (`:50`), `test_floor_is_squared_through_the_t3_scorer[0,1,2]`
  (`:62`, from an InferenceData through `t3.score_idata`), `test_t3_scorer_calls_score_ensemble_and_g_r1_not_its_own_scorer`
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
