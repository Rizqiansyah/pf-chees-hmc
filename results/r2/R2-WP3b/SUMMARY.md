# R2-WP3b SUMMARY — harness passthrough, effective guard, pooled seeds

Written by the lead from the task's final report (subtask harness refuses report files).
Commits `35c0819`, `cf98b0b`, `d87911c`, `e38723f`, `7090916`, on top of merge `ab858dc`. Task suite
(forced CPU, own output): `191 passed, 4 warnings in 286.75s (0:04:46)`. `pfcheeshmc/` and
`tools/_common.py` untouched.

## DEVIATIONS
1. **The no-flag arm is now the sampler's default arm.** The harness passes only `draws=W`, and the
   sampler resolves `pf_draws=4000` (pool capped by the pool rule), `imm_orientation=covariance`,
   `imm_source=mc`, `iterate=ma`. Run 1 injected `pf_draws=1`, `precision` and `ma`. To reproduce one
   draw per chain, pass `--pf-draws 1`.
2. **Passthrough also covers these.** `--init-pick` (default None; the sampler default is also
   `stratified`), `--metric-refresh-reseed` (tri-state, default None), and `--metric-refresh`
   (default `"none"`, which equals the sampler default and is guarded on every cell).
3. **`gev_divergence_probe.py --pf-draws` default changed from 1 to None** (sampler default), so M2c
   dispersion matches the rows it explains. Pass `--pf-draws 1` to reproduce run-1 §3j.
4. **Config guard aborts the run.** A `ConfigMismatch` is written as an error row, then re-raised.
5. **Files outside the list.**
   - `tools/wp3_summarize.py` skips pooled rows: `seed="pooled"` would break its tuple sort.
   - `tools/gev_divergence_probe.py` gets the M2c flag (requested).
   - `tools/summarise_rows.py`'s default table excludes pooled rows and prints an audit line.
6. `metric_refresh_T0` / `metric_refresh_lmax_whitened` are present in every pf-r2 row, null when absent.
7. Pooled rows add `pooled_seeds` and `g_r1_pool` (the verdict on the pooled ensemble, labelled diagnostic).

## Items
| item | status |
|---|---|
| 1. defaults pass through + `effective` + disagree → raise | DONE. Flags at `tools/r1_bias.py:812-857`; `pathfinder_kwargs_from` `:450`; `chees_kwargs_from` `:469`; `effective_from` `:488`; `expected_config` `:494`; `check_effective` `:523` (a missing key counts as a mismatch); guard calls `:598`, `:780`; abort `:709` |
| 2. `--pool-seeds` | DONE. `pool_seed_scores` `:715` concatenates seeds along chains and scores with chains=S·C; `run_pooled` `:728` writes `pooled: true`, `seed: "pooled"`. Holds only (C,D,leaves) float64 draws, never idata |
| 3. `--metric-refresh-warmup` | DONE `:844`. None → max(2, W//2), clamped to [2, W−2] as the sampler does, and recorded in `effective` |
| 6. `grads_per_chain_total` docstring | DONE, **flagged**: `:20-52`. The row value is warmup (phases A+B) + sampling `num_integration_steps`, per chain. It does **not** include the ε-search probes (32, C-independent, uncounted) or the `fisher_diag` refresh-pool gradients (≤ `min(W_A,20)` per chain). `pf_grads_total` is null in every row |
| schema `pf-r2/v1` | DONE `SCHEMA_R2` `:63`; `effective`, `pooled`, `step` (`--step`) |
| M2c probe flag | DONE `tools/gev_divergence_probe.py:67` `--start-scale` (forwards `pf_start_scale`; output records `start_scale` and `effective`) |

## Tests (`tests/test_r1_bias.py:430-608`)
- **Test 2 passthrough.** With no flags, `pathfinder_kwargs == {}` and `chees_kwargs == {"draws": 6}`;
  `effective.imm_orientation == "covariance"`; `effective.iterate` and every non-result key equal a
  direct no-kwargs sampler call.
- **Contradiction test.** `_config_record` monkeypatched to `iterate="ma"` while the harness asks for
  `raw` → `ConfigMismatch` "iterate: asked 'raw', sampler ran 'ma'", and the error row is on disk.
- **Test 3 pooled seeds.** 3 identical copies: b² 0.009087542311288 vs 0.009087542311289; floor
  0.0057694 → 0.0019148 (÷3.013). Plus a 4-independent-seed test and a `run_pooled` → `smallest_pass`
  test.
- **Probe.** `--start-scale 2` gives exactly 2× `median_ratio`.
- **Mutations.**
  - Re-inject `iterate="ma"` → 2 tests fail (`{'draws': 6, 'iterate': 'ma'} == {'draws': 6}`).
  - `check_effective` made a no-op → `DID NOT RAISE`.
  - Pool only `seeds[:1]` → `assert 64 == 192`.

## Smoke (CPU, `~/pf-runs/r2-wp3b-smoke/`; 8schools C=64 D=10 W=25 `--pool-seeds --step SMOKE`)
`effective` for the default rows: `imm_source=mc`, `imm_orientation=covariance`, `iterate=ma`,
`pf_draws=4000`, `pf_pool_cap=400`, `pf_pool_rows=400`, `pf_start_scale=1.0`, `init_pick=stratified`,
`metric_refresh=none`.

| row | chains | b² | floor |
|---|---|---|---|
| seed 1 | 64 | 0.06694 | 0.02659 |
| seed 2 | 64 | 0.01288 | 0.02649 |
| pooled | 128 | 0.00673 | 0.01297 |

With `--iterate raw`: `effective.iterate == "raw"` and the guard passed. C·D = 640, so no G-R1 verdict
at this shape.

UNVERIFIED: the `--timed`/GPU path; `--legacy` (effective null, guard skipped); `--pool-seeds` memory
at C=1,024, d=175 (about 14 MB per seed by arithmetic); the gradient accounting, which is from code
reading.
