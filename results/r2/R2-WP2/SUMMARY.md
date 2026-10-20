# R2-WP2 SUMMARY — closed-form Pathfinder covariance, pool sizing, start scaling, config attrs

Written by the lead from the task's final report. This replaces the task's stale milestone
stub; the subtask harness refused the final write. Commits:
- `669086c` `pf_closed` and pool sizing;
- `5982ee7` iterate-forwarding fix (finding 1);
- `28f6dd2` config record;
- `5e4ec8a` start scale;
- `9860506` tests.

Task suite (forced CPU, own output, at `9860506`): `154 passed in 268.44s (0:04:28)`. The new
`tests/test_pf_closed_form.py` alone: `29 passed in 134.52s (0:02:14)`.

## DEVIATIONS
None from the ⟦VERBATIM⟧ block. Two findings:
1. **Run-1 defect: single-phase `_run_chees` never forwarded `iterate`.** With the metric refresh off,
   `iterate="raw"` was accepted and silently ran the default `ma`. Only the two-phase path forwarded
   it (`sampler.py:925,976` at `4e07b45`). The task confirmed this with a spy on
   `chees_warmup_with_progress`: requested raw → applied `<default ma>` before, `raw` after.
   - Fixed with one line plus a test, in its own commit `5982ee7`. The default `ma` is unchanged.
   - Consequence: every run-1 "covariance, `raw`" row without a refresh ran `ma`. See LEDGER, E13.
2. The rule "`imm_source != mc` → pool = 4C" also covers `fisher_lowrank`, whose pool shrinks from
   `max(4C, 4d²)` to 4C. Implemented as specified; the accuracy effect is unmeasured.

## Items (all `pfcheeshmc/sampler.py` unless noted)
| item | status |
|---|---|
| 1. `pf_closed` = `bj.optimizers.lbfgs.lbfgs_inverse_hessian_formula_1(pf_state.alpha, pf_state.beta, pf_state.gamma)` | DONE `:768`. Same orientation step (6c) and `_metric_to_kernel_order` choke point (8) as `mc`; no ridge; `eigvalsh(...).min() > 0` asserted with a ValueError |
| 2. pool sizing | DONE `resolve_pool_cap` `:82`: `None` → `default_pool_cap` for `mc`, `4*C` otherwise; `0` uncapped; explicit integers as-is |
| 3. `imm_source` default unchanged | DONE (`"mc"`) |
| 4. `pf_start_scale` | DONE `_pathfinder_mean` `:252` (`position + alpha*grad + beta@(gamma@(beta.T@grad))`, the `mu` at `blackjax/optimizers/lbfgs.py:429`) and `_scale_starts` `:264`, applied in step 5b; exactly 1.0 skips the transform; non-positive or non-finite scales raise, as does a scale producing a non-finite start |
| 5. `idata.attrs["config"]` | DONE `PfCheesHMC._config_record` (key list in its docstring). Holds resolved values for `imm_source`, `imm_orientation`, `imm_order`, `ridge` (0.0 for non-mc), `pf_draws`, `pf_pool_cap` (0 = uncapped), `pf_pool_rows`, `pf_start_scale`, `iterate`, `adam_lr`, `target_accept`, `max_leapfrog_steps`, `eps0_probes`, `eps0`, `frozen_step_size`, `frozen_L_nominal`, `seed`, `stage_seeds`, `jax_enable_x64`, `num_warmup`, `warmup_steps_used`, `chains`, `num_draws`, plus `metric_refresh*` (null when no refresh ran) |

## What the existing `lbfgs_lowrank` arm computes
The same matrix as `pf_closed`. Test 1 shows `D(I + U diag(λ−1) Uᵀ)D` with D = diag(√α) equals
`diag(α) + βγβᵀ = formula_1` to a relative Frobenius error of 2.5e-16 to 8.4e-16 on all four models.
It is written through an extra QR and a 2m×2m `eigh`, and has no PD assert. Every run-1
`lbfgs_lowrank` number is void (broken orientation, E12), so M1's `pf_closed` arm is the first valid
measurement of this matrix.

## Tests (`tests/test_pf_closed_form.py`, 29)
- **T1 algebra, 4 models.**
  - `formula_1` equals the low-rank expansion to 1e-16.
  - The sampler's returned metric equals `P·formula_1·Pᵀ`.
  - Min eigenvalue is above 0 everywhere (radon 5.0e-4).
  - The kernel permutation is non-identity on gev and radon.
- **T2 MC consistency, 8schools and gev, N=50,000.**
  - Relative Frobenius error is 1.57e-2 and 3.2e-3, against plan bounds of 0.27 and 0.086.
  - The draw mean is within 5 SE of `_pathfinder_mean`, which equals `position + formula_1 @ grad` to 1e-10.
- **T3 G-ORDER.**
  - The run-1 non-sorted fixture passes in both orientations.
  - End-to-end anisotropic mixing passes with `pf_closed`.
  - Two teeth tests mutate the choke point to identity and require failure.
  - A PD-refusal test is included.
- **T4 pool rows.**
  - At C=1,024, d=40: `pf_closed` 4096, `lbfgs_lowrank` 4096, `mc` 6400.
  - `0` and explicit integers are checked for all sources.
- **T5 start scale.**
  - 1.0 gives bit-identical starts and metric.
  - 2.0 doubles the deviation from `mu_pf` and its per-coordinate rms to 1e-12, with the covariance untouched.
  - The adapt_diag path and validation are also tested.
- **Config.**
  - Defaults are filled in, with no None where a default applied.
  - Moved knobs match what `chees_warmup_with_progress` and `estimate_covariance` received.
  - The iterate regression has its own test.
- The file sets `pytensor.config.blas__ldflags` at import. The repo path contains a space, and
  without it logit fails to link under pytest.

## Mutation evidence
- **Choke point to identity:**
  `pytest tests/test_pf_closed_form.py tests/test_metric_orientation.py -k "kernel_order or g_order or mixes or hands_the_kernel or every_imm_source or covariance_orientation"`
  → `13 failed, 4 passed`.
  - The 4 passes are the sorted-name models (identity permutation) and the 2 teeth tests.
  - Excerpt: `pf_closed/covariance: not in kernel order / Mismatched elements: 16 / 16 (100%)`.
- **Pool-cap branch made unconditional:** `2 failed` (`pf_closed: pool rows [6400], expected 4096`).
- **Bit-identity against `4e07b45`** (loaded as a separate module), `_run_pathfinder` C=16, seed 4,
  non-sorted model:
  - cases: default, explicit 1.0, `mc` + `pf_pool_cap=0`, `lbfgs_lowrank` + `pf_pool_cap=64`;
  - each in float32 and float64;
  - starts and metric are bit-identical in all 8. The script was not committed.

## UNVERIFIED
- No full-sampler `pf_closed` run at C=1,024 or on the GPU. PD was checked on four fits, one seed each.
- `import pymc` enables x64, so production is float64. float32 is untested.
- `metric_refresh_lmax_whitened` (R2-WP1) is in the top-level attrs loop, not in `config`.
- `fisher_lowrank` accuracy at a 4C pool is unmeasured.
