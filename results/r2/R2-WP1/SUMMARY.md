# R2-WP1 SUMMARY — trajectory-length re-seed fix

Written by the lead from the task's final report. The subtask harness refused the task's
write of this file. Commits: `9742363` (formula, clip, attr, tests), `9942ac0` (smoke rows,
generalized-eigenproblem test). Task suite (forced CPU, own output): `130 passed in 178.09s (0:02:58)`
(125 baseline, plus 5 net).

## DEVIATIONS
1. **Smoke acceptance missed on `logit_1000x25`:** re-seed L / no-re-seed L = 3.07 (accept band
   0.5–2×). 8schools = 1.07, inside the band. **Not an A6 BLOCK**, which needs > 3× on ≥ 2 models.
   See "Reading" below; the lead logged it as a contradiction (A6) and continued.
2. Added a public helper `whitened_lambda_max` (`ensemble_metric.py`), so the sampler builds the
   d×d whitened matrix twice per refresh (negligible).
3. The clip uses plain-float `min/max`, not `jnp.clip` (avoids float32 rounding of T0 when x64 is off).
4. The existing non-PD test now passes `draws` (new signature).
5. Extra tests: signature pin, shape mismatch, and a `scipy.linalg.eigh(S, B)` generalized-eigenvalue
   check on a dense B that is neither S nor diag(S).
6. New files: `results/r2/R2-WP1/{smoke_driver.py,run_smoke.sh,smoke_rows/}`. The driver wraps
   `_run_chees` only to log T0 and lmax to a sidecar, because harness rows don't carry them.
   `tools/r1_bias.py` is unchanged.

## Formula implemented (⟦VERBATIM⟧ block, plan Part B R2-WP1)
`T0_B = (pi/2) * sqrt( lambda_max( L^-1 S L^-T ) )`, with `B = L L^T` = imm_B (Cholesky) and
`S` = the ddof=1 sample covariance of the same `draws` the estimator consumed. `lambda_max` comes from
`jnp.linalg.eigvalsh` of the symmetrised d×d matrix, with L⁻¹ applied by two `solve_triangular`
calls and no inverse.
- Code: `pfcheeshmc/ensemble_metric.py:164-220` (S at `:142-150`, solves at `:191-193`, eigvalsh at
  `:194-195`).
- Clip `T0_B = clip(T0_B, eps_B, max_leapfrog_steps*eps_B)`: `pfcheeshmc/sampler.py:986`.
  `max_leapfrog_steps` is always set, by `setdefault(...,1000)` at `sampler.py:848`.
- Same pool: `draws` is built at `sampler.py:946` and passed to both `estimate_covariance` and
  `whitened_lambda_max` / `reseed_trajectory_length`.
- Attrs: `metric_refresh_lmax_whitened` is written at `sampler.py:1012` and `:1453-1454`;
  `metric_refresh_T0` is kept. The two-phase test asserts both, and that T0 == clip((π/2)√lmax).

## Items
| item | status |
|---|---|
| change (formula, clip, signature, attr) | DONE (above) |
| test 1 unit invariance | DONE `tests/test_ensemble_metric.py:86` |
| test 2 identity, B=S → π/2 | DONE `:114` (dense_shrunk shrink=0 matches ddof=1 to 1e-14) |
| test 3 correlation, B=diag(S) → (π/2)√(1+\|ρ̂\|) | DONE `:132` (sample_diag matches ddof=1 to 1e-14) |
| test 4 non-PD rejected | DONE `:175` |
| test 5 delete `test_reseed_matches_closed_form` | DONE |
| mutation proof | DONE (below) |
| CPU smoke | DONE; acceptance missed on logit (DEVIATION 1) |

**Mutation.** The old body `pi/2*sqrt(1/eigvalsh(cov_B).min())` was placed in the new signature, then
`pytest tests/test_ensemble_metric.py -k "invariant_to_coordinate_units or pi_over_two or tracks_pool_correlation"`
was run → `3 failed, 14 deselected in 5.15s`:
- T1 `T0 moved with coordinate units: 2.01293425883 -> 112.110278941 (rel 5.469e+01)`;
- T2 `T0 3.14462019166 != pi/2 1.57079632679`;
- T3 `T0 0.391366102474 != (pi/2)sqrt(1+|rho|) 1.95523112205`.
The file was then restored (checked with `cmp`).

## CPU smoke (C=64, D=10, W=50, `--orientation covariance --iterate ma`, seeds 1/2/3, CPU, no `--timed`)
Command: `bash results/r2/R2-WP1/run_smoke.sh`. Rows are in `results/r2/R2-WP1/smoke_rows/`.

| model | arm | L | mean L | eps | T0 | lmax |
|---|---|---|---|---|---|---|
| 8schools | re-seed on | 12/9/10 | 10.3 | .292/.300/.277 | 1.894/1.856/1.869 | 1.453/1.396/1.416 |
| 8schools | re-seed off | 10/10/9 | 9.7 | .305/.263/.286 | — | — |
| 8schools | no refresh | 21/19/17 | 19.0 | .408/.451/.391 | — | — |
| logit_1000x25 | re-seed on | 30/35/24 | 29.7 | .159/.151/.227 | 3.646/3.919/4.096 | 5.387/6.226/6.800 |
| logit_1000x25 | re-seed off | 10/9/10 | 9.7 | .194/.162/.191 | — | — |
| logit_1000x25 | no refresh | 9/12/9 | 10.0 | .548/.412/.503 | — | — |

**Reading (lead).**
- With `sample_diag`, B = diag(S), so lmax is the top eigenvalue of the pool *correlation* matrix
  (5.4–6.8 on logit). A diagonal metric cannot absorb that correlation, so the ChEES-optimal T grows
  by √lmax.
- In integration time T = eps·L, the re-seeded arm lands at T ≈ 4.8 / 5.3 / 5.5. That equals the T the
  no-refresh arm reached on its dense Pathfinder metric (4.9 / 4.9 / 4.5).
- The no-re-seed control stays at T ≈ 1.5–1.9. Phase B starts it at L=1 and gives it only 25 steps
  to grow.
- So the 3× L ratio reflects the diagonal metric's conditioning, and the control under-growing. It
  is not the run-1 unit defect, whose signature was 8–10× L, with T unrelated to the no-refresh T.

UNVERIFIED (task): whether S (25 phase-A steps from Pathfinder starts) still carries unconverged
inter-chain spread. No GPU, C=1,024 or W≥100 run was made.

Gotcha: import `tools/r1_bias` before `pfcheeshmc`/pytensor. It sets `PYTENSOR_FLAGS` at import, and
logit fails in `_pymc_adapt_diag_start` without them.
