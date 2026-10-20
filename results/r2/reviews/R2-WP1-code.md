# Review R2-WP1 (code) — lead, replacing the fast-tier review (LEDGER handover)

Verdict: **APPROVE for merge.**

| checklist item | result | evidence |
|---|---|---|
| ⟦VERBATIM⟧ block matches code | yes | `ensemble_metric.py` whitened_lambda_max: `cholesky(B)`, `solve_triangular(L,S)`, `solve_triangular(L,A.T).T` = L⁻¹SL⁻ᵀ, symmetrised, `eigvalsh`; T0 = π/2·√lmax |
| test 1 fails on the old formula | yes | task mutation run: T1 rel 5.469e+01 (SUMMARY) |
| no explicit `inv` | yes | `grep -n 'inv(' ensemble_metric.py` empty; sampler `:742` `inv` is the pre-existing `imm_orientation="precision"` path, untouched |
| pool S = estimator's draws | yes | `sampler.py:946` `draws` feeds both `estimate_covariance` and the re-seed |
| clip applied | yes | `sampler.py:986` min/max to [eps_B, max_L·eps_B]; max_L from setdefault at `:848` |
| attrs written | yes | `metric_refresh_lmax_whitened` in params_B and the attrs loop |
| suite passes | yes | task: `130 passed in 178.09s`; re-run by lead after merge |

Algebra check: A = L⁻¹S; Aᵀ = SL⁻ᵀ (S symmetric); L⁻¹(SL⁻ᵀ) = L⁻¹SL⁻ᵀ, which is symmetric, so the
final transpose is a no-op. Smoke miss on logit: accepted as a logged contradiction (SUMMARY "Reading").
The integration time T matches the no-refresh arm's.
