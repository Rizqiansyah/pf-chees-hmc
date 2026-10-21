# Review R2-WP2 (code) — lead, replacing the fast-tier review (LEDGER handover)

Verdict: **APPROVE for merge**, including the separate `5982ee7` iterate-forwarding fix.

| checklist item | result | evidence |
|---|---|---|
| covariance is `formula_1`, not a hand re-derivation | yes | `sampler.py:768` calls `bj.optimizers.lbfgs.lbfgs_inverse_hessian_formula_1` |
| passes the order choke point and the orientation step as `mc` does | yes | it replaces `covariance` before 6c (orientation) and 8 (`_metric_to_kernel_order`), the same path as `mc`. Orientation and permutation commute (P Σ⁻¹ Pᵀ = (P Σ Pᵀ)⁻¹), so their relative order is immaterial. The mutation run confirms the choke point is exercised (13 failed) |
| pool-cap branch conditional on `imm_source` | yes | `resolve_pool_cap` `:82-95`; mutation → 2 failed |
| scale 1.0 bit-identical | yes | exact-1.0 skip, plus a bitwise test, plus an 8-case comparison with `4e07b45` |
| no ridge on `pf_closed` | yes | `:768-776` (PD asserted); config `ridge` 0.0 for non-mc (`:874`) |
| suite passes | yes | task 154 passed; lead re-runs after merge |

`5982ee7` verified independently: at `4e07b45` the single-phase `chees_warmup_with_progress` call
(`sampler.py:~864-916`) has no `iterate=` argument, while the two-phase calls do (`:925`, `:976`). Its
default is `utils.py:133` `iterate="ma"`.

Note: `pf_closed` still forms the MC pool covariance (`:753`) before overwriting it. That is O(4C·d²)
work and harmless at 4C rows.
