# R2-WP2 SUMMARY (work in progress; items not yet DONE are PENDING)

## DEVIATIONS
none so far (see "Out-of-scope finding" below).

## Item status
| # | item | status | where |
|---|---|---|---|
| 1 | `imm_source="pf_closed"` | DONE (milestone 1) | `pfcheeshmc/sampler.py` step 6b |
| 2 | pool sizing by `imm_source` | DONE (milestone 1) | `resolve_pool_cap`, called from `_run_pathfinder` step 4 |
| 3 | `imm_source` default unchanged ("mc") | DONE | `kwargs.get("imm_source", "mc")` |
| 4 | `pf_start_scale` | PENDING | |
| 5 | `idata.attrs["config"]` | PENDING | |
| T1-T5 | tests in `tests/test_pf_closed_form.py` | PENDING | |
