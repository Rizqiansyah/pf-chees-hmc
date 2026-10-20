# R2-WP2 SUMMARY (work in progress; items not yet DONE are PENDING)

## DEVIATIONS
none so far. One out-of-scope fix is in its own commit (see "Out-of-scope finding").

## Out-of-scope finding (separate commit 5982ee7, revertable alone)
Single-phase `_run_chees` did not forward `iterate` to `chees_warmup_with_progress`: with `metric_refresh` off, `chees_kwargs={"iterate": "raw"}` ran `ma`. Reproduced by spying on `chees_warmup_with_progress` (`requested raw -> applied ['<default ma>']` before, `['raw']` after). Only the two-phase path forwarded it. Consequence: every run-1 row made with `--iterate raw` and no metric refresh (T1 `covraw`, R1 'raw' passes) actually ran `ma`; run-2 M2a/M3 `raw` arms would have been `ma` too. The default (`ma`) is unchanged, so default runs are bit-identical. The config record needed the applied value, hence the fix.

## Item status
| # | item | status | where |
|---|---|---|---|
| 1 | `imm_source="pf_closed"` | DONE (milestone 1) | `pfcheeshmc/sampler.py` step 6b |
| 2 | pool sizing by `imm_source` | DONE (milestone 1) | `resolve_pool_cap`, called from `_run_pathfinder` step 4 |
| 3 | `imm_source` default unchanged ("mc") | DONE | `kwargs.get("imm_source", "mc")` |
| 4 | `pf_start_scale` | DONE (milestone 3) | `PfCheesHMC._pathfinder_mean`, `_scale_starts`; applied in `_run_pathfinder` step 5b; exactly 1.0 skips the transform |
| 5 | `idata.attrs["config"]` | DONE (milestone 2) | `PfCheesHMC._config_record` (key list in its docstring), written at the end of `sample()`; per-stage values recorded in `self._resolved` by `_run_pathfinder`, `_run_step_size_tune`, `_run_chees`, `_run_dynamic_hmc` |
| T1-T5 | tests in `tests/test_pf_closed_form.py` | PENDING | |
