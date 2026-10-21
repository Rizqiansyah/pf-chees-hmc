# WP2 code-correctness review (vendored ChEES adapter) — run by the lead

The two reviewer tasks for this run died with a broken tool layer (`bash: not found`), so this
review is executed directly. Every claim below is from a file opened or a command run in this
session; the branch's own 9-test suite (`tests/test_chees_vendored.py`) was re-run: **9 passed
in 14.9 s** forced CPU.

**Subject:** `wp/wp2` — `pfcheeshmc/chees.py` (441 lines), claimed to be BlackJAX 1.7.1
`chees_adaptation.base` verbatim plus exactly five deltas; `pfcheeshmc/utils.py` wiring
(`WarmupInfo`); `tools/wp2_lgrowth.py` (measurement harness).

## Verdicts

1. **Verbatim claim — CONFIRMED.** The vendored `base` reproduces the installed
   `blackjax/adaptation/chees_adaptation.py` `base` (same whitening derivation, same
   `weighted_empirical_mean`, same clip bounds, same Halton/jitter contract). The five declared
   deltas are present and no undeclared semantic difference was found. The only consumed-line
   difference between the two iterate modes is the trajectory-length line:
   `chees.py:367` `new_trajectory_length = jnp.exp(new_log_trajectory_length)` (raw) vs
   `chees.py:370` `= jnp.exp(new_log_trajectory_length_ma)` (ma = installed behaviour). The
   installed file consumes the moving average (`chees_adaptation.py:493`
   `jnp.exp(new_log_trajectory_length_ma)`), so `ma` matches it and `raw` is the one-line paper fix.

2. **Bit-for-bit `ma` — CONFIRMED, and it is a full-state check.**
   `test_ma_mode_is_bit_for_bit_upstream` runs 64 steps of the installed `base` and the vendored
   `base(iterate="ma")` on a fixed input stream and calls `_assert_states_equal` each step, which
   compares **every pytree leaf** of `ChEESAdaptationState` (step_size, both moving averages,
   trajectory_length, da_state, optim_state incl. Adam m/v buffers, random_generator_arg, step)
   with exact `jnp.equal`. It also pins `t0=None` init and the `optim=None`→`adam(0.05)` default.
   9/9 pass on forced CPU x64. The default therefore is provably the shipped behaviour.

3. **raw/ma is one line; the trace records the consumed value — CONFIRMED.** The test asserts
   `trace.trajectory_length == state.trajectory_length` (the consumed value, not the MA), and
   `trace.step/step_size` match the state. The trace is built inside `update` and returned as a
   `lax.scan` output, so it costs nothing extra (point 5).

4. **`WarmupInfo` contract change — CONFIRMED safe, one NIT.** Every consumer uses attribute
   access or `getattr(..., "info", ...)`: the sampler seam `sampler.py` (`hasattr(chees_info,
   "info")`), `tools/r1_bias.py:335`, and the tests (`.info`/`.trace`). A BlackJAX `HMCInfo` has
   no `info` field, so the `hasattr` discriminator cannot misfire today. **NIT:** an
   `isinstance(x, WarmupInfo)` check would be the clearer, more robust contract than a string
   `hasattr`.

5. **Trace costs nothing — CONFIRMED.** `one_step` returns `(new_states.position, info, trace)`
   as the scan's `ys`; `trace_hist` is a scan output, not an eager per-step materialisation, so
   the warmup wall is unchanged.

6. **Frozen values agree with the kernel — CONFIRMED.** The sampling stage
   (`_run_dynamic_hmc`) consumes `chees_parameters["step_size"]` = `exp(log_step_size_ma)` and
   `integration_steps_fn(..., traj_len_adj)` with `traj_len_adj = exp(log_T_ma - log_eps_ma)`.
   `frozen_L_nominal` reports exactly `traj_len_adj`, so the row schema and the sampling kernel
   agree on the frozen (ε, L). The *last warmup step* uses the current (non-frozen) iterate —
   inherent to ChEES, not a defect; the freeze-at-moving-average is the paper convention.

7. **Harness traps — CONFIRMED absent.** `tools/wp2_lgrowth.py` builds models via
   `bench_ladder.model_specs()[name]()` (the spec loader, *not* the notebook loader that stubs
   `sample_pf_chees_hmc`) and drives the warmup entry point directly, so the stub-leakage class of
   bug cannot hit it. Cold/warm are labelled per `rep` (rep 0 cold, rep 1 warm) and rows carry
   `wall_s`, `warmup_divs`, and the `L_at` checkpoints. One **NIT**: the covariance arm is built as
   `imm_cov = inv(imm_prec)` then symmetrised — a direct `imm_orientation="covariance"` call would
   avoid the double inversion.

## BLOCKING
None.

## NON-BLOCKING
- Prefer `isinstance(x, WarmupInfo)` over `hasattr(x, "info")` at the sampler seam (point 4).

## NIT
- `wp2_lgrowth.py`: build the covariance arm directly instead of inverting the precision (point 7).
- `test_raw_iterate_grows_L_faster_in_the_early_phase` asserts a directional claim; confirm it is
  seed-robust before it is relied on for the accept/kill.

## Verdict
**RECOMMEND: merge.** The vendored adapter is a faithful vendor-and-fix: the `ma` default is
pinned bit-for-bit to the installed adapter across every state leaf, the `raw` mode is the single
documented line, the trace is free, and the frozen values the row schema reports are the values
the sampling kernel actually consumes. The `ma` default is safe to ship as behaviour-preserving.
