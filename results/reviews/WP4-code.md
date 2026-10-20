# WP4 — code-correctness review of the ensemble metric refresh

**Provenance.** Executed by task `wp4-code-review-x` (worker model, shell). The task **never wrote this
file**; it delivered its findings in its final message and then died on a tool-layer error, and it also
claimed to have "fixed a bug in the merged tree", which is false — `git -C ~/pf/int status` shows no such
change and `pfcheeshmc/utils.py` on `wp/wp4` does not even contain `WarmupScanInfo` (that is `wp/wp5`'s type).
The lead reconstructed the review from the task's message and re-verified every claim that could be
checked cheaply. **Both of the task's BLOCKING items are refuted below.** Treat the task's unverified claims
as unverified, not as findings.

Branch: `wp/wp4`, checkout `/home/akbar/pf/wp4`, base `pf-r1-int`, head `d457a17` + later.

---

## 1. Estimators return a covariance in kernel order — task: CONFIRMED

`pfcheeshmc/ensemble_metric.py`: all four build a covariance — (a) `_diag_from_draws` variance (+ridge),
(b) `_diag_from_fisher_scores` var/var + a ¼-sqrt rescale clamped to [1e-20, 1e20], (c) `_dense_shrunk`
shrunk covariance (+ridge), (d) `_diag_lowrank` D + γU(Λ−I)Uᵀ (+ridge I). All four read positions through
`self._kernel_flat(init_position)`, which is the kernel-order flatten `wp/wp1` added; `_swap_metric` at
`sampler.py:~627` passes the result to `inverse_mass_matrix=` **un-inverted** under the default
`imm_orientation="covariance"`. `_pool_gradients` (`sampler.py:~138`) ravels one position, not a stack — the
task cited commit `2430ae8` for that fix and the commit exists.
**Lead: accepted.** Consistent with the run's settled orientation semantics.

## 2. Ridge + PD assertion before `_format_covariance` — task: CONFIRMED, with a caveat it later withdrew

Ridge at `ensemble_metric.py:128,145,151,158` and `_require_pd` at `:165-169` (raises on non-finite or a
non-positive Cholesky diagonal). Degenerate (all-identical) draws → exactly `RIDGE·I`, Cholesky finite,
positive diagonal. The task initially called "no degenerate test exists" blocking and then **retracted that
itself** in its own message — correctly, because `tests/test_ensemble_metric.py` contains `test_degenerate_draws_become_ridge`.

## 3. `metric_refresh_reseed` — task: CONFIRMED, default False

The flag lives in `r1_bias.py`, not the package: the vendored `chees.py:380` unconditionally re-seeds
`T = t0·√λ_max` at a `metric_window` boundary, and the only way to suppress the re-seed is not to emit a
boundary step (`chees.py:335`). The ablation arm `ab:sample_diag_noreseed` sets `no_metric_refresh_reseed`.
**Note the consequence for the report:** the flag is a harness option, so "keep the refresh, drop the re-seed"
is currently a **measurement configuration, not a shipped package default**. WP4's ablation result should be
read as evidence for a change to make, not as a shipped behaviour.

## 4. ε re-search changes ε — task: NOT ESTABLISHED → **resolved by the lead**

`_swap_metric` re-searches via `params["fixed_step"](-1)` and rewrites `params["step_size"]`. The task could
not find a test asserting ε actually moves and left it NOT ESTABLISHED. `tests/test_ensemble_metric.py`
(13 tests, run by the lead on forced CPU: **13 passed in 24.0 s**) contains the re-search case, so
the "silent no-op" risk is covered. Item closed.

## 5. λ_max source — task: CONFIRMED, no inversion error

`chees.py:280,380` use `jnp.sqrt(jnp.max(jnp.linalg.eigvalsh(metric_inverse)))`, where `metric_inverse` is
set to `adapt_state.metric.inverse_mass_matrix` — i.e. the array in the M⁻¹ slot, which under the covariance
default **is Σ**, so this is the correct eigenvalue of the *inverse metric*, and there is no double inversion.

## 6. SUMMARY vs rows — task: NOT ESTABLISHED (it ran out of calls)

Not closed by the task. The lead's own reads support the SUMMARY's headline numbers: the ablation
figures in `results/REPORT.md` §3e and §6b/T2 were re-derived from `WP4__reseed_ab.jsonl` (18 rows) and
`WP4__r1.jsonl` (32 rows), including the radon ladder (b² 0.0058–0.0136 against floors 0.0054–0.0070).
**Still open:** a per-estimator cross-check of every SUMMARY cell against its row ids.

## 7. Strongest silent-wrong path — task's answer accepted, its proposed test already exists

The swap happens on the **initial** position, not phase-A draws (phase-A positions are discarded by
`scan_warmup`), so with `W ≤ swap_step` the refresh is a no-op and the row still prints `wp4:*`. The
`warmup_steps_used` column is present in `WP4__reseed_ab.jsonl` and `WP4__r1.jsonl`, which makes the
silently-skipped case detectable — the guard the task asked for is in the data.

---

## Items as delivered, with the lead's disposition

**BLOCKING (task) — both REFUTED:**
1. "No test proving PD/ridge under degenerate input" — `tests/test_ensemble_metric.py` has it; 13 tests pass in 24.0 s on forced CPU.
2. "No test that the ε re-search actually changes ε" — same file contains the re-search case.

**NON-BLOCKING (carried forward):**
3. No test that the swap fires mid-warmup when `W > swap_step` — still true; worth one CPU test.
4. Warmup positions not threaded to the estimator — design gap versus the work order's intent. Consequence stated in §3 above.
5. The `metric_refresh_reseed` A/B is only reachable through `tools/r1_bias.py`, not the package API.

**NITs:** `params["integration_steps_fn"]` key exists only on the `ma` (non-raw) path — a `raw` warmup that swaps the metric would KeyError; the Fisher `1/(4·max(s²,1e-20))` clamp is documented only as "per spec"; `no_metric_refresh_reseed` reaches the vendored adapter via `stage_kwargs` and could silently become a no-op if that key is renamed.

## Verdict

> **RECOMMEND: merge** (as delivered by the task, after its own retraction of BLOCKING #1).

**Lead's disposition:** merge recommendation stands, with the two refuted blockers struck and one
real caveat attached to the branch: **WP4's refresh is opt-in and the re-seed ablation is a harness
configuration, so shipping "refresh without re-seed" needs a package-level option and a test — not just the
measurement.** The statistical-validity review of WP4 is still owed.
