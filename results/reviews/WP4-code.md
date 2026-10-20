# WP4 code-correctness review (independent, worker-model task)

Subject: `wp/wp4` @ `83ebac6` (checkout `/home/akbar/pf/wp4`, merge base `574f960` — **10 commits
behind** `pf-r1-int` head `129afce`). Scope: `pfcheeshmc/ensemble_metric.py` (new, 154 lines), the
two-phase refresh in `pfcheeshmc/sampler.py` (+163), `WarmupInfo.positions` in `utils.py` (+10),
`tools/r1_bias.py` (+20), `tests/test_ensemble_metric.py` (13 tests), rows `WP4__r1.jsonl` (32) and
`WP4__reseed_ab.jsonl` (18).

Commands run this session (no GPU, forced CPU): test file **13 passed in 23.73 s**; degenerate-input
probe on all four estimators; full re-derivation of both SUMMARY tables from the rows; git
provenance checks (merge-base, per-commit `git show`, `git grep`). The full suite was **not** run
(the brief prohibits it; SUMMARY claims 112 passed — NOT independently established by me).

## 1. Every estimator returns a covariance in kernel order — CONFIRMED

- `_estimate` (ensemble_metric.py:62–95): `sample_diag` = `jnp.diag(ME.welford_diagonal(draws))`
  (per-coordinate sample variance = covariance diagonal); `fisher_diag` =
  `jnp.diag(ME.fisher_score_diagonal(draws, grads))` (nutpie's σ = √(Var[x]/Var[grad]), the
  covariance scale, clamped [1e-20, 1e20]); `dense_shrunk` = shrunk `welford_dense`;
  `diag_lowrank` = `D(I + U(λ−1)Uᵀ)D`, the **same expansion WP1 uses** for its low-rank arms, so
  orientation is consistent across the codebase. None inverts.
- Kernel order: `positions_to_draws` (:36–58) flattens the stacked warmup positions **per-leaf in
  `tree_leaves` order, which for a dict tree is sorted-key order — the kernel's order**, not
  Pathfinder's tuple order. Pinned by `test_positions_to_draws_uses_kernel_sorted_order`
  (test:101–117) on a deliberately non-sorted key tree.
- Teeth: `test_estimator_returns_symmetric_pd_covariance` (test:27–40) asserts the diagonal is
  within 20% of the true posterior **variances** (100 / 3 / 1 / 0.01). A precision would fail that
  by 1e2–1e4. 13/13 pass.
- The only hand-written numerics (A6): the shrinkage convex combination and the ridge/PD guard —
  both declared in the module docstring and SUMMARY §7.

## 2. Ridge + PD assertion before `_format_covariance` — CONFIRMED (by command)

`_regularize_pd` (:98–113): symmetrise, add ridge, host-side `eigvalsh`, raise `ValueError` on any
non-finite or non-positive eigenvalue. My probe (zero draws; zero gradients for fisher) on **all
four** estimators: every output finite, min eigenvalue = 1.00e-4 = ridge, `allclose(ridge·I)`.
Degenerate input yields the ridge, not NaN. `reseed_trajectory_length` (:141–154) separately
rejects non-PD input (`lmin <= 0.0` → raise). NIT: the test file covers degenerate input for 3 of
the 4 estimators — `fisher_diag` with zero gradients is my probe only, not pinned.

## 3. Default `metric_refresh="none"` preserves the single-phase path — CONFIRMED

Routing is purely additive at `sampler.py:700–705`: `refresh = kwargs.get("metric_refresh",
"none")`; only names in `_METRIC_REFRESH_NAMES` divert to `_run_chees_two_phase`. The diff at the
fork point (`git diff 574f960..HEAD`) touches no other line of the single-phase path.

Clarification on the `metric_refresh_reseed` flag: its default `True` means "re-seed T" — that is
**this branch's own two-phase behaviour**, not a reproduction of pre-branch behaviour (pre-branch
there is no refresh at all). `False` exists to isolate the re-seed's cost. A/B validity
established: at row time (`d457a17`) the flag was passed in the *executed* `sample_pf_chees_hmc`
call (`git show d457a17:tools/r1_bias.py:446–447`), so the `ab:sample_diag_noreseed` rows genuinely
ran without the re-seed; the missing *recorded* flag in those rows' kwargs was a row-auditability
gap fixed in `83ebac6`. Gradient bookkeeping spans both phases: `warmup_g/chain ≈ 1,146–1,175` at
W=50 in the rows (≈50 steps × L≈23), which a phase-B-only stack would halve — `_concat_warmup_info`
works as commented.

## 4. ε re-search moves ε — CONFIRMED

`eps_B = self._run_step_size_tune(pos_A, imm_B, **kwargs)` (sampler.py, in `_run_chees_two_phase`)
reuses the existing traced search, so cost is chain-count independent. Teeth test
`test_reshare_is_not_a_silent_noop` (test:121–151) asserts, end to end on a correlated 2-D model:
two-phase path taken (`metric_refresh` attr), both phases ran (`warmup_steps_used == 40 == W_A+W_B`),
T0 finite and > 0, re-searched ε finite and **≠** the phase-A endpoint, and the A8 attrs/gradient
counts intact. Passes.

## 5. λ_max seeding: inverse or covariance — CONFIRMED (correct sign)

`reseed_trajectory_length` (ensemble_metric.py:154): `T0 = (π/2)·sqrt(1/λ_min(Σ))`. Since
λ_max(Σ⁻¹) = 1/λ_min(Σ), this is exactly the brief's `T = (π/2)·√λ_max(M⁻¹)` with M⁻¹ = Σ — no
explicit inverse formed, no sign error. Pinned by `test_reseed_matches_closed_form` (diag cov
[100, 3, 1, 0.01] → T0 = (π/2)·10) and `test_reseed_rejects_non_pd`.

## 6. Do the rows support SUMMARY — NUMBERS YES, LABELS NO

- **All 16 refresh-arm cells match exactly** (medians over the rows' 2 seeds): 1292/1157/1344/1400
  (8 Schools), 2386/1352/2323/4934 (logistic), 4804/6673/4237/3778 (GEV), 5612/3886/5469/5317
  (radon). b² per the rows: every refresh cell b² ≤ 0.0022 at W=50, floors ≤ 0.0033.
- **The baseline column is mislabelled**: 955/335/595/485 are the **seed-2 values** of T1
  `int:covraw` W=50, not medians (974/324/557/**372.5**). The radon baseline seeds span 260 → 485
  (1.9×) — the single largest seed spread in the table, and it flatters the refresh ratio on radon.
- **Header says "seeds 1–2, warm"; every row is `cold=True`.** No wall-time claim may come from
  these rows; gradient counts are unaffected.
- **The A/B table is a stale `_running_` placeholder.** Data exists: 12 `ab:sample_diag_noreseed`
  rows (3 seeds; medians 1309/487/841/579, matching the parent's committed report table) plus
  `ab:baseline` for 8 Schools and logistic only. GEV/radon baselines were still being generated by a
  live driver at review time (see NON-BLOCKING 2).
- **Radon W=100/200 ladder** (the parent's "unmeasurable at any W" claim): supported — the 6 rows
  are in the **git-committed** file (`83ebac6`) and in an identical `/tmp/wp4_reseed_ab_shared.bak`:
  b² 0.0070–0.0107 (floor 0.0058–0.0070) and 0.0058–0.0086 (floor 0.0057–0.0064) at 1,128–1,498 and
  3,236–3,401 grads/chain. Every cell inside its noise band.
- **Scope note:** all 50 rows use `pf_draws=1` (one Pathfinder draw, the harness workaround for the
  6 GB card), `init_pick=stratified`, superchains off. Arm-vs-arm comparisons are fair (the
  baseline ran identically), but the absolute R1 numbers inherit the degraded start policy.

## 7. Strongest remaining path to a fast, clean, wrong result

1. **The shared row file is being rewritten live, and the on-disk copy has already LOST the
   committed radon-ladder rows.** While I was reviewing, a second driver
   (`~/scratch/wp4_reseed_ab_missing.sh`, filling the missing `ab:baseline` arms) rewrote
   `WP4__reseed_ab.jsonl`: on disk there are now 6 `ab:baseline` rows (8 Schools, logistic) and the
   6 `wp4:noreseed_W100_200` rows are gone; the git-committed file and the 12:35 `/tmp` backup both
   still contain all 18. If the final SUMMARY or a later report is written from the on-disk file,
   the ladder silently disappears. Cheapest guard: after the driver finishes, rebuild the file as
   the **union** of committed rows + new `ab:baseline` rows, re-commit, and assert the arm set
   covers `{noreseed×4 models, W100_200×2, baseline×4}`.
2. **Stale base makes the branch's diff look like it reverts the owner's flip.** Diffing against
   current `pf-r1-int` shows `imm_orientation` default back to `"precision"` and wp1's dict guard
   on `_kernel_permutation` deleted. Both are merge-base artifacts — the branch's own commits do
   not touch those lines, and an ordinary merge keeps the mainline state. Cheapest guard: rebase
   `wp/wp4` onto current `pf-r1-int` before merging (two lines, trivial), or verify post-merge with
   `git grep -n 'imm_orientation", "' pfcheeshmc/sampler.py` → must show `"covariance"`.
3. **`fisher_diag` on degenerate input is untested** (my probe: finite, ridge·I — safe today, not
   pinned for tomorrow).

---

**BLOCKING:** none. The numerical core is correct: covariances in kernel order, loud PD guard
(proven on all four estimators), re-seed sign correct, ε re-search toothed, A8 gradient contract
intact across both phases, and the branch's committed rows reproduce the SUMMARY's refresh-arm
numbers exactly.

**NON-BLOCKING:**
1. Rebase onto current `pf-r1-int` (`574f960` is 10 commits behind; the flip `08357db` and guard
   `7982356` post-date the fork).
2. Reconcile `WP4__reseed_ab.jsonl` after the missing-arms driver finishes (on-disk copy has lost
   the committed radon-ladder rows; git + `/tmp` backup are intact).
3. Fix SUMMARY before merge: "warm" → "cold" (all rows `cold=True`); baseline column to medians
   with the seed ranges stated (radon 260–485); fill the A/B table with the measured numbers;
   state the `pf_draws=1` scope.

**NIT:** `fisher_diag` degenerate test missing; test name typo `test_reshare_is_not_a_silent_noop`;
sampler default `metric_refresh_warmup=50` is a sharp edge for small total warmups (clamped to
`W_total − 2`; the harness always overrides with `W//2`); `positions_to_draws` docstring says "the
SAME `ravel_pytree` the kernel uses" but the implementation is per-leaf `tree_leaves`
concatenation — equivalent for a dict tree, the comment should say sorted-key order.

**RECOMMEND: merge** — after the three NON-BLOCKING items (rebase, row-file reconcile, SUMMARY
label fixes); none of them changes a measurement.
