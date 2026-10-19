# WP1 code-correctness review (L2, independent) — branch `wp/wp1` @ `08357db`

Reviewer: independent task, not the author, not the lead. Checkout `/home/akbar/pf/wp1`.
Every number below is from a command run in this session or a file opened in this session.
Environment: CPU-forced (`env CUDA_VISIBLE_DEVICES=` after `env`), through `pixi run --manifest-path`.

## 0. Counts

| run | result |
|---|---|
| `tests/test_metric_orientation.py` on branch, forced CPU | **17 passed** in 58.01 s |
| full `tests/` on branch, forced CPU | **55 passed** in 123.19 s |
| post-flip test file against `~/pf/legacy` (`ecd9ba1`), import pinned | **13 failed, 4 passed** in 50.90 s |
| pre-flip test file (`git show 08357db^:tests/test_metric_orientation.py`) against legacy, import pinned | **12 failed, 5 passed** in 53.37 s |

### The 13/4 vs 12/5 discrepancy: confirmed to be exactly the flip

The two failing sets differ by **one test**: `test_default_orientation_is_covariance_and_mixes` (post-flip) replaces `test_default_is_still_precision_orientation` (pre-flip). The replacement asserts the default arm *mixes* (`max_rhat < 1.05`, `min_ess > 10_000`) and fails on legacy with an **assertion**, not an import/collection error:

```
tests/test_metric_orientation.py:358: AssertionError: default arm is not mixing:
    max R-hat 10.3706, per-var ESS {'delta': 32.58..., 'beta': 37841.97..., ...}
```

The other 12 failures are identical between the two files (same lines, same reasons). Nothing else moved. Not a finding.

### Legacy failure classification (13, post-flip file)

| count | test(s) | failure type | intended reason? |
|---|---|---|---|
| 6 | `test_every_imm_source_hands_the_kernel_the_same_ordered_array[*]` | AssertionError, line 173: `diag(imm)=[9.88e-05, 1.01e+04, ...]` vs expected | yes — the wrong metric reaches the kernel |
| 1 | `test_covariance_orientation_mixes_the_anisotropic_target` | AssertionError, line 308: max R-hat 10.3706 | yes |
| 1 | `test_default_orientation_is_covariance_and_mixes` | AssertionError, line 358 (the delta test) | yes |
| 1 | `test_unknown_orientation_raises` | `Failed: DID NOT RAISE ValueError` (line 213) | yes — legacy silently ignores the kwarg, which is itself the defect the test targets |
| 4 | `test_permutation_is_tuple_to_dict_index`, `..._identity_when_names_are_sorted`, `..._rejects_a_tree_that_does_not_match_the_model`, `test_kernel_flat_matches_the_kernels_own_ravel` | `AttributeError: no attribute '_kernel_permutation'` (×3) / `'_kernel_flat'` (×1) | legitimate "fails pre-fix" (the mechanism does not exist on `ecd9ba1`), but these four are mechanism-absence, not wrong-behaviour assertions. They prove the branch adds the mechanism; the wrong-behaviour proof comes from the 8 rows above. |

**No import errors, no collection errors, no fixture errors.** The 4 tests that PASS on legacy pass by design: `test_permuting_factors_equals_permuting_the_expansion` (pure algebra), `test_kernel_convention_is_ke_half_p_A_p_and_momentum_cov_is_A_inverse` (orientation-agnostic convention pin), and the two asserted-failure pins (`test_precision_orientation_does_not_mix...`, `test_ordering_error_collapses_one_coordinate`) which are written to stay green while the bug exists.

## 1. Permutation — correct on every production case I constructed

Worked examples (my runner, `~/scratch/wp1_l2_perm_trace2.py`, exact replica of `_kernel_permutation`'s body, tuple inputs):

| case | expected | got |
|---|---|---|
| `(z, a(3))`, T=[z,a], K=[a,z] | [1,2,3,0] | [1,2,3,0] ✓ (matches the test's expected and its docstring arithmetic) |
| `(theta(8), mu, tau_log__)`, T≠K | [8,9,0,1,2,3,4,5,6,7] | ✓ |
| 4 scalars, T=[delta,beta,zeta,alpha] | [3,1,0,2] | ✓ |
| creation == sorted (8 Schools real shape) | identity | ✓ |
| `imm[perm][:, perm]` with Var(zeta)=100 | zeta variance on zeta diagonal | ✓ (diag [1,1,3,100]; off-diagonal coupling 7.0 lands on the right kernel pair) |
| leaf-count mismatch (missing / extra var) | raises | raises `ValueError` with the name list ✓ |

**Fragility found (NON-BLOCKING):** the construction is only correct when `ref_tree` is a **TUPLE** in `rvs` order. The first `ravel_pytree` relies on tuple order; a dict input ravels in sorted order and the function then returns a silently wrong permutation. Probe: dict input for the `(theta,mu,tau)` case returns `[1,2,3,4,5,6,7,8,9,0]` instead of `[8,9,0,...,7]`. The single call site (`sampler.py:568`) passes `samples_flat`, which is `pathfinder.sample`'s output — a tuple, because `approximate` was seeded with `initial_position=tuple(x0_pf[n] for n in self.rvs)` (`sampler.py:425-430`). So the production path is safe, but the contract is implicit. One-line fix: `assert isinstance(ref_tree, (tuple, list))` at the top of `_kernel_permutation`. (The call site passes `samples_flat` from `pathfinder.sample` (`:568`), a tuple because `approximate` was seeded with `initial_position=tuple(x0_pf[n] for n in self.rvs)` at `:432`.)

## 2. Low-rank arms — one code path, equivalence verified

- The production path expands to a dense covariance in tuple order (`sampler.py:509-511` for both `lbfgs_lowrank` and `fisher_lowrank`) and the **single** choke point (`:547-574`) applies orientation and permutation identically to all three `imm_source` values. I found no path by which an `inverse_mass_matrix` reaches the kernel without passing `:547-551`: `grep -n "inverse_mass_matrix =" pfcheeshmc/sampler.py` → lines 549, 551, 574 only (plus the `:1046` unpack of `_run_pathfinder`'s return). `grep -n "linalg.inv" pfcheeshmc/sampler.py` → line 549 only.
- Numerical equivalence of "permute factors first" vs "permute the expansion" (my runner, `~/scratch/wp1_l2_lowrank.py`): max abs diff **0.000e+00** in all four cases — random full-rank square (float64), repeated eigenvalues (`lam=2.0` ×4), non-square `U` (12×4), and float32 with scales 1e-6..1e6. The test at `tests/test_metric_orientation.py:225` pins the same equivalence.
- Tuple-order consistency of the factors: L-BFGS factors come from the tuple-seeded Pathfinder fit; the Fisher arm's `gdraws` are rows of `flat_mat` (tuple-order ravel of a tuple tree, `:475-479`) and its `ggrads` come from `unravel` derived from the same tuple tree at `:499`, with `ld_fn` normalising tuple input (`:72-83`), so draws and gradients share tuple order. **Confirmed consistent.** (I initially suspected a dict/tuple mismatch here — the `unravel` at `:499` is built from `samples_total`, which is a tuple; no mismatch.)

## 3. The flip — consistent, disclosed in code, but the SUMMARY is stale

- **Consistent across all three arms:** the orientation block operates on `covariance`, which every arm builds in 6b; the flip (`:547`) and the permutation (`:568-574`) apply uniformly. Verified in code, not from the comment.
- **Net behaviour change vs legacy `ecd9ba1`** (verified against the legacy file, `:406` mc-`inv(cov)`, `:436-438` low-rank expansion-as-is):
  - `mc` (default arm): orientation AND ordering both change.
  - `lbfgs_lowrank` / `fisher_lowrank`: the branch's pre-flip "precision" default had **inverted** their expansion relative to legacy (an unintended intermediate); the flip puts them back to legacy orientation, so their net change vs legacy is the permutation alone. The comment at `:524-528` ("a behaviour change for them") describes the pre-flip state and now reads backwards — fold into the SUMMARY fix.
- **Disclosure in code:** good. `:529-545` states the owner decision, the physics, the merged-with-permutation rationale, and the accepted GEV divergence cost.
- **NON-BLOCKING:** `results/WP1/SUMMARY.md` still describes the pre-flip default: line 13 "`imm_orientation` = `"precision"` (default, = today's `mc` behaviour)", line 59 names the deleted test `test_default_is_still_precision_orientation`, lines 112/114 label the precision arm "(shipped default)". A8 makes the SUMMARY the citable deliverable; update it before merge so the report doesn't cite a stale default.

## 4. `tools/wp1_rebaseline.py` — silent-wrong-result paths checked

- Stub: identity asserted **twice** (before the notebook load, `:175`, and per cell, `:198`), and the entry point's `__name__` is written into the row's kwargs (`:178,:192`). No stub leakage path found.
- Duplicates: `write_cell` (`:127-163`) raises on a live `(wp, arm, model, seed, rep)` collision; `OVERWRITE=1` is opt-in and prints what it replaced; VOID-stamped rows do not occupy a key. No silent overwrite path.
- Cold/warm: `cold = rep == 0`, one process per (model, imm_source, arm), `C.require_gpu_lock` at the top (`:166`). Decision-on-warm holds. NIT: from the second seed on, "cold" rep-0 rows run against a hot JIT cache; harmless under the warm-only rule but the label means "first rep", not "first compile".
- `d` is computed from the model's initial point (`tools/wp1_rebaseline.py:180`), not from the metric — cannot disagree with the model dimension; it is not checked against the metric's shape, but the sampler's own shape assert (int tree `utils.py`, `inverse_mass_matrix is {imm_arr.shape} ... expected ({d_flat},)`) catches that at run time.
- Metric silently degrading: if the Pathfinder pool degenerates to a point mass, `covariance = ridge * I` and `inv` succeeds with a ridge-scaled identity — no raise, plausible row. **Not introduced by this branch** (legacy had the identical behaviour) and not detectable from the row file; noted as a known blind spot for WP4, not a finding against this branch.

## 5. `WarmupInfo` seam

- `HMCInfo` is a `NamedTuple` (blackjax/mcmc/hmc.py:52) with fields momentum, acceptance_rate, is_accepted, is_divergent, energy, proposal, step_size, num_integration_steps, logdensity, logdensity_grad. **No field named `info`**, so `hasattr(chees_info, "info")` at `sampler.py:677` cannot mis-fire on the bare-info path.
- In the wp1 tree `chees_warmup_with_progress` returns the bare stacked info (utils.py tail: `return (last_states, parameters), warm_info`), so the unwrap is dead-but-harmless there. In the merged tree (wp/wp2) it returns `WarmupInfo(info=warm_info, trace=trace_hist)` and the unwrap is the live path; I verified the branch's 17 tests pass in both trees' shapes because the merge tree re-ran 99 passed earlier (lead's number, `pf-r1-int`) — I re-ran the wp1 suite (55) but the full merged suite was run by the lead, not me; flagging that the seam's *live* path is only exercised on `pf-r1-int`.
- `self._warmup_trace` is written at `:683` and read nowhere in the branch (grep: one hit). Diagnostics-only; last-call-wins on instance reuse — NIT, no action needed.

## 6. Observation (not a finding): a stale line-number read during the review

One mid-review `grep -n "inverse_mass_matrix" pfcheeshmc/sampler.py` returned line numbers matching the **legacy** checkout file exactly (`406 inv(cov)`, `436 expansion`, `450 return`, `454 _run_step_size_tune`). The lead resolved it as a stale/foreign read; I re-verified the current wp1 file independently: one `_run_pathfinder` at `:366`, metric choke point at `:547-574`, single `linalg.inv` at `:549`. Recorded so a future reviewer doesn't chase the same ghost.

## Verdicts

**BLOCKING:** none.

**NON-BLOCKING:**
1. `results/WP1/SUMMARY.md` lines 13/30/59/112-114 describe the pre-flip default and a deleted test name; the `:524-528` comment's "behaviour change for them" reads backwards post-flip. Update before merge (the A8 deliverable is the citation surface).
2. `_kernel_permutation` silently returns the wrong permutation for dict input (probe: `[1..9,0]` vs `[8,9,0..7]`). Add an `isinstance(ref_tree, (tuple, list))` assert so the implicit contract becomes a loud one.

**NIT:**
3. `_metric_to_kernel_order`'s 1-D branch (`return imm[perm]`) has no caller in the branch (all metrics are 2-D) and no test — either remove it or pin it.
4. Rebaseline "cold" rows from seed 2+ are hot-JIT; label means "rep 0".
5. `self._warmup_trace` is write-only in this branch.
6. The seam's live path (merged tree) was exercised by the lead's 99-passed run, not re-run by this reviewer.

**RECOMMEND: merge** — the permutation is correct on every case I could construct (including the vector-leaf adversarial cases), the low-rank equivalence is numerically exact, the flip is consistent at a single choke point with the code disclosure present, the test teeth are real (12 substantive pre-fix failures, zero import/collection errors), and the one stale artifact (SUMMARY) is a docs fix, not a code defect.

**Flip assessment, as invited:** I did not re-run the R2/R1 measurements — those are the lead's and the statistical reviewer's numbers, outside my scope. What I can say from code: the flip is a two-line default change with the wrong orientation still fully reachable as `imm_orientation="precision"` and still pinned by an asserted-failure test, so the regression surface is small and the revert is a one-character diff. The evidence for the flip's correctness rests on the R̂/ESS measurements, which the 8 assertion-based legacy failures confirm are the right instrument (a wrong metric is observably un-mixing, R̂ ~10-13, before any bias gate is involved).
