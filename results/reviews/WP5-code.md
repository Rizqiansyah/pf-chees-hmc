# WP5 code-correctness review (independent task)

Subject: `wp/wp5` @ `0e6739f` (14 commits on `pf-r1-int`), checkout `/home/akbar/pf/wp5`.
Method: full source diff of `pfcheeshmc/sampler.py` + `pfcheeshmc/utils.py`, all 9 tests in
`tests/test_wp5_capacity.py` read, one test-file run (9 passed in 40.83 s, forced CPU), and the
80 `WP5__memprobe_before.jsonl` + 5 `WP5__accept.jsonl` rows recomputed per (model, C).
Tool layer was alive; `echo tool-layer-ok` passed on first action.

## 0. BLOCKING — this branch is pre-flip, and merging it reverts the owner's decision

The merge base of `wp/wp5` and `pf-r1-int` is `164f449` ("Merge branch 'wp/wp1' into pf-r1-int").
The owner-approved orientation flip `08357db` and the dict-argument guard `7982356` landed on
`pf-r1-int` **after** that point. Concretely, the wp5 tree carries, relative to `pf-r1-int`:

- `sampler.py:664` — `orientation = kwargs.get("imm_orientation", "precision")` (pf-r1-int: `"covariance"`)
- `sampler.py:649-661` — the pre-flip comment block ("The default stays 'precision' … changing it is an efficiency decision the owner makes, not a task")
- `sampler.py:138-146` — the `isinstance(ref_tree, dict)` rejection in `_kernel_permutation` is **absent** (pf-r1-int `7982356` added it)
- `tests/test_metric_orientation.py:347-358` — WP1's post-flip test `test_default_orientation_is_covariance_and_mixes` is reverted to `test_default_is_still_precision_orientation`, which **asserts the default fails to mix** (max R̂ > 2.0)

A merge that resolves the resulting conflicts by taking the wp5 side silently ships the broken
orientation as the default — the exact failure class (an owner decision reverted by a merge
conflict) this run has already suffered in the opposite direction. **Resolution rule for the
merger: on every conflict in `sampler.py` (orientation hunk + comment + dict guard) and in
`tests/test_metric_orientation.py`, take the `pf-r1-int` side; wp5 contributes only the
pool/init/scan changes.** After merging, re-run `test_metric_orientation.py` and
`test_wp5_capacity.py` on the merged tree.

Everything below judges the wp5 changes *within their own tree*, where the tree is internally
consistent (precision default + a test pinning the precision default).

## 1. Default `pf_pool_cap=None` preserves historical behaviour — CONFIRMED for the pool, with one caveat

- `sampler.py:437` `pf_pool_cap = kwargs.get("pf_pool_cap", None)`; `:515-519` `total = chains*draws` when `None`, else `min(total, cap)` with a `cap < chains` raise. The pool-draw call `bj.vi.pathfinder.sample(key_samp, pf_state, total)` is otherwise untouched.
- The test proves it at the right level: `test_pool_cap_bounds_the_pool_and_default_is_uncapped` spies on `pathfinder.sample` and asserts the default draws exactly `8*500` rows and `pf_pool_cap=32` draws 32.
- The old `samples_cd` reshape round trip `(total) → (chains, draws) → (total)` was algebraically a no-op and is deleted; `samples_cd` has no other consumer in the tree (grep: none).
- **Caveat (NON-BLOCKING):** the shipped default does *not* preserve historical init behaviour, because `pm_chains` defaults to `1` (`sampler.py:425`), not `chains`. With `init_source="pf"` the PyMC points only seed the PF optimiser, so one point changes the L-BFGS seed → different pool → different starts. It is documented (`sampler.py:419-424` "Default 1 = the R1 fix; pass pm_chains=chains for the old per-chain behaviour") and is the plan's own ask (WP5 item 4), but a merge note must say the *numerics* of the default init path changed, not just the memory.

## 2. `pm_chains=1` guard — CONFIRMED, and the "one bad chain" case cannot exist by construction

The guard runs on the unique points before broadcast (`sampler.py:450-471`), so with one shared point "one chain's start bad, the rest fine" is structurally impossible; the only question is whether an unrepairable shared point can pass silently. Two tests cover it:

- `test_shared_init_point_repairs_a_nan_gradient_base` — a fake init sitting exactly at a NaN-gradient point (the GEV xi=0 shape): the guard repairs, and **every** chain's start is asserted finite and within 2e-3 relative + 5e-6 absolute of the repaired base.
- `test_unrepairable_shared_base_raises_not_nans` — `ld_fn ≡ -inf`: `pytest.raises(RuntimeError, match="finite log-density and finite gradient")`.

The `init_source="pf"` path takes chain starts from PF pool draws (finite by construction, as before the branch) with the 1e-3/1e-6 jitter applied only to the `adapt_diag` fallback; `test_adapt_diag_fallback_keeps_distinct_starts_with_a_shared_base` pins that the fallback cannot collapse to C identical starts (the R̂=1.000-while-stuck trap) and that `pm_chains=6` is untouched by the jitter path. Both ran green in my 9-passed run.

## 3. "Byte-identical picks" — CONFIRMED as value-identical, with stated gaps

`test_chunked_scoring_matches_the_single_vmap_original` runs the same pool twice (`score_chunk=5` vs `10**9`, i.e. multi-block vs the original single-vmap path) and asserts `np.testing.assert_array_equal` on the **chosen start positions** — exact, not allclose. It does not compare indices: a dropped/reordered row changes which pool row is picked, and with continuous draws equal positions imply equal rows, so the value comparison is sound. Gaps: (a) no explicit pool-smaller-than-cap case (`total < cap`, where `min()` leaves the pool unchanged — trivially safe, untested); (b) ties in scores are impossible by construction on continuous draws, so the absent tie test is not a hole. `test_best_chain_scoring_picks_the_manual_argmax` reproduces the key derivation by hand (`split(key(5),5)[0]`, `pm_tune=500`) and asserts the PF seed equals the manual argmax row at `rtol=0, atol=0` — the strongest form this check can take.

## 4. Scan trimming — CONFIRMED; no dropped field is read

- Sampling: `_SampledState(position, logdensity)` + `_SampledInfo(is_divergent, acceptance_rate, is_accepted, num_integration_steps, energy)` (`sampler.py:27-55, 954-965`). The only consumers are `_to_inference_data` (`pos_hist = getattr(states_hist, "position", …)`, `lp_src = getattr(states_hist, "logdensity", …)`, and the five `infos_hist.*` fields) and `sample()`'s attrs block (`chees_info.num_integration_steps`, `infos_hist.num_integration_steps`, `sample_stats["diverging"]`). Grep for `.momentum`, `.proposal`, `.logdensity_grad` across `pfcheeshmc/` + `tests/` finds only comments and `tests/test_inference_data.py`, which feeds a hand-built full HMCInfo into `_to_inference_data` directly and therefore still passes.
- Warmup: `WarmupScanInfo(num_integration_steps, is_divergent, acceptance_rate)` replaces the full HMCInfo stack on both paths; positions are stacked only on the stop-rule path, where `block_max_rhat` reads them. `_run_chees` unwraps `WarmupInfo.info` once (`sampler.py:794-801`) and reads `num_integration_steps` — present. The key streams are unchanged (default path `split(rng_key, num_steps)` as before; stop-rule path `fold_in` blocks as before).
- The projection is pinned by `test_warmup_scan_projects_to_the_three_fields_and_counts_are_exact` (field set, (steps, chains) shapes, integer dtype, exact counts) and `test_sampling_scan_projection_keeps_all_six_sample_stats_fields` (all six sample_stats vars present, steps shape (4, 25), posterior (4, 25, 4)).
- **NIT:** `WarmupScanInfo.acceptance_rate` stores `acc_vals` (the raw per-chain rates, `utils.py:247`) while the class docstring calls it "the harmonic mean over chains" — the pooled value is `acc_mean`, computed and then dropped. Nothing in the package reads this field, so it is a docstring error, not a bug; fix the docstring or store `acc_mean`.

## 5. Memory attribution from ABSENCE — the rows support it, one arithmetic is mislabelled

Recomputed per (model, C) from `kwargs.peak_bytes_in_use` (the row's own `peak_mem_bytes` is 0/None in the memprobe file — the stage data lives in `kwargs`; that is a harness schema choice, not a bug, but the SUMMARY should say where the number comes from):

| model | C | last stage written | recomputed max peak |
|---|---|---|---|
| 8schools | 1,024 | sampling_scan (full run) | 2,314.1 MiB |
| gev | 1,024 | sampling_scan (full run) | 946.4 MiB |
| logit_1000x25 | 1,024 | pathfinder_solve | 581.4 MiB |
| radon NC | 64 | sampling_scan (full run) | 2,575.7 MiB |
| radon NC | 128 / 256 / 1,024 | pathfinder_solve | 652.7 / 652.9 / 653.5 MiB |

The pipeline order (… → `pathfinder_solve` → `pf_pool_draw` → …) is visible in the code, so every dying config stops at exactly the boundary of one stage: `pf_pool_draw`. The attribution "the wall is the C × pf_draws pool chain" is therefore a one-stage-boundary inference, which is the strongest the absence method can be. The failed-allocation sizes (699.59 MiB / 1.35 GiB / 5.36 GiB) are **not in the rows** — they come from the XLA crash text quoted in the commit messages; the report already flags that provenance.

Two discrepancies in the SUMMARY:
- **NIT (text mislabelling):** the prose pairs "699.59 MiB at 64k rows, 1.35 GiB at 128k" — the row counts are off 8×. 699.59 MiB ≈ 512k rows (C=128 × 4,000) and 1.35 GiB ≈ 1M rows (C=256 × 4,000); the SUMMARY's own table above the prose pairs them correctly.
- **NIT:** "~14 GB at d=175, W=2,000, C=1,024" for the warmup scan is labelled "measured" in `utils.py:22-25` — no pre-branch run reached the warmup scan at C=1,024 (all died at pool draw), so it is an arithmetic estimate (W·C·d·8 B × ~5 for the duplicated info leaves ≈ 14.7 GB), not a measurement. Say "computed from shapes".

## 6. Warm wall beside process-cumulative peak — the 653 MiB is safe, the row to quote is the cold one

`wp5_accept.py` uses two processes per config: rep 0 (cold) is a single fresh process whose peak spans exactly that one run — the defensible memory number. Rep 1 (warm) first does a throwaway full `sample()` to compile, then times a second `sample()` on a fresh `PfCheesHMC` instance, and the row's `peak_mem_bytes` is read after both runs (the harness's own print labels it `peak(cum)`). For radon the warm cumulative (653.3 MiB) is within 0.2 MiB of the cold single-run peak (653.5 MiB), so quoting 653 MiB is correct — but quote the **cold** row, and do not present a warm-row peak as a warm-run peak on any other config without checking the two agree. The 42.94 s warm wall itself is clean (measured run only; the throwaway is excluded) and comes from a fresh sampler instance, so no state leaks into the timing.

## 7. Fast-clean-wrong paths — strongest one is the pre-flip state, not the capacity code

- **Kwarg drift:** none. `pf_pool_cap`, `pm_chains`, `score_chunk` are read and consumed inside the same function (`_run_pathfinder`), and the spy test in item 1 proves the cap value reaches the pool draw. `pm_chains` range-checked at `:426-427`, `score_chunk >= 1` at `:439-440`.
- **Silent no-cap:** `test_pool_cap_bounds_the_pool_and_default_is_uncapped` would go red if the cap stopped reaching the draw.
- **Wrong-orientation silent wrong posterior:** not in wp5's own tree, but the merge hazard in item 0 is precisely this class — a "capacity" merge that flips the metric back to the orientation that fails R̂ ≤ 1.01 on GEV in 5/5 seeds, with every wp5 row (run on the pre-flip default) looking fine because the capacity metrics do not depend on orientation.

## Verdicts

| # | claim | verdict |
|---|---|---|
| 0 | branch is consistent with the merged tree | **REFUTED — pre-flip; BLOCKING at merge** (item 0) |
| 1 | `pf_pool_cap=None` preserves historical pool | CONFIRMED (pool); init numerics change by design (NON-BLOCKING) |
| 2 | guard equivalence under `pm_chains=1` | CONFIRMED (2 tests, green) |
| 3 | byte-identical picks | CONFIRMED (value-identical, exact assert); small coverage gaps stated |
| 4 | InferenceData unchanged, no dropped field read | CONFIRMED (consumer grep + 2 tests) |
| 5 | memory attribution | CONFIRMED from one-stage-boundary absence; allocation sizes from crash text; two NITs |
| 6 | warm wall + cumulative peak | CONFIRMED with the cold-row preference stated |
| 7 | no kwarg drift | CONFIRMED |

**BLOCKING**
1. Rebase/merge onto current `pf-r1-int` and resolve every conflict in `sampler.py` (orientation default + comment + `_kernel_permutation` dict guard) and `tests/test_metric_orientation.py` by taking the pf-r1-int side. Merging the branch as-is reverts the owner's 2026-10-20 flip and the `7982356` guard. Re-run both test files on the merged tree before merge.

**NON-BLOCKING**
1. Merge note: the shipped *default* init path changed numerics (`pm_chains=1` → different PF seed → different pool/starts), even though the pool size default is historical. Say so next to "default preserves behaviour".
2. `utils.py:22-25` "~14 GB … measured" is computed, not measured — relabel.
3. Quote peaks from the cold (single-config) rows; the warm row's peak is cumulative by construction.

**NIT**
1. SUMMARY prose row labels off 8× ("64k/128k" should be "512k/1M") — the MiB figures and the table are correct.
2. `WarmupScanInfo.acceptance_rate` docstring says "harmonic mean" but stores the raw per-chain `acc_vals`.
3. `sampler.py:794-801` comment still says "stacked HMCInfo"; it is `WarmupScanInfo` now.
4. `test_adapt_diag_fallback_keeps_distinct_starts_with_a_shared_base` is defined after the `if __name__ == "__main__"` block, so the manual runner skips it (pytest still collects it).

**RECOMMEND: hold — single reason:** the branch is pre-flip; merge it only after re-basing onto current `pf-r1-int` with the conflicts resolved in favour of the covariance default, the dict guard and the post-flip test, and both test files re-run on the merged tree. The capacity code itself is sound: every behavioural claim I could execute came back confirmed, and the memory attribution is as strong as the absence method allows.
