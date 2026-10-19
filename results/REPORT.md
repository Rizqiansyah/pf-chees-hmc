# pf-r1 — external review implementation run: REPORT

Branch `pf-r1` (and the merged measurement tree `pf-r1-int`). Nothing is pushed; `main` is untouched at `ecd9ba1`. Drafted while tasks finish; sections marked PENDING name what they are waiting on.

## 1. What was run

An external review produced `docs/review-2026-10/pf-chees-hmc-implementation-plan.md`. This run executes it: work packages WP0–WP9, each in its own git checkout on a `wp/*` branch, findings recorded in `results/LEDGER.md`, every merge gated on two independent reviews (`results/reviews/`), all GPU work serialised through `tools/gpu_run.sh`.

Owner objective for this run: maximise GPU throughput in the **R1** regime (≈1,024 chains × ≈10 draws), judged by the bias gate G-R1 (b² < 0.01 against a CPU NUTS reference) and gradients per chain. ESS/s is the secondary R2 objective (32–128 chains × 1,250 draws).

## 2. The two correctness findings that decide everything

### 2.1 The metric is applied to the wrong coordinates (P0, confirmed)

Pathfinder draws are flattened as a **tuple** in `model.value_vars` order; the HMC kernel state is a **dict** that JAX ravels in **sorted-key** order. On GEV (`mu, sigma_log__, xi_interval__` vs sorted `mu, xi_interval__, sigma_log__`) and on radon noncentered the two orders differ; on 8 Schools they coincide, which is why the bug survived every test the repo had.

Measured in isolation (orientation held correct, ordering wrong), on a 4-scalar target with posterior variances 1e2 / 3 / 1 / 1e-2: **max R̂ 1.0069 and worst-variable bulk ESS 3,176 against ~31,000 for the other three coordinates** — one coordinate silently dies. `tests/test_metric_orientation.py` is the permanent gate (G-ORDER); it must fail on `ecd9ba1`.

### 2.2 The metric orientation is inverted (P0, confirmed — after one retracted contradiction)

`pfcheeshmc/sampler.py` passes `inverse_mass_matrix = jnp.linalg.inv(cov)`. The array in BlackJAX's slot is `M⁻¹` in `KE = ½ pᵀ M⁻¹ p`, and leapfrog frequencies are `ω² = eig(M⁻¹ H)` with target Hessian `H = Σ⁻¹`. The well-conditioned choice is therefore **`M⁻¹ = Σ`, i.e. the slot must hold the covariance** (Stan's docs call the inverse mass matrix "the covariance" for exactly this reason). An earlier entry in this run's ledger challenged the review on this point; that challenge is **retracted** at `5a18c88` after direct re-measurement:

| arm | max R̂ | min bulk ESS | ESS/gradient | worst mean error | frozen ε |
|---|---|---|---|---|---|
| permuted + **covariance** (correct) | 1.0021 | 30,967 | 12.45 | 0.010 sd | 1.213 |
| permuted + precision (**shipped**) | **12.1019** | **32** | 0.014 | 0.232 sd | 0.0140 |

Per-variable ESS on the shipped arm: `{delta: 33, beta: 33,052, zeta: 32, alpha: 35}` — three of four coordinates do not move. Command: `~/scratch/orientation_arbiter.py`.

### 2.3 The three arms never shared a convention (P0, direction corrected)

`imm_source="mc"` inverted its covariance, while `lbfgs_lowrank` and `fisher_lowrank` expand `D(I + U(Λ−I)Uᵀ)D`, which **is** the covariance. So the low-rank arms were accidentally correct and the default arm was not. Every published `mc`-versus-low-rank ratio in the repo (README ladder ratios included) compares a well-conditioned arm against a badly-conditioned one and is **void**; the conclusions they supported must be re-derived, and in the opposite direction from the naive reading.

## 3. Measured effect on the real models (R2 shape: 32 chains × 1,250 draws, W=2,000, 5 seeds, warm rows)

| model | arm | max R̂ | min bulk ESS | ESS/s | gradients/ESS | divs/1k | ε | nominal L |
|---|---|---|---|---|---|---|---|---|
| 8 Schools | shipped (unpermuted + precision) | 1.0017 | 22,120 | 2,085 | 58.2 | 0.00 | 0.185 | 42 |
| 8 Schools | permuted + precision | 1.0020 | 22,082 | 2,166 | 57.2 | 0.00 | 0.187 | 42 |
| 8 Schools | **permuted + covariance** | 1.0014 | 20,469 | 2,226 | **35.5** | 0.00 | 0.524 | 22 |
| GEV | shipped | 1.0023 | 8,102 | **403** | 73.4 | 0.23 | 0.0009 | 54 |
| GEV | permuted + precision | **1.0465** | 679 | 44 | 295.8 | 0.03 | 0.0009 | 8 |
| GEV | **permuted + covariance** | 1.0074 | 4,983 | 303 | 71.2 | 0.47 | 0.833 | 10 |
| logistic | shipped | 1.0016 | 15,462 | 417 | 151.7 | 0.00 | 0.0020 | 129 |
| logistic | permuted + precision | 1.0021 | 14,058 | 403 | 163.8 | 0.00 | 0.0021 | 122 |
| logistic | **permuted + covariance** | 1.0014 | 17,163 | **1,048** | **46.1** | 0.00 | 0.656 | 27 |
| radon NC | shipped | 1.0069 | 3,845 | 98 | 501.9 | 0.00 | 0.0025 | 75 |
| radon NC | permuted + precision | **1.0142** | 2,418 | 28 | 2,278.8 | 0.00 | 0.0007 | 268 |
| radon NC | **permuted + covariance** | 1.0066 | 4,468 | **171** | **182.3** | 0.00 | 0.971 | 27 |

Readings that matter:
- **Fixing the ordering alone is a regression.** Gradients/ESS improves on 1 of 4 models (criterion: 3 of 4) and pushes GEV and radon past the R̂ ≤ 1.01 bar. The permutation must ship **together with** the orientation fix.
- **Permutation + covariance improves gradients/ESS on 4 of 4** and ESS/s on 3 of 4, and passes R̂ everywhere.
- **GEV is the honest counter-example**: there the double-wrong shipped arm scores 403 ESS/s at R̂ 1.0023 while the correct cell scores 303–343, and divergences rise 0.225 → 0.475 per 1k. Unresolved, logged for WP4: on GEV the *wrongly-ordered* covariance metric beats the correct one (1,310 vs 343 ESS/s), and `xi_interval__` is boundary-constrained.

## 3b. R1, the owner's regime (C=1,024 × D=10, seeds 1–2)

| model | shipped orientation | corrected (covariance + raw iterate) |
|---|---|---|
| 8 Schools | fails W=50 (b² 0.0139), passes W=100 (1,283 gradients/chain) | **passes W=50 on both seeds, 955 gradients/chain** |
| logistic 1000×25 | **never** — b² 0.177/0.193, 0.130/0.158, 0.039/0.050 at W=50/100/200, both seeds | **passes W=50 on both seeds, 335 gradients/chain** |
| GEV | fails W=50 on one seed (b² 0.087), passes W=100 | **passes W=50 on both seeds** (b² 0.0016 / 0.0001) |
| radon NC | **never** — b² 105/106 at W=50, `unresolvable` at W=100 and W=200 (b² 12.1) | **no W passes on both seeds**: W=200 gives 0.0234 (fail) and 0.0071 (pass); W=100 is `unresolvable` |

Verdicts are **per seed** under the harness's own gate (`tools/r1_bias.py:g_r1`: eligible when C·D ≥ 2,000; `unresolvable` when floor > 0.005; otherwise pass iff b² < 0.01). An earlier version of this table averaged two seeds and reported radon NC as passing at W=200 — the seeds disagree, so the honest statement is that radon NC does not pass on every seed at any warmup tested. The project target (R1 at C=1,024 × D=10 passing G-R1 with warmup ≤ 300 on all four models) is therefore **met on three of four** by the corrected arm, two of four by the shipped one. Split R̂ across the same cells reads 1.12–8.87, including in cells that pass on bias — direct confirmation that R̂ cannot arbitrate at 10 draws per chain.

Two limits on this table: the arms move **two** factors at once (orientation and adapter iterate), so the R1 gain is not attributable to either alone; and all 48 rows are cold with 2 seeds, so no wall-clock claim comes from them.

### 3d. T4 — capacity at the R1 shape (C=1,024 × D=10, W=2,000, public-default `pf_draws=4,000`, 6 GB card)

Verified from `wp/wp5`'s row files (`WP5__memprobe_before.jsonl` 80 rows, `WP5__accept.jsonl` 5 rows). The before-table's "stage" is the last row written before the process died, so the stage attribution comes from the rows and the failed-allocation size from the crash text — stated because it is a weaker provenance than a row field.

| model | d | largest C before | where it died before | largest C after | cold peak after | init warm after |
|---|---|---|---|---|---|---|
| 8 Schools | 8 | 1,024 (peak 2,314 MiB) | fits | 1,024 | 224.1 MiB | 25.9 s cold, compile-dominated |
| GEV | 3 | 1,024 (946 MiB) | fits | 1,024 | 288.1 MiB | 35.3 s cold |
| logistic | 25 | **OOM at 1,024** | pool draw — XLA `NOT_FOUND` profiling the 4.1M-row pool | 1,024 | 581.4 MiB | 36.4 s cold |
| radon NC | 175 | **64** (2,575 MiB at 64) | pool draw — 699.59 MiB at C=128, 1.35 GiB at 256, 5.36 GiB at 1,024 | **1,024** | 653.5 MiB | **4.07 s warm**, 0 divergences per 1k, 42.9 s total warm |

The wall is the `C × pf_draws` draw pool (draw → flatten → X → XᵀX); the failed allocation scales with the pool size exactly. Both retracted memory explanations stay dead. **Accept criteria: capacity PASS on all four models; the "init under 2 s" bar is NOT met** — warm init is 4.07 s, of which 2.47 s is WP5's pathfinder/pool/scoring chain and **1.59 s is the step-size tuning stage owned by WP7**. The bar as written adds two tasks' budgets together; it needs restating as per-owner budgets, and neither reading passes today. Cold init at C=1,024 is 25.9–45.1 s (compile), which is the run-once cost no cache can serve on the owner's run-once workflow.

### 3e. WP4 — ensemble metric refresh during a short warmup (R1, C=1,024 × D=10, W=50, 2 seeds)

| model | no refresh: b² | no refresh: verdict | refresh (any of 4 estimators): b² | refresh verdict | gradients/chain, no refresh → refresh |
|---|---|---|---|---|---|
| 8 Schools | 0.0010 | pass | 0.0008 – 0.0010 | pass | 974 → 1,157 – 1,400 |
| GEV | 0.0008 | pass | 0.0002 – 0.0013 | pass | 557 → 3,778 – 6,673 |
| logistic | 0.0047 | pass (marginal) | 0.0006 – 0.0009 | pass | 324 → 1,352 – 4,934 |
| radon NC | **1.6118** | **fails at every warmup tested** | **0.0018 – 0.0020** | **pass on both seeds** | 372 → 3,886 – 5,612 |

1. **The refresh is the only measured path to a passing R1 run on radon NC.** The task initially argued the same benefit is cheaper by running the baseline at W=200; that is refuted by its own rows (seed 1 b² **0.02339 = fail** at W=200, both seeds `unresolvable` at W=100), and the task accepted the correction.
2. **Every estimator is a cost on the models Pathfinder already handles** — 1.4–8× more gradients for a bias change inside seed noise — and **no estimator clears the plan's 1.2× kill clause against the plain sample diagonal** (dense-with-shrinkage 1.03× on logistic and radon, low-rank worse, Fisher-diagonal worst on GEV at 6,673). So: kill `dense_shrunk`, `diag_lowrank`, `fisher_diag`; keep `sample_diag`; ship the refresh **opt-in, keyed on Pathfinder metric quality** (radon's variance ratio 0.011 versus 8 Schools' 0.56), never on by default.
3. **The cost has an identified cause and a queued test.** The trajectory-length re-seed `T = (π/2)·√λ_max` lands at L = 23–402 against the baseline's converged 10–27 because λ_max of a 50-step ensemble estimate is noise-dominated. The A/B (refresh with vs without the re-seed, 4 models × 3 seeds at W=50) is queued behind the current GPU work, with the decision rule fixed before the run: within 1.5× of baseline cost on 8 Schools and GEV while radon still passes → cheap opt-in; still ~3× → the cost is intrinsic.

## 4. Decisions taken in this run

1. `imm_orientation` is an explicit, tested parameter (default unchanged for now); `imm_order` likewise. The default flip is owner-gated (G3) — recommendation and remaining evidence in §7.
2. **Nested R̂ is not a convergence gate in R1.** It reads 1.0040 on a cell with b² 0.4100. Measured reason: it is blind to bias shared by all chains (all-chains +1 sd reads 1.0004, identical to clean; every-other-superchain +1 sd reads 1.1135 at K=8). R1 Pathfinder deficit *is* common-mode, so G-R1 (b²) is the only gate; nested R̂ stays as a differential-drift/multimodality alarm. A proposed "precision ceiling at K" was refuted by the same measurement.
3. Vendored ChEES adapter with `iterate="raw"|"ma"` (default `ma`, behaviour-preserving). The raw iterate reaches the plateau trajectory length by ~step 100 (logistic L@100: 25–42 vs 8.3) without worsening the frozen value at W=2,000 (25.0 vs 26.8); `t0=π/2` adds nothing.
4. **Merged-tree rule.** `wp/wp1` + `wp/wp2` alone crashed every run (`WarmupInfo` vs `num_integration_steps`) while each branch was green alone. Since then, accept/kill decisions are re-verified on `pf-r1-int`.

## 5. Gates

| gate | status | basis |
|---|---|---|
| G-ORDER (permanent metric tests) | PASS | 17 tests in `tests/test_metric_orientation.py`; 12 fail on `ecd9ba1` |
| G-A R̂ ≤ 1.01 (permuted + covariance) | PASS on 5-seed **medians**; per seed on GEV it is **3/5** (1.0065, 1.0017, 1.0054, 1.0141, 1.0117) | §3, §3c |
| G-B tail ESS / E-BFMI | PASS | min tail 1,207; E-BFMI 0.60–0.99 |
| G-R1 b² < 0.01 at C·D ≥ 2,000 | corrected arm passes at W=50 on 8 Schools (955 gradients/chain), GEV and logistic; **radon NC passes on no W for both seeds**. Shipped arm: never on logistic | T1, 48 rows, per-seed verdicts under `g_r1` |
| G-R1-diag | REDEFINED | §4.2 |
| G-MEM peak ≤ 5.0 GB | **MET for radon at C=1,024** | 653 MiB peak — the R1 wall was the `chains × pf_draws` Pathfinder pool chain (radon died at C≥128 before); bounding the pool (`pf_pool_cap≈4C`) and `pm_chains=1` fixes it. GEV/8 Schools already fit at C=1,024 (0.9 / 2.3 GB). Warm init 4.1 s misses the 2 s stretch sub-goal |
| G3 (default change: ≥5 seeds, no divergence increase, means within 1σ) | NOT MET on GEV, with the terms now separated | shipped arm fails R̂ ≤ 1.01 on **5/5 seeds** (to 1.0688); candidate passes **3/5** and halves mean error, but divergences rise seed-stably 0.025 → 0.400 per 1k. The "means within 1σ" clause is unsatisfiable as written (worst z 3.4 on a 0.026 sd shift) and should be restated in sd units |

## 6. Reference artifacts

`~/pf-runs/references/{8schools,gev,logit_1000x25,radon_noncentered}.npz` — CPU NUTS, 8 × 10,000, seed 42, manifest `results/references/MANIFEST.json`. The first delivery collapsed the parameter axis for vector variables and was rejected; the rebuilt files are per-parameter and independently re-derived (0 contract violations). The stored stat columns are display-only: the bias gate recomputes everything from the sample arrays, and that is recorded in the manifest as `single_authority`.

## 7. Open before this run can close


1. Reviews outstanding: `wp/wp1` (two worker-model tasks running), `wp/wp8`, `wp/wp3` (statistical), `wp/wp9docs`, `wp/wp2`, `wp/wp5`, `wp/wp4`. `wp/wp0refs` is merged.
2. **WP5 capacity: DONE and verified** — radon NC runs at C=1,024 × D=10 (peak 653 MiB, suite 108 passed); the memory source is the `chains × pf_draws` pool chain, bounded by `pf_pool_cap`. Open: warm init 4.07 s against a 2 s bar that mixes two tasks' budgets (§3d).
3. **WP4 measured** (§3e): the refresh keeps radon NC passing but costs 1.4–8× elsewhere and no estimator clears the 1.2× kill clause; the re-seed ablation is queued. **WP6 (per-step efficiency) not started** — only the T5a command-buffer A/B exists and is running.
4. ~~GEV divergence diagnosis~~ — done: the rise is real and seed-stable (0.875, 0.350, 0.400, 0.200, 0.575 per 1k on the candidate; ≤ 0.100 on every shipped seed). What is still missing is *which coordinate* diverges, which needs one instrumented run reporting the divergence position's implied velocity.
5. **Default orientation: DECIDED — flip to `covariance`, same merge as the permutation** (owner ruling 2026-10-20, implemented on `wp/wp1` at `08357db`, merged tree green at 99 passed, merge held by the review gate). Hedging was impossible: the permutation is unconditional in that branch, so merging it while holding `precision` ships a state failing R̂ ≤ 1.01 on GEV in 5/5 seeds. Remaining G3 gaps: the GEV divergence rise has no mechanism, and no arm pair isolates orientation from the adapter iterate at C=1,024. The pinned test is now `test_default_orientation_is_covariance_and_mixes`.

## 8. Handoff — state of the tree and what is left

**Where things are.** `pf-r1` carries the WP0 artifacts, the merged reference posteriors, the ledger, this report, and two lead-authored reviews. `pf-r1-int` (branch, checkout `~/pf/int`, suite **99 passed**) is the merged measurement tree — wp8 + wp0refs + wp1 + wp2 + wp3 — and it is the source of every R1/T1/G3 number here. Unmerged: `wp/wp1` (metric fix **+ the approved flip**), `wp/wp2`, `wp/wp3`, `wp/wp5` (capacity, 108 passed), `wp/wp8`, `wp/wp9docs`, `wp/wp4`. Nothing pushed; `main` untouched at `ecd9ba1`.

**Blocked, honestly:** merges wait on two executable independent reviews per branch (owner ruling, A4 strict). Reviewer-model tasks on this box have no shell (`TOOLLAYER_DEAD`); worker-model tasks do, and two of those are reviewing `wp/wp1` now. One exception precedes the ruling: `wp/wp0refs` was merged on one task review plus one lead review, flagged in `results/STATUS.md`.

**Order from here:** `wp/wp1` reviews → merge the metric stack (wp1 → wp2 → wp3) → T2 at 128 chains → T3 CPU-versus-GPU (first version discarded: it rated a subset of parameters because the references store constrained arrays only for transformed variables) → T5a and the per-step cost split → the WP4 re-seed ablation → final T-tables.

**Questions for the reviewer.** (1) Is a 16× divergence-rate rise that still touches 0.04% of draws an acceptable price for passing R̂ on GEV? (2) Should G3's "means within 1σ" clause be restated in sd units, since it is unsatisfiable at 40k draws? (3) Delete `imm_order` after the merge, or keep it as a diagnostic? (4) WP5's 2 s warm-init bar adds two tasks' budgets together — what should each task own?
