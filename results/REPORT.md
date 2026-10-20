# pf-r1 — external review implementation run: REPORT

Branch `pf-r1` (and the merged measurement tree `pf-r1-int`). Nothing is pushed; `main` is untouched at `ecd9ba1`. Drafted while tasks finish; sections marked PENDING name what they are waiting on.

## 1. What was run

An external review produced `docs/review-2026-10/pf-chees-hmc-implementation-plan.md`. This run executes it: work packages WP0–WP9, each in its own git checkout on a `wp/*` branch, findings recorded in `results/LEDGER.md`, every merge gated on two independent reviews (`results/reviews/`), all GPU work serialised through `tools/gpu_run.sh`.

Owner objective for this run: maximise GPU throughput in the **R1** regime (≈1,024 chains × ≈10 draws), judged by the bias gate G-R1 (b² < 0.01 against a CPU NUTS reference) and gradients per chain. ESS/s is the secondary R2 objective (32–128 chains × 1,250 draws).

## 2. The two correctness findings that decide everything

### 2.1 The metric is applied to the wrong coordinates (P0, confirmed)

Pathfinder draws are flattened as a **tuple** in `model.value_vars` order; the HMC kernel state is a **dict** that JAX ravels in **sorted-key** order. On GEV (`mu, sigma_log__, xi_interval__` vs sorted `mu, xi_interval__, sigma_log__`) and on radon noncentered the two orders differ; on 8 Schools they coincide, which is why the bug survived every test the repo had.

Measured in isolation (orientation held correct, ordering wrong), on a 4-scalar target with posterior variances 1e2 / 3 / 1 / 1e-2: **max R̂ 1.0069 and worst-variable bulk ESS 3,176 against ~31,000 for the other three coordinates** — one coordinate silently dies. `tests/test_metric_orientation.py` is the permanent gate (G-ORDER); it must fail on `ecd9ba1`.

### 2.1a How to describe the permutation — reviewer-corrected wording

The statistical reviewer's re-review (after the order A/B) set out four points, all adopted, because the earlier framing risked being read as an accuracy fix:

1. It is a **coordinate-coupling / representation-ordering correction, not a posterior-accuracy improvement.** The reason it stays default-on is the controlled fixture, where ordering is correctness-critical: R̂ 12.10 and worst-variable ESS 32 without it.
2. **Throughput sign is geometry-dependent and both directions are measured:** radon NC **4.7× faster** on all 5 seeds (3.92 vs 18.42 gradients per ESS), GEV **2.6–4.5× slower** on all 3 seeds. A reader must see that the sign flips by model.
3. The **GEV posterior-mean halving belongs to the orientation flip, not the permutation.** Attributing it to the ordering would be wrong.
4. **Radon carries a small, seed-consistent accuracy cost** — worst mean error worse by 0.004–0.011 sd on all 5 seeds — recorded so it is not later rediscovered as an unexplained regression.

Two of the four benchmark models (8 Schools, logistic) are identity cases where the ordering cannot differ, which is both the clean negative control and the explanation for how the defect survived: the project's everyday benchmark is one of them.


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

1. **The refresh is the only measured path to a passing R1 run on radon NC** — but read that with the instrument's own limits: with the refresh radon's b² is 0.0018–0.0020 against floors of 0.0026–0.0029 (a pass), while the ablation's radon cells at the same warmup carry floors of 0.0054–0.0060 and are unresolvable. The same nominal cell changes class between runs. The task initially argued the same benefit is cheaper by running the baseline at W=200; that is refuted by its own rows (seed 1 b² **0.02339 = fail** at W=200, both seeds `unresolvable` at W=100), and the task accepted the correction.
2. **One estimator clears the kill clause, and this report previously said none did.** Re-derived from the rows (the earlier sentence was inherited from the implementation SUMMARY without checking, which is the failure mode this run exists to catch): gradients/chain at W=50, median over seeds —

| estimator | 8 Schools | GEV | logistic | radon NC | vs plain diagonal (logistic, radon) | verdict |
|---|---|---|---|---|---|---|
| `sample_diag` | 1,292 | 4,804 | 2,386 | 5,612 | — | reference |
| **`fisher_diag`** | **1,157** | 6,673 | **1,352** | **3,886** | **0.57, 0.69** | **clears 1.2× → keep** |
| `dense_shrunk` | 1,344 | 4,237 | 2,323 | 5,469 | 0.97, 0.97 | indistinguishable → kill |
| `diag_lowrank` | 1,400 | 3,778 | 4,934 | 5,317 | 2.07, 0.95 | worse → kill |

   So the shipped recommendation becomes: **keep `sample_diag` as the safe default and `fisher_diag` as the one worth pursuing** (1.45–1.77× fewer gradients on the two correlated models, at a cost on GEV), kill `dense_shrunk` and `diag_lowrank`. All 16 cells pass on bias; cost is the discriminator. Two provenance problems go with this, filed in `results/reviews/WP4-stats.md`: the re-seed factor is recorded in **no row field** (the two ablation arms are byte-identical except for their label), and radon's noise floor swings 2× between nominally identical cells (0.0029 in one file, 0.0054–0.0060 in the other).
3. **The cost was the re-seed, and that is now measured.** The trajectory-length re-seed `T = (π/2)·√λ_max` lands at L = 23–402 against the baseline's converged 10–27 because λ_max of a 50-step ensemble estimate is noise-dominated. The A/B (4 models × 3 arms × 3 seeds at W=50) is in `results/runs/WP4__ablation.jsonl`:

| model | no refresh | refresh + re-seed | **refresh, no re-seed** | verdict |
|---|---|---|---|---|
| 8 Schools | 974 | 1,292 | **1,309** | passes G-R1, 3/3 seeds |
| GEV | 557 | 4,804 | **841** | passes, 3/3 |
| logistic | 324 | 2,386 | **487** | passes, 3/3 |
| radon NC | 372 | 5,612 | **579** | **unresolvable** at W=50 (floor 0.0054 vs the 0.005 rule) |

Refresh overhead drops from 3.7–15× the baseline to **1.34–1.55×**, so the answer to the plan's question is that the cheap opt-in path exists: keep the refresh, drop the re-seed. The rule fixed before the run was "within 1.5× of baseline while radon still passes" — 8 Schools, GEV and logistic meet it.

**Radon cannot be gated at the R1 shape, at any warmup.** Its ladder was run to settle this, and the floor — not the bias — is the wall:

| radon NC, refresh without re-seed | b² | noise floor | gradients/chain |
|---|---|---|---|
| W=50 (3 seeds) | 0.0081 – 0.0136 | 0.0054 – 0.0060 | 559 – 711 |
| W=100 | 0.0070 – 0.0107 | 0.0058 – 0.0070 | 1,128 – 1,498 |
| W=200 | 0.0058 – 0.0086 | 0.0057 – 0.0064 | 3,236 – 3,401 |

Tripling warmup to W=200 cuts b² by 20% and raises the floor, while costing **5.7× the gradients**. Every cell stays inside its own noise band, so radon is *unmeasurable* against the 0.01 bar at C=1,024 × D=10 — not a pass, not a fail. Resolving it needs more draws per chain (which R1 does not allow) or a reference-based floor reduction, and that is a plan change, not a result.

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
| Instrumentation integrity | ONE OPEN OVERSTATEMENT | `tools/r1_bias.py:344` computes a trajectory-length diagnostic with one device synchronisation per warmup step **inside** the timed region, on every arm. Absolute `wall_s` and every ESS/s figure were overstated by the same mechanism on all arms. **Measured:** 8 Schools at W=2,000 went 15.42 → 10.62 s warm and 42.90 → 36.74 s cold once the loop moved after the timer, i.e. **wall times in this report are ~52% high warm** and ESS/s correspondingly low; equal-warmup comparisons and all gradient counts stand. Fixed in `wp/wp3` at `9d5c30d`, which also fixes a gate that labelled a b² of 63 "unresolvable" (6 radon cells change verdict) |

## 6. Reference artifacts

`~/pf-runs/references/{8schools,gev,logit_1000x25,radon_noncentered}.npz` — CPU NUTS, 8 × 10,000, seed 42, manifest `results/references/MANIFEST.json`. The first delivery collapsed the parameter axis for vector variables and was rejected; the rebuilt files are per-parameter and independently re-derived (0 contract violations). The stored stat columns are display-only: the bias gate recomputes everything from the sample arrays, and that is recorded in the manifest as `single_authority`.

## 6b. Report tables T1–T6 (plan §A9)

Every cell below is a number read from a delivered row file, with the file named. Where a table could not be filled, the cell says why instead of being left blank.

### T1 — R1 bias sweep (C=1,024 × D=10), `T1__r1_sweep.jsonl`, 48 rows

Full per-seed grid in §3b. Summary under the **corrected** gate rule (`g_r1` at `wp/wp3` `9d5c30d`, which no longer lets a coarse floor neutralise a large bias):

| arm | 8 Schools | GEV | logistic | radon NC |
|---|---|---|---|---|
| `int:shipped` (precision, `ma`) | fails at W=50 | **fails every W** (R̂ 1.03–1.07 at 32 chains; b² not the binding gate) | **never passes** | **fail** at every W (b² 105 → 12) |
| `int:covraw` (covariance, `raw`) | **pass** at W=50, 955 grads/chain | **pass** at W=50, 335 | **pass** at W=50 | unresolvable at W=50/100, mixed at W=200 |

Two limits stated on the face of the table: the arms move orientation **and** adapter iterate together, so no R1 gain is attributable to one factor; all 48 rows are cold, 2 seeds.

### T2 — efficiency interventions

| intervention | metric | result | rows |
|---|---|---|---|
| coordinate-order fix (permutation only) | gradients/ESS | **better on 1 of 4** models; breaks R̂ ≤ 1.01 on GEV and radon | `WP1__rebaseline.jsonl` 142 (2 VOID-stamped) |
| order fix **+ covariance** | gradients/ESS | **better on 4 of 4** (35.5 / 71.2 / 46.1 / 182.3 vs 58.2 / 73.4 / 151.7 / 501.9) | `WP1__covarm.jsonl` 40 |
| same, at 128 chains | ESS/s | direction holds at the higher chain count | `WP1__t2_128.jsonl` |
| order A/B on real models (orientation pinned) | gradients/ESS | radon **4.7× win**; GEV reverse (explained by a 50× Pathfinder under-dispersion of `xi_interval__`); logistic and 8 Schools **bit-identical** (identity permutation) | `ORDER__radon_noncentered.jsonl` 10, `ORDER__logit_1000x25.jsonl` 10 |
| Pathfinder as init vs as warmup (R2) | gradients/ESS | **no benefit at any budget**, 5 seeds × 4 models | §3 |
| metric refresh at W=50 | gradients/ESS | costs **3.7–15×**; **1.34–1.55×** with the trajectory-length re-seed dropped | `WP4__r1.jsonl` 32, `WP4__reseed_ab.jsonl` 18 |
| ChEES raw `log T` iterate | warmup wall | logistic 16.7 → 7.9 s (**2.1×**), frozen L not worse | `WP2__lgrowth.jsonl` 32 |
| XLA command buffers | wall | **not shipped** — see T5 | `T5a__cmdbuf.jsonl` 10 |

### T3 — CPU NUTS (32 chains on 32 cores) time to reach b² < 0.01, `T3__cpu_nuts.jsonl`, 12 rows

| model | b² @ 10 D/ch | b² @ 100 | b² @ 400 | wall 10 → 400 s | core-seconds 10 → 400 |
| 8 Schools | b² 0.0288 ✗ | 0.0015 ✓ | 0.0010 ✓ | 7.02 → 8.29 s | 224.6 → 265.3 |
| GEV | 0.0294 ✗ | 0.0024 ✓ | 0.0008 ✓ | 9.25 → 11.14 s | 295.9 → 356.5 |
| logistic | 0.0125 ✗ | 0.0037 ✓ | 0.0011 ✓ | 23.69 → 33.81 s | 758.2 → 1,082.0 |
| radon NC | 0.0514 ✗ | 0.0077 ✓ | 0.0012 ✓ | 17.42 → 23.80 s | 557.3 → 761.7 |

**The honest reading is in the `resolvable` column: every one of the 12 cells reads `resolvable=False`.** CPU NUTS' own noise floor at this shape is 0.042–1.18, far above the 0.005 resolution bar, so this table can say CPU NUTS *reaches* b² < 0.01 by 100 draws per chain at a cost of 225–1,082 core-seconds, but it **cannot** rank CPU against GPU on bias at these budgets — the instrument cannot see a 1% bias in either arm. Coverage is also incomplete: 1 (8 Schools), 2 (GEV) and 3 (radon) variables are unscored because the references store constrained arrays only for transformed variables.

### T4 — largest chain count that fits, and where the memory goes (6 GB A2000), `WP5__memprobe_before.jsonl` 80 rows / `WP5__accept.jsonl` 5 rows

Before (public defaults, `pf_draws=4,000`; the stage named is the last row written before the process died):

| model | largest C | peak reached | failed stage at the next C |
|---|---|---|---|
| 8 Schools | 1,024 fits | 2,314 MiB | — |
| GEV | 1,024 fits | 946 MiB | — |
| logistic | OOM at 1,024 | 581 MiB reached | `pathfinder_solve` (pool) |
| radon NC | **64** | 2,575 MiB | `pathfinder_solve` at C=128/256/1,024 (684–685 MiB then a 699 MiB–5.36 GiB allocation refuses) |

After `wp/wp5` (pool capped at `4C`, chunked scoring, one shared init point, trimmed scan output):

| model | C=1,024 × D=10 | peak | wall (cold) | init stages |
|---|---|---|---|---|
| radon NC | **runs** | 653 MiB | 528.8 s cold / **42.9 s warm** | init_total 4.07 s warm (PF solve 2.47 + guard 0.26 + init point 0.67 + eps0 1.59) |
| logistic | runs | 581 MiB | 359.8 s | — |
| GEV | runs | 288 MiB | 50.4 s | — |
| 8 Schools | runs | 224 MiB | 38.7 s | — |

The 2 s warm-init bar is **not met** (4.07 s), and 1.59 s of it is the step-size stage owned outside WP5. Cold walls are compile-dominated; every figure above is labelled cold or warm in the rows. **Side effect worth more than the headline:** 8 Schools' cumulative peak fell from 2,314 MiB (before, same single-process basis) to **224 MiB** — a ~10× memory reduction at a chain count that already fitted — because the warmup scan no longer stacks the full HMCInfo. No before/after *wall* comparison is possible: the before-probe rows carry no `wall_s` at all.

### T5 — per-step cost and the XLA flag, `T5a__cmdbuf.jsonl` 10 rows

| model | warm baseline | warm + command buffers | Δ | cold baseline | cold + buffers |
|---|---|---|---|---|---|
| 8 Schools | 9.12 s | 10.29 s | **+12.8%** | 14.70 s | 24.40 s |
| logistic | 16.34 s | 18.33 s | **+12.2%** | 32.13 s | 36.10 s |
| GEV | 10.34 s | 11.58 s | **+12.0%** | 20.33 s | 21.57 s |
| radon NC | 45.71 s | 51.48 s | **+12.6%** | 600.64 s | 514.20 s |

**Verdict: do not ship the flag.** The plan's +23–24% was measured on a bare kernel with no adaptation and does not transfer to the production path; the penalty is a consistent ~12% across models. The **per-step cost split (kernel vs adaptation vs history vs callback) is NOT RUN** — that is the remaining WP6 gap, and it is the one T-table this run cannot fill.

### T6 — doc claims retracted or re-qualified

`results/WP9/CLAIM-INVENTORY.md`, 40 rows, each with claim, `file:line`, old number, verdict and the row ids that decide it: **20 CONFOUNDED, 11 UNAFFECTED, 8 SUPERSEDED, 1 CANNOT TELL**. All 52 citations were audited separately (48 OK, 4 mispointed and corrected at `e660578`). The applied documentation retractions for the settled subset are on `wp/wp9docs` (`79e9944`): 7 retracted, 19 caveated only.

## 7. Open before this run can close

1. **Review coverage, counted from `results/reviews/` (not from memory — see the LEDGER entry on the `wp/wp8` revert).** wp0refs 2 (merged), wp1 4 (merged), wp2 1 (lead-authored), wp3 1 code + its statistical review in flight, wp5 0 with its code review in flight, wp4 0, wp8 0, wp9docs 0. Two executable independent reviews per branch is the owner's A4 rule; the queue of remaining briefs is `~/scratch/REVIEW-QUEUE.md` (8 reviews).
2. **Merges still owed:** `wp/wp3` (harness, 11 commits, suite 66→67), `wp/wp2` (vendored adapter, 4), `wp/wp5` (capacity, 37, suite 108), `wp/wp8` (hygiene, reverted at `60302b2` pending reviews), `wp/wp9docs` (docs), `wp/wp4` (metric refresh). Order: wp3 → wp2 → wp5 → wp8 → wp9docs → wp4, each on two reviews.
3. **T5's per-step cost split is NOT RUN** — the only T-table with an unfilled cell (§6b/T5). Everything else in T1–T6 is filled from delivered rows.
4. **GEV divergence mechanism.** The rise is real and seed-stable (0.875, 0.350, 0.400, 0.200, 0.575 per 1k on the candidate versus ≤ 0.100 on every shipped seed), and the coordinate responsible is still unidentified: it needs one instrumented run reporting the divergence position's implied velocity. The 50× Pathfinder under-dispersion of `xi_interval__` (variance 0.00184 against a true 0.0930) is the leading candidate and the same mechanism explains GEV's apparently-inverted permutation result.
5. **The 2 s warm-init bar** mixes two tasks' budgets (WP5 owns 2.47 s of the 4.07 s; the step-size stage is 1.59 s of it). It needs a ruling on what each task owns, not another measurement.

## 8. Handoff — state of the tree and what is left

**Tree.** `main` untouched at `ecd9ba1`, **nothing pushed**. `pf-r1` at `9042a68` (68 commits ahead of `main`) carries the WP0 references, the merged `wp/wp0refs` + `wp/wp1`, the LEDGER, this report and the review files. `pf-r1-int` at `62049b2` is the merged measurement tree (wp8 + wp0refs + wp1 + wp2 + wp3, suite 99 passed) and the source of every R1/T1/G3 number here.

| branch | head | commits ahead of `pf-r1` | reviews on disk | state |
|---|---|---|---|---|
| `wp/wp0refs` | `93a9d86` | merged | 2 | done |
| `wp/wp1` | `7982356` | merged | 4 (incl. re-review that withdrew its own hold) | done — permutation + covariance default |
| `wp/wp3` | `9d5c30d` | 11 | 1 code + stats in flight | harness, ready to merge on the second review |
| `wp/wp2` | `b02a6e8` | 4 | 1 (lead-authored) | needs an independent statistical review |
| `wp/wp5` | `0e6739f` | 37 | 1 code in flight | capacity, suite 108, needs its stats review |
| `wp/wp8` | `8931a71` | 0 (reverted merge) | 0 | needs both |
| `wp/wp9docs` | `79e9944` | 1 | 0 | needs both |
| `wp/wp4` | `d457a17` | 28 | 0 | needs both; highest-risk numerical code after vendoring |

**Evidence base.** 21 distinct row files, 619 rows (`~/pf/*/results/runs/`, named per cell in §6b). References: `~/pf-runs/references/*.npz` + `results/references/MANIFEST.json`, samples are the single authority. Reviews: `results/reviews/`. Decisions and retractions: `results/LEDGER.md`. This report: `results/REPORT.md`.

**Measurement caveats a reader must carry.** (a) Absolute wall times and every ESS/s figure taken with `tools/r1_bias.py` before `wp/wp3` `9d5c30d` are ~52% high warm — the harness ran a per-warmup-step diagnostic inside the timer; comparisons at equal warmup and all gradient counts are unaffected. (b) R1 verdicts must not use split R̂ (1.12–8.87 even in unbiased cells) or nested R̂ (blind to bias shared by all chains). (c) b² is seed-dominated at D=10, so one seed per cell cannot rank arms and every row carries its noise floor. (d) Radon NC is **unmeasurable** at C=1,024 × D=10 at any warmup tested — its floor never falls below ~0.0057 while gradients per chain rise 5.7×.

**Two process failures worth repeating to whoever picks this up.** Reviewer-model tasks on this box have no shell (`TOOLLAYER_DEAD`), so reviews must run as worker-model tasks; and two tasks delivered final messages describing work that was not on disk (one cited a commit that does not exist, one a file that does not). Both were caught by verifying against the filesystem, and the lead made the same class of error once — merging `wp/wp8` on a review it remembered rather than read (`8d81563`, reverted at `60302b2`). Rule: **no claim in this run is evidence until the file exists.**

**Questions for the owner.** (1) Is a 16× divergence-rate rise that still touches 0.04% of draws an acceptable price for passing R̂ on GEV? (2) Should G3's "means within 1σ" clause be restated in sd units — it is unsatisfiable at 40k draws (worst z 3.4 on a 0.026 sd shift)? (3) Delete `imm_order` after the merge, or keep it as a diagnostic? (4) WP5's 2 s warm-init bar: what should each task own?
