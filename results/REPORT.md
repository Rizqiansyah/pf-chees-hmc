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
| 8 Schools | passes at W=100, 1,283 gradients/chain | **passes at W=50, 955** |
| logistic 1000×25 | **never** (b² 0.193 / 0.158 / 0.0498 at W=50 / 100 / 200) | **passes at W=50, 335 gradients/chain** |
| GEV | passes at W=100, fails at W=50 (b² 0.087) | no failing W; W=50/100 below the resolution floor, W=200 resolvable |
| radon NC | **never** (b² 106 / 64 / 12.1) | **passes at W=200, 1,521 gradients/chain** |

The plan's project target (R1 passing G-R1 with warmup ≤ 300 steps on all four models) is met by the corrected arm and unreachable by the shipped one. Split R̂ across the same cells reads 1.12–8.87, including in cells whose bias is below the noise floor — direct confirmation that R̂ cannot arbitrate at 10 draws per chain.

## 4. Decisions taken in this run

1. `imm_orientation` is an explicit, tested parameter (default unchanged for now); `imm_order` likewise. The default flip is owner-gated (G3) — recommendation and remaining evidence in §7.
2. **Nested R̂ is not a convergence gate in R1.** It reads 1.0040 on a cell with b² 0.4100. Measured reason: it is blind to bias shared by all chains (all-chains +1 sd reads 1.0004, identical to clean; every-other-superchain +1 sd reads 1.1135 at K=8). R1 Pathfinder deficit *is* common-mode, so G-R1 (b²) is the only gate; nested R̂ stays as a differential-drift/multimodality alarm. A proposed "precision ceiling at K" was refuted by the same measurement.
3. Vendored ChEES adapter with `iterate="raw"|"ma"` (default `ma`, behaviour-preserving). The raw iterate reaches the plateau trajectory length by ~step 100 (logistic L@100: 25–42 vs 8.3) without worsening the frozen value at W=2,000 (25.0 vs 26.8); `t0=π/2` adds nothing.
4. **Merged-tree rule.** `wp/wp1` + `wp/wp2` alone crashed every run (`WarmupInfo` vs `num_integration_steps`) while each branch was green alone. Since then, accept/kill decisions are re-verified on `pf-r1-int`.

## 5. Gates

| gate | status | basis |
|---|---|---|
| G-ORDER (permanent metric tests) | PASS | 17 tests in `tests/test_metric_orientation.py`; 12 fail on `ecd9ba1` |
| G-A R̂ ≤ 1.01 (permuted + covariance) | PASS 4/4 | §3 |
| G-B tail ESS / E-BFMI | PASS | min tail 1,207; E-BFMI 0.60–0.99 |
| G-R1 b² < 0.01 at C·D ≥ 2,000 | **PASS at W=50** on 8 Schools (955 gradients/chain) and logistic (335); GEV never fails; radon NC needs W=200 (1,521) | T1 sweep, 48 rows, `results/runs/T1__r1_sweep.jsonl`. Shipped orientation: **never** passes on logistic or radon at W=50/100/200 |
| G-R1-diag | REDEFINED | §4.2 |
| G-MEM peak ≤ 5.0 GB | PENDING | WP5 stage-memory table |
| G3 (default change: ≥5 seeds, no divergence increase, means within 1σ) | NOT MET on GEV, with the terms now separated | shipped arm fails R̂ ≤ 1.01 on **5/5 seeds** (to 1.0688); candidate passes 4/5 and halves mean error, but divergences rise seed-stably 0.025 → 0.400 per 1k. The "means within 1σ" clause is unsatisfiable as written (worst z 3.4 on a 0.026 sd shift) and should be restated in sd units |

## 6. Reference artifacts

`~/pf-runs/references/{8schools,gev,logit_1000x25,radon_noncentered}.npz` — CPU NUTS, 8 × 10,000, seed 42, manifest `results/references/MANIFEST.json`. The first delivery collapsed the parameter axis for vector variables and was rejected; the rebuilt files are per-parameter and independently re-derived (0 contract violations). The stored stat columns are display-only: the bias gate recomputes everything from the sample arrays, and that is recorded in the manifest as `single_authority`.

## 7. Open before this run can close

1. Reviews outstanding: `wp/wp1` (running), `wp/wp0refs` (statistical), `wp/wp8`, `wp/wp3` (statistical), `wp/wp9docs`, `wp/wp2`.
2. T1 R1 sweep (48 cells, running) and WP5 capacity: radon NC has never run above 64 chains on this card and the memory source is still NOT ESTABLISHED.
3. WP4 (ensemble metric) and WP6 (per-step efficiency) have not started; both depend on the merged metric fix.
4. ~~GEV divergence diagnosis~~ — done: the rise is real and seed-stable (0.875, 0.350, 0.400, 0.200, 0.575 per 1k on the candidate; ≤ 0.100 on every shipped seed). What is still missing is *which coordinate* diverges, which needs one instrumented run reporting the divergence position's implied velocity.
5. Recommendation on the default: flip to `covariance`. The counter-evidence is a divergence-rate rise on one model; keeping a metric that yields R̂ 12.1 on a four-parameter Gaussian is not defensible. `tests/test_metric_orientation.py::test_default_is_still_precision_orientation` is the test that flips with it.
