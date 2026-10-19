# LEDGER — append-only decisions and evidence

Format: date · WP · decision · evidence (row ids + commands) · gate verdicts · SHA

## 2026-10-19 · WP0 · Branch `pf-r1` created from `ecd9ba1`; `main` untouched, nothing pushed
Commands: `git switch -c pf-r1`; review docs copied to `docs/review-2026-10/`; the six review probes
to `tools/probes/`; `~/pf-o1-stages.py` to `tools/stage_timers.py`.
`ref/*.pdf` deliberately not committed (A3).

## 2026-10-19 · WP0 · GPU discipline adopted as code
`tools/gpu_run.sh` wraps `flock /tmp/pf-gpu.lock` and sets `PF_GPU_LOCK=1`;
`tools/_common.require_gpu_lock()` raises in a timed cell that forgot it, so an unlocked timing
fails loudly instead of producing a void number (A5).

## 2026-10-19 · Baseline check before acting on Part B
nvidia-smi at run start: no foreign compute apps. Checkout/env recipe per A3 verified by the reviewer.

## 2026-10-19 · WP0.2 · delivered, then REJECTED on re-verification; task re-tasked
Task reported 4/4 PASS (`82bb771` on `wp/wp0refs`). I re-verified against the .npz myself (CPU, `~/scratch_verify_refs3.py`,
`~/scratch_theta_probe.py`): max R̂ and min bulk ESS reproduce to the digit (1.00025 / 1.00017 / 1.00049 / 1.00048) and all
four sha256 match. But the per-variable statistics are collapsed over the parameter axis: in `8schools.npz`,
`u__theta` is (8, 10000, 8) with per-parameter means ≈ [0.28, 0.04, -0.12, 0.01, -0.22, -0.11, 0.30, 0.05], while
`mean__theta` = 0.02917 (the grand mean over the 8 schools) and `mcse_x__theta` is the scalar 0.00298. Scalar variables
and all of GEV are correct, so the task's own gate could not see it.
Decision: **do not merge**. WP3 scores b² per parameter against `mean_ref`/`Var_ref`, so a scalar reference for `theta`
would silently mis-score 8 of 10 parameters on 8 Schools and 170 of 175 on radon NC — this is the run's headline gate.
Re-tasked the task to keep the parameter axis on mean__/var__/mcse_x__/mcse_x2__, re-run the affected models at the same
seeds, and add a permanent shape-guard test (stat shape must equal sample shape minus the two leading axes).
Standing rule reconfirmed: a task number that gates work is re-derived from the artifact, not accepted from a report.

## 2026-10-19 · Tasks · WP8 split into conflict-free items only
`wp/wp8` (steady-tier) given only the statistical-correctness test and the README `base()` wording, with `sampler.py` and
`utils.py` explicitly off-limits, because WP1 (metric block) and WP3 (start selection) are editing `sampler.py` and WP2
will own `utils.py`. The remaining WP8 items (stage-kwarg whitelist, mutable defaults, dead xmap branch, Halton bits)
are held until those tasks merge. Running: 2/2 fast-tier (WP1, WP3), 2/3 steady-tier (WP0.2 rework, WP8).

## 2026-10-19 · tools · peak device memory is process-cumulative, not per cell
Verified on the A2000 under the lock (`tools/gpu_run.sh`, first use): `jax.devices()[0].memory_stats()` exposes
`peak_bytes_in_use` but the device has **no `reset_memory_stats()`** in JAX 0.11.2, and on CPU `memory_stats()`
returns `None`. So a row's peak memory read at row-creation time is meaningless and a value read after a cell is the
process high-water mark, not that cell's. `tools/_common.py` now documents this and adds `set_peak_mem(row)` to stamp
after the cell; per-cell attribution requires one process per cell (WP3 found the same defect independently).

## 2026-10-19 · WP0.2 · RE-DELIVERED and independently verified; merge held for two reviews
Task `67da0ca`: `x.mean(axis=(0,1))` / `x.var(axis=(0,1), ddof=1)`, `_mcse` returns the full array and raises on a shape
mismatch, an in-script guard before savez, and a permanent `tests/test_reference_npz.py` asserting
stat shape == sample shape[2:] for every npz (plus a synthetic legacy-format file that proves the assertion goes red).
My own re-derivation (`~/scratch_verify_refs6.py`, CPU): **0 contract violations** across all four files — every stat
array per-parameter, stored mean/var match a fresh recomputation to 1e-8, every MCSE finite and > 0, sha256 match, all
draws (8, 10000). Gate numbers unchanged (R̂ 1.0002 / 1.0002 / 1.0005 / 1.0005, min ESS 52,051 / 38,930 / 19,838 / 15,440),
GEV caveat intact (18 divergences at target_accept 0.99).
Decision: references are **usable for scoring now** (files are on disk outside git, so WP3 is unblocked), but the branch
merge into `pf-r1` waits for the two A4 reviews (code + stats, both fast-tier), and both fast-tier slots are occupied.
Queued as the first thing to run when a fast-tier slot frees.

## 2026-10-19 · CONTRADICTION · Part B §0.1 "the default metric is the precision, not the covariance" is refuted for the kernel path
Command: `~/scratch_metric_semantics.py` (CPU, x64, 20,000 momentum draws), run against the installed wheel.
Set `Sigma = [[4, 1.2], [1.2, 1]]` and pass `imm = inv(Sigma)` to `metrics.default_metric`, exactly as `sampler.py:406` does:

    passed array (inv Sigma) diag : [0.391 1.562]
    true Sigma diag               : [4. 1.]
    momentum variance             : [4.007 1.004]      -> matches Sigma, NOT inv(Sigma)

So the array BlackJAX is handed really is M⁻¹, and `inverse_mass_matrix = jnp.linalg.inv(cov)` produces the textbook
choice M = Σ_posterior: momentum is drawn with the posterior covariance. Applying Part B's fix verbatim
(`inverse_mass_matrix = cov`) would make momentum ~ N(0, Σ⁻¹) — narrow directions get large momenta — i.e. it would
*invert* a metric that is currently correct.
Reconciling with Part B's §2c numbers (frozen ε 0.0021 → 0.66, nominal L 136 → 26 on logistic): both orientations are
valid HMC and preserve the target, so a change in ε does not prove an inversion. With the correct metric ε scales like
Σ, so a small ε and a large step count are the expected behaviour for a narrow posterior, not a pathology. That leaves
§0.3 (the trajectory-length rate limit) standing on its own evidence and §0.1 refuted as stated.
Not acted on silently: WP1 is instructed to settle it by posterior moments on a known-variance Gaussian, which is the
only arbiter that matters, and to ship the **permutation** fix regardless — `order_check.py` (my own run: GEV and radon
value-vars order ≠ JAX dict-flatten order; kinetic energy 0.5 where 50 was intended) is independent of this question.
If WP1's moment test shows the `cov` arm recovering the analytic covariance and the `inv(cov)` arm not, Part B wins and
this entry is superseded with its numbers.

## 2026-10-19 · tools · `bench_ladder.load_notebook_model` leaked a sampling stub into the whole process
`tools/bench_ladder.py:106-109` replaced `pm.sample`, `pm.sample_prior_predictive`, `pm.sample_posterior_predictive`
and **`pfcheeshmc.sample_pf_chees_hmc`** with `_tiny_idata` to keep notebook loading from sampling, and never restored
them. The harness survived only because it cached `REAL_SAMPLE_PF` at `:49` and calls that. Any other code that called
`pfcheeshmc.sample_pf_chees_hmc(...)` after a notebook load in the same process got a tiny fake InferenceData: fast wall
time, no divergences, no diagnostics, no error. Found by the WP3 task; confirmed by reading `:106-109` and the absence of
any restore. Fix: stub inside `try:` / restore in `finally:`. Verified by loading 8 Schools and checking identity of both
entry points (`restored: True`, `pm.sample restored: True`), and pinned by
`tests/test_bench_ladder_stub_restore.py` (2 passed, 4.15 s, CPU).
Consequence for open work: every harness that loads ladder models and then samples through the public API must assert
`pfcheeshmc.sample_pf_chees_hmc is bench_ladder.REAL_SAMPLE_PF` before each cell until this merge reaches its checkout.

## 2026-10-19 · WP1 · the three imm_source arms do NOT share an orientation (task-reported, measurement pending)
`mc` passes `inv(cov)` = M⁻¹ (correct, matches my momentum measurement), but the `lbfgs_lowrank` and `fisher_lowrank`
branches expand `D(I + U(Λ−I)Uᵀ)D` from the L-BFGS / Fisher factors, which **is** the inverse Hessian ≈ Σ, and feed that
into the same slot. So the three arms have been consuming the array with opposite conventions.
Consequence: every `mc` vs `lbfgs_lowrank` vs `fisher_lowrank` comparison in the repo — including the README's
1.22x / 1.96x / 0.19x — compared differently-oriented metrics, not just differently-estimated ones. Part B §0.1 pointed at
the wrong line but the underlying "metric orientation is a mess" complaint is real, and it lives in the non-default arms.
Decision: WP1 makes orientation an explicit, measured choice across all three arms (permutation fix unconditional;
arbitrated by posterior moments on a known-variance target, not by ε magnitude). WP9's retraction of the Fisher numbers
should cite the orientation mismatch as the mechanism.

## 2026-10-19 18:05 · WP1 · RETRACTION of entry 699be6d: the metric orientation IS a correctness-class failure of the shipped default

**What I logged at 699be6d and why it was wrong.** I fed `inv(Sigma)` to `metrics.default_metric`, drew 20,000 momenta, and found `Var(p) = Sigma`. From that I concluded the argument really is `M^-1`, that the shipped metric was the textbook choice, and that Part B §0.1 was a tuning question. The first half is still true: the kernel does read the array as `M^-1`. The conclusion drawn from it is not. A single measurement of one consumer cannot decide a property of a pipeline where the same array is consumed twice, and the shipped default fails the project's own convergence bar by three orders of magnitude in ESS.

**The measurement that settles it** (WP1 fixture `~/scratch/wp1_2x2b.py`; four scalar coordinates, posterior variances `zeta 1e2, delta 3, alpha 1, beta 1e-2`; `value_vars` order `[delta, beta, zeta, alpha]` vs kernel order `[alpha, beta, delta, zeta]`, i.e. a permutation that is not a reversal; 32 chains, 500 warmup, 1,000 draws, seeds 5/6/7, CPU, tree `/home/akbar/pf/wp1` @ `4460dc7`). The lead re-ran it independently (background task `b92a63722`); every one of the twelve cells reproduces:

| order | orientation | eps | L | max R̂ | min bulk ESS | per-var ESS (alpha/beta/delta/zeta) |
|---|---|---|---|---|---|---|
| kernel | covariance | 1.12–1.21 | 4–5 | **1.0020–1.0022** | **30,121–33,495** | all ≈ 31,000 |
| kernel | precision | 0.0130–0.0140 | 4–6 | **12.10–13.08** | **32–38** | 32–38 on every wide coord |
| pathfinder | covariance | 0.236–0.246 | 12–45 | 1.0069–1.1451 | 166–3,176 | one coord collapses |
| pathfinder | precision | 0.0132–0.0142 | 4–6 | 10.37–11.71 | 33 | 33 everywhere |

**Two factors, two signatures.** An ordering error collapses a *specific* coordinate (pathfinder+covariance: ESS 166 against ~30,000 for the others). An orientation error collapses *every wide coordinate at once* (ESS 32–38 across the board). That is why the earlier 3-scalar fixture with a symmetric variance ladder and reverse-sorted names was degenerate and misread the two factors as interchangeable: reversing a symmetric ladder mimics inversion.

**Mechanism: NOT ESTABLISHED.** The candidate is that the same array is also consumed by the ChEES criterion whitening (`pfcheeshmc/utils.py`), so the step-size and trajectory-length adaptation is calibrated under the assumption that it is Sigma; consistent with this, the precision arms freeze eps ≈ 0.014 and the covariance arms eps ≈ 1.2 on the identical target. I have not verified that in source and I am not asserting it. What is established is the behavioural fact, not its explanation.

**Consequences.**
1. `699be6d`'s verdict "the metric is already correct; §0.1 is tuning" is retracted. Part B §0.1's *effect* is confirmed on a controlled fixture; its *stated mechanism* (the kernel reads `imm` as Sigma) remains refuted by the momentum measurement above.
2. The shipped default is unchanged so far (WP1 made orientation an explicit `imm_orientation` knob, default `precision`, and reordered coordinates for all three `imm_source` values). Every historical number in `docs/ladder-results.md` and `README.md` therefore comes from the failing orientation, except where a model's metric is near-isotropic or the permutation is the identity (8 Schools).
3. Flipping the default still needs the real-model 5-seed evidence (matrix in progress: legacy unpermuted+precision, wp1 permuted+precision, wp1 permuted+covariance, plus the two low-rank arms) and the owner's decision per plan A10. Not established yet; do not flip on the fixture alone.

## 2026-10-19 18:25 · WP3 · G-R1-diag does not fire where it is supposed to (lead read of the delivered rows)

WP3 delivered its self-test and I re-derived its rows rather than trusting the report. The self-test reproduces §2e (b² 2.711 vs 2.71 starts-only, 0.141/0.045/0.0022 vs 0.138/0.039/0.0027 logistic, ε 0.134–0.147 vs 0.14, L 20.4/56.4/8.1/26.8 vs 22/55/8/28, gradients/chain 1,151 vs 1,200). Two things in the same rows contradict conclusions the task drew.

**1. The nested-R̂ gate passes on a cell that is grossly biased.** `results/runs/WP3__nested_k8.jsonl`, 8 Schools, C=1,024, D=10, W=25, `superchains=8`, seed 1: **nR̂ = 1.0040 (below the 1.01 bar), split R̂ = 3.2875, b²_max = 0.4100.** The cell this positive control was built to prove — burn-in dominates at W=25 — is exactly the cell where nested R̂ says "converged". A diagnostic gate that clears a 0.41-variance-unit bias is not a gate. Candidate mechanism, NOT ESTABLISHED: `superchains` here *replicates* K starts across C/K chains, so chains inside a superchain share a start by construction; if the paper's variance decomposition assumes independent initialisations per superchain, our replication scheme makes the between-superchain spread insensitive to the burn-in bias it is supposed to expose. The discriminating test is nR̂ at the same W with unreplicated starts (K=1 / `init_pick=iid`): if it rises there, the estimator is fine and the start design defeats it, which matters for every R1 configuration we intend to ship.

**2. "Divergences cannot be an R1 gate" is only true of sampling divergences.** All 30 grid rows have `divs_per_1k = 0.0` — but `warmup_divs` is 626–647 on 8 Schools at C=1,024 (≈0.6 per chain) and 238–1,724 on logistic. That reproduces §2c's "~1 divergence per chain in warmup" observation, which the plan asked WP1 to check and which is therefore now independently confirmed. The correct statement is that *sampling* divergence rate is a null test at D=10, while *warmup* divergence is non-null and is the only divergence signal R1 produces.

**Consequences.** G-R1 stays the primary gate (b² with its floor, which the harness does emit). G-R1-diag is downgraded to informational until item 1 is resolved, and no accept/kill decision in WP4/WP5 may cite nested R̂ as evidence of convergence. WP3 is re-tasked with the K=1 control when a slot frees.

## 2026-10-19 18:42 · WP1 · the orientation 2×2 on a REAL model (GEV) confirms the retraction — found in the task's uncommitted rows during the pause

The fixture settled the mechanism; this is the discriminating evidence on a model where the tuple-vs-dict permutation is not the identity. Rows read from `~/pf/wp1/results/runs/WP1__gev_2x2.jsonl` (24 rows, 4 arms x 3 seeds x rep 0/1, C=32, GPU, warm and cold identical to the 3rd digit so no timing contention in these fields). `kernel` = permuted into kernel order, `pathfinder` = the old unpermuted order; `precision` = the shipped default, `covariance` = Part B §0.1's fix.

| arm | max R̂ | frozen ε | nominal L | realised steps/draw | min bulk ESS |
|---|---|---|---|---|---|
| permuted + **precision** (shipped) | **1.0313–1.0518** | 0.00089–0.00092 | 7–8 | 4.0–4.4 | **579–838** |
| permuted + **covariance** | **1.0017–1.0055** | 0.767–0.832 | 11–18 | 5.5–9.2 | **5,818–16,628** |
| unpermuted + precision | 1.0074–1.0142 | 0.00086–0.00087 | 30–43 | 15.0–21.9 | 2,369–5,196 |

Consequences:
1. On GEV the shipped orientation fails the project's own G-A bar (R̂ ≤ 1.01) at every seed and yields ~20x less worst-variable ESS than the covariance orientation at the same chain count. `a03dc0f`'s conclusion now holds on a real model, not only the 4-scalar fixture.
2. The ε and L ratio matches the plan's §2c direction (ε jumps ~900x, L falls), so the reviewer's qualitative story is right even though §0.1's stated mechanism about what the kernel reads remains refuted.
3. These rows are NOT yet committed by the task (untracked file). Do not treat the arm as final until WP1 commits them; the numbers above are read directly from the file, which is why they are recorded now.

Pause note: box yielded to another user at 18:38, GPU released (99 MiB, 0%). No new measurement was run to produce this entry.

## 2026-10-19 21:55 · WP1 · the completed 4-model re-baseline says the permutation fix alone FAILS acceptance and must not ship without the orientation flip

Lead's own aggregation of warm rows (rep 1, 5 seeds each) from `~/pf/wp1/results/runs/WP1__rebaseline.jsonl` and `WP1__covarm.jsonl`. C=32, D=1,250, W=2,000, GPU. `legacy:mc` = unpermuted + precision (what is shipped today). `wp1:mc` = permuted + precision (permutation fix only). `wp1:mc:covariance` = permuted + covariance (Part B §0.1 as written).

| model | arm | max R̂ | min bulk ESS | ESS/s | gradients/ESS | divs/1k | ε | nominal L | realised steps |
|---|---|---|---|---|---|---|---|---|---|
| 8 Schools | legacy:mc | 1.0017 | 22,120 | 2,085 | 58.2 | 0.00 | 0.185 | 42 | 21.3 |
| | wp1:mc | 1.0020 | 22,082 | 2,166 | 57.2 | 0.00 | 0.187 | 42 | 21.4 |
| | **wp1:mc:covariance** | 1.0014 | 20,469 | 2,226 | **35.5** | 0.00 | 0.524 | 22 | 11.5 |
| GEV | legacy:mc | 1.0023 | 8,102 | **403** | 73.4 | 0.23 | 0.0009 | 54 | 27.0 |
| | wp1:mc | **1.0465 FAIL** | 679 | 44 | 295.8 | 0.03 | 0.0009 | 8 | 4.1 |
| | **wp1:mc:covariance** | 1.0074 | 4,983 | 303 | 71.2 | 0.47 | 0.833 | 10 | 5.1 |
| logistic 1000×25 | legacy:mc | 1.0016 | 15,462 | 417 | 151.7 | 0.00 | 0.0020 | 129 | 64.7 |
| | wp1:mc | 1.0021 | 14,058 | 403 | 163.8 | 0.00 | 0.0021 | 122 | 61.0 |
| | **wp1:mc:covariance** | 1.0014 | 17,163 | **1,048** | **46.1** | 0.00 | 0.656 | 27 | 13.6 |
| radon NC | legacy:mc | 1.0069 | 3,845 | 98 | 501.9 | 0.00 | 0.0025 | 75 | 37.6 |
| | wp1:mc | **1.0142 FAIL** | 2,418 | 28 | 2,278.8 | 0.00 | 0.0007 | 268 | 134.0 |
| | **wp1:mc:covariance** | 1.0066 | 4,468 | **171** | **182.3** | 0.00 | 0.971 | 27 | 13.7 |

**Verdicts.**
1. **The permutation-only fix fails the plan's own acceptance criterion.** Gradients/ESS against `legacy:mc`: 8 Schools 57.2 vs 58.2 (better), logistic 163.8 vs 151.7 (worse), GEV 295.8 vs 73.4 (4.0x worse), radon 2,278.8 vs 501.9 (4.5x worse) — better on 1 of 4, criterion requires >= 3 of 4. It also breaks R̂ ≤ 1.01 on GEV and radon, which the shipped code passes. Shipping `wp1:mc` alone would be a regression.
2. **Permutation + covariance is the shippable combination.** Gradients/ESS better on 4 of 4 (35.5/71.2/46.1/182.3 vs 58.2/73.4/151.7/501.9), ESS/s better on 3 of 4 (logistic 2.5x, radon 1.7x, 8 Schools +7%), R̂ within the bar on all four.
3. **GEV is the honest counter-example and it must be reported, not smoothed.** There the buggy shipped arm accidentally performs well (ESS/s 403, R̂ 1.0023); correcting both factors gives R̂ 1.0074 and ESS/s 303, and the divergence rate rises 0.23 -> 0.47 per 1k. G3's "divergence rate must not rise" is therefore NOT met by GEV, which is the one gate a default change needs.
4. **§2c reproduces.** logistic ε 0.0021 -> 0.656 and L 122 -> 27 (reviewer: 0.66 / 26); 8 Schools ε 0.187 -> 0.524, L 42 -> 22 (reviewer: 0.51 / 22).
5. **The earlier `imm_source` comparisons are now explained.** `wp1:lbfgs_lowrank` on GEV gives R̂ 1.0400 and 238.9 gradients/ESS, and on radon 1.0146 with 2,353.7 — the same broken-orientation failure, confirming `9f892b2` that those arms were not comparable to `mc`.

**Consequence for the default.** The direction is clear and measured on real models, but the default change is gated on G3 and the owner's decision, and GEV currently fails the divergence clause of G3. So: land the permutation and the explicit orientation knob; do not flip the shipped default in this commit; the flip needs a GEV-specific look at why correcting the orientation raises its warmup divergence rate.

## 2026-10-19 22:15 · WP3 code review · one blocking defect, and G-R1-diag is a plan-level defect with a mechanism

Review at `results/reviews/WP3-code.md`. Verdict hold. Findings I verified are actionable:

1. **BLOCKING.** `tools/r1_bias.py:463-467` writes `grads_per_chain_warmup`/`_sampling` as `np.median(...)` across chains, while the exact per-chain sums computed at `:341-346` are discarded. Plan A8 requires EXACT counts defined as the sum of `num_integration_steps`. Every T1 column built on that field is wrong by whatever chain-to-chain variation exists. Fix in `wp/wp3`, not by relabelling.
2. **The delivered `WP3__selftest.jsonl` grid rows are at git_sha `f82d1ba`, which predates the `grads_per_chain_total` key added at `542c12c`.** Those rows must be read as warmup+sampling combined. Confirmed by inspection: the field is absent from the file.
3. **G-R1-diag cannot do the job Part B assigns it, and the reason is arithmetic, not a bug.** The harness's `superchain_ids` are correct (`:387`, matches the sampler's `jnp.repeat(pick_global[:K], C//K)` at `sampler.py:391-399`), but nested R̂'s between-superchain term scales as `B/n` with `n` = draws *per superchain*. At K=8, C=1024, D=10 that is n=1,280, so a 1-sd offset contributes about 1/1,280 → nR̂ ≈ 1.0008 before any bias. `tests/test_nested_rhat.py:45-57` already pins exact shift-invariance. That is why the same value, 1.0040–1.0043, came back at b² 0.4100 and at 0.00063.
   Consequence adopted: **nested R̂ is reported as a multimodality alarm with a stated K ceiling, never as the R1 convergence gate.** No WP4/WP5 accept-kill may cite it as convergence evidence. R1 convergence rests on b² against the reference plus the printed floor.
4. Second-order, recorded: the noise floor at `:268-270` does not propagate reference MCSE and assumes cross-chain independence, which the `superchains` arm deliberately violates, so the floor is mildly optimistic there.

A positive control is running now on GPU (`WP3__nested_k8.jsonl`, C=1024, K=8, W=25 and 100): if nR̂ fails to move at K=8 and moves at larger K, the blindness is measured directly rather than inferred.

## 2026-10-19 22:30 · CONTRADICTION against the WP3 code review · nested R-hat is not precision-limited at R1 chain counts; it is common-mode-limited

The review (`results/reviews/WP3-code.md`, item 7) attributed G-R1-diag's failure to fire (nested R̂ 1.0040 at b² 0.4100) to a precision ceiling: the between-superchain term scaling as `B/n` with `n` = draws per superchain = 1,280 at K=8, C=1024, D=10. Measured on synthetic stacks at exactly that shape (`tests/test_nested_rhat.py`, and the same construction as `~/pf/wp3/results/runs/WP3__nested_k8.jsonl`):

| fixture at C=1024, D=10 | K=8 (n=1,280) | K=64 (n=160) | K=256 (n=40) |
|---|---|---|---|
| clean, iid normal | 1.0004 | 1.0028 | 1.0112 |
| **every other superchain** offset +1.0 sd | **1.1135** | 1.1040 | 1.1089 |
| **all chains** offset +1.0 sd (common mode) | **1.0004** | 1.0028 | 1.0112 |

At K=8 the estimator detects a 1-sd differential offset as strongly as at K=256, so the `B/n` argument does not hold at this shape and a "K ceiling" is the wrong remedy. What it cannot see is a shift shared by every chain: identical to the clean control.

**Mechanism adopted instead.** In the R1 regime the Pathfinder-start deficit is common-mode — one start pool, one warmup budget, the same bias in all 1,024 chains. Any R̂ family statistic (split, nested, or otherwise) is invariant to that by construction. So: **G-R1 (b² against the CPU NUTS reference) remains the only convergence gate for R1**, and G-R1-diag is retained as a differential-drift / multimodality alarm, not downgraded by a K ceiling and never usable as convergence evidence. The genuine K limit that was measured is the statistic's own noise floor rising with superchain count (clean 1.0004 → 1.0112 from K=8 to K=256), which is a reason to keep K modest, not a reason to distrust K=8.

## 2026-10-19 22:58 · integration seam defect found only by merging branches · WP1 + WP2 together crashed every run

No task could have caught this: each task tests its own tree. Merging `wp/wp1` (which added `warmup_grad_evals_per_chain`, reading `chees_info.num_integration_steps`) with `wp/wp2` (whose vendored warmup returns `WarmupInfo(info=<HMCInfo stack>, trace=...)` instead of the bare HMCInfo) makes **every** call raise `AttributeError: 'WarmupInfo' object has no attribute 'num_integration_steps'` at `sampler.py:1089`. The failure is total, not silent — which is the only good news.

Fixed at the seam on `wp/wp1` (`1cb74bf`): unwrap `chees_info` when it carries `.info`, keep the trace on the instance. Inert without the vendored adapter. The bias harness got the same tolerance (`fa8ca79` lineage) so one harness spans pre- and post-vendoring trees.

**Standing rule for the rest of the run:** accept/kill decisions are re-verified on `pf-r1-int`, the merged tree of `pf-r1` + wp8 + wp0refs + wp1 + wp2 + wp3 (`b15af66`, merges clean, no conflicts). A branch that is green alone is not evidence the combination is green.

**First integrated measurement** (`T1__smoke.jsonl`, 8 Schools, C=1024, D=10, W=50, seed 1, covariance + raw iterate, cold): b²_max **0.00051** against noise floor 0.00118 → **G-R1 PASS with 807 warmup gradients per chain** (+186 sampling), ε 0.247, nominal L 35, 0 divergences per 1k. Split R̂ on the same run reads 1.2217 — the reason R1 is gated on b² and not on R̂.

## 2026-10-19 23:15 · RETRACTION of `699be6d` · the default metric orientation IS wrong, Part B §0.1 is confirmed, and the reason my own measurement misled me

I re-ran the arbiter myself on the merged tree (`~/scratch/orientation_arbiter.py`, CPU, 32 chains, 500 warmup, 1,000 draws, seed 5, four conjugate-Normal scalars with posterior variances 1e2 / 3 / 1 / 1e-2, `value_vars` order `[delta, beta, zeta, alpha]` against kernel order `[alpha, beta, delta, zeta]`). Reproduced WP1's table to the digit:

| cell | max R̂ | min bulk ESS | ESS/gradient | worst mean error | ε | nominal L |
|---|---|---|---|---|---|---|
| permuted + **covariance** (Part B's fix) | **1.0021** ✅ | 30,967 | 12.45 | 0.010 sd | 1.213 | 4 |
| permuted + precision (**shipped default**) | **12.1019** ❌ | **32** | 0.014 | 0.232 sd | 0.0140 | 4 |
| unpermuted + covariance (ordering alone) | 1.0069 | 3,176 (zeta) | 0.140 | 0.020 sd | 0.243 | 45 |

Per-variable ESS in the shipped cell: `{delta: 33, beta: 33,052, zeta: 32, alpha: 35}` — three of four coordinates do not move at all, and the mean is 0.23 posterior sd off. That is not a tuning difference; it fails the project's own G-A bar by three orders of magnitude in ESS.

**Where my `699be6d` reasoning went wrong, stated precisely so nobody repeats it.** The momentum-draw measurement itself was right: the array in BlackJAX's `inverse_mass_matrix` slot is used as `M⁻¹` in `KE = ½ pᵀ A p`, and passing `inv(Σ)` gave momentum variance `Σ`. The false premise was the next sentence — *"so the mass matrix should be the posterior covariance, hence `A = inv(cov)` is correct"*. Leapfrog's frequency is `ω² = eig(M⁻¹ H)` with `H = Σ⁻¹` the target Hessian, so the well-conditioned choice is `M⁻¹ = Σ`, i.e. **the array in the slot must be the covariance**, and the momentum variance is then `Σ⁻¹` (which is what makes positions move at the posterior's own scale `Σ^{1/2}`). Stan's docs call the inverse mass matrix "the covariance" for exactly this reason. A mass matrix equal to the covariance is the *opposite* of optimal here.

**Consequences, in order of importance.**
1. **Part B §0.1 is CONFIRMED.** `pfcheeshmc/sampler.py:406`'s `inverse_mass_matrix = jnp.linalg.inv(cov)` is wrong; the fix is to pass `cov` (with the ridge), and the ordering permutation is a separate, additional P0.
2. **`9f892b2` keeps its finding but reverses its direction.** The three `imm_source` arms never shared an orientation — true. But it was the **`mc` arm that was broken** (`inv(cov)`), while `lbfgs_lowrank`/`fisher_lowrank` expand `D(I+U(Λ−I)Uᵀ)D`, which *is* the covariance, i.e. accidentally correct. Every published `mc`-versus-low-rank ratio therefore compares a correctly-conditioned arm against a badly-conditioned one, in the opposite direction from what the docs claim. The numbers stay void; the story changes.
3. **The real-model sweep now reads as a prediction confirmed, not a surprise:** `wp1:mc` (permuted + precision) was the worst arm on gradients/ESS everywhere, and the covariance arm was best on 4 of 4.
4. **GEV's oddity is no longer a paradox but is still unresolved:** there, wrongly-ordered-covariance (1,310 ESS/s) beats correctly-ordered-covariance (343), and the shipped double-wrong cell scored 403 with R̂ 1.0023. Two wrongs partially cancelling, plus a boundary-constrained `xi_interval__` that may prefer the fatter scale it receives by mistake. WP4's problem, logged not explained.
5. **Default flip:** the case is now strong — the correct arm passes R̂ ≤ 1.01 on all four models and beats the shipped arm on gradients/ESS on all four; the only counter-evidence is GEV's divergence rate rising 0.225 → 0.475 per 1k, which cannot justify keeping a metric that produces R̂ 12.1 on a 4-parameter Gaussian. `tests/test_metric_orientation.py::test_default_is_still_precision_orientation` is the test that flips with the default, and the plan wants it flipped with G3 evidence and owner sign-off.

## 2026-10-19 23:45 · T1 (R1 regime) — the corrected arm passes the bias gate at 50 warmup steps on every model measured so far; the shipped arm cannot pass at 200 on logistic or radon

`pf-r1-int`, C=1,024, D=10, seeds 1 and 2, one process per model×arm, warm-lock GPU, `results/runs/T1__r1_sweep.jsonl` (42 rows). `int:shipped` = permutation fixed, orientation `precision`, adapter `ma`. `int:covraw` = orientation `covariance`, adapter `raw` iterate. Gate G-R1: b²_max < 0.01 for x and x², with the printed noise floor; a b² below the floor is "not resolvable", i.e. converged to within resolution, not a stronger pass.

| model | arm | W | b²_max | floor | verdict | gradients/chain | ε | L |
|---|---|---|---|---|---|---|---|---|
| 8 Schools | shipped | 50 | 0.0139 | 0.0021 | **FAIL** | 641 | 0.138 | 20 |
| 8 Schools | shipped | 200 | 0.0020 | 0.0014 | pass | 3,310 | 0.126 | 46 |
| 8 Schools | **cov+raw** | **50** | 0.0015 | 0.0012 | **PASS** | **993** | 0.324 | 35 |
| GEV | shipped | 50 | 0.0870 | 0.0013 | **FAIL** | 377 | 0.0006 | 9 |
| GEV | shipped | 200 | 0.0015 | 0.0036 | below floor | 1,223 | 0.0007 | 9 |
| GEV | **cov+raw** | **50** | 0.0016 | 0.0012 | **PASS** | **595** | 0.516 | 18 |
| logistic | shipped | 50/100/200 | 0.193 / 0.158 / **0.0498** | ~0.003 | **FAIL at every W** | 267–1,090 | 0.0013–0.0016 | 8–13 |
| logistic | **cov+raw** | **50** | 0.0058 | 0.0026 | **PASS** | **335** | 0.500 | 10 |
| radon NC | shipped | 50/100/200 | **106 / 64 / 12.1** | ~0.005 | **FAIL by 3-4 orders** | 349–1,343 | 0.0004–0.0005 | 10–16 |
| radon NC | cov+raw | — | — | — | PENDING (last group running) | — | — | — |

1. **The target is met on three of four models at W=50**: the plan's goal was "R1 passes G-R1 with warmup ≤ 300 steps". Corrected arm: 8 Schools 993 gradients per chain, GEV 595, logistic 335 — versus the reference figure of ~1,200 at W=100 on 8 Schools with the old code and 53,372 at the shipped default W=2,000.
2. **The shipped orientation is not a tuning preference in R1 either.** On logistic it fails at 200 warmup steps with b² 0.0498; on radon it is 12.1 at 200, four orders of magnitude past the bar. This is the same defect that produced R̂ 12.1 on the four-scalar fixture, now on the owner's real regime.
3. **Split R̂ is confirmed useless here** (1.12 to 8.87 across all R1 cells, including passing cells with b² below the floor), which is the empirical basis for gating R1 on b² instead.
4. `int:covraw` GEV shows 0.098 and 0.195 divergences per 1k at W=50/100 and 0 at W=200 — small, warmup-side, and consistent with the ε jump (0.0006 → 0.52) rather than with a geometry failure.
