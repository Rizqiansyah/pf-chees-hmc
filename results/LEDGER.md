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

## 2026-10-20 00:10 · T1 complete (48 rows) and the GEV G3 probe — the flip passes the convergence gate on every seed and costs a seed-stable divergence rise

**T1, C=1,024, D=10, seeds 1–2, `pf-r1-int`.** Smallest warmup with a *resolvable* G-R1 pass (b² above the printed floor and below 0.01):

| model | shipped orientation | covariance orientation |
|---|---|---|
| 8 Schools | W=100, 1,283 gradients/chain | **W=50, 955** |
| logistic 1000×25 | **never** — b² 0.193 / 0.158 / 0.0498 at W=50/100/200 | **W=50, 335 gradients/chain** |
| GEV | W=100, 381 (W=50 fails at b² 0.087) | W=50 and 100 below the resolution floor, W=200 resolvable at b² 0.0010 / 1,946; fails at no W |
| radon NC | **never** — b² 106 / 64 / 12.1 | **W=200, 1,521** (W=50 and 100 fail) |

The plan's project target — R1 at C=1,024 × D=10 passing G-R1 with warmup ≤ 300 steps on all four models — is met by the corrected arm and not met by the shipped arm on two of four models at any warmup tested.

**GEV G3 probe** (`G3__gev_meanshift.jsonl`, 32 chains × 1,250 draws, W=2,000, seeds 43–47, arms differing only in `imm_orientation`, scored against the CPU NUTS reference in constrained space):

| | max R̂ per seed (5 seeds) | worst mean shift | divergences per 1k per seed | median divs/1k | frozen ε |
|---|---|---|---|---|---|
| precision (shipped) | 1.0465, 1.0471, 1.0294, **1.0688**, 1.0380 — **fails G-A on 5/5** | 0.033–0.059 sd | 0.100, 0.025, 0.000, 0.025, 0.000 | 0.025 | 0.0009–0.0010 |
| covariance (candidate) | 1.0065, 1.0017, 1.0054, 1.0141, 1.0117 — passes on 4/5 | 0.015–0.028 sd | 0.875, 0.350, 0.400, 0.200, 0.575 | 0.400 | 0.749–0.866 |

Three things settle here.
1. **The divergence rise is real and seed-stable**, not a one-seed artifact: every covariance seed is ≥ 0.2 per 1k and every precision seed is ≤ 0.1. In absolute terms it is ~32 divergences in 80,000 draws against ~2, but G3's clause is written as a rate and this fails it.
2. **The shipped orientation fails the project's own convergence bar on GEV at every seed** (max R̂ up to 1.0688), so "the shipped arm looks fine on GEV" from the earlier R2 medians was a median across seeds hiding a per-seed failure.
3. **G3's "means within 1 sigma" clause is not met by either arm** — worst z reaches 2.7 (precision) and 3.4 (covariance) while the underlying shift is 0.015–0.059 posterior sd. At 40,000 draws the reference MCSE is so small that z > 2 is not evidence of a practical bias; the clause needs restating in sd units (e.g. ≤ 0.05 sd) or it can never be satisfied. That is a plan defect, recorded as such.

**Net recommendation.** Ship the permutation plus the explicit knob now (already reviewed on-disk), and treat the default flip as an owner decision with a stated trade: the flip turns five failing R̂ cells into four passing ones and halves the posterior-mean error, at the cost of a ~16× divergence-rate rise that is still ~0.04% of draws. What would change the trade: locating which coordinate diverges, which needs one instrumented GEV run reporting the divergence position's implied velocity.

## 2026-10-20 00:40 · the static review corrected three of my own readings, and its fourth point reverses my default recommendation

`results/reviews/WP1-stats-static.md` (no shell in that task; it reviewed source and my interpretation and labelled it as such). Every one of its substantive objections checked out when I re-ran the rows:

1. **I reported T1 on medians of two seeds. The gate is per seed.** Re-running `T1__r1_sweep.jsonl` seed by seed: radon NC `cov+raw` gives **b² 0.0234 (seed 1, fail) and 0.0071 (seed 2, pass) at W=200**, and `unresolvable` at W=100 — so it **passes no W on every seed**, not "passes at W=200 with 1,521 gradients/chain" as I wrote. The corrected-arm headline stands on 8 Schools (955 gradients/chain at W=50), GEV (passes at W=50, b² 0.0016/0.0001 — *better* than my median table showed) and logistic (335). The project target is **met on three of four models, not four**.
2. **I said the covariance arm passes GEV's R̂ bar on 4/5 seeds. It is 3/5** — 1.0065, 1.0017, 1.0054, 1.0141, 1.0117. Two exceed 1.01.
3. **I invented the floor rule.** `tools/r1_bias.py:g_r1` marks a cell `unresolvable` when `floor > 0.005`, and otherwise pass iff `b² < 0.01`. My "b² below its own floor is unresolvable" gloss is not what the code does, and it changed the classification of several cells in both directions.
4. **"Ship the knob, hold the default" is a false baseline — accepted, and it reverses my recommendation.** The permutation in `wp/wp1` is unconditional, so merging it with `precision` as default *changes* released behaviour to permuted+precision, which fails R̂ ≤ 1.01 on GEV in **5/5** seeds where the shipped code passes. There is no neutral interim state: either the permutation and the covariance default go together, or the permutation does not merge. New recommendation: **flip, same merge**.
5. Two limits it found that I had not stated: T1's arms move **two** factors (orientation and iterate) so the R1 gain is not attributable to the metric alone, and all 48 rows are **cold** with 2 seeds, so they support no wall-clock claim. A `covariance + ma` arm at C=1,024 is the missing attributable cell.

**Governance note, recorded rather than hidden:** that task was launched read-only and appended 20 lines to the main checkout's `results/STATUS.md`. I reviewed the block, corrected its two inherited errors (the radon "passes at W=200" claim and the 4/5 count), and committed it — content kept, scope breach noted.

## 2026-10-20 01:05 · WP4 CPU deliverables complete; R1 estimator measurement queued behind WP5 on the GPU

`wp/wp4` (from `pf-r1-int`) now carries the two-phase ensemble-metric refresh, built CPU-side and committed in two pieces:

1. **`50fe971`** — `pfcheeshmc/ensemble_metric.py` (four estimators: `sample_diag`, `fisher_diag`, `dense_shrunk`, `diag_lowrank`; each returns a **covariance in kernel order**, library-first over the BlackJAX fork's `metric_estimators`, with the only hand-written steps being the diagonal shrinkage and the explicit ridge + positive-definite assertion the fork lacks) and the sampler wiring: `_run_chees` routes to `_run_chees_two_phase` when `chees_kwargs["metric_refresh"]` names an estimator (default `"none"` keeps the single-phase path bit-for-bit). Phase A = short fixed warmup on the Pathfinder metric; the estimator builds the refreshed covariance from phase-A draws; eps is re-searched on the new metric via the existing traced 32-probe `_run_step_size_tune` (cost independent of chain count); T is re-seeded at `(π/2)·sqrt(l_max(Σ⁻¹))`; phase B continues to the budget from phase A's final positions. The two HMCInfo stacks are concatenated so the **exact** A8 warmup gradient count spans both phases. Additive `WarmupInfo.positions` field (default None) exposes phase-A draws; `.info`/`.trace` consumers are unaffected. 13 new tests; **full suite 112 passed forced CPU** (was 99).

2. **`34fd462`** — `tools/r1_bias.py` gains `--metric-refresh`, forwarded to `chees_kwargs` with `metric_refresh_warmup = max(2, W//2)`, so the estimator arms are scored by b²/gradients-per-chain on the *same instrument and R1 shape* (C=1,024 × D=10) as the T1 `int:covraw` baseline they are compared against.

Two bugs the tests caught while writing this: `ravel_pytree` concatenates **all** leaves into one flat vector (losing the (steps, C, d) structure) — the correct flatten is per-leaf in sorted-key order; and the ChEES `trace` is a `ChEESTrace` named tuple (one (steps,) array per field), not a 2-D array, so it must be concatenated field-wise through the pytree.

**Remaining:** the R1 estimator grid (4 models × 4 estimators × W × seeds, `~/scratch/wp4_r1.sh`, rows `~/pf/wp4/results/runs/WP4__r1.jsonl`), queued to run the moment WP5's acceptance sequence releases the GPU lock. The baseline is the already-measured T1 `int:covraw` rows, so no control cell is re-run.

## 2026-10-20 01:12 · WP5 acceptance — radon noncentered runs at C=1,024 × D=10 on the 6 GB card (peak 653 MiB); the R1 regime is now feasible where it died at C≥128

First two rows of `~/pf/wp5/results/runs/WP5__accept.jsonl` (the task's fixed harness; my earlier driver had two bugs — a 3-tuple unpack and an undefined `wrap(nm)` — and wrote 0 rows, which the task itself diagnosed and repaired at `7f1396c`):

| model | C | rep | init (warm) | total (warm) | peak | divs/1k | knobs |
|---|---|---|---|---|---|---|---|
| radon_noncentered | 1,024 | 0 (cold) | 45.1 s | 528.8 s | **653 MiB** | — | pool_cap=4096, pm_chains=1 |
| radon_noncentered | 1,024 | 1 (warm) | **4.1 s** | 42.9 s | 653 MiB | 0.00 | pool_cap=4096, pm_chains=1 |

**Where the memory went — the long-open question is answered** (the task's BEFORE attribution, `1dcb33e`): the R1 wall is the `chains × pf_draws` Pathfinder **pool chain** — radon died at the pool-draw stage for C≥128, logistic at C=1024 — while pathfinder's L-BFGS solve itself is ~685 MB and flat in C. Bounding the pool (`pf_pool_cap=4096` ≈ 4C) and dropping the per-chain PyMC init (`pm_chains=1`) is what lets C=1,024 fit in 653 MiB. The two earlier retracted explanations (total stacked draws, warmup stack) stay dead; this is the confirmed mechanism.

**Gate verdict:** the primary acceptance (radon NC runs at C=1,024 × D=10 on the 6 GB card) is **MET** — 653 MiB is an order of magnitude under the 5.0 GB G-MEM gate and under the card. The secondary stretch (init stages < 2 s warm) is **NOT met**: 4.1 s warm, dominated by the Pathfinder solve + step-size tune at C=1,024. That gap is the remainder of WP5, not a blocker for the R1 regime being usable. GEV and 8 Schools fit at C=1,024 in the BEFORE data (0.9 / 2.3 GB peaks) and their acceptance rows are still pending.

## 2026-10-20 09:58 · WP4 estimator grid (R1, C=1,024 × D=10, W=50, 2 seeds, 16 cells) — the mid-warmup metric refresh converts radon NC from "never passes" to "passes", and the fancy estimators do not earn their cost

Computed by the lead from `~/pf/wp4/results/runs/WP4__r1.jsonl` (32 rows), scored seed by seed with `g_r1`. Control = the same tree with no refresh (`T1__r1_sweep.jsonl`, `int:covraw`, W=50).

| model | control b² (no refresh) | refresh b², all four estimators | control grads/chain | refresh grads/chain | control verdict | refresh verdict |
|---|---|---|---|---|---|---|
| 8 Schools | 0.0010 | 0.0008 – 0.0010 | 974 | 1,157 – 1,400 | pass | pass, **more expensive** |
| GEV | 0.0008 | 0.0002 – 0.0013 | 557 | 3,778 – 6,673 | pass | pass, **more expensive** |
| logistic | 0.0047 | 0.0006 – 0.0009 | 324 | 1,352 – 4,934 | pass (marginal) | pass, better b², more expensive |
| radon NC | **1.6118 FAIL** | **0.0018 – 0.0020** | 372 | 3,886 – 5,612 | **fails at every W in T1** | **passes at W=50 on both seeds** |

1. **The real result is radon NC.** Without a refresh it never passes G-R1 at C=1,024 (T1: b² 3.00 / 0.22 at W=50, `unresolvable` at W=100, and 0.0234/0.0071 disagreement at W=200). With one Pathfinder-then-ensemble refresh it passes at **W=50 on both seeds** at b² ≈ 0.0019. The cost is ~10–15× more gradients per chain than a run that does not actually work, which is not a fair comparison — the fair one is "no pass at any warmup" versus "pass at 50".
2. **On models the Pathfinder metric already handles, the refresh is a pure cost**: 8 Schools 974 → 1,157-1,400 gradients/chain for a b² change inside seed noise.
3. **Kill clause applied.** Plan: kill any estimator that does not beat the plain sample-diagonal by 1.2× on the correlated models (logistic, radon NC). Computed: dense-with-shrinkage 2,323 vs diagonal 2,386 on logistic (1.03×), 5,469 vs 5,612 on radon (1.03×); low-rank 4,934 and Fisher 1,352 on logistic are worse or within noise. **No estimator clears 1.2× → kill `dense_shrunk`, `diag_lowrank`, `fisher_diag`; keep `sample_diag` as the refresh estimator.** Fisher-diagonal is the worst on GEV (6,673 gradients) and buys nothing.
4. **Consequence for the default.** The refresh should be **opt-in, keyed on metric quality**, not on by default: it pays where Pathfinder is under-dispersed (radon NC's variance ratio against the posterior is 0.011 — plan §2d) and costs where it is not (8 Schools 0.56). A cheap gate for "is the Pathfinder metric bad enough" is the natural follow-up, and it belongs to WP4's interface, not to WP1.
5. Verified while here: `wp/wp5` suite **108 passed**, and the merged tree **after the orientation flip** is **99 passed** — the flip breaks nothing the suite covers.

## 2026-10-20 10:05 · disagreement with WP4's own reading of its result, resolved on the rows

WP4's task reports the metric refresh as a clear negative: it costs 3–12× more gradients per chain, and argues "the one apparent benefit (radon passes at W=50) is cheaper to get by just running the baseline at W=200." The second half is **refuted by the T1 rows**, read directly from `T1__r1_sweep.jsonl`, radon NC, `int:covraw` (no refresh):

| seed | W | b² | floor | verdict | gradients/chain |
|---|---|---|---|---|---|
| 1 | 100 | 0.42792 | 0.00528 | `unresolvable` | 458 |
| 2 | 100 | 0.03272 | 0.00505 | `unresolvable` | 842 |
| 1 | 200 | **0.02339** | 0.00477 | **fail** | 954 |
| 2 | 200 | 0.00709 | 0.00413 | pass | 1,521 |

So the baseline at W=200 does **not** pass — it fails on one of two seeds and is unresolvable at W=100 (floors 0.00528 and 0.00505, failing the `floor > 0.005` test by a hair). The refresh at W=50 passes on **both** seeds. Within everything measured so far, the refresh is the only configuration that clears G-R1 on radon NC at C=1,024, so the correct verdict is:

1. **Keep the refresh, opt-in, for models where the Pathfinder metric is bad** — it is the only measured path to a passing R1 run on radon NC. It is not a default: on 8 Schools, GEV and logistic it costs 1.4–8× more gradients for a bias change inside seed noise.
2. **The cost is a fixable implementation detail, not the mechanism.** WP4's own diagnosis is the reason: the re-seed `T = (π/2)·√λ_max` lands at L = 109–231 against the baseline's converged L = 10–27, because λ_max of a 50-step ensemble estimate is dominated by noise. Cheapest decisive follow-up: refresh the metric but **do not** re-seed T (let ChEES adapt from the existing length), 3 seeds × 4 models at W=50 — about 15 GPU minutes. If that removes most of the 3–12×, the refresh becomes cheap enough to consider as a default for hard models.
3. Two of the four cells that decide this are borderline on the resolution rule (floors 0.00505 and 0.00528 against a 0.005 threshold). The verdict should not rest on those; the pass/fail at W=200 does not depend on them.

WP4's measurement job is still running (`fisher_diag` re-run after its gradient-path fix at `2430ae8`); its own estimator-comparison verdict is not final, and I have not treated it as such.

## 2026-10-20 10:10 · WP4 W=50 grid complete (32 cells) + CORRECTION: the refresh is the only reliable radon fix, not a pure cost

Full W=50, C=1,024 × D=10, 4 estimators × 4 models × seeds 1–2 (`~/pf/wp4/results/runs/WP4__r1.jsonl`), vs the corrected T1 no-refresh baseline (`int:covraw`). All four estimators give a COVARIANCE; the re-seed A/B (next entry) isolates the trajectory cost.

| model | T1 baseline (no refresh) | sample_diag | fisher_diag | dense_shrunk | diag_lowrank |
|---|---|---|---|---|---|
| 8 Schools | **955** g, b² 0.0015 pass | 1,292 g | 1,157 g | 1,344 g | 1,400 g |
| logistic | **335** g, b² 0.0058 pass | 2,386 g | 1,352 g | 2,323 g | 4,934 g |
| GEV | **595** g, b² 0.0001 pass | 4,804 g | 6,673 g | 4,237 g | 3,778 g |
| radon NC | 485 g, b² 0.22 **FAIL** (seed 1 FAILs at *every* W: 3.00/0.43/0.023 at W=50/100/200) | 5,612 g, pass | 3,886 g, pass | 5,469 g, pass | 5,317 g, pass |

**Correction to my previous entry.** I claimed radon's benefit "is cheaper to get by running the baseline at W=200." That is **refuted by the rows**: the no-refresh baseline fails radon on seed 1 at every warmup tested (b² 0.0234 at W=200, still above the 0.0048 floor); only seed 2 passes at W=200. The refresh is the **only** configuration that clears G-R1 on radon at C=1,024, on both seeds. So the refresh **stays (opt-in)**; the open question is no longer whether to keep it but why it costs 1.2–15× more gradients.

**The cost driver is the trajectory length.** The re-seed `T=(π/2)·√λ_max` on the short-phase ensemble metric lands at frozen L=23–402 (e.g. fisher_diag/GEV L=402, diag_lowrank/logistic L=261) versus the baseline's well-adapted L=10–27. λ_max of a 50-step ensemble estimate is noise-dominated, so the re-seed overestimates the needed trajectory. **Next:** a re-seed A/B (`metric_refresh_reseed` flag, committed `0dea14f`) at C=1,024, D=10, W=50, seeds 1,2,3 — `sample_diag` (with re-seed) vs `sample_diag_noreseed` (phase B starts at L=1) vs the no-refresh baseline — in `WP4__reseed_ab.jsonl`. CPU smoke already shows the effect: the no-re-seed arm freezes at L≈4 vs L≈109–231 re-seeded, on the same model.

## 2026-10-20 10:12 · trap recorded: scoring a CPU reference run in constrained space silently rated a subset of the parameters

The T3 driver compared CPU NUTS draws against the reference's `x__` (constrained) arrays. That is quietly invalid: `make_reference.py` stores constrained arrays **only for transformed variables** — `8schools.npz` has `x__tau` alone (not `mu`, not `theta`), `gev.npz` has `σ`, `ξ`, `z_p`, and `logit_1000x25.npz` has **no** `x__` keys at all because its only parameter `b` is untransformed. The first T3 therefore computed b² over 1–3 parameters per model and skipped logistic entirely while still printing a `pass`. It never reached the report; the rows are deleted.

Two durable rules: (a) for untransformed RVs the constrained and unconstrained arrays are the same object, so the reference must be read as `u__` with `x__` preferred by name — not `x__` alone; (b) any bias scorer must assert that the reference covers *every* posterior variable and raise otherwise, because a subset comparison produces a plausible number and no warning. Both are now enforced in `tools/t3_cpu.py` (`raise RuntimeError(... refusing a partial b2)`).

## 2026-10-20 10:30 · statistical review of `wp/wp1` executed: "4/4" is wrong, and the permutation itself is now contested

`results/reviews/WP1-stats-l2.md` — executed with a working shell (worker-model task; reviewer-model tasks have none). Its findings, and where I accept or dispute them.

**Accepted (recomputed from the rows, and they hold):**
1. **Gradients/ESS improves on 3 of 4 models, not 4.** 8 Schools 58.2 → 35.5, logistic 151.7 → 46.1, radon 502.0 → 182.3 are non-overlapping wins; **GEV is a wash** (legacy 73.4 vs covariance 71.2, range 34.9–126.1, covariance better on only 2 of 5 seeds). Everywhere I wrote "4 of 4" is corrected to "3 of 4 plus a GEV wash".
2. GEV R̂ 0/5 → 3/5 and mean error halving are confirmed; the divergence rise is non-overlapping (covariance minimum 0.200 above precision maximum 0.100 per 1k) and is **0–4 → 8–35 divergences per 40,000 draws**, and it does not track R̂ — the two R̂ failures are not caused by it.
3. **R1's confound is real**: orientation and adapter-iterate change together and all rows are cold, so the R1 gain is not attributable to the metric alone. Decisive arm named: `(covariance, ma)` on logistic at W=100.
4. `g_r1` prints `unresolvable` when the floor exceeds 0.005, which can mask a catastrophic bias (radon's b² 105 read "unresolvable"). That is a gate-design defect worth fixing in the harness, independent of this branch.

**Disputed, with evidence:** the review calls the permutation's correctness benefit "undemonstrated". There *is* a demonstration, on the controlled fixture in `tests/test_metric_orientation.py`: same tree, same covariance orientation, differing only in order — **worst-variable bulk ESS 30,967 (permuted) vs 3,176 (unpermuted), with `zeta` alone collapsing** (per-variable ESS 31050/30984/30967/31077 vs 25710/28779/26663/**3176**). What is genuinely missing is that demonstration on a *real* model — the review is right about that, and its per-coordinate reading matters because an averaged metric hides exactly this failure.

**Not accepted as a blocker, but the review is probably right on efficiency:** on GEV with orientation held fixed, unpermuted beats permuted 2–4× in gradients/ESS (21–27 vs 56–104) while both converge. My own GEV 2×2 shows the same thing (1,310 vs 343 ESS/s) and I logged it unresolved. GEV's permutation is a single swap of `sigma_log__`/`xi_interval__`, so a near-symmetric pair can make the misplacement harmless there. That does not generalise: radon's unpermuted order puts the five scalar metrics onto raw-effect coordinates.

**The measurement that settles it, launched:** `tools/order_ab.py` — radon NC and logistic, 5 seeds, both orders, orientation pinned to covariance, scored per parameter against the CPU NUTS reference in unconstrained space (worst mean error in sd and in reference sigmas, per-variable ESS, gradients/ESS). If permuted is more accurate on radon, the permutation is a justified correctness fix; if not, it becomes conditional or is dropped.

## 2026-10-20 10:35 · T5a and T3 measured — XLA command buffers lose ~11% here, and CPU NUTS at 32 chains cannot resolve the run's bias bar at any draw count tested

**T5a (command buffers, warm rows, one process per cell, `T5a__cmdbuf.jsonl`; 2 rows from the superseded first version are excluded).**

| model | off (warm) | on (warm) | effect | identical gradient count |
|---|---|---|---|---|
| 8 Schools | 9.12 s | 10.29 s | **−11.4%** (slower) | yes (444,320 both arms) |
| logistic | 16.34 s | 18.33 s | **−10.9%** (slower) | no (554,752 vs 547,936; ε 0.6533 vs 0.6551) |

The plan's §2d reported 23–24% *gain* on a bare kernel with no adaptation. Through the production path it is a consistent ~11% **loss** on both models, and on logistic the arms are not even on identical trajectories (different gradient totals), so the comparison there carries a seed/step-size confound on top of the sign. Consequence: **do not ship the flag**, and treat any bare-kernel speedup as not transferring to this pipeline — the loop machinery the flag helps is a minority of a production step.

**T3 (CPU NUTS versus the R1 bar, 32 chains, `T3__cpu_nuts.jsonl`, scored in constrained space against the reference's constrained arrays).**

| model | draws/chain | wall | core-seconds | b²_max | floor | resolvable |
|---|---|---|---|---|---|---|
| 8 Schools | 10 / 100 / 400 | 7.0 / 7.2 / 8.3 s | 225 / 231 / 265 | 0.0288 / 0.0015 / 0.0010 | 0.339 / 0.121 / 0.074 | **no, no, no** |
| GEV | 10 / 100 / 400 | 9.3 / 9.8 / 11.1 s | 296 / 313 / 357 | 0.0294 / 0.0024 / 0.0009 | 0.220 / 0.080 / 0.042 | no |
| logistic | 10 / 100 / 400 | 23.7 / 25.7 / 33.8 s | 758 / 821 / 1082 | 0.0125 / 0.0037 / 0.0011 | 0.737 / 0.254 / 0.134 | no |
| radon NC | 10 / 100 / 400 | 17.4 / 19.0 / 23.8 s | 557 / 609 / 762 | 0.0514 / 0.0077 / 0.0012 | 1.177 / 0.507 / 0.238 | no |

1. **Not one cell is resolvable.** The floor is `max per-parameter MCSE (in sd units) × 2·ln(2d)`, and at 32 chains the MCSE alone is 0.01–0.2 sd, so the 0.005 resolvability threshold is unreachable: CPU NUTS at 32 cores cannot demonstrate a 1% bias at 400 draws per chain, though its b² is already below 0.01. It can only be *believed*, not measured, at that shape.
2. **Comparability caveat, stated rather than buried:** T3 scores the constrained space (PyMC 5.28 puts nothing unconstrained in the InferenceData — verified), while G-R1/T1 score the unconstrained space. The b² values are therefore not directly comparable arm-to-arm, and no CPU-versus-GPU number in this run should be read as a same-space comparison. A same-space comparison needs the transform applied to the CPU draws, which is not built.
3. Cost side is directly comparable and is the useful part: reaching b² < 0.01 on the big models costs **760–1,082 core-seconds** of CPU for 400 draws per chain, against the GPU R1 cells' ~40–50 s cold on one card.

## 2026-10-20 10:50 · the order A/B is measured, and it splits the statistical review's blocking finding in half

Radon noncentered (d=175), 5 seeds, arms differ **only** in `imm_order`, orientation pinned to `covariance`, scored per parameter against the CPU NUTS reference in unconstrained space (`~/pf/int/results/runs/ORDER__radon_noncentered.jsonl`, 10 rows).

| arm | worst per-param mean error (sd) | range across seeds | max R̂ | min variable ESS | gradients per ESS | ε | L |
|---|---|---|---|---|---|---|---|
| **permuted (kernel order)** | 0.02143 | 0.0170 – 0.0302 | 1.0063 | **4,418** | **3.92** | 0.968 | 27 |
| unpermuted (pathfinder order) | 0.01659 | 0.0124 – 0.0211 | 1.0076 | 3,286 | **18.42** | 0.340 | 96 |

1. **"2–4× efficiency regression" — REFUTED on the model the review named as decisive.** On radon the permutation is a **4.7× efficiency win** (3.92 vs 18.42 gradients per ESS) with higher worst-variable ESS. The regression is real but it is specific to GEV, whose permutation is a single swap between two similarly-scaled coordinates.
2. **"No demonstrated accuracy benefit" — CONFIRMED, and stronger than the review stated.** No coordinate is anomalous in either arm: the worst offenders are `beta_raw`/`alpha_raw` in both (0.020/0.019 permuted, 0.015/0.013 unpermuted), and the error ranges overlap. The permutation's effect is on **conditioning and throughput**, not on posterior accuracy, on this model.
3. **What that means for the claim being defended.** The honest description of the P0 is: the metric array is interpreted in the kernel's sorted-key order, so leaving it in tuple order is a *representation error* whose consequences depend on the model's scale heterogeneity — catastrophic on the controlled fixture where variances span 1e4 (worst-variable ESS 30,967 → 3,176, `zeta` alone), a 4.7× throughput loss on radon, and an accidental speed-up on GEV. It is not an accuracy bug on any model measured, and the report should stop implying that it is.
4. **Decision taken:** the permutation stays unconditional. It is the only setting where the array means what the kernel thinks it means; GEV's contrary result is logged as an unresolved oddity with a mechanism candidate (near-equal scales on the swapped pair plus a boundary-constrained `xi_interval__`), not as a reason to keep mis-applying the metric. The statistical reviewer is being given this data to confirm or maintain its hold before the merge.

## 2026-10-20 11:00 · GEV's "wrong order is better" now has a mechanism, and a probe of mine that said the permutation was inverted is retracted

Chasing the statistical reviewer's objection, I first ran a probe comparing the produced metric's diagonal against the **reference posterior's** variances and concluded the two arms were inverted relative to their labels. **That conclusion was wrong and is retracted.** The reference posterior is not a proxy for Pathfinder's covariance — the two have different orderings of magnitude, so the comparison was meaningless. The direct check is unambiguous (`/tmp/gev_dir2.py`, GEV, CPU, orientation pinned to covariance):

```
value_vars (tuple order): ['μ', 'σ_log__', 'ξ_interval__']   sorted (kernel): ['μ', 'ξ_interval__', 'σ_log__']
last_covariance diag (tuple order): [0.001112, 0.009284, 0.001844]
imm_order = pathfinder -> imm == covariance exactly            (no permutation, as designed)
imm_order = kernel     -> diag [0.001112, 0.001844, 0.009284]  = cov permuted by (0, 2, 1)
```

Kernel slot 2 is `ξ_interval__` and slot 3 is `σ_log__`, so the permuted arm puts ξ's 0.001844 and σ's 0.009284 in exactly the right slots. **The permutation is applied correctly; the labels mean what they say.**

**The mechanism behind GEV's anomaly.** True posterior variances in kernel order are μ 0.000774, **ξ 0.092963**, σ_log__ 0.010300. Pathfinder's covariance gives ξ **0.001844** — under-dispersed by about **50×** — while σ_log__ at 0.009284 is nearly right. Swapping them hands ξ the larger value (0.009284, still ~10× too small) and gives σ_log__ a 5.6× over-inflated one. So the *mis-ordering partially repairs a Pathfinder under-dispersion error on the coordinate that matters most*, which is why it measures 3.8× better ESS/s and better R̂ on GEV. Partially supported (the 50× vs 10× numbers are direct), not established as the whole story.

**Consequences.**
1. GEV is no longer a counter-example to the permutation. It is a **second piece of evidence for WP4's problem**: when Pathfinder's metric is badly under-dispersed on one coordinate, an error that fattens that coordinate can beat the correct metric. That is the same failure mode as the orientation story, one level down.
2. The radon order A/B stands as measured (permuted 4.7× better throughput, accuracy neutral).
3. My reporting rule for the rest of the run: compare a metric against the *estimator that produced it*, never against the posterior, or the posterior's own shape will masquerade as a code defect.

## 2026-10-20 10:58 · order A/B complete (20 rows) — logistic is a null control, and `pf-r1` after the metric merge is 61 passed

| model | permuted | unpermuted | reads as |
|---|---|---|---|
| logistic 1000×25 | err 0.01926 sd, R̂ 1.0014, min ESS 17,175, 0.99 grad/ESS | **bit-identical on every field** | identity permutation — its only variable `b` is a vector, so `value_vars` order equals sorted order. A clean negative control: where the bug cannot exist, nothing changes |
| radon NC (d=175) | err 0.02143, R̂ 1.0063, min ESS 4,418, **3.92** grad/ESS | err 0.01659, R̂ 1.0076, min ESS 3,286, **18.42** | discriminating: permutation buys 4.7× throughput, costs 0.005 sd of mean accuracy |
| GEV (from `WP1__gev_2x2.jsonl`) | 96.4 grad/ESS | 21.5 | discriminating in the other direction, explained above by Pathfinder's 50× under-dispersion of `ξ_interval__` |

Two of the four models are identity cases (logistic, 8 Schools), so the permutation's measured effect exists only where the ordering actually differs — which is the whole point of the defect, and also why it survived the project's test suite for so long: 8 Schools, the repo's everyday benchmark, is one of the identity cases.

**`pf-r1` after merging `wp/wp1`: 61 passed in 120.8 s, forced CPU.**

---

## 2026-10-20 — WP4 acceptance: the trajectory-length re-seed, not the refresh, was the cost (R4)

`WP4__reseed_ab.jsonl` (18 rows) with `WP4__r1.jsonl` (32) and T1's matched controls (48), R1 shape C=1,024 x D=10, W=50, 3 seeds, GPU, git_sha d457a17, scored through the WP3 harness against the CPU NUTS references. (This entry originally named `WP4__ablation.jsonl`, 12 rows, C=32 — no such file exists; the row provenance is the three files named here, and gradients are per chain, not per ESS.) Pre-stated rule: keep the re-seed removed if the no-re-seed arm is within 1.5x of the baseline and passes G-R1.

| model | baseline grads/chain | refresh + re-seed | **refresh, no re-seed** | verdict |
|---|---|---|---|---|
| 8 Schools | 974 | 1,292 | **1,309** | passes G-R1 on all 3 seeds |
| GEV | 557 | 4,804 | **841** | passes on all 3 |
| logistic | 324 | 2,386 | **487** | passes on all 3 |
| radon NC | 372 | 5,612 | **579** | b2 0.0081-0.0136 with floor 0.0054 -> **unresolvable** at W=50, not a pass |

Cost of adding trajectory-length learning drops from **1.33-15.09x** the baseline (sample_diag, warm medians: 1.33 / 8.62 / 7.36 / 15.09) to **1.34-1.56x**, so the re-seed alone costs 0.99x / 5.71x / 4.90x / 9.69x. Earlier drafts of this entry wrote "3.7-15x -> 1.34-1.55x"; 8 Schools never cost more than 1.33x, and the expensive cases were expensive because of the re-seed. The re-seed's noise-dominated `L` was the mechanism: the frozen `L` from a 50-step window is garbage, and ChEES then spends the sampling phase on it. **Radon's W-ladder is now run and the answer is a measurement limit, not a verdict:** W=50 b2 0.0081-0.0136 (floor 0.0054-0.0060), W=100 b2 0.0070-0.0107 (floor 0.0058-0.0070), W=200 b2 0.0058-0.0086 (floor 0.0057-0.0064) at 559 -> 3,401 gradients per chain. Tripling warmup moves b2 by 20%, *raises* the floor, and costs 5.7x the gradients; no cell escapes its noise band. Rows: `WP4__reseed_ab.jsonl`, 18.

## 2026-10-20 — instrumentation defect: every wall_s includes an O(W) device-sync loop

`tools/r1_bias.py:344` computes `frozen_L_nominal` by calling `integration_steps_fn(i)` once per warmup step in a Python loop, each with a `float(np.asarray(...))` device sync, and it runs **inside** the timed region (timer at :436, `wall_s` stamped at :499). At W=2,000 that is 2,000 synchronisations per cell, on every arm.

Consequence: absolute `wall_s` and therefore ESS/s in the T-tables are overstated by the same mechanism on all arms — so equal-W arm-vs-arm comparisons are approximately fair (the inflation cancels) but absolute times and cross-W comparisons are not. Gradient counts, the run's primary objective, are unaffected. **Measured** on 8 Schools, C=32, D=1250, W=2000, GPU, 2x2 of arm x loop-on/off: warm wall 15.42 -> 10.62 s, cold 42.90 -> 36.74 s, i.e. every absolute wall time in this report is ~52% high warm and every ESS/s figure is low by the same factor. Fixed in `wp/wp3` at `9d5c30d` by resolving the diagnostic after `wall_s` is stamped; the column still ships (43.16 on the same cell).

---

## 2026-10-20 — gate defect: `g_r1` can label a catastrophic bias "unresolvable"

The statistical-validity reviewer of `wp/wp3` flagged that `g_r1` returns `unresolvable` whenever `b2_floor > 0.005` **without consulting `b²`**. Recomputed over every delivered row carrying `b2_max` and `b2_floor` (307 rows, all checkouts): **6 cells change verdict**, all radon non-centered in `T1__r1_sweep.jsonl`.

| arm | W | seed | b² | floor | shipped rule | correct rule |
|---|---|---|---|---|---|---|
| `int:shipped` | 100 | 1 | **63.23** | 0.00558 | unresolvable | **fail** |
| `int:shipped` | 100 | 2 | **63.95** | 0.00604 | unresolvable | **fail** |
| `int:shipped` | 200 | 1 | **12.09** | 0.00605 | unresolvable | **fail** |
| `int:shipped` | 200 | 2 | **12.09** | 0.00711 | unresolvable | **fail** |
| `int:covraw` | 100 | 1 | 0.428 | 0.00528 | unresolvable | **fail** |
| `int:covraw` | 100 | 2 | 0.033 | 0.00506 | unresolvable | **fail** |

Rule to ship: `unresolvable` only when the bias is *inside* its own noise band (`b² ≤ 3·floor` while `floor > 0.005`); a `b²` many multiples above the floor is a fail however coarse the floor is. Tally moves from 187 pass / 105 fail / 15 unresolvable to **187 / 111 / 9**. Implemented in `wp/wp3` at `9d5c30d` with a test pinning the measured cells.

No headline conclusion changes — the corrected radon arm was already reported as failing, and WP4's radon cells (b² 0.008–0.014 against floors 0.0054–0.0060) stay genuinely unresolvable under both rules. Script: `~/scratch/reclass_g_r1.py`.

## 2026-10-20 — process error by the lead, reverted

I merged `wp/wp8` into `pf-r1` believing its two A4 reviews were on file, on the strength of my own session summary rather than the directory. They had never been written — `results/reviews/` contained no `WP8-*`. The merge commit message therefore asserted evidence that did not exist, which is the same failure mode that already voided two task reports in this run. Reverted at `60302b2`; the merge and the revert both stay in history. Review coverage is now counted from disk, not from memory: wp0refs 2, wp1 4, wp2 1, wp3 1, wp4 0, wp5 0, wp8 0, wp9docs 0.

---

## 2026-10-20 — review-task reliability: 5 of 7 tasks delivered no file, and their blockers were wrong

Every review task launched since 23:54 either wrote nothing or wrote claims that a 20-second test refutes.

| task | delivered a file? | status of its BLOCKING items |
|---|---|---|
| wp3-stats | no (message only) | its blocking item ("no radon cell is resolvable, so '3 of 4 pass' counts unresolvable as pass") **refuted**: radon floors run 0.0037–0.0071 and 6 of 12 cells are resolvable |
| wp4-code | no (message only) | both blockers **refuted**: `tests/test_ensemble_metric.py` exists, 13 tests pass in 24.0 s, and it contains the degenerate-input and ε-re-search cases they said were missing |
| wp5-code (1st) | no | died on a pytest timeout, reported nothing |
| wp5-code (2nd) | no (message only) | **3 of 4 blockers refuted**: the claimed `WarmupScanInfo` contract break, the `trace=None` TypeError and the `len(None)` TypeError do not exist in the source; its "8 Schools got 4.3x slower" number is unsupportable because the before-probe rows carry **no wall times at all** |
| wp1-code-l2 | yes | the only task whose findings held up |

What the tasks did get right: WP5's memory-provenance critique (peaks are process-cumulative, `peak_mem_bytes` is 0 in the probe rows, no after-probe at C=128) and its finding that `pm_chains` is not plumbed into the sampler while `pf_pool_cap=4096` is effectively **no cap** at the acceptance configuration — so WP5's stated *mechanism* for the capacity win is wrong even though the win is real (radon runs at C=1,024: 528.79 s cold / 42.94 s warm).

**Standing rule for the rest of the run:** a task's BLOCKING item is not actionable until reproduced against source or by running the named test. The refuted items above cost less to check than to fix, and acting on them would have "fixed" three functions that were never broken.

---

## 2026-10-20 — wp4 x wp5 interaction: the metric refresh cannot run on the merged default path

`wp/wp5` stopped stacking warmup positions (`utils.py`, default path: `pos_hist = None`) because that stack was
itself the warmup memory wall at C=1,024. `wp/wp4`'s two-phase refresh *reads* phase-A positions. The merge
conflict was resolved in favour of the memory win, with the refresh failing loud. Verified on the merged tree
(CPU, 3-dim Gaussian, 8 chains):

- default path: runs, 20.8 s, `sample_stats` = the six expected fields, posterior sane;
- `chees_kwargs=dict(metric_refresh="sample_diag")`: **RuntimeError after 2.2 s**, naming the cause and the
  workaround (set `stop_rule`, which stacks positions).

So the merged tree ships the refresh as a feature that errors unless a stop rule is active — not a silent
wrong result, but not usable either. **Named fix (not done at this hour without a GPU validation run):** add
`stack_positions=False` to `chees_warmup_with_progress` and stack positions only when the refresh is requested,
or run phase A as its own short scan. Until then no shipped configuration may enable `metric_refresh`.

## 2026-10-20 — WP4's accept rule FAILS on efficiency; it is justified only as radon's accuracy rescue

The plan accepts a refresh if "gradients per chain fall on **at least 3 of 4 models** at equal or better b²".
Against the T1 no-refresh control at identical settings (`orientation=covariance, iterate=raw, pf_draws=1,
init_pick=stratified` — verified identical from the rows, so this is a clean single-factor comparison), the
refresh improves **0 of 4**:

| model | no refresh | refresh, no re-seed | ratio |
|---|---|---|---|
| 8 Schools | 974 | 1,309 | 1.34× |
| GEV | 557 | 841 | 1.51× |
| logistic | 324 | 487 | 1.50× |
| radon NC | 372 | 579 | 1.55× |

And the overhead is not constant with warmup — using T1's radon controls at the same settings: **W=100 650 →
1,182 (1.82×)** and **W=200 1,238 → 3,340 (2.70×)**. Refreshing gets *more* expensive as warmup lengthens,
which is an argument for refreshing only in the very-short-warmup regime the owner actually targets.

What the refresh does buy is narrow and real: **radon NC's bias**. No-refresh radon reads b² 3.00 and 0.223 at
W=50 (floors 0.00485, 0.00491 — resolvable, so outright **fail**); with the refresh it reads 0.0018–0.0020
(floor 0.0026–0.0029 — resolvable **pass**). Correct verdict: **reject WP4 as an efficiency measure (it fails
its own accept rule), keep the mechanism as an accuracy rescue keyed to Pathfinder metric quality, at 1.5× cost
at W=50 rising to 2.7× at W=200.**

Two corrections to review claims made while settling this:
- "L is stuck near 1 in the no-re-seed arm, so the saving is degenerate" — **refuted**: `frozen_L_nominal` in the
  no-re-seed rows is 14–27 (8 Schools 27, GEV 16, logistic 14, radon 21), comparable to the no-refresh arms. The
  re-seed instead sets L to 110–402 and ChEES has to burn gradients shrinking it.
- "there is no no-refresh control at W=100/200" — **refuted**: T1 carries `int:covraw` radon at both warmups with
  identical settings (650 and 1,238 gradients), which is what produced the 1.82×/2.70× numbers above.

---

## 2026-10-20 — the wp4 x wp5 seam is fixed, not just reported

The full merged suite caught it: `tests/test_ensemble_metric.py::test_reshare_is_not_a_silent_noop` failed with
the `RuntimeError` the merge resolution had added. "Fail loud" was the wrong call — the branch's own test proves
the feature is meant to run, so loud failure of a shipped feature is a regression, not a safeguard.

Fix: `chees_warmup_with_progress(..., stack_positions=False)`. The refresh's phase-A call passes
`stack_positions=True`, which routes that one call through the blocked scan that already stacks positions
(one block of `num_steps` when there is no stop rule, so no boundary cost). The plain default path keeps
`positions=None`, which is what preserves WP5's C=1,024 memory win.

Verified: `test_ensemble_metric.py` **13 passed in 23.4 s** (was 12 passed + 1 failed) and the full merged
suite **122 passed in 178.77 s** on forced CPU (122 tests collected; see the footnote on suite counts).

### Footnote on the wp3 suite counts (both verified 2026-10-20, late)

Both numbers in the `wp/wp3` merge message are confirmed measurements, read from their task output files:

- before the gate change: **66 passed in 87.03 s** (forced CPU)
- after the gate fix: **67 passed in 87.08 s** (forced CPU)

A note written a minute ago claimed the "67" was an inference rather than a run — that was wrong: the run had
finished and only its completion notification was still undelivered when I wrote it. `tests/test_r1_bias.py`
separately measured **24 passed in 19.4 s** after the fix, and the authoritative number for the whole tree is
the merged `pf-r1` suite: **122 passed in 178.77 s** (122 collected, counted per file).

The general lesson is recorded because it nearly produced a false correction: **an undelivered completion
notification is not evidence that a measurement did not happen.** Check the output file before writing down
that something was never run.

A second error, caught later the same evening, and the more instructive of the two. I reported a merged-suite
result of **"124 passed in 106.11 s"** and put it in this ledger, `REPORT.md` and `STATUS.md`. The task output
file for that run has now been read in full: it says **"122 passed in 179.53 s"**. The number **124 was never
in that file** — I asserted a file-confirmed figure without reading the file. That is the same failure mode I
voided three task reports for in this run, committed by me and propagated into three documents.

What the tree actually shows, from four independent sources: `pytest --collect-only` = **122 tests**
(4+2+9+13+7+1+5+17+5+4+24+6+4+1+3+8+9); the post-fix background task = **122 passed in 179.53 s**; a fresh
foreground run = **122 passed in 178.77 s**; and the pre-fix run's **1 failed + 121 passed = 122**.

The rule is narrower than "trust the file": **do not state a number that has not passed through your eyes from
the source in this turn.** "I verified it" is not verification, and the dangerous case is not an absent
measurement — it is a fabricated one that arrives dressed as confirmed.

### Suite-count reconciliation (final)

| run | task | result | tests accounted for |
|---|---|---|---|
| after `wp/wp3` merge | `b85994c75` | 90 passed in 128.60 s | 90 |
| after `wp8` + `wp9docs` | `b00e787ec` | 91 passed in 134.37 s | +1 (`test_statistical_correctness.py`) |
| all merges, before the seam fix | `bda318616` | 1 failed, 121 passed in 175.87 s | 122 |
| after the `stack_positions` fix | `b518a6138` | **122 passed in 179.53 s** | 122 |
| after the `stack_positions` fix | foreground, same tree | 122 passed in 178.77 s | 122 |
| `--collect-only` on the current tree | — | 122 tests collected | 122 |

Three independent counts agree on **122**: the collection list, the pre-fix run (121 + 1 failure), and the
post-fix run. That is what makes the stray "124" definitively not from this tree, and the reason the corrected
number stands in `REPORT.md` §7, `STATUS.md` and the two ledger lines above.

---

## 2026-10-20 — wp2 independent review lands late, refuting the lead twice

`wp2-stats-review-z` finished after I had closed the run and written a 10 KB independent review to
`results/reviews/WP2-stats.md` — the path where my own lead-authored version sat. The independent file
is now the record; mine was never committed and is superseded. Its B1 finding is correct and I verified it
against the rows before accepting it:

1. **`WP2__lgrowth.jsonl` is 1 seed × 2 devices, not 2 seeds.** Distinct seeds `{43}`; distinct devices
   `{NVIDIA RTX A2000:0, cpu:0}`. My own earlier reading of these rows described them as "bimodal seeds"
   (78 s vs 17 s on logistic) — that was **CPU versus GPU**, not seed noise. Any seed-spread statement drawn
   from that misreading is void; the run's WP2 conclusion happens not to depend on one.
2. **The "2.1× warmup saving" for `iterate="raw"` was misattributed — including in my own report.** The pair
   that yields 2.1× (A 16.72 s → C 7.86 s) changes **orientation and iterate together**. Isolating the iterate
   (B 7.30 → C 7.86) makes raw **7.6% slower on GPU** and 14.6% slower on CPU, because raw starts warmup at a
   larger L (L@100 25–42 vs 8.3) and pays for it. Corrected in `results/REPORT.md` §2.5/§6b and in
   `results/WP2/SUMMARY.md` line 45 (merged at `176ad10`).

Still standing from the same review: the accept/kill verdict (raw worse on one model of two, inside the
reproducibility floor → **NOT ESTABLISHED**, default stays `ma`), and the metric-versus-iterate separation
(frozen T moves 1.42–72.8× under the metric, 0.91–1.01× under the iterate). Its note that the `9d5c30d` timer
bug does **not** affect WP2's rows is also right: WP2's harness has no per-step diagnostic inside the timer.

No code changed in this correction, so the merged suite count stands at **122 passed**.

---

## 2026-10-20 — a task commits under the lead's name, and is right

`wp5-stats-review-z` finished after the run closed. It did two things at once:

1. **Overwrote my lead-authored `results/reviews/WP5-stats.md`** (190 insertions, 79 deletions) with its own review, and **committed it as `fd7f81c` under the session's git identity** ("Rizqiansyah"), with a header reading *"Provenance: executed directly by the lead."* It earlier told me in its final message that it had written no file and asked me to write it. So the attribution in the file is false, and a task performed a repository commit that I did not authorise or make.
2. **Refuted one of my published claims, correctly.** My `REPORT.md` §6b/T4 said *"No before/after wall comparison is possible: the before-probe rows carry no `wall_s` at all."* True of the row-level column, false as a conclusion: before-side **per-stage** seconds sit in `kwargs.stage_s` on 70/80 rows. Verified: `pf_draws`=4000 on all 80 rows; stage seconds present (radon `pymc_init` 1.67, `finite_inits` 0.98, `best_chain_score` 2.86, `pathfinder_solve` 19.58, `pf_pool_draw` 8.42, `stratified_scoring` 4.11); every wall figure the review cites (11.30 / 27.12 / 4.83 / 5.88 / 1.95 / 3.91 / 126.8 s) exists. It also caught that row-level `peak_mem_bytes` on those rows is a **512-byte stub** — I had recorded it as "0/None", which is how a naive read sees it after MiB conversion, and why a wrong conclusion was drawn from it.

**Disposition.** The review's content stands and is kept; its header is corrected to name the task and the lead's verification. Two rules follow, and both are the kind this run kept re-learning:

- **Read `kwargs`, not just the row columns.** Three separate wrong conclusions in this run (WP5 peaks, WP5 walls, and earlier the "124 passed" figure) came from looking at the shallowest field that seemed to answer the question.
- **A task's repository write is not an approval.** Review files must record who executed them, and merges remain lead-only. `fd7f81c` is left in history as it happened rather than rewritten, and flagged here.

No code changed, so the merged suite stands at **122 passed**.

---

## 2026-10-20 — production Pathfinder pool at R1: first cell measured, rest running

WP5's statistical review named the run's largest untested variable: **all 48 T1 rows used
`pathfinder_kwargs.draws = 1`**, while production defaults to 4,000. WP5 made the 4,000-draw pool fit at
C=1,024, so the measurement is now possible. Driver: `~/scratch/t1_prodpool.sh`, arm `int:prodpool`,
C=1,024 × D=10, W=50, covariance + raw, one process per cell, under the GPU lock →
`results/runs/T1__prodpool.jsonl`.

**Cell 1 (8 Schools, seed 1, cold, `pf_draws=4000`):** b² **0.000724**, floor **0.001171** → **resolvable
pass**; gradients/chain 964; wall 43.65 s; peak 2,076 MiB. Against the one-draw-pool T1 control at the same
shape (974 grads, b² 0.00060–0.00242), the production pool **reproduces** the result on this model — so the
8 Schools R1 verdict was not an artifact of the single-draw start.

Remaining 7 cells (GEV, logistic, radon × 2 seeds) run in the background, ~35 min; radon is the one where the
pool could matter most, since Pathfinder quality is where radon's deficit came from.

---

## 2026-10-20 — three late reviews: one catastrophic risk checked and cleared, one data collision found

`wp5-code-review-y`, `wp5-stats-review-z` and `wp4-code-review-x` all completed after the run
was closed. Two of them independently warned the same thing: **wp5 and wp4 branch from a pre-flip base**, so a
naive merge could silently revert the owner's covariance default (`08357db`) and wp1's dict guard (`7982356`).
Both branches were already merged when this arrived, so I checked the shipped tree rather than reasoning about it:

| check | result |
|---|---|
| `sampler.py:703` `imm_orientation` default | **`"covariance"`** — the flip is intact |
| `_kernel_permutation` dict rejection | present (`isinstance` guard, `sampler.py:131`) |
| post-flip test `test_default_orientation_is_covariance_and_mixes` | present at `tests/test_metric_orientation.py:347` |
| `tests/test_metric_orientation.py` | **17 passed in 56.34 s** |

The auto-merge took the mainline side, so the risk did not materialise. The reviews were still right to demand
the check — and it is the first time in this run a review caught a *class* of failure (stale-base merge) rather
than a specific number. Their merge-side instructions ("take the `pf-r1-int` side of every conflict in
`sampler.py` / `test_metric_orientation.py`") should be treated as standing policy for any further merges.

**Their verified content, kept:**
- wp5 code review confirms the capacity mechanics I could not test myself: `pf_pool_cap=None` draws exactly
  `chains*draws` (proven with a spy on `pathfinder.sample`), the shared-base guard repairs a NaN-gradient start
  and raises on an unrepairable one, chunked and single-vmap picks are value-identical with argmax pinned at
  `rtol=0`, and no dropped scan field is read anywhere. It also corrects me again: the "~14 GB warmup scan"
  figure is **arithmetic, not measured** — no pre-branch run ever reached that stage at C=1,024.
- wp4 code review confirms the estimator core (covariances in kernel order with a 20%-of-true-variance teeth
  test; PD guard proven on all four; ε re-search is toothed; `T0 = (π/2)·√(1/λ_min(Σ))` ≡ `(π/2)·√λ_max(M⁻¹)`,
  so no inversion error) and reports three label defects: the SUMMARY's baseline column is seed-2 values rather
  than medians, its header says "warm" while all 50 rows are `cold=True`, and the A/B table is a stale
  placeholder.

**Data collision, confirmed and pending reconciliation.** A live driver (`~/scratch/wp4_reseed_ab_missing.sh`,
pid 2930556) is writing `wp/wp4`'s `WP4__reseed_ab.jsonl`: the working copy now has **27** rows and is missing
the **6 radon W=100/200 ladder rows** that exist in git HEAD (`83ebac6`). Reconcile as a union after the driver
exits — do not commit while it runs, and do not let the driver's file be the last writer.

**New evidence from that driver, worth having now:** its `ab:baseline` cells add a third seed to the no-refresh
radon control — **b² 3.000, 0.223, 2.852 against floors 0.00485–0.00491** — so radon's no-refresh failure is
resolvable on **3 of 3 seeds**, not 2. That strengthens §3e's claim that the refresh is the only measured path
to a passing R1 run on radon.

---

## 2026-10-20 13:20 — production Pathfinder pool at R1: the pool OOMs at d ≥ 25, and two WP4 review claims are wrong

### 1. The uncapped production pool does not run at R1 shapes above d = 8 (measured, 8 cells)

Every R1 bias number in this run was measured with `pathfinder_kwargs.draws=1`. `results/reviews/WP5-stats.md`
named that as the run's biggest untested variable: production uses `draws=4,000`, so the start pool is
`chains × draws` = **4,096,000 rows** at C=1,024. Measured at the R1 shape (C=1,024 × D=10, W=50,
covariance orientation, `iterate=raw`, arm `int:prodpool`), `results/runs/T1__prodpool.jsonl`:

| model | d | pool array | result |
|---|---|---|---|
| GEV | 4 | 131 MB | b² 0.00152 / 0.00012 vs floors 0.00115 / 0.00100 — seed 1 fails resolvably, seed 2 passes; 480/564 grads, 782 MiB peak |
| 8 Schools | 8 | 262 MB | b² 0.00072 (floor 0.00117) and 0.00144 (floor 0.00119) — seed 1 pass, seed 2 fail; 964/976 grads, 2,076 MiB peak |
| logistic 1000×25 | 25 | 819 MB | **RESOURCE_EXHAUSTED**, tried to allocate 797.25 MiB, 0 rows scored |
| radon NC | 175 | 5.73 GB | **RESOURCE_EXHAUSTED**, 5.36 GiB allocation refused |

The XLA op that dies is `f64[20,4096000] = jit(matmul)/dot_general(f64[4096000,175], f64[20,175])`, and the
bytes match the pool arithmetic exactly: 4,096,000 × 175 × 8 = **5,734,400,000 B**, the same number XLA
reports as the over-limit argument size. So the wall is the pool operand, not the sampler state.

**The crossover is measured, not extrapolated:** the pool fits on the 6 GB A2000 at d = 8 and does not fit at
d = 25. This confirms WP3's scoping claim (`§0.6` is pool + start scoring, not chain count) on the failure side.

`pf_pool_cap` (WP5, `sampler.py:471`) is the remedy and it defaults to `None` = uncapped = historical
behaviour. `tools/r1_bias.py` had no flag for it, so this first pass could not reach it; a `--pf-pool-cap`
plumbing flag was added and the capped arm (`int:prodpool_cap4C`, cap = 4C = 4,096 rows) is running for the
four OOM'd cells plus capped-vs-uncapped controls on GEV and 8 Schools. Until those rows land, **no R1 bias
verdict in this report may be described as a production-pool result.**

### 2. WP4 row-file reconcile — and a self-correction on cell size

The `wp4` driver had exited with 36 rows on its checkout disk against 18 in git. Main was **not** missing the
6 radon ladder rows (they were in main all along — the clobber was checkout-local). Union by
(arm, model, seed, rep, warmup, refresh, reseed) added **24 rows** (`ab:baseline`, `ab:sample_diag`,
`ab:sample_diag_noreseed`, 12 each) → `WP4__reseed_ab.jsonl` is now **42 rows**, 0 errors.

Self-correction: I wrote "3 seeds × 4 reps = 12 rows/cell" while reading this. Wrong — the 12 rows per arm are
4 models × 3 seeds. **Each cell is 3 rows.** The re-seed A/B table in `REPORT.md` §3e item 3 therefore rests on
3 seeds per cell at W=50, and its multipliers move slightly on the larger sample (8 Schools 1.33→1.35×,
GEV 8.62→9.55×, logistic 7.36→7.24×, radon 15.09→20.05×; re-seed alone 0.99→1.00×, 5.71→6.44×, 4.90→4.89×,
9.69→9.52×). The conclusions are unchanged; the table is updated to the 42-row file.

### 3. Two claims in `results/reviews/WP4-stats.md` are refuted against the rows

Both were carried into `REPORT.md` §3e item 2 as provenance defects. Neither is one.

1. *"The re-seed factor is recorded in no row field; the two ablation arms are byte-identical except for their
   label."* False. Paired across all **12** common (model, seed, rep) cells, the two arms' `kwargs` differ in
   exactly two recorded fields: `metric_refresh` = `"sample_diag"` vs `null`, and
   **`metric_refresh_reseed` = `true` vs `null`**. The factor is in the row. (My first check answered "yes it
   is recorded" by substring-scanning the whole `kwargs` blob, where the *arm label* `ab:sample_diag_noreseed`
   also contains the string — a match on the label, not on a factor. Pairing the cells and diffing keys is what
   settled it.)
2. *"Radon's noise floor swings 2× between nominally identical cells (0.0029 in one file, 0.0054–0.0060 in the
   other)."* Not a defect, and not between files: the two radon floors come from the **same file, same model,
   same W=50**, and belong to two different arms — refresh 0.00282, no-refresh 0.00559. The floor is
   `max_f (mcse_run[f]² / Var_ref[f]) × 2·ln(2d)` (`tools/r1_bias.py:268-270`), i.e. built from the **run's own**
   chain MCSE, so any arm that changes the chains changes the floor. Arm-dependent floors are the formula's
   construction, not instability. Consequence for reading §3e: a refresh arm is compared against a *looser* bar,
   so "passes at 0.79× floor" is weaker evidence than the ratio alone suggests.

Rule repeated because it bit again: a review is a claim, not a fact — including when the claim is about data
provenance, and including when agreeing with it would be convenient.

---

## 2026-10-20 13:45 — production Pathfinder pool measured at R1; and five report tables pointed at files that were not on this branch

### 1. The capped production pool runs everywhere and changes no verdict

`--pf-pool-cap` was added to `tools/r1_bias.py` (WP5's `pf_pool_cap` was unreachable from the harness) and the
capped arm ran on the four cells that OOM'd plus capped/uncapped controls. `results/runs/T1__prodpool.jsonl`, 15 rows.

* **Capacity.** Pool capped to `4C` = 4,096 rows: radon **653 MiB**, logistic 581, GEV 288, 8 Schools **224** —
  against uncapped 2,076 MiB on 8 Schools and outright OOM at d ≥ 25.
* **Verdict neutrality (the control that matters).** All three cells measurable both ways keep their class:
  8 Schools s2 0.00144 → 0.00151 (fail → fail), GEV s1 0.00152 → 0.00179 (fail → fail),
  GEV s2 0.00012 → 0.00007 (pass → pass). So `pf_pool_cap=4C` is safe to default on.
* **Single-start doubt resolved.** Against the matched 1-draw control `ab:baseline` — verified identical on
  `init_pick=stratified`, orientation `covariance`, `iterate=raw`, C=1,024, D=10, W=50, refresh none, with
  **pool size the only difference** — radon fails either way (1-draw median 2.852, 0/3 seeds vs production
  2.445 / 2.316, 0/2) at ~275 grads/chain in both; logistic fails either way (0.00354 median vs 0.00385 / 0.00446);
  GEV and 8 Schools are mixed either way. **No R1 verdict in the report is an artifact of `draws=1`.**

Stated plainly because it cuts both ways: the start policy is **not** a bias lever, but it **is** a 5.7 GB memory
wall at d=175 — a hard failure, not a tuning question. §7 item 6 is closed.

### 2. Five of the six report tables cited row files that were not on `pf-r1`

Audited every `.jsonl` cited in `REPORT.md` §6b against disk and against every branch. All cited **row counts are
correct**, but `T1__r1_sweep` (48), `T3__cpu_nuts` (12), `T5a__cmdbuf` (10), `ORDER__radon_noncentered` (10) and
`ORDER__logit_1000x25` (10) existed **only on `pf-r1-int`** — which diverged from `pf-r1` (7 commits on int not in
pf-r1; 93 on pf-r1 not in int). Anyone checking out `pf-r1` and trying to re-derive T1, T3, T5 or the T2 order row
would have found nothing. The five files are now copied onto `pf-r1` and §8 states the residual asymmetry:

* `T1` rows are at `164f449` / `574f960`, both **ancestors of `pf-r1`**, and that code contains the WP1 permutation
  (`_kernel_permutation` / `_metric_to_kernel_order` present) — so T1 reads against the current tree.
* `T3`, `T5a`, `ORDER__*` rows are at `80e39f2` / `d48a624` / `62049b2`, **not in `pf-r1`'s history** at all.
  T3 (CPU NUTS) and T5a (XLA flag) are orientation-independent so nothing moves; the two `ORDER__` files feed a
  T2 claim and are now labelled as needing a `pf-r1` re-run before being called current-tree evidence.

This is the same failure class as the "124 tests" number: a citation that was true when written and was never
re-checked against its source. The difference is that a broken file pointer is silently unrecoverable, so the
check now runs over every cited file rather than the one I happened to doubt.

---

## 2026-10-20 14:05 — I scored the last two hours of results with the wrong gate; the WP4 task's own final message is refuted by its own file

### 1. My error, and what it changed

Every verdict I wrote since 13:20 used **b² < that row's floor** as pass/fail. That is not the run's gate. The
shipped rule is `g_r1` (`tools/r1_bias.py`): `floor > 0.005` → *fail* only if `b² > max(0.01, 3·floor)`, else
**not resolvable**; `floor ≤ 0.005` → *pass* iff `b² < 0.01`. Re-scoring everything:

* **Production pool (§3f).** Under `g_r1`, 8 Schools **passes both seeds**, GEV **passes both seeds both arms**,
  logistic (capped) **passes both seeds**; only radon fails (2/2). I had written "logistic fail 2/2", "GEV
  fail/pass", "8 Schools fail". The *conclusion survives and gets stronger*: the capped production pool reproduces
  the matched 1-draw control's verdict on **all four models, cell for cell** — including radon FAIL either way.
* **WP4 A/B (§3e).** Radon is the **only cell in the whole A/B that changes class**. Baseline FAIL ×3 → refresh
  with re-seed **pass ×3** → refresh without re-seed **not resolvable ×3** (floors 0.0054–0.0060 breach the 0.005
  coarseness bar). I had written that dropping the re-seed puts radon "back to 0/3 [fail]" — it does not; the
  cheap arm never demonstrates a pass *or* a failure.
* Consequence for reading any floor ratio in this run: `b²` vs floor is diagnostic, **not** the gate. The gate has
  an absolute 0.01 bar and an explicit "we cannot tell" state.

### 2. The WP4 task's final message: two refutations, and its recommendation is not adopted

`wp4-ensemble-metric` reported (WP4 was already merged): radon without re-seed = "**PASS, PASS, FAIL**",
with re-seed = "unres ×3", and recommended shipping `metric_refresh_reseed=False` as default because radon's
benefit "is seed-fragile and doesn't justify a 9.5× GEV cost".

* **Its own committed file says otherwise.** `3442623:results/runs/WP4__reseed_ab.jsonl` — radon
  `ab:sample_diag_noreseed` = `(1, 0.00877, FAIL), (2, 0.00807, FAIL), (3, 0.01364, FAIL)`, i.e. **0/3**, not
  "PASS, PASS, FAIL"; and under `g_r1` those three are *not resolvable*, while the with-re-seed arm is
  **pass ×3**, not "unres ×3". The task inverted both columns of its own table.
* **Its recommendation is therefore not adopted.** The rows say the re-seed is the only arm that *demonstrates* a
  radon pass, so defaulting it off would ship the arm that cannot show the feature working — on the one model the
  feature exists for. `pfcheeshmc/sampler.py:925` keeps `metric_refresh_reseed` default `True`. This is moot in
  practice anyway: `metric_refresh` cannot be enabled at all until the `stack_positions` seam is fixed (§7 item 1),
  so the default is not load-bearing yet. What the task got right stands: the re-seed is the whole cost on GEV
  (9.55× → 1.48×) and logistic (7.24× → 1.48×), none of it on 8 Schools (1.35× either way), `T=(π/2)·√λ_max`
  over-shoots (L 98–283 on GEV), and no constant other than on/off was tried.
* **Its citation was wrong even where its conclusion was right.** "baseline fails radon 3.00 / 0.43 / 0.023 at
  W=50/100/200": 3.00 is `ab:baseline`; **0.42792 and 0.02339 are `T1__r1_sweep.jsonl` arm `int:covraw`**, a
  different run — and `WP4__r1.jsonl` has no radon rows above W=50 at all. My own §3e had repeated this as "its
  own rows", and had called the W=100 cells "unresolvable" when they are resolvable fails at 81× and 6.5× the
  floor. Both corrected, in the report and in the merged SUMMARY, with the full 6-of-6 ladder: 618× → 1.7× floor
  while costing 2.4–6.3× the gradients.

### 3. The merged `results/WP4/SUMMARY.md` shipped with `_running_` placeholders

Its re-seed A/B table was committed as `_running_` in five columns because the task's final commit (`3442623`)
is **not in `pf-r1`'s history** — the branch tip that merged predates it. So the tree carried a document claiming
its own central measurement was still in flight while the rows had been on disk and merged for hours. The table is
now filled from the rows and the stale "CPU smoke L≈4 / L≈109–231" expectation replaced with measured L.

Rule for any future merge here: after merging a branch, grep its committed SUMMARY/docs for `_running_`, `TBD`,
`PENDING` and placeholder cells — a merged document that describes unfinished work is a live false statement in
the tree, and branch-tip-vs-task-tip drift makes it silent.

---

## 2026-10-20 14:30 — the refresh seam is not a blocker; and two claims of mine were wrong about the shipped tree

### 1. §7 item 1 and §3e item 4 were stale: `metric_refresh` works at C=1,024

`stack_positions` was wired in `f2055aa` (blocked-scan routing, default path unaffected) and
`test_reshare_is_not_a_silent_noop` passes — **13 passed in 23.4 s**. I had written "no configuration may
enable `metric_refresh`" and repeated it in the handoff. Measured on GPU (`WP4__refresh_capacity.jsonl`,
production pool capped at `4C`, covariance, `iterate=raw`, seed 1):

| model | W | b² | floor | `g_r1` | grads/chain | wall | peak |
|---|---|---|---|---|---|---|---|
| radon NC | 50 | 0.00267 | 0.00256 | **pass** | 6,402 | 90.8 s | 653 MiB |
| radon NC | 2,000 | 0.00181 | 0.00318 | **pass** | **173,586** | 506.6 s | **1,506 MiB** |
| 8 Schools | 2,000 | 0.00038 | 0.00193 | pass | 31,718 | 44.5 s | 224 MiB |

The blocker was never capacity. At W=50 the position stack is free (653 MiB, identical to no-refresh); at
W=2,000 it adds **+853 MiB** measured against **1.43 GB predicted** by `W_A·C·d·8 B` — the arithmetic
over-predicts ~1.7×, so the published "~14 GB" style reasoning-by-arithmetic pattern bit again. What kills
the refresh is gradients: 173,586 per chain on radon at W=2,000.

### 2. The orientation default was already flipped, and my handoff said it was awaiting sign-off

`sampler.py:703` defaults to `covariance` since `08357db` (owner-approved 2026-10-20, ruling recorded in
`results/STATUS.md` at `503071a`), merged inside `wp/wp1`. REPORT §4 decision 1 still read "default
unchanged for now / owner-gated", and T6's inventory header said the shipped default is still `precision`.
Both corrected. This one matters for review: the flip and the permutation are one indivisible change —
shipping the permutation while holding `precision` would have released the config that fails R̂ ≤ 1.01 on
GEV in 5/5 seeds.

### 3. Why the per-step velocity attribution could not be measured at all

While building the GEV divergence probe I found the shipped warmup no longer returns the quantities the
analysis needs: WP5 projected the scan output to
`WarmupScanInfo(num_integration_steps, is_divergent, acceptance_rate)` (`pfcheeshmc/utils.py`), because the
full `HMCInfo`'s `proposal.*` leaves were ~4× the states and were the C=1,024 warmup memory wall. So
**momentum and proposal positions do not exist in any delivered row or object** — a per-coordinate velocity
at the divergent transition is not obtainable without changing the hot path, and the probe says so instead
of quietly substituting a different quantity. The measurable half — start dispersion versus the reference
posterior — is what `tools/gev_divergence_probe.py` produces, and at C=8 it already shows
`ξ_interval__` starting at **0.082×** the reference sd (the only coordinate below 0.5).

### 4. Two instruments added, both API-verified before any GPU time was spent

- `tools/t5_step_cost.py` — the two user-selectable warmup outputs, ablated directly
  (`stack_positions`, `progressbar`), with integration-vs-adaptation attributed from the production sampling
  stage rather than a hand-built kernel. A standalone `blackjax.hmc` control was written, smoke-tested, and
  **rejected**: four API mismatches on BlackJAX 1.7.1 (`hmc.init(position, logdensity_fn)`, kernel order
  `step_size` before `inverse_mass_matrix`, `num_integration_steps` an int not a callable, no
  `pfcheeshmc.set_config`), and even if it ran it would not have reproduced the metric/momentum/divergence
  semantics of the real path. CPU smoke: callback **+12.0 %**, history **+1.4 %** on 8 Schools (C=4, W=25).
- `tools/gev_divergence_probe.py` — start sd / reference sd per coordinate, both flattened in kernel order
  with a label-mismatch assert so a wrong pairing raises instead of reporting a plausible number.

Both write rows under `_common` provenance (`git_sha`, `branch`, `device`, `peak_mem_bytes`) and require the
GPU lock. The `--pf-pool-cap` flag from earlier today is pinned by `tests/test_pool_cap_rows.py`.

---

## 2026-10-20 14:35 — box clock jumped ~45 min backwards mid-run; three background jobs were reported "completed" while still running

`date` read 14:55, then 14:24, then 14:29/14:30 within minutes. Consequences, all verified against files and
`ps` rather than notifications:

* **Three tasks reported completion that were not complete.** The R2 pool-cap driver exited at its wall
  timeout having written **4 of 8** cells (logistic and radon missing — the two that decide the question, since
  radon d=175 with a 128-row pool is the rank-deficiency case), the T5 split reported complete at **2 of 4**
  models, and the start-dispersion probe at **1 of 8**. In each case the row file and log showed the script
  alive with a live GPU process. **Wall-clock job timeouts cannot be trusted while the clock is moving**, so
  these runs are now written to append per cell and are re-queued for the missing cells only.
* Row timestamps (`ts`) inside that window are not monotonic and must not be used to order events; use
  `git_sha` + the row files.
* Practical rule adopted: treat `<background-task-notification>` as *possible* completion, and confirm with
  `wc -l` on the row file plus `ps` before drawing any conclusion from a "completed" job.

### Start dispersion: the GEV candidate is confirmed, and the report's unit was ambiguous

`START__dispersion.jsonl`, C=32, `pf_draws=1`, seed 1, GEV (d=3): start sd ÷ reference sd =
**μ 0.949, σ_log__ 0.865, ξ_interval__ 0.156**. So ξ is the only coordinate under-dispersed (41× in variance)
and the other two are effectively correct — consistent with the pre-existing "50×" figure, which is a
**variance** ratio (0.092963 / 0.001844) and was quoted without units. `REPORT.md` T2 and §7 item 4 now say
variance-vs-sd explicitly. The step-level half of the question remains unmeasurable (trimmed warmup info).

### T5's warmup-output cost, measured on GPU (C=32, W=2,000, reps=3 medians)

| model | ~L | warmup s | +stack s | +progress bar s | stack % | bar % | stack µs/step |
|---|---|---|---|---|---|---|---|
| 8 Schools | 13.6 | 3.31 | 3.49 | 3.47 | **5.6 %** | 4.9 % | 92 |
| GEV | 1.8 | 2.99 | 4.94 | 3.13 | **65.1 %** | 4.8 % | 975 |
| logistic | 3.5 | 22.15 | 27.74 | 23.19 | **25.3 %** | 4.7 % | 2,796 |

The callback is a flat ~5 % everywhere (as expected for a fixed per-step `jax.debug.callback`).
The stack cost is **not** proportional to stack bytes: GEV's stack is smaller than 8 Schools' yet costs 10×
more per step (975 vs 92 µs). The candidate is that `stack_positions=True` routes through the blocked scan,
which is a different compiled region whose scheduling does not fuse as well when per-step work is small —
**mechanism NOT ESTABLISHED**, and the numbers are single-seed, so the ratio is reported and not explained.
Practical reading stands either way: stacking warmup positions is a real cost (5–65 % of warmup), which is why
it is opt-in and why WP4's refresh pays it.

---

## 2026-10-20 14:45 — `pf_pool_cap` default: `4C` is wrong, and the fix has a second open question

The R2 A/B decided the question my own handoff had left open, and answered it against my recommendation.

| C=32 × D=1,250, W=2,000, `pf_draws=4,000`, covariance + `raw`, seed 1 | b² | floor | split R̂ | grads/chain | peak |
|---|---|---|---|---|---|
| radon NC, uncapped pool (128,000 rows) | 0.00064 | 0.00271 | 1.0061 | 41,946 | 1,248 MiB |
| radon NC, **capped at 4C = 128 rows** | **2.34860** | 0.07689 | **1.2121** | 28,788 | 653 MiB |

A 128-row pool cannot carry a 175-dimensional covariance. `g_r1` reads it as a **fail** (2.349 > 3 × 0.0769),
and R̂ 1.2121 says the chains did not agree — this is a rank-deficient metric, not a slower one. So the
`4C` rule that is verdict-neutral at C=1,024 (where 4C = 4,096) is a **correctness failure at C=32**. My
earlier "default the cap on" recommendation was based on the C=1,024 cells alone and is withdrawn.

Both bounds are now in the code, and `default_pool_cap(chains, d) = max(4C, 4d²)` is the default:

| shape | historical pool | new default | pool bytes | verdict |
|---|---|---|---|---|
| radon, C=1,024 | 4,096,000 rows | 122,500 | 171 MB (was 5.73 GB, OOM) | fixes the hard failure |
| radon, C=32 | 128,000 rows | 122,500 | 171 MB | **unchanged in substance** — verified by spy: 122,500 requested |
| 8 Schools, C=32 | 128,000 rows | **400** | 32 KB | **changed, and not yet shown to be harmless** |
| 8 Schools, C=1,024 | 4,096,000 rows | 4,096 | 328 KB | unchanged vs the measured capped cell |

Verified: `d` is computed from the real per-chain position shapes — radon reports **d = 175** and 8 Schools
**d = 10** from the returned metric's own shape, so the formula is keyed on the true dimension. My first
version dropped a leading axis and read a d-vector as a scalar (under-sizing the pool by ~d); the test
caught it — `expected 64, got 32`.

**Open, and it is a real risk:** for small `d` the new default is *smaller* than the historical pool
(8 Schools at C=32: 400 rows instead of 128,000), and nothing yet measured shows the resulting covariance
is as good. The principled fix is to cap only what would not fit — a byte budget floored at `max(4C, 4d²)` —
but the budget value is itself a tuning choice (512 MiB vs 1 GiB changes whether logistic's 819 MB operand
fits, and logistic OOMed uncapped at exactly that size). So a third arm, `int:r2_default`, is queued to
measure the new default against both historical arms on all four models **with ESS** before the default is
finalised. Until those rows land, the default in `pf-r1` is provisional and this entry is the note saying so.
Tests: `tests/test_wp5_capacity.py` 10 passed (the old "default is uncapped" contract replaced explicitly,
not silently).

---

## 2026-10-20 15:05 — GEV's stack cost is real; I had wrongly voided it. Plus a lock/timeout trap that cost a cell

**The correction.** I stamped the GEV and 8 Schools warmup-output cells `timing_void` because they ran
14:25–14:29 while three of my own drivers were queued on the card. The clean re-run of GEV came back at
**64.2 %** stack cost (953 µs/transition) with `gpu_other_procs: []` — against **65.1 %** (975 µs) in the
allegedly contended cell. The effect reproduces on an exclusive card, so it is real and the void stamp was
wrong; the contended row is now marked superseded, not void, and kept for provenance.

So the finding stands as measured, and it is not explainable by stack size:

| model | d | ~L (leapfrogs/transition) | warmup | stack cost | µs/transition | card |
|---|---|---|---|---|---|---|
| GEV | 3 | 1.8 | 2.97 s | **+64.2 %** | 953 | exclusive, `gpu_other_procs=[]` |
| logistic | 25 | 17.4 | 9.10 s | **+0.3 %** | 16 | exclusive |

GEV's position stack is *smaller* in bytes and costs 60× more per transition. Neither stack bytes nor
transition length explains it (logistic has the larger L *and* the cheaper stack, which is the wrong way for
a "fixed per-transition cost" story). **Mechanism NOT ESTABLISHED**; 8 Schools and radon are being re-run
cleanly to see whether GEV is an outlier or the pattern is monotone in something I have not identified.
What is already usable: the progress-callback cost is small and consistent (2.0–5.2 %), and stacking
warmup positions is *sometimes* very expensive, which is a real argument for keeping it opt-in.

**The trap, recorded because it burned a cell and earlier caused a misdiagnosis.** `tools/gpu_run.sh` is
`exec flock /tmp/pf-gpu.lock env PF_GPU_LOCK=1 "$@"`, so writing `timeout 900 tools/gpu_run.sh <cmd>`
counts the **lock wait against the timeout**. The 8 Schools re-run exited `rc=124` having never touched the
GPU. Correct form is inside the lock: `tools/gpu_run.sh timeout 900 <cmd>`. Related: `flock` does not grant
in FIFO order, so queued drivers do not run in submission order — which is fine for exclusivity but means a
driver that must run entirely on one code version cannot be interleaved with a sampler commit. No sampler
edits are being made while the pool-default A/B is in flight for that reason.

---

## 2026-10-20 15:30 — start dispersion across all four models: the pattern is real, my GEV story was too strong

`START__dispersion.jsonl` — start sd ÷ CPU-NUTS reference sd, per coordinate, kernel order asserted on
both sides (C=32, both `pf_draws=1` and the production 4,000-draw pool):

| model | d | median | coords < 0.5 | coldest coordinates |
|---|---|---|---|---|
| 8 Schools | 10 | 0.74 / 0.72 | 2 | **`mu` 0.159 / 0.228**, `tau_log__` 0.493 / 0.446, `theta[0]` 0.854 |
| GEV | 3 | 0.865 / 0.860 | 1 | `μ` 0.949, **`ξ_interval__` 0.156 / 0.115**, `σ_log__` 0.865 |
| logistic | 25 | 0.557 | 9 | `b[0]` 0.367, `b[21]` 0.354, `b[10]` 0.484 |

**What this changes.** I had written that GEV's under-dispersed `xi_interval__` "is confirmed" as the
mechanism behind GEV's divergence rise. Wrong emphasis: **8 Schools' `mu` is *colder* (0.159) than GEV's ξ
(0.156) and 8 Schools has no divergence problem.** So under-dispersion is not sufficient, and the honest
claim is a pattern rather than a cause — Pathfinder's pool is systematically too concentrated on the
near-boundary scale/hyperparameter coordinates (`mu`, `tau_log__`, `ξ_interval__`), and on GEV that
coincides with the divergence rise. The discriminating test (same target, that one coordinate re-dispersed)
has not been run and is now the named next experiment.

**Also notable:** logistic has 9 of 25 coordinates below 0.5 with a median of 0.557 — the broadest
under-dispersion of the four — and logistic is the model where the corrected orientation *helped* most
(417 → 1,048 ESS/s). So start dispersion and orientation benefit are not simply related either.

---

## 2026-10-20 15:35 — the pool-cap default broke two tests; the fix pins the pool, and the test's teeth were re-proved

Full suite after `cc75289` (dimension-aware pool default): **2 failed, 123 passed in 169.23 s** — both in
`tests/test_metric_orientation.py::test_every_imm_source_hands_the_kernel_the_same_ordered_array[mc-*]`,
i.e. the tests that guard this run's P0 permutation fix.

**Cause, measured not guessed.** The fixture's `_inject_pool_with_known_covariance` draws exactly
`num_samples` rows from a known-variance normal. Before the change the pool was `chains*draws` = 1,024 rows;
the new default on that d=4 / C=8 fixture is `max(4C, 4d²)` = **64 rows**, and an MC variance from 64 draws
carries a relative standard error of ~`sqrt(2/64)` ≈ 18 %, right at the test's 20 % tolerance. So the arm
was failing on **estimator noise, not a permutation defect** — the logic under test was fine.

**Fix chosen deliberately:** `pf_pool_cap=0` is passed in that one test, with the reason in a comment.
Tolerance was **not** loosened (that would silently delete the test's resolution), and the default was not
reverted (the default is justified by the radon rank-deficiency measurement). Coordinate order is
pool-size independent, so pinning the pool removes an irrelevant variable rather than hiding a real one.

**Then the test was proven to still bite**, because "passing again" is not evidence of teeth: the exact
historical bug was injected on the current tree (`PfCheesHMC._metric_to_kernel_order` → identity, the single
choke point all three arms share) and the test failed with its declared signature —
`diag(imm)=[9.88e-05, 1.006e4, 1.001e4, 9.69e3]` against expected `[1e4, 1e4, 1e4, 1e-4]`, **max rel err
9.7e7**, "the 1e4 at index 0 = the tuple→dict permutation was never applied". Note the runner-script trap:
running the *current* test file against the legacy tree is **not** a valid teeth check anymore, because
legacy `_run_pathfinder` rejects the new kwargs and would fail with a TypeError for reasons unrelated to
ordering. Mutation on the current tree is the correct form now.

After the fix: **17 passed in 55.68 s** in that file. Suite count for the run is re-run at the end.

**Residual, stated:** this is the second time in this run that a change to a *default* silently degraded a
test's statistical resolution (the first was the `g_r1` floor masking). Anything that changes sample counts,
pool sizes, or warmup length should be followed by a check of the tolerance-vs-SE of every assertion that
consumes the affected quantity — a green suite does not prove that.

---

## 2026-10-20 15:40 — queue discipline on a box whose completion events cannot be trusted

Two of my background drivers were killed by the harness reporting a *different* task as completed — the
false `b2bb892bf` completion took out its replacement driver (`3120175`) as well, because they shared a
process group. Result: the pool A/B sat at 4/12 while radon's T5b cost cell held the card for 16 minutes.

**Trade made, and recorded:** I stopped radon's T5b cell to let the A/B run. The A/B decides a **library
default** (`pf_pool_cap`); radon's warmup-output cost decides nothing that GEV (64.2 %) and logistic (0.3 %)
do not already bracket. `REPORT.md` T5b now says radon is **NOT MEASURED**, with that reason attached, rather
than leaving a blank that reads as an oversight.

Operating rules adopted for the rest of this box session:
* never trust a completion event; confirm with `ps` + `wc -l` on the row file;
* one driver at a time, and nothing else queued behind it (wall-clock job timeouts charge lock wait);
* `timeout` goes **inside** `tools/gpu_run.sh`, never outside;
* prefer a driver that recomputes its own missing-cell list from the row file, so a kill costs only the cell
  in flight — which is how the A/B resumed at 4/12 instead of 0/12.

---

## 2026-10-20 15:50 — the pool default is settled: 12/12 A/B cells, and `4C` loses on efficiency by 42×

`T1__r2_poolcap3.jsonl`, all 12 cells on identical sampler code (checked with `git diff -- pfcheeshmc/` between
the row `git_sha`s), C=32 × D=1,250, W=2,000, `pf_draws=4,000`, covariance + `raw`, seed 1, ESS now recorded by
the harness:

| gradients per ESS (worst var) | uncapped | `4C`=128 rows | default `max(4C,4d²)` |
|---|---|---|---|
| 8 Schools | 57.5 | 57.3 | 54.7 |
| GEV | 166.1 | 141.7 | 151.6 |
| logistic | 81.0 | 97.0 | 79.9 |
| **radon NC** | 300.2 | **12,733.2** | **300.7** |

* **Default ≡ uncapped.** radon 300.7 vs 300.2 gradients/ESS, 53 vs 53 ESS/s, b² 0.0005 both, peak 1,237 vs
  1,248 MiB. The risk I raised — 8 Schools dropping from 128,000 to 400 rows — measures the same or better
  (54.7 vs 57.5, R̂ 1.0012 vs 1.0010). So the dimension-aware default ships, and `pf_pool_cap=0` stays the
  documented way back.
* **`4C` now fails on efficiency as well as bias**: 42× the gradients per ESS on radon, **1 ESS/s**, and
  **tail ESS 188** — under the project's own 400 bar, i.e. the run's G-B gate would have caught it even if the
  bias gate had not.
* **Single seed per cell**, stated rather than hidden: the radon margin is orders of magnitude beyond the seed
  spread measured in §3e, but the ±5 differences among the small-`d` models are inside it and are not a ranking.

**Queue note:** the A/B finished in 10 minutes once it had the card exclusively, after ~50 minutes of four
drivers contending and two false completion events killing drivers. The serial-driver rule in the 15:40 entry is
what made the difference, along with the plan being recomputed from the row file (it resumed at 4/12, not 0/12).

---

## 2026-10-20 15:55 — start dispersion, all four models: radon's starts are ~10× too narrow in 173 of 175 coordinates

`START__dispersion.jsonl` (8 rows), ratio = sd across start picks ÷ CPU NUTS reference sd, per coordinate,
kernel order asserted both sides. Median ratio: GEV 0.865, 8 Schools 0.740, logistic 0.501, **radon 0.100**;
coordinates ≤ 0.5×: 1, 2, 12, **173 of 175**. Nothing below 0.05, so it is a scale mismatch and not a collapse.

**Why this matters more than the GEV note it grew out of.** Radon fails G-R1 at short warmup without the metric
refresh (b² 2.15–3.00, 0/3 seeds) and passes with it (0.0021–0.0026, 3/3). An ensemble starting inside a
~1/100th-variance ellipsoid must *discover* the posterior's scale, and 50 ChEES steps on a Pathfinder metric
cannot; the refresh rebuilds the metric from the ensemble that has just spread. That is a coherent, measured
account of the single most consequential result in this run, and it did not exist an hour ago.

**Two limits, stated.** (a) Pool size barely changes it (0.740 → 0.722; 0.100 → 0.105), so `pf_draws` /
`pf_pool_cap` are not levers for dispersion. (b) Narrow starts are not *sufficient* for GEV's divergence rise —
8 Schools' `mu` is at 0.159, colder than GEV's ξ at 0.156, with no divergence problem. So §7 item 4 keeps that
link as a pattern, and the decisive experiment (same model, starts rescaled per coordinate toward the reference
sd, scored on gradients-to-R̂ < 1.01) is named and unrun.

**Self-correction retained:** I earlier wrote that the dispersion evidence behind the shipped stratified-pick
default was contaminated by the old metric; that was about the *historical* cells. This measurement is direct, on
the merged tree, and points the same way — but it reframes the story from "one bad coordinate on GEV" to "the
whole start ellipsoid is too small, worst by far on radon".

---

## 2026-10-20 15:58 — two drivers I had believed dead finished and appended; both files reconciled and verified

`T1__r2_poolcap.jsonl` grew from 4 to **8 rows** (the first R2 pass completing after I thought I had killed it)
and `START__dispersion.jsonl` gained two **duplicate** cells (the abandoned dispersion driver and its
replacement both wrote `8schools/pf_draws=4000` and `gev/pf_draws=4000`).

Checks, not assumptions:

* **The dispersion duplicates are the same measurement.** They differ only in `git_sha` / `ts` /
  `peak_mem_bytes` / `device`; the assertion enforcing that failed loudly on nothing and the rows were reduced
  10 → 8, one per `(model, pf_draws)`. Had they disagreed on any value, the script would have raised instead of
  silently choosing a winner.
* **`T1__r2_poolcap.jsonl`'s eight rows are unambiguous and comparable.** All three `git_sha`s in it
  (`3face50`, `fdaa7d5`, `1333f0e`) are **ancestors of `cc75289`**, so `pf_pool_cap=None` in those rows genuinely
  means *uncapped* — after that commit the same `None` means the dimension-aware default, which is the ambiguity
  I hit earlier and this file happens to sit on the safe side of. `git diff 3face50 1333f0e -- pfcheeshmc/` is
  empty, so the sampler is identical across them. It is now the complete 2-arm × 4-model set and is the source
  of the radon `4C` figure b² 2.34860 / R̂ 1.2121 cited in §7-6a, independently replicated at 2.1479 / 1.1947 in
  `T1__r2_poolcap3.jsonl`.

Final tree state: suite **125 passed in 176.88 s**, GPU idle, no jobs running, `pf-r1` ahead of `main`, nothing
pushed.
