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
