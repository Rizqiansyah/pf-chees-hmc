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
