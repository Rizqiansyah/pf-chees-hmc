# pf-chees-hmc — run 2 work order (written 2026-10-20 after reviewing run 1 on `pf-r1`)

Audience: the lead. Read Part A fully before starting, then Part B. This order replaces
the run-1 plan for everything it covers. Where it says "as run 1 Ax", follow section Ax of
`docs/review-2026-10/pf-chees-hmc-implementation-plan.md` (on `pf-r1`) unchanged.

Run 1 delivered the two P0 fixes (metric orientation and coordinate order) and the R1 capacity
work, and those stand. The review of its artifacts found three defects run 1 does not know about.
They void its metric-refresh (WP4) verdicts and its CPU-vs-GPU table (T3), and they leave the
default R1 path untested. Run 2 fixes those defects, finishes the skipped plan item WP5.1, and
answers the owner's question with warm timings: does R1 on the GPU beat 32-core CPU NUTS at
equal accuracy?

---

# Part A — How to run

### A0. Scope
- **In scope:** R2-WP0 to R2-WP4 (code), M1 to M5 (measurements), and the report.
- **Out of scope:**
  - run-1 WP6.2/6.3 (custom kernel, profiling) and WP7 (ε target at large C);
  - WP10 (literature arms);
  - the owner's own model;
  - any change on `main`.
- **Models:** `8schools`, `gev`, `logit_1000x25`, `radon_noncentered`. Use these names exactly in
  every `--models` flag.
- **Shapes:**
  - R1 is C=1,024 chains × D=10 draws, the owner's objective.
  - R2 is C=32 × D=1,250 at W=2,000, the secondary objective.

### A1. Owner decisions for this run
1. Same tasks, caps, checkout flow and library-first rule as run 1 (A4, A6).
2. **Pre-approved default changes, on `pf-r2` only.** the lead may change a default only when its
   pre-registered rule in Part B §3 passes: D1 (ChEES `iterate`) and D2 (Pathfinder covariance
   source). Record the rule evaluation in the LEDGER, row by row. The metric refresh (D3) stays
   **opt-in whatever the result**. the lead reports a recommendation for it and does not change its
   default.
3. G3's "means within 1σ" clause stays as written. Also report the mean shift in posterior-sd
   units next to it (Part B M3). The owner will decide whether to restate the clause.
4. Run-1 owner questions (keep `imm_order`, split of the 2 s init bar): leave them unchanged and
   do not act on them.

### A2. Superseded beliefs — void, even if the lead's memory, LEDGER or REPORT on `pf-r1` say otherwise
the long-term memory holds run-1 conclusions. The ones below are **void** for this run:

| run-1 claim | why it is void |
|---|---|
| "The trajectory-length re-seed is the cost of the refresh; reject the refresh on efficiency" | The re-seed formula is wrong by units (Part B §1.1). The 213–283 frozen L values are that bug. |
| "The refresh is radon's only rescue (passes 3/3)" | Those passes were bought with L≈220, about 5,000 gradients per chain. That is brute force, not evidence for the metric. |
| "CPU NUTS cannot resolve b² at these shapes" (T3, `resolvable` 12/12 False) | The floor in `t3_cpu.py` is not squared, and T3 dropped the transformed variables (§1.2). |
| "ChEES `iterate=raw` costs time and saves none" | Measured at W=2,000 (R2) only. It says nothing about R1, where every pass used `raw`. |
| "Radon cannot be gated at the R1 shape at any warmup" | Without the refresh, radon fails *resolvably* (6/6). W=300, the project limit, was never run. |
| "`pf_pool_cap` defaults to None = uncapped" | Stale. `None` now resolves to `max(4C, 4d²)`. |
| "All R1 evidence used `pf_draws=1`" | Stale. REPORT §3f measured the production pool. |
| "`max(4C, 4d²)` is the settled pool default" | Provisional. The covariance it serves has a closed form (§1.4), and the cap needs about 32·d³ bytes. |
| Every `lbfgs_lowrank` / `fisher_lowrank` number in run-1 rows | Measured with the broken orientation (LEDGER `:174`). |
| REPORT §3e "warm rows" | All 42 rows of `WP4__reseed_ab.jsonl` have `cold: true`. |

Still valid, and the baseline for this run:
- the orientation and permutation fix, and the covariance default;
- the R2 gradients/ESS headline (35.5 / 71.2 / 46.1 / 182.3, re-derived);
- R1 passing at W=50 on 8 Schools, GEV and logistic with (covariance, `raw`);
- radon running at C=1,024 with a 653 MiB peak;
- command buffers being 12% slower;
- nested R̂ being blind to common-mode bias;
- the start-dispersion table (REPORT §3j);
- T5b;
- the ESS fix and the instrumentation fix in `r1_bias.py`.

### A3. Branch, environment, first commits
```bash
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN"
git switch -c pf-r2 # from pf-r1 HEAD (cca5407 or later); carries the lead's working files
# hygiene (run-1 A3 said never commit these; 308 files / 19.4 MB were committed on pf-r1)
git rm -r -q --cached ref/*.pdf
git rm -q --cached pf-chees-hmc-implementation-plan.md pf-chees-hmc-literature-plan.md \
                   pf-chees-hmc-review-handoff.md   # duplicates of docs/review-2026-10/*
printf '\nref/*.pdf\n' >> .gitignore
git add .gitignore
cp pf-chees-hmc-run2-plan.md docs/review-2026-10/pf-chees-hmc-run2-plan.md
git add docs/review-2026-10/pf-chees-hmc-run2-plan.md
git commit -m "run2: start pf-r2; untrack session state and paper PDFs; add run-2 plan"
```
- Do **not** rewrite history; that is the owner's call.
- Never commit to `main` and never push.
- **Implementation checkouts:** `git worktree add ~/pf/r2-<wp> -b r2/<wp> pf-r2`.
  - Use the run-1 A3 env recipe unchanged: one shared pixi env from `$MAIN/pixi.toml`, with
    `PYTHONPATH="$WT"`.
  - Leave the run-1 checkouts (`~/pf/{int,legacy,wp*}`) in place; they hold run-1 evidence.
- **Measurement tree:** measurements run **from `$MAIN` on `pf-r2`**, with `PYTHONPATH="$MAIN"`.
  - Freeze code during a measurement step. No merge into `pf-r2` while any row of that step is
    queued or running; `flock` grants are not FIFO.
  - Each step records its code sha.
  - Before comparing two row sets, assert `git diff --name-only <sha1> <sha2> -- pfcheeshmc tools`
    is empty, or name the difference.
- **Suite command** (forced CPU, about 3 min):
  `OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu pixi run pytest -q -p no:cacheprovider`.
  Quote the pass count and time **from the command's own output**.

### A4. Tasks (as run 1 A4, with these amendments)
- **fast-tier:** R2-WP1, R2-WP2, R2-WP3 and all code reviews.
- **steady-tier:** R2-WP0 checks, R2-WP4 (docs and errata), every M-step runner, and enumerations.
- **Formula fidelity (new, mandatory).** Blocks marked **⟦VERBATIM⟧** in Part B go into the task
  brief **character for character**. Do not paraphrase them.
  - The task's SUMMARY restates the formula it implemented, with `file:line`.
  - The review compares plan, brief and code. In run 1, the brief paraphrased the re-seed formula
    into the wrong one, and the review then approved it.
- **Review protocol (amended).**
  - **One** fast-tier review per code WP, using that WP's **review checklist** (Part B), plus
    the lead's re-derivation of any number the WP's SUMMARY quotes.
  - A review finding blocks a merge only if it comes with a command or test that reproduces it.
    Without one, log it as `UNVERIFIED` in the review file and continue.
  - Run 1 got 6 empty-or-false reviews out of 7; the second review is dropped for that reason.
    A final review pass covers the whole run.
- **Plan coverage (new, mandatory):** keep `results/r2/PLAN-COVERAGE.md`. It has one row per
  numbered item in Part B, marked `DONE (sha/rows)`, `SKIPPED (reason)` or
  `DEVIATED (what and why)`. An item that is silently absent counts as a run failure. Run 1
  skipped WP5.1 this way.

### A5. GPU discipline (as run 1 A5, plus)
- **Timeouts inside the lock:** `tools/gpu_run.sh timeout N <cmd>`, never
  `timeout N tools/gpu_run.sh ...`.
- **One process per (model, arm)** for any row whose peak memory or warm wall-clock is quoted.
- **Repeats:**
  - Bias and gradient sweeps may run `--repeat 1`. Their rows are cold, and no wall-clock claim
    may come from them.
  - Every quoted wall-clock comes from a `--repeat 2` rep-1 row.
- **CPU jobs (M5's CPU NUTS) never overlap a timed GPU batch.** Run them between GPU steps, or
  tag the overlapping rows `cpu_contended=true`.

### A6. Autonomy and stopping (as run 1 A7, plus)
- **Contradiction protocol:** if a measurement contradicts this plan, record both numbers and
  continue. Do not reinterpret the plan.
- **BLOCK R2-WP1 and skip M2b** if, after the re-seed fix, the CPU smoke in R2-WP1 still freezes
  L above 3× the no-refresh L on 2 or more models. Continue everything else.
- **Keep `imm_source="mc"` (D2 = no)** if R2-WP2's exact-equality test fails. Continue.
- **Pause and mark STATUS `WAITING-GPU`** after 3 consecutive hours of foreign GPU occupancy.

### A7. Output artifacts (the review contract)
Everything goes under `results/r2/` on `pf-r2`:
- `STATUS.md`: live status, the current step, and what is blocked.
- `LEDGER.md`: decisions with evidence, the D1/D2/D3 rule evaluations, and contradictions.
- `PLAN-COVERAGE.md`: see A4.
- `REPORT.md`: tables R2-T1 to R2-T6 (Part B §4), errata applied, and open questions.
- `runs/<STEP>__<name>.jsonl`: rows in schema `pf-r2/v1`. This is `pf-r1/v1` plus `effective`
  (the resolved sampler config read from `idata.attrs["config"]`), `pooled` (bool) and `step`.
- `<WP>/SUMMARY.md`: one per code WP, including the restated formula (A4).
- `reviews/<WP>-code.md`.
- `RUN1-COVERAGE.md`: every numbered run-1 Part B item marked DONE, SKIPPED or DEVIATED, with
  evidence. This is a steady-tier enumeration.

Large arrays stay outside git, in `~/pf-runs/`. Row files must be committed (run 1 nearly lost
101 rows that existed only on disk).

### A8. Execution order
```
Wave 1 (parallel): R2-WP0 (lead) · R2-WP1 (fast-tier) · R2-WP2 (fast-tier) · R2-WP4 (steady-tier)
Wave 2: R2-WP3 (fast-tier; needs the R2-WP1/R2-WP2 flag names only, which are fixed in Part B)
                    reviews of WP1/WP2 (fast-tier) as each finishes
Merge order:        R2-WP1 -> R2-WP2 -> R2-WP3 -> R2-WP4, suite after each merge (A3 command)
GPU, serial:        M1 -> [D2] -> M2a -> M2b -> (M2c if triggered) -> M3 -> [D1] -> M4
CPU, between GPU:   M5-cpu (any time after R2-WP3 merges, never during a timed GPU batch)
Last:               REPORT, PLAN-COVERAGE, RUN1-COVERAGE, final suite
```
Rough budget: about 8–10 GPU hours and about 1 CPU hour.

### A9. Done
- `pf-r2` holds every merged WP, and `main` is still at `ecd9ba1`. Nothing is pushed.
- `results/r2/REPORT.md` has R2-T1 to R2-T6 filled, or each gap names the missing measurement.
- D1, D2 and D3 are evaluated in the LEDGER.
- PLAN-COVERAGE has no silent gaps.
- The final suite count is quoted from its own output.
- Print the branch, `git log --oneline pf-r1..pf-r2`, and the REPORT path.

---

# Part B — Technical work order

## 1. Defects found in run 1 (evidence)

### 1.1 The re-seed formula is not invariant to the units of the coordinates (P0)
- **Where:** `pfcheeshmc/ensemble_metric.py:141-154`
  `reseed_trajectory_length(cov) = (π/2)·√(1/λ_min(Σ_B))`. It is pinned by
  `tests/test_ensemble_metric.py:76` `test_reseed_matches_closed_form`.
- **Why it is wrong:** under the covariance metric `M⁻¹ = Σ_B`, the leapfrog frequencies are
  `√eig(Σ_B·H)`. They do not change when one coordinate is rescaled.
  - The formula takes the precision of the *narrowest posterior coordinate in model units*.
  - Rescaling one coordinate by 10 changes T0 by 10×.
- **Measured signature** (`results/runs/WP4__reseed_ab.jsonl`): with the re-seed, frozen L is
  213–244 on radon, 98–283 on GEV and 103–114 on logistic. Without it, L is 21–23, 13–35 and
  12–15.
- **Correct formula:** run-1 plan WP4 specified "λ_max of the **whitened** ensemble covariance".
  The run-1 task's docstring says "the brief defines the reseed in terms of the largest eigenvalue
  of the metric's *inverse*", so the paraphrase happened in the brief.
- **ChEES optimum:** for a unit Gaussian, ChEES is `4d·sin²T`, which peaks at T=π/2. The
  optimum therefore scales with the largest standard deviation in the metric's whitened frame.

### 1.2 The T3 CPU-NUTS floor is not squared, and T3 skipped the transformed coordinates (P0 for the owner's question)
- **Where:** `tools/t3_cpu.py` (exists only on `pf-r1-int`, `~/pf/int`).
  - `mc2` holds `MCSE/sd`.
  - `floor = max(mc2)·2 ln(2d)` is then compared with b², which is squared.
- **Signature:** at 32×400, b² ≈ 0.001 sits 30–200× *below* its own "floor" of 0.04–0.24, and the
  floor falls like 1/√N.
- **Squared floors are about 0.001–0.005** at 400 draws, which makes the cells resolvable passes.
- **Coverage:** T3 scored constrained space, so it dropped `tau_log__`, `ξ_interval__`,
  `σ_log__` and radon's three `sigma_*_log__`, which are the hardest coordinates. The references
  hold `u__*` unconstrained arrays and `r1_bias.score_ensemble` already scores them.
- **No GPU side:** no warm R1 wall-clock exists for the corrected configuration. Every
  C=1,024 × D=10 row with `cold: false` is legacy code (WP3 self-test) or a WP5 accept row at
  W=2,000.

### 1.3 The default R1 path is untested (P0)
- `iterate` defaults to `"ma"` (`pfcheeshmc/utils.py:133`, `sampler.py:925,976`).
- T1 has exactly two arms: (precision, `ma`) and (covariance, `raw`). The R1 gain is attributed
  to two factors at once, and REPORT §3b says so.
- The `raw` cost (+7.7% warmup wall on logistic) was measured at W=2,000.
- At W≤100 the iterate is what decides L growth: L at step 100 is 25–42 with `raw` versus 8.3
  with `ma`.
- The harness defaults disagree with the sampler: `tools/r1_bias.py:613,620,621` default to
  `--pf-draws 1 --orientation precision --iterate ma`.

### 1.4 WP5.1 (closed-form Pathfinder covariance) was skipped without comment (P1)
- There is a **single** Pathfinder fit (`sampler.py:564`), so the pool is i.i.d. from one Gaussian.
- `bj.vi.pathfinder.sample` → `blackjax/optimizers/lbfgs.py:bfgs_sample` draws with covariance
  exactly `diag(α) + β γ βᵀ`. This is `lbfgs_inverse_hessian_formula_1(α, β, γ)` (`lbfgs.py:385`).
  The algebra: `phi = μ + √α ⊙ (Q(L−I)Qᵀu + u)` with `LLᵀ = I + RγRᵀ` and `QR = β/√α`.
- So the MC pool covariance (`sampler.py:675`) is a noisy estimate of a known matrix.
  - Its noise is why `4C` rows went rank-deficient (radon at C=32: b² 2.15 / 2.35).
  - That failure is why the d-dependent cap `max(4C, 4d²)` exists.
- That cap is about 32·d³ bytes: 4 GB at d=500, an out-of-memory risk for a large owner model.
- The `lbfgs_lowrank` arm was never measured in the correct orientation.

### 1.5 Documentation errata (for R2-WP4)
| # | where | wrong | correct |
|---|---|---|---|
| E1 | `results/REVIEW-HANDOFF.md` §3 item 2 | "`pf_pool_cap` defaults to `None`, the uncapped production pool…" | `None` resolves to `max(4C, 4d²)` (`sampler.py:495,589`); `0` is uncapped |
| E2 | `results/REPORT.md` §8 caveat 6 | "All R1 evidence used `pf_draws=1`" | superseded by §3f (production pool measured) |
| E3 | REPORT §6b T1 table | GEV `covraw` cell "335"; shipped GEV "fails every W"; radon `covraw` "unresolvable at W=50/100" | 335 is logistic's figure; §3b says shipped GEV passes at W=100; §3e says radon fails resolvably 6/6 |
| E4 | REPORT §3e item 3 | "warm rows" | all 42 rows are `cold: true` |
| E5 | REPORT §7 "Done" | "125 passed in 169.28 s … (122 tests collected…)" | reconcile with the header (125, 176.88 s); remove "122 collected" |
| E6 | REPORT T2 row "Pathfinder as init vs as warmup … \| §3" | no such data in §3 | cite the row file, or delete the row |
| E7 | REVIEW-HANDOFF §1 | three artifacts listed twice | deduplicate |
| E8 | REPORT T3, HANDOFF §1 T3 row | "CPU NUTS cannot resolve b²" | void: §1.2; replaced by run-2 R2-T4 |
| E9 | REPORT §3e items 1–3, §3g, §7.3 | the re-seed cost and the radon rescue | void: §1.1; replaced by run-2 R2-T1/T3 |
| E10 | REPORT §3e end, §7.5, HANDOFF §3.6, §8 caveat 5 | "Radon cannot be gated … at any warmup" | only the no-re-seed refresh arm is unresolvable; the others fail or pass resolvably |
| E11 | REPORT §4.3, T2 iterate row, HANDOFF §1 | "raw … costs time, saves none" | add "at W=2,000 (R2); untested at R1" |
| E12 | LEDGER `:174` and any table quoting `wp1:lbfgs_lowrank` / `fisher_lowrank` | treated as low-rank evidence | measured under the broken orientation; void |

## 2. Code work packages

### R2-WP0 — Setup (lead; steady-tier for checks)
1. A3 commands.
2. Create the `results/r2/` skeleton (A7) and `env.json`.
3. Before any code merges, run the suite on `pf-r2` and record the baseline count.
4. Copy `~/pf/int/tools/t3_cpu.py` to `tools/legacy/t3_cpu.py` with a header
   pointing to §1.2. It is kept for provenance and must not be run.

### R2-WP1 — Fix the trajectory-length re-seed (fast-tier)
**Files:**
- `pfcheeshmc/ensemble_metric.py` (the re-seed function);
- `pfcheeshmc/sampler.py`, `_run_chees_two_phase` only (about `:895-1000`), including its
  docstring and the comments at `:897-904` and `:963-967`;
- `tests/test_ensemble_metric.py`.

**Change ⟦VERBATIM⟧:**
```
T0_B = (pi/2) * sqrt( lambda_max( L^-1 S L^-T ) )
  where  B = L L^T  is the refreshed metric (imm_B, a dense covariance in kernel order, Cholesky),
         S = sample covariance (ddof=1) of the SAME phase-A pool draws the estimator used, shape (N, d),
         lambda_max via jnp.linalg.eigvalsh of the symmetrised d x d matrix, L^-1 applied with
         jax.scipy.linalg.solve_triangular (no explicit inverse).
Then clip:  T0_B = clip(T0_B, eps_B, max_leapfrog_steps * eps_B)   (so 1 <= L0 <= max_leapfrog_steps).
New signature: reseed_trajectory_length(cov_B, draws) -> float.
Record idata.attrs["metric_refresh_lmax_whitened"] = lambda_max and keep metric_refresh_T0.
```

**Tests** (each must fail on the old formula, demonstrated by mutation, and pass on the new):
1. **Unit invariance.** d=6, random SPD Σ, N=20,000 draws, B = `estimate_covariance("sample_diag", …)`.
   Rescale coordinate 0 by 10 and coordinate 3 by 0.01, rebuild B from the rescaled draws, and
   require T0 to be unchanged to 1e-9 relative.
2. **Identity.** B = S exactly (the dense, unshrunk estimator path), so T0 = π/2 to 1e-8.
3. **Correlation.** d=2, sample correlation ρ̂ and B = diag(S), so T0 = (π/2)·√(1+|ρ̂|) to 1e-8.
4. Reject a non-PD B (keep the existing test).
5. Delete `test_reseed_matches_closed_form`. It pins the defect.

**CPU smoke** (task, before handing over; small C so it is CPU-cheap):
- Run `8schools` and `logit_1000x25` at C=64, D=10, W=50 with `metric_refresh="sample_diag"`,
  re-seed on and off.
- Report frozen L for each.
- Accept: re-seed L is within 0.5–2× of no-re-seed L on both.

**Review checklist (code review):**
- The ⟦VERBATIM⟧ block matches the code.
- Run test 1 against the old formula and show that it fails.
- No explicit `inv`.
- The pool S is the same draws the estimator saw.
- The clip is applied.
- The attrs are written.
- The suite passes.

### R2-WP2 — Closed-form Pathfinder covariance and start scaling (fast-tier)
**Files:**
- `pfcheeshmc/sampler.py`: the Pathfinder/metric/pool region (`default_pool_cap`, `:480-780`)
  and `_metric_to_kernel_order` callers only;
- new tests `tests/test_pf_closed_form.py`.

**Change ⟦VERBATIM⟧:**
```
1. New imm_source value "pf_closed": covariance = blackjax.optimizers.lbfgs.lbfgs_inverse_hessian_formula_1(
       pf_state.alpha, pf_state.beta, pf_state.gamma)            # the exact covariance bfgs_sample draws with
   It is in Pathfinder's flat (tuple, self.rvs) order and MUST go through the same _metric_to_kernel_order
   choke point and the same imm_orientation step as "mc". No ridge (PD by construction; assert eigvalsh min > 0).
2. Pool sizing: when imm_source != "mc" the pool only feeds start picks, so pf_pool_cap=None resolves to 4*C.
   When imm_source == "mc", keep default_pool_cap(chains, d) = max(4C, 4d^2). pf_pool_cap=0 and explicit
   integers behave as today.
3. Do NOT change the imm_source default here. D2 (Part B §3) decides it after M1.
4. New kwarg pf_start_scale (float, default 1.0): after the start picks are chosen, replace each start x_i by
   mu_pf + pf_start_scale * (x_i - mu_pf), with mu_pf the Pathfinder mean in the same flat order
   (pf_state.position + formula_1 @ grad, i.e. the mu in bfgs_sample). Applies to chain starts only, never to
   the covariance. 1.0 must be bit-identical to today.
5. Write idata.attrs["config"] (JSON string): resolved imm_source, imm_orientation, iterate, pf_draws,
   pool rows actually drawn, pf_start_scale, metric_refresh and its resolved settings, and the frozen eps/L.
   R2-WP2 owns this attrs write (sampler.py ~:1400-1440); R2-WP3 only reads it. Merge order puts WP2 first.
```

**Tests:**
1. **Exact algebra.**
   - On `8schools`, `gev`, `logit_1000x25` and `radon_noncentered` (CPU, built through
     `tools/r1_bias.build_model` so the stubs are restored), fit Pathfinder once.
   - `formula_1` must equal the dense expansion of
     `blackjax.mcmc.metrics.lbfgs_inverse_hessian_to_low_rank_metric` to 1e-8 relative
     (Frobenius). This also settles what the `lbfgs_lowrank` arm is.
2. **MC consistency.**
   - Draw N=50,000 from `bj.vi.pathfinder.sample` with the same state.
   - The sample covariance's relative Frobenius error versus `formula_1` must be below
     `5·√(d/N)·(1+‖Σ‖_F⁻¹ tr Σ)`, or use the exact Wishart standard error if simpler.
   - Run on 8schools and gev (small d, fast).
3. **G-ORDER.** The run-1 non-sorted-names fixture (`tests/test_metric_orientation.py`) passes
   with `imm_source="pf_closed"`. Show it fails when the choke point is mutated to identity.
4. **Pool rows.** With `pf_closed` and C=1,024, the rows drawn are 4,096. With `mc` they are
   `max(4C, 4d²)`.
5. **`pf_start_scale`.**
   - 1.0 gives bit-identical starts.
   - 2.0 doubles every coordinate's start sd about μ_pf (to 1e-12).

**Review checklist:**
- The covariance is `formula_1`, not a hand re-derivation.
- It passes through the order choke point before orientation.
- The pool-cap branch is conditional on `imm_source`.
- Scale 1.0 is bit-identical (show the test).
- No ridge is added to `pf_closed`.
- The suite passes.

### R2-WP3 — Harness: sampler-default passthrough, pooled seeds, T3 driver (fast-tier)
**Files:**
- `tools/r1_bias.py`;
- new `tools/t3_cpu_nuts.py`;
- `tools/summarise_rows.py`;
- tests `tests/test_r1_bias.py` and new `tests/test_t3_cpu.py`.

**Change:**
1. **Defaults pass through.**
   - `--orientation`, `--iterate`, `--pf-draws`, `--pf-pool-cap`, `--imm-source` (new) and
     `--start-scale` (new) default to `None`, which means "do not pass; the sampler decides".
   - Every row records `effective = json.loads(idata.attrs["config"])`.
   - A row whose `effective` disagrees with an explicitly passed flag raises an error.
2. **`--pool-seeds`.**
   - After all seeds of a (model, arm, W) cell, concatenate the seeds' unconstrained draws along
     the chain axis.
   - Score once with `score_ensemble(..., chains=S·C, ...)` and write one extra row with
     `pooled: true`, `seed: "pooled"`.
   - This is **G-R1-pool, a diagnostic, never a gate**. It separates the method's bias from
     per-run Monte-Carlo noise, which is radon's measurement problem.
3. **`--metric-refresh-warmup`** (default `W//2`, as today at `:458`). Record it.
4. **`tools/t3_cpu_nuts.py`.**
   - CPU NUTS via `bl.REAL_PM_SAMPLE` with `chains=32`, `cores=32`, `target_accept=0.8` (record
     it), `discard_tuned_samples=False` and `idata_kwargs={"include_transformed": True}`.
   - Score the **value-variable names** with `r1_bias.score_ensemble` against
     `r1_bias.load_reference`: the same instrument as the GPU rows, unconstrained. The gate is
     `r1_bias.g_r1`.
   - Gradients per chain are exact: the sum of `n_steps` over `warmup_sample_stats` and
     `sample_stats`, divided by chains.
   - Rows carry `wall_s`, `core_seconds`, `tune`, `draws`, `divs_per_1k`, b², the floor and the
     `g_r1` verdict.
   - Flags: `--models --tune --draws --seeds --out`. One process per (model, tune, draws).
5. **`summarise_rows.py smallest-pass`.**
   - For a row file and arm, give per model the smallest W in the grid with `g_r1 == pass` on
     **every** seed.
   - Also give the median `grads_per_chain_total` at that W, the per-seed b²/floor, and the
     pooled b².
   - Print "none ≤ max W" when nothing passes.
6. **Document `grads_per_chain_total`** in the `r1_bias` docstring: warmup + sampling + ε-search
   probes + refresh-pool gradients, per chain. Pathfinder gradients are reported separately
   (`pf_grads_total`) because they are shared.

**Tests:**
1. **Floor regression.** On i.i.d. N(0,1) draws shaped (C=256, D=40, d=5), the floor from
   `score_ensemble` is within ±30% of `2 ln(10)/(C·D)` (squared units).
   - The T3 bug would give about √ of that.
   - Run the same check through `t3_cpu_nuts`'s scorer.
2. **Passthrough.** With no flags, `effective.imm_orientation == "covariance"` (sampler default)
   on a tiny model.
3. **Pooled seeds.** Pooling 3 identical-seed copies gives the same b² and a floor ÷3 (±5%).

**Review checklist:**
- The T3 driver calls `score_ensemble` rather than its own scorer.
- `include_transformed` is on and the value-var names are scored.
- Gradients include tuning.
- The floor test fails if the square is removed (show it).
- No harness default silently differs from the sampler.

### R2-WP4 — Errata, coverage, hygiene (steady-tier; bounded)
1. Apply E1–E7 inline in `results/REPORT.md` and `results/REVIEW-HANDOFF.md`.
2. Add an **"Errata (run 2)"** block at the top of both files listing E1–E12, each pointing to
   §1.x of this plan. Tag the E8–E12 passages inline `[void — run-2 erratum E#]`.
3. Do not rewrite run-1 numbers; the run-1 row files stay untouched.
4. Write `results/r2/RUN1-COVERAGE.md` (A7). Read the run-1 plan Part B §4 item by item and mark
   each DONE, SKIPPED or DEVIATED with evidence. WP5.1 is a known SKIPPED.
5. Confirm `.gitignore` works: `git status --porcelain ref` prints nothing new.

## 3. Measurements and decisions (lead schedules; steady-tier runs)
All from `$MAIN` on `pf-r2` after R2-WP1 to R2-WP3 merge (A3 measurement tree). Every command
is `tools/gpu_run.sh timeout N pixi run python tools/r1_bias.py ... --timed`. Rows go to
`results/r2/runs/`.

### M1 — Closed form vs Monte-Carlo covariance (decides D2)
**Arms:** `--imm-source mc --arm m1:mc` and `--imm-source pf_closed --arm m1:closed`. Everything
else takes the sampler default; `effective` records it.

| shape | flags | seeds | repeat |
|---|---|---|---|
| R2 | `--chains 32 --draws 1250 --warmups 2000` | 1,2,3 | 1 |
| R1 | `--chains 1024 --draws 10 --warmups 50 --pool-seeds` | 1,2,3 | 1 |
| memory | radon only, R1 shape, seed 1 | 1 | 1, one process per arm |

**D2: flip the `imm_source` default to `pf_closed` iff all of these hold:**
- (a) R2-WP2 tests 1–3 pass.
- (b) At R2, on every model, the median `grads_per_ess_worst` is ≤ 1.10× mc's, and the per-seed
  count of `max_rhat ≤ 1.01` is not lower.
- (c) At R1, the per-seed `g_r1` verdicts are equal or better on every model.
- (d) Radon's R1 peak memory is ≤ mc's.

If it flips, commit the default change with the rule evaluation in the LEDGER and the commit
message. All later M-steps then use it.

### M2a — R1 iterate sweep, one factor at a time (provisional D1)
- **Arms:** `--iterate ma --arm m2:ma` and `--iterate raw --arm m2:raw`. Everything else is the
  default after D2.
- **Flags:** `--chains 1024 --draws 10 --warmups 25,50,100,200,300 --seeds 1,2,3 --repeat 1 --pool-seeds`.
- 4 models × 2 arms × 5 W × 3 seeds = 120 cells. Run one process per (model, arm).
- **Output:** `summarise_rows.py smallest-pass` per arm. This is R2-T1 part 1.

### M2b — R1 metric refresh, with the fixed re-seed
Skip M2b if A6 blocked R2-WP1. `IT` = the M2a winner: the arm with the smaller smallest-passing
W on more models; `raw` on a tie.

| arm | flags | W grid | seeds |
|---|---|---|---|
| `m2b:diag` | `--iterate IT --metric-refresh sample_diag` | 25,50,100,200,300 | 1,2,3 |
| `m2b:diag_norese` | `... --no-metric-refresh-reseed` | 50 | 1,2,3 |
| `m2b:fisher` | `--iterate IT --metric-refresh fisher_diag` | 50,100 | 1,2,3 |

- All arms use `--pool-seeds`.
- Report the frozen L for every arm next to the no-refresh arm's L. A re-seed L above 3× the
  no-refresh L is a defect signature; log it and continue.

**D3 (recommendation only; the default stays off).** Recommend "refresh on for R1" iff either:
- it lowers total gradients per chain *at the smallest passing W* on ≥ 2/4 models and raises it by
  more than 1.2× on none; or
- it is the only arm that passes radon at W ≤ 300.

Otherwise recommend "off".

### M2c — Start scaling (conditional)
- **Trigger:** radon has no passing W ≤ 300 in any M2a or M2b arm.
- **Run:** `--start-scale 2` and `--start-scale 3` with `IT`, no refresh, at W 50,100,200 on
  radon, plus `8schools` at W=50 as a control. Seeds 1,2,3, `--pool-seeds`.
- **Report:** b² and floor per seed, plus the start-dispersion ratio for each scale (start sd ÷
  reference sd per coordinate, the REPORT §3j method). For that, give `tools/gev_divergence_probe.py`
  a `--start-scale` flag (a bounded steady-tier change) and run it with `--models radon_noncentered --chains 1024`.

### M3 — R2 matrix, incl. GEV and G3 (finalises D1)
- **Arms:**
  - `m3:ma` (`--iterate ma`);
  - `m3:raw` (`--iterate raw`);
  - `m3:raw_refresh` (`--iterate raw --metric-refresh sample_diag`, fixed re-seed; refresh
    warmup W//2 = 1,000).
- **Flags:** `--chains 32 --draws 1250 --warmups 2000 --seeds 1,2,3,4,5 --repeat 1`.
  4 models × 3 arms × 5 seeds = 60 cells.
- **Per seed, report:** `max_rhat`, `min_bulk_ess`, `min_tail_ess`, `ebfmi_min`,
  `divs_per_1k` (sampling and warmup), `grads_per_ess_worst` and b².
- **Also report the worst posterior-mean shift in posterior-sd units against the reference**,
  `max_i |E_run[x_i] − E_ref[x_i]| / sd_ref(x_i)`, next to G3's z version (A1.3).
- **G3 on GEV:** state per seed whether each arm meets R̂ ≤ 1.01, has divergences not above the
  shipped arm's run-1 maximum (0.10 per 1k, sampling), and keeps the mean shift within 1σ. Do not change anything based on
  it.

**D1: flip the `iterate` default to `raw` iff all of these hold:**
- (a) M2a: raw's smallest passing W ≤ ma's on 4/4 models ("none" counts as larger than 300), and
  is strictly smaller on ≥ 2.
- (b) M2a: raw's gradients per chain at its smallest passing W ≤ ma's at its own smallest passing
  W on ≥ 3/4 models.
- (c) M3: raw's median `grads_per_ess_worst` is ≤ 1.15× ma's on every model, and its per-seed
  `max_rhat ≤ 1.01` count is not lower.

If it flips, commit it with the evaluation, as for D2.

### M4 — GPU warm timing at the R1 operating point
- **Arms:** for each model, (i) the **final default path** after D1/D2, no extra flags, at its
  smallest passing W from M2a; and (ii) the **best arm** from M2a/M2b/M2c at its smallest passing
  W, if different.
- **Run:** `--repeat 2 --seeds 1,2,3`, one process per (model, arm).
- **Report from rep-1 rows:**
  - median `wall_s` and the `stage_s` breakdown (init, pathfinder, eps0, warmup, sampling,
    postproc);
  - the rep-0 seed-1 wall, labelled **cold**;
  - the peak memory from a separate single-cell process.
- Also run one C=2,048 cell per model on the default path, seed 1, repeat 2. This is an R1
  scaling data point; if it OOMs, record the OOM.

### M5-cpu — CPU NUTS time-to-pass (fixed T3)
- **Run:** `tools/t3_cpu_nuts.py` for 4 models × tune {200, 500, 1000} × draws {50, 100, 250, 400}
  × seeds 1,2,3 = 144 runs.
- CPU only. Never during a timed GPU batch (A5).
- **Output:** per model, the cheapest (tune + draws) configuration with `g_r1 == pass` on all 3
  seeds, with median wall, core-seconds and gradients per chain.

## 4. Report tables (`results/r2/REPORT.md`)
- **R2-T1 — R1 smallest passing W.** Rows are (model, arm) over M2a, M2b and M2c. Columns:
  smallest passing W, median gradients per chain there, per-seed b²/floor, pooled b²/floor, and
  frozen ε and L.
- **R2-T2 — M1, closed form vs MC.** Both shapes, plus radon memory, with the D2 evaluation.
- **R2-T3 — M3, R2 matrix.** Per-seed R̂ pass counts, median gradients/ESS, divergences, tail ESS,
  E-BFMI and mean shift in sd units. Include the GEV G3 table and the D1 evaluation.
- **R2-T4 — time to pass, CPU vs GPU.** One row per model, with these columns:

  | model | CPU config | CPU wall (median) | CPU core-s | GPU config | GPU warm wall | GPU cold wall | GPU peak | warm speed-up | b² CPU / GPU |
  |---|---|---|---|---|---|---|---|---|---|

  This answers the owner's question; it is the headline.
- **R2-T5 — decisions.** D1, D2 and D3, each with its rule clauses evaluated line by line and the
  deciding row ids.
- **R2-T6 — errata applied** (E1–E12), and plan coverage counts (DONE/SKIPPED/DEVIATED).
- **Open questions for the owner:** at most 5, each with the measurement that would settle it.

## 5. What not to do
- Do not invent a new gate. `g_r1` is the gate, and G-R1-pool is a labelled diagnostic.
- Do not report "b² < floor" as PASS or FAIL.
- Do not quote a wall-clock from a `--repeat 1` row, a cold row, or a contended row.
- Do not change the refresh default, `main`, or the environment (except as run 1 A3 allowed, and
  only by the lead).
- Do not run `tools/legacy/t3_cpu.py`.
- Do not re-run run-1 matrices that this plan does not list.
