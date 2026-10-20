# pf-chees-hmc: review notes and handoff for a review session

**Scope of this note.** This is a read-only review of `Rizqiansyah/pf-chees-hmc` at commit `ecd9ba1`
(the latest commit message starts "bench(ladder): chain scaling at fixed total draws..."), plus a read of BlackJAX **1.7.1** source
(git tag `1.7.1`, the version the repo's docs pin). Nothing was executed: no GPU, and PyPI was
blocked in the reviewing environment. Every finding is tagged:

- **[SOURCE]**: verified by reading code (file:line given). Re-read before acting.
- **[REPO DATA]**: supported by numbers already in the repo's own docs.
- **[HYPOTHESIS]**: plausible mechanism, needs a measurement on the real box.

Owner's goal, for context: maximise GPU throughput for HMC-class sampling in the
many-short-chains regime (e.g. ~1000 chains × ~10 draws), keep divergences as a diagnostic, and
assume the model parameterisation is already good (funnel fixes are out of scope).

---

## 0. Top findings, priority order

| # | Finding | Tag | Priority |
|---|---|---|---|
| 1 | The default ("mc") inverse mass matrix is the **precision**, not the covariance. Preconditioning is inverted. | SOURCE + REPO DATA | **P0** |
| 2 | ChEES trajectory length starts at **one leapfrog step** and can only grow about **exp(0.1·√N)** in N warmup steps. This, not mixing, plausibly sets the ~2,000-step warmup requirement. | SOURCE + REPO DATA | **P0** |
| 3 | Conclusions about many short chains rest on **split-R-hat over 20–39 draws per chain**, which the repo itself shows is inflated even at stationarity. | SOURCE + REPO DATA | P1 |
| 4 | The "cost per step grows with chains" result is confounded by **O(chains) host-side loops** and a **chains × 4000 Pathfinder pool** inside the timed wall. The pool is also the first suspect for the radon OOM. | SOURCE | P1 |
| 5 | The stop rule's +6.7–9.0 s overhead is plausibly **per-block recompilation** of an eager `lax.scan`, not an intrinsic cost. | SOURCE + HYPOTHESIS | P2 |
| 6 | All metric options come from **one Pathfinder fit**, which looks too narrow on radon. The metric should come from the ensemble during warmup. | REPO DATA + HYPOTHESIS | P1 |
| 7 | Several decisions were taken against the inverted baseline and should be **re-run** after the fixes: the E4 metric A/B, the rescaling rejection, the warmup default, MEADS, and the LAPS regime argument. | — | P1 |
| 8 | Hygiene: kwargs are silently ignored, Halton bits are hard-coded, harnesses live outside the repo, and there is no efficiency or orientation test. | SOURCE | P3 |

Correction to something said earlier in the chat: BlackJAX 1.7.1's **public** `chees_adaptation`
*does* offer opt-in ensemble diagonal mass-matrix estimation (`mass_matrix_estimation="diagonal"`)
plus a slow-direction trajectory-length floor (`chees_adaptation.py:93-236, 652-700`). The repo's
README line "ChEES ... does not adapt the mass matrix" is true only of the private `base()` the repo
calls.

---

## 1. Inverted preconditioning (P0)

**Code** [SOURCE]
- `pfcheeshmc/sampler.py:404-406`: `cov = X.T@X/(n-1) + ridge·I`, then
  `inverse_mass_matrix = jnp.linalg.inv(cov)`. So the inverse mass matrix is the **precision** of
  the Pathfinder draws.
- BlackJAX convention: an array metric *is* the inverse mass matrix M⁻¹. See
  `blackjax/mcmc/metrics.py:180-218` (`default_metric`) and `:221-270`, where kinetic energy uses
  `linear_map(inverse_mass_matrix, momentum)`. The ChEES code states it explicitly at
  `adaptation/chees_adaptation.py:404-405` ("mass matrix M = Σ⁻¹, velocity v = M⁻¹p = Σp").
  Window adaptation sets `inverse_mass_matrix = covariance`
  (`adaptation/mass_matrix.py:336-351`). The ChEES criterion whitens by `1/sqrt(imm)`, treating imm
  as a variance (`chees_adaptation.py:450`).
- The other options in the repo are covariance-oriented, i.e. correct. `lbfgs_lowrank` is the
  L-BFGS **inverse Hessian** (`metrics.py:582-604`). `fisher_lowrank` is a BlackJAX inverse mass
  matrix.

**The repo's own numbers corroborate this** [REPO DATA]. In `docs/ladder-results.md`, under
"Eigen-spectra", on radon (d = 175):
- `mc` has geometric mean **148** and condition number 74.
- `lbfgs_lowrank` has geometric mean **0.0069** and condition number 72.

Both come from the same Pathfinder fit, and 148 × 0.0069 ≈ 1.02. So `mc` is the reciprocal of the
covariance-oriented metric. The doc reads this as Fisher's "units are 2,000x below the posterior's
own scale". That 2,000× is really precision versus covariance.

**Consequence.** With M⁻¹ = Σ_pf⁻¹, the preconditioned covariance is about Σ_pf^{1/2} Σ Σ_pf^{1/2},
which is roughly Σ² when Pathfinder is decent. The condition number is squared instead of being
near 1. The target is unchanged, so this is still valid MCMC and the existing tests cannot catch
it. The cost is efficiency: small ε, long trajectories, and a long ChEES warmup (see §2).

**Consistent with**
- Fisher (covariance-oriented) cut realised steps **21 → 8** on 8 Schools and **60 → 9** on
  logistic, and raised ESS/s 1.22× and 1.96× respectively.

**Not consistent with**
- On radon noncentered, `mc` (271k ESS/s) beat `lbfgs_lowrank` (155k) and `fisher` (52k). My
  reading is that the Pathfinder *shape* itself is poor there (§6), so neither orientation of it is
  good. This needs measurement.

**Do**
1. Change `sampler.py:406` to `inverse_mass_matrix = cov`. Re-check that `utils.py:118-122` then
   passes the diagonal of the covariance to ChEES, which is the correct whitening.
2. Add a test. Use a Gaussian target with known Σ (condition ≈ 1e3), feed draws from N(0, Σ) as
   the "Pathfinder" pool, and assert that the returned imm ≈ Σ, i.e. M^{1/2} Σ M^{1/2} ≈ I.
3. Re-run the 4-model ladder (8 Schools, GEV, logistic 1000×25, radon NC) at 32 chains with 5 seeds.
   Report frozen ε, `steps_mean`, gradients per ESS, ESS/s, max R-hat, and divergences per 1k draws.
   Expect fewer steps and higher ESS per gradient on most models. Radon NC is the open question.

---

## 2. ChEES trajectory length is rate-limited by the adapter (P0)

**Code** [SOURCE, BlackJAX 1.7.1 `adaptation/chees_adaptation.py`]
- `init` (`:513-523`): `trajectory_length = step_size`, i.e. the run starts at **one leapfrog
  step**.
- `update_weight = step ** (-decay_rate)` (`:371`). The repo uses `decay_rate=0.5`
  (`pfcheeshmc/utils.py:68`).
- Adam updates `log(trajectory_length)` (`:473-481`). But `trajectory_length` in the state is
  `exp(moving average)` (`:490-505`). So each step the MA moves by `update_weight × Adam_step`, and
  that MA value is what is fed back into Adam next step and consumed by the kernel.
- Adam's step is about `lr` (0.05, `sampler.py:525`) when the gradient sign is consistent. The clip
  at 0.35 (`:23`, `:479`) does not bind at this lr.

**Bound.** The cumulative growth of log T after N steps is at most about lr·Σ t^{-1/2} ≈ 0.1·√N.

| warmup steps N | max T growth factor |
|---|---|
| 500 | about 9× |
| 1,000 | about 22× |
| 2,000 | about 80× |

Reaching L leapfrog steps at fixed ε therefore needs roughly
N ≈ (10·ln L)²: about 1,000 steps for L = 21, 1,350 for L = 37, and 1,700 for L = 60. Realised steps
can exceed this because ε also falls during warmup: the ε0 doubling search overshoots, and dual
averaging then shrinks it.

**Matches the repo's data** [REPO DATA]
- Logistic: 15.4 steps at warmup 500 versus 59.9 at 2,000.
- Radon NC (`probe_init_policy` table): 7.5, 17.2, 37.4 and 37.5 steps at warmup 250, 500, 1,000
  and 2,000.
- GEV at warmup 1,000 froze at 6.2 steps (README).
- The repo's own conclusion: "Required warmup tracks realised trajectory length".
- More chains not reducing the required warmup is also consistent. Chains reduce gradient noise,
  not this rate limit. This matches the owner's experience that 1000 chains × 10 draws "doesn't
  work".

**To check** [HYPOTHESIS]
- Whether Hoffman et al. 2021, Algorithm 1, feeds the moving average back into the optimiser. The
  repo's `docs/optimisation-plan.md` M4 says the paper uses moving averages only to freeze ε and T
  at the end. Check against `ref/hoffman2021-chees-hmc.pdf` and the TFP
  `GradientBasedTrajectoryLengthAdaptation` source. If the paper does not feed the MA back, the
  BlackJAX behaviour is a deviation that slows warmup.

**Do** (in the repo's forked loop, `utils.py:105-113`)
1. **Instrument first.** Add `astate.step_size`, `astate.trajectory_length`, and the MA values to
   the warmup scan's `ys`. Plot them against the step. This confirms or refutes the rate limit
   directly.
2. **Initialise T0 sensibly.** BlackJAX's own comment (`chees_adaptation.py:101-104`) says ChEES
   converges to about π/2 in whitened units for the bulk. After the §1 fix, set
   `adaptation_state = adaptation_state._replace(trajectory_length=T0)` with T0 = π/2. A more
   robust option is T0 = (π/2)·sqrt(λ_max) of the whitened covariance of the starting ensemble.
3. **Decouple or speed up the optimiser.** Either keep a raw log-T iterate for Adam and use the MA
   only for freezing (this needs forking `base.compute_parameters`, about 150 lines), or raise the
   lr to 0.1–0.2. With 512 or more chains the gradient is low-noise.
4. **Consider the length floor.** `max(T, (π/2)·sqrt(λ_max))` from BlackJAX 1.7.1's
   `_apply_length_floor` (`:192-236`) is exactly a T floor, and it is cheap to port.
5. **Measure** how many warmup steps each variant needs to reach within 10% of the 2,000-step frozen
   (ε, L), with the bias gate from §3. Target a few hundred steps.

---

## 3. Short-chain diagnostics (P1)

**Claims at issue** (`docs/ladder-results.md`)
- "Chains at fixed total draws": 1,024 chains × 39 draws gave R-hat 1.0404, and the doc concludes
  "short chains do not substitute for long ones".
- "Warmup length versus chain count": 512 × 20 gives R-hat 1.08–1.11 at every warmup, and the doc
  concludes "more warmup does not repair what more chains broke".

**Why these are not established**
- The repo's own `utils.py:219-221` shows the estimator problem: with 50 draws per chain, split-R-hat
  sits at 1.035–1.058 *forever* on a unit Normal [SOURCE].
- The nested R̂ paper (Margossian et al., arXiv:2110.13017) makes the same point: R̂ conflates short
  chains with poor mixing.
- Per-chain ESS at 20–39 draws is also unreliable.

**Do**
1. Score the many-short-chains cells by **bias against a reference**. Use the squared error of the
   posterior mean and variance per parameter in posterior-sd units, with reference posteriors from
   long runs or PosteriorDB fits. Sweep chains C ∈ {128, 512, 1024}, draws per chain ∈ {10, 50},
   and warmup ∈ {200, 500, 1000, 2000}.
2. Add **nested R̂** with superchains. `arviz_stats.rhat_nested` is not installed. Install it, or
   implement it with a test against the reference implementation; the repo already learned once
   that hand-rolled R-hat goes wrong.
3. The algebra wall(C) = fixed(C) + (W + D/C)·c(C) in the doc is right, and agreed: when D/C is
   small the warmup W dominates. That is why §2 is the main lever.

---

## 4. Confounders in chain scaling, and memory (P1)

The doc's c/step = wall / (W + D/C) includes all fixed costs. Several of them scale with C on the
host [SOURCE]:

| Location | What it does | Why it matters |
|---|---|---|
| `sampler.py:346-351` | `_score_chain`: one eager `float(self.ld_fn(x))` per chain, only to pick the Pathfinder seed | With `adapt_diag` all starts are identical (the repo's own `optimisation-plan.md` W3 cites PyMC docs). This is pure waste at C = 1024. |
| `sampler.py:260-264` | per-chain `bool(usable[c])` host syncs and slicing | Runs even when every chain passes. |
| `sampler.py:212-215` | per-chain stack of PyMC init points | Scales with C on the host. |

**Pathfinder pool.** `draws` defaults to 4000 per chain (`sampler.py:310`). The pool is
chains × 4000 × d, held in several copies (`samples_flat`, `flat_mat`, `X`; `:362-406`).
- On radon (d = 175) at C = 128, that is 512,000 × 175 × 8 B ≈ **684 MiB per copy**. This is the
  same order as the reported "allocation failure at 699 MiB in a dot_general".
- It is independent of warmup, which matches "radon OOMs at 128 chains even with warmup 100". The
  doc marks the memory source NOT ESTABLISHED; this is the first suspect.
- The pool only estimates the covariance of a Gaussian that Pathfinder already gives in closed form
  (the L-BFGS factors).

**Scan histories.**
- The warmup scan stores `(position, full HMCInfo)` for every step (`utils.py:192`).
- The sampling scan stores the full state plus full `HMCInfo` (momentum, proposal position,
  momentum and gradient) per draw (`sampler.py:704`).
- Item W4 in `docs/gpu-upgrade-plan.md` is still open. At C = 1024 these histories reach GBs.

**Batched trip count** [HYPOTHESIS]. `num_integration_steps` comes from the per-chain
`random_generator_arg` (`blackjax/mcmc/dynamic_hmc.py:110`). Under `vmap`, the leapfrog
`fori_loop` (`mcmc/trajectory.py:226`) therefore has a *batched* trip count, so it runs as a while
loop with a batched predicate and per-iteration selects. That happens even though every chain
shares L (padding ratio 1.0). Computing L once, unbatched, may remove that overhead. Separately,
check whether XLA:GPU syncs the while predicate to the host each iteration on this JAX version.
- Context: "~80% of every step is not the model" (about 2 ms per step against about 0.5 ms of
  model time) is consistent with launch- and sync-bound loops.

**Do**
1. Run the stage timers (`~/pf-o1-stages.py`) at C ∈ {32, 128, 512, 1024}, and log ε and
   `steps_mean` per C.
2. Profile one warmup step with `jax.profiler` or nsys. Count kernels per leapfrog step and
   device-to-host copies per while iteration.
3. Fix the init stage: drop `_score_chain`, or vmap it. Vectorise the finiteness repair.
4. Cap the pool, e.g. `max(chains, 4096)` draws, or use the closed-form covariance.
5. Project the scan outputs to what `sample_stats` needs (W4).

---

## 5. Stop-rule overhead (P2)

[SOURCE] `utils.py:196-204`: `_scan_block` calls `jax.lax.scan` **eagerly**, outside `jit`, once
per block, with a newly traced body. `block_max_rhat` (`utils.py:227`) is also eager.

[HYPOTHESIS] Each block recompiles. With about 8 blocks of 250 steps, roughly a second per compile
would give the measured +6.7–9.0 s.

**Do**
1. Wrap the block runner in `jax.jit` with `n` static and `start` traced, using
   `start + jnp.arange(n)` so at most two shapes compile. Also jit the R-hat.
2. Re-measure. If the overhead drops below about 0.5 s, the E5 verdict ("costs more than it saves")
   flips.
3. For the many-chain regime, gate on ensemble-statistic stabilisation or nested R̂ (§3). An example
   of the former is LAPS's relative fluctuation of E[x_i²] over a moving window. Split-R-hat over a
   window *during* adaptation is the wrong signal here.

---

## 6. Metric source for hierarchical models (P1)

- Every metric option comes from **one** Pathfinder fit: a single path from the best start
  (`sampler.py:355-359`).
- On radon NC, the covariance-oriented L-BFGS metric has a geometric-mean variance of **0.0069**.
  In a noncentred parameterisation most of the 175 parameters are raw effects with near-unit
  posterior scale. [HYPOTHESIS] Pathfinder is badly under-dispersed there, which would explain why
  no Pathfinder-derived metric is good on that model.
- `ref/README.md` already records Pathfinder's own warning about its inverse Hessian, and that
  nutpie and Stan do not take the metric from Pathfinder.

**Do**
1. Compare Pathfinder per-parameter variances with reference posterior variances, and look at the
   ratio distribution.
2. Estimate the metric **from the ensemble during warmup**. With 512 or more chains, a diagonal from
   the last part of a short burn-in is accurate. Options:
   - BlackJAX 1.7.1 `chees_adaptation(mass_matrix_estimation="diagonal")`, or port its Welford
     accumulator and length floor into the forked loop.
   - SNAPER (§9).
   The repo's L6 "metric refresh during warmup" task is the natural home; do it right after §1.

---

## 7. Decisions to re-run after §1 and §2

- **E4 metric A/B**, and "rescaling to the empirical covariance's geometric mean made every model
  worse". The reference scale was the precision's. Overall metric scale should mostly be absorbed by
  ε, so a large effect from rescaling suggests an interaction with the ε0 search, the T0 = ε start,
  or the 1000-step cap. Re-check this.
- **Warmup default of 2,000 and "no smaller default"** (W7). These are likely an artefact of §1 and
  §2.
- **MEADS failure** (ε = 0.0 or NaN on radon).
  - The harness is not in the repo. Check which metric and starts it received.
  - Compare at equal *gradients*, not equal draws: GHMC takes one leapfrog step per transition.
  - The `num_chains % num_folds` and `n_per_fold ≥ 2` guards noted in the doc stand.
- **D7, LAPS rejected as "wrong regime"** (`docs/gpu-upgrade-plan.md`). The owner's target regime
  (512 or more chains, about 10 draws each) is exactly LAPS's regime, so that argument does not
  apply to the goal.
  - The implementation concerns stand and were spot-checked: LAPS is not exported from
    `blackjax/__init__.py` or `adaptation/__init__.py`, and `adaptation/laps.py:291` calls `.item()`.
  - MAMS (LAPS's adjusted phase) has a well-defined energy error, so divergence-style flags are
    definable, though `sample_stats` would need mapping.
- **CPU-vs-GPU conclusions** are specific to an RTX A2000 6 GB (fp64 measured at 196 GFLOP/s).
  State that wherever the result is quoted.

---

## 8. Hygiene (P3)

- **Unknown kwargs are silently ignored** (`.get` everywhere). Concrete cases:
  - `tools/probe_init_policy.py:125` passes `eps_probes=`, but the tuner reads `eps0_probes`
    (`sampler.py:480`).
  - `tests/test_rng_stage_keys.py:70` passes `pathfinder_kwargs={"num_paths": 2}`, which
    `_run_pathfinder` never reads.

  Validate the stage kwargs and raise on unknown keys.
- **Halton bit count** uses `max_sampling_steps=1000` hard-coded (`utils.py:70, 80-83`).
  `halton_sequence` truncates to `max_bits` (`dynamic_hmc.py:205-215`), so sampling past
  2^bits − warmup steps wraps the jitter. This is harmless statistically. Pass the real draw count.
- **Mutable default dicts** in `PfCheesHMC.sample` (`sampler.py:891-906`) are mutated. This is fine
  through the public wrapper but fragile.
- **Dead `xmap` branch** at `sampler.py:754-763`. `jax.experimental.maps` is gone from recent JAX.
- **Harnesses and results live outside the repo**: `~/pf-o1-stages.py`, `~/pf-cache-ab.py`, the
  MEADS harness, `~/pf-research/*`, `~/pf-runs/*.jsonl`. Commit them under `tools/` and `results/`
  so the docs' numbers are reproducible.
- **Tests guard specific past bugs well**, but nothing tests efficiency or orientation (§1). Also
  add one statistical test: on a known anisotropic Gaussian, check the mean and variance within
  Monte Carlo error.

---

## 9. What I agree with

- The evidence discipline, the retractions, and the measurement-integrity notes (the dead-code
  elimination trap and the `CUDA_VISIBLE_DEVICES` trap) are excellent.
- Rejecting FSM de-synchronisation for ChEES is correct: L is shared, so R(m) = 1 and the padding
  ratio is 1.0. FSM stays relevant for a GPU NUTS reference arm.
- Warmup is a fixed cost and is the lever. The equal-total-draws algebra is right.
- CPU wins small one-shot jobs on this box. Four chains is unusable for ChEES. Float32 parked on
  measurement. The funnel is a parameterisation issue, per the owner.

---

## 10. Suggested sequence

1. §1 orientation fix and test. Re-run the 4-model ladder (32 chains, 5 seeds).
2. §2 instrumentation (ε and T traces), then T0 initialisation and lr or decoupling variants.
   Measure the warmup needed.
3. §3 short-chain harness: bias against reference plus nested R̂. Sweep C × draws per chain ×
   warmup.
4. §4 stage timers versus C, profiling, init vectorisation, pool cap, W4.
5. §6 ensemble metric during warmup (diagonal, then SNAPER-style).
6. §5 jit the stop-rule blocks.
7. §7 re-runs: E4, MEADS at equal gradients, LAPS or MAMS in the many-chain regime.

**Gates.** Keep the repo's own gates (G-A to G-D, `docs/upgrade-programme.md`). For the
many-short-chains regime, add a bias-versus-reference gate.

---

## 11. Literature options, how they map here

- **SNAPER** (Sountsov & Hoffman, arXiv:2110.11576; implemented in TFP). Ensemble diagonal
  preconditioning plus trajectory length along the principal direction. This is the closest fit to
  this repo, and the PDF is in `ref/`.
- **BlackJAX 1.7.1 `chees_adaptation`**. The diagonal estimate plus the (π/2)·sqrt(λ_max) length
  floor give §2 and §6 nearly for free.
- **MEADS / GHMC** (Hoffman & Sountsov 2022). Ensemble adaptation with no freeze needed. Re-evaluate
  after §1 at equal gradients.
- **LAPS / MAMS** (Robnik & Seljak 2026, arXiv:2601.16696; Robnik et al., NeurIPS 2025,
  arXiv:2503.01707). Built for many chains with few samples each. It needs a kernel change and a
  `sample_stats` mapping.
- **Nested R̂** (Margossian et al., arXiv:2110.13017) and Margossian & Gelman (arXiv:2311.02726) for
  diagnostics and warmup length in the many-chain regime.
- **Vectorised FSM NUTS** (Dance et al., ICML 2025; github.com/hwdance/jax-fsm-mcmc). A GPU NUTS
  arm with divergences, as a method-versus-hardware control. It is not a replacement for ChEES
  sampling.
