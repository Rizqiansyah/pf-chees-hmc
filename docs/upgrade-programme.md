# Upgrade programme 2

Scope: throughput generality, correctness guards, and the metric — cold-start work explicitly
excluded by the owner. Built on the state left by programme 1: ε0 traced once (`cb74b1b`), default
chains 32 (`c128ca1`), warmup 2,000 (`93b988f`), CPU frontier measured and the inflated ratios
withdrawn (`ddeb055`).

Measured starting point, this box (RTX A2000 vs 32-core CPU), warm medians, draws 2000, warmup 2000:
8 Schools 45,614 ESS/s at 32 chains and 134,898 at 128 (best CPU 37,494); GEV 7,856 and 17,464
(best CPU 6,733). Imm condition numbers: 5.25 (8 Schools), ~10.2 (GEV). Both models have 3–10
parameters, which is the GPU's weakest regime.

## Epics

| # | Epic | Goal | Done when | Est. |
|---|---|---|---|---|
| E1 | Model ladder | Show whether the GPU advantage grows with model size and stiffness | curve published for ≥4 models × chain counts, CPU arm at its own frontier | 1 day |
| E2 | Correctness guards | Remove four known silent-failure hazards | each has a failing-then-passing test, committed separately | 1 day |
| E3 | Test wiring | Make the loop runnable by someone else | `pixi run test` green on CPU, `pixi run test-gpu` opt-in | 1 h |
| E4 | Metric upgrade | Fisher / low-rank preconditioning, the one published ~4× method gain | only if gate G1 opens | 2–3 days |

## E1 — Model ladder

Models, from PosteriorDB (Magnusson, Torgander, Bürkner, Zhang, Carpenter, Vehtari — arXiv:2407.04967;
not "Charnti", which an earlier note here had wrong), verified against `stan-dev/posteriordb` at
`91027f9c`. It ships plain files, not an R package: 120 Stan models, 91 datasets, 148
model+data pairs, 57 published reference fits. **Reference fits stop at d = 66**, and only 12 of
the 63 posteriors with 10 ≤ d ≤ 500 have one, so a "10–500 parameters with ground truth" ladder is
not available and larger models need a self-generated reference under the DB's published criteria
(10,000 draws per parameter, R̂ < 1.01, lag-1 autocorrelation < 0.05, no divergences).

| posterior id (as the database names it) | d | N | ref fit | role on the ladder |
|---|---|---|---|---|
| 8 Schools (repo example) | 10 | **8** | — | centered funnel control; N=8 is below the database's own data floor |
| GEV Port Pirie (repo example) | 3 | 259 | — | stiff, cheap |
| `nes1972-nes` | 10 | 1350 | **yes** | same parameter count as 8 Schools with 168× the data, benign |
| `diamonds-diamonds` | 26 | 5000 | **yes** | well-conditioned GLM control at realistic N |
| `mcycle_gp-accel_gp` | 66 | 133 | **yes** | centered GP with whitened eigen-expansion; Seyboldt et al. **excluded it because nothing converged** |
| `irt_2pl-irt_2pl` | 144 | 2000 | self-gen | the 8-Schools pathology at 14× the parameters |
| `radon_mn-radon_variable_intercept_slope_centered` / `_noncentered` | 175 | 919 | self-gen | the conditioning A/B pair, same model two parameterisations |
| `science_irt-grsm_latent_reg_irt` | 408 | 2744 | self-gen | correlated high-d plus funnel |

Synthetic ill-conditioned Gaussians are **dropped**: my d=50 build made CPU NUTS fail too
(R̂ 1.33, tail-ESS 23), so that cell measured the model rather than the sampler.

Protocol spine, chosen because none of the three published GPU/CPU studies is copyable 1:1
(Sountsov reports ESS for one parameter with no seed count and no warmup at all; Dance excludes
warmup and uses identity metrics for its NUTS rows; Seyboldt states no hardware and no repeats):
bulk ESS summed over parameters divided by wall seconds **including warmup and compilation**,
per-parameter minimum reported alongside the total, **gradients per ESS** as the hardware-free
second axis, chain sweep GPU {1,4,16,32,64,128} and CPU {1,8,32}, ≥10 seeds with median and IQR,
cold versus warm flagged, divergences per 1,000 draws plus tail-ESS, E-BFMI and max R̂ per run, and
**non-converged models reported rather than dropped** — dropping them is exactly how the Fisher
paper's 4× became funnel-free.

**Status: 5 of 6 model families measured, results in `docs/ladder-results.md`.** Two findings
already change the programme. (a) The GPU advantage is **not** monotone in size — 1.17× (d=3),
1.21× (d=10), 3.0× (d=25), then **0.58× at d=175** on radon noncentered, where 32 CPU cores win
outright; what tracks the ratio is steps-per-draw and per-gradient cost, not parameter count. (b)
**radon centered (the funnel) fails on our sampler**: R̂ 4.89 with the trajectory frozen at 4.8
leapfrog steps, 273 ESS/s, while CPU NUTS is merely poor (R̂ 1.08, 29–99 divergences per 1,000).
That is the same failure Seyboldt et al. excluded and Dance et al. reported as `nan` ESS, now
reproducible on our box — and it becomes E4's acceptance target.

## E2 — Correctness guards

1. **NaN gradient at a feasible start — done in `07db02c`.** GEV at `model.initial_point()` has
   log-density −7.749 and all three partial derivatives NaN. `_ensure_finite_inits` tested
   log-density only, so it could not see this; the pipeline escaped only because its seed differed
   by luck. Starts now require a finite gradient, verified end-to-end on GEV and 8 Schools with no
   diagnostic or throughput change. Root cause is deeper than a JAX quirk: `GenExtreme.logp`
   differentiates the unselected `xi != 0` branch, and the density is not C¹ at ξ = 0 — PyTensor's
   own autodiff returns NaN there too.
2. **Key collision — confirmed.** `key_warm` (`sampler.py:426-427`: `rng = jax.random.key(seed)`,
   `_, key_warm = jax.random.split(rng)`) and `key_sample` (`sampler.py:452` and `:587`, same parent
   key, same child index) are the **same key**, so sampling draws from the prefix of the warmup key
   stream. The ε0 probes (`sampler.py:407`) take children of that same parent directly. Fix: derive
   each stage with `jax.random.fold_in(rng, stage)` for stage 0/1/2 (tuner, warmup, sampling) instead
   of positional children. **Done in `a9d1a3e`** (1 = warmup, 2 = sampling, 3 = the field-discovery
   probe, which had used a hard-coded `jax.random.key(0)` and so ran identically for every seed).
   Guarded by `tests/test_rng_stage_keys.py`, which reads the stage numbers back out of the source.
   Realisations change; the recorded throughput and diagnostic claims are stream-invariant in
   distribution, so the 5-seed matrix was not re-run.
3. **Misleading display — confirmed at both sites.** Warmup shows the arithmetic mean
   (`utils.py:109`, printed at `:81`) and sampling also shows the arithmetic mean
   (`sampler.py:565`), while ChEES adapts on the harmonic mean excluding divergences
   (`chees_adaptation.py:358-363`). This already caused one wrong diagnosis in this project.
4. **Undefined BFMI on GEV — done in `5f5b0d3`.** Cause: `blackjax/mcmc/hmc.py:171` stores the
   *proposal* state's energy, so a divergence writes +inf and arviz's BFMI is NaN. The harness now
   reports E-BFMI over finite entries plus `energy_nonfinite`; measured 1.154 (GEV), 0.979
   (8 Schools), 0.990 (logistic), 0.597 (radon noncentered), 0.397 (radon centered). Every cell
   clears the 0.3 gate, so E-BFMI is not a funnel detector — R-hat and trajectory length are.

## E3 — Test wiring — done (`95b7134`, now 16 tests)

`pixi add --pypi pytest` (or a dev feature), a CPU `test` task, a GPU `test-gpu` task that runs the
two example models at reduced budgets. Both current test files already run standalone via
`__main__`, so this is packaging, not rewriting.

## E4 — Metric upgrade: **measured, shipped as an option, not a default**

Full A/B with eigen-spectra and the rejected scale-match repair: see `docs/ladder-results.md`,
"Metric A/B (E4)". Summary — `fisher_lowrank` is 1.22x on 8 Schools and 1.96x on logistic with
R-hat unchanged, 0.19x on radon noncentered, and the centered funnel stays at R-hat 2.10-5.69 under
every metric configuration. The discriminator is realised trajectory length.

G1 is satisfied: ladder condition numbers are 426 (logistic), 6,660 (radon centered) and 25,116
(radon noncentered) against the 50 threshold, and the centered funnel already fails quality gates.
Acceptance target is `radon_centered` at R̂ < 1.05, not a speed-up on the model that is fine.

If G1 opens: adopt a Fisher-divergence or low-rank preconditioner through the library where the
contract allows, otherwise compute the metric ourselves from quantities we already have. BlackJAX
ships `window_adaptation_low_rank` (implements Algorithm 1 of arXiv:2603.18845) and
`window_adaptation(diagonal_estimator="fisher")`, plus the Stan shrinkage formula in
`adaptation/mass_matrix.py:319-353`. Known constraint from earlier work: the staged/window engine
drives the classic HMC family, not `dynamic_hmc`, so adoption likely means *computing* the metric
with library code and feeding it into our existing dense-metric path.

## Decision gates

- **G1 (E4 go/no-go):** proceed only if a ladder model with a usable reference fit shows imm
  condition number ≥ 50 or a divergence rate ≥ 1 per 1,000 draws at 32 chains. The evidence makes
  this less likely than it looked, and sharpens what E4 would be claiming: the Fisher result is
  dominated by **small** models (median d = 9, 68 of 114 at d ≤ 15), the low-rank and
  gradients-per-ESS gains are reported as the *same* 4×, which is only overhead-free at tiny d, and
  **every centered/funnel model was excluded for non-convergence** — including
  `eight_schools_centered` and `mcycle_gp-accel_gp`. So the published 4× does not cover our
  geometry, and the regime where a low-rank term should matter (correlated 100–400 parameters,
  funnelled) is the set that paper never isolated. Our own condition numbers, 5.25 and ~10.2, sit
  where that paper's §2.3 simulation shows Fisher and covariance diagonals barely separating.
- **G1b (if E4 proceeds):** implement nutpie's measured recipe, not the paper's prose — diagonal
  σ = (Var[x] / Var[∇])^(1/4), initialised at diag(|score₀|), covariances regularised as
  XXᵀ/γ + I with γ = 1e-5, eigenfilter keep λ > 2 or λ < 1/2, clamps [1e-20, 1e20] on the diagonal
  and [1e-10, 1e10] on the low-rank σ, ≥3 good draws before any update, draws from divergent
  trajectories excluded unless |index| > 4, whole update rejected on any non-finite value, step size
  re-initialised on the first metric change. Note the trap: `sample.py:1064` documents the
  eigenvalue cutoff default as 100 while the Rust default that actually applies is 2.0.
- **G2 (fair CPU arm):** every CPU cell uses chains = cores and reports its own frontier. The 4-chain
  single-thread reference is retired — it produced a 7.4× claim that measured 3.6×.
- **G3 (default changes):** no default moves without ≥5 seeds, divergence rate non-increase at equal
  chain count, and posterior means within 1σ of the previous default.
- **G4 (one commit per item):** each E2 guard ships alone, with its test.

## Question map (drives the lanes; per the hmc-perf-change skill)

| Q | Question | Owner | Evidence that settles it | Gates | Cost |
|---|---|---|---|---|---|
| Q1 | Which PosteriorDB models are cheap enough for a 6 GB A2000 and span conditioning? | lane — **answered**: 120 models / 148 pairs / 57 reference fits, reference fits stop at d=66; ladder set in E1 above | posteriordb index + dimensions + observations | E1 model list | done |
| Q2 | What benchmark protocol do published GPU-vs-CPU HMC papers use for scaling and reporting? | lane — **answered**: none copyable 1:1; adopted spine in E1 above (ESS/s incl. warmup + compile, gradients/ESS second axis, chain sweeps, ≥10 seeds) | primary papers' experiment sections | E1 protocol | done |
| Q3 | What did arXiv:2603.18845 actually measure, and does the ~4× hold for small d and dense metrics? | lane — **answered**: 4× is mostly small-d (median d=9, 68/114 at d≤15) but **every centered/funnel model was dropped for non-convergence**, hardware and repeats not stated → E4 gate rewritten as G1 | full text, numbers with tiers | E4 worth doing at all | done |
| Q4 | How do nutpie/nuts-rs clamp and regularise the metric (eigenvalue floor, gamma, rank)? | lane — **answered**: recipe recorded as G1b above, incl. the doc-vs-default eigenvalue cutoff trap (100 documented, 2.0 applied) | source, file:line | E4 implementation | done |
| Q5 | In BlackJAX 1.7.1, can `window_adaptation_low_rank` / `diagonal_estimator="fisher"` be used to *produce* a metric usable by `dynamic_hmc`, and with what chain-count constraints? | lane: source audit | file:line | E4 route (library vs hand-rolled) | ~20 min |
| Q6 | Which PyTensor/JAX op makes GEV's gradient NaN at a finite log-density point, and is it upstream? | lane: source audit | minimal repro (3-dim, CPU only) + graph read | E2.1 fix shape | ~20 min |
| Q7 | Confirm the key-collision line numbers and that a `fold_in` split changes no benchmark | me — **done**: collision confirmed at `sampler.py:426-427` vs `:452`/`:587`; fix is `fold_in` per stage, then re-run the 5-seed matrix | E2.2 | ~15 min |
| Q8 | Confirm harmonic vs arithmetic at both display sites | me — **done**: arithmetic at `utils.py:109` and `sampler.py:565` | E2.3 | ~5 min |
| Q9 | Is non-finite `energy` a divergent-transition artefact or a real overflow? | me | inspect one GEV run's sample_stats | E2.4 | ~10 min |
| Q10 | Build the ladder harness and run the two existing models through it as a smoke test | me — **done**: `tools/bench_ladder.py` (commit `de15e59`); padding ratio 1.0 measured on both GPU models; synthetic Gaussians dropped because CPU NUTS failed them too | harness + one row per model | E1 start | done |
| Q11 | Is there any published ESS/s head-to-head of a Pathfinder / quasi-Newton metric against an adapted sample covariance? | lane — **answered**: no. The only peer-visible wall-clock comparison is Bales et al. 2019 on CPU Stan and its headline is mixed (sample covariance won on Prophet and Radon). So E4 is gated on our own A/B of `diag` vs `draw_diag` vs `low_rank` on the radon funnel pair, not on the 4×. | searched: arXiv API/ti/abs combos, Crossref, full texts | E4 justification | done |

No overlap: literature lane takes Q1–Q4, source lane takes Q5–Q6, rest is mine.

## Sequencing

1. E3 first (1 h) — it makes every later claim reproducible by someone else.
2. E2.1–E2.3 (half a day) — cheap guards, each its own commit with a test.
3. E1 ladder (a day, mostly machine time) — produces the G1 verdict and the honest crossover.
4. E2.4 whenever it is cheap; it is a reporting bug, not a sampler bug.
5. E4 only if G1 opens; otherwise the ladder result is written up as evidence that the current
   metric is adequate for this model class.

---

# E5 — Gradient economy, chain batching, warmup policy, initialisation

Opened 2026-10-18 by owner decision after the ladder and roofline work. In scope: **(1)** fewer
gradients per effective sample, **(2)** batched likelihood only, **(3)** all of the fixed-cost work,
**(4)** stratified Pathfinder initialisation. Explicitly **out of scope for now**: float32 (measured
3.3x–20x on data-heavy shapes, parked with its numbers), data subsampling per leapfrog step,
adaptive chain spawning.

Why these: a step costs `L x cost_of_one_gradient`, warmup is 2,000 of those steps regardless of how
many samples you want, and both the E4 metric A/B and the roofline probe say the leverage is in
gradients-per-effective-sample and in fixed cost, not in device throughput. Measured anchors: the
latency floor is ~130–155 us per 32-chain gradient on small models; radon's real per-transition cost
(11.4 ms at ~37 steps) matches its own gradient cost, so kernel plumbing is not the gap.

## Question map

Owner "me" = answered by a probe I run, no lane. Lanes get at least two questions whose answers could
change the design.

| # | Question | Owner | Settling evidence | Gates | Cost |
|---|---|---|---|---|---|
| G1 | Does momentum persistence (GHMC) reduce gradients per ESS versus momentum-refreshing HMC at equal trajectory length, at d < 50? | lane: ghmc-evidence | primary papers, numbers with tiers | kernel fork D6 | ~20 min |
| G2 | Does persistence interact with a **fixed lockstep** trajectory length, and does it move the optimal target acceptance away from 0.651? | lane: ghmc-evidence | primary papers + BlackJAX defaults | whether ChEES's tuning rule still applies | ~20 min |
| G3 | What does installed BlackJAX 1.7.1 GHMC actually require — init/step signature, state fields, info field names — and can our ChEES `update_adapt` call drive it unchanged? | lane: blackjax-source | file:line | implementation route | ~15 min |
| G4 | Can a stop rule live inside the warmup loop: what do `blackjax.eca` / `ns/` expose, are they jit/scan compatible, and do they carry the `shard_map` "chains" mesh constraint that killed LAPS? | lane: blackjax-source | file:line + a 3-dim CPU check | stop-rule granularity | ~15 min |
| G5 | What does Pathfinder expose for stratified init (`info` fields, PSIS weights, log-ratios), and is PS-IR reachable through public API? | lane: blackjax-source | file:line | init change | ~10 min |
| G6 | On the graphs we ship, does per-chain cost fall with chain count (reads shared) or stay flat (re-read per chain)? | **me — answered: shared, closed** | `~/pf-batchprobe.py`, us per 32-chain gradient inside a `lax.scan`: radon 133.5 (1.93/chain at 128), logistic 121.1 (3.10/chain), 8 Schools 22.5 (0.16/chain). f32 is 1.06-1.07x, not 3x. | batched-likelihood work item: **cancelled** | done |
| L1 | GHMC: does the kernel change and what does it invalidate? | **lane e5-blackjax — answered** | `GHMCInfo` is a plain `HMCInfo` so `update_adapt` reads unchanged; but momentum+slice move into the carry and `num_integration_steps` is hard-wired to 1, so `steps_mean`/`padding_ratio` become meaningless. Two routes: ChEES+GHMC (trivial swap, but the trajectory-length half of ChEES goes inert and the 0.651 target is unvalidated for 1-step dynamics) or MEADS+GHMC (public `bj.meads_adaptation`, natively multi-chain lockstep, emits `{step_size, momentum_inverse_scale, alpha, delta}`, `momentum_inverse_scale` accepts `LowRankInverseMassMatrix`) -- but MEADS+GHMC is the D6 fork, not a drop-in. | kernel fork D6 | done |
| L2 | Stop rule: is an in-graph early stop reachable without the LAPS `shard_map` constraint? | **lane e5-blackjax — answered, yes** | `blackjax.eca.while_with_info` is a pure `jax.lax.while_loop` primitive (no `shard_map`, no mesh) and `blackjax.diagnostics.{rhat,effective_sample_size,ess_bulk,ess_tail,pareto_khat,psis_weights}` are all public pure `jnp`, `jit`-compatible. Verified on CPU: a `while_loop` with `rhat` as the cond stopped dynamically at i=19. Granularity: per-block in-graph (chunk the 2000-step warmup into 250-step scans, R-hat over a window as the outer cond). JAX 0.11 gotcha: `while_loop`'s body returns just the carry, not `(carry, ys)`. Note `arviz_stats.rhat_nested` is NOT installed. | stop-rule granularity | done |
| L3 | Stratified init: what is reachable from public API and what is our current init? | **lane e5-blackjax — answered** | `info.path.elbo` (shape `(path_size,)`) is the per-path PSIS log-weights (Algorithm 5); `pathfinder.sample` is i.i.d.-only, but the stratified/low-discrepancy transform is an affine map on public `PathfinderState` fields, replicable with a Halton `u` and no private call. `multi_approximate` + `psis_weights` is fully public. Our current init (single best-ELBO path + one random draw per chain) is the weakest available option; the strictly-better public upgrade is a PSIS mixture over `info.path`. | init change | done |
| L4 | Warmup stop rule: published criteria + does early-stopped warmup bias the posterior? | lane (queued, flash slot) | primary papers on stopping / convergence-aware warmup | stop-rule thresholds, bias gate G-B | queued |
| L5 | Re-trace elimination: what exactly keys a JIT compile on, and is the per-chain-count cost avoidable? | **lane e5-retrace (running)** | JAX version, cache-key docs, `jax_log_compiles` attribution | re-trace fix | running |
| L6 | Metric refresh during warmup: is it implementable without touching private state, and what must be reset? | **lane e5-metric-refresh (running)** | ChEES state/parameter fields, metric-switch step-size semantics, Stan shrinkage call sites | metric-refresh work item | running |
| O1 | Where does the per-transition overhead live — 1,990 us measured in production against 462 us of model evaluation at 8 Schools (4.1x), 11.4 ms against 4.9 ms at radon (2.3x)? | **me** | arm-by-arm scan-cost attribution on the real logp: kernel only, +info carry, +Halton trajectory-length machinery, +host callback, full warmup step | largest remaining lever | ~10 min |
| G7 | Is there a supported way to make a PyMC model evaluate all chains in one wide op instead of `vmap`, and what breaks (transforms' jacobian, deterministics, coords)? | lane: pytensor-batching | source + 3-dim CPU experiment | implementation route for (2) | ~25 min |
| G8 | Is there published evidence that PS-IR / stratified init beats one random draw from the same approximation, measured in warmup length or convergence diagnostics — and that start dispersion matters for *ensemble* adaptation specifically? | lane: init-reuse | primary papers, tiers | whether (4) is worth code | ~20 min |
| G9 | Is persisting tuned HMC hyperparameters across runs standard practice, what are the documented failure modes, and do samplers re-verify after a warm start? | lane: init-reuse | Stan/PyMC docs + papers | cache design, verification block | ~15 min |
| C1 | How early in warmup are the frozen parameters (step size, trajectory-length statistics) within 10% of their 2,000-step values? | **me** | instrumented run, 250-step blocks | stop rule thresholds | ~10 min |
| C2 | Does nested R-hat separate warmup 500 / 1,000 / 2,000 on models where we know the answer? | **me** | `arviz_stats.rhat_nested` on existing draws | stop criterion validity | ~10 min |
| C3 | How many distinct XLA traces does one run create, and which knob (chains, draws, dtype) creates them? | **me** | compile-time attribution per stage | re-trace fix | ~10 min |

## Corrections to the roofline/latency claims (two rounds, both measured)

1. **First pass was wrong, and this second pass corrects the first correction.** The original probe
   timed a single jitted `value_and_grad` call and read the 130-155 us as a "latency floor". A
   corrected probe that forces the gradient into the `lax.scan` carry (so it is computed, not DCEd)
   gives, us per 32-chain batched gradient on the real graphs: **8 Schools 18.5, radon 131.6,
   logistic 116.6** (f64). The pure scan floor is only **5-7 us**. So the real per-step model cost
   for 8 Schools is ~23 us, not 154 -- the first correction stands on the number, and the
   "dispatch cost" attribution is retracted; the original probe was measuring the real value+grad,
   just with the wrong label.
2. **Float32 is ~1.05-1.1x on the real graphs, not 3.3x** (radon 131.6 f64 vs 142.5 f32; logistic
   116.6 vs 115.9; 8 Schools 18.5 vs 18.8). The 3.3x-20x figures came from a dense 919x175 synthetic
   matmul at the fp64 ALU limit; radon is a gather and never touches that limit. Parked float32
   stays parked.

**Chain batching is cancelled as a work item**, and the PyTensor lane confirmed it from the other
side: the design matrix is a single constant in the jaxpr, and per-chain marginal cost collapses
(radon 44 us at 1 chain -> 1.94 us per chain at 128). What a hand-written wide op *would* buy is
1.4-1.9x at C <= 32 narrowing to ~0.9x at 128, and it is unreachable for arbitrary user models
(`pt.vectorize_graph` replaces RVs with constants; `Matmul` has no vectorize dispatch). The jacobian
trap is documented, and it is silent: omitting it shifts every transformed dimension's gradient by
exactly 1.0 with no error.

**The prize moved, and it is now very large.** With the model priced at its true cost, 8 Schools
spends ~0.05 s of model evaluation in a 12.4 s run and radon ~0.3 s in a 43 s run. Nearly all of the
wall time is our own loop machinery and warmup -- O1 attributes it to specific components.

## Gates carried over

- **G-A (kernel change):** no kernel swap without ≥5 seeds, divergences per 1,000 draws non-increasing
  at equal chains, max R̂ ≤ 1.01, posterior means within 1σ, **and** gradients-per-ESS reported — that
  last metric is the currency of this epic, so a wall-time-only comparison is not admissible.
- **G-B (warmup policy):** an early stop must reproduce the 2,000-step posterior within 1σ on both
  example models and must reject a deliberately short warmup on GEV, where 1,000 steps produced R̂
  1.068 and a 6.2-step frozen trajectory. A stop rule that cannot detect that case is not a stop rule.
- **G-C (cache):** a persisted-hyperparameter cache ships only with a graph fingerprint that fails
  loudly when the model changes, plus a verification block of real sampling; no silent reuse.
- **G-D (init):** stratified init is only worth shipping if it changes a measured quantity (warmup
  length to reach R̂ < 1.01, or max R̂ at fixed budget), not "looks better spread".

## Baselines to beat, measured

- Gradients per effective sample: computed from existing rows as `steps_mean x draws_total / total_bulk_ESS`.
- Warmup cost: 2,000 lockstep steps ≈ 23 s at 32 chains on radon, ≈ 8 s at 8 Schools — flat in chains.
- Re-trace: 15–19 s per new chain count per model per process.
- 4,000-draw budget quality: GPU max R̂ 1.006 → 1.017 (8 Schools) and 1.025 → 1.083 (radon) as chains go 4 → 32.
