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

## E2 — Correctness guards

1. **NaN gradient at a feasible start.** GEV at `model.initial_point()` has log-density −7.749 and
   all three partial derivatives NaN. `_ensure_finite_inits` tests log-density only, so it cannot
   see this; the pipeline escapes only because its seed differs by luck. Guard on gradient finiteness.
2. **Key collision — confirmed.** `key_warm` (`sampler.py:426-427`: `rng = jax.random.key(seed)`,
   `_, key_warm = jax.random.split(rng)`) and `key_sample` (`sampler.py:452` and `:587`, same parent
   key, same child index) are the **same key**, so sampling draws from the prefix of the warmup key
   stream. The ε0 probes (`sampler.py:407`) take children of that same parent directly. Fix: derive
   each stage with `jax.random.fold_in(rng, stage)` for stage 0/1/2 (tuner, warmup, sampling) instead
   of positional children. This changes realisations, so it needs the 5-seed matrix re-run as a
   non-regression check even though it is not a default change.
3. **Misleading display — confirmed at both sites.** Warmup shows the arithmetic mean
   (`utils.py:109`, printed at `:81`) and sampling also shows the arithmetic mean
   (`sampler.py:565`), while ChEES adapts on the harmonic mean excluding divergences
   (`chees_adaptation.py:358-363`). This already caused one wrong diagnosis in this project.
4. **Undefined BFMI on GEV.** `sample_stats.energy` contains non-finite values; find why and either
   fix the export or document it as undefined rather than emitting a number.

## E3 — Test wiring

`pixi add --pypi pytest` (or a dev feature), a CPU `test` task, a GPU `test-gpu` task that runs the
two example models at reduced budgets. Both current test files already run standalone via
`__main__`, so this is packaging, not rewriting.

## E4 — Metric upgrade, conditional

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
