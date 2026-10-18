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

Models, cheapest-gradient first: (1) 8 Schools funnel d=10 (have), (2) GEV d=3 (have),
(3) ill-conditioned Gaussian d=50 and d=500 with condition number ~1e4, (4) logistic regression
1000×25 and 10,000×50, (5) one real model from PosteriorDB so the result is not self-invented.

Arms: GPU at 32 / 128 / 256 chains; CPU NUTS with **chains = cores** at 4 / 8 / 16 / 32.
Report per cell: wall (cold and warm separately), bulk ESS/s total and worst-variable, min
tail-ESS, max R̂, divergences per 1,000 draws, mean leapfrog steps, and the padding ratio
(useful gradients / issued gradients).

## E2 — Correctness guards

1. **NaN gradient at a feasible start.** GEV at `model.initial_point()` has log-density −7.749 and
   all three partial derivatives NaN. `_ensure_finite_inits` tests log-density only, so it cannot
   see this; the pipeline escapes only because its seed differs by luck. Guard on gradient finiteness.
2. **Key collision.** `key_warm` (`sampler.py:399-400`) and `key_sample` (`sampler.py:560`) are the
   same key, so sampling draws from the prefix of the warmup key stream.
3. **Misleading display.** Both progress bars report arithmetic-mean acceptance
   (`utils.py:109`, `sampler.py:538-541`) while ChEES adapts on the harmonic mean excluding
   divergences (`chees_adaptation.py:358-363`). This already caused one wrong diagnosis here.
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

- **G1 (E4 go/no-go):** proceed only if at least one ladder model shows imm condition number ≥ 50 or
  divergence rate ≥ 1 per 1,000 draws at 32 chains. Otherwise E4 is descoped and written up as
  not-needed for this model class, because our two shipped examples (5.25 and 10.2) do not need it.
- **G2 (fair CPU arm):** every CPU cell uses chains = cores and reports its own frontier. The 4-chain
  single-thread reference is retired — it produced a 7.4× claim that measured 3.6×.
- **G3 (default changes):** no default moves without ≥5 seeds, divergence rate non-increase at equal
  chain count, and posterior means within 1σ of the previous default.
- **G4 (one commit per item):** each E2 guard ships alone, with its test.

## Question map (drives the lanes; per the hmc-perf-change skill)

| Q | Question | Owner | Evidence that settles it | Gates | Cost |
|---|---|---|---|---|---|
| Q1 | Which PosteriorDB models are cheap enough for a 6 GB A2000 and span conditioning? | lane: literature | posteriordb index + dimensions + observations | E1 model list | ~15 min |
| Q2 | What benchmark protocol do published GPU-vs-CPU HMC papers use for scaling and reporting? | lane: literature | primary papers' experiment sections | E1 protocol | ~15 min |
| Q3 | What did arXiv:2603.18845 actually measure, and does the ~4× hold for small d and dense metrics? | lane: literature | full text, numbers with tiers | E4 worth doing at all | ~20 min |
| Q4 | How do nutpie/nuts-rs clamp and regularise the metric (eigenvalue floor, gamma, rank)? | lane: literature | source, file:line | E4 implementation | ~20 min |
| Q5 | In BlackJAX 1.7.1, can `window_adaptation_low_rank` / `diagonal_estimator="fisher"` be used to *produce* a metric usable by `dynamic_hmc`, and with what chain-count constraints? | lane: source audit | file:line | E4 route (library vs hand-rolled) | ~20 min |
| Q6 | Which PyTensor/JAX op makes GEV's gradient NaN at a finite log-density point, and is it upstream? | lane: source audit | minimal repro (3-dim, CPU only) + graph read | E2.1 fix shape | ~20 min |
| Q7 | Confirm the key-collision line numbers and that a `fold_in` split changes no benchmark | me | read + one A/B run | E2.2 | ~15 min |
| Q8 | Confirm harmonic vs arithmetic at both display sites | me | read | E2.3 | ~5 min |
| Q9 | Is non-finite `energy` a divergent-transition artefact or a real overflow? | me | inspect one GEV run's sample_stats | E2.4 | ~10 min |
| Q10 | Build the ladder harness and run the two existing models through it as a smoke test | me | harness + one row per model | E1 start | ~30 min |

No overlap: literature lane takes Q1–Q4, source lane takes Q5–Q6, rest is mine.

## Sequencing

1. E3 first (1 h) — it makes every later claim reproducible by someone else.
2. E2.1–E2.3 (half a day) — cheap guards, each its own commit with a test.
3. E1 ladder (a day, mostly machine time) — produces the G1 verdict and the honest crossover.
4. E2.4 whenever it is cheap; it is a reporting bug, not a sampler bug.
5. E4 only if G1 opens; otherwise the ladder result is written up as evidence that the current
   metric is adequate for this model class.
