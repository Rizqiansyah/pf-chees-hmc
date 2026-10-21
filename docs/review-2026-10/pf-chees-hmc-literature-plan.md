# pf-chees-hmc: alternative methods, references and test plan

**What this is.** A companion to `pf-chees-hmc-review-handoff.md`. That note covers bugs and
re-measurements inside the current pipeline (§ numbers below refer to it). This one covers
**alternative methods from the literature**: what each is, why it might help this project, how to
test it against the repo, and when to stop.

**Owner constraints**
- HMC-class sampler, so divergences remain a diagnostic.
- GPU, many chains. The target regime is about 512–1024 chains × about 10 draws, with long-chain
  runs as a secondary regime.
- Pathfinder initialisation.
- Assume the parameterisation is already good. Funnel geometry is out of scope.

**Claim tags**
- **[paper]**: read in the paper or abstract.
- **[repo]**: taken from the repo's own `docs/` evidence tables (`docs/gpu-upgrade-plan.md` §4,
  `docs/ladder-results.md`). These were not re-verified here.
- **[source]**: read in BlackJAX 1.7.1 source.
- **[unverified]**: from background knowledge. Check before relying on it.

---

## 0. Prerequisites: do these before any bake-off

Without these, every comparison below is biased towards whichever arm happens to dodge the bugs.

1. **Fix the metric orientation** (handoff §1). Use `inverse_mass_matrix = cov`, not `inv(cov)`.
2. **Instrument the warmup and initialise trajectory length** (handoff §2). Log ε, T and steps per
   warmup step, and set T0 ≈ π/2 in whitened units.
3. **Use a many-short-chains scoring harness** (handoff §3). Score by bias against a reference
   posterior plus nested R̂. Do not use split-R̂ on 10–40 draws per chain.
4. **Vectorise the init stage and cap the Pathfinder pool** (handoff §4). Otherwise wall time at
   C = 1024 measures Python loops.

---

## 1. Bake-off protocol (shared by every arm)

**Models**
- 8 Schools (NC), GEV, logistic 1000×25, radon NC, plus the owner's own deep hierarchical model.
- Skip centred funnels, per the owner.
- Reference posteriors: PosteriorDB reference fits where they exist. Otherwise long CPU NUTS runs
  meeting PosteriorDB criteria (R̂ < 1.01, no divergences, about 10k draws).

**Regimes**
- **R1, many short chains:** C ∈ {128, 512, 1024}, draws per chain ∈ {1, 10}. Main target.
- **R2, long chains:** C = 32–128, 1,000+ draws per chain. Secondary.

**Primary metrics**
- **R1:** gradient evaluations *per chain* to reach b²_max < 0.01, where
  b²[f] = (E_ensemble[f] − E_ref[f])² / Var_ref[f] for f = xᵢ and xᵢ². Also wall-clock to the same
  target, with warmup and compilation included and cold and warm reported separately.
  - Per-chain gradient count is the sequential-depth proxy. It is the metric MEADS and LAPS report
    [paper], so results are comparable to the literature.
- **R2:** ESS/s including warmup and compile, plus gradients per ESS for the worst variable.

**Secondary metrics**
- Nested R̂ (superchains of k chains sharing a start).
- Divergences per 1,000 draws, reported *per kernel definition* (§5). Do not compare rates across
  kernels as if they were the same thing.
- Total gradients, which is energy and cost.
- Peak memory, and the largest C that fits on the card.

**Fairness**
- Same Pathfinder starts, same log density, same C.
- At least 5 seeds; report the median and the 10th–90th percentile.
- Device not shared during a run.
- Keep the repo's gates G-A to G-D (`docs/upgrade-programme.md`).

**Logging.** Commit harnesses and raw JSONL under `tools/` and `results/`. Do not keep them in
`~/`.

---

## 2. Arms to test, in recommended order

### A0. Baseline: current ChEES after the handoff fixes
This is the comparison point for every arm. It is the current pipeline with the §1 and §2 fixes
applied.

### A1. BlackJAX 1.7.1 `chees_adaptation(mass_matrix_estimation="diagonal")` with the length floor
- **What it is** [source]. The ensemble Welford diagonal metric is estimated over the last
  `mass_matrix_window_fraction` of warmup. It engages after `max(64, 2√d)` pooled samples. The ChEES
  criterion is whitened by this metric. The trajectory length is floored at (π/2)·√λ_max of the
  whitened ensemble covariance (`chees_adaptation.py:93-236, 652-700`).
- **Why it might help.** It replaces the single-fit Pathfinder metric with an ensemble estimate
  (handoff §6), and the floor directly counters the slow growth of T (handoff §2).
- **Route.** Public API, or port the Welford accumulator and the floor into the repo's forked loop
  in `utils.py`. Effort: hours.
- **Watch.** The docstring scopes validation to "located (typical-set) initializations", and says
  funnel-like targets can lose per-draw ESS [source]. That is fine given the owner's constraints.
- **Stop if** it does not beat A0 on gradients per chain to the bias target on at least 3 of the
  5 models.

### A2. SNAPER-HMC (Sountsov & Hoffman 2021)
- **What it is.** ChEES-family adaptive HMC that adapts step size, a diagonal metric from the
  ensemble, and trajectory length. The length is tuned with a criterion projected onto the
  top principal direction, estimated with a minibatch Oja update [repo, F-tier].
- **Why it might help.** This is the most direct published answer to this repo's problems. It gives
  the metric from the ensemble, and a trajectory-length signal focused on the hardest direction.
  The repo's reading of the paper is that ChEES both overshoots, because it has no cost term, and
  undershoots on pathological spectra [repo].
- **Route.** Benchmark first, port later.
  1. TFP ships it as `tfp.experimental.mcmc.sample_snaper_hmc`.
  2. Run it on the JAX substrate (`tfp.substrates.jax`) with the PyMC jaxified log density
     flattened to one vector. Check the argument names in the installed TFP version.
  3. If it wins, port the criterion into the BlackJAX loop, since the repo stays on BlackJAX.

  Effort: a day to benchmark, two to three days to port.
- **Watch.** TFP's defaults for the number of adaptation steps and the learning rate may differ
  from the repo's. Record them.
- **Stop if** it does not beat A1 by about 1.5× or more on gradients per chain to the bias target.

### A3. Ensemble Fisher-score metric
- **What it is.** It estimates the metric from draws **and their gradients**. The diagonal version
  is σᵢ = (Var xᵢ / Var ∇ᵢ)^{1/4}. A low-rank variant adds eigen-directions of the draw and gradient
  covariances. Source: the low-rank Fisher preconditioning paper the repo cites as
  arXiv:2603.18845, the approach nutpie uses [repo]. The repo also records nutpie's practical
  recipe, including clamps, the eigen cutoff of 2, the default trap, and resetting step size on
  metric change, as gate G1b in `docs/upgrade-programme.md`.
- **Why it might help.** With 512 or more chains, *one* ensemble snapshot gives a usable diagonal,
  and the gradients are already in the HMC state (`DynamicHMCState.logdensity_grad`), so it costs
  nothing extra.
  - The repo tested Fisher only on **Pathfinder draws**, which is the weak source (handoff §6),
    and against an inverted baseline (handoff §1). So the earlier E4 verdict does not settle this.
  - The paper's reported gain is about 4× in ESS per gradient (median). The repo notes it is
    dominated by small d, and that funnels were excluded [repo].
- **Route.** In the forked warmup loop, at refresh points (for example steps 50, 100 and 200),
  compute the ensemble Fisher diagonal, swap the metric, and re-run the ε search or reset dual
  averaging. Try diagonal first, low-rank second. Effort: about one day.
- **Stop if** the diagonal is not at least as good as the A1 Welford diagonal at equal warmup.
  Drop low-rank if it adds less than 1.2× on correlated models.

### A4. LAPS-style hybrid: unadjusted transport phase, then ChEES-HMC
- **What it is.** LAPS (Robnik & Seljak 2026) runs unadjusted microcanonical Langevin dynamics with
  an ensemble-set step-size schedule to move chains from a cold start quickly. It then switches on
  Metropolis adjustment (MAMS) for fine convergence [paper].
  - Results [paper], with 4096 chains initialised from the prior, in gradients per chain to
    b²_max < 0.01: IRT 537 (ChEES) vs 185 (LAPS); German Credit 1168 vs 300.
  - LAPS needs C ≳ 128–256 chains [paper].
  - Its ablation shows that **the microcanonical dynamics matter**. The same scheme with
    underdamped Langevin needs 650 gradients on IRT, worse than ChEES at 537 [paper].
- **Why it might help.** It is built for exactly the R1 regime. The repo's D7 rejection was
  "wrong regime (we run 32 × 2000)", and that argument does not apply to the owner's goal.
- **Hybrid design**, which keeps divergences meaningful:
  1. **Transport phase.** Unadjusted MCLMC (public `blackjax.mclmc` kernel) from the Pathfinder
     starts. Use the EEVPD-targeted step size ε ← ε·(EEVPD_target/EEVPD)^{1/6}, and noise scale
     L = 2·√ΣVar[xᵢ]. Stop when the relative fluctuation of ensemble E[xᵢ²] over a moving window
     falls below 0.01 [paper].
  2. **Handoff.** Ensemble diagonal metric (Welford or Fisher, A1 or A3), an ε search, and T0
     from the ensemble (π/2·√λ_max).
  3. **Short ChEES or SNAPER adaptation** (about 100–200 steps), then HMC sampling. This is where
     divergences are recorded.
- **Route.** Do not use `blackjax.adaptation.laps` as an API: it is not exported, and
  `laps.py:291` calls `.item()`, so it cannot be traced [source]. The repo's audit found other
  paper/code mismatches [repo]. Build from public pieces instead. This is the repo's own W6a arm.
  Effort: two to three days.
- **Stop if** the transport phase does not cut total gradients per chain to the bias target by
  about 2× or more versus A1 or A2.

### A5. Full MAMS: Metropolis-adjusted microcanonical HMC
- **What it is.** Microcanonical dynamics with an MH step, so it is asymptotically unbiased. The
  paper reports up to 7× faster than NUTS on benchmarks [paper], but in the *sequential*
  (few-chain) regime; the authors explicitly do not consider the many-short-chains regime [paper].
  It has a well-defined energy error, so a divergence-style flag is definable.
- **Run only if** A4 shows the microcanonical *kernel* (not only the transport) is what wins.
- **Cost.** `sample_stats` needs a mapping: four of the repo's six keys have no source field
  [repo, D5.3]. The dense metric is not supported; diagonal or low-rank only [repo].

### A6. MEADS / generalized HMC (Hoffman & Sountsov 2022)
- **What it is.** GHMC (one leapfrog step plus partial momentum refresh) with ensemble chain
  adaptation of all parameters, using folds of chains. It does not perturb the stationary
  distribution, so there is no freeze and no separate warmup phase [paper]. It is public in
  BlackJAX (`meads_adaptation` driving GHMC) [repo].
- **Why re-test.** The repo's probe failed: R̂ 1.03 on 8 Schools, and ε of 0.0 or NaN on radon
  [repo]. But it ran against the inverted metric, its harness is not in the repo, and the comparison
  was at equal *draws*. GHMC needs many more draws per effective sample; compare at equal
  *gradients*.
- **Guards** [repo]:
  - `num_chains % num_folds == 0`, and `num_chains // num_folds ≥ 2`. One chain per fold silently
    gives NaN.
  - The 1-D `momentum_inverse_scale` is squared, which is a trap.
  - `alpha` outside [0, 1] gives a silent NaN.
- **Divergence caveat.** A one-step energy error is a weaker posterior-pathology signal than a full
  trajectory's.
- **Stop if** it is still worse than A1 at equal gradients after the prerequisites.

### A7. Tempered SMC with HMC mutation moves
- **What it is.** It anneals from prior to posterior over a population of particles, with HMC moves
  at each temperature. The kernel parameters can be tuned from the particle population; see
  Buchholz, Chopin & Jacob, *Adaptive tuning of HMC within SMC* [unverified citation details].
  - It is in BlackJAX as `adaptive_tempered_smc` [unverified in 1.7.1; check the export].
  - It is parallel over particles. Sequential depth is about the number of stages × mutation steps.
  - It yields the marginal likelihood, and its HMC moves still report divergences.
- **Why mention it.** It replaces "long warmup per chain" with "a few dozen stages". That could be
  shallower than ChEES warmup for some models.
- **Run only if** sequential depth remains the bottleneck after A1–A4. It is a different pipeline
  shape, so give it its own budget.

### A8. Vectorised FSM NUTS (Dance et al. 2025): reference arm
- **What it is.** NUTS restructured as a finite-state machine so `vmap`ped chains do not wait for
  the slowest tree [paper]. Reported speedups over BlackJAX NUTS are close to 10× at 100 chains and
  about 3× or more at 500 chains on a skewed mixture-of-Gaussians target [paper]. Code:
  `github.com/hwdance/jax-fsm-mcmc`.
- **Role.** A GPU-NUTS reference arm with gold-standard divergences, separating method gains from
  hardware gains. The repo's protocol (`docs/gpu-upgrade-plan.md` §7) already calls for this.
  The repo's rejection of FSM *for ChEES* is correct: L is shared, so there is nothing to
  de-synchronise.
- **Route.** Warm up with A1 or A2, freeze ε and the metric, then sample with FSM-NUTS. Confirm it
  exposes per-draw divergence flags. Effort: one to two days, since it is research code.

### A9. Adaptive MALT (Riou-Durand, Sountsov, Vogrinc, Margossian, Power 2023): parked
Tuning-free Metropolis-adjusted Langevin trajectories, designed to work with many parallel chains
[paper]. It is HMC-class. There is no BlackJAX implementation, so the effort is high and the
expected gain over A1, A2 or A6 is unclear. Revisit only if A6 looks promising but unstable.

### Skip, with reasons
- **GIST** (arXiv:2404.15253): self-tuning per transition gives per-chain variable trajectory
  lengths, which brings back NUTS-style lockstep padding on GPU.
- **Parallel-in-time MCMC via DEER / quasi-Newton** (Zoltowski et al. 2025): it parallelises a
  *fixed-parameter* chain across its length [paper]. Adaptation, which is this project's bottleneck,
  breaks the fixed recursion. The method is also memory-heavy and research-grade.
- **Flow preconditioning or TESS:** this is a geometry fix, which is out of scope per the owner.

---

## 3. Decision flow

```
Prerequisites (handoff §1–4) ─► A0 baseline
        │
        ├─► A1 (hours) ─► A3 diagonal on A1's loop (1 day)
        │         └─ best of A1/A3 = "ensemble-metric ChEES"
        ├─► A2 SNAPER via TFP (1 day) ── wins ≥1.5×? ─► port to BlackJAX
        │
        ├─► A4 LAPS-transport hybrid on top of the best of the above (2–3 days)
        │         └─ microcanonical kernel itself wins? ─► A5 full MAMS
        ├─► A6 MEADS at equal gradients (1 day, mostly harness)
        ├─► A8 FSM-NUTS reference arm (1–2 days)
        └─► A7 tempered SMC only if depth still dominates
```

**What "done" looks like.** One recommended default for R1 and one for R2. Each comes with gradient
counts per chain to the bias target, wall-clock cold and warm, nested R̂, divergence behaviour, and
the largest C that fits. Losing arms get documented with the measurement that eliminated them.

---

## 4. References

**Core ChEES family**
- Hoffman, Radul, Sountsov (2021). *An Adaptive-MCMC Scheme for Setting Trajectory Lengths in
  Hamiltonian Monte Carlo.* AISTATS, PMLR 130:3907–3915.
  https://proceedings.mlr.press/v130/hoffman21a.html
- Sountsov & Hoffman (2021). *Focusing on Difficult Directions for Learning HMC Trajectory
  Lengths* (SNAPER). arXiv:2110.11576. TFP API:
  https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/sample_snaper_hmc
- Hoffman & Sountsov (2022). *Tuning-Free Generalized Hamiltonian Monte Carlo* (MEADS). AISTATS,
  PMLR 151:7799–7813. https://proceedings.mlr.press/v151/hoffman22a.html
- Riou-Durand, Sountsov, Vogrinc, Margossian, Power (2023). *Adaptive Tuning for Metropolis Adjusted
  Langevin Trajectories.* AISTATS, PMLR 206:8102–8116. arXiv:2210.12200.

**Microcanonical**
- Robnik & Seljak (2026). *Faster parallel MCMC: Metropolis adjustment is best served warm* (LAPS).
  arXiv:2601.16696 (preprint).
- Robnik, Cohn-Gordon, Seljak (2025). *Metropolis Adjusted Microcanonical Hamiltonian Monte Carlo*
  (MAMS). NeurIPS 2025. arXiv:2503.01707.

**Metric estimation**
- Low-rank Fisher preconditioning, arXiv:2603.18845, as cited in the repo
  (`docs/gpu-upgrade-plan.md` §4). Verify the authors and title before citing. The nutpie
  implementation details are summarised in the repo's gate G1b.
- Zhang, Carpenter, Gelman, Vehtari (2022). *Pathfinder: Parallel quasi-Newton variational
  inference.* JMLR 23(306). arXiv:2108.03782.

**Diagnostics and workflow in the many-chain regime**
- Margossian, Hoffman, Sountsov, Riou-Durand, Vehtari, Gelman (2024). *Nested R̂: Assessing the
  convergence of MCMC when running many short chains.* Bayesian Analysis. arXiv:2110.13017.
- Margossian & Gelman (2023). *For how many iterations should we run Markov chain Monte Carlo?*
  arXiv:2311.02726.
- Sountsov, Carroll, Hoffman (2024). *Running Markov Chain Monte Carlo on Modern Hardware and
  Software.* arXiv:2411.04260.

**GPU execution**
- Dance, Glaser, Orbanz, Adams (2025). *Efficiently Vectorized MCMC on Modern Accelerators.* ICML,
  PMLR 267:12436–12458. Code: https://github.com/hwdance/jax-fsm-mcmc
- Zoltowski, Wu, Gonzalez, Kozachkov, Linderman (2025). *Parallelizing MCMC Across the Sequence
  Length.* arXiv:2508.18413.

**SMC** [unverified details]
- Del Moral, Doucet, Jasra (2006). *Sequential Monte Carlo samplers.* JRSS-B.
- Buchholz, Chopin, Jacob (2021). *Adaptive tuning of Hamiltonian Monte Carlo within sequential
  Monte Carlo.* Bayesian Analysis.

**Software**
- BlackJAX 1.7.1: `adaptation/chees_adaptation.py` (diagonal and length floor),
  `adaptation/meads_adaptation.py`, `mcmc/mclmc.py`, `mcmc/ghmc.py`, and `adaptive_tempered_smc`
  (check the export).
- TFP JAX substrate: `tfp.substrates.jax.experimental.mcmc` (SNAPER).
