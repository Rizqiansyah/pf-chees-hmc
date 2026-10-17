# Optimisation and methodology plan — `pfcheeshmc`

Status: proposal, nothing implemented. Date: 2026-10-17.
Baseline commits: `4fb50e9` (BlackJAX 1.7 + pixi CUDA-13 env), `eb8c4c1` (`sample_stats` contract fix), `ba66ccc` (README).
Environment: Python 3.13.7, blackjax 1.7.1, jax/jaxlib 0.11.2, pymc 5.28.5, pytensor 2.38.2, numpy 2.4.6, numba 0.66.0; 1× NVIDIA RTX A2000, driver 580.178.04 (`ref/README.md` lists the papers).

---

## 1. Measured cost model

Source: instrumented 8 Schools run, 32 chains × 2000 draws, 10 000 ChEES warmup steps, `random_seed=42`, harness at `~/pf-instrument2.py` (monkey-patched `perf_counter` boundaries with `jax.block_until_ready` at each stage exit; a first pass at `~/pf-instrument.py` established that the step-size tuner repeats its cost on a second identical call: 74.59 s then 74.65 s).

| Component | Location | Time | Share |
|---|---|---|---|
| Step-size tuner | `sampler.py:350-392` | 77.1 s | 49% |
| `pathfinder.approximate` | `sampler.py:296-302` | 24.1 s | 15% |
| ChEES warmup | `sampler.py:394-422` → `utils.py:144` | 21.4 s | 14% |
| Dynamic HMC sampling | `sampler.py:569-576` | 13.2 s | 8% |
| `pathfinder.sample` (128 000 draws) | `sampler.py:304` | 7.6 s | 5% |
| Covariance + `inv` + reshape | `sampler.py:306-336` | 6.2 s | 4% |
| Finite-start repair, 32 chains | `sampler.py:185-231` | 4.0 s | 3% |
| `pm.init_nuts(adapt_diag)` | `sampler.py:167-183` | 1.4 s | 1% |
| InferenceData assembly | `sampler.py:665-740` | 1.1 s | 1% |
| **Total** | | **156.2 s** | |

Reading: **22% of the run is MCMC** (ChEES + sampling). 49% is a stage that does not exist in the ChEES paper. Numbers are inflated relative to the uninstrumented 80.1 s baseline because `block_until_ready` removes JAX's cross-stage pipelining and `JAX_LOG_COMPILES=1` adds logging; **shares** are the decision-relevant quantity, and the r̂ / divergence outputs were unchanged (max r̂ 1.0, 0 divergences).

---

## 2. Work items

Ordering is by measured value per unit of risk. Each item lists the change, the evidence, and how it is proven.

### W1 — Remove the per-chain step-size search (or vmap it)
- **Location:** `sampler.py:350-392`, called from `sample()` at `sampler.py:759`.
- **Evidence:** 49% of runtime (table above); repeats per call, 74.59 s → 74.65 s on identical inputs, so it is per-call cost and not one-time compilation; ~2.3 s per chain. Inside the loop: `float(eps_i)` and `float(info_i.acceptance_rate)` plus a `tqdm` postfix force ≥3 host syncs per chain (96 at 32 chains), and an un-jitted `bj.hmc(...).step(...)` runs per chain purely to print a value that is discarded.
- **Why it is not required:** ChEES as published already adapts ε. Algorithm 1 of Hoffman et al. (2021) takes only "initial step size ε₀", then each warmup step computes the harmonic-mean acceptance across chains and updates ε by dual averaging to a target of 0.651. BlackJAX implements exactly this: `blackjax/adaptation/chees_adaptation.py:21` (`OPTIMAL_TARGET_ACCEPTANCE_RATE = 0.651`) and the `da_state` field of `ChEESAdaptationState`. The repo's own `_run_chees` passes `init_step_size` straight into `init_adapt` (`utils.py:56`).
- **Change (preferred):** delete stage 4 and seed ChEES with a user-settable `initial_step_size` (default 1.0, or the Pathfinder-based heuristic in `chees_adaptation()`). **Change (conservative):** keep the search but call it once, vmapped across chains inside a single `jax.jit`.
- **Expected:** −49% (delete) or −47% (vmap) of wall time; −96 syncs.
- **Risk:** ε₀ affects early warmup; if ChEES starts from a bad ε it needs more warmup to recover. This is measurable, not speculative — see §4.
- **Proven by:** §4 A/B; additionally assert that the tuned `chees_parameters["step_size"]` is within a factor of ~2 of the old ε₀ for both example models.

### W2 — Replace the Monte-Carlo mass matrix with the analytic one
- **Location:** `sampler.py:296-336` (`approximate` → `sample(total=chains*draws)` → `jax.vmap(_flatten_i)` at `:332` → `cov` → `jnp.linalg.inv` at `:336`).
- **Evidence:** 13.8 s (9%) to characterise a **single** Gaussian. `blackjax/vi/pathfinder.py:205-207` shows `approximate` returns `jax.tree.map(lambda x: x[max_elbo_idx], pathfinder_result)`, i.e. the best-ELBO approximation only, and `lbfgs.bfgs_sample` (`optimizers/lbfgs.py:412-438`) draws i.i.d. from that one normal. So the empirical covariance converges to the approximation's own covariance, at √(D/total) ≈ 0.3% Monte-Carlo error — and `ridge=1e-6` before `inv` is a genuine numerical hazard on near-degenerate directions (deterministics, badly scaled parameters).
- **Reference:** Zhang et al. 2022, arXiv:2108.03782, Algorithm 4 (PSIS-weighted draws from the quasi-Newton path). The paper's estimator is a **mixture over the path with PSIS weights**, not a single Gaussian.
- **Available already in this install:** `blackjax/vi/multipathfinder.py` (`multi_approximate`, `psis_weights`) and `blackjax/adaptation/pathfinder_adaptation.py:46-83` (`_psis_weighted_mixture_covariance`, documented as returning a dense `(d,d)` positive-definite mixture covariance via the law of total variance), with `imm_estimator: Literal["lbfgs_psis_mixture", "psis_empirical"]` at `pathfinder_adaptation.py:269-271`.
- **Change:** build the inverse mass matrix from `multi_approximate` + `_psis_weighted_mixture_covariance`; keep `pathfinder.sample` only for chain initialisation (≤ `chains` draws, not `chains*draws`).
- **Expected:** −9%, better-conditioned metric, removes the ridge hack.
- **Risk:** `multipathfinder` is newer API; needs a smoke test on both example models.

### W3 — Collapse the identical-start machinery
- **Location:** `_pymc_adapt_diag_start` `sampler.py:167-183`, `_ensure_finite_inits` `sampler.py:185-231`, `_score_chain` `sampler.py:287-291`.
- **Evidence:** 5.4 s (4%) plus a hidden cliff. PyMC's own documentation of this init mode (`pymc/sampling/mcmc.py:1593-1595`): "*`adapt_diag`: … **All chains use the test value (usually the prior mean) as starting point.***" So the 32 "inits" are identical, `tune=500` is inert for point generation, and both the repair loop and the best-chain selection perform 32 copies of one computation. `_find_finite_start` (`sampler.py:59-120`) evaluates `bool(...)` per retry — 1 sync best case, up to ~81 per chain (≈2 592 at 32 chains) when retries fire, in eager Python with no progress output.
- **Change:** request one start point (`model.initial_point()` or `initvals=`), run `_find_finite_start` once, and either keep `jitter+adapt_diag` when genuine per-chain diversity is wanted or drop PyMC's initialiser entirely (one PyTensor `logp_dlogp` compile disappears).
- **Expected:** −3%, one fewer PyTensor compile, removes the retry cliff.

### W4 — Stop retaining data the scan does not need
- **Location:** `utils.py:144` and `utils.py:163` (`warm_info` returned and never read; `sample()` discards `chees_info`), `sampler.py:569-576` (`infos_hist`, `states_hist` carry every leaf).
- **Evidence:** `HMCInfo` is 46 array leaves at D=10 for 8 Schools (`momentum` 10 + 4 scalars + `proposal` as `IntegratorState` = position 10 + momentum 10 + logdensity 1 + logdensity_grad 10, see `mcmc/hmc.py:52-88`, `mcmc/integrators.py:43-53`). Warmup at 10 000 × 32 ≈ 59 MB, sampling ≈ 12 MB at D=10; at D=1000 that is **≈5.9 GB** of warmup information nothing reads and ≈1.2 GB of sampling information of which 5 fields are used. Since `eb8c4c1`, `sample_stats` reads exactly `lp, diverging, acceptance_rate, is_accepted, energy, num_integration_steps`, and `lp` needs only `states_hist.logdensity`.
- **Change:** `ys=()` for the ChEES scan; project the dynamic-HMC scan `ys` to the five diagnostics plus `position`/`logdensity`.
- **Expected:** negligible time at D=10, up to ~7 GB at D=1000 on a 12 GB shared card; smaller scan graphs may also cut compile time.

### W5 — Replace the field-discovery probe with `jax.eval_shape`
- **Location:** `sampler.py:447-449` and `_find_path` `sampler.py:451-488`.
- **Evidence:** one full, **eager** (un-jitted — the `jit` is applied later at `sampler.py:516`) 32-chain transition per `sample()` call, discarded, purely to discover field names that are fixed in 1.7.1. `_find_path` also matches on substrings over the info pytree, walking `momentum` (which contains the model's own variables) first, so a parameter named e.g. `acceptance_rate_like` would silently retarget the progress bar. Same collision class as the bug fixed in `eb8c4c1`.
- **Change:** name the five fields directly (W4 already requires this) and delete `_find_path`.

### W6 — Resolve `pathfinder.approximate` (24 s on a 10-dimensional model)
- **Location:** `sampler.py:296-302`, defaults `maxiter=30, maxcor=10, num_samples=200` (`vi/pathfinder.py:158-166` signature) are untouched by this repo.
- **Unknown:** whether this is XLA compile cost or real L-BFGS work. **Probe first:** call `approximate` twice with identical inputs in one process; if the second call is cheap it is compile-bound (then only W2's restructuring helps); if it repeats, tune `maxiter`/`maxcor`/`num_samples`.
- **Expected:** up to −15%, contingent on the probe.

### Items deliberately not in the plan
`jnp.linalg.inv` → Cholesky/solve (the linear algebra is microseconds at D=10 and dominated by `X.T @ X` at D=1000); the `lax.cond` carry-dtype dance in `_run_dynamic_hmc` (<0.1%); `_swap_cd` transposes; `_silence_pymc_init`; `update_every` tuning on the progress bars (0.3–1.3%). Dead-code deletions (`dict_to_position` `sampler.py:47`, the unreachable `jax.experimental.maps` branch at `sampler.py:613-625`) are hygiene, scheduled with the file they touch, not perf.

---

## 3. Methodology track (separate from performance)

These change sampling behaviour and each needs its own A/B; they are **not** folded into W1-W6.

- **M1 — Adapt the metric during warmup.** BlackJAX's public `chees_adaptation()` exposes `mass_matrix_estimation` and `mass_matrix_window_fraction` (`adaptation/chees_adaptation.py:584-585`), engaging a Welford diagonal once `max(64, 2·√d)` samples are pooled (`:61-72`). The repo's hand-rolled loop calls the private `base()` and never estimates a metric, so a bad Pathfinder metric is never corrected. Note the geometry split this creates today: ChEES whitens by `1/sqrt(imm)` and only accepts a diagonal (`:451`, `:415-449`), while sampling uses the dense matrix.
- **M2 — Consider `meads_adaptation` / `low_rank_adaptation`** (`adaptation/__init__.py`) instead of the bespoke loop: ~120 lines of `utils.py` currently depend on private symbols (`chees_adaptation.base`, `dynamic_hmc.build_kernel`, `halton_sequence`) and had to be patched for 1.7 (commit `4fb50e9`).
- **M3 — Evaluate ChEES-fast and SNAPER** (Sountsov & Hoffman, arXiv:2110.11576, in `ref/`): a cheaper criterion and one with better short-chain efficiency — directly relevant because this sampler's premise is many short chains. Adopt only on measured ESS/gradient-evaluation wins, using the paper's own metric (ChEES vs SNAPER vs ESJD curves, figures 4-6).
- **M4 — Document the intentional divergences from Algorithm 1** rather than "fixing" them: Adam lr 0.05 vs the paper's 0.025, `decay_rate=0.5` vs the paper's fixed 0.9/0.1 moving averages, plus 1.7-only `LOG_UPDATE_CLIP=0.35` (`:22-23`) and `_length_floor` (`:587`). These are BlackJAX defaults, not bugs.

---

## 4. Verification protocol

Applies to every behavioural change (W1, W2, M1-M3). Pure-refactor items (W3-W5) need the existing test plus a green example run.

1. **Metric is ESS per second**, not wall time alone. Report total `ess_bulk` across parameters ÷ wall seconds, plus max r̂ and divergence count.
2. **Fixed seeds**, `random_seed=42`, and **two repeats** per configuration, because GPU reductions are not bitwise reproducible here: identical seeds have already differed by 1 divergence and 0.002 in mean acceptance. A one-divergence delta is noise, not signal.
3. **Acceptance thresholds:** max r̂ ≤ 1.01; total ESS within 10% of baseline; divergences not increased beyond the noise floor.
4. Harness: `~/pf-instrument2.py` (stage timers) is the measuring stick; move it to `tools/bench_stages.py` and commit it with the first performance change so the numbers in this plan stay reproducible.
5. Regression floor: `pixi run python tests/test_inference_data.py` (from `eb8c4c1`) must stay green.
6. Both examples re-executed headless to `~/pf-runs/` (the route used for the notebook run) so plots and timings are captured outside the repo.

---

## 5. Sequencing

| Step | Items | Effort | Gate |
|---|---|---|---|
| 1 | Probe W6 (one run, ~60 s GPU) | 5 min | decides W6 shape |
| 2 | W1 + W3 + W5 (removals, no new machinery) | ~30 min incl. A/B | §4 thresholds |
| 3 | W4 (scan projections) | ~20 min | green tests + one example |
| 4 | W2 (analytic PSIS metric) | ~45 min | §4 thresholds on both examples |
| 5 | M1 (metric adaptation during warmup) | ~1 h, behaviour change | §4 + paper comparison |
| 6 | M2/M3 (meads / SNAPER evaluation) | separate spike | only if M1 underperforms |

Expected end state after steps 2-4: wall time dominated by actual MCMC — roughly 40 s rather than 80 s for the 8 Schools benchmark, with the mass matrix computed analytically instead of from 128 000 draws.

---

## 6. References

**Papers (local copies in `ref/`, citation details in `ref/README.md`)**
1. Hoffman, Radul, Sountsov. *An Adaptive-MCMC Scheme for Setting Trajectory Lengths in HMC.* AISTATS 2021, PMLR 130:3907-3915. §2.2 (ChEES criterion — **Ch**ange in the **E**stimator of the **E**xpected **S**quare), Algorithm 1 (ε₀ input, Halton jitter `tₙ = hₙTₙ₋₁`, `⌈t/ε⌉` steps, harmonic-mean acceptance targeted at 0.651, Adam on `log T`, ε/T frozen to 0.9/0.1 moving averages at the end of warmup).
2. Zhang, Carpenter, Hoffman, Marcus, Sobie, Yao. *Pathfinder: Parallel quasi-Newton variational inference.* arXiv:2108.03782. Algorithm 4 (PSIS-weighted draws over the quasi-Newton path) — basis of W2 and M1.
3. Sountsov, Hoffman. *Focusing on Difficult Directions for Learning HMC Trajectory Lengths.* arXiv:2110.11576. ChEES-fast, SNAPER, and joint step-size/metric/trajectory adaptation — basis of M3.

**Installed-source references (`.pixi/envs/default/lib/python3.13/site-packages/`)**
- `blackjax/adaptation/chees_adaptation.py:21` `OPTIMAL_TARGET_ACCEPTANCE_RATE = 0.651`; `:22-23` `LOG_UPDATE_CLIP`; `:61-83` Welford engagement gate; `:412-451` diagonal whitening in the criterion; `:576-587` public `chees_adaptation()` signature including `mass_matrix_estimation`.
- `blackjax/vi/pathfinder.py:158-166` `approximate` defaults; `:205-207` best-ELBO selection; `:211-241` `sample` returning `(samples, logq)`.
- `blackjax/optimizers/lbfgs.py:412-438` `bfgs_sample` (single-Gaussian draws).
- `blackjax/adaptation/pathfinder_adaptation.py:46-83, 264-272` `_psis_weighted_mixture_covariance`, `imm_estimator`.
- `blackjax/mcmc/hmc.py:52-88` `HMCInfo`/`HMCState` fields; `blackjax/mcmc/integrators.py:43-53` `IntegratorState`; `blackjax/mcmc/dynamic_hmc.py:37-121` `DynamicHMCState`, kernel signature with `integration_steps_params`.
- `blackjax/adaptation/__init__.py:1-17` available adaptation families (`meads_adaptation`, `low_rank_adaptation`).
- `pymc/sampling/mcmc.py:1593-1597` `adapt_diag` gives every chain the same test-value start; `pymc/sampling/jax.py:278-286` PyMC's own `lp`/`diverging` sample-stats convention.

**Repo references**
- Commits `4fb50e9`, `eb8c4c1`, `ba66ccc`; test `tests/test_inference_data.py`; benchmarks `~/pf-instrument.py` (repeat-call proof) and `~/pf-instrument2.py` (stage split); executed example outputs `~/pf-runs/`.
- Dependency pins and their cause: `pixi.toml` (`numpy>=2.0,<2.5`, `numba==0.66.0`), upstream context pymc-devs/pytensor#2351.
