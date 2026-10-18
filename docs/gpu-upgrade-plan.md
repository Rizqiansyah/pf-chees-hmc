# GPU sampler upgrade plan (v2)

Status: draft for approval. Supersedes `docs/optimisation-plan.md` for direction; that document
still holds the measured stage cost model and is referenced below.
Nothing in this plan is implemented. No benchmark in here has been run except §2.
Evidence discipline: every external claim carries an ID and a tier — **[F]** full text read this
session, **[A]** abstract only, **[M]** metadata only, **[C]** installed source with file:line,
**[U]** unverified, treat as a hypothesis. Deep verification is in progress for the numbered
claims (`/home/akbar/pf-research/VERIFICATION.md`).

---

## 1. Objective, and the claim we can actually defend

Goal: highest effective sample size per wall-clock second on this box's GPU (RTX A2000), for PyMC
models, with a cross-check against CPU NUTS.

Two motivations from the project owner, now ranked by evidence:

- **(a) GPU as an alternative when the CPU is contended or weak — supportable.** On a shared box
  with 32 cores available and idle, CPU NUTS wins small models outright (§2). The defensible claim
  is availability and predictability of throughput, not universal speed.
- **(b) GPU as an overall performance win over CPU NUTS — weaker than it looked.** Table 1 of
  arXiv:2411.04260 is now verified line by line **[F-verified]**: 2.4 s / 24,041 / 10,017 (P100
  HMC), 4.3 s / 16,005 / 3,731 (P100 NUTS), 27.0 s / 24,131 / 894 (28-core Xeon 8273CL HMC),
  54.4 s / 16,048 / 295 (CPU NUTS), 64 chains, "tuning time is not reflected in the benchmark".
  Three things shrink it. ESS is on the global scale parameter `tau` only. The same kernel on GPU
  versus CPU is 11–12x, so most of the 34x is device throughput, not method. And the authors warn
  the ratio worsens below 64 chains — we run 32 on an A200. `d = 51` is from arXiv:2110.11576; this
  paper never states the dimension.

Consequence: the project's success criterion becomes **the crossover** — the region of (model cost
per gradient, chain count, CPU availability) where GPU ESS/s overtakes CPU NUTS — plus being
competitive above it. §6 item W0 measures it.

Restated objective: **GPU HMC-family vs GPU NUTS at equal adaptation budget, including warmup.**
The CPU-versus-GPU part of the story is device throughput.

**Not established, do not cite as fact:** Tran et al. 2018 (arXiv:1811.02091, ~100x over Stan,
~37x over PyMC3). The verification thread had no network and no local capture, so it could not be
checked at all. Not refuted — unverifiable.

---

## 2. Measured baseline on this machine (done)

`/home/akbar/pf-runs/baseline.jsonl`, harness `/home/akbar/pf-baseline.py`, task `bbb022a88`.
32-core box, RTX A2000. CPU arm: PyMC/PyTensor NUTS, 4 chains, 1000 tune + 1000 draws,
`target_accept=0.8`, `OMP_NUM_THREADS=1`. GPU arm: current `sample_pf_chees_hmc`, 32 chains,
2000 draws, 10 000 ChEES warmup.

| Model | Backend | Wall | max R̂ | ESS/s (total) | ESS/s (worst var) | div |
|---|---|---|---|---|---|---|
| 8 Schools, d=10 | CPU NUTS | 2.33 s | 1.0022 | **18,207** | 1,137 | 9 |
| 8 Schools | GPU dhmc+ChEES | 80.0 s cold / 52.7 s warm | 1.0014 | 5,852 / 8,828 | 437 / 666 | 1 |
| GEV Port Pirie, d=3 | CPU NUTS | 3.49 s | 1.0018 | **2,686** | 590 | 42 |
| GEV | GPU dhmc+ChEES | 122.7 s cold / 99.5 s warm | 1.0009 | 1,558 / 1,882 | 314 / 374 | 33 |

Findings:
1. Our sampler loses to CPU NUTS by 2–3x in ESS/s on both shipped example models.
2. ~27 s of the cold GPU wall time is JAX compilation (80.0 s → 52.7 s on the second call).
3. Deleting every identified overhead reaches CPU parity on 8 Schools at best. Therefore
   **overhead removal is necessary but not sufficient**; the lever is preconditioning (§5-D1).
4. GPU runs had fewer divergences and slightly lower max R̂ than CPU NUTS at these settings.

---

## 3. Stage cost model (unchanged, still the basis for W-items)

Instrumented 8 Schools, 32x2000, 10k warmup, seed 42 (see `docs/optimisation-plan.md` §1).
Shares, not absolute times, are the decision quantity: step-size tuner 49%, `pathfinder.approximate`
15%, ChEES warmup 14%, Dynamic HMC sampling 8%, Pathfinder 128k draws 5%, covariance+`inv` 4%,
finite-start repair 3%, `pm.init_nuts` 1%, InferenceData 1%.

---

## 4. Evidence base actually used

| ID | Claim used | Tier |
|---|---|---|
| arXiv:2411.04260 T1 | GPU HMC 10,017 ESS/s vs CPU NUTS 295; NUTS loses to HMC on GPU; warp divergence explanation | [F] |
| PMLR 130:3907 (ChEES) | HMC beats TFP NUTS 1.32–3.00x in ESS/gradient, 5/7 targets top adaptive; V100 throughput linear to >=50 chains, roofline ~75–100; same jittered trajectory shared across chains | [F] |
| arXiv:2110.11576 (SNAPER/ChEESR) | ChEES-fast is just eta_tau=0.05; ChEESR = ESJD/tau; SNAPER = ESJD_(z^T w)^2/tau with minibatch Oja; ChEES overshoots (no cost term) and undershoots on pathological spectra | [F] |
| PMLR 151:7799-7813 (MEADS/ECA) | Alg 3 closed-form: eps=min{1, 0.5/sqrt(max_eig ḡ)}, gamma=max{1/(t·eps), 1/sqrt(max_eig θ̄)}, alpha=1−exp(−2·eps·gamma), delta=alpha/2; valid cross-chain adaptation needs a disjoint chain subset, priced at "at most a 25% slowdown" at K=4; Table 2 is **gradient counts, no wall-clock table**; **hardware was TPU, not GPU**; **NUTS beats ChEES on Grads/ESS for Banana (156 vs 448) and Stochastic Volatility (146 vs 180)** | [F-verified] |
| arXiv:1910.11141 (Radul) | NUTS padding penalty is a batching artifact: longest trajectory ~4x mean => ~25% utilization; CPU-batched NUTS matched Stan | [F] |
| arXiv:2503.17405 (Dance FSM) | sync-barrier cost C0 ∝ sum_i max_j N_ij; ceiling R(m)=E[max]/E[N1]=O(ln m), ~2 at zero skew, ~100 at skew 10; net loss when log-density cheap (0.8x, 0.91x); "hyperparameters pre-tuned over 400 warm-up steps" verbatim. **Never stated: whether warmup is inside the ESS/s denominator.** Table-1 column-to-value pairing is inferred from column-major extraction (this thread confirmed it separately against the published PMLR PDF) | [F-verified, pairing inferred] |
| arXiv:2603.18845 (Seyboldt) | low-rank Fisher preconditioning: ~4x in ESS/gradient **and** ESS/second (low-rank), ~0.75x grads and ~1.3x wall (diagonal), all **median**; rank rule `U_c = {U_i : lambda_i <= 1/c or lambda_i >= c}`, c=2, gamma=1e-5; "114 models remained after this filtering" is **post-filter**. **The paper makes no hardware statement — "on CPU" was our insertion and is unsupported.** | [F-verified, hardware claim struck] |
| arXiv:2210.10644 / PMLR 206:3664 (TESS) | flow transport + elliptical slice sampling; strongest published ESS/s on funnel targets (1116 and 2978 vs 480.8 and 64.85 for MEADS) | [M/F via citing paper] |
| arXiv:2311.02726 (Margossian & Gelman) | cross-chain adaptation reduces gradients-per-chain to a target bias (d=501, 2/4/8 chains); many-short-chains: variance buys with chains, bias cannot; ESS 100 for 0.1 sd MCSE | [F] |
| arXiv:2110.13017 (nested R̂) | plain R̂ invalid in many-short-chains regime; nested R̂ measures nonstationarity directly | [F] |
| BlackJAX 1.7.1 wheel | public API surface, GHMC contract, `_psis_weighted_mixture_covariance`, `staged_adaptation` limits, GIST cost, TESS/FSM absent | [C-verified] |
| arXiv:2108.03782 (Pathfinder) | Algorithm 5 is PS-IR: sample psi "**with replacement**, probabilities proportional to w_j", for use "as an initialization algorithm for MCMC". **The phrase "mass matrix" does not occur in the paper.** The PSIS-mixture covariance is a BlackJAX extension, not a Pathfinder result | [F-verified] |
| arXiv:2601.16696 (LAPS, Robnik & Seljak, v1 23 Jan 2026, **preprint, 0 citations, no venue**) | beats ChEES and MEADS on all six Inference-Gym targets by 2.3–22.9x, but the metric is **gradient calls per chain to `b²_max < 0.01`** on an A100 — **no ESS is measured anywhere and no seconds are measured**, wall clock being explicitly "derived" from a gradient-count proxy. Margins are large only on synthetic targets (Banana 15.5x, Gaussian 16.7x) and ~2–3x on the realistic hierarchical models. Table 2 shows LAPS uses **6.8–28x more total gradient calls** than sequential MCLMC — it wins only if >=M parallel lanes are free. Table 3 gives a counterexample: adjusted-only MAMS beats LAPS on German Credit (270 vs 300). Ablation: the same two-phase scheme with underdamped Langevin needs 650 calls on IRT, **worse than ChEES's 537** — "microcanonical dynamics is important for the success of our scheme" | [F-verified] |
| arXiv:2503.01707 (MAMS) | acceptance-free vs NUTS wall clock does exist: 1.5 s (MAMS) vs 4.8 s (NUTS) on a **single CPU**, 1000 draws, disclaimed by its own authors. This kills the blanket "no such comparison exists" | [F-verified] |
| arXiv:2404.15253 (GIST) | "Gibbs self-tuning" ✓ and "NUTS only slightly outperforms the GIST sampler overall" ✓. **"no second derivatives" is not a paper statement** — "derivative" occurs 0 times; it is an inference from the oracle signature | [F-verified / inference flagged] |
| SNUTS arXiv:2603.02437 | "a sparse Q beating the inverse of the covariance estimated with Pathfinder" — **NOT ESTABLISHED**, no local copy and no network in the verification thread | [U] |

---

## 5. Design decisions

### D1 — Preconditioning is the primary lever (adopt)
Evidence: the only published ~4x in ESS/second that is a method change rather than a hardware
change is low-rank Fisher preconditioning (arXiv:2603.18845) **[F]**, and BlackJAX already ships
Algorithm 1 of it as `blackjax.window_adaptation_low_rank` — `M⁻¹ = diag(σ)(I + U(Λ−I)Uᵀ)diag(σ)`,
Fisher-divergence minimizing, eigenvalue masking `[1/cutoff, cutoff] → 1`, `gamma=1e-5`, `max_rank=10`,
state re-initialised at μ* before sampling **[C]**.

Constraint found in source: it is **single-chain** (`low_rank_adaptation.py:458-473`, driven through
`staged_adaptation`). Multi-chain low-rank selection exists only as `staged_adaptation(metric="auto")`,
which cannot drive our kernel (§D5). So adopting it means either accepting single-chain warmup with a
shared metric, or driving the metric through the multi-chain path we keep.

### D2 — Delete, do not optimise, the auxiliary stages (adopt)
W1 ε tuner (49%), W3 identical-start repair (3%), W5 field-discovery probe, and the 128k-draw
Monte-Carlo covariance (9%) are all removable, and three of them have no counterpart in the ChEES
algorithm. The covariance is replaced by a library call (§D3).

### D3 — Pathfinder metric: keep the library route, downgrade the claim (adopt with caveat)
`_psis_weighted_mixture_covariance` (pathfinder_adaptation.py:46-137) is analytic — softmax of
per-path logsumexp weights, within-path weighted covariance plus between-path outer products,
zero Monte-Carlo draws, no d×d `inv` **[C]**. That strictly dominates our current
`cov + 1e-6·I → inv` estimator, which approximates the covariance of a *single* Gaussian
(`pathfinder.sample` → `bfgs_sample` draws i.i.d. from one Gaussian, lbfgs.py:412-439 **[C]**).

Caveat, now verified against the paper: **PSIS as a mixture covariance has no published
derivation.** Pathfinder's Algorithm 5 is PS-IR — sampling **with replacement** proportional to
`w_j`, for use "as an initialization algorithm for MCMC" — and the phrase "mass matrix" never
appears in the paper **[F-verified]**. Counter-evidence, now read in context: LAPS says ensemble
Pathfinder (Algorithm 2 of Zhang et al.) "failed to converge for most of our benchmark problems",
but gives numbers for only two of six targets — Banana reaching `b²_max = 0.02` in 67 gradient
evaluations and German Credit plateauing at **21.5**, roughly 2,150x over its threshold. "Most" is
unquantified **[F-verified]**; the SNUTS quote remains unverified **[U]**. Also `pathfinder_adaptation.py:372-375` labels the estimator "**Paper-canonical
(Zhang et al. 2022)**" and that badge is **unsupported by the paper** — worth an upstream issue.
PD of the mixture is a docstring claim with no code guarantee **[C]**. So we adopt it as an
engineering default, not as method. (`ref/README.md` also mis-numbers this as Algorithm 4; fixed.)

### D4 — Diagnostics and stop rule (adopt)
Gate on: no divergences, `E-BFMI ≥ 0.3`, rank-normalized bulk-ESS ≥ 400 and tail-ESS ≥ 400,
`R̂ < 1.01` for long chains or **nested R̂** (`arviz_stats.rhat_nested`) in the many-short-chains
regime, `k̂ ≤ min(1 − 1/log₁₀S, 0.7)` wherever PSIS reweighting is used, and one fp64 rerun as a
numerics gate (single precision has a 24-bit mantissa; |logp| > 2²⁴ corrupts accept/reject) **[F/C]**.
Early stopping is implementable in-library: `blackjax.eca.add_splitR` and
`while_with_info(..., while_cond)` **[C]**.

### D5 — Explicitly rejected, with reasons (do not re-litigate without new evidence)
1. **FSM de-synchronization for our kernel.** Our kernel has zero cross-chain loop-count skew by
   design: ChEES uses one shared jittered trajectory length so "each chain does the same amount of
   work per iteration" (PMLR 130:3907 §2.3) **[F]**. Dance's ceiling `R(m) = E[max N]/E[N₁]` is
   then 1, and the paper warns the restructuring is a net loss when the log-density is cheap
   (0.8x, 0.91x) **[F]**. Scope this rejection: **FSM is not rejected for a GPU NUTS arm** — it
   recovers roughly an order of magnitude there, which is exactly why a GPU-NUTS baseline must not
   be written off as naively vectorized NUTS.
2. **`blackjax.staged_adaptation` as our warmup.** Kernel contract excludes `dynamic_hmc`, `ghmc`
   and `mclmc`; multi-chain only with `metric="auto"`; acceptance pooled by the **arithmetic** mean,
   not configurable; `max_grad_budget=None` + auto raises; no tests in the wheel **[C]**.
3. **MCLMC/acceptance-free as a drop-in.** Four of our six `sample_stats` keys have no source field
   and there is no synthesis path, so it is a contract break, not a kernel swap **[C]**. `mclmc`
   also cannot consume a dense (d,d) metric (diagonal or `LowRankInverseMassMatrix` only) **[C]**.
4. **TESS now.** Highest published ESS/s on funnel geometry, but no transport/flow component exists
   in BlackJAX 1.7.1 (`elliptical_slice` only, no flow preconditioning; TESS absent by
   enumeration) **[C]**. Revisit as option B if W0 shows our models are funnel-limited.
5. **"ChEES-fast" as a change.** It is the learning rate we already run (0.05) **[F]**.

### D6 — Open fork, not yet decided (W6 decides it)
Keep `dynamic_hmc` + ChEES, or switch to GHMC + MEADS/ECA. GHMC facts **[C]**: always one
velocity-Verlet step, exactly one `value_and_grad` per call, returns `HMCInfo` so our contract
survives, dense argument is the *inverse* mass matrix (1-D arrays are squared — a trap), metric is
rebuilt every call unless a `Metric` object is pre-built, and `alpha` outside [0,1] yields silent
NaN with no validation. No turnkey post-adaptation loop exists; `staged_adaptation.py:28` explicitly
excludes GHMC. Cost: ECA's disjoint-subset requirement is priced at "at most a 25% slowdown" at the default
`num_folds=4` **[F-verified]**, and MEADS's own numbers are TPU gradient counts with no wall-clock
table **[F-verified]**.

**Correction to the earlier recommendation, important.** "MEADS + lockstep fixed-step HMC + Fisher
preconditioning" is internally incoherent as stated: in this wheel MEADS emits **GHMC** parameters
(`meads_adaptation.py:657-666`, `:777-783`), so pairing MEADS with plain fixed-step HMC throws away
`alpha` and `delta` — that is, the momentum persistence that is half of GHMC. Either MEADS comes
with GHMC, or plain HMC needs a different step-size and metric rule.

### D7 — LAPS: evaluated and NOT adopted (closed 2026-10-18)
Full read at `/home/akbar/pf-research/laps-read.md` (474 lines). Decision: **do not adopt, and do
not treat `blackjax.adaptation.laps` as an API.** Reasons, all verified:

1. **Wrong regime.** LAPS is cold-start from the prior, one sample per chain, and validates only
   `M > 128–256` while stating lower M "cannot work". We run `chains=32, draws=2000`, a warm
   long-chain regime where post-adaptation ESS/s dominates.
2. **Wrong metric.** Everything is gradient calls per chain to a bias threshold. No ESS, no seconds.
3. **Wrong kernel, and the kernel is load-bearing.** Its own Langevin ablation loses to ChEES, so
   the win is late adjustment **+ microcanonical dynamics**, which we are not adopting here.
4. **The implementation is not usable.** Unexported (`adaptation/__init__` and `blackjax.__all__`
   both omit it), absent from shipped docs, no BibTeX in the docstrings, one test (2-D Gaussian, 4
   chains, `early_stop=False`) that never exercises the paper's adaptive phase switch. `microcanonical=False`
   raises. `run_eca` calls `.item()` and `arr[:int(counter)]`, so it cannot be traced — it must run
   eagerly outside jit/vmap/scan, and cross-chain adaptation needs a `shard_map` `Mesh` with an axis
   literally named `"chains"`. Wheel integrity is fine (113 files hashed against RECORD, 0 mismatches),
   so this is upstream state, not local patching.
5. **Defaults contradict the paper.** `C=0.1` in code vs 0.025 in the text; `alpha=1.9` vs 2; the
   ε update reads `C·bias**(3/8)` vs Algorithm 1's `F(C·D̃)` (exponent 3/2) with no reconciliation
   available **[NOT ESTABLISHED]**; a `clip(eps_factor, 0.3, 3.0)` safeguard with no paper
   counterpart; and the default bias observable is `x`, not the paper's `x**2`.

**What we keep — one transferable hypothesis, as experiment arm W6a:** an unadjusted burn-in whose
**ensemble-variance diagonal metric** (`y_i = x_i / sqrt(Var[x_i])`, applied once at a phase
boundary and then frozen) plus an **EEVPD-targeted step size** (`eps <- eps·(EEVPD_wanted/EEVPD)^(1/6)`,
from `EEVPD ∝ eps^6`) could seed a warm HMC run more cheaply than our 49%-of-runtime ε tuner. That
is the only kernel-agnostic object in the whole method.

Also kept as a design idea: LAPS **freezes all hyperparameters before sampling explicitly "to
eliminate the bias caused by ECA"**, and prefers **bisection** on acceptance over dual averaging
because with many chains the acceptance estimate is low-noise. Both are cheap and apply to whatever
kernel survives W6.

---

## 6. Work items

| # | Item | Evidence | Gate |
|---|---|---|---|
| W0 | **Crossover ladder** (see §7) — decides whether §1(b) survives at all | none exists; nothing in the literature gives an ESS/s-vs-chain-count curve | report the crossover, or report that none exists on this box |
| W1 | Delete per-chain ε tuner; replace with a documented initialiser (Pathfinder-derived, MEADS-style closed form, or 1.0) | 49% of runtime; H&G Alg 4 is an initializer that "works for any ε₁"; NumPyro ships it off | ESS/s gain ≥ 30% with r̂, tail-ESS, BFMI and divergence gates held |
| W1b (done) | ~~Read LAPS end to end and audit `adaptation/laps.py`~~ — done, verdict in D7: not adopted | LAPS metric is gradient calls, not ESS or seconds [F-verified] | closed |
| W6a | **LAPS preconditioner arm, public API only.** Gate first: wall-clock per gradient call for `dynamic_hmc` vs `adjusted_mclmc` at our d and 32 chains (~2 min). Then arms (a) current ChEES+dhmc, (b) dhmc with metric = ensemble `Var[x]` and step size from ~200 unadjusted `blackjax.mclmc` steps (`L=2·sqrt(ΣVar)`, `C=0.025`, switch on `delta[x_i^2] < 0.01`), (c) `adjusted_mclmc` driven by (b) | D7 ablation says the kernel, not the metric, carries LAPS's win; if MAMS's per-call cost is worse on an A2000 the paper's derived advantage cannot transfer and (b)/(c) are decided before they run | ~2 GPU-min/arm; budget-matched ESS/s at 90 s, scored over the whole run and post-t=30 s |
| W2 | Replace hand-rolled 128k-draw covariance with `pathfinder_adaptation`/`multipathfinder` analytic PSIS mixture; keep `pareto_k` in sample_stats | D3 | −9% runtime, Pareto k reported, no new divergences |
| W3 | Collapse the identical-start machinery (PyMC `adapt_diag` returns one point; `tune=500` is inert) | 3% + a 2592-sync retry cliff | −3%, and the retry path deleted |
| W4 | Drop unused scan outputs (`warm_info`, `infos_hist` leaves, `logdensity_grad`) | ~59 MB at D=10, ~5.9 GB at D=1000 | memory report before/after; behaviour identical |
| W5 | Delete the field-discovery probe and `_find_path`; name fields explicitly | one wasted real 32-chain transition + a silent-wrong-UI collision class | probe gone; test still green |
| W6 | Kernel-family bake-off: dhmc+ChEES vs GHMC+MEADS, at matched ESS/s including adaptation cost | §4 rows MEADS vs ChEES disagree by target | whichever wins on our ladder; document the other as an option |
| W7 | Warmup-length sweep (1k/2k/5k/10k) with nested R̂ in-loop early stopping | Margossian & Gelman: cross-chain adaptation lowers gradients-to-bias with more chains; SNAPER: a non-zero optimum exists | find the knee; publish the curve |
| W8 | Metric bake-off at fixed kernel: current dense vs analytic PSIS vs Welford diagonal vs low-rank Fisher | D1 is the only ~4x published method gain | pick by ESS/s, not by condition number |

Order: W0 first (it can invalidate the premise), then W1+W3+W5 (mechanical, no statistics change),
then W2, then W7, then W8, then W6 last because it is the most expensive to reverse.

Each item ships with: the existing `tests/test_inference_data.py` green, a golden-trace fixture
(8 Schools, 4 chains x 200 draws, seed 42, committed netCDF plus expected mean/sd/r̂), a posterior
mean cross-check against `pm.sample` NUTS within Monte-Carlo error, and a benchmark row in the
protocol below. Two independent reviews per behaviour-changing item, looping until clear.

---

## 7. Benchmark protocol v2 (replaces the §4 protocol in the old plan)

Required by the evidence, and none of my §2 numbers satisfy all of it yet:
- report `ESS/s = (ESS/useful_grads) × (useful_grads/s)` **and** the padding ratio useful/issued
- time compile, pathfinder, warmup, sampling separately **and** summed; warmup inside the denominator
- ≥5 seeds; median plus 10th/90th percentiles; never best-of-N; device not shared during a run
- `block_until_ready()` before stopping timers; identical dtype; **pin the pytree representation**
  (BlackJAX docs report min ESS/s 18.9 vs 7.3 for flat vs dict positions) **[C]**
- baselines: CPU NUTS with all physical cores and `adapt_delta` swept, plus **its own ESS/s-vs-core-count
  curve**; a fixed-L grid oracle as the achievable ceiling; a Fisher/low-rank arm; a GPU NUTS arm so
  the method claim is separated from the hardware claim
- sweep chain count (8/32/64/128) and report the curve — no published curve exists, so this is ours
- validity gates per D4, plus one fp64 rerun
- 8 Schools is a funnel; report it separately from the non-funnel models rather than averaging

Ladder (W0): 8 Schools (funnel, d=10, cheap gradient); ill-conditioned Gaussian d=50 and d=500;
logistic regression 1000x25 and a heavier-data variant to raise cost-per-gradient; the GEV model.
Each at CPU cores=1 and cores=4 versus GPU chains=32, plus the chain sweep on the best point.

---

## 8. Deferred, needs an owner decision
- Repo rename / new branch / widened scope, and whether the current kernel family ships as one
  option among several. Not a code problem; it is history and remote policy. Four commits are still
  unpushed (`4fb50e9`, `eb8c4c1`, `ba66ccc`, `4f249ca`) pending the box's SSH key.
- Whether the ~11 MB of `ref/*.pdf` are committed or gitignored. Asked repeatedly, unanswered.

## 9. Verification outcome
`/home/akbar/pf-research/VERIFICATION.md` re-derived 15 load-bearing claims. Ten verdicts changed
or gained corrections. The material ones are folded into §§1, 4 and 5 above: the 34x is mostly
device throughput; `d = 51` was misattributed; Seyboldt states no hardware; MEADS's paper is TPU
and gradient-counts only; Pathfinder's Algorithm 5 is resampling with replacement and never
mentions a mass matrix; the "no wall-clock comparison for acceptance-free samplers" claim is false
(MAMS gives 1.5 s vs 4.8 s on one CPU); and "MEADS + fixed-step HMC" is incoherent because MEADS
drives GHMC.

Method caveat on the verification itself: that thread had **no network tools** — `read`, `write`,
`edit`, `grep`, `find`, `ls` only — so every [F-verified] tag means "checked against a full-text
capture already on this machine under `/tmp`, not re-fetched from the publisher." Those captures
are this session's own downloads, which is weaker than independent re-retrieval. `arXiv:1811.02091`
and `arXiv:2603.02437` could not be checked at all and are marked [U].
