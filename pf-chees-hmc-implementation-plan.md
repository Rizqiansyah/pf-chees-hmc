# pf-chees-hmc: independent review and implementation plan

**Author:** independent review session, 2026-10-19, at commit `ecd9ba1`, **on the real box** (RTX A2000 6 GB,
JAX 0.11.2, BlackJAX 1.7.1, PyMC 5.28.5). Reviews `pf-chees-hmc-review-handoff.md` ("the handoff")
and `pf-chees-hmc-literature-plan.md` ("the lit plan"), which were written without hardware.

**Audience:** the lead workflow and its tasks. **Read Part A (run instructions) first, then
Part B (technical work order).** Nothing in the repo was changed by this review. Probe scripts are in
`~/pf-research/independent-review-2026-10-19/`. Background only, superseded where they differ:
`pf-chees-hmc-review-handoff.md` and `pf-chees-hmc-literature-plan.md`.

**Evidence tags**
- **[SRC]** read in code, `file:line` given. Re-read before acting (skill §3).
- **[MEAS]** measured in this review on this box. The probe script is named, so re-run it to reproduce.
- **[REPO]** a number from the repo's own `docs/`, not re-measured.
- **[HYP]** a hypothesis, which needs the named measurement.

**Owner goal.** Maximise GPU throughput in the **R1 regime**: about 1,000 chains × about 10
draws each. Keep divergences as a diagnostic. Pathfinder stays as the initialiser. Funnels are out
of scope. **R2** (32–128 chains × 1,000+ draws) is secondary.

---

# Part A — Run instructions (read first)

You orchestrate one autonomous run. Part A says how to run it. Part B (§0–§8) says what to build
and why.

### A0. Scope of this run
- Do **WP0 → WP6, WP8, WP9** (Part B §4). WP7 and WP6.4 are optional, only once everything else is
  done.
- **WP10 is out of scope.** Stop after WP9's report so the results can be reviewed first.
- Use the 4-model ladder: 8 Schools, GEV, logistic 1000×25, radon NC. The owner's own model is
  deferred because it costs hours of CPU sampling; leave a slot for it in the harness. Radon
  centred is never a gate.

### A1. Owner decisions (resolved 2026-10-19)
1. **Checkouts: approved.** Implementation tasks each work in their own checkout and branch (A3).
2. **Vendoring BlackJAX ChEES `base()`: approved.**
3. **Owner's model: later.** Do not block on it.
4. **`arviz-stats`: approved.** Nested R̂ is `arviz_stats.rhat_nested`. General owner rule: do not
   reimplement what a maintained library provides (A6).
5. **XLA command-buffer flag:** document it. Set it in `pixi.toml` `[activation.env]` if WP6.1
   accepts it.

### A2. Superseded beliefs: do not act on these, even if memory, FINDINGS or docs recall them
- "Short chains do not substitute for long ones", "more warmup does not repair what more chains
  broke", "a Pathfinder start needs draws per chain to burn off". Refuted by §2b and §2e.
- "There is no smaller warmup default; 2,000 steps are needed". Measured under an inverted and
  permuted metric.
- "LAPS is the wrong regime". The owner's regime is now R1.
- Every radon and GEV metric verdict (E4), and the eigen-spectra reading. Confounded by the WP1
  bugs.
- "c/step grows 3.17 → 13.69 ms with chains". The number includes O(C) init and host costs.
- `~/pf-research/QUEUE.md`. Obsolete: this file is the queue.
- "ESS/s is the objective". That holds for R2 only. The R1 objective is gradients per chain and wall
  time to pass G-R1 (§6).

WP9 re-measures each item; log it in LEDGER when it does.

### A3. Branches, checkouts, environment (recipe verified on this box 2026-10-19)
- **Integration branch.** In the main checkout run `git switch -c pf-r1` (from `ecd9ba1`). Never
  commit to `main`. Never push.
- **First commit on `pf-r1` (WP0).** It contains:
  - this file and the two handoff docs, in `docs/review-2026-10/`;
  - the probes from `~/pf-research/independent-review-2026-10-19/`, in `tools/probes/`.

  **Checkouts only see committed files**; untracked files in the main checkout are invisible to
  tasks. Do not commit `ref/*.pdf` (skill §7).
- **Per WP:** `git worktree add ~/pf/<wp> -b wp/<wp> pf-r1`.
  - Tasks edit and commit only on `wp/<wp>`.
  - The lead merges `wp/<wp>` into `pf-r1` in the main checkout after the gates and both
    reviews pass, then runs `git worktree remove ~/pf/<wp>`.
  - Branch a dependent WP from `pf-r1` *after* its dependency merges.
- **One shared environment**: the main checkout's `.pixi`. Never run `pixi install` in a checkout.
  Run checkout code from the checkout directory with:
  ```
  MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; WT=~/pf/<wp>; cd "$WT"
  pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python "$WT/tools/<script>.py"
  pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider
  ```
  Verified: this imports the checkout's `pfcheeshmc`, applies the activation env
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false`), and pytest passes. Every harness must print
  `pfcheeshmc.__file__` so that a run against the wrong tree is detectable.
- **Environment changes** (`pixi add`, `[activation.env]`) are made only by the lead, in the
  main checkout, on `pf-r1`. Commit them with `pixi.lock`, then re-run `pixi run test`.
  - WP0: `pixi add --pypi "arviz-stats[xarray]>=1.3.3,<1.4"`. PyMC pins `arviz<1.0`, so do **not**
    upgrade `arviz`; `arviz-stats` is a separate package. Its requirements are met here: numpy
    ≥ 2 (2.4.6), scipy ≥ 1.13 (1.18.1), xarray ≥ 2024.11 (2026.9.0), Python ≥ 3.12.
  - Record any other package the solver moves (diff `pixi.lock`).
- **Legacy arm:** `git worktree add --detach ~/pf/legacy ecd9ba1`. Run the `pf-r1` harness with
  `PYTHONPATH=~/pf/legacy`.
- **Box traps:**
  - The repo path contains a space, so always quote it.
  - Models with a Dot need `PYTENSOR_FLAGS="blas__ldflags=-llapack -lblas -lcblas -lm"`; the
    `bench_ladder` import sets it.
  - Always go through `pixi run`; a bare env python misses the activation env.

### A4. Tasks
- Caps (hard): **2 × fast-tier and 3 × steady-tier at once.** Total launched is unlimited if run
  serially.
- **fast-tier is for moderately difficult work:**
  - implementing WP1, WP2, WP3, WP4, WP5 and WP6.2;
  - every code-correctness review;
  - every statistical-validity review.
- **steady-tier is for easy, bounded work only:**
  - running a specified matrix with a given script and arguments, and collecting the JSONL;
  - generating references (WP0.2);
  - single WP8 hygiene items;
  - a doc retraction from a given evidence row;
  - line-number verification and enumerations.
- **The lead does:** environment changes, merges, gate decisions, GPU-lock scheduling, and
  STATUS/LEDGER/REPORT.
- **Every brief contains:**
  - the goal;
  - the checkout path and branch;
  - the files allowed to change;
  - the Part B section verbatim;
  - the tests that must fail before and pass after;
  - the A3 commands;
  - the prohibitions: no `main`, no push, no environment changes, no GPU outside the lock, and no
    edits outside its own checkout;
  - the deliverable shape (A8).
- Review tasks are read-only. They must invite rebuttal of Part B's framing and answer each point
  `CONFIRMED / REFUTED / NOT ESTABLISHED` with `file:line` or a row id.
- Every merge needs 2 reviews (code and stats, both fast-tier), run between implementation tasks.
  Save them as `results/reviews/<WP>-code.md` and `<WP>-stats.md`.

### A5. GPU discipline
- One shared 6 GB card. **Every GPU process** runs as `flock /tmp/pf-gpu.lock <command>`, including
  smoke tests longer than 30 s.
- Before each timed batch, run `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
  - If a foreign process holds the GPU, wait and retry for up to 1 h before timing.
  - Record any foreign process in the row's `gpu_other_procs`.
  - Correctness-only runs may proceed anyway.
- CPU-heavy jobs (reference NUTS) must not overlap timed GPU runs. Run them in WP0, or tag the
  overlapping rows `cpu_contended=true`.
- Run every cell with REPEAT=2. Rep 0 is cold (first compile in the process), rep 1 is warm.
  Decisions use warm unless a WP says otherwise. Never mix the two.

### A6. Library-first (owner rule)
| Need | Use |
|---|---|
| Nested R̂, R̂, ESS, MCSE (R1) | `arviz_stats` (`rhat_nested`, `rhat`, `ess`, `mcse`). The array API takes numpy arrays. |
| R2 diagnostics | `arviz` 0.23, as `tools/bench_ladder.py` already does, so new rows stay comparable with old ones |
| Batched mean/covariance accumulator | `blackjax.adaptation.metric_buffers.cgl_update_batch` |
| Stan covariance shrinkage | `blackjax.adaptation.mass_matrix` (the window-adaptation estimator) |
| Fisher / low-rank metric | `blackjax.adaptation.metric_estimators` |
| ε search | `blackjax.adaptation.step_size.find_reasonable_step_size` (vmappable, `sampler.py:504-509`) |
| Pathfinder covariance, closed form | `blackjax.mcmc.metrics.lbfgs_inverse_hessian_to_low_rank_metric` |
| λ_max at a metric refresh | `jnp.linalg.eigvalsh` (d ≤ a few hundred, once per refresh) |
| Profiling | `jax.profiler` |
| Reference posteriors | `pm.sample` (PyMC NUTS) |

Vendoring is limited to ChEES `base()` and the private helpers it needs. Any other hand-written
numerical code needs a one-line justification in that WP's SUMMARY.

### A7. Autonomy
- **Decide alone:**
  - implementation details within a WP;
  - accept or kill by the WP's stated criteria;
  - adding tests and harness options;
  - ordering within a wave.
- **Stop that WP, mark it BLOCKED in STATUS, and continue with the others when:**
  - a gate cannot be evaluated;
  - a default change fails G3;
  - a dependency other than `arviz-stats` is needed;
  - a public API change goes beyond Part B;
  - anything would touch `main`.
- **Contradiction protocol.** This applies if a measurement contradicts Part B, for example a WP1
  test does *not* fail on `ecd9ba1`, or §2 numbers do not reproduce within noise.
  1. Do not reinterpret it silently.
  2. Write a LEDGER entry `CONTRADICTION` with both numbers and the commands.
  3. Re-run once with another seed.
  4. Continue only with WPs that do not depend on the claim.
- **Stop the whole run** (write STATUS and a draft REPORT) when:
  - `pixi run test` is red on `pf-r1` and cannot be fixed inside one WP;
  - the environment breaks;
  - the GPU is unavailable for more than 2 h.

### A8. Output artifacts (the review contract: this is what gets examined afterwards)
Everything lives under `results/` on `pf-r1` and is committed, except large files:
```
results/
  STATUS.md                # living: per WP TODO/IN-PROGRESS/DONE/KILLED/BLOCKED, branch, last commit, next step
  LEDGER.md                # append-only: date, WP, decision, evidence (row ids + commands), gate verdicts, SHA
  REPORT.md                # final report (template below); a draft if the run stops early
  env.json                 # jax, jaxlib, blackjax, pymc, arviz, arviz-stats versions; GPU, driver, XLA_FLAGS, SHA
  runs/<WP>__<name>.jsonl  # raw rows, one per (cell, seed, rep), schema pf-r1/v1
  <WP>/SUMMARY.md          # per-WP summary (template below); small PNGs in <WP>/figs/
  reviews/<WP>-code.md, reviews/<WP>-stats.md
  references/MANIFEST.json # per model: path, sha256, chains, draws, max R̂, min ESS, divergences, target_accept
```
Large files stay outside git:
- references in `~/pf-runs/references/<model>.npz`, a dict keyed by variable name holding the
  **unconstrained value vars and the constrained vars**;
- profiler traces in `~/pf-runs/profiles/`.

**JSONL row, schema `pf-r1/v1`.** Every field is always present; use `null`, never omit.

| Group | Fields |
|---|---|
| Identity | `schema`, `wp`, `arm` (e.g. `legacy`, `wp1`, `wp1+wp2`, `wp1+wp2+wp4:diag`), `git_sha`, `pfcheeshmc_file` |
| Cell | `model`, `d`, `chains`, `draws_per_chain`, `warmup`, `seed`, `rep`, `cold`, `kwargs` (the exact dict passed) |
| Environment | `device`, `xla_flags`, `gpu_other_procs`, `cpu_contended` |
| Time and memory | `wall_s`, `stage_s` {`init`, `pathfinder`, `eps0`, `warmup`, `sampling`, `postproc`}, `peak_mem_bytes` |
| Adaptation | `frozen_eps`, `frozen_L_nominal`, `steps_mean` |
| Gradients | `grads_per_chain_warmup`, `grads_per_chain_sampling`, `pf_grads_total` |
| Divergences | `divs_per_1k`, `warmup_divs` |
| R1 quality | `b2_max`, `b2_floor`, `b2_argmax`, `nested_rhat_max`, `superchains` |
| R2 quality | `max_rhat`, `min_bulk_ess`, `min_tail_ess`, `ess_s_worst`, `grads_per_ess_worst`, `ebfmi_min` |
| Failure | `error` |

Rules for these fields:
- Gradient counts must be **exact**: the sum of `num_integration_steps`.
- **Legacy arm:** wrap `PfCheesHMC._run_chees`, whose returned info carries `num_integration_steps`
  per step, and `_to_inference_data`. Force `keep_untransformed=True` so b² can be computed in
  unconstrained space, by variable name.
- **New code** writes `idata.attrs`: `frozen_step_size`, `frozen_L_nominal`,
  `warmup_grad_evals_per_chain`, `sampling_grad_evals_per_chain`, `warmup_steps_used`.

**`<WP>/SUMMARY.md` template.** One section each:
1. Change: what, files, SHAs.
2. Tests added: name, fails on the parent commit?, passes now?
3. Measurements: a table, the JSONL path, and cold or warm stated.
4. Frozen ε/L and the realised step distribution.
5. Gate verdicts: G-A..D, G3, G-R1, G-MEM, G-ORDER, as applicable.
6. Surprises and contradictions with Part B.
7. Library-first exceptions.
8. Exact commands to reproduce.

**`REPORT.md` template**
1. **Run summary:** WPs done, killed or blocked; the branch; `git log --oneline ecd9ba1..pf-r1`.
2. **T1, R1** (C = 1024 and 512, D = 10). Per model × arm:
   - the smallest W passing G-R1, and gradients per chain at that W;
   - warm and cold wall time;
   - peak memory;
   - divergences per 1k;
   - nested R̂;
   - frozen ε/L.
3. **T2, R2** (32 and 128 chains, 1,250 draws). Per model × arm:
   - ESS/s for the worst variable, including warmup and compile;
   - gradients per ESS for the worst variable;
   - max R̂ and min tail-ESS;
   - divergences per 1k;
   - E-BFMI.
4. **T3:** CPU NUTS (32 chains = cores) time to pass G-R1, against GPU R1, on the same models.
5. **T4:** the largest C that fits per model; stage seconds against C ∈ {32, 128, 512, 1024}.
6. **T5:** per-step cost split (kernel, adaptation, history, callback) and the XLA flag's effect.
7. **T6:** doc claims retracted or re-qualified: claim, `file:line`, old number, new number, row ids.
8. Contradictions with Part B and how each was resolved.
9. Open questions for the reviewer, with row pointers.

### A9. Execution waves (respect the caps; the lock serialises all GPU work)
- **Wave 1**
  - Lead: WP0.1, WP0.3 and WP0.4 (branch, commits, `arviz-stats`, skill update,
    `env.json`).
  - steady-tier: WP0.2 references (CPU, before any timed GPU run).
  - fast-tier: WP1 in `wp1`.
  - fast-tier: WP3 harness in `wp3`.
- **Wave 2** (after WP1 and WP3 merge)
  - steady-tier runners under the lock:
    - the WP3 self-test;
    - the `legacy` and `wp1` baselines, R1 and R2 matrices.
  - fast-tier: WP2 in `wp2`. It also owns the warmup-output projection.
  - fast-tier: WP5 in `wp5`, covering init, pool, scoring and sampling-scan projection only.
  - steady-tier: WP8 items, one at a time.
- **Wave 3** (after WP2 merges)
  - fast-tier: WP4 in `wp4`, branched from the updated `pf-r1`.
  - fast-tier: WP6.2 in `wp6`, after WP4 merges.
  - steady-tier: WP6.1 flag A/B and the WP6.3 profile run.
- **Wave 4**
  - steady-tier runners: WP9 re-runs.
  - steady-tier, one per item: doc retractions, each checked by the lead.
  - Lead: REPORT.md.
  - Optional: WP7.
- Reviews take fast-tier slots between implementation tasks. Never run more than 2 fast-tier tasks at once.

### A10. Done for this run
- `pf-r1` has WP0–WP6, WP8 and WP9 merged, or explicitly KILLED or BLOCKED with evidence.
- `results/REPORT.md` is filled per A8, with T1–T6 complete for the 4 models.
- `pixi run test` is green on `pf-r1`.
- `main` is untouched and nothing is pushed.

Then stop and print:
- the branch name;
- `git log --oneline ecd9ba1..pf-r1`;
- the path to `results/REPORT.md`.

---

# Part B — Technical work order

## 0. Executive summary

The handoff is right about the two P0 mechanisms (inverted metric and rate-limited trajectory
growth), and both are confirmed by source and by measurement. It **missed a third P0 bug**, and it
**under-called** two findings that change the strategy.

1. **NEW P0: the metric is permuted against the coordinates on GEV and radon.** [SRC + MEAS]
   Pathfinder's draws, and so the covariance and every low-rank option, are a *tuple* in
   `model.value_vars` order. The HMC state is a *dict*, and JAX flattens dicts in **sorted-key
   order**. On GEV, σ and ξ are swapped. On radon, both parameterisations shift the five scalars
   into the raw effects. This holds before and after the inversion fix. Every GEV and radon metric
   result in the repo is confounded, including radon NC's `mc` beating `lbfgs_lowrank`.
2. **The inversion is real** [SRC + MEAS]. With it fixed on logistic 1000×25 (32 chains), the
   frozen ε goes 0.0021 → 0.66 and nominal L goes 136 → 26.
3. **The trajectory rate limit is real, and BlackJAX deviates from the paper** [SRC + MEAS].
   Hoffman et al. 2021 Algorithm 1 runs Adam on the raw log-T iterate and uses a moving average
   only for the freeze. BlackJAX feeds a `t^-0.5`-weighted MA back as Adam's operating point
   (`chees_adaptation.py:473, 493`). A **one-line change** (raw iterate) makes T converge in
   about 100 steps instead of about 1,000+. Initialising T0 = π/2 adds nothing on top of it.
4. **Many short chains already work. The repo's evidence against them is an R-hat artifact.**
   [MEAS] Chopping one converged 32 × 2,000 run into 1024 × 39, 512 × 20, 512 × 78 and 128 × 313
   reproduces *every* "degraded" R-hat the docs report: 1.050, 1.090, 1.024 and 1.006, against
   the docs' 1.040, 1.08–1.11, 1.023 and 1.006. Scored by bias against a NUTS reference,
   **1024 chains × 10 draws reach b²_max < 0.01:**
   - on 8 Schools after **100** warmup steps, even with today's code;
   - on logistic after **≤100** steps with the metric and adapter fixes. Today's code needs about
     400 steps there and fails at 100 (b² = 0.14).
5. **Pathfinder's covariance is a poor metric even when correctly oriented. Promote the ensemble
   metric to P0.** [MEAS] Pathfinder is under-dispersed everywhere. Median variance ratio
   PF/posterior: 0.56 on 8 Schools, 0.26 on logistic, and **0.011 on radon NC**. Whitening by the
   dense PF covariance leaves condition 30, 227 and 464. 8 Schools is *worse* than unpreconditioned
   (22). Whitening by the true posterior **diagonal**, which is what an ensemble estimate gives at
   C ≥ 512, gives 2 on 8 Schools and 28 on radon NC. Logistic needs a dense or low-rank ensemble
   estimate.
6. **The 1,000-chain regime is blocked by init-stage memory, not by the sampler.** [MEAS] At C=1024
   and d=25 the Pathfinder pool OOMs (797 MiB). With the pool capped, the stratified start scoring
   OOMs next (500 MiB = 65,536 log-density evaluations × 1,000 observations, vmapped at once).
7. **About a quarter of every leapfrog step is a free win** [MEAS]. XLA command buffers for
   `WHILE` and `CONDITIONAL` cut the bare kernel by 23–24% at C=32 and C=1024. The leapfrog loop is
   confirmed to lower to a while-loop with a batched OR-predicate, as the handoff hypothesised. The
   bare kernel is about half of the production warmup step, so the other half (adaptation update,
   history writes, callbacks) is unprofiled.

**Strategy consequence.** The path to the owner's goal is:
- correct metric (WP1);
- paper-faithful ChEES (WP2);
- ensemble metric during a **short fixed warmup** (WP4);
- an R1 bias harness to judge it (WP3);
- init and scan memory fixes so C = 1024 fits (WP5).

It is **not** a stop rule, **not** a longer warmup, and not a different kernel family. Literature arms
(SNAPER, LAPS hybrid, FSM-NUTS reference) come **after** the corrected baseline exists.

---

## 1. Verdict on each claim in the handoff and lit plan

| Handoff/lit-plan claim | Verdict | Evidence / correction |
|---|---|---|
| §1 Default metric is the precision, not the covariance | **CONFIRMED** | `sampler.py:406` [SRC]. Kinetic energy is `½ pᵀ·imm·p` (`blackjax/mcmc/metrics.py:268`). Probe `tgrowth.py`: ε rises 300× on logistic once fixed [MEAS]. |
| §1 "Fix = `inverse_mass_matrix = cov`" | **INSUFFICIENT** | Also fix the **ordering** (§0.1, WP1). Otherwise GEV and radon stay wrong. |
| §1 "radon NC open question: Pathfinder shape poor" | **CONFIRMED and confounded** | The PF variance is 100× too small (median) [MEAS `pfq.py`], **and** the metric is permuted [MEAS `order_check.py`]. Both act at once. |
| §2 T starts at ε, growth ≤ about exp(0.1√N) | **CONFIRMED** | `chees_adaptation.py:517, 371, 473, 493` [SRC]. Traces: logistic L@100/500/1000 = 6.6/37/92 [MEAS]. |
| §2 "check whether the paper feeds the MA back" | **ANSWERED: it does not** | `ref/hoffman2021-chees-hmc.pdf` Algorithm 1: Adam on log Tₙ, EMA T̄ ← 0.9 T̄ + 0.1 Tₙ, used only at n = N_warmup. SNAPER Algorithm 1 is the same: raw iterate, Adam β₁ = 0, β₂ = 0.5 for short runs, lr 0.05, iterate averages only for post-warmup. BlackJAX deviates. |
| §2 "Initialise T0 = π/2 / raise lr to 0.1–0.2" | **NOT NEEDED** | With the raw iterate, T0 = π/2 gives the same traces as T0 = ε (variants C and D in `tgrowth.py`). lr 0.05 is sufficient. |
| §2 Rate limit "sets the ~2,000-step warmup requirement" | **PARTLY** | True for R2 on models that need long T (logistic under the inverted metric). In R1, the binding constraint is **burn-in**. On 8 Schools, W = 100 already suffices with today's code [MEAS `e2e.py`]. |
| §3 R1 conclusions rest on short-chain split-R̂ | **CONFIRMED, stronger** | The chop test reproduces every number; see §0.4 [MEAS `rhat_chop.py`]. The conclusions are refuted, not merely "not established". |
| §4 Host O(C) loops in init | **CONFIRMED (read)** | `sampler.py:260-289` and `:346-351` [SRC]. `adapt_diag` gives identical starts, so the `_score_chain` loop is pure waste. Not yet timed at C = 1024. |
| §4 Pathfinder pool is the first OOM suspect | **CONFIRMED, plus a second one** | Pool OOM at C=1024, d=25 (797 MiB). After capping the pool, the stratified scoring OOMs (500 MiB) [MEAS]. Radon at C = 128 not yet re-tested. |
| §4 Batched trip count gives a while-loop with a batched predicate | **CONFIRMED** | StableHLO has 2 while ops and 85 `or` ops in the scan body [MEAS `loopcheck.py`]. |
| §4 GPU host sync per while iteration | **CONSISTENT** | Command-buffer WHILE/CONDITIONAL give −23–24% [MEAS]. Not yet profiled. |
| §5 Stop-rule cost is recompilation | **NOT CHECKED, demoted** | R1 uses a short fixed warmup and post-hoc nested R̂. Split-R̂ over a warmup window is invalid in R1. Park WP-stop until WP3 shows the warmup needed varies a lot between models. |
| §6 Metric should come from the ensemble | **CONFIRMED, promoted to P0** | Conditioning numbers in §0.5 [MEAS `pfq.py`]. |
| §7 Re-run E4, W7, MEADS and LAPS after the fixes | **AGREE** | E4 is largely superseded by WP4. MEADS is low priority. |
| §8 `eps_probes` typo, `num_paths` unused | **CONFIRMED** | `tools/probe_init_policy.py:125` against `sampler.py:480`; `tests/test_rng_stage_keys.py:70` [SRC]. |
| Lit A1: use BlackJAX `chees_adaptation(mass_matrix_estimation="diagonal")` | **DO NOT USE AS-IS** | See the three bullets after this table. **Port** the idea with a batched accumulator instead. |
| Lit R1 metric: gradients per chain to b²_max < 0.01 | **AGREE, with a measurability floor** | The noise floor of b²_max is about (τ/(C·D))·2 ln(2d). At C = 32 and D = 10 it is about 0.04, so 0.01 cannot be resolved. Use C·D ≥ about 2,000 and print the floor next to every b². |

The lit plan's A1 route through BlackJAX's public `chees_adaptation(mass_matrix_estimation="diagonal")` has three problems:
- Its Welford fold is a **sequential `lax.scan` over chains** every step (`chees_adaptation.py:816-822`), i.e. a 1,024-iteration loop per warmup step at C = 1024.
- It starts from the **identity** metric and cannot be seeded.
- It only accumulates over the last 50% of warmup (`:799`).

**Also new (not in either document)**
- **Harmonic-mean ε gets more conservative as C grows.** On 8 Schools with the same metric, frozen
  ε is 0.51 at C = 32 and 0.29–0.37 at C = 1024 [MEAS]. That means 1.5–1.7× more leapfrogs per
  transition in R1. See WP7.
- **On easy models, slow T growth saves warmup gradients.** In R1, 8 Schools at W = 100 used 1,200
  gradients per chain with today's code against 2,737 with cov + raw iterate, at the same b².
  So judge WP2 on (a) R2 efficiency at short warmup and (b) minimum W on hard models. Judge it
  **together with WP4**, never alone on easy R1 cells.

---

## 2. Measured probe results (reference numbers for the tasks)

All run with BlackJAX 1.7.1 code paths and the repo's own Pathfinder stage (`_run_pathfinder`,
`_run_step_size_tune`). Scripts: `~/pf-research/independent-review-2026-10-19/`.

**2a. Ordering** (`order_check.py`, CPU)

| model | `model.value_vars` order | JAX dict-flatten order |
|---|---|---|
| 8 Schools | mu, tau_log__, theta | same |
| GEV | μ, σ_log__, ξ_interval__ | μ, **ξ_interval__, σ_log__** |
| radon NC | sigma_y, sigma_alpha, sigma_beta (log), mu_alpha, mu_beta, alpha_raw[85], beta_raw[85] | **alpha_raw, beta_raw, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y** |

Toy check: a diagonal metric meant to give `z_b` a variance of 100 gives a kinetic energy of 0.5,
not 50. The intended value goes to `a_vec`.

**2b. R-hat chop test** (`rhat_chop.py` on `~/pf-runs/8schools_pfchees_2000x32.nc`; full-run
R̂ 1.0012, IAT about 1.8): see §0.4. As a control, iid normals give max R̂ of 1.0021 at 1024 × 39
and 1.0040 at 512 × 20.

**2c. Trajectory growth** (`tgrowth.py`, GPU, 32 chains, 2,000 warmup steps). Nominal L = T/ε:

| model | variant | ε0 | frozen ε | frozen L | L@100 | L@500 | L@1000 |
|---|---|---|---|---|---|---|---|
| logistic | A current (precision, MA-fed) | 0.0025 | 0.0021 | 136 | 6.6 | 37 | 92 |
| logistic | B cov, MA-fed | 1.28 | 0.66 | 26 | 10 | 24 | 27 |
| logistic | C cov, raw iterate | 1.28 | 0.66 | 27 | **34** | 19 | 28 |
| 8 Schools | A current | 0.64 | 0.18 | 43 | 22 | 34 | 39 |
| 8 Schools | B cov, MA-fed | 1.28 | 0.51 | 22 | 11 | 14 | 24 |
| 8 Schools | C cov, raw iterate | 1.28 | 0.53 | 19 | 20 | 24 | 16 |

Every variant shows C divergences in warmup, about 1 per chain. That is consistent with the first
step at the over-shot ε0 = 1.28 from the doubling search. Check this in WP2.

**2d. Pathfinder metric quality vs a CPU-NUTS reference** (`pfq.py`, unconstrained space, PF in its
own order so no permutation is involved)

| model | d | cond(posterior) | median var ratio PF/post | cond whitened by PF dense | by PF diag | by **true posterior diag** |
|---|---|---|---|---|---|---|
| 8 Schools | 10 | 22 | 0.56 | 30 | 21 | **2** |
| logistic 1000×25 | 25 | 357 | 0.26 | 227 | 446 | 308 (needs dense) |
| radon NC | 175 | 5,039 | **0.011** | 464 | 724 | **28** |

**2e. R1 bias probe** (`e2e.py`, GPU, **C = 1024, D = 10**, 3 seeds, reference = PyMC NUTS
8 × 5,000 draws at target_accept 0.95). b² = (E_ens − E_ref)²/Var_ref, maximised over xᵢ and
xᵢ² in unconstrained space. Gradients per chain include warmup.

| model | variant | W | b²_max (median) | gradients/chain | frozen ε / L |
|---|---|---|---|---|---|
| 8 Schools | PF starts alone | — | 2.71 | — | — |
| 8 Schools | A current | 100 | 0.0008 | 1,200 | 0.14 / 22 |
| 8 Schools | A current | 2,000 | 0.0008 | 53,372 | 0.14 / 55 |
| 8 Schools | C cov + raw | 100 | 0.0009 | 2,737 | 0.30 / 38 |
| logistic | PF starts alone | — | 0.305 | — | — |
| logistic | A current | 100 / 200 / 400 | **0.138** / 0.039 / 0.0027 | 458 / 1,011 / 3,027 | 0.0015 / 8 → 28 |
| logistic | C cov + raw | 100 | **0.0005** | 1,702 | 0.59 / 29 |

The logistic run needed `draws=64, init_pick="iid"` to fit at C = 1024. The two OOMs are in §0.6.

**2f. Bare-kernel cost** (`loopcheck.py`, 8 Schools, L ≈ 20, no adaptation, warm)

| C | ms per transition | µs per leapfrog | with `--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CONDITIONAL,WHILE` |
|---|---|---|---|
| 32 | 1.01 | 49 | 0.77 ms (−24%) |
| 1024 | 1.38 | 68 | 1.06 ms (−23%) |

For comparison: production warmup is 2.05 ms per step [REPO], and the batched gradient alone is
about 23 µs at C = 32 [REPO].
Note: PyMC's import appends `--xla_force_host_platform_device_count=100` to `XLA_FLAGS`. It is
harmless on GPU, and flags set before import survive.

---

## 3. Target architecture for R1 (what WP1–WP7 build toward)

```
PyMC model ─► one initial point + finiteness check (no per-chain loop)
          ─► Pathfinder (single path; INIT ONLY; pool ≈ a few × C, chunked scoring)
          ─► C starts (stratified; optional K superchains × M replicas for nested R̂)
          ─► warmup phase A (W_A ≈ 25–75 steps): PF-diag metric (correctly ordered/oriented),
                ε by dual averaging; optionally L = 1 (SNAPER) while ε settles
          ─► metric refresh from the ENSEMBLE (diag / Fisher-diag / dense, batched accumulator,
                shrinkage), ε re-search, T re-seeded
          ─► warmup phase B (W_B ≈ 50–200 steps): ChEES with the raw-iterate Adam (paper Algorithm 1)
          ─► freeze ε, T (moving averages), metric
          ─► sample D ≈ 10 draws per chain, storing only position + 6 sample_stats fields
          ─► diagnostics: nested R̂ (superchains), divergences/1k, frozen ε/L, bias harness in benchmarks
```

---

## 4. Work packages

Conventions for every WP:
- Follow the workflow-skill doc: baseline, probe, implement, test, matrix,
  two review tasks, one commit.
- Use `pixi run` always, so the activation env applies (`XLA_PYTHON_CLIENT_PREALLOCATE=false`).
- **All GPU work goes through one lock**: `flock /tmp/pf-gpu.lock pixi run python …`. The card is
  6 GB and shared, and concurrent timings are void.
- One WP per checkout and branch, exactly as in A3 (approved by the owner). Task models and
  reviews as in A4. Output artifacts as in A8.

### WP0 — Infrastructure (P0, do first; mostly CPU; parallelisable)
1. **Branch and commits** (lead, A3).
   - Create `pf-r1`. Commit the review docs to `docs/review-2026-10/`.
   - Commit the six probes to `tools/probes/`, plus `~/pf-o1-stages.py` as `tools/stage_timers.py`.
     The `~/pf-meads-pilot.py` family is optional.
   - Create `results/` with STATUS.md, LEDGER.md and env.json (A8). Results are committed.
2. **Reference posteriors** (steady-tier task, CPU, before any timed GPU run).
   `tools/make_reference.py <model>` writes `~/pf-runs/references/<model>.npz`, a dict **by variable
   name**:
   - the unconstrained value vars, via `idata_kwargs={"include_transformed": True}`;
   - the constrained vars;
   - per-variable mean, variance and MCSE of x and x² (`arviz_stats.mcse`).

   Run settings:
   - PyMC NUTS with 8 chains × 10,000 draws, tune 2,000, target_accept 0.95. GEV may need 0.99.
   - Load the models through `tools/bench_ladder.py`. Its notebook loader stubs `pm.sample`, so
     call `bench_ladder.REAL_PM_SAMPLE`.

   Gate: R̂ < 1.01 and bulk ESS > 10,000 per parameter. Report divergences; if GEV still diverges at
   0.99, record that as a reference caveat.

   Write `results/references/MANIFEST.json`. Models: 8 Schools, GEV, logistic 1000×25, radon NC.
   The owner's model is deferred, but the script must take any PyMC model.
3. **GPU lock wrapper and env banner** in a shared `tools/_common.py`. Every harness prints:
   - the JAX device and `XLA_FLAGS`;
   - `pfcheeshmc.__file__` and the git SHA;
   - cold or warm;
   - peak memory (`jax.devices()[0].memory_stats()["peak_bytes_in_use"]`);
   - foreign GPU processes (A5).

   It writes A8 rows.
4. **Environment and process docs** (lead).
   - `pixi add --pypi "arviz-stats[xarray]>=1.3.3,<1.4"`, commit `pixi.toml` and `pixi.lock`, then
     `pixi run test`.
   - Update the workflow-skill doc on `pf-r1`:
     - (a) implementation tasks write only in their own checkout and branch (A3); review tasks stay
       read-only;
     - (b) harnesses live in `tools/` and results in `results/`, not `~/pf-research/FINDINGS.md`;
     - (c) add the R1 objective and gate G-R1 (§6) next to ESS/s, which becomes R2-only;
     - (d) add the A3 shared-env recipe and the A5 GPU lock;
     - (e) tier use: fast-tier for moderate work, steady-tier for bounded work.

### WP1 — Metric correctness: orientation + ordering (P0; small; GPU only for re-baseline)
**Why:** §0.1, §0.2.
**Change** (`pfcheeshmc/sampler.py`, `_run_pathfinder`, single choke point at the return, `:442-450`):
1. `inverse_mass_matrix = cov` (`:406`). Keep the ridge.
2. Compute the permutation from Pathfinder's tuple order to the kernel's dict order, once.
   `unravel_t = ravel_pytree(tuple_sample)[1]`, `idx = unravel_t(jnp.arange(d))`, and
   `perm = ravel_pytree(dict(zip(self.rvs, idx)))[0].astype(int)`. Apply it to:
   - the dense IMM: `imm[perm][:, perm]`;
   - the low-rank factors before expansion: `sigma[perm]`, `U[perm, :]`.

   Fisher's `gdraws` and `ggrads` are internally consistent in tuple order, so permute only the
   result.
3. Add a helper `self._kernel_flat(position_dict)`. All future metric code (WP4) estimates from
   kernel-state positions, which are already in kernel order, and must use this helper.

**Tests** (new `tests/test_metric_orientation.py`, CPU). Each must **fail on `ecd9ba1`**:
- **Model:** `z = Normal("z", 0, 100)` created first, `a = Normal("a", 0, 0.01, shape=3)` created
  second. The creation order is not sorted.
- **Orientation and order:** with d = 4, `diag(imm)` ≈ kernel-order variance of the PF draws within
  20%. Also `default_metric(imm).kinetic_energy({"z": 1, "a": 0}) ≈ 0.5·Var(z)`.
- **Same check for `imm_source ∈ {lbfgs_lowrank, fisher_lowrank}`.**
- **Smoke:** frozen ε on this near-Gaussian is in [0.3, 2.0] and nominal L is ≤ 10, with a short
  warmup of 200 steps, 32 chains.

**Re-baseline (GPU, R2 shape):**
- 4 models × 32 chains × 1,250 draws × W 2,000, **5 seeds**.
- Report frozen ε, nominal L, `steps_mean`, gradients per worst-variable ESS, ESS/s (cold and warm),
  max R̂, min tail ESS, divergences/1k and E-BFMI. Do the same for the `lbfgs_lowrank` and
  `fisher_lowrank` options.

**Accept:** the tests pass; gradients/ESS do not get worse on ≥ 3 of 4 models; divergence rate does
not rise at equal C. Expected [HYP from 2c]: about 2× fewer gradients per transition on 8 Schools,
about 5× on logistic, and an unknown but large change on GEV and radon.

### WP2 — Paper-faithful ChEES adapter (P0; medium)
**Why:** §0.3. BlackJAX `base()` feeds the MA back (`chees_adaptation.py:473, 493`).
**Change:** vendor `base()` from BlackJAX 1.7.1 into `pfcheeshmc/chees.py`. It is Apache-2.0, so
keep the header and attribution. Add options; the defaults are in brackets:
- `iterate = "raw" | "ma"` [raw]. Adam and the kernel use the raw log-T iterate; the MA is used
  only for the freeze. This is the one-line change already prototyped (`tgrowth.py`).
- `ma = "decay" | "ema"` [decay, `t^-κ`, κ = 0.5]. The paper uses `ema` with 0.1.
- Adam `b1, b2, lr` [b1 0.9, b2 0.999, lr 0.05]. Also test SNAPER's short-run values b1 = 0,
  b2 = 0.5.
- `T0` [ε0].
- `single_step_iters` [0]. SNAPER uses 100: L = 1 while ε settles.
- `trace=True` returns per-step (ε, T, L_nominal, harmonic acceptance, ChEES gradient,
  divergence count) as scan outputs. These are scalars only.
- Keep the whitened criterion, the finite-guards and the clip unchanged.
- Point `utils.chees_warmup_with_progress` at the vendored `base`. The default switches to
  `iterate="raw"` only after the G3 gate.
- **WP2 also owns the warmup-output projection** (W4, warmup half). Today the scan emits position
  plus the full `HMCInfo` per step (`utils.py:192`). It should emit nothing by default, or the
  scalar trace when `trace=True`. Keep `warmup_steps_used` and the exact warmup gradient count as
  `idata.attrs` (A8). The block/stop-rule path must keep working; its tests are in
  `tests/test_warmup_stop.py`.

**Tests:**
- `iterate="ma"` reproduces BlackJAX bit-for-bit (same seed, same frozen ε and L).
- With the raw iterate on a unit Gaussian (32 chains), L_nominal reaches within 20% of its 2,000-step
  value by step 200.
- Frozen values are finite. No ε collapse at C ∈ {4, 32}.

**Measure (GPU):**
- (a) R2: 4 models × W ∈ {250, 500, 1,000, 2,000} × 32 chains, 5 seeds. Report gradients per
  worst-ESS and ESS/s including warmup and compile. The headline cell is logistic at W = 500
  (today: 6× ESS loss).
- (b) R1: run through the WP3 harness.

**Accept:** at W = 500, R2 quality on logistic and GEV matches today's W = 2,000 (G-A..D). No
regression at W = 2,000.
**Kill:** if the raw iterate's noise at C = 32 makes the frozen L worse than the MA-fed run at
W = 2,000 on ≥ 2 models. Then keep `ma` for small C and `raw` for C ≥ 256.

### WP3 — R1 evaluation harness (P0; CPU-heavy; can be built in parallel with WP1/WP2)
**Why:** R-hat cannot judge R1 (§0.4). Every later decision is made on this harness.
**Build** `tools/r1_bias.py`:
- It drives the **public** `sample_pf_chees_hmc` path. The probe in `e2e.py` was a reimplementation
  and does not count.
- Scores against the WP0 references: b²_max over xᵢ and xᵢ² in unconstrained space, by variable
  name. Get unconstrained output from both arms by forcing `keep_untransformed=True` in
  `_to_inference_data` (wrapper for the legacy arm; a public option in new code).
- Prints the noise floor next to every b²:
  `b2_floor = max_f (mcse_run[f]² / Var_ref[f]) · 2·ln(2d)`, with `mcse_run` from
  `arviz_stats.mcse`.
- Exact gradients per chain, warmup plus sampling, from `num_integration_steps`.
- Warm and cold wall, stage split, peak memory, frozen ε/L, divergences/1k, nested R̂, and a
  chop-calibrated split-R̂ for information only.

**Nested R̂** (Margossian et al., arXiv:2110.13017): use the library. Do not reimplement it.
- Call `arviz_stats.rhat_nested(ary, superchain_ids=..., chain_axis=0, draw_axis=1)` (array API,
  verified in the 1.3.3 wheel, `arviz_stats/sampling_diagnostics.py:305`). `superchain_ids` has
  length = chains, with equal-sized superchains.
- Property tests (CPU): on stationary chop data (§2b, with superchain ids assigned to independent
  chains), nR̂ ≈ 1. With dispersed superchains at W = 0 it is clearly > 1.
- Take the threshold for the chosen K from the paper; record the choice in the WP3 SUMMARY.

**Sampler support (small):** `pathfinder_kwargs={"superchains": K}` picks K stratified PF draws and
replicates each C/K times. Default off.

**Sweep definition (R1):**
- C ∈ {512, 1024} (and 128 for information), D ∈ {1, 10}, W ∈ {25, 50, 100, 200, 400, 800}.
- 3 seeds for exploration, 5 for decisions.
- The output per (model, variant) is the **smallest W with b²_max < 0.01 on every seed**, and the
  gradients per chain at that W.

**Self-test:** re-derive §2e on 8 Schools and logistic within noise.

### WP4 — Ensemble metric during warmup (P0 after WP1+WP2; medium-large)
**Why:** §0.5. PF diag on radon NC is 100× too narrow; the posterior diagonal gives cond 28.
**Change** (vendored warmup loop from WP2): phases A → refresh → B, per §3.
- **Accumulator: batched.** Use a Chan merge (`blackjax/adaptation/metric_buffers.py:396`
  `cgl_update_batch`) over the trailing window: positions, and gradients for Fisher. Never
  `_fold_welford_batch`, which is sequential.
- **Estimators to compare:**
  - (i) diagonal variance;
  - (ii) Fisher diagonal, `sqrt(Var x / Var ∇)` as the variance. The gradients are already in
    `DynamicHMCState.logdensity_grad`;
  - (iii) dense with Stan shrinkage `(n/(n+5))·Σ + (5/(n+5))·1e-3·I` when C·window ≥ 10d;
  - (iv) diagonal + low-rank (nutpie recipe, gate G1b).
- **After a swap:**
  - re-search ε vectorised over the ensemble: one leapfrog from every chain at a few candidate ε,
    pick harmonic acceptance ≈ 0.651 by bisection, then restart dual averaging there;
  - re-seed T at max(current T, (π/2)·√λ_max of the whitened ensemble covariance), with λ_max
    from `jnp.linalg.eigvalsh` (A6);
  - pass the new diagonal to the ChEES criterion.
- **Schedule knobs:** `refresh_at` (e.g. [50] or [50, 100]), `window` (e.g. 25), and the
  estimator. Default off until the gate passes.
- **Watch:** PF starts are under-dispersed, so the first refresh will be too narrow. A second
  refresh may be needed. Measure.

**Tests:**
- On a known anisotropic Gaussian (cond 1e3, d = 20, 512 chains), the refreshed diagonal is within
  15% of truth after a 50-step window.
- The dense estimate's whitened cond is < 2.
- Kernel order is respected (reuse the WP1 model).
- No NaN when a chain diverges inside the window: exclude divergent draws, nutpie-style.

**Measure:** WP3 R1 sweep plus the WP1 R2 matrix. Expected [HYP from 2d]:
- radon NC L falls several-fold;
- logistic L ≈ 28 → under 10 with the dense estimator.

**Accept:** gradients per chain to target fall on ≥ 3 of 4 models against WP1+WP2, with no worse
divergences/1k. **Kill:** an estimator that does not beat the diagonal by ≥ 1.2× on correlated
models (dense, low-rank) is dropped.

### WP5 — Capacity and fixed costs at large C (P1; blocks C = 1024 at d ≥ 25)
1. **Pool.** Replace the MC covariance with the **closed-form** PF covariance, i.e. the dense
   expansion of `lbfgs_inverse_hessian_to_low_rank_metric`, which is the same Gaussian. Verify the
   equivalence numerically on 3 models. Cap the pool at what start selection needs (e.g. `4·C`).
   `sampler.py:310` defaults to 4,000 draws per chain.
2. **Scoring.** Chunk the stratified scoring (`sampler.py:377-381`) with
   `jax.lax.map(..., batch_size=…)` or score `4·C` draws.
3. **Init de-duplication.** Use one PyMC initial point. Drop the per-chain `_score_chain` loop
   (`:346-351`). Loop only over failing chains in `_ensure_finite_inits` (`:260-289`), using
   `jnp.where` and indexing instead of per-chain slicing and stacking.
4. **Scan projection (W4, sampling half only).** The warmup half belongs to WP2, which avoids a
   merge conflict in `utils.py`. Today the warmup emits about 14 GB at C = 1024, d = 175, W = 2,000.
   Sampling should emit position plus the 6 contract fields (`sampler.py:704`). Keep the
   `tests/test_inference_data.py` contract. `logdensity` is needed for `lp`.
5. **Measure:**
   - the maximum C that fits per model;
   - stage timers at C ∈ {32, 128, 512, 1024};
   - the true device cost per step against C, replacing the confounded "c/step 3.17 → 13.69 ms";
   - the re-trace cost per new C (`jax_log_compiles`). Bucketing C is an owner decision.

**Accept:** radon NC runs at C = 1024 × D = 10 on 6 GB. The init stages at C = 1024 take < 2 s warm.

### WP6 — GPU per-step efficiency (P1; implementation can run in parallel with WP4, measurement is serial)
1. **Command buffers.**
   - **Get the flag right first.** Find this XLA's *default* `xla_gpu_enable_command_buffer` list.
     §2f's explicit list may have dropped a default type. Then test whether additive syntax
     (`+WHILE,+CONDITIONAL`) is accepted.
   - Validate: same seeds give the same frozen ε/L, and posteriors match within MC error.
     Bitwise identity is not required, but note it.
   - Report the end-to-end warm wall on 4 models at C = 32 and 1024, and the change in compile time.
   - **If accepted** (≥ 5% warm end-to-end on ≥ 2 models and gates pass), set `XLA_FLAGS` in
     `pixi.toml` `[activation.env]` (lead) and document it in the README. In any case,
     document it.
   - PyMC appends to `XLA_FLAGS` at import; it does not overwrite. Activation overrides the
     caller's environment, so a user override must use the `pixi run env XLA_FLAGS=... python` form.
2. **Unbatched trip count.** L is shared by all chains. Write a batched kernel:
   - an unbatched `fori_loop(0, L)` whose body is the vmapped leapfrog;
   - vmapped momentum draw and MH step;
   - the same `HMCInfo` fields.

   This removes the batched OR-predicate and the per-iteration carry selects. Measure at
   C = 1024 × d = 175, where select traffic is largest. Contract: the posterior is identical in
   distribution, same divergence semantics.
3. **Profile** one production warmup step (`jax.profiler.trace`). Split the 2.05 ms
   [REPO] into kernel (about 1.0 ms bare), adaptation update, history writes and callbacks.
   Target the largest share.
4. Chunked leapfrog (K steps per while iteration, masked tail): only if per-iteration overhead after
   steps 1–2 is still > 30% of the gradient cost.

**Accept:** each item ≥ 5% warm end-to-end on ≥ 2 models with gates G-A..D, or dropped.

### WP7 — Step-size target at large C (P2)
**Why:** frozen ε falls 0.51 → about 0.3 from C = 32 to 1,024 [MEAS], because the harmonic mean is
dominated by the worst chains.
**Compare:**
- harmonic mean at 0.651 (current);
- harmonic mean at 0.8 (SNAPER's target);
- a trimmed harmonic mean (drop the worst 1–2%);
- the arithmetic mean at 0.8.

**Judge** on R1 gradients per chain to target, divergences/1k and E-BFMI. Divergences are the owner's
diagnostic, so any variant that raises divergences/1k at equal C fails G3.

### WP8 — Hygiene (P3; any time; task-sized)
- Raise on unknown stage kwargs (whitelist per stage). This catches `eps_probes`
  (`tools/probe_init_policy.py:125`) and `num_paths` (`tests/test_rng_stage_keys.py:70`).
- Halton bits from the real draw count, not `max_sampling_steps=1000` (`utils.py:70, 80-83`).
- Remove the mutable default dicts (`sampler.py:891-894`) and the dead `xmap` branch (`:754-763`).
- Statistical test: anisotropic Gaussian, mean and variance within 4 MCSE (CPU, ≤ 60 s).
- README: "ChEES does not adapt the mass matrix" is true only of the private `base()` the repo
  calls.

### WP9 — Re-runs and doc corrections (after WP1–WP5 merge)
- **Headline comparison for the owner:** time to b²_max < 0.01, CPU NUTS (32 chains = cores,
  tune + draws until the target is met) against GPU R1 (C = 1024, D = 10, warm and cold).
  Also the R2 ESS/s grid at 32 and 128 chains.
- **Retract or re-qualify, each with the new measurement:**
  - README: "More chains also means less needed warmup".
  - README: "Warmup length: 2,000 is enough / do not go below 1,000" (GEV).
  - README: the Fisher option's 1.22×/1.96×/0.19× (baseline inverted and permuted).
  - `ladder-results.md`: eigen-spectra "units 2,000× below the posterior" (it is precision against
    covariance).
  - `ladder-results.md`: "rescaling made every model worse".
  - `ladder-results.md`: "short chains do not substitute for long ones", "more warmup does not
    repair what more chains broke", and "a Pathfinder start needs draws-per-chain to burn off".
    These are refuted by §2b and §2e.
  - `ladder-results.md`: "c/step 3.17 → 13.69 ms" (confounded).
  - `ladder-results.md`: the radon OOM "memory source NOT ESTABLISHED" (pool and scoring confirmed).
  - `ladder-results.md`: every radon and GEV metric verdict (permuted).
  - `gpu-upgrade-plan.md` D7: LAPS "wrong regime". R1 is now the owner's regime.
- MEADS re-test at **equal gradients**, after WP1/WP4, through the WP3 harness. Low priority.
- Stop rule: re-open only if WP3 shows the minimum W varies > 4× across models. Then jit the block
  runner and gate on ensemble-statistic stabilisation or nested R̂, never split-R̂ over a window.

### WP10 — Literature arms (NOT in this run, see A0; after WP9, each with its own stop rule from the lit plan)
Order:
1. **SNAPER criterion** in the vendored `base`: project onto the top principal direction with an
   Oja update. This is about a day now that WP2 exists. Installing TFP is not needed.
2. **LAPS-style hybrid**: unadjusted MCLMC transport, then WP4 refresh, then short ChEES. Build it
   only if WP3 shows **burn-in** gradients dominate R1 cost after WP4.
3. **FSM-NUTS reference arm** (Dance et al.) for divergence ground truth.
4. Tempered SMC: only if sequential depth is still the bottleneck.

Stop rules as in lit plan §2. "Beat" means fewer gradients per chain to the bias target, **and**
warm wall-clock, on the WP3 harness.

---

## 5. Sequencing, dependencies and tasks

```
WP0 ──┬─► WP1 ──► WP2 ──► WP4 ──┐
      ├─► WP3 (build in parallel) ┼─► WP9 ──► WP10
      ├─► WP5 (parallel with WP2/WP4; touches sampler init, not the warmup loop)
      ├─► WP6 (parallel implementation; serial GPU measurement)
      └─► WP8 (any time)                WP7 after WP4
```

**Task assignments, waves and caps: see A4 and A9.** Files each WP owns, to avoid merge
conflicts:

| WP | Owns |
|---|---|
| WP1 | `_run_pathfinder` metric block (`sampler.py:393-450`) |
| WP5 | `_run_pathfinder` pool and scoring (`:309-391`), the init helpers, and the sampling scan |
| WP2, then WP4 | `utils.py` warmup loop and the new `pfcheeshmc/chees.py` |
| WP6.2 | the kernel, after WP4 |

WP1 merges before WP5 branches. Literature tasks are not needed in this run.

**Decision rule reminders:**
- No default changes without G3: ≥ 5 seeds, divergence rate does not increase, means within 1σ.
- In R1, "quality" is the WP3 bias gate plus nested R̂, **never** split-R̂.
- Report frozen ε/L and the realised step distribution with every speed claim (skill §3).

---

## 6. Gates (keep G-A..G-D and G1–G4; add these)

- **G-R1 (bias):** b²_max < 0.01 for xᵢ and xᵢ² on every seed, at C·D ≥ 2,000, with the noise floor
  printed. A cell whose floor exceeds 0.005 is "not resolvable", not a pass.
- **G-R1-diag:** nested R̂ below the paper's threshold for the chosen K (superchain runs only).
- **G-MEM:** peak device memory ≤ 5.0 GB on the 6 GB card. Record the maximum C that fits.
- **G-ORDER:** the WP1 tests are permanent. Any new metric code must pass the non-sorted-names model.

**What "done" looks like for the owner:**
- R1 (C = 1024, D = 10) passes G-R1 on 8 Schools, GEV, logistic and radon NC with warmup ≤ 300
  steps. The owner's model is added in a later run.
- This is the project target, not this run's exit condition (A10). A run that ends with honest
  KILLED or BLOCKED verdicts and complete tables is a successful run.
- Gradients per chain to target are reported (published ChEES baselines are 537–1,168 on IRT and
  German Credit, lit plan A4).
- Warm wall-clock is quoted against CPU NUTS time to the same target. Divergences/1k are no worse
  than today's R2.

---

## 7. Owner decisions

Resolved 2026-10-19: checkouts, vendoring, `arviz-stats`, the XLA flag, and the owner model
deferred. See A1.

Still open. Do not decide these in this run; put them in REPORT §9 with evidence:
- **Default API for R1:** expose `warmup` and the metric-refresh options on `sample_pf_chees_hmc`,
  and change the default warmup. Implementing the options as **non-default** kwargs is allowed.
  Flipping defaults needs G3 **and** the owner.
- **Chain-count bucketing** to avoid re-tracing per C: measure the cost in WP5; do not implement it.

---

## 8. Things deliberately left out, and why

- **Stop rule (handoff §5):** wrong signal for R1. It is parked (WP9) unless the minimum W varies a
  lot between models.
- **Float32:** parked on measurement [REPO]. Revisit only for data-heavy models after WP6.
- **Funnels and centred radon:** out of scope per the owner. Their numbers are not used to judge
  anything here.
- **GIST, DEER, flows/TESS:** as in the lit plan's skip list. That reasoning stands.
