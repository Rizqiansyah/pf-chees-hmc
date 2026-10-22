# F2c + F2d (+ one-dispatch ε0) SUMMARY

Task: fast-tier, one round.
- Branch `f2-cd`: 6 commits on cdc4a55, head e3694cd. Merged into `f2` with `--no-ff`.
- Task's suite: `413 passed, 51 warnings in 494.93s (0:08:14)`.

## Result
- A warm second `sample()` makes **0 XLA compiles and 0 PyTensor function compiles**. This holds on
  the 4 ladder models, a `dims=` model, `jitter+adapt_diag` and `pm_chains=C`. cdc4a55 built 3
  PyTensor functions per call (all in `pm.init_nuts`), and 5 for `dims=`.
- Draws vs cdc4a55 at C=1024: **bit-identical** on 8schools, gev and logit (every array and config
  value). Radon's start picks change by design (F2c).

## DEVIATIONS (accepted)
1. **F2c stride rounds up:** `stride = max(ceil(total / 4C), 1)`. Flooring could score up to about 8C.
   It equals the old formula whenever total ≤ 4C. Radon: stride 30, 4,084 rows scored (was 122,500).
2. **`adapt_diag` equals PyMC only where PyMC computes in float64.**
   - PyMC computes a support point from int or float literal priors in float32: for example `Gamma(2., 3.)`
     gives `log(α/β)` off by 3e-8.
   - Ours is the exact value, because F2a widens prior literals. That is pinned by tests at 1e-15
     against the analytic value and at 1e-6 against PyMC.
   - The 4 ladder models and the `initval`, `pm.Data` and `dims=` models agree exactly (0.0).
3. **A user scalar `initval` stays static** (F2-2). Changing it compiles one tiny `jit(initial_point)`.
4. **Files beyond the brief:**
   - `lifted.py`: `initial_point_expression`, `lift_initial_point` and `_initial_point_shapes`;
     `_abstract` folds the init graph's shapes rather than compiling `model.initial_point()`.
   - New `tools/start_dispersion.py`.
   - `count_pytensor_functions` and `dims_model` test helpers.
   - **A latent F2a bug, fixed (`lifted.py:408-418`).** `_key` lowered without `keep_unused`, so JAX
     dropped unread arguments from the text. Two graphs that differed only in unused arguments shared a
     key, and the registry returned the wrong `fn`. The case found: a Normal's identity transform and
     its initial point.
     - Here it raised a `TypeError`.
     - With equal arity it could have computed from the wrong inputs without any error.
     - Fixed with `jax.jit(..., keep_unused=True)`; mutation M11 pins it.
5. **The ε0 program is specialised on C** (it takes the `(C, d)` stack and slices inside). A new C
   compiles once, like every site.
6. **Probe keys are `fold_in(key, i)`**, which equals `split(key, n)[i]` under the default PRNG config.

## Items
- **F2d:**
  - initial points come from PyMC's `make_initial_point_expression`, lifted, through a module-level
    jit (`compiled.initial_point`);
  - `jitter+adapt_diag`: U(−1,1) in value space, redrawn up to `jitter_max_retries` using the F2a logp
    (`compiled.jittered_initial_points`, a `while_loop`). When retries run out, `model.check_start_vals`
    raises PyMC's `SamplingError`;
  - fallback: `pm.init_nuts`, with a warning;
  - other init modes still use `pm.init_nuts`;
  - `pm_tune` is unused by these two modes in PyMC 5.28.5 as well (`pymc/sampling/mcmc.py:1703-1704`
    only); a test pins it.
- **F2c:** `START_SCORE_PER_CHAIN = 4`.
  - Picks are bit-identical on 8schools, gev and logit.
  - Radon start dispersion (run-1 §3j method, C=1024, seeds 1–5): median ratio 0.1017 → 0.1020, min
    0.0799 → 0.0783, and 171 of 175 coordinates ≤ 0.5 both before and after.
  - Radon Pathfinder stage (CPU): 1.11–1.20 → 0.76–0.83 s. Scoring: 0.22–0.28 → 0.055–0.062 s.
- **ε0:** one vmapped dispatch, with infeasible probes masked (feasible ones moved to the front, and
  a NaN-median over the masked log).
  - Bit-identical to the per-probe code, with 0 and with 4 infeasible probes.
  - A different infeasible count compiles 0. The old code compiled 5 small programs whenever the count
    changed.

## Tests
- `tests/test_initial_points.py` (29);
- `tests/test_step_size_batched.py` (8);
- `tests/test_start_scoring.py` (8);
- 5 new tests in `tests/test_compile_once.py`.

Must-fail evidence:
- on cdc4a55 the compile tests give `assert (3 == 0)` (PyTensor functions);
- the old ε0 code compiles 5 programs on a different infeasible count;
- mutations M1–M11 each fail the intended test.

## Measurements (task, CPU, C=1024, D=10, W=25, load average 2–6; `tools/warm_stages.py`)
Seconds, calls 2–3 averaged.

| model | tree | wall | init | pathfinder | eps0 | warmup | sampling | transform | rest |
|---|---|---|---|---|---|---|---|---|---|
| 8schools | cdc4a55 | 0.99 | 0.47 | 0.02 | 0.03 | 0.20 | 0.07 | 0.02 | 0.17 |
| 8schools | new | **0.53** | 0.03 | 0.03 | 0.00 | 0.20 | 0.07 | 0.02 | 0.18 |
| gev | cdc4a55 | 2.62 | 1.17 | 0.03 | 0.04 | 0.55 | 0.20 | 0.06 | 0.58 |
| gev | new | **1.47** | 0.17 | 0.03 | 0.00 | 0.55 | 0.20 | 0.06 | 0.46 |
| logit_1000x25 | cdc4a55 | 3.70 | 0.26 | 0.09 | 0.04 | 2.15 | 0.90 | 0.01 | 0.26 |
| logit_1000x25 | new | **3.29** | 0.02 | 0.09 | 0.01 | 2.13 | 0.89 | 0.02 | 0.14 |
| radon_noncentered | cdc4a55 | 5.03 | 0.65 | 1.09 | 0.12 | 1.97 | 0.76 | 0.13 | 0.30 |
| radon_noncentered | new | 4.89 | 0.16 | 0.77 | 0.02 | 2.50 | 1.00 | 0.14 | 0.30 |

Radon's higher warmup and sampling time on the new tree is a seed-dependent ChEES mode (L about 16
vs about 9), not overhead. Over 30 seeds that mode occurred 6/30 on cdc4a55 and 9/30 on the new tree
(Fisher p = 0.55).

## Open and UNVERIFIED
- **Pre-existing, not fixed:** `1 < pm_chains < chains` raises a broadcast `ValueError` in
  `_run_pathfinder` (also on cdc4a55). Recorded in the ledger.
- **"rest"** (`lift_logp` plus the transform lift, 0.14–0.46 s) is now the largest fixed cost
  after the loops on 8schools and gev. It goes to F2e.
- **UNVERIFIED:**
  - GPU timing of `jittered_initial_points` and the vmapped ε0;
  - the R1 gate on radon (lead, F2f);
  - float32.
