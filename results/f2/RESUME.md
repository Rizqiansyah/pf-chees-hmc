# F2 — resume note (written 2026-10-23, before a context compaction; updated 2026-10-23 22:45 after F2e and F2f, 2026-10-24 02:50 after F2g)

## Programme in one paragraph
F2 makes a sampler call compile nothing new when only seed, data values, prior values, W or D change.
It is the per-call recompilation fix behind run 2's finding: GPU was slower than 32-core CPU NUTS on
3 of 4 models because of fixed per-call overhead. The plan is `docs/review-2026-10/pf-chees-hmc-f2-plan.md`.
The owner wants **universal** improvements (any PyMC model), not benchmark tuning, and least maintenance.

## Where things stand
- **Branch:** `f2` (local only). Head fd98faa at the time of writing; it had a clean suite:
  `413 passed, 51 warnings in 494.25s`.
- **Merges:**
  - `main` is 232ae58, untouched; `origin/main` is ecd9ba1.
  - **Never push.**
  - Merge `f2` → `main` only when the owner says so.
- **Done and merged:**

  | WP | what it did |
  |---|---|
  | F2a | `pfcheeshmc/lifted.py`: rule C lifting (model inputs before the rewrite), dtype widening, erfcx guard, registry keyed on `keep_unused` HLO |
  | F2b-1 | `pfcheeshmc/compiled.py`: module-level jits with a static `fn`; stale-static guard |
  | F2b-2 | loops as host-driven chunk programs with a dynamic count; no `jax.debug.callback`; `fold_in` keys |
  | F2c+F2d | JAX initial points; start scoring capped at 4C; one-dispatch ε0 |

  Each has a SUMMARY in `results/f2/<WP>/`. The running record is `results/f2/LEDGER.md`.
- **Status:** a warm call makes 0 XLA and 0 PyTensor compiles (tests: `tests/test_compile_once.py`).
- **GPU smoke** (pre-F2c/d, W=25, `results/f2/runs/smoke_*`): warm 8schools 6.2→0.8 s, radon 16.4→2.9 s.
  Cold calls roughly halved.

## Update 2026-10-23 22:45
- F2e is merged (89e0682). The suite passes: `453 passed, 60 warnings in 523.25s`.
- F2f is done: `results/f2/F2f/REPORT.md`. The R1 gate passes 4/4. GPU f2 beats CPU NUTS by 15×,
  9.4×, 6.8× and 2.5–3.8×.
- Waiting on the owner:
  - merge `f2` → `main`?
  - optional F2g: a jitted start-repair check, which removes GEV's 156 eager compiles;
  - a `tests/conftest.py` teardown for the memory-mapping cliff;
  - should the disk cache stay opt-in?

## Update 2026-10-24 02:50
- The owner said "Yes to all" (2026-10-23): `f2` was merged into local `main` (19f0503, tree = 97b6569), and
  F2g ran.
- F2g is merged into `f2` (51e15cf), with `results/f2/F2g/SUMMARY.md`. The suite passes:
  `496 passed, 62 warnings in 649.33s`.
  - jitted start-repair check: GEV's cold init is 8.98→2.47 s on the GPU, and its warm init 0.21→0.03 s;
  - `tests/conftest.py` teardown: peak mappings 15.7k;
  - the disk cache is on by default in `${XDG_CACHE_HOME:-~/.cache}/pfcheeshmc/jax`, with opt-out
    `PFCHEESHMC_COMPILATION_CACHE=0`. Timing tools default to off.
- Waiting on the owner:
  - merge `f2` (with F2g) → `main` again? The earlier approval does not carry over.
  - Accept no size cap on the default directory (task's DEVIATION 1)?
  - Fix the `1 < pm_chains < chains` broadcast bug (small task)?

## Update 2026-10-24, autonomous run (owner: "Do not ask. Work autonomously.")
- **Owner's up-front answers** (in the LEDGER):
  - F2h → local `main` without asking, after the review and the suite;
  - follow-ups F2i, F2j, F2k, one task at a time, no default flips;
  - upstream bugs stay drafted;
  - never push.
- **Local `main`:** 89721fd (tree = f2 2872ecb: F2h, F2i and F2j). Not pushed.
- **F2h:** done. Suite on 44a0556: 514 passed. GPU check: the host gather costs nothing measurable.
- **F2i:** done as a cheaper widening, no memo (a7a4413). Suite: 575 passed. CPU: 0.016–0.046 s saved per call.
- **F2j:** done (6ade5b1). The tests release compiled programs only above `max_map_count // 2 - 16,000` mappings.
  Suite: 584 passed in 725 s, against 774 s before. Peak 22.7k mappings.
- Checkouts and branches of F2h–F2j are removed. TASK-RULES now tell a task to sign with its own model (62c38a3).
- **F2k task** (float32 probe, CPU): running in `~/pf/f2-k` (branch `f2-k` from 62c38a3). When it reports:
  - review, merge into `f2`, run the suite, write the SUMMARY and the ledger;
  - merge into local `main`.
- **F2k GPU part (lead):** float32 warm/cold stages, and the R1 gate rows on the GPU.
- **At the end:** a final F2 report to the owner.

## Next steps (as written before F2e; steps 1–2 are now done)
1. **F2e: one fast-tier task**, in a checkout at `~/pf/f2-e` on branch `f2-e`. Git refs cannot nest
   `f2/x` under `f2`.
   - **(a) Opt-in on-disk JAX compilation cache.**
     - A documented switch, off by default: `jax_compilation_cache_dir`, plus min compile time 0 and min
       entry size -1.
     - Test with two fresh processes, the second with new data or priors at the same shapes: it hits the
       cache.
     - Add a README section.
   - **(b) Lift memo for the "rest" stage**, which costs 0.14–0.46 s per call (`lift_logp` plus the
     transform lift; breakdown in the LEDGER).
     - Use a structural hash of the PRE-rewrite lifted graph (op `__props__`/hash, types, static
       constant bytes) to skip the rewrite, funcify and key lowering.
     - Before reusing a hit, verify it with `pytensor.graph.basic.equal_computations`, so a hash
       collision can never return a wrong `fn`.
     - Tests: hit, miss, a forced collision rejected, and values in the right order.
2. **F2f: lead, GPU, quiet host.** Check `uptime` and `nvidia-smi`; another user, ccaprani,
   sometimes loads the box.
   - **Tool:** `tools/warm_stages.py TREE MODEL C D W --calls 3`, under `tools/gpu_run.sh`. For TREE use
     a `git archive 232ae58` for `main`, or the `f2` tree.
   - **Shapes:** the M4 shapes, C=1024, D=10, each model at its smallest passing W: 8schools 25, gev 25,
     logit 50, radon 300. Run cold and warm, per stage.
   - **R1 bias gate:** run it on f2 for the 4 models at those W, with the run-2 driver and harness
     (`results/r2/drivers/run_step.sh`, `tools/r1_bias.py`), compared with run-2's M2a/M4 rows.
     - Needed because radon's start picks changed (F2c), the refresh and stop-rule streams changed
       (F2b-2), and draws moved by up to about 2e-10.
   - **CPU NUTS first call** in a fresh process, with the PyTensor cache cold and warm
     (`tools/t3_cpu_nuts.py`).
   - **Report:** redo R2-T4 for the "new data/priors every run" and "repeat runs" workflows, plus a
     short report to the owner.
3. **Then** ask the owner about merging `f2` → `main`.

## Workflow rules (owner's)
- Tasks are sequential, one at a time: fast-tier subtasks, CPU only, rules in `results/f2/TASK-RULES.md`.
- The lead reviews, merges with `git merge --no-ff`, runs the full suite on `f2`, writes the
  SUMMARY from the task report (tasks cannot write report files), and updates the LEDGER.
- When a task stops at the time limit, resume it with session message after the reset. Do not restart it.
- GPU work goes only through `tools/gpu_run.sh`, the flock lock, and only by the lead.
- Library-first. Every rule is stated in graph/PyMC terms, never special-cased to the ladder models.

## Known issues (not fixed; F2 scope or not)
- **Upstream, not filed:**
  - PyMC 5.28.5 `pm.Uniform(-100, 100)` gives logp NaN (int8 overflow); fixed on our path;
  - JAX `erfcx` returns 0 on [26.54, 26.64]; a bug draft is in `results/f2/F2a/SUMMARY.md`.
- **Pre-existing:**
  - `1 < pm_chains < chains` raises a broadcast `ValueError` in `_run_pathfinder`;
  - a multi-block stop rule keeps only the last block's `WarmupInfo`, so gradients are under-counted.
- **Accepted limits:**
  - W_A < 20 changes the refresh pool shape and compiles once;
  - a scalar `initval` change compiles one tiny program;
  - equal-valued prior literals merged by PyMC give a new program when they diverge.
- **UNVERIFIED:** float32; the GPU timing of `jittered_initial_points`. (The vmapped ε0 and memory at
  C=1024, d=175 were closed in F2f.)
- **Tooling gotcha:** `JAX_PLATFORM_NAME=gpu` leaks from the example notebooks into `os.environ` when a
  ladder model is built. CPU subprocesses must drop it.
