# F2 ledger

- 2026-10-22 22:50 — Owner's go. Host quiet (load 0.5, GPU idle). Branch `f2` created from `main`
  (232ae58). Plan, task rules and probes committed.
- 2026-10-22 — Probe 5 (`ops_probe.py`, `hoist_probe5.py`). PyTensor's JAX rewrite fuses prior values
  into `Composite` inner graphs, so lifting after the rewrite misses them; the "HLO invariant" check
  there was vacuous.
- 2026-10-22 — Probe 6 (`hoist_probe6.py`). Rule C (lift model inputs before the rewrite; formula
  constants stay static):
  - logp is equal;
  - perturbed rebuilds give byte-identical HLO;
  - `Sqr`/`Log1p`/`Softplus` are kept, whereas rule A turns `Sqr` into `Pow`;
  - CPU vmap(value_and_grad) is equal or faster.
  
  Plan F2-2 revised to rule C. This is the owner's intent (data and prior changes never recompile)
  with PyMC's numerics untouched. Owner to be informed in the next status.
- 2026-10-22 — F2b is split into F2b-1 (values as arguments, plus the cache) and F2b-2 (loop
  lengths, plus Pathfinder). The value→RV transform is added to F2a/F2b. The F2f baseline is
  deferred to run back to back with the after-F2 measurement.
- 2026-10-22/23 — F2a, round 1 (fast-tier task). Rule C is implemented in `pfcheeshmc/lifted.py`.
  Accepted:
  - `Shape`/`Shape_i`/`Output` added as value uses (otherwise `pm.Data` leaks);
  - a wider erfcx band with a 9-term series;
  - draws not bit-identical (ulp-level, from prior arithmetic moving out of constant folding).

  Open issue raised: a lifted prior's dtype depends on the literal's value (PyTensor autocast), so
  the key changes with the value.
- 2026-10-23 — Lead probes (`scratchpad/f2/widen_probe2.py`, `uniform_overflow.py`).
  - `clone_replace(rebuild_strict=False)` can widen literals; `graph_replace` cannot.
  - Found: `pm.Uniform("u", -100, 100)` has logp NaN in PyMC 5.28.5 on both backends (int8
    overflow).
- 2026-10-23 — F2a, round 2. Widening is now rule "model_inputs", with fallback
  "model_inputs_native_dtype". Folded one-element literals are matched by fingerprint. The Uniform
  NaN is fixed on our path.
- 2026-10-23 — F2a merged (ff, 14cd7cf). Task suite: `262 passed, 5 warnings in 380.69s (0:06:20)`.
  SUMMARY is in `results/f2/F2a/`. Lead suite on merged `f2`:
  `262 passed, 4 warnings in 382.86s (0:06:22)`.
- 2026-10-23 — F2b re-split.
  - F2b-1 is the infrastructure (module-level jit with a static lifted `fn`) plus the non-loop sites.
  - F2b-2 is the loops, as host-driven chunks with no `jax.debug.callback`: today a cached program
    would call the first call's progress bar, and callbacks block the on-disk cache.
  - Draw buffer sized `min(D, 1000)` with a dynamic fill count.
- 2026-10-23 00:17 — F2b-1 task launched (checkout `~/pf/f2-b1`).
- 2026-10-23 — F2b-1 round 1: the task stopped at the time limit and was resumed at 03:40.
  - Non-loop sites are module-level jits with a static `fn`. A warm second call now compiles only
    the 2 loop scans (was 8).
  - CPU warm wall: 8schools 6.07→4.27 s, radon 16.8→7.1 s (Pathfinder 8.9→1.1 s).
  - Cold first call: 26–40 s → 13–19 s.
- 2026-10-23 — F2b-1 round 2 (lead review). A kept-static (baked-in) shared variable went
  silently stale after `pm.set_data` at the same shape. Fixed with `is_stale()`, so it lifts again;
  mutation M11 reproduces the wrong answer.
- 2026-10-23 04:01 — F2b-1 merged (ed67175). Lead suite:
  `315 passed, 13 warnings in 393.53s (0:06:33)`. SUMMARY is in `results/f2/F2b-1/`.
- 2026-10-23 04:05 — F2b-2 task launched (checkout `~/pf/f2-b2` from c35fbba). Design: host-driven chunks with a dynamic count; per-step `fold_in` keys (stream change); no `jax.debug.callback`; per-step outputs gathered to NumPy; Halton fixed at 30 bits.
- 2026-10-23 — F2b-2 (fast-tier; interrupted 04:30–08:40, then resumed).
  - Loops run as host-driven chunk programs: dynamic count, no callbacks, `fold_in` keys.
  - Under the default PRNG config `fold_in(key, i) == split(key, N)[i]`, so the default path's key
    stream is unchanged. The stop-rule and refresh streams changed.
  - A warm call compiles 0 programs, also on W and D changes. CPU warm wall:
    - 8schools 4.39→0.99 s;
    - gev 4.59→2.56 s;
    - logit 5.21→3.72 s;
    - radon 7.05→5.00 s.
  - Pre-existing defect noted (not F2 scope): with a multi-block stop rule, `WarmupInfo` holds only
    the last block, so gradient counts under-count.
- 2026-10-23 09:23 — F2b-2 merged (a417002). Lead suite:
  `361 passed, 41 warnings in 447.73s (0:07:27)`. SUMMARY is in `results/f2/F2b-2/`.
- 2026-10-23 — `lift_logp` breakdown (CPU, `scratchpad/f2/lift_cost.py`): 0.13–0.45 s per call.
  - `model.logp()`: 0.02–0.30 s.
  - The JAX rewrite: 0.07–0.26 s.
  - The rest: widening and the key lowering.
  - A later memo (per model object, or a pre-rewrite signature) could save most of this.
- 2026-10-23 09:25 — GPU smoke, `tools/warm_stages.py TREE MODEL 1024 10 25 --calls 3` under
  `tools/gpu_run.sh`. TREE was `main` (a `git archive` of 232ae58) or `f2` (a417002). Host load
  about 1.2, GPU idle. Logs are in `results/f2/runs/smoke_*`. Wall time per call:

  | model | tree | cold call | warm call |
  |---|---|---|---|
  | 8schools | main | 35.9 s, 457 compiles | 6.2 s, 8 compiles |
  | 8schools | f2 | 18.5 s, 141 compiles | 0.82–1.04 s, 0 compiles |
  | radon (W=25) | main | 76.5 s | 16.1–16.7 s |
  | radon (W=25) | f2 | 38.6 s | 2.9–3.0 s, 0 compiles |

  - Breakdown of f2's warm radon call: init 0.66, pathfinder 0.40, eps0 0.65 (32 serial probe
    dispatches), warmup 0.46, sampling 0.23, transform 0.18, rest 0.31 s.
  - XLA:GPU's per-step `cond` shows no visible cost (8schools warmup takes 0.06 s).
- 2026-10-23 09:35 — F2c+F2d task launched (`~/pf/f2-cd` from cdc4a55): JAX initial points (no `pm.init_nuts`/NUTS compile), start-scoring cap at 4C rows, one vmapped ε0 dispatch with a fixed probe mask, and 0 PyTensor compiles on a warm call.
- 2026-10-23 — F2c+F2d (fast-tier). Merged with `--no-ff` (head e3694cd). Task suite:
  `413 passed, 51 warnings in 494.93s`. SUMMARY is in `results/f2/F2cd/`.
  - JAX initial points via PyMC's `make_initial_point_expression`, lifted. This replaces
    `pm.init_nuts`, which built and discarded a NUTS step.
  - A warm call makes 0 PyTensor compiles (was 3) and 0 XLA compiles.
  - Start scoring capped at 4C rows. Radon scoring went from 122,500 to 4,084 rows; its dispersion
    is unchanged.
  - One vmapped ε0 dispatch, bit-identical.
  - Draws are bit-identical to cdc4a55 on 8schools, gev and logit.
  - CPU warm wall: 8schools 0.99→0.53 s, gev 2.62→1.47 s, logit 3.70→3.29 s.
  - **Latent F2a bug fixed by the task:** `_key` lowered without `keep_unused`, so graphs differing
    only in unused arguments shared a key and a registry `fn`. With equal arity that could have
    produced silent wrong results.
  - Pre-existing, not fixed: `1 < pm_chains < chains` raises a broadcast `ValueError` in
    `_run_pathfinder`.
- 2026-10-23 — Lead suite on merged `f2` (9cd4c1f): `413 passed, 51 warnings in 494.25s (0:08:14)`.
- 2026-10-23 — Paused at the owner's request (system maintenance) after the F2c+F2d merge.
  - Next is F2e: an opt-in on-disk cache, plus a lift memo. The memo is a pre-rewrite structural hash,
    verified with `equal_computations` before reuse.
  - Then F2f: GPU measurement of main vs f2 at the M4 shapes, the R1 bias gate, and CPU NUTS first
    calls.
- 2026-10-23 20:15 — Resumed after the compaction. Per-call lift cost on fresh model objects (CPU,
  `scratchpad/f2e/lift_cost2.py`; the three lifts are logp, initial point and transform, summed):
  - rewrite: 0.10–0.35 s;
  - key lowering: 0.02–0.07 s;
  - widening: 0.02–0.06 s;
  - `model.logp()`: 0.02–0.05 s;
  - funcify: ≤ 0.02 s.
  `tools/warm_stages.py` builds a new model object on every call, so a per-model memo would not help
  it. The pre-rewrite memo skips the rewrite, the funcify and the key lowering.
- 2026-10-23 20:20 — F2e task launched (`~/pf/f2-e` from f6108ac): an opt-in on-disk JAX
  compilation cache, and a lift memo (pre-rewrite structural hash, verified with
  `equal_computations`).
- 2026-10-23 21:45 — F2e (fast-tier, one round). Merged with `--no-ff` (89e0682; task head 13ed9f0). Task suite:
  `453 passed, 60 warnings in 520.15s`. SUMMARY is in `results/f2/F2e/`.
  - **Lift memo:** pre-rewrite signature; every hit verified with `equal_computations`. The three lifts
    of a call on a rebuilt model went from 0.16–0.54 s to 0.05–0.14 s.
  - **CPU warm wall:** 8schools 0.51→0.39 s, gev 1.43→1.01 s, logit 3.27→3.19 s, radon 4.90→4.58 s.
  - **Disk cache:** `pfcheeshmc.enable_compilation_cache(path)`, opt-in and documented in the README. A
    fresh process with new data and priors gets 134 of 135 requests from the cache. First call on CPU:
    8schools 13.5→3.4 s, radon 16.9→9.5 s.
  - **Suite fragility:** at f6108ac the one-process suite ends at 61,028 memory mappings against
    `vm.max_map_count` 65,530. F2e's test module releases compiled programs, so the suite now ends at
    about 37,900. A `tests/conftest.py` teardown is the durable fix (not done).
- 2026-10-23 21:59 — Lead suite on merged `f2` (89e0682):
  `453 passed, 60 warnings in 523.25s (0:08:43)`.
- 2026-10-23 — F2f drivers added in `results/f2/drivers/`:
  - `f2f_gpu.sh`: `main` vs `f2`, same data and new data, disk cache cold and warm;
  - `run_step.sh` with `F2f_gate.cmds`: the R1 gate on the M4 default rows;
  - `f2f_cpu.sh` and `cpu_nuts_calls.py`: CPU NUTS per-call walls, PyTensor cache cold and warm;
  - `new_data.py`: any model's observed data permuted, the same shapes;
  - `tools/warm_stages.py --new-data S`.
