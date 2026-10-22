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
