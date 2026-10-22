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
