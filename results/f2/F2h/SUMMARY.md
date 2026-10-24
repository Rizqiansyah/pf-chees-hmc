# F2h SUMMARY: `pm_chains` points shared cyclically (fixes `1 < pm_chains < chains`)

Task: fast-tier, one round.
- Branch `f2-h`: 3 commits on 2605245, head edc227f. Merged into `f2` with `--no-ff` (82c8b2c).
- Task's suite: `514 passed, 63 warnings in 724.36s (0:12:04)`. It was 496; this adds 18 in `tests/test_pm_chains.py`,
  with child runner `tests/pm_chains_run.py`.
- Lead suite on merged `f2`: SUITE_PLACEHOLDER

## Result
- **The bug:** `_run_pathfinder` broadcast the repaired `(pm_chains, ...)` stack to `(chains, ...)` with
  `jnp.broadcast_to`. That is right only for one row, so `1 < pm_chains < chains` raised `Incompatible types for
  broadcasting`. It failed before F2 too.
- **The fix** (`pfcheeshmc/sampler.py:695–715`):
  - `_ensure_finite_inits` runs once on the `pm_chains` rows, for every `pm_chains`; the two branches are collapsed.
    Each unusable row is repaired under `split(key_fix, pm_chains)[row]`, as before.
  - If `pm_chains < chains`, chain c takes row `c % pm_chains`. The gather is `np.take` on the host, then one
    `jax.device_put`, which compiles no XLA program. The old `broadcast_to` compiled one, and `jnp.take` would
    compile three.
  - Cold compiles of `_run_pathfinder` on GEV, 4 chains: 95 → 94 at the default (one `jit(broadcast_in_dim)` fewer);
    97 → 97 at `pm_chains = chains`.
- **Bit-identical draws** against 80f0916 at `pm_chains = 1` and `pm_chains = chains`, with both `init_source`
  values and with `jitter+adapt_diag`. Every `posterior/*` and `sample_stats/*` array and the config are equal.
- **Audit of the rest of the init path:** OK, nothing else assumed one row.
  - The `init_source="adapt_diag"` jitter is drawn per chain over the tiled stack, so chains sharing a row still get
    distinct starts. Only its comment changed.
  - The start pick scores all `chains` tiled rows, which is redundant but correct. Scoring only the unique rows would
    change a compiled shape.
  - `config["pm_chains"]` is the number of PyMC points built.
  - `_jax_initial_points` broadcasts the single un-jittered base point to `chains`, which is correct for any count.
- **Docs:** the `_run_pathfinder` and `sample_pf_chees_hmc` docstrings state the cyclic meaning. The README does not
  mention `pm_chains`.

## DEVIATIONS (accepted)
- A host gather (`np.take` + `device_put`) instead of the brief's `jnp.take`/`jnp.tile`. The reason is above: it
  compiles nothing. On the GPU it costs a copy of `pm_chains` rows to the host and of `chains` rows back.
  GPU_PLACEHOLDER

## Tests (`tests/test_pm_chains.py`, 18)
- **Cyclic assignment:** `pm_chains` ∈ {1, 2, 3, 5, 7, 8}, with chains = 8 and `jitter+adapt_diag` (the PyMC points
  differ, which the test checks: pairwise > 0.05 apart).
  - The stack handed to the step-2 scoring equals `repaired[c % pm_chains]` exactly.
  - The Pathfinder start is the highest-logp row.
  - Each chain's final start is nearest its own row.
- **End to end:** `sample_pf_chees_hmc`, `pm_chains=3`, chains = 8, both `init_source` values. Finite draws, and
  `config["pm_chains"] == 3`.
- **One repair, shared:** a NaN-gradient row is repaired once, with key `split(split(key(7), 5)[0], 3)[1]`, and is
  shared by chains 1, 4 and 7.
- **Bad values** `pm_chains` ∈ {0, 9, -2} are refused.
- **Bit-identity** against `git archive 80f0916` (4 cases) and **no new compiled program** at the Pathfinder stage.
- **Must-fail evidence:**
  - With the old `sampler.py`: `7 failed, 11 passed`, with `ValueError: Incompatible types for broadcasting: input
    type=float64[3] and requested type=float64[8]`.
  - With a block assignment mutation (`c * pm_chains // chains`): the same 7 fail.
- **Lead:** a stale comment in `tests/test_start_repair.py` ("1 < pm_chains < chains fails") now points to
  this file (44a0556).
