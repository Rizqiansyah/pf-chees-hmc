# Review R2-WP3a (code) — lead, replacing the fast-tier review (LEDGER handover)

Verdict: **APPROVE** (merged together with R2-WP3b as `r2/wp3`).

| checklist item | result | evidence (`tools/t3_cpu_nuts.py`) |
|---|---|---|
| T3 driver calls `score_ensemble`, not its own scorer | yes | `:114-117` `rb.score_ensemble(...)`, gate `rb.g_r1(...)` |
| `include_transformed` on, value-var names scored | yes | `:70`; names from `model.value_vars` `:209`; a missing name raises (`:97-103`) |
| gradients include tuning | yes | `:126-136` sums `n_steps` over `warmup_sample_stats` + `sample_stats`; raises if warmup group is missing; `discard_tuned_samples=False` `:69` |
| floor test fails if the square is removed | yes | task mutation (both squares removed in `r1_bias.bias`): 6 failed / 5 passed; `0.0441 <= 1.3*0.00045` |
| no harness default silently differs | n/a here | R2-WP3b scope (`r1_bias.py`) |

Notes:
- `core_seconds` = RUSAGE_SELF + RUSAGE_CHILDREN. That counts PyMC's worker processes only once they
  are reaped, which happens at the end of `pm.sample`. Acceptable; M5 also records wall.
- At 32 chains, draws=50 gives C·D = 1,600 < 2,000, so those M5 cells are `g_r1`-ineligible by
  construction. This is recorded as a plan fact, and the gate is unchanged.
- The first NUTS process per model can include a one-off PyTensor C compile. The lead runs
  one untimed warm-up cell per model before M5.
