# STATUS — run 2 (live)

**Current step:** M3 (R2 matrix) running → M5-cpu → M4 (chained). M2b done; D3 = recommend OFF.
`results/r2/drivers/run_step.sh M1 results/r2/drivers/M1.cmds`, logs `~/pf-runs/r2-logs/M1/`.

**Done:** R2-WP0..WP4 merged into `pf-r2` (WP1 `07575bc`, WP2 `62c7620`, WP3 `353c02f`, WP4 `25d2618`).
Lead: the lead with fast/steady tasks, run one at a time from WP3b on (owner).

**Next:** D2 evaluation → M2a → M2b → (M2c if triggered) → M3 → D1 → M4; M5-cpu between GPU steps.

**Blocked:** nothing.

**Discipline in force (plan A3/A5):** GPU only via `tools/gpu_run.sh timeout N <cmd>`; one process per
(model, arm); quoted walls only from `--repeat 2` rep-1 rows; code frozen during a step (the driver
aborts if `pfcheeshmc/` or `tools/` change); `g_r1` is the gate, G-R1-pool a labelled diagnostic.

| branch | head | suite | GPU | pushed |
|---|---|---|---|---|
| `pf-r2` | `eee10ae` | re-run after WP3+WP4 merge (log `~/pf-runs/r2-logs/suite-after-wp3wp4.txt`) | M1 | never |
