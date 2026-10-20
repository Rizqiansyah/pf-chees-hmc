# STATUS — run 2 (live)

**Current step:** handover to Code lead (see LEDGER). Wave 1 tasks: R2-WP1 (finish the task), R2-WP2, R2-WP3a, R2-WP4; baseline suite re-run.
`tools/legacy/t3_cpu.py` quarantined behind an execution guard. Baseline suite running on `pf-r2`.

**Next:** Wave 1 tasks — R2-WP1 (fast-tier, re-seed fix), R2-WP2 (fast-tier, closed-form Pathfinder
covariance + `pf_start_scale` + `attrs["config"]`), R2-WP4 (steady-tier, errata). Then R2-WP3 (fast-tier,
harness + T3 driver) and one code review per WP as a fast-tier slot frees.

**Blocked:** nothing.

**Discipline in force (plan A3/A5):**

- GPU only via `tools/gpu_run.sh timeout N <cmd>` — the timeout goes *inside* the lock.
- One process per (model, arm) for any row whose peak memory or warm wall-clock is quoted.
- Quoted wall clocks come only from a `--repeat 2` rep-1 row; no clock from a `--repeat 1`, cold or
  contended row.
- Code frozen during a measurement step: no merge into `pf-r2` while any row of that step is queued or
  running, because `flock` grants are not FIFO.
- Before comparing two row sets, assert `git diff --name-only <sha1> <sha2> -- pfcheeshmc tools` is empty,
  or name the difference.
- `g_r1` is the gate. G-R1-pool is a labelled diagnostic, never a gate. "b² < floor" is never reported as
  PASS/FAIL.

| branch | head | suite | GPU | pushed |
|---|---|---|---|---|
| `pf-r2` (base `pf-r1` `cca5407`) | `934e576` | baseline running | idle | never |
