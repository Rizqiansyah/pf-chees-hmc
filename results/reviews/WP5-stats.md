# WP5 — statistical-validity review of the R1 capacity work

Provenance: executed directly by the lead. Reviewer-model tasks on this box have no shell, and four
prior review tasks this run delivered numbers that exist in no file. Every figure below is printed by a
command run in this session (`~/scratch/wp5_stats_numbers{,2,3}.py`), and the two row files were enumerated
before any claim about them.

**Row files that exist** (`ls /home/akbar/pf/wp5/results/runs/`, confirmed this session):
`WP5__memprobe_before.jsonl` (80 rows) and `WP5__accept.jsonl` (5 rows). Nothing else is WP5 evidence.
The WP0/WP1/WP2/WP3 files in the same directory are inherited from the base tree. **There is no after-probe
file and no WP5 b²-scoped row.** That is established by enumeration, not assumption.

Row-level facts that shape everything below:
- `WP5__memprobe_before.jsonl`: row-level `wall_s` is **null on all 80 rows**. Per-stage seconds live in
  `kwargs.stage_s` (present on 70/80 rows). Per-stage memory lives in `kwargs.bytes_in_use` /
  `kwargs.peak_bytes_in_use` (80/80). The row's own `peak_mem_bytes` is 512 (a stub value), so it must not be
  read. `kwargs.pf_draws = 4000` on every before row (production pool).
- `WP5__accept.jsonl`: 5 rows — radon rep 0 (cold) and rep 1 (warm), plus GEV / 8 Schools / logistic cold
  only. Every row carries `wall_s`, `stage_s`, `peak_mem_bytes`, and `note="C=1,024 fit: yes"`.
- **`peak_bytes_in_use` on this box is process-cumulative** (no `reset_memory_stats()`), and
  `tools/wp5_accept.py` runs **two processes per config, never mixed**: rep 0 is a fresh process (memory
  valid only cold); rep 1 warms the compiler with one full throwaway `sample()` in the same process and then
  times the stages. So a rep-1 peak includes the throwaway run.

---

## Q1. Before/after peaks, recomputed, and SUMMARY vs rows

All before-side cumulative peaks recomputed from `kwargs.peak_bytes_in_use` grouped by (model, chains):

| model | C | before rows | before last stage | before cum peak | before completes? | after cum peak | after completes? |
|---|---|---|---|---|---|---|---|
| radon NC | 64 | 20 | `sampling_scan` | **2,575.7 MiB** | yes | — | — |
| radon NC | 128 | 5 | `pathfinder_solve` | 652.7 MiB | **no** (crash by absence) | — | — |
| radon NC | 256 | 5 | `pathfinder_solve` | 652.9 MiB | **no** | — | — |
| radon NC | 1,024 | 5 | `pathfinder_solve` | 653.5 MiB | **no** | 653.5 (cold) / 653.3 (warm) | **yes** |
| logistic | 1,024 | 5 | `pathfinder_solve` | 581.4 MiB | **no** | 581.4 | **yes** |
| GEV | 1,024 | 20 | `sampling_scan` | 946.4 MiB | yes | 288.1 | yes |
| 8 Schools | 1,024 | 20 | `sampling_scan` | 2,314.1 MiB | yes | 224.1 | yes |

The supportable before→after comparison, per model:
- **radon NC 64 → 1,024**: before, C≥128 dies at `pathfinder_solve` (5 rows, no `sampling_scan`); after, the
  full run completes. This is the acceptance claim and it is real.
- **8 Schools 2,314.1 → 224.1 MiB** at C=1,024, a **10.3× reduction** on a config that already fit, from
  trimming the warmup scan (the full-HMCInfo stack was the second consumer).
- **GEV 946.4 → 288.1 MiB** (3.3×).
- **logistic 581.4 MiB before-death → 581.4 MiB after-complete**: before it crashed at that footprint trying
  to grow the pool; after it completes with the peak set by the pathfinder stage.

**Every number in the SUMMARY's before/after tables appears in the rows.** I checked all of them: the
before-side memory figures (1,431.7 / 2,218 / 589.9 / 1,378.2 / 946.4 / 2,314.1 / 2,575.7 / 653.5 MiB) and the
before-side per-stage walls (11.30, 27.12, 4.83, 5.88, 1.95, 3.91, 126.8 s — these sit in `kwargs.stage_s`,
not row-level `wall_s`, which is why a naive read finds no before wall). None is a number that exists only in
the prose. **No SUMMARY memory/wall claim is unsupported by a row.** The one presentation defect (flagged
correctly by the code review) is labelling: every "after" peak is process-cumulative, not per-stage, and must
not be read as the pool's steady-state footprint.

## Q2. Is the radon C=1,024 claim measurable, or only the C=64 one?

- **Measurable and established by rows:** that the process **completes** at C=1,024 × D=10. Both before (5
  rows, dying at `pathfinder_solve`) and after (rep 0 cold 528.79 s, rep 1 warm 42.94 s, `note="C=1,024 fit:
  yes"`) are present, so this is a before/after on the same shape, not an inference from the C=64 row.
- **Inferred from absence:** the before-side *failure* (that pre-WP5 radon cannot run C≥128) is read from the
  five short row-blocks stopping at `pathfinder_solve` with no `sampling_scan`. That is the correct reading,
  and the code review's caution stands: "stage X failed" is inferred from the last row written, not recorded.
- **What is NOT established by the WP5 rows:** that the C=1,024 output is statistically valid. The accept rows
  carry no R̂, no ESS, no b² (Q5). So "the fix works" is precisely "the process now completes and fits at
  C=1,024" — a capacity statement. It is not, and the rows do not support it being, a "the samples are
  correct" statement.

## Q3. Cumulative-peak hygiene: the radon warm row

The warm row is **rep 1**, a separate process that (per `tools/wp5_accept.py`) first runs one full throwaway
`sample()` to compile, then times the stages — all in that one process. `peak_bytes_in_use` is cumulative, so
the warm row's **653.3 MiB includes the throwaway full run**, not just the timed warm run. It is not
attributable to the warm run alone, and it is not comparable to the cold row's 653.5 MiB as if they were
independent measurements.

**Strongest defensible memory statement from these rows:** *a single process that compiles and runs one full
C=1,024 × D=10 radon warmup-plus-sampling peaks at 653.5 MiB cumulative, leaving ~5.3 GB headroom on the 6 GB
card.* That is a genuine capacity bound. It is not a per-stage or per-run footprint, and it must not be quoted
as "the warm run uses 653 MiB."

## Q4. init 4.07 s vs the 2 s bar

Warm init on the acceptance cell (radon C=1,024, rep 1), from `stage_s`:

| stage | s | owner |
|---|---|---|
| `_pymc_adapt_diag_start` (one point, tune 500) | 0.666 | inside pathfinder stage |
| `_ensure_finite_inits` (one point) | 0.259 | inside pathfinder stage |
| `_run_pathfinder` (solve + capped pool + chunked scoring + covariance) | 2.474 | **WP5** |
| `_run_step_size_tune` (eps0, 32 probes) | 1.592 | step-size work, not WP5 |
| **`init_total` (= pathfinder + eps0)** | **4.067** | — |

- `init_total` = `_run_pathfinder` + `_run_step_size_tune` = 2.474 + 1.592 = 4.066 ≈ 4.067. The PyMC init and
  guard are *inside* the pathfinder stage, not additive on top.
- **WP5-owned init is 2.474 s — still 0.47 s over the 2 s bar**, even excluding eps0. The full-init number is
  4.067 s, and 1.592 s of that 2.07 s excess belongs to a stage WP5 does not own.

**The bar is ill-posed as written, for three reasons the rows show:**
1. It bundles WP5's pathfinder init with the step-size work's eps0 into one 2 s budget, so WP5 can be judged
   down for a stage it does not own.
2. It is warm-only, but this is a run-once workflow where the **cold** init is what is actually paid: radon
   cold init_total is **45.147 s** (GEV 35.265, logistic 36.42, 8 Schools 25.917), dominated by tracing/compile.
   A 2 s warm bar measures a number the owner never sees in the run-once case.
3. Even the WP5-only pathfinder stage (2.474 s) cannot get under 2 s on this card once the single 500-step
   PyMC adapt_diag init (0.67 s) and the C-independent L-BFGS solve (~0.9 s) are in — those are not tunable
   without trading documented quality.

**Position:** restate the bar as *"the C-dependent part of warm init is removed,"* which is what WP5 actually
achieves and which the before `kwargs.stage_s` prove (best_chain_score 11.30 s on 8 Schools and 27.12 s on
radon at C=1,024 → gone; `_pymc_adapt_diag_start` 4.83–5.88 s → 0.666 s; `_ensure_finite_inits` 1.95–3.91 s →
0.259 s). An absolute 2 s total-init bar is the wrong quantity for a cold-dominated, run-once workflow and it
mixes two work packages. Report, don't tune: the branch correctly does not cut `eps0_probes` or `pm_tune` to
force the number.

## Q5. Does the acceptance run report any convergence?

**No.** Across all 5 accept rows: `max_rhat`, `min_bulk_ess`, `min_tail_ess`, `b2_max`, `b2_floor`,
`ebfmi_min`, `nested_rhat_max` are **all null**; `warmup_divs` and `grads_per_chain_total` are null; the only
quality field populated is `divs_per_1k = 0.0` (radon only).

So the accept rows license **capacity** — "the process runs at C=1,024 × D=10, fits in memory, completes
warmup + sampling, and (radon) produces no sampling divergences." They do **not** license **validity** —
"the C=1,024 × D=10 output is unbiased or converged." R1 must be judged by b² against the CPU NUTS reference,
and there is no b² here.

**The load-bearing gap — confirmed by reading the only C=1,024 quality rows, and already flagged in the
report (§7 item 6, line 251/268) but still open:** the T1 bias sweep
(`~/pf/int/results/runs/T1__r1_sweep.jsonl`, 48 rows — the source of every R1 pass/fail/unresolvable verdict
in the report) was run with `pathfinder_kwargs.draws = 1`. That is **one Pathfinder draw**: a degenerate
start pool, not the production `pf_draws=4,000`. WP5's entire premise is that `pf_draws=1` "is NOT the
production start policy." Therefore:

- The R1 quality conclusions (8 Schools / GEV / logistic pass G-R1 at W=50; radon unresolvable at every W)
  were measured under a **one-draw start pool**, and
- WP5, which is what makes the **production** pool fit at C=1,024, has **not re-scored b² under that
  production pool**.

No delivered row combines "C=1,024" + "b²" + "production start policy." That is the missing measurement, and it
matters because the start pool is exactly the quantity the R1 bias gate is sensitive to (the whole R1 deficit
is a Pathfinder-start deficit).

## Q6. Cheapest additional row that makes the acceptance claim airtight

**One row, four models, warm, on the `wp/wp5` tree:** run the WP3 bias harness
(`tools/r1_bias.py`, already present on the merged base) at the acceptance shape — C=1,024, D=10, W=50,
1 seed, warm — with the **production** start policy (`pathfinder_kwargs.draws=4000`, `pf_pool_cap=4C`,
`pm_chains=1`, `score_chunk`), and score b² against the CPU NUTS references.

- What it gives: `max_rhat`, `min_bulk_ess`, `b2_max`, `b2_floor`, `divs_per_1k` at C=1,024 × D=10 under
  production starts, on the tree that has WP5's changes. It is a direct control against T1 (identical shape,
  only the start pool differs: 4,000 draws vs 1), so it isolates whether WP5's capacity machinery — shared
  init point, capped pool, chunked scoring — changes the statistics.
- Cost: from the accept rows, warm sampling is 2.95 s (radon) to ~10 s (logistic) and warm pathfinder 2.47 s
  (radon) to ~4 s; warmup at W=50 is a fifth of the W=2,000 warmup. Realistic total **~15–30 min GPU wall,
  four single processes** (one per model, per the run's one-process-per-cell rule for clean peaks).

If that row reproduces the T1 verdicts (3 models pass at W=50, radon unresolvable), the R1 conclusions are
confirmed under production starts and WP5's capacity claim is airtight. If it does not — if the production
pool moves b² relative to the one-draw pool — then the report's R1 table must be re-labelled as
"measured under a one-draw start pool," which is a material correction.

---

## Verdict

**BLOCKING (for the run's R1 conclusions, not for the WP5 branch itself):** the only C=1,024 statistical rows
(T1) were measured under `pathfinder_kwargs.draws=1`, a one-draw start pool, and the production pool that WP5
makes fit has never been b²-scored at C=1,024. Until the Q6 row is run, "radon unresolvable / three models pass
at W=50" are one-draw-pool results, and the report's R1 table should be read that way.

**NON-BLOCKING (WP5-merge):**
1. The 2 s warm-init bar is ill-posed — it bundles WP5's pathfinder init (2.474 s, 0.47 s over) with the
   step-size work's eps0 (1.592 s), is warm-only against a cold-dominated run-once workflow (cold init 45.1 s),
   and cannot be met without trading documented quality. Restate it as "C-dependent warm init removed," which
   the before `kwargs.stage_s` prove.
2. Every "after" peak is process-cumulative (warm rep-1 peak includes the throwaway compile run). Quote the
   capacity bound, not a per-run footprint.
3. `pm_chains` is not plumbed into the sampler (harness concept), and `pf_pool_cap=4096` is effectively no cap
   at C=1,024 / pf_draws=4,000 — the real mechanism is `pm_chains=1` + the chunked scorings deleting the
   `(C, pf_draws)` vmap. The branch's own "the cap fixed it" framing is wrong (already in the corrections).

**NIT:** the before-side per-stage walls live in `kwargs.stage_s`, not row-level `wall_s` (null on all 80
before rows) — a reader scanning the wrong column concludes "no before timings" and overstates the
unsupported-claims case; both are true at once.

**RECOMMEND: merge** the WP5 capacity branch — its acceptance criterion (radon completes at C=1,024 × D=10 on
the 6 GB card) is established by the before/after rows and none of the memory/wall claims is unsupported —
**conditional on running the Q6 row**, because that row decides whether the report's existing R1 quality
verdicts (measured on a one-draw start pool) transfer to the production pool WP5 enables. The merge and the
measurement are independent; the measurement gates the *conclusions drawn from* the merged tree, not the tree.
