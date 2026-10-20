# WP5 — statistical-validity review (capacity claim)

**Provenance: lead-executed, NOT independent.** Two tasks were assigned this review; the first died on
a full-suite timeout, the second produced a plausible report whose memory numbers exist in no row file. Every
figure below was printed in this session from `~/pf/wp5/results/runs/WP5__memprobe_before.jsonl` (80 rows)
and `WP5__accept.jsonl` (5 rows) by `~/scratch/wp5_peaks.py`.

## 1. What the rows support, and what they don't

The probe writes one row per stage from a fixed ladder
(`start → pymc_init → finite_inits → best_chain_score → pathfinder_solve → pf_pool_draw → stratified_scoring →
covariance → eps0 → warmup_scan → sampling_scan`), so a missing later stage means that stage failed.

| model | C | stages reached before dying | peak at death |
|---|---|---|---|
| 8 Schools | 1,024 | **all 11, incl. sampling_scan** | 2,314.1 MiB |
| GEV | 1,024 | **all 11** | 946.4 MiB |
| logistic | 1,024 | through `pathfinder_solve` | 581.4 MiB |
| radon NC | 64 | **all 11** | 2,575.7 MiB |
| radon NC | 128 / 256 / 1,024 | through `pathfinder_solve` | 652.7 / 652.9 / **653.5 MiB** |

So: **radon's pre-fix wall is the Pathfinder draw pool** (`pf_pool_draw` never appears at C ≥ 128), the
`chains × pf_draws` account is confirmed, and both retracted memory models stay dead. 8 Schools and GEV already
fitted at C=1,024 before the change — the change is about radon and logistic, not everything.

**Correction the report needed:** the peak figures in this table are `kwargs.peak_bytes_in_use`, i.e.
**process-cumulative**, and this box has no `reset_memory_stats()`. The radon C=1,024 "653.5 MiB" is therefore
the cumulative allocation at death (its failure message is a 5.36 GiB request), not the pool's steady-state
footprint — the pool-stage numbers are the smaller ones in the same file (64.4–66.0 MiB at
`best_chain_score`/`finite_inits`).

## 2. The acceptance rows certify capacity, not quality

`WP5__accept.jsonl` carries only `cold`, `wall_s`, `peak_mem_bytes`, `stage_s`, `divs_per_1k`, `note`.
**No `max_rhat`, no ESS, no b², no warmup divergence count** in any of the 5 rows.

| model | C | cold | warm | wall | peak |
|---|---|---|---|---|---|
| radon NC | 1,024 | yes | yes | 528.79 s / 42.94 s | 653.5 / 653.3 MiB |
| logistic | 1,024 | yes | — | 359.8 s | 581.4 MiB |
| GEV | 1,024 | yes | — | 50.37 s | 288.1 MiB |
| 8 Schools | 1,024 | yes | — | 38.72 s | 224.1 MiB |

Two hygiene problems with those numbers:
- the radon **warm** row's peak (653.3 MiB) is cumulative *including the 528.79 s cold run in the same
  process*, so it cannot be attributed to the warm run;
- quoting a warm wall next to a cold-inflated peak is exactly the mixing this run has been correcting.

**What the rows license:** "radon non-centered runs to completion at C=1,024 × D=10 on a 6 GB card."
**What they do not license:** "R1 at radon is correct." Quality at C=1,024 is established elsewhere — by T1's
b² rows, not by WP5 — and for radon those rows are unresolvable (§3e), so radon's *capacity* is solved and its
*validity* is still open.

## 3. The memory win that is real and under-reported

8 Schools at C=1,024: cumulative peak **2,314 MiB before → 224 MiB after** (`WP5__accept.jsonl`), a ~10×
reduction from trimming the warmup scan, on a model that already fitted. This is the most reusable result in
the branch and the SUMMARY does not state it as a before/after pair.

## 4. The 2 s init bar

Warm init 4.07 s, of which `_run_step_size_tune` (eps0) is 1.59 s and Pathfinder 2.47 s — the bar is not met
and 1.59 s of the miss belongs to another task's stage. Cold init is 25.9–45.1 s and compile-dominated, so the
bar only makes sense as a warm number. Ruling needed, not measurement.

## 5. Dead code that reads as load-bearing

`_score_chain` has no remaining callers after the change. That is the kind of thing that makes a later reader
believe a guard is active when it is not; delete it or note why it stays.

## Verdict

**BLOCKING:** none — the capacity claim is supported by the rows and independently observed in this run.
**NON-BLOCKING:** label every WP5 peak as process-cumulative; add `max_rhat`/`min_bulk_ess`/`b2_max` to the
accept rows so capacity and quality are not confused (they are different claims and only the first is proven);
restate the 8 Schools 10× memory win as a before/after pair; delete or annotate `_score_chain`; fix the
doubled `pixi run` in `tools/wp5_accept.sh`.
**NIT:** `peak_mem_bytes` is 0 in every probe row — the real values are in `kwargs.peak_bytes_in_use`, which is
easy to miss and produced at least one fabricated-looking figure from a task that read the wrong field.

> **RECOMMEND: merge** the capacity work as *capacity*, with the quality caveat attached at the top of its
> SUMMARY: radon at C=1,024 completes; radon at C=1,024 is not yet shown to be unbiased.
