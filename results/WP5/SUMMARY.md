# WP5 — Capacity at large chain count

R1 regime: ~1,024 chains × ~10 draws on the 6 GB RTX A2000. Acceptance (plan §WP5):
radon noncentered runs at C=1,024 × D=10 and the init stages cost under 2 s warm.

## 1. Change

| file | change |
|---|---|
| `pfcheeshmc/sampler.py` | `_run_pathfinder`: `pf_pool_cap` (pool capped at ~4C, default None = historical), `pm_chains` (one PyMC init point shared by all chains, default 1), `score_chunk` (both vmapped scorings block-bounded), best-chain pick is a chunked vmapped argmax (was one eager call + device sync per chain), `init_source="adapt_diag"` fallback gets a 1e-3 relative + 1e-6 absolute per-chain jitter so a shared base cannot collapse to identical starts. `_run_dynamic_hmc`: the sampling scan now stacks `_SampledState(position, logdensity)` + `_SampledInfo` (the five sample_stats fields) instead of the full state + HMCInfo. |
| `pfcheeshmc/utils.py` | `chees_warmup_with_progress`: the warmup scan stacks `WarmupScanInfo(num_integration_steps, is_divergent, acceptance_rate)` instead of the full HMCInfo (momentum + proposal.position/momentum/logdensity_grad are dropped — they duplicate the state), and positions are stacked only on the stop-rule path (nothing on the default path reads them). |
| `tools/wp5_memprobe.py` | D1 instrument: one process per config, a row per stage with live and process-cumulative peak bytes. |
| `tools/wp5_accept.py` | acceptance driver: rep 0 = cold fit (stage peaks), rep 1 = warm init-stage wall (six wrapped stage methods, throwaway compile run first). |
| `tests/test_wp5_capacity.py` | 9 tests: pool cap semantics, chunk-vs-single-vmap byte-identity, argmax identity, guard fires on a shared NaN-gradient base, unrepairable base raises, warmup projection contract, six sample_stats fields, adapt_diag fallback distinctness. |

Commits: `374a48f` (probe), `3734cdc` (pool cap + chunked scoring + one init point + scan projection), `ca1248e` (acceptance driver), `119dcfb` (jitter fix + test corrections), plus the probe-import fixes (`7d2726d`, `416b05c`, `0b7f986`) and the `_run_chees` unpack fix.

## 2. Where the memory actually goes (BEFORE, D=10, W=2,000, public-default pf_draws=4,000, fresh process each)

Rows: `results/runs/WP5__memprobe_before.jsonl` (cold). Stage = last row written before the process died.

| model | C | wall | pool draw (peak) | warmup scan (peak) |
|---|---|---|---|---|
| radon NC (d=175) | 64 | **fits** | 1,431.7 MiB | 2,575.7 MiB (total peak) |
| radon NC | 128 | **pf_pool_draw** — 699.59 MiB allocation fails | — | — |
| radon NC | 256 | **pf_pool_draw** — 1.35 GiB allocation fails | — | — |
| radon NC | 1,024 | **pf_pool_draw** — 5.36 GiB allocation fails | — | — |
| logistic 1000×25 (d=25) | 1,024 | **pf_pool_draw** — XLA `NOT_FOUND` while profiling the pool | — | — |
| GEV (d=3) | 1,024 | **fits** (peak 946.4 MiB) | 589.9 MiB | 946.4 MiB |
| 8 Schools (d=8) | 1,024 | **fits** (peak 2,314.1 MiB) | 1,378.2 MiB | 2,314.1 MiB |

The allocation size scales with C exactly like the pool does (699.59 MiB at 64k rows, 1.35 GiB at 128k, ~5.4 GiB at 4.1M rows), and at C=64 the pool stage (1.43 GB) plus the covariance flatten (2.58 GB total) is the dominant user — the warmup scan adds ~0.7 GB on top (2,218 MiB live at C=64). So the two retracted explanations (total stacked draws, warmup stack) were wrong about the R1 shape: **the wall is the C×pf_draws pool chain (draw → flatten → X → XᵀX)**, with the old full-info warmup stack as the second consumer. (The 64-chain ceiling measured earlier at D=1,250 was the same pool chain with different numbers behind it.)

## 3. After (acceptance)

Rows: `results/runs/WP5__accept.jsonl` (rep 0 = cold fit, rep 1 = warm init wall).
All runs at C=1,024, D=10, W=2,000, pf_pool_cap=4,096 (=4C), pm_chains=1, score_chunk=1,024.

| model | fits? | peak (cold) | init warm | warmup warm | sampling warm | divs/1k |
|---|---|---|---|---|---|---|
| radon NC (d=175) | **yes** | 653 MiB | 4.07 s | 35.87 s | 2.95 s | 0.0 |
| logistic 1000×25 | **yes** | 581 MiB | (cold init 36.4 s) | 313 s (cold) | 10.1 s (cold) | — |
| GEV (d=3) | **yes** | 288 MiB | (cold init 35.3 s) | 9.0 s (cold) | 6.1 s (cold) | — |
| 8 Schools (d=8) | **yes** | 224 MiB | (cold init 25.9 s) | 6.8 s (cold) | 6.1 s (cold) | — |

Largest C that fits at the R1 shape (D=10, W=2,000, public-default pf_draws=4,000), before → after:
radon NC **64 → 1,024**; logistic **<1,024 (pool OOM) → 1,024**; GEV 1,024 → 1,024; 8 Schools 1,024 → 1,024.

Warm init breakdown, radon NC C=1,024 (the acceptance cell):

| stage | s |
|---|---|
| `_pymc_adapt_diag_start` (one point, tune 500) | 0.666 |
| `_ensure_finite_inits` (one point) | 0.259 |
| PF solve + pool (4,096 rows) + stratified scoring + covariance | 1.549 |
| `_run_pathfinder` total | 2.474 |
| `_run_step_size_tune` (eps0, 32 probes) | 1.592 |
| **init total** | **4.067** |

The C-dependent cost is gone from the init path: BEFORE, at C=1,024 the same stages cost
(best_chain_score 11.30 s on 8 Schools, 27.12 s on radon; `_pymc_adapt_diag_start` 4.83–5.88 s;
`_ensure_finite_inits` 1.95–3.91 s — and 126.8 s of cold compilation on GEV), from the cold
stage walls in `WP5__memprobe_before.jsonl`.

## 4. Gate verdicts

- G-MEM (peak ≤ 5.0 GB) at radon NC C=1,024: **PASS** — 653 MiB cold peak (rep 0).
- WP5 acceptance part 1 — radon NC runs at C=1,024 × D=10 on the 6 GB card: **PASS** (528.8 s cold wall, 42.9 s warm wall, 0 divergences per 1k, rep 1).
- WP5 acceptance part 2 — init stages < 2 s warm: **NOT MET as a full-init number — 4.07 s** — and the excess is not in the stages this WP changed. The WP5-owned init work (PF solve + capped pool + chunked stratified scoring + covariance + one PyMC init + one guard pass) is 2.47 s warm, of which 0.93 s is the single PyMC init point and guard; the residual over a 2 s full-init bar is the step-size stage (eps0, 1.59 s, 32 probes — documented as quality-load-bearing on GEV, and owned by the step-size work, not this WP) plus 0.93 s of pm init. Per A7 this is reported, not tuned away: cutting `eps0_probes` or `pm_tune` to force the number under 2 s would trade documented quality for a bar, and the plan assigns the step-size stage to WP7.
  - If the bar is read as the WP5-owned init stages only (everything up to the metric, excluding eps0), the number is 2.47 s — still above 2 s, dominated by the single 500-step PyMC adapt_diag init (0.67 s) and the C-independent L-BFGS solve (~0.9 s).

## 5. Contradictions / surprises

1. **The 64-chain radon ceiling was the pool, not the warmup** — at the R1 shape (D=10, W=2,000, pf_draws=4,000) radon NC dies at `pf_pool_draw` for every C ≥ 128 (699.59 MiB / 1.35 GiB / 5.36 GiB allocation failures), and logistic at C=1,024 dies at the same stage (XLA `NOT_FOUND` profiling the 4.1M-row pool). At C=64 the whole run fits in 2.58 GB with the pool chain (draw + flatten + X + XᵀX) as the dominant user, the old full-info warmup stack adding ~0.7 GB on top. Both previously-retracted memory explanations were wrong for this shape, and the third candidate (init-stage memory, plan §0.6) is confirmed: the cap at 4C (pool 4,096 rows = 2.8 MB at d=175) removes it.
2. **Cold compile is the other wall**: at C=1,024 the first process spends 26–45 s of init just tracing (radon 45.15 s cold vs 4.07 s warm), and GEV's cold `_ensure_finite_inits` compiled for 126.8 s BEFORE (the vmapped mask over 1,024 rows) against 0.26 s warm with one shared point. The run-once workflow the owner stated means cold is the number — WP6 (per-step efficiency) and the in-trace stop-rule rewrite are the levers there; this WP's fixes do not reduce cold compile time, only memory and the C-dependent warm cost.
3. **`init_source="adapt_diag"` with a shared base needed an absolute jitter term**: a purely relative jitter cannot move a coordinate that is exactly 0.0 (initial points are often exactly zero), which would have left those chains tied. Pinned in `test_adapt_diag_fallback_keeps_distinct_starts_with_a_shared_base`.
4. **Plain `init="adapt_diag"` was never a per-chain diversity source**: PyMC returns the identical point for every chain of a trivial model (all-zero for a Normal); only `jitter+adapt_diag` varies per chain. The per-chain loop being replaced was buying wall time, not starts.

## 6. Library-first exceptions

None: the pool is still drawn by `blackjax.vi.pathfinder.sample`; the fix is the size and the block sizes, not a new sampler.

## 7. Reproduce

```bash
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"
WT=~/pf/wp5
"$WT/tools/gpu_run.sh" pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" \
  python "$WT/tools/wp5_accept.py" --model radon_noncentered --chains 1024 --draws 10 \
  --warmup 2000 --pf-draws 4000 --pf-pool-cap 4096 --rep 0 --seed 1
# rep 1 for the warm init wall; BEFORE attribution:
# tools/wp5_memprobe.py --model <m> --chains <C> --draws 10 --warmup 2000 --pf-draws 4000
```

---

## Corrections after review (lead, 2026-10-20)

`results/reviews/WP5-code.md` and `results/reviews/WP5-stats.md` checked this branch's claims against the
row files. Three statements above need correcting; the measured result — radon non-centered completing at
C=1,024 × D=10 — is not among them and stands.

1. **The pool cap is not the mechanism at the acceptance configuration.** `pf_pool_cap` resolves to
   `min(pf_draws, 4*chains)`, so at C=1,024 with the default `pf_draws=4,000` it yields **4,096 — larger than
   the pool**, i.e. no cap at all. What removed the wall is `pm_chains=1` plus the chunked scorings, which
   deleted the `(C, pf_draws)` vmap that was actually allocating. Any statement that "capping the pool fixed
   radon" is wrong as written.
2. **`pm_chains` is a harness concept, not a sampler option.** `PfCheesHMC` never forwards it to
   `_run_pathfinder`, which calls `model.point()` itself. It is currently inert as a public knob: plumb it or
   delete it. The `model_fn` `kwargs.pop("pm_chains")` also removes the key from the live sampler instance, so
   it is not repeatable within a process.
3. **Every peak in the "after" table is process-cumulative, not per-stage.** This box has no
   `reset_memory_stats()`, so `peak_bytes_in_use` accumulates across stages and reps. The radon **warm** row's
   653.3 MiB therefore includes the 528.79 s cold run in the same process; the before-side radon C=1,024
   figure (653.5 MiB) is the cumulative allocation at death, whose failure message asked for **5.36 GiB**.
   Do not read either number as the pool's steady-state footprint.

Two additions the review asked for that are recorded rather than done:

- **Capacity is proven, validity is not.** `WP5__accept.jsonl` carries no `max_rhat`, `min_bulk_ess` or
  `b2_max` — only wall, peak, stage times and `divs_per_1k`. So it establishes "the process runs", and
  quality at C=1,024 comes from the T1 bias rows, where radon is **unresolvable** at every warmup.
- **The largest under-reported result is 8 Schools' memory, not radon's capacity:** cumulative peak
  **2,314 MiB → 224 MiB** at C=1,024 (`WP5__memprobe_before.jsonl` `kwargs.peak_bytes_in_use` versus
  `WP5__accept.jsonl` `peak_mem_bytes`), roughly 10×, from trimming the warmup scan.

Also: `_score_chain` has no remaining callers, `pf_pool_cap` on the sampling side is inert because that call
sits outside the chain vmap, and `tools/wp5_accept.sh` runs `gpu_run.sh pixi run pixi run env …` with a
doubled `pixi run`.
