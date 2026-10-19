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

Rows: `results/runs/WP5__accept.jsonl` (cold fit, warm init wall).
Before/after, largest C that fits at the R1 shape (D=10, W=2,000, public-default pf_draws=4,000): radon NC 64 → (after); logistic (after, OOM before); GEV 1,024 → 1,024; 8 Schools 1,024 → 1,024.

| model | C | D | W | fits? | init stages warm | peak (cold) |
|---|---|---|---|---|---|---|
| radon NC | 1,024 | 10 | 2,000 | | | |
| logistic | 1,024 | 10 | 2,000 | | | |
| GEV | 1,024 | 10 | 2,000 | | | |
| 8 Schools | 1,024 | 10 | 2,000 | | | |

(To be filled from the rows when the GPU queue completes.)

## 4. Gate verdicts

- G-MEM (peak ≤ 5.0 GB): (pending after rows)
- WP5 acceptance (radon NC C=1,024 × D=10 fits; init < 2 s warm): (pending)

## 5. Contradictions / surprises

- (filled below)

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
