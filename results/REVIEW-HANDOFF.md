# Review handoff — pf-r1 (2026-10-20)

Read `results/REPORT.md` for findings. This file is the audit surface: what each claim rests on, what was
independently re-derived, what is knowingly unmeasurable, and how to reproduce it.

**Tree.** `main` untouched at `ecd9ba1`; `pf-r1` carries the eight merged work packages. Nothing pushed.
Suite: **125 passed in 176.88 s** forced-CPU at `a9dc9be` (confirmed; commits after it touch docs and data only) — originally 122 at the merges, +3 added after: (122 at merge + 2 pool-cap row tests + 1 from splitting the capacity test when the pool default changed contract).

## 1. Every headline claim and the row file behind it

| claim | file | how to re-derive |
|---|---|---|
| metric orientation is a correctness failure (R̂ 12.1–13.1 vs 1.002; ESS 32 vs ~30k) | `WP1__gev_2x2.jsonl`, `T1__prodpool.jsonl`, fixture script `~/scratch/wp1_2x2b.py` | per-seed `max_rhat` / `min_bulk_ess` by arm; 4 arms × 3 seeds |
| permutation + covariance is the shippable combination: better gradients/ESS on 4/4 | `WP1__covarm.jsonl` (40), `WP1__rebaseline.jsonl` (142) | `grads_per_ess_worst`, warm rows, `rep==1`, median over 5 seeds. **Re-derived this session: 35.5/71.2/46.1/182.3 vs 58.2/73.4/151.7/501.9 — exact.** |
| permutation alone must not ship (1/4, breaks R̂ on GEV+radon) | same two files | same column; `max_rhat` 1.0465 (GEV), 1.0142 (radon) |
| ChEES raw `log T` iterate costs time, saves none | `WP2__lgrowth.jsonl` (32) | `wall_s` medians by (device, model, arm): logistic B 7.30 → C 7.86 s (+7.7 %), CPU 27.42 → 31.43 s (+14.6 %); A→C is 2.13× and that is the metric, not the iterate. **Re-derived.** |
| refresh fixes radon's bias and is rejected on cost | `WP4__reseed_ab.jsonl` (42), `WP4__r1.jsonl` (32) | per-seed `g_r1` verdicts; cells are 3 rows (3 seeds), **not** 12 |
| refresh runs at C=1,024; cost is gradients, not memory | `WP4__refresh_capacity.jsonl` (3) | b², `grads_per_chain_total`, `peak_mem_bytes` |
| production pool: fits at d ≤ 8, OOMs at d ≥ 25; cap is verdict-neutral | `T1__prodpool.jsonl` (15) | pool bytes = `chains × pf_draws × d × 8`; XLA's own over-limit number is 5,734,400,000 B = 4,096,000 × 175 × 8 |
| capacity before/after WP5 | `WP5__memprobe_before.jsonl` (80), `WP5__accept.jsonl` (5) | peaks are in `kwargs.peak_bytes_in_use` (bytes); per-stage seconds in `kwargs.stage_s` on 70/80 rows; row-level `wall_s` is null on all 80 |
| XLA command buffers: do not ship (−12 %) | `T5a__cmdbuf.jsonl` (10) | `wall_s` by (model, arm, cold). **Re-derived: +12.8 % / +12.2 %.** |
| CPU NUTS reference cannot resolve b² at these shapes | `T3__cpu_nuts.jsonl` (12) | `resolvable` field: 12/12 False. **Re-derived.** |
| warmup-output costs (stack, callback) | `T5b__stepcost.jsonl` | `t_warmup_s`, `t_with_stack_s`, `t_with_bar_s` |
| GEV `xi_interval__` starts under-dispersed | `START__dispersion.jsonl` | start sd ÷ reference sd per coordinate, kernel order asserted on both sides |
| pool-size default (three arms × 4 models, with ESS) | `T1__r2_poolcap3.jsonl` (12) + `T1__r2_poolcap.jsonl` (8, the earlier pass) | `grads_per_ess_worst`, `ess_s_worst`, `min_tail_ess`, `max_rhat` by (model, arm); all rows share sampler code (`git diff -- pfcheeshmc/`) |
| refresh runs at C=1,024; what it costs | `WP4__refresh_capacity.jsonl` (3) | `grads_per_chain_total`, `peak_mem_bytes`, `g_r1` |
| warmup-output cost (stack, callback) | `T5b__stepcost.jsonl` (5; radon NOT MEASURED, 2 rows marked superseded) | `t_warmup_s`, `t_with_stack_s`, `t_with_bar_s`, `gpu_other_procs`, `timing_void` |
| start dispersion vs reference | `START__dispersion.jsonl` (8) | `start_sd_over_ref_sd_coldest_first`, `n_coords_below_half`, kernel-order leaf assert |
| historical doc claims retracted / confounded | `results/WP9/CLAIM-INVENTORY.md` (40 rows), `results/reviews/` | every row carries `file:line` and the deciding row ids |

## 2. Scoring rules — do not invent a third

`g_r1(b², floor, C·D)` in `tools/r1_bias.py` is the run's gate: ineligible below C·D = 2,000; when
`floor > 0.005` a cell fails **only** if `b² > max(0.01, 3·floor)` and is otherwise **not resolvable**;
when `floor ≤ 0.005` it passes iff `b² < 0.01`.

- "b² < floor" is **not** the gate and must not be reported as PASS/FAIL. It is a diagnostic ratio.
- The floor is built from the **run's own** chain MCSE (`tools/r1_bias.py:268-270`), so two arms of the same
  model legitimately get different floors. That is construction, not instability.
- Nested R̂ is a differential-drift/multimodality alarm, never convergence evidence (§3e, LEDGER `d4779fa`
  and the common-mode fixture in `tests/test_nested_rhat.py`).

## 3. Known limits, stated before you find them

1. **ESS was null in every `r1_bias` row until `10ec746`.** Any R1/R2 efficiency number before that commit
   comes from `tools/wp1_rebaseline.py`, and rows from before it cannot be compared with T2 at all. The
   `T1__r2_poolcap.jsonl` file therefore contains a mix of ESS-bearing and null rows; the pool-cap decision
   is re-run uniformly into its own file.
2. **The 4,000-draw pool was only reached with the cap.** `pf_pool_cap` defaults to `None`; the uncapped
   production pool cannot run above d ≈ 8 on a 6 GB card, so the d = 25/175 production cells are *capped*
   cells, and `pf_draws = 1` remains the only fully-uncapped-arm-avoiding configuration for small d.
3. **Per-coordinate divergence velocity is unmeasurable on this tree.** WP5 projected the warmup output to
   `WarmupScanInfo(num_integration_steps, is_divergent, acceptance_rate)`; momentum and `proposal.*` are not
   returned anywhere. Naming the coordinate that diverges needs the untrimmed `HMCInfo` behind a flag.
4. **Two `ORDER__*` files feed a T2 row but their git shas are not in `pf-r1`'s history** (`80e39f2`,
   `d48a624`, `62049b2`). The numbers are auditable, not reproducible from `pf-r1`. `T1__r1_sweep` is at
   `164f449`/`574f960`, which *are* ancestors and carry the permutation.
5. **b² is seed-dominated at D = 10.** One seed per cell cannot rank arms; the noise floor ships in every
   row and the report's comparisons are medians over 3–5 seeds.
6. **Radon cannot be gated at the R1 shape at any warmup** — floors stay ≈ 0.005, so its cells are not
   passes, they are unresolvable. Fixing this needs more draws per chain (outside R1) or a better floor
   (reference MCSE + serial correlation), which is a plan change.
7. **The box clock jumped ~45 min backwards during this run** (LEDGER 14:35). Row `ts` values in that window
   are not monotonic; order events by `git_sha` and row files, not timestamps. Three jobs also reported
   "completed" while still running, so several row files were rebuilt per-cell.
8. **Review coverage is uneven, and labelled.** A4 wanted two independent reviews per merge; only one task
   produced claims that survived verification. Four branches carry lead-executed reviews, labelled
   as such at the top of each file. Every number in them was re-derived from rows — and that process refuted
   several of my own published statements, listed in §4.

## 4. Corrections made during the run (the failure mode to look for elsewhere)

| wrong claim | how it was caught |
|---|---|
| "~2.1× warmup saving from `iterate=raw`" | independent review: A→C moves orientation *and* iterate; isolating the iterate costs 7.7 % |
| "`WP2__lgrowth.jsonl` shows bimodal seeds" | it is 1 seed × 2 devices |
| "the before-probe has no wall data at all" | per-stage seconds exist in `kwargs.stage_s` on 70/80 rows; `peak_mem_bytes` is a 512-byte stub, not 0 |
| "no configuration may enable `metric_refresh`" | the seam was fixed at `f2055aa`; 13 tests pass and it now runs at C=1,024 |
| "the default flip is owner-gated / still precision" | `08357db` flipped it to `covariance` (owner-approved, `STATUS 503071a`) |
| PASS/FAIL scored as `b² < floor` | the shipped gate is `g_r1`; re-scoring changed several verdicts |
| "124 tests passed" and "12 rows/cell" | neither was read from its source; both wrong (`122 passed in 179.36 s`; cells are 3 rows) |
| "the re-seed factor is recorded in no row field"; "radon floor swings between identical cells" | both refuted from the rows: `metric_refresh_reseed=true/null`, and floors are arm-dependent by construction |

**The pattern:** claims inherited from a task's report or written earlier in the session and never re-read
against their source. Row files and the tree are the only authority here; `git_sha` in each row says which
code produced it.

## 5. Reproducing

```bash
# suite (CPU-only, ~3 min)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu pixi run pytest -q

# any timed GPU cell MUST be under the lock (A5); one process per cell (peak memory is process-cumulative)
tools/gpu_run.sh pixi run python tools/r1_bias.py --models radon_noncentered --chains 1024 --draws 10 \
  --warmups 50 --seeds 1 --repeat 1 --pf-draws 4000 --pf-pool-cap 4096 \
  --orientation covariance --iterate raw --arm my:arm --timed --out results/runs/MY.jsonl

# warmup-output cost split
tools/gpu_run.sh pixi run python tools/t5_step_cost.py --model gev --chains 32 --warmup 2000 --reps 3
# start dispersion vs the CPU NUTS reference
tools/gpu_run.sh pixi run python tools/gev_divergence_probe.py --models gev --chains 32 --pf-draws 1
```

Reference posteriors live outside git at `~/pf-runs/references/*.npz` (manifest `results/references/MANIFEST.json`);
**the sample arrays are the single authority**, the stat columns are display-only, and `mcse_x2__` is the MCSE
of `E[x²]`, not of the variance. `tools/bench_ladder.py` stubs the sampler globals on model load — `r1_bias`'s
`build_model` restores them and asserts it did, which is why harness code must not call the loader directly.

## 6. How tests were proven, and two traps in the harness itself

**Teeth checks changed form mid-run and the old recipe is now invalid.** The original check for
`tests/test_metric_orientation.py` ran the test file against the pre-fix checkout (`~/pf/legacy` @
`ecd9ba1`) with a `pfcheeshmc.__file__` assert. That no longer proves anything: the current test passes
kwargs (`imm_orientation`, `pf_pool_cap`) that legacy `_run_pathfinder` rejects, so it would fail with a
`TypeError` for reasons unrelated to ordering. **The correct form now is mutation on the current tree** —
neutralise the choke point and require the declared failure signature:

```python
import pfcheeshmc.sampler as sm
sm.PfCheesHMC._metric_to_kernel_order = lambda self, imm, perm: imm   # the historical bug
# -> test fails: diag [9.88e-5, 1.006e4, 1.001e4, 9.69e3] vs [1e4,1e4,1e4,1e-4], max rel err 9.7e7
```

**Timeouts must go inside the GPU lock.** `tools/gpu_run.sh` is `exec flock /tmp/pf-gpu.lock env
PF_GPU_LOCK=1 "$@"`, so `timeout N tools/gpu_run.sh CMD` charges lock wait to `N`. Two cells were lost that
way (one reported as a 43-minute hang that was only waiting). Correct form:
`tools/gpu_run.sh timeout N CMD`. Related: `flock` grants are **not FIFO**, so queued drivers do not start in
submission order — do not interleave a sampler commit with a driver that must run entirely on one code
version (the pool A/B rows were checked with `git diff <sha1> <sha2> -- pfcheeshmc/` to confirm they are
single-factor).

**Two failures of the same kind to look for elsewhere.** Twice in this run a change to a *default* or a gate
silently degraded a test's or a table's **statistical resolution** rather than its logic: the `g_r1` floor
masking, and the dimension-aware pool leaving a d=4 fixture with 64 rows (~18 % MC SE against a 20 %
tolerance) — which looked like a broken permutation fix. A green or red suite does not distinguish these;
the check is tolerance-vs-SE for every assertion consuming the affected quantity.
