# F2g SUMMARY: jitted start-repair check, `tests/conftest.py` teardown, on-disk cache on by default

Task: fast-tier, one round. It stopped once at the time limit before its first commit and was resumed.
- Branch `f2-g`: 4 commits on 087b54e, head eec7b17. Merged into `f2` with `--no-ff` (51e15cf).
- Task's suite: `496 passed, 62 warnings in 656.82s (0:10:56)`. It was 471: 25 new tests in `test_start_repair`
  (11 tests, 18 cases) and `test_compilation_cache` (8 → 33).
- Lead suite on merged `f2`: `496 passed, 62 warnings in 649.33s (0:10:49)` (51e15cf).

## Result
- **Start repair (item 1).** `_find_finite_start._is_usable` (`pfcheeshmc/sampler.py:372`) decides with
  `_usable_mask` on a one-row host stack, the jitted program that screened the starts. `_ensure_finite_inits`
  (`sampler.py:565`) reads the mask once, returns the stack untouched when every row passes, and repairs only the
  failing rows, row c under `split(key, chains)[c]` as before, on a host copy.
  - Draws are bit-identical to 087b54e (GEV default and `pm_chains=C`; six repair cases including a non-ladder
    `fence` model that goes through backtracking and restarts, and a last-resort origin start).
  - CPU, GEV, C=1,024: init of the first call 5.45 s / 164 compiles → 1.43 s / 19; warm init 0.13 → 0.02 s.
    First call 18.3 → 13.9 s.
  - Every model whose start needs repair benefits; nothing is model-specific. The eager check stays as a fallback
    if the jitted one cannot run.
- **`tests/conftest.py` (item 2).** A module-scoped autouse `gc.collect(); jax.clear_caches()`. Peak memory
  mappings in the one-process suite 37,831 → 15,717 (limit 65,530; f6108ac had 61,028). Cost: +73 s (+13%),
  because modules recompile what earlier ones had cached. No test relied on another module's programs.
- **Disk cache on by default (item 3).** `sample_pf_chees_hmc` calls `ensure_default_compilation_cache`
  (`pfcheeshmc/compilation_cache.py:59`) at its start.
  - Directory `${XDG_CACHE_HOME:-~/.cache}/pfcheeshmc/jax`, created 0700 (and its parent). Refused with a
    `RuntimeWarning`, never an error, if it belongs to another user, is group/other-writable or cannot be created.
  - Never over the user: a JAX cache directory already set (env var, `jax.config`, `set_cache_dir`) or
    `jax_enable_compilation_cache=False` means nothing is touched. Each threshold moves only if still at JAX's
    default. `jax_compilation_cache_max_size` is never set.
  - Opt-out `PFCHEESHMC_COMPILATION_CACHE=0` (`false`/`no`/`off`), read at the call. Import changes no JAX
    setting; the class API does not switch it on.
  - JAX takes the directory at the next compile even after earlier compiles (`is_cache_used` latches the
    platform and `jax_enable_compilation_cache`, not the directory): tested, no JAX internals used.
  - `enable_compilation_cache(path)` now calls JAX's public `reset_cache()` when it replaces a directory in use;
    without it an explicit choice after a sampler call was silently ignored.
  - `warm_stages`, `r1_bias`, `bench_ladder` and `init_probe` set the opt-out with `setdefault`, so "cold" rows
    stay cold; `warm_stages` prints which cache it ran with. The F2f drivers need no change: their "no cache" jobs
    go through these tools, and their cache jobs name a directory explicitly.
  - CPU, 8schools, first call in a fresh process: 13.8 s filling the default directory, then 3.4 s (same data)
    and 3.5 s (new data), 130 of 130 hits. 1.3 MB, 129 files.

## DEVIATIONS (accepted)
1. **No size cap on the default directory.** JAX 0.11.2 evicts correctly, but with a cap set `LRUCache.put`
   globs the directory and reads every entry's atime file on every write (`jax/_src/lru_cache.py`,
   `_evict_if_needed`; I read it: the scan runs whenever `eviction_enabled`, even under the cap). The task
   measured 33 µs per file per write: a new model's ~135 writes cost 4.5 s per 1,000 files already there. A cap
   big enough to be useful slows every new model's first call. Growth is 1.3–4.6 MB per model structure, shapes
   and chain count, not per dataset. The README documents size, `du`, clearing and setting JAX's own cap.
   **Owner to confirm** (one line adds a cap).
2. **Additions:** `reset_cache` in the explicit helper; thresholds moved only from JAX's defaults; the
   ownership/permission refusal; `pfcheeshmc.default_compilation_cache_dir()` exported; the banner in
   `tools/_common.py` prints both cache variables; new `tools/init_probe.py`.

## Tests and must-fail evidence
- `tests/test_start_repair.py` with `tests/start_repair_run.py`: compares the repaired stacks and GEV draws of
  this tree with `git archive 087b54e` in fresh processes (skips if the commit is not available). Old
  `sampler.py` restored: 5 of 18 cases fail (`assert 161 <= 40`, `usable_mask` compiled once instead of twice,
  the untouched-stack `is` check, the two nothing-to-repair cases). The override mutation (`_is_usable` ignoring
  an `ld_fn`/`_ld_fn` override) fails the override tests.
- `tests/test_compilation_cache.py` with `compilation_cache_run.py` and `compilation_cache_policy_run.py`: every
  default-cache child runs with HOME or XDG_CACHE_HOME in `tmp_path`. Nine mutations of `compilation_cache.py`,
  `sampler.py` and `__init__.py`, one at a time, each fail 1–7 tests (sampler not calling it; a user directory
  overridden; opt-out ignored; mode 0755; thresholds unset; switched on at import; no `reset_cache`; user
  thresholds overwritten; unsafe permissions accepted).
- The real `~/.cache/pfcheeshmc` does not exist after the lead's suite run (`ls: cannot access '/home/akbar/.cache/pfcheeshmc'`), nor after the GPU re-check.

## GPU re-check (lead)
Host: RTX A2000 6 GB, load 3–6 (the suite had just finished; F2f had 1–4), 2026-10-24 02:35–02:41.
Driver `results/f2/drivers/f2g_gpu.sh`, tables `results/f2/drivers/f2g_table.py`, rows `results/f2/runs/F2g/`,
logs `~/pf-runs/f2g/logs/`. Shapes as F2f: C=1,024, D=10, W = 25, 25, 50, 300. F2f's tree (3e0a739) has the same
library code as 087b54e.

**Same data, cache off (`--calls 3`), F2f → F2g.** Call 1 wall / XLA compiles; init in s / compiles; warm = mean
of calls 2–3.

| model | call 1 | init, call 1 | warm wall | warm init |
|---|---|---|---|---|
| 8schools | 17.2 s / 134 → 17.0 s / 130 | 0.58 / 10 → 0.36 / 4 | 0.23 → 0.23 s | 0.02 → 0.01 |
| gev | **34.0 s / 263 → 27.2 s / 121** | **8.98 / 164 → 2.47 / 19** | 0.55 → 0.47 s | **0.21 → 0.03** |
| logit_1000x25 | 31.6 s / 112 → 31.4 s / 109 | 0.67 / 7 → 0.50 / 3 | 1.43 → 1.40 s | 0.01 → 0.01 |
| radon_noncentered | 45.2 s / 139 → 44.5 s / 135 | 0.85 / 10 → 0.62 / 4 | 6.50 → 6.40 s | 0.04 → 0.03 |

- GEV's two warm calls were 0.58 and 0.36 s; the warm init is 0.03 s in both, so the spread is elsewhere.
- The other models lose the 3–6 eager row ops the old `_ensure_finite_inits` compiled for a row that passed.

**First call in a fresh process with new data, the disk cache warm: F2f explicit helper → F2g library default**
(`PFCHEESHMC_COMPILATION_CACHE=1`, fresh `XDG_CACHE_HOME`, filled by one `--new-data 0` process). Wall (XLA
compile s).

| model | fill, F2f → F2g | warm cache, new data, F2f → F2g | CPU NUTS, warm compiledir (F2f) |
|---|---|---|---|
| 8schools | 17.6 (14.7) → 17.6 (14.7) | 3.7 (1.0) → **3.6 (0.9)** | 4.2 |
| gev | 34.5 (28.9) → 28.5 (23.6) | 7.4 (2.0) → **6.2 (1.6)** | 5.9 |
| logit_1000x25 | 32.6 (28.6) → 31.9 (27.9) | 4.8 (0.9) → **4.9 (0.9)** | 10.0 |
| radon_noncentered | 42.7 (34.2) → 42.0 (33.5) | 9.8 (1.2) → **9.6 (1.2)** | 18.9 |

- Every compile request of the warm-cache process hits (130/130, 121/121, 109/109, 135/135); 0 written.
- The default equals the explicit helper on the GPU, which closes the task's GPU UNVERIFIED item.
- Directory 2.9–3.7 MB, 108–134 files, modes 700 on `pfcheeshmc/` and `jax/`. JAX adds
  `xla_gpu_per_fusion_autotune_cache_dir/<hash>/` under it, as the README says.
- GEV's warm-cache first call is still 0.3 s behind CPU NUTS (6.2 vs 5.9 s). Its init there is 0.69 s, against
  0.14–0.25 s for the others: the repair's remaining small programs (key split, jitter normal, ravel) are traced
  and loaded per process. Not pursued: they are kept eager for bit-identity, and 0.3 s on one model is noise-level
  against the run-to-run spread.

## Findings
1. **Pre-existing bug, not fixed:** `pathfinder_kwargs={"pm_chains": k}` with `1 < k < chains` raises
   `Incompatible types for broadcasting` at `sampler.py:707`, `jnp.broadcast_to(a, (chains,) + a.shape[1:])`,
   which is right only for one row. It fails on 087b54e too. A cyclic `take` would fix it without changing the
   default (k = 1).
2. **JAX cache writes without eviction are not atomic** (`write_bytes`). A process killed mid-write leaves a
   truncated entry that `put` never replaces; JAX warns on each read and compiles. The README says to delete the
   directory.
3. **The `with_new_data` driver clone is not structurally the plain model** for 14 of 130 programs (filling with
   the plain model, then calling with a clone: 116 hits). The F2f and F2g drivers fill with `--new-data 0` (a
   clone) so their rows are consistent. A user rebuilding the model with their own code is the plain case.
4. **The suite teardown costs 13%.** A threshold variant (clear only when the mapping count is high) would keep
   most of it; not done, the brief said every module.

## UNVERIFIED
- Several processes writing the default directory at once (read path catches errors; not tested).
- macOS and Windows locations (the `os.getuid` check is skipped where it does not exist).
- float32 end to end for the repair.
- `jax.experimental.compilation_cache.compilation_cache.reset_cache` staying public in later JAX (one call site,
  tested).
