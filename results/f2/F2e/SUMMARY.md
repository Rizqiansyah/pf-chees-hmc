# F2e SUMMARY: opt-in on-disk compilation cache and lift memo

Task: fast-tier, one round.
- Branch `f2-e`: 6 commits on f6108ac, head 13ed9f0. Merged into `f2` with `--no-ff` (89e0682).
- Task's suite: `453 passed, 60 warnings in 520.15s (0:08:40)`. It was 413; this adds 32 tests in
  `test_lift_memo` and 8 in `test_compilation_cache`.

## Result
- **Lift memo.** A model rebuilt around new data or prior values at the same structure no longer
  reruns PyTensor's JAX rewrite, the funcify or the key lowering. The three lifts of one call
  (logp, initial point, transform) now cost:

  | model | f6108ac | f2-e |
  |---|---|---|
  | 8schools | 0.208 s | 0.070 s |
  | gev | 0.536 s | 0.114 s |
  | logit | 0.162 s | 0.049 s |
  | radon | 0.393 s | 0.139 s |

- **Disk cache.** `pfcheeshmc.enable_compilation_cache(path)` is opt-in and off by default. A fresh
  process with new data and priors at the same shapes is served from the cache: 134 of 135 requests
  hit.
  - The first call on CPU, 8schools: 13.5 → 3.4 s.
  - The first call on CPU, radon: 16.9 → 9.5 s.

## DEVIATIONS (accepted)
1. **The ladder hit test perturbs lifted values in place** (`tests/test_lift_memo.py::perturb_values`,
   x·1.01 + 0.003).
   - The ladder builders cannot be rebuilt with other values.
   - Only what rule C lifts is touched: RV parameter constants, observed arrays, array constants and
     `pm.Data`.
   - The model's own static scalar expression constants are left alone; changing GEV's `0.10536`
     correctly misses.
2. **Min entry size is -1, not 0.** In jax 0.11.2 the two are equivalent:
   `default_min_cache_entry_size()` is 0. -1 is JAX's documented "no restriction, no platform
   override".
3. **One `fn` can now carry several keys.** A `pm.Data`-sized RV at N=30 and at N=31 has one
   pre-rewrite graph, so it gets one `fn` and two keys.
   - "Equal key ⇒ same `fn`" still holds.
   - Nothing reads `.key` outside `lifted.py`, and jit retraces per shape anyway.
4. **Files beyond the brief:**
   - `pfcheeshmc/compilation_cache.py`, the helper;
   - `tests/compilation_cache_run.py`, the child process of the cache test;
   - `tools/lift_cost.py`, the lead's probe made tree-aware;
   - `tools/warm_stages.py --cache-dir`, which prints disk-cache hits and writes, for F2f.
5. **`_build` returns a 6-tuple** (the memo is added). Tests that monkeypatch `_build` pass through.

## Items
### (a) Disk cache: DONE
- **The helper.** `pfcheeshmc/compilation_cache.py:27`, exported from `pfcheeshmc/__init__.py`. It
  sets exactly `jax_compilation_cache_dir`, `jax_persistent_cache_min_compile_time_secs=0` and
  `jax_persistent_cache_min_entry_size_bytes=-1`; a test diffs `jax.config.values`.
- **CPU support,** verified in `jax/_src/compilation_cache.py:is_cache_used`:
  - "cpu" is in `supported_platforms`;
  - the CPU client reports `supports_executable_serialization=True`.
- **Zero-code route:** JAX's three env vars. The README documents it. The directory alone is nearly
  useless: only 3 of 134 programs reach JAX's 1 s default, so a warm first call is 8.2 s against
  3.4 s with all three settings.
- **Counting hits:** a cache hit still emits `/jax/core/compile/backend_compile_duration`
  (`pxla.py:1533` wraps `compile_or_get_cached`). The tests count
  `/jax/compilation_cache/{compile_requests_use_cache,cache_hits,cache_misses}` instead. The pytest
  process never enables the cache.
- **README:**
  - the two pre-F2 paragraphs are rewritten: "Compile once per model structure" and the progress bars;
  - a new subsection, "Keeping the compiled programs between Python processes (opt-in)": what it
    covers, safe to delete, trust, and call order;
  - CPU numbers only; GPU is pending F2f.

### (b) Lift memo: DONE (`pfcheeshmc/lifted.py`)
- **Code:** `_Memo`, `_MEMO`, `_context`, `_signature`, `_recall`, `_spec_ids`.
  - `_build` calls `_recall` between `graph_replace` and the JAX rewrite, for the rules other than
    "none".
  - `lift_graph` keeps one key per spec id.
- **Signature:**
  - ops in toposort order (type and `__props__`), input wiring and output types;
  - constants by type and bytes;
  - inputs and placeholders by position and type only;
  - `jax_enable_x64` in the bucket key, because funcify bakes constants at that width.
- **Cost:** signature plus verification take 2–8 ms for the three sites.
- **Verification:** every candidate goes through `equal_computations(new_outputs, memo.outputs,
  in_xs=new_inputs, in_ys=memo.inputs)`. Inputs match by position, so swapped roles are different
  graphs.
  - Constants compare with `Constant.equals`. It treats 0.0 = −0.0 and NaN = NaN as equal; the
    signature's byte comparison separates them.
  - A signature or comparison that raises, `RecursionError` included, is a miss, never an error.
- **A hit still reads from the current model:** `values`, `sources`, `kept_static` and
  `static_shared`. `is_stale()` is unchanged.
- **Memory:** an unbounded module dict, like `_REGISTRY`. It retains 0.24–1.0 MB per model structure;
  the registry already retains 0.5–8.9 MB. `test_the_memo_holds_no_data` pins that the logit design
  matrix is not kept.

## Tests
`tests/test_lift_memo.py` (32) and `tests/test_compilation_cache.py` (8, with
`tests/compilation_cache_run.py`).

Must-fail evidence (temporary mutations of `lifted.py`):

| mutation | failing test | excerpt |
|---|---|---|
| memo off (old behaviour) | ladder hit test, 8schools | `assert (3, 3, 3) == (0, 0, 0)` |
| no `equal_computations` under a forced collision | `test_forced_signature_collision_never_returns_a_wrong_function` | `assert -21.169 == -21.283 ± 2.1e-06` |
| stale first-build values on a hit | extra-model hits, key tests, collision, swapped roles | value and logp mismatches |
| two same-shape values swapped on a hit | 6 swapped-roles cases, ladder, extras | `assert -44.1157 == -55.9766 ± 5.6e-11` |
| key cache ignoring shapes | key-shape and unknown-static-shape tests | `assert 0 == 1` |
| static-shared snapshot from the first build | `test_a_hit_still_reads_everything_...` | `assert (k is k)` |
| signature ignores constants, no verification | static-exponent miss | `this must be a build: assert (0 == 1)` |
| `jax_enable_x64` dropped from the context | `test_another_jax_enable_x64_is_another_function` | `assert (0 == 1)` |
| helper without min compile time | 4 of 8 cache tests | `assert (4 >= (135 - 5))` |

- **Base tree for (a):** `AttributeError: module 'pfcheeshmc' has no attribute
  'enable_compilation_cache'`.
- **The two-process cache test:**
  - process 1 writes 133 entries;
  - process 2 (new data and priors, same shapes) makes 134 of 135 requests as hits, 0 misses and 0
    new files. The one exception is a 0.04 s `jit(log)` compiled at import;
  - process 2's compile time is under 30% of process 1's, and its draws are finite and differ;
  - the control without the switch: 0 hits, no directory, and draws bit-equal to the cached run.
- **Draws** are bit-identical with and without the memo on 8schools and logit, for every
  InferenceData array.

## Measurements (task, CPU, 32-core host, `OMP_NUM_THREADS=4`, load 1–3 from the task's own runs)
`tools/warm_stages.py TREE MODEL 1024 10 25 --calls 3`. Seconds, calls 2–3 averaged.

| model | tree | wall | init | rest | transform |
|---|---|---|---|---|---|
| 8schools | f6108ac | 0.51 | 0.03 | 0.18 | 0.02 |
| 8schools | f2-e | **0.39** | 0.01 | **0.07** | 0.01 |
| gev | f6108ac | 1.43 | 0.17 | 0.44 | 0.06 |
| gev | f2-e | **1.01** | 0.13 | **0.09** | 0.02 |
| logit_1000x25 | f6108ac | 3.27 | 0.02 | 0.14 | 0.02 |
| logit_1000x25 | f2-e | **3.19** | 0.01 | **0.05** | 0.01 |
| radon_noncentered | f6108ac | 4.90 | 0.17 | 0.30 | 0.13 |
| radon_noncentered | f2-e | **4.58** | 0.03 | **0.12** | 0.10 |

- First calls are unchanged (13.4–18.2 s).
- Calls 2–3 make 0 XLA and 0 PyTensor compiles in both trees.
- Radon varies from call to call because of the seed-dependent ChEES mode.

**Disk cache, first call in a fresh process** (`tools/warm_stages.py $WT MODEL 1024 10 25 --calls 1
[--cache-dir DIR]`):

| model | no cache | cache cold (writes) | cache warm | size | XLA compile s, none → warm |
|---|---|---|---|---|---|
| 8schools | 13.47 s | 13.95 s | **3.36 s** (134 hits) | 1.5 MB, 133 files | 10.57 → 0.60 |
| radon_noncentered | 16.90 s | 17.59 s | **9.51 s** (139 hits) | 1.8 MB, 138 files | 8.57 → 1.01 |

## Findings and UNVERIFIED
- **Suite resource cliff** (lead follow-up):
  - At f6108ac the one-process suite ends at 61,028 memory mappings, against `vm.max_map_count`
    65,530.
  - A new XLA-heavy module aborted it with "Fatal Python error: Aborted" in
    `backend_compile_and_load`.
  - The task's module releases compiled programs at its end (`gc.collect(); jax.clear_caches()`), so
    the suite now ends at about 37,900.
  - Recommended: the same module-scoped teardown in a `tests/conftest.py`.
- **Dim lengths are static by design.** Under `dims=`, another J is another program.
- **Follow-up candidate:** memoise `_widen_model_inputs` on a pre-widen signature. It takes 0.02–0.06 s
  per call, now the largest part of the lift that is ours.
- **UNVERIFIED:**
  - the GPU persistent cache, which also enables XLA's autotune cache in the directory;
  - the README's claim that a changed jaxlib, device count or GPU misses. It rests on reading
    `jax/_src/cache_key.py`;
  - float32 end to end. Only the x64-off lift and logp are tested.
