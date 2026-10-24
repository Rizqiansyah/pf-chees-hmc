# F2j SUMMARY: release compiled programs in the tests only when the mapping count is high

Task: fast-tier, one round.
- Branch `f2-j`: 3 commits on b71e71e, head 89b3a04. Merged into `f2` with `--no-ff` (6ade5b1).
- Task's suite (final state, with the task's `-p mapcount_modules` sampler): `584 passed, 64 warnings in 720.65s
  (0:12:00)`. It was 575; this adds 9 in `tests/test_conftest_release.py`.
- Lead suite on merged `f2`, the bare command (no plugin): `584 passed, 64 warnings in 725.14s (0:12:05)`, rc 0
  (6ade5b1). The load was 1.4 at the start and 4.2 at the end, with the F2k task starting during the run. The real
  `~/.cache/pfcheeshmc` does not exist after it. The bare command matches the task's two timed runs (717 s and
  721 s) within 8 s, so the sampler plugin did not distort them.

## Lead review
- **The rule** (`tests/conftest.py:46-90`): after a test module, release the compiled programs (`gc.collect()`,
  then `jax.clear_caches()`) only if `/proc/self/maps` holds more than `limit // 2 - 16,000` lines, where the limit
  is read from `/proc/sys/vm/max_map_count`. Where either cannot be read, release after every module, as before.
  At the default 65,530 the threshold is 16,765.
- **Its claim** holds as long as no module grows by more than 16,000 mappings: a module that starts at or under the
  threshold ends under half the limit. The largest module growth measured is 12,589. A module that grew more would
  still end far from the limit: an abort needs about 48,800 mappings of growth from the threshold.
- **Acceptance:**
  - peak 22,730 and 22,726 mappings, against half the limit, 32,765: met;
  - 53.5–57.4 s recovered of F2g's 73 s (73–79%): "most of it", met;
  - no test changed outcome: met;
  - the threshold and its measurement are in the docstring: met.
- **The audit of in-process compile counts is accepted.** JAX's in-memory caches are keyed on function identity,
  static arguments and abstract shapes. A cache warmed by an earlier module can therefore only remove compiles from
  a test's own warm-up call. It cannot add a compile to a second call, or hide a rebuilt closure. The never-release
  run backs this: every JAX cache stayed warm up to test_widen, and every test that ran passed, apart from the 4 the
  forcing plugin breaks by design.
- **The cost is a second probe file** (`results/f2/probes/mapcount_modules.py`, measurement only) and 9 unit tests,
  which take a fraction of a second.

## DEVIATIONS (accepted)
1. **Commit trailer.** The task signed the required commit trailer. TASK-RULES said
   "the pinned tier", which was wrong for a fast-tier task. The rule now tells a task to record which pass produced the commit.
   Earlier task commits (F2a–F2i) carry that pinned tier. That history is merged and is not
   rewritten.
2. **Two files beyond the spec's list:**
   - `tests/test_conftest_release.py`, the unit test of the rule;
   - `results/f2/probes/mapcount_modules.py`, the per-module probe. `mapcount_plugin.py` is untouched.
3. **All 4 full-suite runs were used:**
   - baseline;
   - mine1;
   - mine2 (the final state);
   - a forced "never release" run, which aborts by design and is audit evidence, not a timing run.
   The two timed runs of the final rule agree within 4 s.
4. **The timed runs carried `-p mapcount_modules`** with a 1 s sampler thread and no per-test reads, the same in
   all three. The lead's suite is the bare command.
5. **A docstring-only commit (89b3a04) came after mine2.** The unit test was re-run, and the diff is comments only.

## Spec items
- **Rule in `tests/conftest.py`:** DONE.
  - `mapping_count` (:50) counts `b"\n"` in `/proc/self/maps`.
  - `mapping_limit` (:59) reads `/proc/sys/vm/max_map_count`.
  - Both return None on `OSError`; the limit also returns None on `ValueError`.
  - `release_threshold(limit)` (:68) is `int(limit * PEAK_FRACTION) - MODULE_GROWTH`, with 0.5 and 16,000.
  - `should_release` (:73) is True when either is None, else `count > threshold`.
  - `release_if_needed` (:78) and the fixture (:86).
  - At a limit of 32,768 the threshold is 384, under an idle process's ~2,000 mappings, so such a host releases
    after every module.
- **Threshold from measurement:** DONE. Growth per module is peak minus start, each of the 32 modules alone in a
  fresh process.
  - The largest: test_pf_closed_form 12,589, test_compile_once 12,582, test_chunked_loops 8,584,
    test_lift_memo 6,349, test_lifted 5,027; the other 27 are under 4,700.
  - In the suite, after a release, the same modules grew 12,286, 12,271, 8,326 and 6,020.
  - 16,000 is 27% above the largest.
- **README Testing line:** DONE (`README.md:57`, "about 12 minutes").

## Measurements
Full-suite runs, CPU, `OMP_NUM_THREADS=4`, `-p mapcount_modules` with `MAPCOUNT_SAMPLE_S=1`. The load is the
1-minute average at start and end.

| Run | conftest | Result line | Load start → end | Peak mappings |
|---|---|---|---|---|
| baseline | b71e71e (release after every module) | `575 passed, 64 warnings in 774.17s (0:12:54)` | 3.32 → 3.89 | 15,688 |
| mine1 | 7c4079b | `584 passed, 64 warnings in 716.78s (0:11:56)` | 2.67 → 5.22 (7.97 mid-run) | 22,730 |
| mine2 | 54c0300 | `584 passed, 64 warnings in 720.65s (0:12:00)` | 0.88 → 2.78 (9.18 mid-run) | 22,726 |
| never (`MAPCOUNT_FORCE=high`) | forced | aborted, RC 134, 666 s in, in test_wp5_capacity (`Fatal Python error: Aborted` in `backend_compile_and_load`) | 3.11 → 4.61 | 64,301 at the end of test_widen |

In mine2, the four modules that ended above 16,765 released, and the same four did in mine1:

| Module | End count | After release |
|---|---|---|
| test_compile_once | 22,706 | 2,356 |
| test_lifted | 17,914 | 3,439 |
| test_pf_closed_form | 17,274 | 3,439 |
| test_step_size_batched | 18,415 | 3,441 |

Every other module ended at or under 14,703 and kept its caches.

A per-test check with `mapcount_plugin.py` and a 0.25 s sampler over the worst region (acceptance_display through
compile_once) peaked at 22,695, in `test_compile_once.py::test_draws_match_the_f2_head_before_this_task[8schools]`.
That agrees with the 1 s sampler's 22,715.

Commands (in `~/pf/f2-j`; `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=4`):
```
# per module, alone in a fresh process (two at a time)
MAPCOUNT_PER_TEST=1 MAPCOUNT_MODULES_OUT=<file> pixi run --manifest-path "$MAIN/pixi.toml" env -u JAX_PLATFORM_NAME \
  PYTHONPATH="$WT/results/f2/probes:$WT" python -m pytest "$WT/tests/<module>.py" -q --rootdir "$WT" -p no:cacheprovider -p mapcount_modules
# full suite
pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT/results/f2/probes:$WT" \
  python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider -p mapcount_modules
```

## Tests (`tests/test_conftest_release.py`, 9)
- The threshold is half the limit less one module's growth.
- A count at the threshold is kept, and one above it is released.
- An unreadable count or limit releases.
- A limit under twice the growth releases after every module.
- The readers agree with `/proc`, and return None where `/proc` is unreadable or not a number.
- The fixture runs `gc.collect` and then `jax.clear_caches` only when it releases.

Each of 12 mutations of `conftest.py` is killed. Among them: `>` to `>=` and to `<`; the fallback returning False;
the rule always True or always False; `gc.collect` dropped, or swapped with `clear_caches`; `PEAK_FRACTION` 0.4;
`MODULE_GROWTH` 0; the readers catching the wrong exception.

Forced runs of the real conftest, on a 6-module subset (49 tests):

| Mode | Peak | Count after each module |
|---|---|---|
| real rule | 11,776 | no release |
| forced low | 4,605 | back to ~2,200 |
| forced high | 11,780 | never drops |

## Audit: in-process compile-count assertions (no test changed)
- **"Second call compiles nothing"** in test_compile_once (`:176`, `:204`, `:224`, `:244`, `:298`–`:371`, `:491`),
  test_initial_points `:280` and test_step_size_batched `:110`.
  - Each counts the call after the test's or module's own warm-up.
  - A warm cache can neither fail them nor make them vacuous.
- **Cold counts in test_compile_once:**
  - `:124`, `:142` and `:155` are positive controls on fresh lambdas or a `cos` nothing else uses.
  - `:259` is C=16, chunks (6,4,5). test_chunked_loops uses C=8 and other chunk sets, and the test passed even
    in the never-release run.
  - `:270` (C=24, J=6) and `:283` (C=20) use shapes no other module uses.
- **Child-process tests are not affected:** start_repair, pm_chains, compilation_cache, compile_equiv and widen.
- **PyTensor and lift-code counters** do not count JAX caches. test_lift_memo empties its memo per test.

## UNVERIFIED
- The cost of today's teardown against none at all. The 73 s is F2g's figure, from an older head. The
  never-release suite now aborts, so it cannot be timed. The saving reported is baseline minus the new rule.
- The last 2 tests of test_wp5_capacity did not run in the never-release run. They pass in mine1 and mine2.
- Behaviour where `/proc` is unreadable is tested only with `open` replaced, not on a non-Linux host.
- Host load varied during the timed runs (other users).
