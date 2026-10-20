# WP5 — code-correctness review of the R1 capacity work

**Provenance.** Executed by task `wp5-code-review-y` (worker model, shell available). The task **never
wrote this file** — it exhausted its tool budget, said so plainly, and sent its findings as text with the
explicit note "the pytest numbers in this report were NOT executed; verify before relying on them." Two
earlier tasks on this same branch died without delivering anything, one of them by running the full test
suite and blowing its tool timeouts. The lead reconstructed the review and re-verified the checks
that could be run cheaply. **Three of the four BLOCKING items are refuted.**

Branch: `wp/wp5` (checkout `/home/akbar/pf/wp5`), base `pf-r1-int`, head `0e6739f`.

---

## 1. "The warmup scan drops `is_divergent` the adapter still reads → crash on every call" — **REFUTED**

`pfcheeshmc/utils.py:250` builds the projection as
`proj = WarmupScanInfo(info.num_integration_steps, info.is_divergent, acc_vals)`, and the class at
`utils.py:41-51` declares exactly `num_integration_steps, is_divergent, acceptance_rate`. The reads at
`utils.py:244-245` (`jnp.sum(info.is_divergent)`, `chees_harmonic_acceptance(acc_vals, info.is_divergent)`)
are inside the scan body on the per-step `info`, not on the stacked projection. There is no contract
violation. Independently confirmed by execution: `tests/test_statistical_correctness.py` drives the full
public `sample_pf_chees_hmc` path and **passes (1 passed, 19.19 s, forced CPU)** — a crash-on-every-call defect
would fail it instantly.

## 2. "`WarmupInfo.trace=None` default is consumed as a list → TypeError in every production warmup" — **REFUTED**

`WarmupInfo` at `utils.py:16-38` is a `NamedTuple` with fields `info: Any` and `trace: Any` and **no
defaults**; there is no `trace=None` anywhere in `utils.py`. `pfcheeshmc/chees.py` contains no `_adjust_length`
function and no `trace[...]` subscript (0 matches for either pattern). The claimed call path does not exist.
Same execution check as item 1.

## 3. "`sampler.py:675` sets `warm_info=None`, then `len(warm_info)` → TypeError" — **REFUTED**

Neither `warm_info = None` nor `len(warm_info)` occurs in `pfcheeshmc/sampler.py` (0 matches for both
patterns). The reviewer explicitly flagged this as unverified because it ran out of calls; it is wrong.

## 4. "Byte-identical picks: ties are not covered; the at-cap case is" — **PARTIALLY CONFIRMED, kept as a test gap**

`tests/test_wp5_capacity.py` (9 tests, **9 passed in 40.57 s** forced CPU — the task did not run them)
covers the default path, a pool below the cap and a pool above it, and asserts index identity. It has **no
tie case** and no exact-boundary case. Note also the real cap logic: `min(draws, 4*chains)` means at
`C=1,024, pf_draws=4,000` the result is exactly `4,096`, so `pf_pool_cap=4096` is effectively **no cap** —
the fix works only because `pm_chains=1` and the chunking removed the `(C, pf_draws)` vmap that was actually
OOMing, not because the pool shrank. That is a correction to how WP5's win should be described.

## 5. Memory attribution — accepted in part; the wall-time half is **NOT ESTABLISHED**

Accepted: per-configuration peaks must come from `kwargs.peak_bytes_in_use` (the row's own `peak_mem_bytes`
is `0`), because this box has no `reset_memory_stats()` and `peak_bytes_in_use` is process-cumulative — so the
radon "5.36 GiB" figure is cumulative, not pool-stage, and the honest claim is that `pf_pool_draw` fails with
"Failed to allocate 5.36 GiB". Also accepted: there is **no after-probe row for radon at C=128**, so the
before/after table's C=128 row cannot claim an after-peak.

**Refuted by the row files:** the task's headline "8 Schools C=1,024 went 26.9 → 114.8 s (4.3× slower after)".
`WP5__memprobe_before.jsonl` (80 rows) contains **no `wall_s` on any row** — the before-probe is memory-only.
A before/after wall comparison cannot be built from the delivered files at all, so no slowdown is established
and none is refuted. The report's T4 table therefore states before-side peaks only and after-side walls only,
with the labels on the face of the table.

## 6. Cold/warm and cumulative-peak hygiene — accepted

`WP5__accept.jsonl` has radon warm and cold pairs plus GEV / 8 Schools / logistic **cold only**. The warm row
is the second rep in one process, so its cumulative peak (653.3 MiB) includes the cold rep's allocations and
is not attributable to the warm run alone; the cleanest single-process radon number is the C=64-style probe
(107.9 MiB). The warm 42.94 s wall is legitimate — the process-cumulative problem is memory, not time.

## 7. Strongest remaining silent-wrong path — accepted

(a) `pm_chains` is not plumbed from `PfCheesHMC` into `_run_pathfinder` (`_run_pathfinder` calls
`model.point()` directly at `sampler.py:~299`), so that kwarg is a harness concept that the sampler does not
consume — it would be silently ignored if someone tried to turn it off. (b) `pf_pool_cap` reaches the
**sampling-side** `_sample_pathfinder` call but that call is outside `chain_fn`, so vmapping one chain over
`draws` means the cap does nothing for sampling at all. (c) `_score_chain` has no remaining callers — dead
code that looks load-bearing, i.e. the guard the tests exercise is the inline one.

---

## Items as delivered, with disposition

**BLOCKING (task):** 1, 2, 3 → **all REFUTED** by source and by the passing end-to-end test. 4 → kept as a
**test gap** (ties, exact boundary), not a correctness blocker.
**NON-BLOCKING (carried forward):** the before/after table must be labelled cumulative-peak, not per-stage
(needs a fresh process per stage to fix properly); the pool cap is effectively no-op at `pf_draws=4000,
C=1024`; `pm_chains` should either be plumbed or removed rather than sitting unused.
**NITs:** `tools/wp5_accept.sh` runs `gpu_run.sh pixi run pixi run env …` — a doubled `pixi run`;
`_kernel_flat` is called before `self.rvs` is set; `model_fn`'s `kwargs.pop("pm_chains")` would drop the key
from the live sampler instance; no test asserts the trimmed `sample_stats` variable list in
`idata.sample_stats.data_vars` (only per-field presence).

## Verdict

> **RECOMMEND: hold** — as delivered.

**Lead's disposition:** the stated blockers do not exist, but two findings survive and are material:
**(i)** the memory table's provenance is weaker than its presentation (cumulative peaks, no after-probe at
C=128, no before-side wall times at all), and **(ii)** the pool cap is a no-op at the acceptance configuration
while `pm_chains` is not plumbed into the sampler, so the branch's own description of *why* the wall moved is
wrong even though the wall did move — radon genuinely runs at C=1,024 (cold 528.79 s, warm 42.94 s, peak
653.5 MiB in `WP5__accept.jsonl`). Fix the SUMMARY's attribution, plumb or delete `pm_chains`, add the
tie/boundary tests, then re-review. `wp/wp5` also has no statistical-validity review yet.
