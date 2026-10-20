# LEDGER — run 2

Decisions with evidence, the D1/D2/D3 rule evaluations, and contradictions (plan A7). Append-only.

---

## 2026-10-20 — run 2 opens

`pf-r2` created from `pf-r1` HEAD `cca5407` per plan A3. Hygiene commit `934e576`: 303 session files
and three `ref/*.pdf` files untracked (kept on disk and now ignored), the three root-level duplicates of
`docs/review-2026-10/*.md` untracked, and the run-2 plan versioned in `docs/review-2026-10/`. History was
not rewritten (owner's call), nothing is pushed, and `main` is untouched at `ecd9ba1`.

`tools/legacy/t3_cpu.py` is a copy of the run-1 T3 driver from `pf-r1-int` with a quarantine header
and an execution guard (`raise SystemExit` on `__main__`, verified firing). Its module body is unedited so
the defect stays auditable: the floor it emits is `MCSE/sd`, **not squared**, and it scored constrained
space, dropping `tau_log__`, `xi_interval__`, `sigma_log__` and radon's three `sigma_*_log__`. Run 1's T3
conclusion ("CPU NUTS cannot resolve b²") is void for this run (plan §1.2). It cannot run as-is in any
case: its `sys.path` and output-path assumptions no longer resolve from `tools/legacy/`.

Environment frozen in `results/r2/env.json`: python 3.13.16, jax 0.11.2, blackjax 1.7.1, pymc 5.28.5,
arviz 0.23.4, numpy 2.4.6; host workstation; 32 CPUs; one NVIDIA RTX A2000 with 6,138 MiB.

**A2 acknowledged.** The nine run-1 claims the plan voids are treated as void for every decision in this
run, including where this branch's own `results/REPORT.md`, `results/LEDGER.md` and
`results/REVIEW-HANDOFF.md` still assert them. R2-WP4 applies the E1–E12 errata in place; run-1 numbers
and row files are not rewritten.

**Pre-registered defaults.** Only D1 (`iterate`) and D2 (Pathfinder covariance source) may flip, and only
on their Part B §3 rules, each recorded clause by clause. D3 (metric refresh) stays opt-in whatever M2b
shows; it gets a recommendation only.

## 2026-10-20 — lead handover (owner's request)

the lead stopped after R2-WP0 items 1, 2 and 4 because of contention on its backend. The baseline suite it
had started did not report, and the R2-WP1 task left uncommitted work in `~/pf/r2-wp1`. At the
owner's request, the lead orchestrates the rest of run 2 with parallel tasks: **fast-tier** (R2-WP1/2/3 and code reviews) and **steady-tier** (R2-WP4,
enumerations). M-steps are launched by the lead through `tools/gpu_run.sh` driver scripts kept
in `results/r2/drivers/`. Every plan rule is otherwise unchanged. The R2-WP1 task's uncommitted diff
is carried forward and checked against the plan, not discarded. R2-WP3 is split into 3a (new files:
`tools/t3_cpu_nuts.py`, `summarise_rows.py smallest-pass`, `tests/test_t3_cpu.py`) and 3b (`r1_bias.py`
passthrough/pool-seeds, after R2-WP2 merges) so the file-disjoint half starts in wave 1.

Baseline suite on `pf-r2` at `4e07b45` (forced CPU, A3 command), copied from the command's own output: `125 passed in 165.79s (0:02:45)`.

## 2026-10-20 — R2-WP1 merged (`07575bc`)

Lead review APPROVE (`results/r2/reviews/R2-WP1-code.md`). Suite after merge (own output):
`130 passed in 174.33s (0:02:54)`.

**Contradiction (A6 protocol, recorded, continued).** The CPU smoke acceptance band (re-seed L within
0.5–2× of no-re-seed L) was missed on `logit_1000x25`: 29.7 / 9.7 = 3.07×. 8schools was 1.07×. The
A6 BLOCK needs > 3× on ≥ 2 models, so **no BLOCK, and M2b runs**.

| logit arm | integration time T = eps·L, seeds 1/2/3 |
|---|---|
| re-seed on | 4.8 / 5.3 / 5.5 |
| no refresh (dense Pathfinder metric) | 4.9 / 4.9 / 4.5 |
| re-seed off | 1.9 / 1.5 / 1.9 |

Lead reading: in integration time the re-seeded arm matches the no-refresh arm. The
no-re-seed control under-grows from L=1 in 25 phase-B steps. The L gap comes from `sample_diag`
being unable to absorb logit's correlation (whitened lmax 5.4–6.8). It is not the run-1 unit defect,
whose signature was 8–10× L. M2b reports both numbers at C=1,024.

**Plan error found by R2-WP4 (E3c).** The plan says radon `covraw` "fails resolvably 6/6". Row
`T1__r1_sweep.jsonl` `int:covraw` W=200 seed 2 has b² 0.00709 / floor 0.00413, which `g_r1` passes.
The erratum was narrowed to "fails resolvably at W=50/100, mixed at W=200". The task was right.

## 2026-10-20 — R2-WP2 merged; new run-1 defect E13 (`iterate` was never forwarded without a refresh)

Lead review APPROVE (`results/r2/reviews/R2-WP2-code.md`).

**E13 (found by the R2-WP2 task, verified by the lead).** At `4e07b45` the single-phase
`chees_warmup_with_progress` call in `_run_chees` passes no `iterate=`, so it takes
`utils.py:133` `iterate="ma"`. Only `_run_chees_two_phase` (`:925`, `:976`) forwarded it.
`5982ee7` fixes it. Consequences for run-1 claims:
- every run-1 row that requested `raw` **without** a metric refresh ran `ma`. A scan of
  `kwargs` in `results/runs/*.jsonl` finds 88 such rows:
  - `T1__r1_sweep` `int:covraw` 24;
  - `T1__prodpool*` `int:prodpool*` 16;
  - `T1__r2_poolcap*` 20;
  - `WP2__lgrowth` `wp2:C_cov_raw` / `wp2:D_cov_raw_t0` 16;
  - `WP4__reseed_ab` `ab:baseline` 12;
- A2's "still valid: R1 passing at W=50 on 8 Schools, GEV and logistic with (covariance, `raw`)"
  is really **(covariance, `ma`)**, i.e. the current default path. The passes stand; the label is
  wrong;
- run-1 "`raw` costs +7.7% warmup wall and saves none" (E11) compared two runs of the same
  configuration, so it is void, not merely "untested at R1";
- the WP4 refresh arms (two-phase) did run the requested iterate;
- `WP2__lgrowth`'s `raw` arms also ran `ma`. So the plan §1.3 claim "L at step 100 is 25–42 with
  `raw` versus 8.3 with `ma`" is not an iterate effect: whatever separated those arms, it was not
  the iterate. Not decision-bearing, because M2a measures `raw` vs `ma` directly.
M2a is therefore the **first** R1 measurement of `raw` on the single-phase path. R2-WP3's
`effective` check (flag vs `attrs["config"]`) is the guard that would have caught this.
Suite after the R2-WP2 merge (`62c7620`, own output): `159 passed in 271.65s (0:04:31)`.

Suite after the R2-WP3 + R2-WP4 merges (`25d2618` code; run before M1, own output): `191 passed, 4 warnings in 285.91s (0:04:45)`. M1 started from code trees of `eee10ae`; no code change since.

## 2026-10-20 — M1 done; **D2 = NO** (keep `imm_source="mc"`)

M1 ran 18 jobs, all rc=0, from code trees `eee10ae` (driver log `~/pf-runs/r2-logs/M1/driver.log`).
Rows: `results/r2/runs/M1__{r2,r1,mem}.jsonl`. Every row's `effective.imm_source` matches its arm.
Every other setting is the sampler default: `pf_draws=4000`, pool by the pool rule, covariance, `ma`.

| clause | evaluation | result |
|---|---|---|
| (a) R2-WP2 tests 1–3 pass | `tests/test_pf_closed_form.py` in the 191-pass suite | **pass** |
| (b1) R2 median grads/ESS(closed) ≤ 1.10× mc, every model | 8schools 57.39/55.76 = 1.03; gev 146.9/141.5 = 1.04; logit 77.34/75.86 = 1.02; radon 296.3/302.5 = 0.98 | **pass** |
| (b2) R2 per-seed count of max_rhat ≤ 1.01 not lower | 8schools 3 vs 3; gev 2 vs 2; logit 3 vs 3; **radon 2 vs 3** (closed seed 2 = 1.0101; mc 1.0075/1.0066/1.0046) | **FAIL** |
| (c) R1 per-seed g_r1 equal or better, every model | 8schools, gev, logit pass/pass/pass both arms; radon fail/fail/fail both arms | **pass** |
| (d) radon R1 peak ≤ mc | 653.5 MiB (closed, pool 4,096 rows) vs 1,242 MiB (mc, pool 122,500 rows) | **pass** |

D2 needs every clause to hold, so **D2 = NO**: the `imm_source` default stays `"mc"`. The deciding
clause misses by 0.0001 of R̂ on one radon seed. Per A6 this is recorded, not reinterpreted. REPORT
carries it as an open question for the owner. On these rows, the closed form costs the same at R2
(0.98–1.04×), gives the same R1 verdicts, halves radon's R1 peak memory, and removes the O(d³) pool.

Radon at R1 W=50 fails on both arms (b² 0.14–2.07 per seed; pooled b² 1.0 (mc) and 2.0 (closed)
against pooled floors 0.0017). That is a real bias, not noise; M2a's W grid is the next test.

## 2026-10-21 — M2a done (160 rows, 8 jobs rc=0); provisional **D1 = NO**; M2c not triggered; IT = raw

`results/r2/runs/M2a__r1.jsonl`, code trees `eee10ae`. Default path otherwise (`imm_source=mc`,
covariance, `pf_draws=4000`). Smallest W with `g_r1 == pass` on all of seeds 1–3
(`tools/summarise_rows.py smallest-pass`):

| model | ma: W / med grads per chain | raw: W / med grads per chain |
|---|---|---|
| 8schools | 25 / 540 | 25 / 861 |
| gev | 25 / 270 | 25 / 629 |
| logit_1000x25 | 50 / 319 | 25 / 297 |
| radon_noncentered | **300 / 1,770** | **100 / 2,037** |

- **D1 (a)** raw ≤ ma on 4/4 (25≤25, 25≤25, 25≤50, 100≤300), strictly smaller on 2 (logit, radon) → **pass**.
- **D1 (b)** raw's grads at its smallest W ≤ ma's at its own, needed on ≥ 3/4. Only logit holds
  (297 ≤ 319); 8schools 861 > 540, gev 629 > 270, radon 2,037 > 1,770. So 1/4 → **FAIL**.
- D1 therefore cannot flip whatever M3 shows. **D1 = NO; the `iterate` default stays `ma`.** M3 still
  runs, for R2-T3 and G3.
- **Radon is gateable at R1 without the refresh**: ma passes all three seeds at W=300, and raw from
  W=100. This replaces A2's "radon cannot be gated" and run-1's "the refresh is radon's only rescue".
  The M2c trigger (no passing W ≤ 300) is therefore **not met: M2c is SKIPPED**.
- **IT for M2b = raw**: it has the smaller smallest-passing W on 2 models (logit, radon) and ma on 0.
- raw grows L 2–4× faster (radon W=100: L 38 vs 8). It buys passes at smaller W at a higher
  gradient cost per chain.

Lead incident: M2a finished at 00:25, but my completion watcher matched its own command
line in `pgrep -f`, so it never fired, and the GPU sat idle until 07:22. Fixed by waiting on driver
log lines instead. M2b → M3 → M5 now run as one chain.
