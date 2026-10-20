# WP4 — statistical-validity review of the ensemble metric refresh

**Provenance, and it changes how much weight this review carries.** The assigned task
(`wp4-stats-review-z`) died without writing its file, as six of seven review tasks this run have done.
It delivered a verdict and several numbers in its final message. The lead then **re-derived the
statistics from the row files itself** and found that the task's central blocking claim was false — and that
**the report's own §3e carried a different wrong number, inherited from the WP4 implementation SUMMARY.**
Every figure below was printed by the lead's script `~/scratch/wp4_estimator_rank.py` against
`WP4__r1.jsonl` (32 rows), `WP4__reseed_ab.jsonl` (18 rows) and `T1__r1_sweep.jsonl` (48 rows).

## 1. The estimator ranking — report corrected, task's blocker refuted

Gradients per chain, W=50, C=1,024, D=10, median over seeds (range in parentheses):

| estimator | 8 Schools | GEV | logistic | radon NC |
|---|---|---|---|---|
| `wp4:sample_diag` | 1,292 (1,281–1,304) | 4,804 (3,373–6,236) | 2,386 (2,381–2,392) | 5,612 (5,513–5,712) |
| **`wp4:fisher_diag`** | **1,157** (1,091–1,223) | 6,673 (6,186–7,160) | **1,352** (1,312–1,393) | **3,886** (3,767–4,006) |
| `wp4:dense_shrunk` | 1,344 (1,302–1,386) | 4,237 (3,971–4,503) | 2,323 (2,143–2,503) | 5,469 (4,899–6,039) |
| `wp4:diag_lowrank` | 1,400 (1,353–1,446) | 3,778 (3,463–4,092) | 4,934 (4,892–4,976) | 5,317 (4,725–5,909) |

Kill clause (must beat the plain sample diagonal by 1.2× on the correlated models, logistic **and** radon):

| estimator | logistic ratio | radon ratio | verdict |
|---|---|---|---|
| `fisher_diag` | **0.57** (1.77× better) | **0.69** (1.45× better) | **CLEARS the kill clause — keep** |
| `dense_shrunk` | 0.97 | 0.97 | FAIL → kill (a 3% gain, inside noise) |
| `diag_lowrank` | 2.07 (worse) | 0.95 | FAIL → kill |

Three corrections follow:

1. **The task's BLOCKING item — "no pair of rows differs only by estimator, so the kill clause is not
   evaluable" — is REFUTED.** `WP4__r1.jsonl` contains all four estimators at identical chain/draw/warmup/seed
   settings; the ranking above is exactly that unconfounded comparison. The task reached its conclusion
   because it compared the wrong set (the ablation file, which holds only two arms).
2. **`results/REPORT.md` §3e was wrong** and is corrected by this review: it said *"no estimator clears the
   plan's 1.2× kill clause … dense-with-shrinkage 1.03× on logistic and radon, low-rank worse, Fisher-diagonal
   worst on GEV at 6,673"*. The last clause is true (Fisher is worst on GEV: 6,673 vs 4,804) but the conclusion
   is inverted — **`fisher_diag` clears the clause on both decisive models**, and dense-with-shrinkage is
   0.97× (a marginal gain, not a 3% regression). The error was inherited from the implementation task's own
   SUMMARY without re-derivation, which is the exact failure mode this run has been built to catch.
3. **Accept rule** (better gradients on ≥ 3 of 4 models, versus the plain diagonal): `fisher_diag` is better on
   8 Schools, logistic and radon, worse on GEV → **3 of 4, PASS**. `dense_shrunk` 2 of 4 → FAIL. `diag_lowrank`
   1 of 4 → FAIL.

## 2. Bias (b² median / noise floor), W=50

| estimator | 8 Schools | GEV | logistic | radon NC |
|---|---|---|---|---|
| `sample_diag` | 0.0010 / 0.0020 | 0.0013 / 0.0007 | 0.0006 / 0.0013 | 0.0020 / 0.0029 |
| `fisher_diag` | 0.0010 / 0.0019 | 0.0002 / 0.0006 | 0.0009 / 0.0017 | 0.0019 / 0.0028 |
| `dense_shrunk` | 0.0008 / 0.0019 | 0.0003 / 0.0006 | 0.0009 / 0.0011 | 0.0018 / 0.0029 |
| `diag_lowrank` | 0.0009 / 0.0020 | 0.0003 / 0.0005 | 0.0006 / 0.0009 | 0.0019 / 0.0026 |

All four sit below the 0.01 bar on every model and every floor is ≤ 0.005, so under `g_r1` (the corrected rule
at `wp/wp3` `9d5c30d`) **all 16 cells are resolvable passes**. Bias does not discriminate between estimators
here; cost does. Note the floor instability: the same nominal radon cell has floor 0.0029 in `WP4__r1.jsonl`
and 0.0054–0.0060 in `WP4__reseed_ab.jsonl` — a ~2× swing between runs, which is itself a reason not to make
near-boundary claims from single cells.

## 3. The re-seed ablation — the headline holds, but its provenance has a hole

| model | no refresh (T1 `int:covraw`, W=50) | refresh **+ re-seed** (`wp4:sample_diag`) | refresh, **no re-seed** |
|---|---|---|---|
| 8 Schools | 974 (955–993) | 1,292 (1,281–1,304) | **1,309 (1,291–1,333)** |
| GEV | 557 (519–595) | 4,804 (3,373–6,236) | **841 (708–855)** |
| logistic | 324 (313–335) | 2,386 (2,381–2,392) | **487 (467–501)** |
| radon NC | 372 (260–485) | 5,612 (5,513–5,712) | **579 (559–711)** |

Overhead versus baseline: with re-seed **1.3× / 8.6× / 7.4× / 15.1×**; without **1.34× / 1.51× / 1.50× / 1.56×**.
So the report's "3.7–15× → 1.34–1.55×" is confirmed (the low end is 1.3×, not 3.7× — the 3.7 came from
`dense_shrunk`-class numbers; the honest range for the sampled-diagonal arm is 1.3–15× → 1.34–1.56×).

Radon's ladder: no-reseed at W=100 is 1,182 gradients (1,128–1,498) and at W=200 is 3,340 (3,236–3,401) —
5.7× the W=50 cost, with b² moving from 0.0081–0.0136 to 0.0058–0.0086 while floors stay 0.0057–0.0070. Still
unresolvable at every warmup; the report already says this.

**Provenance defect found while verifying, and it is the review's main actionable item:** the re-seed factor
is **not recorded in any row field**. The configurations of `wp4:sample_diag` and `ab:sample_diag_noreseed`
are byte-identical in the data (`chees_kwargs` shows `{"draws": 50, "iterate": "raw", "metric_refresh":
"sample_diag"}` in both), so the distinction rests entirely on the arm *label*. Fix: emit
`metric_refresh_reseed` (and `metric_refresh_swap_step`) as row columns in `tools/r1_bias.py`, stamp them on
the ablation arms, and re-derive — cheap, but until then the run's central WP4 result is not self-describing
evidence.

## 4. Confounding, stated precisely

`WP4__r1` versus the T1 baseline differ in orientation and iterate as well as the refresh (T1's rows store
those factors; `WP4`'s rows do not store them at all — see §3). Therefore:
- **Estimator-vs-estimator** comparisons are clean (§1).
- **Refresh-vs-no-refresh** is clean *within* the covariance+raw configuration, because both sides were run
  covariance+raw (`WP4__r1` arms and `int:covraw`), and the ablation's own three-arm structure at W=50
  reproduces it.
- The task's alternative "unconfounded" number (8 Schools W=200: 1,768.9 vs 1,521.6) **does not exist**: the
  actual 8 Schools W=200 rows are `int:shipped` 3,305/3,310 and `int:covraw` 4,004/3,807, and no ablation row
  has W=200 for 8 Schools. That claim is refuted.
- Worth noting from those same rows: at W=200 the corrected arm costs **more** gradients than the shipped arm
  on 8 Schools (4,004 vs 3,305) for a b² of 0.00047 vs 0.00076 — a bias difference inside the floor. That is a
  legitimate question about the corrected default at long warmups, and it is not in the report.

## 5. Wall time

No `wall_s` in these rows is trustworthy in absolute terms: the harness bug (per-step diagnostic inside the
timer, ~52% warm overstatement) was fixed at `wp/wp3` `9d5c30d` after all WP4 rows were taken. Equal-warmup
ratios survive; absolute seconds do not. The report carries this caveat.

## 6. T5b (per-step cost split) — still not run

`ls ~/pf/*/results/runs/T5b*` → nothing. The task's non-blocking point is correct and it is the one
unfilled T-table cell (§6b/T5).

---

## Verdict

**BLOCKING (must clear before merge):**
1. `metric_refresh` / `metric_refresh_reseed` / swap step are not recorded as row fields, so the ablation —
   the evidence behind WP4's headline — is not falsifiable from its own data. Cheap fix, then re-derive.
2. The estimator decision in `results/WP4/SUMMARY.md` is inverted relative to the rows: it kills what the
   numbers keep (`fisher_diag`) and keeps the kill list rationale that doesn't hold (`dense_shrunk` 1.03×).
   SUMMARY must be corrected to match §1 before this branch's conclusions are quoted anywhere.

**NON-BLOCKING:**
3. `refresh_swap_step` is undocumented in the harness's own help; state its default and its interaction with
   `warmup_steps_used` (the no-op-at-`W ≤ swap_step` trap the code review raised).
4. No `wp4:sample_diag` rows at W=100/200, so cost growth with warmup is only known for one estimator.
5. Floor instability between nominally identical cells (radon 0.0029 vs 0.0054–0.0060) should be reported as a
   measured property of the instrument, since near-boundary verdicts depend on it.

**NITs:** GEV's `sample_diag` range is 3,373–6,236 (an 85% spread across two seeds) — quote the range, not the
median, wherever that cell appears; `dense_shrunk` at 0.97× is a null result and should be described as
"indistinguishable from the diagonal", not as a regression.

> **RECOMMEND: hold** — clear the two blocking items (a harness column and a corrected SUMMARY), then merge.
> The measurement itself is good: the re-seed ablation reproduces, and Fisher-diagonal is a real 1.45–1.77×
> win on the two correlated models, which is the opposite of what this branch's SUMMARY concluded.
