# WP4 — statistical-validity review (arbiter)

**Provenance.** Two tasks were assigned this review; both were killed by the tool layer without writing a
file, after one of them delivered a long analysis whose headline column could not be reproduced. This review
is therefore **lead-executed** (an A4 deviation, recorded in `results/STATUS.md`), and it settles the
disagreement by recomputing everything from the row files: `~/scratch/wp4_arbiter.py`,
`~/scratch/wp4_estimator_rank.py`, `~/scratch/wp4_verify_claims.py`. Where a task's claim and the rows
disagree, the rows are cited.

## 1. The refresh and the re-seed, per seed (warm rows, C=1,024, D=10)

| model | no refresh (`int:covraw`, T1) | refresh + re-seed (`wp4:*`) | refresh, **no** re-seed | re-seed alone |
|---|---|---|---|---|
| 8 Schools | 974 [955–993] | 1,303 = **1.34×** | 1,309 = **1.34×** | 1.00× |
| GEV | 557 [519–595] | 4,298 = **7.72×** | 841 = **1.51×** | **5.11×** |
| logistic | 324 [313–335] | 2,386 = **7.37×** | 487 = **1.50×** | **4.90×** |
| radon NC | 372 [260–485] | 5,206 = **13.98×** | 579 = **1.55×** | **8.99×** |

The branch's central claim **holds**: dropping the trajectory-length re-seed collapses a 4.9–9.0× penalty to
nothing, and refresh cost falls to 1.34–1.55× of no-refresh. It was also overstated in the report (3.7–15×),
which is corrected here — 8 Schools never cost more than 1.34×.

A task's contrary summary ("re-seed cost 1.32× / 0.99× / 1.81× / 0.92×, therefore the ablation fails its own
rule") **cannot be reproduced from the rows** in any pairing I can construct, and it contradicts the same
task's own three-column table, which matches the numbers above. Verdict: the ablation stands; the summary did not.

## 2. Estimator ranking and the kill clause

| estimator | 8 Schools | GEV | logistic | radon NC | vs diagonal (logistic, radon) |
|---|---|---|---|---|---|
| `sample_diag` | 1,292 | 4,804 | 2,386 | 5,612 | reference |
| `fisher_diag` | 1,157 | 6,673 | **1,352** | **3,886** | **0.57×, 0.69×** → clears 1.2× |
| `dense_shrunk` | 1,344 | 4,237 | 2,323 | 5,469 | 0.97×, 0.97× → null |
| `diag_lowrank` | 1,400 | 3,778 | 4,934 | 5,317 | 2.07×, 0.95× → worse |

- **Estimator-vs-estimator at W=50 is unconfounded**: all four arms share chains/draws/warmup/seed/cold and
  differ only in `chees_kwargs.metric_refresh`. So a task's blocking claim that "no pair differs only by
  estimator, therefore the kill clause is not evaluable" is **false**.
- The **kill clause is decidable and kills two of three** challengers (`dense_shrunk` is a 3% move,
  `diag_lowrank` is 2.07× worse on logistic). `fisher_diag` clears it on both correlated models.
- The one part of that task's coverage argument that **is** right: radon only has `sample_diag` in the *no-re-seed*
  file, so the estimator comparison on radon exists only in the with-re-seed configuration.

## 3. Bias, and where the gate cannot decide

All 16 estimator cells at W=50 have floor ≤ 0.005 and b² < 0.01 → resolvable passes; bias does not discriminate
between estimators, cost does. Radon's ladder is the exception that must be quoted exactly:

| radon, refresh no re-seed | b² | floor | class |
|---|---|---|---|
| W=50 (3 seeds) | 0.0081 / 0.0081 / 0.0136 | 0.0054–0.0060 | **unresolvable** |
| W=100 | 0.0070–0.0107 | 0.0058–0.0070 | **unresolvable** |
| W=200 | 0.0058–0.0086 | 0.0057–0.0064 | **unresolvable** |

No radon cell at any warmup in either file is resolvable, and the best b² (0.0136 at W=50) is 2.4× its own
floor — so radon is not shown to be unbiased, and it is not shown to be biased. Cost, meanwhile, rises 1.82×
at W=100 and 2.70× at W=200 versus no-refresh. The correct statement in the report is "unmeasurable at the
current shape", and that is what §3e now says.

## 4. Provenance — the one thing that must be fixed before this is quoted as shipped evidence

- **No factor is recorded as a row field.** `WP4__r1.jsonl` and `WP4__reseed_ab.jsonl` carry `arm`, `cold`,
  `git_sha`, `kwargs` — not `orientation`, `iterate`, `metric_refresh` or `metric_refresh_reseed`. Two arms
  that differ only by the re-seed are byte-identical in their recorded configuration. This is the review's only
  real blocking item, and it is cheap: add the four columns in `tools/r1_bias.py` and stamp them.
- Both tasks claimed `warmup_steps_used` was **absent** from the ablation rows, and used that to argue the
  no-op-swap case cannot be excluded. **Refuted: it is present in 18/18 ablation rows, 32/32 `WP4__r1` rows
  and 48/48 T1 rows.** Combined with `warmup=50` and the refresh swap step, the silently-skipped-swap case is
  checkable.
- `git_sha` is `d457a17` on all 18 ablation rows, so they post-date the `wp4:*` baseline rows and are
  internally consistent.
- `T5b` (per-step cost split) genuinely does not exist: `ls ~/pf/*/results/runs/T5b*` → nothing.

## 5. What is confounded, stated once

Any comparison between a `WP4__r1` arm and a T1 arm moves orientation and iterate together with the refresh,
because the T1 arms are `int:covraw`/`int:shipped` and WP4's arms run covariance+raw by driver flag. Within-file
comparisons (estimator vs estimator; re-seed vs no re-seed) are clean. So: **the re-seed result and the
estimator ranking are attributable; "the refresh versus today's default sampler" is not**, and the single arm
that would settle it is `wp4:norefresh` run with `--iterate raw --orientation covariance` at W=50 on all four
models.

## Verdict

**BLOCKING:**
1. Record `orientation`, `iterate`, `metric_refresh`, `metric_refresh_reseed` (and the swap step) as row
   columns so the refresh evidence is self-describing.

**NON-BLOCKING:**
2. Add the no-refresh control at the same iterate/orientation as the refresh arms (one arm, four models).
3. Run `dense_shrunk` / `fisher_diag` on radon in the no-re-seed configuration — the kill clause is currently
   decidable on logistic only for those two.
4. Add rows at 2–3× the ablation's runtime so cold/warm can be separated; every ablation row is cold.
5. Improve the noise floor (reference MCSE + serial correlation) before any *near-floor* accept/kill — this is
   exactly where radon sits.

**NIT:** `refresh_swap_step` is undocumented in the harness; GEV's `sample_diag` spans 3,373–6,236 across two
seeds, so quote ranges not medians for that cell.

> **RECOMMEND: merge** the code with the one blocking provenance fix, and quote WP4's result only as:
> *"dropping the trajectory-length re-seed removes a 4.9–9.0× penalty, leaving refresh at 1.34–1.55× of
> no-refresh; `fisher_diag` is the only challenger that clears the kill clause; radon is unmeasurable at
> C=1,024 × D=10 at every warmup tried."*
