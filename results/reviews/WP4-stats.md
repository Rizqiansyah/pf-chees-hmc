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

*Reconciliation note:* the "refresh + re-seed" column above is the median across **all four estimators**
(`wp4:*`), which is why GEV reads 4,298 there. Pairing **the same estimator** on both sides of the re-seed
(`wp4:sample_diag` vs `ab:sample_diag_noreseed`) gives 1,292 / 4,804 / 2,386 / 5,612, i.e. re-seed costs
0.99× / **5.71×** / 4.90× / **9.69×**. `results/REPORT.md` §3e uses the same-estimator pairing, which is the
correct comparison; both are shown so neither is mistaken for the other.

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

**BLOCKING:**
1. `metric_refresh_reseed` is recorded in **no** row field. Every other factor *is* recorded inside `kwargs`
   (`pathfinder_kwargs.imm_orientation`, `chees_kwargs.iterate`, `chees_kwargs.metric_refresh`,
   `pathfinder_kwargs.draws`, `init_pick`), so the two ablation arms — which differ *only* by the re-seed — are
   byte-identical in everything the data says about them. One line in the harness fixes it.

## 5. Attribution: the refresh comparison is **clean**, and this corrects an earlier draft of this review

Reading the recorded `kwargs` of the actual rows (8 Schools, W=50, cold):

| source | arm | pf_draws | init_pick | orientation | iterate | metric_refresh | grads/chain |
|---|---|---|---|---|---|---|---|
| T1 | `int:covraw` | 1 | stratified | covariance | raw | — | 993, 955 |
| WP4 | `wp4:sample_diag` | 1 | stratified | covariance | raw | sample_diag | 1,304, 1,281 |
| WP4 | `ab:sample_diag_noreseed` | 1 | stratified | covariance | raw | sample_diag | 1,309, 1,291 |

So **refresh vs no-refresh is a single-factor comparison** — same orientation, same iterate, same start policy,
same chains/draws/warmup/seeds. The task's claim that "any T1-vs-WP4 comparison moves orientation and iterate
too, so the gain is not attributable" is **REFUTED by the rows**, and the earlier draft of this review repeated
it; that draft is superseded by this section. The only unattributable axis left is the re-seed (§BLOCKING 1),
and the only missing control is `wp4:norefresh` at W=100/200.

## 6. The caveat that applies to every R1 number in this run

All 48 T1 rows and all 50 WP4 rows run with **`pathfinder_kwargs.draws = 1`** — one Pathfinder draw — while the
production default is 4,000 and `wp/wp5`'s capacity work is precisely about the 4,000-draw pool. The R1 bias
evidence therefore describes a *single-start* regime: it does not include the start dispersion that production
uses, and it is not the configuration WP5 made fit at C=1,024. Any statement of the form "R1 passes at
C=1,024" must carry that. It does not change the refresh/re-seed conclusion (both sides share the setting), but
it is the largest single untested variable in the R1 story.

## Verdict

**BLOCKING:**
1. Record `orientation`, `iterate`, `metric_refresh`, `metric_refresh_reseed` (and the swap step) as row
   columns so the refresh evidence is self-describing.

**NON-BLOCKING:**
2. Add the no-refresh control at W=100/200 on all four models — the only warmup points where the refresh has no
   comparison at all.
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
