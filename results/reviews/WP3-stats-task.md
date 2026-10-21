# WP3 — statistical-validity review of the R1 bias harness (`tools/r1_bias.py`)

**Provenance, stated up front because it matters.** This review was executed by an independent task
(`wp3-stats-review-task`, worker model, shell available). The task **never wrote its output file** —
it delivered the review in its final message and then died on a tool error. The lead reconstructed
the review from that message and then **re-verified every numeric claim against the row files**. Two of the
task's central numbers are refuted below; everything else stands. Items marked NOT RE-VERIFIED were not
independently reproduced.

Branch reviewed: `wp/wp3` (checkout `/home/akbar/pf/wp3`), including the post-code-review fixes at
`24f2859` and the harness fixes at `9d5c30d`.

---

## The task's findings, with the lead's verification

### 1. b² implementation — correct as specified (task: CONFIRMED)

`bias()` implements `b² = max over parameters and over {x, x²} of (mean_run − mean_ref)² / Var_ref(ddof=1)`,
per parameter, in unconstrained space, with a shared fold rule on both sides. The task recomputed
`b2_max = 0.41240` for the starts-only fixture against a recorded `0.41014` and called it a match within its
1% tolerance.
**Lead: NOT RE-VERIFIED** (the recomputation is a full draw-stack job; the structural claims were
already confirmed by the code review at `results/reviews/WP3-code.md`, which read the same functions).

### 2. Noise floor — optimistic, and the sensitivity numbers are wrong (task: NON-BLOCKING, refuted in part)

The task's formula critique is sound: the floor omits the reference-side MCSE term and assumes independence
across within-chain draws, so it is optimistic by at least √(2 ln 2) ≈ 1.18. That part is a straightforward
read of the implementation.

Its two sensitivity numbers are **refuted by the delivered T1 rows** (`~/pf/int/results/runs/T1__r1_sweep.jsonl`,
48 rows, floors read directly):

| task's claim | measured from the rows | verdict |
|---|---|---|
| "37 of 48 cells (77%) would flip to unresolvable if the floor were doubled" | floors: 29 cells ≤ 0.0025, 13 in (0.0025, 0.005], 6 > 0.005. Doubling moves only the 13 across the 0.005 line, and **4 of 48 cells actually change verdict** | **REFUTED** |
| "all 18 unresolvable cells … 15 of them have b² below the bar" | **6** T1 cells have floor > 0.005, not 18. (The task appears to have used the 30-row WP3 grid as its denominator while quoting "of 48".) | **WRONG DENOMINATOR** |

The *direction* of the critique stands: the floor should carry the reference MCSE and a serial-correlation
inflation factor. It is not load-bearing for any verdict in this run, because the verdicts that matter sit far
from the boundary.

### 3. The claimed BLOCKING item — "unresolvable cells are counted as passes" — REFUTED

The task's blocking argument was: *no radon cell in the delivered T1 set has floor < 0.005, so every radon
cell is unresolvable, and the "3 of 4 models pass" headline counts them as passes.*

Measured: radon floors run **0.0037 – 0.0071**, and **6 of 12 radon cells are resolvable** (floor ≤ 0.005).
Recomputing G-R1 over all 48 cells under the corrected rule (`9d5c30d`):

| arm | 8 Schools | GEV | logistic | radon NC |
|---|---|---|---|---|
| `int:covraw` | 6 pass | 6 pass | 6 pass | 5 fail, 1 pass (W=200 seed 2, b² 0.00709, floor 0.00413) |
| `int:shipped` | 5 pass, 1 fail | 5 pass, 1 fail | **6 fail** | **6 fail** |

So the "3 of 4" statement for the corrected arm rests on genuine resolvable passes (all 12 cells per model,
floors ≤ 0.0036), and radon is reported as failing/not-passing on every seed — which is what `results/REPORT.md`
§3b and §6b/T1 already say. **No acceptance claim in this run rests on an unresolvable cell.** The gate defect
the task was circling is real but it is the *masking* one (a large floor neutralising a large b²), which was
found separately, measured (6 cells change) and fixed at `9d5c30d`.

### 4. Reference-side sensitivity — accepted (task: CONFIRMED)

Smallest true bias resolvable per model, from the references' own MCSE: radon `sigma_raw` 0.0017 sd (min ESS
3,421), GEV `sigma_log__` 0.0036 sd, logistic `beta[0]` 0.0061 sd, 8 Schools `tau` 0.0087 sd. Since b² = 0.01
corresponds to 0.1 sd, the reference is 12–58× more sensitive than the bar in every model. Conclusion accepted:
**sensitivity is not the binding constraint; the run-side floor is.**

### 5. Gradient accounting — accepted (task: CONFIRMED, lead re-verified)

Recomputed over every delivered row carrying a per-chain list: **99 rows, 0 mismatches** between the scalar
and `mean(list)`, and list length equals chain count. The older grid rows (`WP3__selftest.jsonl` at
`f82d1ba`/`7d01a77`, `WP3__smoke.jsonl`, `WP3__nested_k8.jsonl`) have **no** `grads_per_chain_total` at all,
so they must be read as warmup + sampling added by hand. That is already recorded in the LEDGER.

### 6. Strongest remaining silent-wrong path — accepted as NON-BLOCKING

`flatten_run` subsamples to 2,000 rows with an unseeded, shape-dependent index, so b² and the floor can shift
between identical reruns. Cheap fix: seed the subsample. Kept NON-BLOCKING because every delivered cell
either sits far from its floor or is labelled unresolvable.

### 7. Non-blocking / nits — carried forward

Test the `b2_argmax` labelling on the x² half of a **late** leaf; seed the subsample; state why unconstrained
space is the right target for positive-support parameters (`tau`, `sigma_log__`); record reference MCSE and
min-ESS in the manifest; the 2,000-row cap contradicts the docstring's "500 rows per parameter"; fix the
`flatten_run` example's axis order. The review's last non-blocking item — moving `nominal_length` off the
timed path — was measured and fixed at `9d5c30d` (wall times were ~52% high warm).

---

## Verdict as delivered by the task

> **RECOMMEND: merge** — with two conditions: (1) stop treating unresolvable as a pass in the headline count;
> (2) add the reference-MCSE and autocorrelation terms to the floor.

## Lead's ruling

1. **Condition (1) is moot on the data** — no headline in this run counts an unresolvable cell as a pass, and
   the rule that *could* have masked one was fixed at `9d5c30d` before this review landed.
2. **Condition (2) is accepted as a follow-up**, not a merge blocker: the floor's missing terms make it
   optimistic by ~1.18–1.4×, and 44 of 48 T1 cells are ≥ 1.4× away from the resolvability boundary, so no
   delivered verdict is at risk. It must be fixed before any gate decision is made on a cell *near* its floor
   (radon is the model where that would bite).
3. **Merge status:** `wp/wp3` now has two reviews on disk — the code-correctness review
   (`WP3-code.md`, one blocking item, fixed at `24f2859`) and this one. `RECOMMEND: merge`.
