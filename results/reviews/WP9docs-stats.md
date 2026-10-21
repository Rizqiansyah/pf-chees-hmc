# WP9docs — statistical-validity review of the retraction scope

**Provenance: lead-executed, NOT independent** — A4 deviation flagged in `results/STATUS.md`.

Question: does the set of claims this branch retracts match the evidence, and does it leave anything the
evidence does *not* support?

## 1. Are the 7 retractions all in the settled class?

| retracted claim | evidence it rests on | is that evidence in the class "settled without the still-running measurements"? |
|---|---|---|
| "GPU max R̂ rises monotonically with chains" (short chains cost quality) | plan §2b chop test, re-run by WP3 as `tools/wp3_chop_check.py` | **Yes** — the artifact is reproducible from one converged run, independent of the metric work |
| "short chains do not substitute for long ones" (fixed 40,000-draw sweep) | same chop test | Yes |
| "more warmup does not repair what more chains broke" + "a start needs draws-per-chain to burn off" | same, plus plan A2's superseded list | Yes |
| "Fisher's units are 2,000× below the posterior's scale" | `9f892b2` — the three `imm_source` arms never shared an orientation | Yes, code-level, and the report's T2 repeats it |
| "no smaller warmup default" (W=500 vs 2,000) | `c2993cf` GEV 2×2 on a **real model** (24 rows) + `a03dc0f` fixture | Yes for the direction; the branch keeps the 8 Schools same-metric delta standing, which is the correct carve-out |
| "more chains also means less needed warmup" | plan A2 superseded + the grid being the §2b artifact | Yes |
| the two memory-model explanations of the OOM ceiling | already retracted earlier in the run; the mechanism is NOT ESTABLISHED | Yes — and the branch keeps the measured 64-chain ceiling as a fact, which is the right split |

No retraction is resting on an unfinished arm. Two of them (the F-3 warmup-default and the Fisher units) are the
riskiest because they involve metric numbers, and both are justified by the *real-model* GEV rows rather than
only the fixture.

## 2. Is anything over-claimed now?

Two places to check, and one is a genuine gap:

1. **"Retracted" vs "unmeasurable".** The chop test shows short-chain R̂ inflation is an artifact of the
   estimator under chopping; the branch's wording ("retracted as stated; the numbers stay as the artifact's
   evidence") is right and does not claim the opposite conclusion is established. Good.
2. **The chop test's own evidence is not a committed artifact.** `tools/wp3_chop_check.py` **prints to stdout
   and writes no row file** (`grep` for an output path returns nothing; `~/pf/wp3/results/runs/` contains
   only `WP3__selftest.jsonl`, `WP3__nested_k8.jsonl`, `WP3__smoke.jsonl`). Four of the seven retractions
   ultimately rest on numbers that exist today only as console text quoted in the plan and the LEDGER. That is
   the weakest link in the doc retraction chain and it is cheap to fix: redirect the script's output into
   `results/runs/WP3__chop.txt` on `wp/wp3` and cite the file.
3. **CONFOUNDED rows stayed caveated, not retracted** — verified by marker count (20 `pf-r1 caveat` markers
   against 7 uses of "retracted"), and spot-reading the four README spots (lines 72, 74, 78 and the
   warmup-table paragraph at 91–95) confirms each keeps its numbers and states the contamination instead of
   deleting the claim.

## 3. Is the inventory's own verdict respected?

`results/WP9/CLAIM-INVENTORY.md` carries 44 verdict tokens across 40 rows (20 CONFOUNDED, 11 UNAFFECTED,
8 SUPERSEDED, 1 CANNOT TELL). The branch touched only the settled subset: the row marked **CANNOT TELL**
(README:14 "potential to sample much faster") is **not** rewritten — verified, line 15 still reads as shipped —
and UNAFFECTED rows are untouched. That is the correct discipline: a doc branch may not launder an unresolved
row into a conclusion.

## Disposition

**BLOCKING:** none.
**NON-BLOCKING:**
1. Commit the chop test's output as a row/text artifact — four retractions currently cite numbers with no file
   behind them. Do this on `wp/wp3`, which owns the script.
2. The warmup-default retraction should name its own limit in the doc: `c2993cf` establishes the *shipped arm's*
   failure on GEV; it does not by itself establish a specific smaller default. The branch's wording keeps the
   8 Schools delta and does not propose a new number, so this is a nuance rather than an error.
**NIT:** the two label definitions (Retracted / Caveated) are the most useful thing in the new section; consider
putting the same two-line legend in `README.md`, where a reader meets the caveats first.

> **RECOMMEND: merge** — the retractions are the settled subset, the unresolved rows are left alone, and the
> one real weakness (the chop test's unpersisted output) belongs to another branch.
