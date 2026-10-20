# R2-WP4 SUMMARY — errata, coverage, hygiene

Written by the lead from the task's final report. The task's own write of this file was
refused by the subtask harness; content is the task's, with the lead's check of E3c added.
Commits: `c17df82` (errata E1–E12 in REPORT and REVIEW-HANDOFF, E12 tag in LEDGER), `45fd624`
(RUN1-COVERAGE.md). Changed vs `4e07b45`: `results/REPORT.md`, `results/REVIEW-HANDOFF.md`,
`results/LEDGER.md` (tag only), `results/r2/RUN1-COVERAGE.md`. No row files, code or tests touched.

## DEVIATIONS
1. **E6 deleted, not corrected.** No row file for "Pathfinder as init vs as warmup" exists in LEDGER,
   REPORT, REVIEW-HANDOFF, STATUS, WP*/SUMMARY, reviews, or the arm labels of the 21 run files. The
   row is deleted; its verbatim text is kept in the errata line so it can be restored.
2. **E3c narrowed.** The radon `covraw` cell now reads "fails resolvably at W=50/100, mixed at W=200",
   not the plan's "fails resolvably 6/6". Lead check (`results/runs/T1__r1_sweep.jsonl`,
   arm `int:covraw`): W=200 seed 1 b² 0.02339 / floor 0.00477 (fail), seed 2 b² 0.00709 / floor
   0.00413 — floor ≤ 0.005 and b² < 0.01, so `g_r1` passes it. The plan's E3c text was wrong; the
   task is right.
3. E7 dedupe kept the later row of each pair; the GEV `xi_interval` claim is folded into the
   surviving START row.
4. E11's correction text is in the errata block only; inline it is a tag (per the brief).
5. E12's tag is at `results/LEDGER.md:174`, outside the two named files (E8–E12 tagging named no
   file restriction).

## Items
| item | status |
|---|---|
| 1. E1–E7 inline | DONE (E1, E2, E3a/b, E4, E5, E7 applied; E3c narrowed, E6 deleted — see DEVIATIONS) |
| 2. Errata (run 2) block E1–E12 at top of both files + inline void tags E8–E12 | DONE |
| 3. No run-1 numbers rewritten, no row files touched | DONE |
| 4. `results/r2/RUN1-COVERAGE.md` | DONE — 31 rows: DONE 9, DEVIATED 13, SKIPPED 9 (WP5.1 SKIPPED) |
| 5. `.gitignore` check | DONE — `git status --porcelain ref` printed nothing (exit 0); `.gitignore:208-209` |

Per erratum: E1 REVIEW-HANDOFF §3 item 2 (checked `sampler.py:495,589`); E2 REPORT §8 caveat 6; E3a
shipped GEV "fails W=50 on one seed (b² 0.087), passes W=100 (§3b)"; E3b 335 moved from GEV to
logistic; E4 §3e item 3 "cold rows" (42/42 `cold: true` verified); E5 §7 169.28 s → 176.88 s, "122
collected" removed; E8 REPORT T3 paragraph and HANDOFF §1 T3 row; E9 REPORT §3e items 1–3, §3g item 3,
§7 item 3; E10 REPORT §3e, §7 item 5, §8 caveat 5, HANDOFF §3 item 6; E11 REPORT §4 item 3, T2 iterate
row, HANDOFF §1 ChEES row; E12 LEDGER `:174`.

RUN1-COVERAGE judgement calls: WP1.2 DEVIATED (permutes the expansion, not the factors; equivalence
tested); WP4 DEVIATED (two named tests missing); WP5.3 DONE with caveat (`pm_chains` not forwarded);
WP8 and WP9 DEVIATED. Bullet-only WPs have one WP-level row.

Unlisted stale passages, not edited: REPORT §3f l.164 "defaults to None" (same as E1); WP4 SUMMARY §5
"6 of 6" (same as E3c); REPORT T1 radon "fail at every W" vs §3b "unresolvable"; REPORT §2.3 l.44;
REPORT T3 "Coverage is also incomplete" (adjacent to E8).

Suite (task, forced CPU, copied from output): `125 passed in 176.20s (0:02:56)`.

UNVERIFIED: whether an "init vs warmup" row file exists outside the locations searched (E6).
RUN1-COVERAGE line numbers refer to `pf-r1` before the errata (read via `git show`).
