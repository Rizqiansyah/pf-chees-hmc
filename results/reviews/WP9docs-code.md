# WP9docs — code-correctness review (documentation retractions branch)

**Provenance: lead-executed, NOT independent** — an A4 deviation, flagged in `results/STATUS.md`.
Every count below was produced by a script in this session (`~/scratch/wp8_wp9_verify.py`), not taken from the
task's report.

Branch `wp/wp9docs` @ `79e9944`, one commit from `e660578`, two files: `README.md` and
`docs/ladder-results.md`.

## 1. Scope discipline — CONFIRMED

`git diff --name-only` against the branch point shows `README.md` and `docs/ladder-results.md` as the only
source files changed; no `pfcheeshmc/`, `tools/`, `tests/` or `results/` file is touched. `README.md:9` — the
sentence owned by `wp/wp8` (`f62a22f`) — is **not** edited by this branch: line 9 still reads the WP8 wording,
and the branch's caveats sit at lines 72/74/78/93/95. No ownership collision.

## 2. Citation integrity — CONFIRMED

- **Commit references:** 6 distinct 7–8-character SHAs are cited in the new text. All 6 resolve in this
  repository (`git cat-file -t` succeeds on each). Nothing like the fabricated `066986c` / `58d582c` that two
  other tasks in this run invented.
- **Line pointers:** 27 written. Checked mechanically against the changed lines: 21 land on a line this branch
  modified in `docs/ladder-results.md`; the remaining 6 point into `README.md` (72, 74, 78, 91, 93, 95) and
  were checked individually — every one lands inside a paragraph that now carries a `pf-r1 caveat` marker.
  **No pointer is stale.**
- **Row-file references:** the new prose cites no `.jsonl` path, so there is no dangling-data risk (verified by
  regex over the whole file).

## 3. Counts, as measured rather than as reported

| item | task reported | measured in this session |
|---|---|---|
| `pf-r1 caveat` markers | 19 spots | **20** (README 4, ladder 16) |
| the word "retracted" | 7 | **7** (README 1, ladder 6) |
| lines differing from `pf-r1` | — | 139 of 682 in the ladder doc |

The one-off in the caveat count is a marker phrased differently from the others; no spot is missing, and no
retraction exceeds the 7 the task listed.

## 4. Structure the reader needs — CONFIRMED

`docs/ladder-results.md:9` opens a `## Corrections applied in pf-r1` section that states the audit trail up
front (LEDGER `a03dc0f` orientation retraction on the 2×2 fixture, `c2993cf` the same confirmed on GEV with
R̂ 1.0313–1.0518, `9f892b2` the three `imm_source` arms never sharing an orientation) and defines the two
labels it uses: **Retracted** = a measurement refutes the claim, numbers stay as the refuting evidence;
**Caveated** = the numbers stand as measured but their interpretation is contaminated. That distinction is the
whole point of the branch and it is stated before any edit, which is the right place for it.

## Disposition

**BLOCKING:** none.
**NON-BLOCKING:** the branch's own caveat markers are free text (`**pf-r1 caveat:**`), so nothing can enforce
or count them in CI; if the doc keeps growing, a machine-checkable tag would be worth it. Not now.
**NIT:** the task's reported count (19) differs from the file's (20) by one differently-phrased marker — the
kind of small drift that makes a self-reported number unusable as evidence, which is why this review counted.

> **RECOMMEND: merge.**
