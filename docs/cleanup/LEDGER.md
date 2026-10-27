# Repo cleanup — ledger

- 2026-10-24 — Scope written (`PLAN.md`, then `docs/cleanup-plan.md`, f2 9a236ba). The owner answered:
  - history: tag and delete only;
  - tools: a slim `benchmarks/`;
  - the API: prune the plan's list;
  - float32: "Opt in, caveats (as you suggest), but make examples use them to demonstrate. I have clear ideas why
    it fails in GEV and we should be able to mitigate. I will leave this mitigation for future me, but a clear and
    loud opt in doc and examples will be helpful".
  - Standing: autonomous, one task at a time, never push, nothing filed upstream.
- 2026-10-24 — C0: tag `archive/pre-cleanup` → b34e66e (local `main`; local only). Branch `cleanup` from f2 9a236ba
  (= main + the scope). Plan, task rules and ledger in `docs/cleanup/`.
- 2026-10-24 — C1 task launched (`~/pf/cleanup-c1` from e087a2f): the four ladder models in Python (`benchmarks/models.py`); proof against the notebook-built models at the tag; consumers switched; notebook loader removed.
- 2026-10-24 — Owner: "Pause after the current subtask finish. We are running out of rate limit very soon."
  When C1 reports, save its SUMMARY to `docs/cleanup/C1/SUMMARY.md` and stop. **To resume:**
  - review C1 (`~/pf/cleanup-c1`, branch `cleanup-c1`) and merge it into `cleanup` with `--no-ff`;
  - run the full suite;
  - launch C2 (the API prune, Q3 list in PLAN.md);
  - continue in the PLAN's order.
  If C1 stops at the time limit, resume the paused task; do not restart it.
- 2026-10-24 — Owner: "use a smaller model for most of these simpler tasks too". The PLAN now has "Who does what":
  - fast-tier tasks implement, and commit their own SUMMARY;
  - a fast-tier integrator per WP reviews (`INTEGRATOR.md` step 1), then merges and runs the suite (step 2);
  - steady-tier runners do the GPU notebook runs and the final suite;
  - the lead writes the briefs and decides.
  C1 was told of the SUMMARY change by message. The pause still holds: when C1 reports, launch nothing until the
  owner says go. Then the C1 integrator, step 1.
- 2026-10-24 — C1 task reported: head 53ad265 on `cleanup-c1`; SUMMARY committed there (`docs/cleanup/C1/SUMMARY.md`). Task suite `620 passed, 63 warnings in 828.14s`. Models proved equal to the tag's notebook-built ones at both floatX (graphs, logp/dlogp at 100 points, draws over 102 arrays). Paused per the owner: **next is the C1 integrator, step 1.**
- 2026-10-24 — C1 merged into `cleanup` at 1e08876 (`--no-ff` of cleanup-c1 53ad265): the four ladder models in Python (`benchmarks/models.py`), proved bit-identical to the notebook-built ones at the tag (graphs, logp/dlogp at 100 points, draws, both floatX); notebook loader and its x64 workaround removed. Integrator suite: `620 passed, 63 warnings in 830.07s (0:13:50)`, rc 0. Findings 1–3 (stale stub comments and trivial guards; unused centered radon branch and build_logistic parameters; blas__ldflags workaround for bias.py) carried to C4. Paused per the owner until told to go: next is C2.
- 2026-10-24 — Owner: "Continue autonomously". C1 integrator (fast-tier): step 1 found nothing blocking; merged as
  1e08876, suite `620 passed, 63 warnings in 830.07s (0:13:50)`; findings 1–3 carried to C4. C2 task (fast-tier)
  launched in `~/pf/cleanup-c2` from 432aaac: verify the Q3 list, remove it, TypeError on removed options,
  default-path draws bit-identical against the tag (child-process proof; C3's golden fixture replaces it).
- 2026-10-24 — C2 merged into `cleanup` at 9c5fe20 (`--no-ff` of cleanup-c2 0c18874): the Q3 experiment options are removed (`imm_orientation`, `imm_order`, `metric_refresh*`, `superchains` and nested R-hat, `iterate`, `init_pick`, `pf_start_scale`, `pf_center`; `ensemble_metric.py` deleted; every stage raises `TypeError` on a key it does not read); default-path and kept-option draws bit-identical to the tag (`tests/test_api_prune.py`). Integrator suite `587 passed, 53 warnings in 846.23s (0:14:06)`. Carried to C3: `without_removed` is imported by 5 modules from `test_api_prune`; carried (unverified): float32/GPU.
- 2026-10-24 — C2 task reported (b7ce5cd; 9 options removed, `pfcheeshmc/` 4,302 → 3,709 lines; proof 0 differences
  over 498 arrays against the tag). The task could not write its SUMMARY; the lead committed the task's text
  verbatim (0c18874) and TASK-RULES now says to return it in the reply. Integrator: nothing blocking; merged 9c5fe20,
  suite `587 passed, 53 warnings in 846.23s (0:14:06)`. C4 task launched in `~/pf/cleanup-c4` from 4951526
  (`benchmarks/` as the PLAN lists, no `tools/`; acceptance: identical b² to the tag's `r1_bias` on 8schools C=64).
- 2026-10-24 — Owner: "Pause after this subtask". When C4 reports, commit its SUMMARY if the task could not, and
  stop. **To resume:** the C4 integrator, step 1 (`~/pf/cleanup-c4`, branch `cleanup-c4`); then C3, C5+C6, C7, C8
  in the PLAN's order. If C4 stops at the time limit, resume the paused task; do not restart it.
- 2026-10-24 — C4 task reported: code head 58cb898 on `cleanup-c4`; SUMMARY committed by the lead (6aa4bc9;
  the task could not write it). Task suite `573 passed, 56 warnings in 848.38s (0:14:08)`. `tools/` gone;
  `benchmarks/` has the PLAN's list plus `compile_counter.py` and `seeded_models.py`. The new `bias.py` gives b²_max
  and draws identical to the tag's `r1_bias` (8schools C=64 seeds 1–3; gev, logit, radon spot checks), and
  reproduces F2f's GPU b² at C=1024 to ~1e-14. Task finding: since C1 the tag-side children of the git-archive
  proofs imported this tree's `benchmarks/models.py` through `PYTHONPATH`, so their model side was vacuous; the runners
  now pick the builder by the tree, and the proofs pass. For the integrator: stale paths left at
  `pfcheeshmc/compiled.py:323`, `sampler.py:100,104`, `tests/conftest.py:23`, `README.md:130` (C5/C6).
  **Paused per the owner: next is the C4 integrator, step 1.**
- 2026-10-27 C4 merged (8446a28): benchmarks/ is the harness (bias.py bit-identical to the tag's r1_bias on 8schools; references.json; README), tools/ gone, git-archive runners pick the tag builder by tree. Suite: 573 passed, 56 warnings in 913.23s (0:15:13). Carried: README lift-cost citation (C6); stale paths in compiled.py:323, sampler.py:100,104, conftest.py:23, README.md:130 (C5/C6).
- 2026-10-27 — C4 merged into `cleanup` at 8446a28 (record c83f14d); integrator suite `573 passed, 56 warnings in 913.23s (0:15:13)` at load ~35 (shared host). Carried to C6: the README lift-cost number has no command. PLAN/TASK-RULES updated for C4 (216bae3). C3 task launched in `~/pf/cleanup-c3` from 216bae3: golden draws (checked equal to the tag), git-archive proofs dropped, coverage diff via a throwaway sys.monitoring plugin.
- 2026-10-28 — C3 merged into `cleanup` at cd2206b (`--no-ff` of cleanup-c3 54fe8a8): git-archive proofs replaced by 18 golden npz files (tests/test_golden.py, make_golden.py; equal to the tag), tool tests dropped, small modules merged; no package line loses coverage (1738/1804 before and after). Integrator suite `542 passed, 48 warnings in 520.33s (0:08:40)`. 5-min goal missed, accepted; TASK-RULES:33 stale (lead fixes).
- 2026-10-28 — C3 accepted and merged (cd2206b; integrator record 95d201f; suite `542 passed, 48 warnings in 520.33s (0:08:40)`; 5-min goal missed at load ~33, accepted). TASK-RULES widen_run line fixed (b42af4d). C5+C6 task launched in `~/pf/cleanup-c56` from b42af4d. Lead decision: `imm_source` stays, README makes no performance claim for it.
2026-10-28 C5+C6 merged at ad40567: results/, old docs, workflow-skill dir deleted; docs/design.md and docs/upstream-drafts.md added; pyproject 0.2.0 with cuda13 extra; README rewritten. Suite: 542 passed, 48 warnings in 460.34s (0:07:40).
