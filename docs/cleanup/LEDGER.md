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
