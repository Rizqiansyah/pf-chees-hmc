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
