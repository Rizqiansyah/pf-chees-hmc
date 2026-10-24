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
