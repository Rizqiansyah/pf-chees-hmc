# LEDGER — run 2

Decisions with evidence, the D1/D2/D3 rule evaluations, and contradictions (plan A7). Append-only.

---

## 2026-10-20 — run 2 opens

`pf-r2` created from `pf-r1` HEAD `cca5407` per plan A3. Hygiene commit `934e576`: 303 session files
and three `ref/*.pdf` files untracked (kept on disk and now ignored), the three root-level duplicates of
`docs/review-2026-10/*.md` untracked, and the run-2 plan versioned in `docs/review-2026-10/`. History was
not rewritten (owner's call), nothing is pushed, and `main` is untouched at `ecd9ba1`.

`tools/legacy/t3_cpu.py` is a copy of the run-1 T3 driver from `pf-r1-int` with a quarantine header
and an execution guard (`raise SystemExit` on `__main__`, verified firing). Its module body is unedited so
the defect stays auditable: the floor it emits is `MCSE/sd`, **not squared**, and it scored constrained
space, dropping `tau_log__`, `xi_interval__`, `sigma_log__` and radon's three `sigma_*_log__`. Run 1's T3
conclusion ("CPU NUTS cannot resolve b²") is void for this run (plan §1.2). It cannot run as-is in any
case: its `sys.path` and output-path assumptions no longer resolve from `tools/legacy/`.

Environment frozen in `results/r2/env.json`: python 3.13.16, jax 0.11.2, blackjax 1.7.1, pymc 5.28.5,
arviz 0.23.4, numpy 2.4.6; host workstation; 32 CPUs; one NVIDIA RTX A2000 with 6,138 MiB.

**A2 acknowledged.** The nine run-1 claims the plan voids are treated as void for every decision in this
run, including where this branch's own `results/REPORT.md`, `results/LEDGER.md` and
`results/REVIEW-HANDOFF.md` still assert them. R2-WP4 applies the E1–E12 errata in place; run-1 numbers
and row files are not rewritten.

**Pre-registered defaults.** Only D1 (`iterate`) and D2 (Pathfinder covariance source) may flip, and only
on their Part B §3 rules, each recorded clause by clause. D3 (metric refresh) stays opt-in whatever M2b
shows; it gets a recommendation only.

## 2026-10-20 — lead handover (owner's request)

the lead stopped after R2-WP0 items 1, 2 and 4 because of contention on its backend. The baseline suite it
had started did not report, and the R2-WP1 task left uncommitted work in `~/pf/r2-wp1`. At the
owner's request, the lead orchestrates the rest of run 2 with parallel tasks: **fast-tier** (R2-WP1/2/3 and code reviews) and **steady-tier** (R2-WP4,
enumerations). M-steps are launched by the lead through `tools/gpu_run.sh` driver scripts kept
in `results/r2/drivers/`. Every plan rule is otherwise unchanged. The R2-WP1 task's uncommitted diff
is carried forward and checked against the plan, not discarded. R2-WP3 is split into 3a (new files:
`tools/t3_cpu_nuts.py`, `summarise_rows.py smallest-pass`, `tests/test_t3_cpu.py`) and 3b (`r1_bias.py`
passthrough/pool-seeds, after R2-WP2 merges) so the file-disjoint half starts in wave 1.

Baseline suite on `pf-r2` at `4e07b45` (forced CPU, A3 command), copied from the command's own output: `125 passed in 165.79s (0:02:45)`.
