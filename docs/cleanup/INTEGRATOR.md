# Integrator — repo cleanup (fast-tier; one per WP)

You review one task's branch, and later merge it. The lead decides; you report.
`MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"` is on branch `cleanup`. The task's checkout is
`~/pf/cleanup-<wp>` (branch `cleanup-<wp>`). Read `docs/cleanup/PLAN.md` (Decisions, the WP's row and section),
`docs/cleanup/TASK-RULES.md`, and the task's `docs/cleanup/<WP>/SUMMARY.md` on its branch. Use the pixi env and the
CPU-only exports of TASK-RULES. Never push. Never use the GPU.

## Step 1: review (change nothing)
Check, and give each a verdict (OK / FINDING), with `file:line` for findings:
1. **Scope:** the diff touches only what the WP row allows, plus what the SUMMARY lists as a deviation.
   `git diff --stat cleanup...cleanup-<wp>`.
2. **No behaviour change on the default path:** the bit-identity proof the WP requires exists, runs, and passes.
   Run it yourself. No sampler default changed: grep the diff for changed default values.
3. **Acceptance:** every item in the WP row is met, by evidence you ran or read, not by the SUMMARY's word.
4. **Must-fail evidence:** each new test that claims to catch a regression fails on the old code (check one).
5. **Dangling references:** `grep -rn` for any path, module, or option the WP removed, across the whole tree
   (code, tests, README, docs, notebooks, pixi.toml, pyproject).
6. **Code quality:** it matches the surrounding code; no dead code left behind by the change; no new
   special-casing of the ladder models.
7. **The SUMMARY** is accurate against what you saw.

Report at most 25 lines: one line per check, then the findings, worst first. Each finding says whether it blocks
the merge. Then stop and wait.

## Step 2: integrate (only after the lead says "merge")
1. `cd "$MAIN"`. `git status` is clean, and the branch is `cleanup`.
2. `git merge --no-ff cleanup-<wp> -m "merge <WP>: <one line>"` with the trailer naming your own model.
3. Full suite in `$MAIN`:
   `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 pixi run python -m pytest tests -q -p no:cacheprovider`.
   Record the load before and after (`cut -d' ' -f1-3 /proc/loadavg`).
4. Append to `docs/cleanup/<WP>/SUMMARY.md` a line "Integrator suite on merged `cleanup` (<sha>): `<the result line,
   verbatim>`, rc <rc>, load <a> → <b>", and the step-1 verdicts. Append one dated line to `docs/cleanup/LEDGER.md`:
   the WP, the merge sha, what it did, and the suite line. Commit both on `cleanup`.
5. If the suite passed: `git worktree remove ~/pf/cleanup-<wp>` and `git branch -d cleanup-<wp>`.
6. Report in at most 5 lines: the merge sha, the suite line verbatim, and anything unexpected. If the suite failed,
   do not remove anything; report the failures.
