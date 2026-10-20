# Task rules — run 2 (read before doing anything)

This is a run-2 implementation task for `pf-chees-hmc`. The lead merges your
branch after one review. These rules restate plan Part A for tasks.

## Paths
- `MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"` — the measurement tree on `pf-r2`.
  **Read only.** Never commit, switch branches, or edit files there.
- Your checkout `WT=~/pf/r2-<wp>` on branch `r2/<wp>`. Edit and commit **only** there.
- Never commit to `main` or `pf-r2`. Never push. Never touch another task's checkout or the run-1
  checkouts (`~/pf/{int,legacy,wp*}`).
- The plan: `$MAIN/docs/review-2026-10/pf-chees-hmc-run2-plan.md`. Read Part A §A0, §A4 and Part B
  §5, then your WP's section in Part B. Your WP section is the spec.

## Environment (one shared pixi env; CPU only)
```bash
cd "$WT"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python "$WT/tools/<script>.py" ...
pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider
```
- **No GPU.** Always export the three variables above. The GPU is reserved for the lead's
  measurement steps. Do not call `tools/gpu_run.sh`.
- Do not change `pixi.toml`, `pixi.lock` or the environment.
- No single command over 20 minutes. Run tests you touch first, the full suite once at the end.

## Fidelity
- Blocks marked ⟦VERBATIM⟧ in the plan are the specification. Implement them exactly. If one is
  impossible or wrong, do not reinterpret it: implement the nearest faithful version, and put the
  conflict first in your SUMMARY under **DEVIATIONS**.
- Library-first: call BlackJAX / ArviZ / arviz-stats / NumPy / SciPy functions rather than
  re-deriving them. Read library source in the pixi env when the plan names a function.
- Match the surrounding code: naming, docstring and comment density, idiom.
- Stay inside your WP's file list. If you must touch another file, say so in the SUMMARY.

## Deliverables
1. Commits on `r2/<wp>`. Messages end with a blank line and a commit trailer.
2. `$WT/results/r2/<WP>/SUMMARY.md` (for example `results/r2/R2-WP1/SUMMARY.md`), committed on your
   branch, containing:
   - **DEVIATIONS** (first; "none" if none);
   - each numbered plan item for your WP → DONE / DEVIATED / SKIPPED, with `file:line`;
   - the formula you implemented, restated verbatim, with `file:line` (code WPs with a ⟦VERBATIM⟧
     block);
   - tests added; for each test the plan says must fail on the old code, the mutation command and
     an excerpt of its failing output;
   - smoke or measurement numbers, with the exact command that produced them;
   - the full-suite result line, **copied from the command's own output**;
   - anything you could not verify, marked `UNVERIFIED`.
3. Your final reply to the lead: at most 15 lines. Give the branch head sha, the suite line
   verbatim, the item statuses, and any open issue. Do not paste diffs.
