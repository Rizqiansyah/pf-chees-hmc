# Task rules — F2 (read before doing anything)

You are an implementation task for F2 of `pf-chees-hmc` (compile once; data and priors as runtime
inputs). The lead reviews your branch and merges it into `f2`.

## Paths
- `MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"` — the lead's tree on branch
  `f2`. **Read only.** Never commit, switch branches, or edit files there.
- Your checkout `WT=~/pf/f2-<wp>` on branch `f2-<wp>` (already created from `f2`). Edit and commit
  **only** there.
- Never commit to `main` or `f2`. Never push. Never touch another checkout.
- **The spec:** `$WT/docs/review-2026-10/pf-chees-hmc-f2-plan.md`.
  - Read "Goal", "Evidence" and "Decisions" in full, then your WP's section.
  - Your WP section, plus the lead's brief in your prompt, is the spec.
  - The probes it cites are in `$WT/results/f2/probes/`.

## Environment (one shared pixi env; CPU only)
```bash
cd "$WT"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=4
pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python "$WT/<script>.py" ...
pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider
```
- **No GPU.** Always export the variables above. Do not call `tools/gpu_run.sh`.
- Do not change `pixi.toml`, `pixi.lock` or the environment. Do not `pip install`.
- No single command over 20 minutes. Run the tests you touch first, and the full suite once at the
  end (about 5 minutes).
- Ladder models: `sys.path.insert(0, "$WT/tools"); import r1_bias; r1_bias.build_model(name)` for
  `8schools`, `gev`, `logit_1000x25`, `radon_noncentered`. Always use `build_model`, never
  `bench_ladder.model_specs()` directly: it leaves sampling stubs installed.

## Fidelity
- If something in the spec is impossible or wrong, do not reinterpret it silently. Implement the
  nearest faithful version, and put the conflict first in your SUMMARY under **DEVIATIONS**.
- **Library-first:** use PyTensor / PyMC / JAX / BlackJAX functions rather than re-deriving them.
  Read library source in the pixi env (`$MAIN/.pixi/envs/default/lib/python3*/site-packages/`)
  before relying on a behaviour.
- **Universality:** nothing may special-case the four ladder models. Every rule must be stated in
  terms of graph structure or PyMC concepts.
- Match the surrounding code: naming, docstring and comment density, idiom.
- Stay inside your WP's file list. If you must touch another file, say so in the SUMMARY.
- Behaviour on `main` must not change unless the spec says so. Where draws can stay bit-identical for
  a fixed seed, they must; prove it with a test.

## Deliverables
1. Commits on `f2-<wp>`. Messages end with a blank line and a commit trailer.
2. A SUMMARY for `results/f2/<WP>/SUMMARY.md`. Do not write this file. Return its content as the last
   section of your final reply, and the lead will commit it. It contains:
   - **DEVIATIONS** (first; "none" if none);
   - each spec item → DONE / DEVIATED / SKIPPED, with `file:line`;
   - tests added. For each test that must fail on the old code: the mutation or old-code command,
     and an excerpt of its failing output;
   - measurement numbers, with the exact command that produced them;
   - the full-suite result line, **copied from the command's own output**;
   - anything you could not verify, marked `UNVERIFIED`.
3. Your final reply: a header of at most 15 lines, then the SUMMARY section. Do not paste diffs. The
   header has:
   - the branch head sha;
   - the suite line, verbatim;
   - the item statuses;
   - any open issue.
4. Commit work in progress at every milestone. If your time budget runs short, stop at the last
   commit and mark the remaining items PENDING.
