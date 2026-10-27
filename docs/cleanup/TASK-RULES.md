# Task rules — repo cleanup (read before doing anything)

You are an implementation task for the repo cleanup of `pf-chees-hmc`. The lead
reviews your branch and merges it into `cleanup`.

## Paths
- `MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"` — the lead's tree on branch
  `cleanup`. **Read only.** Never commit, switch branches, or edit files there.
- Your checkout `WT=~/pf/cleanup-<wp>` on branch `cleanup-<wp>` (already created from `cleanup`). Edit and commit
  **only** there.
- Never commit to `main`, `f2` or `cleanup`. Never push. Never touch another checkout.
- **The spec:** `$WT/docs/cleanup/PLAN.md`.
  - Read "Goal", "Decisions", "Couplings" and "Proposed end state" in full, then your WP's row and section.
  - Your WP section, plus the lead's brief in your prompt, is the spec.
  - Its history is at the tag `archive/pre-cleanup` (b34e66e): `git show archive/pre-cleanup:<path>`.

## Environment (one shared pixi env; CPU only)
```bash
cd "$WT"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=4
pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python "$WT/<script>.py" ...
pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider
```
- **No GPU.** Always export the variables above. Do not call `benchmarks/gpu_run.sh`.
- Do not change `pixi.toml`, `pixi.lock` or the environment. Do not `pip install`.
- No single command over 20 minutes. Run the tests you touch first, and the full suite once at the
  end (about 13 minutes until C3 slims it, then about 5).
- Ladder models (`8schools`, `gev`, `logit_1000x25`, `radon_noncentered`): `benchmarks/models.py`
  (`tools/` is gone since C4). At the tag, a child process uses the tag tree's own `tools/r1_bias.build_model`;
  pick by `os.path.isdir(TREE/benchmarks)`, as the git-archive runners do (C4 finding: `PYTHONPATH` otherwise
  makes the tag side import this tree's models).
- The state before the cleanup, for bit-identity proofs: `git archive archive/pre-cleanup` into a temp dir, run in
  a child process (for example `tests/make_golden.py --tree <dir>`). Since C3 the suite pins the default path with
  `tests/data/golden_*.npz` instead.

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
1. Commits on `cleanup-<wp>`. Messages end with a blank line and a trailer naming your own model, not another one.
2. **Write and commit your SUMMARY yourself** at `docs/cleanup/<WP>/SUMMARY.md` in your branch. It contains:
   - **DEVIATIONS** (first; "none" if none);
   - each spec item → DONE / DEVIATED / SKIPPED, with `file:line`;
   - tests added. For each test that must fail on the old code: the mutation or old-code command,
     and an excerpt of its failing output;
   - measurement numbers, with the exact command that produced them;
   - the full-suite result line, **copied from the command's own output**;
   - anything you could not verify, marked `UNVERIFIED`.
   - If the write step refuses the SUMMARY file, do not work around it: put the SUMMARY after the header in your
     final reply, under a line `--- SUMMARY ---`. The lead commits it verbatim.
3. **Your final reply is the header only, at most 15 lines** (plus the SUMMARY if the write step refused it). Do not paste the SUMMARY or diffs; the lead
   reads the file only where it needs to. The header has:
   - the branch head sha;
   - the suite line, verbatim;
   - the item statuses;
   - any open issue or deviation, one line each.
4. Commit work in progress at every milestone. If your time budget runs short, stop at the last
   commit and mark the remaining items PENDING.
