---
name: hmc-perf-change
description: Evidence-first workflow for performance or algorithm changes to this GPU HMC sampler (BlackJAX + PyMC). Use when editing hot paths in pfcheeshmc/sampler.py or pfcheeshmc/utils.py, swapping kernel families, changing warmup or the mass matrix, or claiming a speedup versus CPU NUTS. Fixes the measurement protocol and the review gates before the code.
---

# HMC performance change loop

Order is not optional: baseline → discriminating probe → implement → test → non-regression
matrix → two review lanes → one commit. Never benchmark before the protocol in §2 is fixed.

## 1. Sequence

1. Read the current code path and the library source it rides on. Note every private API used.
2. Reproduce a baseline with the **public default arguments**, not just the example settings.
3. Run one cheap probe that can kill the idea in under 2 minutes before implementing it
   (e.g. is this stage compile-bound? call it twice in one process; what does this kernel cost
   per gradient?).
4. Implement the smallest change that works. One stage at a time.
5. Add a test for each failure mode discovered while implementing, not only for the happy path.
6. Run the matrix in §2. Then scope per §4 and spawn the two review lanes in §5. Commit only after
   both clear.
7. Record results in `/home/akbar/pf-research/FINDINGS.md` and retain the outcome to project memory.

## 2. Measurement protocol (fixed, do not improvise per run)

- Objective metric: **bulk ESS per wall-clock second, warmup and compile inside the denominator**.
- Always report **cold vs warm**: first run in a process costs 1.4–2.1x here. Never mix them.
- Divergences: **per 1,000 draws, compared at equal chain count**. Raw counts scale with chains.
- Quality gates: max R-hat ≤ 1.01, per-variable tail-ESS ≥ 400, E-BFMI ≥ 0.3 (undefined if the
  energy array has non-finite values — say so, do not print a number), no increase in divergence
  rate at the same chain count.
- Seeds: ≥ 5 per cell, report median and min/max. Never best-of-N. Cells: model × chain count.
- Missing baseline: create it from the pre-change commit with
  `git worktree add /home/akbar/pf-preW1 HEAD` and run the harness there with
  `PYTHONPATH=<worktree>`. `PYTHONPATH` wins for a script run, not for `python -c` inside the repo.
- Stage timing: wrap the stage with a `perf_counter` + `jax.block_until_ready` monkeypatch and
  record both share and absolute. Shares are the decision quantity; instrumented totals are
  inflated versus uninstrumented runs.
- Harnesses live outside the repo: `/home/akbar/pf-baseline.py`, `pf-gate*.py`, `pf-chains.py`,
  `pf-w1-verify.py`, `pf-default-check.py`. Results append to `/home/akbar/pf-runs/*.jsonl`.

## 3. Verification rules

- **No claim without the line.** Any `file:line` assertion, mine or a subagent's, is re-read before
  it is used. Two shipped examples: a subagent "blocker" about `vmap` over a batched `while` loop was
  false — JAX re-evaluates the per-element predicate in the body and selects the old carry
  (`jax/_src/lax/control_flow/loops.py:2210-2224`), confirmed by a 1-minute CPU experiment.
- **Equal small counts are not parity.** For ~33 events Poisson sigma is ~5.7; a single-seed
  match proves nothing. Require ≥ 3–5 seeds before calling non-regression.
- **A faster sampler that changes diagnostics is rejected**, regardless of speed. Warm-start and
  adaptation changes do not change the target but they do select which frozen (step size,
  trajectory length) the run lands on. Report the frozen values and the realized leapfrog-step
  distribution, not only ESS.
- Rejected designs get written down with the measurement that killed them, so they stay dead.

## 4. Scope before spawning

Write a question map before any lane starts. Each row: numbered question, who answers it (me or a
named lane), the evidence that would settle it, which decision it gates, and its cost. Then derive
the lanes from the rows. Rules:

1. **Gate test for a lane:** at least two questions whose answers could change the design. A lane
   that only adds confidence is a lookup I do myself.
2. **No overlap.** Two lanes must not be assigned the same question. Check the map, not memory.
3. **Cheap answers are mine.** Anything answerable by one `read`, one `ls`, or a 2-minute probe goes
   on my list, not a lane's.
4. **Decision-gating claims get verified by me, in the source, before implementation starts** — not
   by the lane that produced them. This is the step that catches a confident but wrong blocker.
5. **Absence claims need a stated method.** "Not present" is only accepted from enumeration (`ls` +
   `read`), never from a directory grep under `.pixi` (see §6).
6. **Cap check before spawning:** 2 × kv520-flash and 3 × kv520 running at once; `kv520` for bounded
   lookups and enumerations, `kv520-flash` for literature and multi-source reasoning.
7. Every lane brief ends with the deliverable shape (table, verdict list) so results are mergeable
   without a follow-up pass.

## 5. Review lanes

- Concurrency caps, hard: **2 × kv520-flash** and **3 × kv520**. Budget is unlimited; do not ask
  lanes to truncate or deliver partial work, but do demand a verdict when the work is done.
- Two lanes per change: one code-correctness (`reviewer`), one statistical-validity (`planner`
  or `researcher`). Spawn with `wait:false`, collect with `agent_output`, re-arm long lanes with
  `agent_steer` and `timeoutMs` only (max 300000). Never re-launch a running lane.
- Briefs must be self-contained and read-only: no file edits, no sampling, no builds, no GPU.
  Require `file:line` evidence, evidence tiers, and `NOT ESTABLISHED` instead of inference. Tell
  them explicitly to invite rebuttal of the framing.
- Lanes have **no web or MCP tools**: retrieval is `curl` from bash. Say so in the brief.

## 6. Box traps (each one already cost time once)

- Python only through `pixi run python`. Bare `python`/`python3` lack the project packages. The
  pixi env has no `pip`, and this pixi version rejects `--with`, so run tests through their
  `__main__` blocks: `CUDA_VISIBLE_DEVICES="" pixi run python tests/test_x.py`.
- The `grep` tool respects `.gitignore`, and `.pixi/` is ignored: a directory grep under
  site-packages silently searches nothing and yields false absences. Use `ls` + `read` there.
- JAX: `jax.random.split(key, n)` returns keys shaped `(n,)`, not `(n,2)`; `jax.random.split`
  inside a `vmap` raises. Pre-generate keys as `(ntrans, nchains)`.
- `pixi run python -c "import pkg"` resolves against the repo cwd, which shadows `PYTHONPATH`.
- `sample_pf_chees_hmc` defaults are in the public signature, not the internal stage
  `setdefault`s. Test the public default; the internal ones may be unreachable.

## 7. Commit rules

- One commit per item, message states what changed, the rule that did *not* change, measured
  numbers with cold/warm status, diagnostics, and what is still unverified.
- Do not commit `ref/*.pdf` or `.pi/` without an explicit decision.
- Do not push unless asked; the box key is not on the remote.
