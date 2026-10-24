# F2i SUMMARY: cheaper widening (`_rebuild`, one fold), no memo

Task: fast-tier, one round.
- Branch `f2-i`: 5 commits on 44a0556, head 749cdff. Merged into `f2` with `--no-ff` (a7a4413).
- Task's suite: `575 passed, 64 warnings in 772.33s (0:12:52)`. It was 514; this adds 61 in `tests/test_widen.py`
  (about 70 s), with child runner `tests/widen_run.py`.
- Lead suite on merged `f2`: `575 passed, 64 warnings in 836.55s (0:13:56)`, rc 0 (a7a4413). The F2j
  task's per-module runs overlapped it (load 11–14 at the end), hence the time. The real `~/.cache/pfcheeshmc`
  does not exist after it.

## Lead review
- **Stop rule not triggered.** It saves 0.015–0.046 s per call on every ladder model, above the plan's 0.01 s.
  The library change is +46/−18 lines, against a budget of about 80.
- **Same graph, checked by reading the code:**
  - The batched `constant_fold` gives the same values. PyMC's `constant_fold` runs `topo_constant_folding` on one
    `FunctionGraph` of all the outputs and returns each one's data.
  - The `(dtype, ndim)` filter drops only parameters whose fold could never match. A fold keeps its output type,
    and the fingerprint is (dtype, shape, bytes).
  - `_rebuild` calls `Apply.clone_with_new_inputs(strict=False)` node by node in `toposort` order, the call
    `clone_replace`'s `rebuild_collect_shared` makes. Only the nodes below a replaced constant are remade.
- **Accepted:**
  - the third design (deviation 1);
  - the sharing of the nodes above a widened constant with the model's own graph. The unwidened path always
    returned that graph unchanged, so sharing is not new downstream.
- **Cost:** 61 tests and about 70 s added to the suite. That is a lot for a 46-line change, but most of it is the
  120-case graph comparison against the old code, which is the proof of deviation 1.

## DEVIATIONS (accepted)
1. **Neither design (a) nor (b).** The task did a third: no memo, the same graph, less work. "Every hit is verified
   as strictly as the F2e memo" does not apply, because there is no new hit path. Instead the output graph is proved
   equal to the old one (tests below), so a collision cannot return a wrong function.
2. **Node sharing and names.**
   - `_rebuild` does not clone the nodes above a widened constant, so the widened graph shares them with
     `model.logp()`'s graph.
   - `clone_replace` cloned every node and dropped some variable names. For example, an `Alloc` output lost
     `'obs_logprob'`.
   - Names enter neither `_signature`, `equal_computations` nor the key. `widen_run.py` strips names before it
     compares graph text.
   - A test checks that the model's own graph is unchanged.
3. **Folded dtype.** The dtype of a folded parameter is read from `param.dtype`, not from the folded value's
   `.dtype`. A fold keeps its output type, and the 120 graphs compared against 44a0556 confirm it.
4. **Files beyond the brief:** `tests/test_widen.py` and `tests/widen_run.py` are new. No other file was touched.

## Design
**Chosen: (c)**, in two parts: `_rebuild` (`pfcheeshmc/lifted.py:388`) and `_widen_model_inputs` (`:405`).
- **`_rebuild`** replaces `clone_replace(rebuild_strict=False)`.
  - In `toposort` order it remakes only the nodes below a replaced constant, with
    `Apply.clone_with_new_inputs(strict=False)`, so the types propagate as before.
  - The nodes above are shared. `clone_replace` cloned the whole graph, twice.
- **One fold, and only when it can matter.**
  - The literal-built parameters are folded in one `constant_fold` call, not one call per parameter. The fixed
    cost of a call is about 1.2 ms.
  - A parameter is dropped before folding in two cases:
    - it is not narrower than floatX (`_widened_dtype` says None);
    - no non-scalar constant in the outputs has its dtype and ndim.
  - The initial-point and transform sites usually fold nothing now.

**Why not (a) or (b).** On the new code the widening costs 0.009 / 0.004 / 0.006 / 0.018 s per call (8schools /
gev / logit / radon). The existing F2e signature plus verification costs 0.003 / 0.008 / 0.002 / 0.005 s.
- **(a) would gain at most** about 0.006 / <0 / 0.004 / 0.013 s.
  - It would need a second signature and `equal_computations` pass on the graph before widening.
  - It would still need the fold, because the one-element constants are found by value (fingerprint), not by
    position.
- **(b) would gain at most** the remaining widening, which is under 0.01 s on three of the four models.
  - It would need a second memo level and dtype casts: the narrow-typed lifted graph differs from the F2e memo's
    graph at every node below a widened constant.
  - Equality with today's path cannot be proved from the narrow graph. Fingerprints match by value, so a data
    constant equal to a prior literal is widened in one build and not in another, and a "hit" could return a
    function other than today's.
- Both are well over the 80-line budget for under 0.01 s on most models. They were not prototyped; the bounds come
  from measured components.

## Measurements (task, CPU)
- **Setup:** `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=4`, host load 2.69 at the start and 2.38
  at the end.
- **Command:** `python $tree/tools/lift_cost.py $tree 8schools gev logit_1000x25 radon_noncentered --reps 4`, run
  on two trees:
  - old: `git archive 44a0556`;
  - new: the checkout at 749cdff.

Builds 2–4, seconds, summed over the three sites (the plan's lift totals were 0.07 / 0.11 / 0.05 / 0.14):

| model | widen before → after | lift before → after | saved per call |
|---|---|---|---|
| 8schools | 0.029–0.031 → 0.009 | 0.069–0.073 → 0.052–0.056 | 0.017–0.018 |
| gev | 0.021–0.024 → 0.004–0.005 | 0.106–0.112 → 0.091–0.094 | 0.015–0.018 |
| logit_1000x25 | 0.021–0.022 → 0.006 | 0.049–0.051 → 0.033–0.034 | 0.016–0.017 |
| radon_noncentered | 0.059–0.063 → 0.017–0.021 | 0.140–0.148 → 0.093–0.103 | 0.045–0.047 |

**Micro-benchmarks** (best of 7, ms, on a loaded host):
- **Rebuild alone,** on the logp graph, `clone_replace` → `_rebuild`: 8schools 11.0 → 3.7, gev 13.7 → 2.1,
  logit 7.0 → 1.9, radon 20.9 → 9.3.
- **Folding,** one batched call against one call per parameter: 8schools (2 parameters) 4.3 → 3.1, radon (4)
  8.3 → 4.1.
- **`_widen_model_inputs` per site,** old → new:

  | model | logp | initial point | transform |
  |---|---|---|---|
  | 8schools | 16.2 → 7.7 | 7.6 → 0.7 | 4.7 → 0.3 |
  | gev | 14.0 → 2.4 | 2.8 → 1.5 | 2.0 → 0.2 |
  | logit | 12.2 → 5.4 | 4.4 → 0.0 | 4.3 → 0.0 |
  | radon | 31.4 → 15.2 | 16.8 → 2.0 | 10.6 → 0.7 |

**GPU check (lead):**
- **Setup:** `results/f2/drivers/f2i_gpu.sh`, rows in `results/f2/runs/F2i/`, compared against F2h's same-data
  rows.
  - C=1,024, D=10, cache off, `--calls 3`.
  - 2026-10-24 12:01–12:04, load 11–13 with the F2j task's per-module runs. The F2h rows had load 2.6–3.4.
- **The "rest" stage** holds the three lifts, and with them the widening.

| model | warm wall (calls 2, 3), F2h → F2i | warm "rest" (calls 2, 3), F2h → F2i |
|---|---|---|
| 8schools | 0.242, 0.228 → 0.218, 0.234 s | 0.074, 0.072 → 0.070, 0.079 s |
| radon_noncentered | 7.33, 5.50 → 7.23, 5.54 s | 0.164, 0.131 → 0.123, 0.124 s |
| gev | 0.690, 0.379 → 0.625, 0.384 s | 0.396, 0.106 → 0.346, 0.109 s |

- **Result:** on this host and at this load, the CPU saving of 0.016–0.018 s on 8schools and GEV is within the
  noise. Radon's warm "rest" is 0.02–0.04 s lower, in line with the 0.046 s measured on the CPU.
- **Not repeated** at a lower load: the CPU rows above are the cleaner evidence, and the change is graph-equal by
  proof.
- Call 1 is 1–7 s slower, but its XLA compile time rose by the same amount (8schools 14.5 → 15.4 s, radon
  34.4 → 40.3 s). That is the load.

## Spec items
- **Skip the full-graph clone:** DONE, partly. The nodes above a widened constant are no longer cloned. The nodes
  below one are still remade, because their dtype changes.
- **Skip the fold where possible:** DONE, partly. It is batched, and skipped when nothing can match.
- **Hit verification as strict as F2e:** N/A, because there is no hit path.
- **Fallback on a miss or an exception:** DONE, through the existing chain. If the widening raises, `lift_graph`
  falls back to `model_inputs_native_dtype`. `test_a_failing_rebuild_falls_back_to_the_native_dtype_rule` pins it.
- **Draws bit-identical:** DONE, against 44a0556 in fresh subprocesses.
  - 8schools and gev, two sampler calls each in one process; the second is on a rebuilt model, a memo hit.
  - Every posterior and sample-stats array, and `config`, is equal for both calls.
- **`key`, `fn` identity and `n_lifted` unchanged:** DONE.
  - At the three sites for the 4 ladder models and the 6 extra models, the key, `n_lifted`, rule and kept-static
    report equal 44a0556's.
  - A rebuild returns the same `fn` and key.
  - `tests/test_lift_memo.py` (32) passes unchanged.
- **The memo holds no data:** DONE. There is no new state, and `test_the_memo_holds_no_data` still passes. A new
  test checks that no module-level container grows and that the widened constants are freed with the graph
  (weakref).
- **Autocast boundaries:** DONE, for 4 kinds: scalar, shaped, Poisson/Gamma/Binomial and truncated.
  - Every literal pair of `test_lifted` (0, 0.5, 0.1, 1000, 40000, -3, …) is lifted from one first build.
  - Each later pair is a memo hit: the memo size is unchanged, `fn` and the key are equal, and the dtypes of the
    values are equal.
  - The logp matches PyMC's at 1e-7.
- **Nothing model-specific:** the rules are in graph terms (dtype, ndim, whether a node is below a constant,
  `_widened_dtype`).

## Tests (`tests/test_widen.py`, 61 tests, about 70 s; child `tests/widen_run.py`)
- **The graph equals `clone_replace`'s:** 4 ladder, 6 extra and 30 literal models, at 3 sites each.
- **Only the nodes below a widened constant are remade,** and the old graph is unchanged.
- **Folding:**
  - one fold call (4 expressions on radon);
  - no fold for float64 literals or at the transform site (6 cases);
  - folded one-element constants are matched.
- **Memo hits** across the autocast boundaries (4 kinds).
- **A failing `_rebuild` falls back.**
- **No state is kept.**
- **Against 44a0556:**
  - 120 graph cases and the 209 replaced constants, byte for byte, plus the keys, `n_lifted`, rules and `fn`
    identity;
  - the draws (8schools, gev).

**Must-fail on the old code:** `git show 44a0556:pfcheeshmc/lifted.py > pfcheeshmc/lifted.py`, then
`pytest tests/test_widen.py -k "not draws and not old_code"` gives **12 failed, 46 passed**. The equivalence tests
(graph, draws, keys) pass on the old code by design.

| failing test | excerpt |
|---|---|
| one fold call | `assert (4, 4) == (1, 4)` |
| no fold when nothing matches (6 cases) | `assert 3 == 0`, `assert 2 == 1` |
| only the nodes below are remade (4 models) | `a node no widened constant reaches was cloned` |
| a failing `_rebuild` falls back | `AttributeError: ... has no attribute '_rebuild'` |

**Mutations** of `lifted.py`, each made temporarily and run against the whole of `test_widen.py`:

| mutation | killed by |
|---|---|
| M1 `_rebuild` drops the remade outputs | graph equals `clone_replace`, only-below, old-code graphs, draws, boundary hit (`AssertionError: ('8schools', 'logp')`) |
| M2 remake every node | only-below (`a node no widened constant reaches was cloned`) |
| M3 one fold per parameter | one-fold and no-useless-fold tests (`assert (4, 4) == (1, 4)`) |
| M4 no filter on the graph's constants | no-useless-fold (`assert 1 == 0`) |
| M5 `strict=True` | 9 tests (`TypeError: Cannot convert Type Scalar(float64) ... into Type Scalar(int8)`) |
| M6 the filter ignores ndim | 8 tests (`nothing was widened: the premise of the test is gone`) |
| M7 fold only the first RV's parameters | old-code graphs, matched fold, boundary hit (`assert (1, 1) == (1, 4)`) |

**Existing tests:** `test_lifted.py`, `test_lift_memo.py`, `test_compile_once.py` and the new file together:
`238 passed, 54 warnings in 272.16s`. None of the existing tests was edited.

## UNVERIFIED
- floatX=float32 end to end for the new code. Only `_widened_dtype` is tested there, as before.
- The (a) and (b) bounds come from measured components, not prototypes.
- A graph with a Scan or an `OpFromGraph` below a widened constant. `_rebuild` makes the same
  `clone_with_new_inputs` call `clone_replace` makes, but none of the 120 compared graphs has one.
