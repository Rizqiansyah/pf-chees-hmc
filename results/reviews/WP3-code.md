# WP3 — code-correctness review

**Reviewed:** `/home/akbar/pf/wp3` @ `d080195` (`tools/r1_bias.py`, `pfcheeshmc/sampler.py`,
`tests/test_r1_bias.py`, `tests/test_nested_rhat.py`, delivered rows
`results/runs/WP3__selftest.jsonl`, `results/runs/WP3__nested_k8.jsonl`).
**Verdict: CHANGES REQUIRED** (one blocking item; the harness itself is sound and the branch is
mergeable once it is documented).
Read-only review: no runs, no edits outside this file.

## Eight numbered verdicts

**1. One normalisation rule, used identically on both sides — CONFIRMED, with a structural caveat.**
The rule is "fold axes 0,1 (chain, draw) into the row axis, fold the remaining axes into leaf
columns, order variables by sorted name, label leaves `name` or `name[j]`". Implemented three times:
`tools/r1_bias.py:201-227` (`flatten_run`, asserts `(chains, draws, *)` then
`reshape(chains*draws, -1)`), `tools/r1_bias.py:229-247` (`flatten_ref`, requires `(rows, leaves)`
and equal row counts), and `tools/r1_bias.py:160` inside `load_reference`
(`reshape(prod(shape[:2]), -1)`, the fold that turns the `.npz` into the `(rows, leaves)` form
`flatten_ref` expects). The two sides are not the same function, but the composition
`load_reference → flatten_ref` is algebraically identical to `flatten_run`, and
`score_ensemble` compares the leaf label lists and raises on mismatch
(`r1_bias.py:373-375`), so a divergence in either direction fails loudly.
`tests/test_r1_bias.py:138-147` pins both label lists and the row count. Highest-severity finding
available here (one-sided fold) is REFUTED.

**2. Row-count assertions per dim branch, especially the scalar branch — CONFIRMED.**
`tests/test_r1_bias.py:333-375` asserts counts, not shapes: `ens.shape == (C*D, 87)` for a fixture
with a 2-D scalar (`sigma_y_log__`), a 3-D scalar (`mu_alpha`, `(C,D,1)`) and a 85-leaf vector
(`alpha_raw`), plus `total_elements == C*D*87`, and then it biases **one scalar leaf by a whole sd**
and requires the gate to name it (`:374`). `tests/test_r1_bias.py:378-389` pins 2-D vs 3-D scalar
equivalence, and `:391-399` pins that a `(draws, chains)` transposition raises instead of reshaping.
This is the class of test the earlier bug needed.

**3. `b2_argmax` index → name — CONFIRMED correct, test has moderate teeth.**
`bias()` returns `argmax` over `concatenate([b2m, b2v])` (`r1_bias.py:259-273`), so `k < d` is the x
half and `k ≥ d` the x² half, both in identical leaf order; `argmax_name` (`r1_bias.py:249-257`)
maps `k % d → labels[k % d]` and appends the moment. `test_r1_bias.py:124-135` asserts the printed
name is in `labels` and that the x² moment wins when variance is wrong; `test_r1_bias.py:30-38`
asserts `k == 1` for a shift in leaf 1. Gap: no test asserts the **x² half of a late leaf**, i.e.
`k = d + (d-1)` naming; an off-by-one-block error would only be caught by the scalar test's
`"sigma_y_log__ (x)"` string, which sits in the x half.

**4. Noise floor formula — CONFIRMED as specified, mildly optimistic.**
`r1_bias.py:268-270` implements
`max(max_f mcse_mean²/var_ref, max_f mcse_x²mean²/var_ref(x²)) · 2·ln(2d)` with
`d = ens.shape[1]`, the model's flattened leaf count (8 Schools d=10, GEV d=3), matching the brief's
text, and `test_r1_bias.py:53-63` pins it to 1e-9 against the closed form.
Two omissions make it slightly optimistic, both small and both worth one sentence in the docstring:
the reference's own Monte-Carlo error is not propagated (denominators come from 8×10,000 draws,
relative error ~0.5%), and `mcse` is computed on the run stack assuming the chain means are
uncorrelated across chains — true here, but it would silently understate the floor for the
`superchains` arm, where chains inside a superchain share a start and are therefore *not*
independent.

**5. Scoring in constrained space by accident — CONFIRMED prevented.**
`force_unconstrained()` (`r1_bias.py:354-364`) wraps `_to_inference_data` to inject
`keep_untransformed=True` for **every** arm in this process, so the legacy and new-code paths get the
same variable set by construction rather than by two implementations. Names come from the model, not
from a file: `[v.name for v in model.value_vars]` (`r1_bias.py:404-405`), deliberately unsorted
(documented at `r1_bias.py:103-111`). A wrong space cannot pass silently: leaf labels must match the
reference exactly (`r1_bias.py:373-375`), and GEV's `sigma_log__`/`xi_interval__` or radon's raw
effects would produce a different label list. Caveat, not a defect: because it is a monkeypatch, a
future public `keep_untransformed=False` default in new code would still be overridden here, so rows
do not prove the public option works.

**6. Exact gradient counts — REFUTED against A8's wording; the missing key is real but benign.**
`r1_bias.py:463-467` writes `float(np.median(ch["grads_warmup_per_chain"]))` and
`float(np.median(steps.sum(1)))`. The per-chain exact sums are computed one line earlier
(`steps.sum(0).tolist()` in `instrument()`, `r1_bias.py:341-346`) and then **discarded** — a median
across chains is not "the sum of `num_integration_steps`". In lockstep ChEES the values are near
identical across chains, so the reported number is unlikely to be wrong by much, but the run's R1
objective is gradients-per-chain and the row loses the spread that would reveal a chain stuck at a
long trajectory. No double counting: warmup comes from the captured warmup `HMCInfo`, sampling from
`idata.sample_stats["num_integration_steps"]`, which are disjoint. Two additional sharp edges:
`steps.reshape(...)[..., 0]` (`r1_bias.py:457`) silently drops every leaf but the first if the sampler
ever emits a vector-valued step count, and `grads_per_chain_total` is written only in the success
path (`r1_bias.py:496`), added in `542c12c` — which is why the 30 grid rows carry `git_sha f82d1ba`
and no such key at all. The task's claim that T1 needs no addition is true going forward and false
for the delivered file; a reporting script must fall back to warmup+sampling for older rows.

**7. `superchain_ids` and the flat ~1.004 — ids CONFIRMED correct; the estimator is legitimately
blind, and here is the arithmetic.**
`sampler.py:391-399` validates K, requires `chains % K == 0`, and builds starts as
`jnp.repeat(pick_global[:K], chains // K)` — consecutive blocks of C/K chains per start.
`r1_bias.py:387` builds `ids = [i // (chains // superchains) for i in range(chains)]` — the same
consecutive-block grouping. They match, and `test_r1_bias.py:205-220` pins both halves (chains 0 and
1 share a start byte-for-byte, chains 0 and 2 differ, non-divisible K raises). The axes handed to
`rhat_nested` are `(chain_axis=0, draw_axis=1)` on a `(C, D, d)` stack (`r1_bias.py:379-389`), which
is the same call shape the lead verified against the library.
Why nR̂ ≈ 1.004 regardless of bias: R̂-family statistics are
`((n−1)W + B/n)/W` where `n` is draws **per superchain**. At C=1,024, K=8 that is n = 128×10 = 1,280,
so a real between-superchain mean separation of one posterior sd contributes `B/n ≈ 1/1280` and
nR̂ ≈ 1.0008 — the pooling that makes nested R̂ robust at 10 draws per chain simultaneously divides
out its sensitivity to anything smaller than ~√n sd. On top of that, the branch's own
`test_nested_rhat.py:45-57` proves the estimator is exactly shift-invariant: offsetting every
superchain identically changes nR̂ by <1e-3. A bias carried by all 8 Pathfinder starts (the shared
metric, the shared initialiser) is invisible to it *by construction*, which is precisely the
0.4100 / 1.0040 case, and equally the 0.00063 / 1.0043 case.
**The one experiment that separates "blind" from "wrong axis":** on CPU, build a synthetic
`(1024, 10, d)` stack with the harness's own `ids`, inject a known mean offset of 0.5 and 1.0 sd on
one coordinate for superchains 0–3 only, and assert nR̂ > 1.05; run the same fixture at K=64
(n = 160). If the offset moves nR̂ at K=64 but not at K=8, the estimator is blind at that superchain
size and the gate needs a K ceiling. If it moves neither, the axes or the ids are wrong. Both cases
finish in seconds and neither requires the GPU.

**8. Fast, clean, wrong — REFUTED for the three named traps; two smaller ones remain.**
The stub trap is closed twice: `build_model` snapshots and restores all four stubbed entry points in
a `finally` with an identity assert (`r1_bias.py:78-101`), and `run_cell` re-asserts
`pfcheeshmc.sample_pf_chees_hmc is bl.REAL_SAMPLE_PF` per cell before sampling
(`r1_bias.py:400-409`); `test_bench_ladder_stub_restore.py` pins the loader side. Missing references
raise `FileNotFoundError` naming `tools/make_reference.py` and there is no fallback sampler
(`r1_bias.py:132-137`, tested at `test_r1_bias.py:192-198`). Cold/warm are separated by the `cold`
flag and `rep`, with the docstring stating decisions use warm. Remaining: `run_starts`
(`r1_bias.py:514-543`) has no stub assertion (harmless today — it calls `_run_pathfinder` directly and
never samples — but it would break silently if that method ever routed through the public entry
point), and `nominal_length` (`r1_bias.py:284-313`) reconstructs L by bracketing over
`range(W)` eager calls, which is O(W) host round-trips — 2,000 of them on a warmup cell, a wall-time
cost baked into `wall_s` for every legacy row.

## BLOCKING

1. **Report gradients per chain exactly, or rename the field.** `r1_bias.py:463-467` must write the
   per-chain values it already computes (`steps.sum(0)`, `steps.sum(1)`) — minimum acceptable fix:
   emit the min/max/median triple, or persist the list; and rename
   `grads_per_chain_warmup`/`_sampling` to say they are medians if the medians stay. A8 requires
   "EXACT, the sum of `num_integration_steps`"; a median of sums is not that, and this is the R1
   objective column. Also record in the SUMMARY that rows with `git_sha f82d1ba` have no
   `grads_per_chain_total` key and must be read as warmup+sampling.

## NON-BLOCKING

1. **State the nested-R̂ sensitivity ceiling where a reader will hit it.** One sentence in the
   `score_ensemble` docstring and in SUMMARY §4: nR̂'s between-superchain term scales as 1/(C·D/K), so
   at K=8/C=1024 it cannot see a bias shared by all superchains; cite
   `test_nested_rhat.py:45-57` as the in-repo proof. Add the K=8/K=64 offset fixture from point 7 so
   the claim is pinned by a test rather than by prose.
2. **Floor is optimistic in two named ways** — reference MCSE not propagated, and cross-chain
   independence assumed, which the `superchains` arm violates. Document, or add the reference term.
3. **`steps.reshape(...)[..., 0]`** (`r1_bias.py:457`) should assert the leaf count is 1 instead of
   taking index 0.
4. **`nominal_length`'s O(W) eager loop** inflates every legacy `wall_s`. Move it behind a flag or
   compute the bracket on a subsample and label it.
5. **`run_starts` has no per-cell stub assert**; add it for symmetry with `run_cell`.

## NIT

1. `flatten_run`/`flatten_ref`/`load_reference` implement the same fold three times; one shared
   helper would make rule 1 impossible to break on one side only.
2. `bias()` uses `nanmax`/`nanargmax`, so a NaN leaf can hide a broken variable behind a plausible
   maximum from another leaf; count the NaNs and put them in the row.
3. `g_r1(b2_max or np.inf, ...)` at `r1_bias.py:505` treats a legitimate `b2_max == 0.0` as missing.

## Rebuttal (asked for, condensed)

A b²-with-floor gate is the right instrument at D=10 and split R̂ should stay informational —
`test_nested_rhat.py:21-28` and the §2b chop reproduction both support that. But the plan's G-R1-diag
is not usable as written: nested R̂ is a *precision* statistic between superchains, so at the very
chain counts R1 uses it is structurally incapable of the job Part B gives it. Keep it as a
multimodality alarm with a stated K ceiling, or drop it; do not let any WP4/WP5 accept/kill cite it
as convergence evidence. That is a plan defect, not a branch defect.

## RECOMMEND: hold

Single reason: the R1 objective column (`grads_per_chain_*`) reports a median where A8 requires the
exact sum, and T1 is built entirely on that column. Everything else reviewed here is sound.
