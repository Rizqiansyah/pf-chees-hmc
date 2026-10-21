# WP0.2 code-correctness review — branch `wp/wp0refs`

**Reviewed:** `/home/akbar/pf/wp0refs` @ `67da0ca` (parent `82bb771`, base `pf-r1` `c3bdbfc`).
Files read: `tools/make_reference.py`, `tests/test_reference_npz.py`, `results/references/MANIFEST.json`,
`results/runs/WP0__references.jsonl`, plus the consumer `tools/r1_bias.py` in `~/pf/wp3` and
`tools/bench_ladder.py` / `tools/_common.py` for contract cross-checks.

**Sandbox limitation, stated up front:** this review had no shell. Two checks the brief asked for were
therefore not performed by me: `sha256sum` on the four `.npz` files, and reading array shapes out of the
binary `.npz` payloads. Everything below that touches on-disk bytes is labelled accordingly, with the
command the lead must run. Nothing here claims a digest was verified by me.

---

## The seven verdicts

### 1. All three root causes fixed — CONFIRMED

`tools/make_reference.py:139-142`:

```
139: arr["mean__" + n]   = x.mean(axis=(0, 1))
140: arr["var__" + n]    = x.var(axis=(0, 1), ddof=1)
141: arr["mcse_x__" + n]  = _mcse(x, "mean")
142: arr["mcse_x2__" + n] = _mcse(x ** 2, "mean")
```

- Reduction axes are explicit and are exactly the two sample axes. No bare `.mean()` / `.var()` remains in
  the stat path.
- The old collapse came from `_mcse` returning `float(ravel()[0])`. It now returns the library array
  unchanged and raises on a shape mismatch (`:80-83`): `arviz_stats.mcse(x, chain_axis=0, draw_axis=1, ...)`
  is checked against `x.shape[2:]` before the caller can use it.
- Not scalar-luck. For a scalar variable `x.shape[2:] == ()` and `x.mean(axis=(0,1))` is a 0-d array, so the
  guard at `:145-152` passes for both scalar and vector variables by construction, not by accident. GEV is
  all-scalars and would have passed the old buggy code too — which is precisely why the task's own gate
  missed the defect; the new check is the one that distinguishes the two cases.

### 2. Loud shape guard — CONFIRMED, and it is in the writer *and* the consumer

Three independent layers:

| layer | location | catches |
|---|---|---|
| inside the MCSE helper | `make_reference.py:83` | a library/axis change that collapses MCSE |
| writer, before `np.savez` | `make_reference.py:145-152` | any of the four stats whose shape ≠ `sample_shape[2:]` |
| consumer, at load | `~/pf/wp3/tools/r1_bias.py:133-139`, `:181-183` (`require_stats`) | a hand-edited or stale `.npz` that never went through the writer |

A future regression is caught **first at generation** (the writer raises before anything is written, so a
bad file never appears on disk), and second at WP3 load. This is the right arrangement: the writer is the
only place that can fail before the artifact exists. `make_reference.py` itself has no reader, so the
absence of a reader guard in this branch is not a hole.

### 3. Do the tests have teeth — REFUTED as written (partial: condition yes, harness no)

`tests/test_reference_npz.py:68-91` is the teeth demonstration, and it exists, which is more than most
guards get. Two concrete weaknesses:

1. **The teeth test does not exercise the test.** `:90-91` re-implements the comparison inline:
   `with pytest.raises(AssertionError): assert np.asarray(loaded["mean__theta"]).shape == sx.shape[2:]`.
   It never calls the loop in `test_reference_npz_keeps_parameter_axis` (`:47-66`). If `STATS` (`:25`) is
   edited, if a stat prefix is renamed, or if the loop's `s.split("__", 1)[1]` var-extraction breaks, the
   teeth test stays green while the real check stops covering `mcse_x2`. It proves the *condition* fails on
   legacy data, not that the *guard* would have caught it.
2. **Zero files ⇒ green.** `:36-38` returns `[]` when the directory is absent, and `:43`
   `@pytest.fixture(params=[])` yields no test cases, so `test_reference_npz_keeps_parameter_axis`
   contributes nothing to the run and `pixi run test` still reports all-pass. The docstring at `:15-18`
   says the test "skips when the directory is absent" — accurate, but the consequence (a permanently
   vacuous guard in any environment without `$HOME/pf-runs`) is not stated, and `PF_REFERENCE_DIR` set to an
   empty or misspelled directory is indistinguishable from success.

Both are fixed by the same 8-line change in the required-fixes section below.

### 4. Constrained vs unconstrained handling — CONFIRMED at code level; the bytes NOT ESTABLISHED by me

- `u_names = [rv.name for rv in model.value_vars]` (`make_reference.py:92`) is the authoritative
  unconstrained list, taken from the model rather than from the InferenceData group's `data_vars` (the
  comment at `:89-90` about coordinate-shaped entries like `theta_dim_0` is a real trap, correctly avoided).
- `c_names = [n for n in free + det if n not in set(u_names)]` (`:94`) keeps the constrained/problem-space
  names and prevents the same variable being saved twice under both prefixes.
- Sampling uses `bl.REAL_PM_SAMPLE` (`:102`), the pre-stub handle (`bench_ladder.py:50`), not the
  notebook stub, and `bench_ladder` is imported before pymc so the `PYTENSOR_FLAGS` blas trap is set first
  (`:41-43`). Correct on both counts.
- No transform is applied anywhere in the file: no `.backward`, no `transform.` call, so a double transform
  is structurally impossible; the arrays are whatever PyMC reported under the value-var names.
- Failure mode if `include_transformed` ever stops putting value-var names in `posterior`: `post[n]` raises
  `KeyError` at `:116`, i.e. loud, not silent. Good.
- What I cannot certify: that `u__theta` in the file is really the raw draw rather than a back-transformed
  one. Corroborating evidence, indirect but strong: the stored per-parameter means for
  `u__theta` are ≈ `[0.28, 0.04, −0.12, 0.01, −0.22, −0.11, 0.30, 0.05]`, which are the school effects
  divided by τ (τ ≈ 4–5) rather than the effects themselves — the signature of a non-centered raw draw. And
  WP3's independent reproduction of the reviewer's §2e bias numbers used these arrays as ground truth and
  landed within noise on 6 of 7 cells; a mislabelled reference would have broken that agreement.
- One residual: a model with a deterministic whose name collides with a value-var name would have its
  constrained copy silently dropped by `:94`. None of the four models does; worth a comment, not a change.

### 5. `mcse_x2__ = arviz_stats.mcse(x**2, method='mean')` — CONFIRMED sound, and the naming is the risk

It is the right quantity **for what the run does with it**. `r1_bias.bias()` scores
`b²v = ((ens²).mean(0) − (ref²).mean(0))² / (ref²).var(0, ddof=1)` (`r1_bias.py:265-267`) and its floor term
pairs `mcse_mean_sq²` with that same `(ref²).var(0, ddof=1)` denominator (`:268-270`). MCSE-of-the-mean
applied to the transformed series `x²` is exactly the Monte-Carlo standard error of `E[x²]`, and arviz
estimates it from the ESS of the `x²` chain, which is the correct estimator. `ddof=1` matches the stored
`var__` (`make_reference.py:140`), so the two definitions do not disagree.

What is wrong is only the readable name. `mcse_x2__` reads like "MCSE of the variance" to a fast reader, and
the file has no `var_x2__`, so a future consumer that tries to score the x² half of the gate **from the
stats alone** (attractive: radon's `.npz` is 223 MB and recomputation is the reason WP3 loads it) will take
`den = var__` (which is `Var(x)`, not `Var(x²)`) and pair it with `mcse_x2__`, producing a floor that is
wrong by `Var(x²)/Var(x)` — a factor that is not close to 1 for heavy-tailed or far-from-zero variables,
and it would move the "not resolvable" boundary silently. The docstrings at `:16-18` and `:142` do say
"second moment", so this is a documentation-hardening item, not a numerical defect.

### 6. `MANIFEST.json` vs the artifacts — CONFIRMED against the row file; digests NOT ESTABLISHED here

Internal consistency I could check without a shell:

- For all four models the manifest's `sha256`, `path`, `chains`, `draws`, `tune`, `seed`,
  `target_accept`, `max_rhat`, `min_ess`, `divergences`, `divs_per_1k`, `gate`, `wall_s` match the
  corresponding rows in `results/runs/WP0__references.jsonl` written at `git_sha: 82bb771` — including
  `wall_s` to the nanosecond, so the manifest was assembled from these rows, not typed.
- The superseded pre-fix rows (`git_sha: c3bdbfc`, `8schools` sha `74b7a10d…`, `gev` 0.95 sha `9f5534db…`)
  are still in the row file but the manifest points only at the current ones. Correct provenance handling.
- GEV is the canonical 0.99-target run with its 18-divergence caveat written into `note`, not buried.
  Correct.
- Chain/draw are uniform at 8 × 10,000 across models and the sampling call is a single code path
  (`make_reference.py:102-106`), so "chain counts that do not match the manifest" is refuted for the run as
  executed.

**Not performed by me:** recomputing the four digests. The reviewer sandbox has no shell. The lead
already did this and recorded it (`pf-r1` `55ba5b1`, "sha256 matching MANIFEST"), so the residual is
procedural rather than open — but re-run it after the fixes requested below land, because `mean__/var__`
regeneration rewrites the files and every later task reads these hashes:

```
cd ~/pf-runs/references && sha256sum *.npz
```

### 7. Other silent-plausible-reference paths — one real gap, the rest clean

- **Real gap:** nothing asserts that the saved sample arrays actually have shape `(chains, draws, …)`.
  `meta["chains"]`/`meta["draws"]` (`make_reference.py:158-159`) are the CLI arguments echoed back, and they
  flow straight into `MANIFEST.json`, which is the file every later task trusts for C and D. If a future
  `pm.sample` change, a `save_warmup=True`, or a throttled run returned a different posterior shape, the
  file and the manifest would disagree and the disagreement would be invisible: the shape guard at
  `:145-152` only compares stats to samples, and WP3's `require_stats` compares stats to samples too. Nobody
  compares samples to metadata.
- Clean: `tune` is not in the posterior (`save_warmup` is never passed, default `False`); `divs_per_1k` is
  computed over posterior draws only (`:109-111`); model data is deterministic (`build_logistic(...,
  seed=11)` at `bench_ladder.py:188-192`, 8 Schools and radon load fixed on-disk data), so seed 42 for all
  four models does not encode a hidden data difference; `np.savez` does not double-append the `.npz`
  extension; `_rhat_max`/`_ess_min` handle 2-D scalar variables correctly via `reshape(c, d, -1)`.
- Misleading provenance, not wrong statistics: the rows carry `device: ["NVIDIA RTX A2000:0"]` and
  `peak_mem_bytes: 512` for a CPU pytensor run, because `_common.new_row` fills both from JAX
  (`_common.py:161-177`, `peak_mem_bytes()` at `:124-128`) whenever they are `None`. A reader of
  `WP0__references.jsonl` would conclude the references were GPU builds. The file's own docstring says
  CPU-only, so the row contradicts the header.

---

## BLOCKING

1. **Metadata is never checked against the artifact** (`make_reference.py:139-160`). Fix, immediately after
   the stats loop and before `meta` is built:

   ```python
   for n in names:
       got = data[n].shape[:2]
       if got != (chains, draws):
           raise ValueError(f"{n}: posterior shape {got} != requested {(chains, draws)}")
   ```

   Three lines, and it is the same class of defect this branch exists to fix: a number that later tasks read
   from `MANIFEST.json` and cannot falsify.

## NON-BLOCKING

2. **Make the teeth test call the guard, and make zero files loud** (`tests/test_reference_npz.py`).
   Extract the body of `:47-66` into `def stat_shape_problems(path) -> list[str]`; have the real-file test
   assert it returns `[]`, and have the synthetic test assert it returns exactly four problems naming
   `mean__theta`, `var__theta`, `mcse_x__theta`, `mcse_x2__theta`. Add: if `PF_REFERENCE_DIR` is set in the
   environment and yields no `.npz`, fail instead of silently collecting nothing.
3. **Document that the stat columns are not authoritative** (`make_reference.py:14-18`,
   `MANIFEST.json["stat_shape_contract"]`). `r1_bias.bias()` recomputes means, `Var(x)`, `Var(x²)` and both
   MCSEs from the sample arrays (`r1_bias.py:264-270`); the stored stats are shape-checked and otherwise
   unused. Say so, and say that `mcse_x2__` is `MCSE(E[x²])` — not the MCSE of a variance — and that no
   `var_x2__` exists, so a stats-only consumer would need to add it.
4. **Correct the CPU provenance fields in the reference rows** (`make_reference.py:190-199`): pass
   `device="cpu"` and `peak_mem_bytes=None` explicitly to `_common.new_row` so a CPU-built reference does not
   read as a GPU build.

## NIT

5. `c_names` (`:94`) silently drops a constrained name that collides with a value-var name. No model here
   does; a one-line comment would prevent a future surprise.
6. `_rhat_max`/`_ess_min` are per-element Python loops (175 calls for radon). Fine at these sizes; if the
   references are ever regenerated at 64 chains it becomes the slow part, and the run has already been bitten
   by per-parameter eager calls elsewhere.

## Rebuttal, as invited

The contract is wider than it needs to be and that is the thing most likely to cost the run later. Four
stat arrays per variable are written, shape-guarded, tested, and then **not used** — the gate's actual
arithmetic is `ref.mean(0)`, `ref.var(0, ddof=1)`, `(ref**2).mean(0)`, `(ref**2).var(0, ddof=1)` and two
MCSEs, all recomputed by the consumer from `u__<var>`. So the run carries two definitions of the same
quantities, one of which can drift out of agreement with the other and neither of which is checked against
the sample arrays it is supposed to summarise. The cheap fix is not more guards on the stats; it is to pick
one authority. My recommendation: keep `mean__/var__/mcse_x__/mcse_x2__` as a *display* artifact only, label
them as such in the manifest, and if any future task wants to score bias from stats alone it must add
`mean_x2__` and `var_x2__` in the same commit and change the consumer, so the pair can never be half-migrated.

Second, plainer point: `tests/test_reference_npz.py` currently proves the guard's arithmetic, not the
guard's reach. In a repo whose last three incidents were all "the check existed and did not run", the
reach is the part worth pinning.

---

**RECOMMEND: hold** — the statistical core is correct and all three root causes are genuinely fixed, but
`MANIFEST.json` publishes `chains`/`draws` that nothing ever verifies against the saved arrays, which is the
same unfalsifiable-metadata failure this branch was created to eliminate; the fix is the 3 lines above and
can land in about ten minutes.
