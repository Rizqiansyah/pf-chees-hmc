# WP3 — R1 evaluation harness (bias gate + nested R-hat)

Branch `wp/wp3` from `pf-r1` (`c3bdbfc`), merged to `448fcc8`. Device: RTX A2000 6 GB unless a row says
`cpu`. All timed cells ran under `flock /tmp/pf-gpu.lock` (`PF_GPU_LOCK=1`, asserted in-process).

## 1. Change

| File | What | SHA |
|---|---|---|
| `tools/r1_bias.py` | the harness: drives public `sample_pf_chees_hmc`, scores b² against the WP0.2 reference, prints the noise floor, nested R-hat, exact gradient counts, frozen ε/L, stage split, peak memory | `2a68fb3`, `f82d1ba`, `448fcc8` |
| `tools/wp3_chop_check.py` | re-derives plan §2b and adds nested R-hat to the same arrays | `7d01a77` |
| `tools/wp3_summarize.py` | gate aggregation: smallest W with b²_max < 0.01 on **every** seed, warm rows only | `7d01a77` |
| `tools/wp3_selftest.sh` | the §2e matrix, one process per (model, W, seed) | `f82d1ba` |
| `pfcheeshmc/sampler.py` | `pathfinder_kwargs={"superchains": K}` — K stratified Pathfinder draws, each replicated C/K times. Default off; byte-identical when off (tested) | `2a68fb3` |
| `tests/test_r1_bias.py`, `tests/test_nested_rhat.py` | 25 tests, CPU, 18.4 s | all three |

Two process-local wraps of the library, per A8's legacy-arm contract, and no library API change:
`_to_inference_data` forced to `keep_untransformed=True` (b² is defined in unconstrained value-var
space) and `_run_chees` wrapped to capture frozen ε, the frozen trajectory length and the exact
warmup gradient counts. Part B asked for "a public option in new code" for `keep_untransformed`;
see §7.

## 2. Tests added

25 new. Each named failure mode was reproduced before it was fixed.

| Test | Failed on the parent commit? | Passes now |
|---|---|---|
| `test_nested_rhat.py` (4): stationary chains ≈ 1 at 10 draws/chain; dispersed superchains > 1.5; global offset invisible; unequal superchains rejected | new file | yes |
| `test_argmax_names_a_parameter_and_which_moment_failed` | **yes** — `IndexError` | yes |
| `test_run_and_reference_are_flattened_by_the_same_rule` | **yes** — run side produced 10 fake leaves for 8 Schools | yes |
| `test_scoring_a_radon_shaped_posterior_does_not_merge_the_dims` | **yes** — `ValueError`, 4 rows vs 100 | yes |
| `test_nominal_length_recovers_the_frozen_trajectory_length` | **yes** — returned `None` | yes |
| `test_collapsed_reference_statistics_are_rejected_at_load`, `test_reference_missing_a_value_var_is_an_error_not_a_silent_subset`, `test_missing_reference_is_a_named_gate_failure` | new | yes |
| `test_superchains_replicates_K_distinct_starts_and_validates_K`, `test_superchains_off_leaves_the_pick_unchanged` | new | yes |
| `bias` / `g_r1` / flattening / reference-reader tests | new | yes |

Full suite on the merged branch: **63 passed** (CPU, 87.9 s), including the lead's
`test_bench_ladder_stub_restore.py`.

## 3. Measurements — §2e self-test (GPU, C=1024, D=10, warm rows, `arm=legacy` = `~/pf/legacy` @ `ecd9ba1`)

Rows: `results/runs/WP3__selftest.jsonl` (33 rows: 30 grid + 3 `pf_starts`), cold and warm both
present; `rep=0` cold, `rep=1` warm, decisions from warm only (A5).

| model | W | b²_max (this WP, 3 seeds) | b² floor | §2e | verdict |
|---|---|---|---|---|---|
| 8 Schools | starts only | **2.711** | n/a | 2.71 | match |
| 8 Schools | 100 | **0.00063 / 0.00277 / 0.00363** | 0.0020–0.0023 | 0.0008 | **reproduces at gate-decision level; the W=100 8-Schools cell is 1.4× its noise floor and 3.5× the reviewer's number, so it is not resolvable enough to disagree** |
| 8 Schools | 2,000 | 0.00037 / 0.00069 / 0.00152 | 0.0008–0.001 | 0.0008 | match |
| logistic | starts only | **0.301** | n/a | 0.305 | match |
| logistic | 100 | 0.071 / 0.141 / 0.155 | 0.0026–0.0028 | 0.138 | match |
| logistic | 200 | 0.011 / 0.045 / 0.057 | 0.0026–0.0031 | 0.039 | match |
| logistic | 400 | 0.0011 / 0.0022 / 0.0034 | 0.0026–0.0027 | 0.0027 | match |

Supporting columns (warm, seed 1): 8 Schools W=100 ε 0.1344 / L 20.4 / grads 1,040+111 / 0 divs per
1k / cold 52.2 s → warm 23.1 s; W=2,000 ε 0.1403 / L 56.4 / grads 52,108+281; logistic W=100 ε 0.00142 /
L 8.1 / grads 408+46; W=400 ε 0.00189 / L 26.8 / grads 2,864+135. §2e quotes ε 0.14 and 0.0015,
L 22 and 8→28, grads 1,200 and 458→3,027. Every ε, L and gradient count agrees; L is bracketed to
about half an integration step for the legacy arm (§4).

Gate outputs (`tools/wp3_summarize.py`): 8 Schools smallest W passing G-R1 on every seed = **100** at
994–1,151 gradients per chain; logistic = **400** at 2,999–4,413. Peak device memory at C=1024:
168–919 MB, i.e. the §0.6 OOMs were pool size, not the sampler.

`pf_starts` rows are the pre-sampling bias of the start ensemble: they carry no ε, L, or gradient
count because no sampler ran (b² of 1,024 Pathfinder draws only).

## 4. Frozen ε/L and realised steps

`frozen_eps`, `frozen_L_nominal`, `grads_per_chain_warmup`, `grads_per_chain_sampling` and
`steps_mean` are populated in **all 30 grid rows**; the only rows where they are null are the three
`pf_starts` rows (nothing sampled). `grads_per_chain_total` (warmup + sampling) was added at
`448fcc8` because A8 lists the two halves but no sum, and T1's objective is gradients per chain.

Legacy ε comes from `parameters["step_size"]`. Legacy L is reconstructed, and its precision is stated:
the warmup emits `ceil(h_i · L)` with `h_i` the Halton jitter, so each realised count gives one bound
`L ∈ [(L_i−1)/h_i, L_i/h_i]`, the intersection is ~1/h_max ≈ 1 wide, and the reported midpoint is
pinned to about half an integration step (tested at W=300 and W=1,000). An empty intersection returns
`None` rather than a guess. Arms that publish `frozen_L_nominal` in `idata.attrs` (A8, WP2 onward)
are read directly.

## 5. Gate verdicts

- **G-R1** (b²_max < 0.01 on every seed, C·D ≥ 2,000, floor printed): 8 Schools passes at W=100 and
  W=2,000; logistic fails at 100 and 200, passes at 400. Emission, one cell verbatim:
  `b2_max=0.00363  floor=0.00228  argmax=theta[7] (x^2)  G-R1=pass` (8 Schools, W=100, seed 3, warm).
  A cell whose floor exceeds 0.005 is reported as `not resolvable (floor ...)`, and C·D < 2,000 as
  `not eligible`; neither can become a pass. Worked counter-example from the harness probe:
  C=8, D=10 gave `b2_max=1.048 floor=0.329 G-R1=not eligible (C*D < 2000)`.
- **G-R1-diag** (nested R-hat): implemented and emitted (`nested_rhat_max`, `superchains`), with
  K=8 and limit 1.01. **K = 8 is taken from the primary source**, arXiv 2110.13017v3: *"we recommend
  using K between 8 and 256 initializations, when running KM=2048 parallel chains"* and *"K = 4, 8,
  or 16 … when running KM=128 chains"*; C=1,024 sits between those two statements, and K=8 (M=128
  subchains at C=1,024, M=64 at C=512) is the value both quotes support. The threshold is the
  paper's own adoption of Vehtari et al.'s 1.01. The same paper states no superchain size uniformly
  minimises the variance of nR̂, so K=8 is a documented choice, not an optimum. Property tests pass
  (4 in `test_nested_rhat.py`); a live K=8 grid is the in-flight part of this WP (§8 command).
- **G-ORDER**: not applicable to WP3; the `superchains` option reuses the existing stratified pick and
  inherits its ordering.
- **G-MEM**: not a WP3 gate; recorded anyway — max 919 MB of the 5.0 GB bar at C=1,024, d=25.
- Gates G-A..G3 were not exercised: WP3 builds the instrument and validates it against §2e, it does
  not change a sampler default.

## 6. Surprises and contradictions with Part B

1. **Split R-hat in the real R1 configuration, not just a chop.** The self-test rows read
   split R̂ **2.06–2.36** on 8 Schools at W=100 and **4.6–5.6** on logistic, while b² says 8 Schools is
   inside the gate (0.0006–0.0036) and nested R̂ reads 1.0004. `tools/wp3_chop_check.py` shows both
   diagnostics on the same converged arrays: split R̂ 1.0464 / 1.0231 / 1.0893 / 1.0067 for
   1024×39 / 512×78 / 512×20 / 128×313 — §2b's reported 1.050 / 1.024 / 1.090 / 1.006 — against
   nested R̂ **1.0004 / 1.0004 / 1.0014 / 1.0003**. Part B §0.4 is confirmed, and the effect is larger
   with `superchains` on (clustered starts are the point of the design, and split R̂ punishes it).
2. **Zero divergences next to a 15 % error.** Every cell in the matrix reports `divs_per_1k = 0.0`,
   including logistic at W=100 where b²_max = 0.141 and split R̂ = 5.4. Divergences, and any
   "no divergences so the run is fine" reading, are useless in R1. This strengthens the owner's
   decision to keep divergences only as a diagnostic.
3. **b² is floor-dominated at short warmup, and seeds move it more than arms do.** At 8 Schools
   W=100 the three seeds give 0.00063 / 0.00277 / 0.00363 against a floor of 0.0020–0.0023: a 5.7×
   spread inside one noise unit. Any arm comparison at one seed and one W will disagree by more than
   the arms differ. Decisions need ≥ 3 seeds (5 for a default change) and the floor printed beside
   every number — which is why the harness prints it in the row, not in a log line.
4. **Harness constraint for the R1 sweep (WP1's radon dim trap).** `idata.posterior.to_array()` on
   radon materialises a 137 TiB cross-product: xarray merges the repeated 85-length dim and the
   scalar dims into one index. Nothing in `r1_bias.py` calls `to_array` — the scorer extracts one
   variable at a time (`idata.posterior[n].values`) — and
   `test_scoring_a_radon_shaped_posterior_does_not_merge_the_dims` pins it on radon-shaped data
   (two scalars + one (85,) vector). Writing the test found a second, quieter bug in my own scorer:
   a scalar variable arriving as 2-D `(chain, draw)` was folded as `(rows, leaves)`, so it scored 4
   draws while the vector variable scored 100 — it raises now, and on radon it would have been a
   silent 4-vs-100 mismatch if the sizes had happened to line up.
5. **`bench_ladder.load_notebook_model` leaked its sampling stubs process-wide.** Loading 8 Schools
   replaced `pfcheeshmc.sample_pf_chees_hmc` and `pm.sample` with a 1-draw fake for the rest of the
   process; a cell that hits it returns instantly with zero divergences and looks like a good result.
   It broke `tests/test_rng_stage_keys.py` by test ordering alone. Fixed permanently on `pf-r1`
   (`9890db7`, `test_bench_ladder_stub_restore.py`), merged into this branch, and the harness also
   asserts `pfcheeshmc.sample_pf_chees_hmc is bench_ladder.REAL_SAMPLE_PF` before every cell.
6. **`_common.peak_mem_bytes()` at row-creation time measured an empty process** (512 bytes on every
   row). The shared helper now stamps after the cell (`8f6578c`); my rows use that. Device peak is
   process-cumulative — JAX 0.11.2 has no reset on this card — which is why the matrix runs one
   process per cell.
7. **Part B §0.6's OOM framing is pool size, not C.** At C=1,024 with `--pf-draws 64` the whole run
   peaks at 610–919 MB on logistic. Nothing about C=1,024 itself was unaffordable on 6 GB.
8. No contradiction with §2e beyond the W=100 wording in §3.

## 7. Library-first exceptions

- **No numerical routine is reimplemented.** b² is arithmetic; R̂, nested R̂ and MCSE come from
  `arviz` 0.23.4 / `arviz_stats` 1.3.3 (`rhat_nested(data, superchain_ids=…, chain_axis=0,
  draw_axis=1)`), and split R̂ is called exactly the way `tools/bench_ladder.py` calls it so new rows
  stay comparable with old ones.
- `frozen_L_nominal` for the legacy arm is derived from the frozen kernel's own `ceil(h_i·L)` rather
  than read from private adaptation state (A8's contract only reaches `_run_chees` and
  `_to_inference_data`). Precision stated in §4; WP2's `idata.attrs` path supersedes it.
- **Row fields beyond A8 (2):** `warmup_steps_used` and `grads_per_chain_total`. A8 lists
  `warmup_steps_used` as an `idata.attrs` field, not a row field, and lists no sum for gradients per
  chain. Reporting tasks should read `grads_per_chain_total` for T1; the two halves are still there.
- **`keep_untransformed` was not added to the public signature.** Part B asks for "a public option in
  new code"; A7 says a public API change beyond Part B stops the WP, and my brief restricted my
  `sampler.py` edit to `superchains`. I forced it with a process-local wrapper instead, so WP3 is not
  blocked and `sampler.py` stays single-owner during the parallel waves. If the owner wants it public
  it is a 3-line additive change (`sample(..., keep_untransformed=False)` →
  `_to_inference_data`), and it belongs to whichever WP next touches that call.
- Field-name note for the reporting task: the A8 names are `frozen_L_nominal`,
  `grads_per_chain_warmup`, `grads_per_chain_sampling` — there is no `frozen_L` or `grad_per_chain`
  column, and `null` in those fields appears only in `arm=pf_starts` rows.

## 8. Exact commands to reproduce

```bash
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; WT=~/pf/wp3

# tests (CPU; activation would otherwise put pytest on the A2000, so the post-activation override)
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$WT" \
  python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider          # 63 passed

# the reference contract check, no sampling
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$WT" \
  python "$WT/tools/r1_bias.py" --models 8schools,gev,logit_1000x25,radon_noncentered \
  --check-references-only                                                       # 4x PASS

# §2b chop + nested R-hat (CPU)
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$WT" \
  python "$WT/tools/wp3_chop_check.py"

# the §2e matrix (GPU, one process per cell, all under the lock) -> 33 rows
sh "$WT/tools/wp3_selftest.sh"

# one cell, e.g. a re-run of a contended or failed cell
flock /tmp/pf-gpu.lock env PF_GPU_LOCK=1 pixi run --manifest-path "$MAIN/pixi.toml" \
  env PYTHONPATH="$WT" python "$WT/tools/r1_bias.py" --models 8schools --chains 1024 \
  --draws 10 --warmups 100 --seeds 1 --repeat 2 --arm legacy --legacy "$HOME/pf/legacy" \
  --pf-draws 64 --init-pick iid --timed --out "$WT/results/runs/WP3__selftest.jsonl"

# gate table
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$WT" \
  python "$WT/tools/wp3_summarize.py" "$WT/results/runs/WP3__selftest.jsonl"

# the remaining piece: nested R-hat live grid, K=8, queued behind other tasks on the lock
K=8 SEEDS=1,2 WS_8S=25,100 WS_LOGIT=25,400 ARM=legacy_k8 STARTS_MODELS= \
  OUT="$WT/results/runs/WP3__nested_k8.jsonl" sh "$WT/tools/wp3_selftest.sh"
```

Not covered by this WP: GEV and radon NC in the R1 grid (the references exist and both pass the load
guard; the §2e self-test scope was 8 Schools and logistic), the C ∈ {512, 128} and D=1 axes of the
sweep definition, and the 5-seed decision runs. Same commands, different `--models/--chains/--draws/
--warmups/--seeds`.
