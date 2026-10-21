# WP4 — ensemble mass matrix during a short fixed warmup

Branch `wp/wp4` (from `pf-r1-int`). The run's owner regime is R1 (C≈1,024 × D=10), judged by the
`b² < 0.01` bias gate (floor-printed; a `b²` below the floor is *unresolvable*, i.e. converged
past resolution, not "extra passing"). **Never judge R1 by split R̂** — measured 1.12–8.87 even in
bias-free cells.

## 1. Change

| file | what |
|---|---|
| `pfcheeshmc/ensemble_metric.py` (new) | Four estimators, each returning a **covariance in kernel order**: `sample_diag`, `fisher_diag`, `dense_shrunk`, `diag_lowrank`. Library-first over the BlackJAX fork's `metric_estimators` (`welford_diagonal/dense`, `fisher_score_diagonal`, `draws_singular_value_low_rank`); the only hand-written steps are the diagonal shrinkage and the explicit ridge + positive-definite assertion the fork lacks. `positions_to_draws` flattens the stacked warmup positions per-leaf in sorted (kernel) key order; `reseed_trajectory_length` computes `T0=(π/2)·√λ_max(Σ⁻¹)` without forming the inverse. |
| `pfcheeshmc/sampler.py` | `_run_chees` routes to `_run_chees_two_phase` when `chees_kwargs["metric_refresh"]` names an estimator (default `"none"` keeps the single-phase path bit-for-bit). Phase A = short fixed warmup on the Pathfinder metric; the estimator builds the refreshed covariance from phase-A draws; ε is re-searched on the new metric via the existing traced 32-probe `_run_step_size_tune` (cost independent of chain count); T is re-seeded at `T0` (or **skipped** with `metric_refresh_reseed=False`, phase B then starts at L=1); phase B continues to the budget from phase A's final positions. The two HMCInfo stacks are concatenated so the exact A8 warmup gradient count spans both phases. Swap bookkeeping (estimator, W split, eps0/phaseA-end/re-searched ε, reseed flag, T0) is written to `idata.attrs`. |
| `pfcheeshmc/utils.py` | Additive `WarmupInfo.positions` field (default None) exposes phase-A draws; `.info`/`.trace` consumers unaffected. |
| `tools/r1_bias.py` | `--metric-refresh {none,sample_diag,fisher_diag,dense_shrunk,diag_lowrank}` and `--metric-refresh-reseed` / `--no-metric-refresh-reseed`, forwarded to `chees_kwargs` with `metric_refresh_warmup = max(2, W//2)`; recorded in the row's kwargs. |

Commits: `50fe971` (estimators + two-phase refresh), `34fd462` (r1_bias option), `2430ae8`
(gradient-path fix), `0dea14f` (re-seed flag).

## 2. Tests added

`tests/test_ensemble_metric.py`, 13 tests, all passing: symmetric-PD covariance in kernel order
for each estimator; ridge-not-NaN on degenerate (all-identical) inputs; kernel-sorted-order
flattening on a non-sorted model; `fisher_diag` requires gradients; unknown estimator raises;
re-seed matches the closed form and rejects non-PD; and a **teeth test** that the swap is not a
silent no-op (two phases ran, T re-seeded, ε re-searched to a value ≠ the phase-A endpoint).
Full suite **112 passed** forced CPU (was 99 on `pf-r1-int`).

## 3. Measurements (R1, C=1,024 × D=10, W=50, seeds 1–2, warm)

`~/pf/wp4/results/runs/WP4__r1.jsonl` (32 cells). Baseline = the corrected T1 no-refresh arm
(`int:covraw`: orientation=covariance, iterate=raw), from `~/pf/int/results/runs/T1__r1_sweep.jsonl`.

| model | no-refresh baseline | sample_diag | fisher_diag | dense_shrunk | diag_lowrank |
|---|---|---|---|---|---|
| 8 Schools | **955** g, b² 0.0015 pass | 1,292 g, pass | 1,157 g, pass | 1,344 g, pass | 1,400 g, pass |
| logistic | **335** g, b² 0.0058 pass | 2,386 g, pass | 1,352 g, pass | 2,323 g, pass | 4,934 g, pass |
| GEV | **595** g, b² 0.0001 pass | 4,804 g, pass | 6,673 g, pass | 4,237 g, pass | 3,778 g, pass |
| radon NC | 485 g, b² 0.22 **FAIL** | 5,612 g, pass | 3,886 g, pass | 5,469 g, pass | 5,317 g, pass |

(g = gradients/chain, warmup+sampling. Every refresh cell passes — including radon, where the
baseline fails.)

### The re-seed cost (measured — filled in by the lead from `WP4__reseed_ab.jsonl`, 42 rows)
The refresh's 1.35–20× gradient cost traces to the trajectory length: the re-seed
`T=(π/2)·√λ_max` on a 50-step ensemble metric lands at frozen **L=23–402** versus the baseline's
well-adapted **L=10–27** (λ_max of a short-phase estimate is noise-dominated). The A/B
(`WP4__reseed_ab.jsonl`, C=1,024 D=10 W=50 seeds 1,2,3 — 3 rows per cell) isolates it:

| arm | 8 Schools g | logistic g | GEV g | radon g | radon `g_r1` (3 seeds) |
|---|---|---|---|---|---|
| `ab:baseline` (no refresh) | 967 | 329 | 567 | 275 | **FAIL ×3** (b² 3.000 / 0.223 / 2.852) |
| `ab:sample_diag` (re-seed) | 1,304 = 1.35× | 2,381 = 7.24× | 5,417 = **9.55×** | 5,513 = **20.05×** | **pass ×3** (0.0026 / 0.0021 / 0.0022) |
| `ab:sample_diag_noreseed` | 1,309 = 1.35× | 487 = 1.48× | 841 = **1.48×** | 579 = 2.11× | **not resolvable ×3** (0.0088 / 0.0081 / 0.0136, floors 0.0054–0.0060 > 0.005) |

Reads: the re-seed is **essentially the whole cost** on GEV (9.55× → 1.48×) and logistic (7.24× → 1.48×), and
none of it on 8 Schools (1.35× either way — that is the intrinsic two-phase overhead). But radon is the only
cell in the A/B that changes class at all, and dropping the re-seed moves it from **pass ×3** to **not
resolvable ×3** — the cheap arm never demonstrates a radon pass, it only fails to demonstrate a failure. Every
other model passes in every arm. So the re-seed is expensive and, for radon, load-bearing; `T=(π/2)·√λ_max` is
not a good universal re-seed (it over-shoots L to 98–283 on GEV), and a radon-specific stabiliser is the open
question. No re-seed constant other than on/off was tried.

The CPU-smoke expectation recorded here earlier is confirmed: the no-re-seed arm freezes at **L = 12–35** against
**25–283** re-seeded (98–283 on GEV), i.e. the saving is not a degenerate "L stuck at 1" artifact.

## 4. Frozen ε / L and realised steps
Baseline (no refresh) at W=50 freezes at ε 0.32–0.59, L 10–27 (well-adapted by the raw iterate).
Refresh arms freeze at ε 0.08–0.26, L 23–402 — the small ε and large L are the signature of the
over-seeded trajectory. The no-re-seed arm is expected to freeze near the baseline's L (CPU
smoke: L≈4).

## 5. Gate verdicts
- **Does the refresh improve gradients/chain at equal or better b²?** Not yet on 8 Schools /
  logistic / GEV (it costs more at equal W=50). The re-seed A/B decides whether the cost is
  intrinsic or the re-seed's fault.
- **Does the refresh help where the baseline fails?** **Yes, decisively, on radon NC.** The
  no-refresh baseline fails radon at *every* warmup and seed — **6 of 6 cells, resolvable** (b² 3.000 / 0.223 at
  W=50, 0.428 / 0.033 at W=100, 0.023 / 0.007 at W=200, i.e. 618× → 1.7× their own floors). *Provenance note,
  corrected by the lead:* the W=50 pair is `ab:baseline` in this WP's own file; the W=100/200 numbers are
  `T1__r1_sweep.jsonl` arm `int:covraw` (same configuration — covariance, `raw`, no refresh — different run), and
  they are **fails**, not the "unresolvable" this SUMMARY earlier called them. Quadrupling warmup to W=200 costs
  2.4–6.3× the gradients and still lands above its floor. The refresh passes all seeds at W=50, so the refresh is
  the only reliable R1 fix for radon and **stays, opt-in**.
- Accept/kill per estimator (plan §WP4): the kill criterion is "does not beat plain diagonal by
  1.2× on the correlated models (logistic, radon)." `fisher_diag` beats `sample_diag` on logistic
  (1,352 vs 2,386 g) and radon (3,886 vs 5,612 g) — but the A/B is needed before any estimator is
  accepted as a *cheap* opt-in.

## 6. Surprises and contradictions with the plan
- **The re-seed backfires.** The plan's `T=(π/2)·√λ_max` re-seed, applied to a 50-step ensemble
  estimate, overestimates the trajectory length (noise-dominated λ_max) and produces L 10–15×
  the baseline's — the dominant source of the refresh's gradient cost. This is the finding the
  A/B quantifies.
- **The refresh is a convergence fix for radon, not a gradient win.** Early in the run I
  mis-read this as "just raise the baseline warmup"; the rows refute it — the no-refresh arm fails radon in all
  6 cells across both seeds and all three warmups (§5). Recorded and corrected in the LEDGER.

## 7. Library-first exceptions (A6)
The estimators wrap the fork's `metric_estimators`; the only hand-written numerics are the
diagonal-shrinkage convex combination and the ridge + PD assertion (the fork applies neither).
`_pool_gradients` hand-writes a chunked vmap of the log-density gradient because the fork has no
batched-score helper for a caller-supplied draw pool.

## 8. Exact reproduce commands
```
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; WT=~/pf/wp4
# tests (forced CPU)
pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" CUDA_VISIBLE_DEVICES= \
  python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider
# one R1 cell (estimator arm, re-seed on)
"$WT/tools/gpu_run.sh" pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" \
  python tools/r1_bias.py --models radon_noncentered --chains 1024 --draws 10 --warmups 50 \
  --seeds 1,2,3 --repeat 1 --arm ab:sample_diag --orientation covariance --iterate raw \
  --metric-refresh sample_diag --timed --out "$WT/results/runs/WP4__reseed_ab.jsonl"
# same arm without the T re-seed
... --metric-refresh sample_diag --no-metric-refresh-reseed ...
# no-refresh baseline
... --metric-refresh none ...
```
