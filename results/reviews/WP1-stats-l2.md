# wp/wp1 statistical-validity review (independent, L2)

Reviewer: independent (not the branch author, not the lead). Tool layer OK (`echo tool-layer-ok` ran clean).
No GPU used. Every number below was computed in this session from the row files named; `python3` is unavailable so all computation went through `pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= python`.

**Row files worked from (all read directly):**
- `/home/akbar/pf/wp1/results/runs/WP1__rebaseline.jsonl` (142 rows; 2 VOID-stamped in `error`, excluded → 140 live)
- `/home/akbar/pf/wp1/results/runs/WP1__covarm.jsonl` (40 rows, all `wp1:mc:covariance`)
- `/home/akbar/pf/wp1/results/runs/WP1__gev_2x2.jsonl` (24 rows)
- `/home/akbar/pf/int/results/runs/T1__r1_sweep.jsonl` (48 rows, all cold, rep 0)
- `/home/akbar/pf/int/results/runs/G3__gev_meanshift.jsonl` (10 rows)
- Gate code: `/home/akbar/pf/wp3/tools/r1_bias.py` → `g_r1` and `bias`.

**VOID marking:** the 2 VOID rows carry `error = "VOID: pre-patch duplicate of cell (...)". Exclusion rule used: drop any row whose `error` string contains `VOID`. R2 warm = `cold is False`.

---

## 1. R2 gradients/ESS and R̂, per seed (32 chains × 1,250 draws, W=2,000, seeds 43–47, warm)

`d` per model: 8schools=10, logit=25, **gev=3**, radon=175.

grads_per_ess_worst, min / median / max across the 5 seeds:

| model | legacy:mc (unperm+prec) | wp1:mc (perm+prec) | wp1:mc:covariance (perm+cov) |
|---|---|---|---|
| 8schools | 53.7 / 58.2 / 59.7 | 54.3 / 57.2 / 61.8 | **33.8 / 35.5 / 37.2** |
| logit | 139.8 / 151.7 / 157.8 | 150.5 / 163.8 / 173.5 | **44.8 / 46.1 / 48.7** |
| gev | 51.4 / 73.4 / 118.6 | 240.3 / 295.8 / 516.5 | **34.9 / 71.2 / 126.1** |
| radon | 434.1 / 502.0 / 567.3 | 2158.5 / 2278.8 / 2549.6 | **174.4 / 182.3 / 199.2** |

**"Improves gradients/ESS on 4 of 4" — REFUTED as stated.** Non-overlapping (cov max < legacy min) wins on **3 of 4**: 8schools, logit, radon. **GEV is the weakest case and is not an improvement**: cov max (126.1) overlaps legacy min (51.4); per seed, cov beats legacy on only **2 of 5** (seeds 43, 44) and is **slower on 3 of 5** (45: 71.2 vs 52.7; 46: 126.1 vs 73.4; 47: 103.4 vs 51.4). The median "win" (71.2 vs 73.4, 3%) is an outlier artifact, not a robust gain. So the honest statement is **3/4 clean efficiency wins; GEV is a wash-to-regression on efficiency.**

The logit improvement (151.7 → 46.1 median; per-seed legacy 139.8–157.8 vs cov 44.8–48.7) is a **3.3× non-overlapping separation on all 5 seeds — not seed noise.** Confirmed real.

max_rhat per seed, pass count at R̂ ≤ 1.01:

| model | legacy (unperm+prec) | wp1:mc (perm+prec) | cov (perm+cov) |
|---|---|---|---|
| 8schools | 5/5 (1.0014–1.0018) | 5/5 (1.0016–1.0020) | 5/5 (1.0013–1.0017) |
| logit | 5/5 (1.0010–1.0018) | 5/5 (1.0012–1.0025) | 5/5 (1.0013–1.0016) |
| gev | 5/5 (1.0015–1.0075) | **0/5 (1.0294–1.0688)** | **3/5 (1.0017,1.0050,1.0074 pass; 1.0161,1.0157 fail)** |
| radon | 5/5 (1.0053–1.0078) | **0/5 (1.0109–1.0165)** | 5/5 (1.0060–1.0092) |

## 2. GEV R̂ and divergence rise (G3__gev_meanshift.jsonl, 32×1250, W=2000, 5 seeds)

| arm | max_rhat (per seed) | R̂≤1.01 | worst mean shift (sd) | divs/1k (per seed) | frozen ε |
|---|---|---|---|---|---|
| precision | 1.0465, 1.0471, 1.0294, 1.0688, 1.0380 | **0/5** | 0.033–0.059 (median 0.0366, max 0.0590) | 0.100, 0.025, 0.000, 0.025, 0.000 (median 0.025) | 0.0009–0.0010 |
| covariance | 1.0065, 1.0017, 1.0054, 1.0141, 1.0117 | **3/5** | 0.015–0.028 (median 0.0205, max 0.0284) | 0.875, 0.350, 0.400, 0.200, 0.575 (median 0.400) | 0.749–0.866 |

- **Mean error "halves" — CONFIRMED.** median 0.0366 → 0.0205 sd; max 0.0590 → 0.0284 sd (≈0.48×).
- **R̂ 3/5 vs 0/5 — CONFIRMED** (covariance passes seeds 43/44/45; precision passes none).
- **Divergence rise "seed-stable" — CONFIRMED and non-overlapping.** Every covariance seed (min 0.200) exceeds every precision seed (max 0.100). Median 0.025 → 0.400 per 1k.
- **Materiality:** 40,000 draws = 32×1250. In absolute count that is **precision 0–4 divergences vs covariance 8–35 per 40,000 draws** (≤0.0875%). Real, non-overlapping in rate, but small in absolute count.
- **R̂ does NOT track divergence count across seeds in either arm.** On covariance the most-divergent seed (43, 0.875/1k) has the *best* R̂ (1.0065, pass) and the least-divergent seed (46, 0.200/1k) has a *failing* R̂ (1.0141). So the two GEV R̂ failures on the covariance arm are **not** caused by the divergence rise — they are a separate mixing phenomenon.
- **Interpretation note:** frozen ε jumps ~800× (0.001 → 0.8) with the flip. The divergence rise is very likely the mechanical cost of a correctly-scaled (larger) step, not evidence against the orientation.

## 3. "No neutral interim state" — REFUTED by the GEV 2×2

The argument is: merging the permutation without the flip ships permuted+precision (`wp1:mc`), which fails GEV R̂ 5/5, so there is no neutral middle.

- The **half** of the claim about `wp1:mc` is **CONFIRMED**: permuted+precision fails R̂ ≤ 1.01 on GEV **5/5** (1.0294, 1.0465, 1.0471, 1.0380, 1.0688 — no seed ≤ 1.01) and on radon 5/5 (1.0109–1.0165). So the permutation cannot ship with the old orientation.
- But the **conclusion** ("no neutral interim state") is **REFUTED**. The GEV 2×2 (`WP1__gev_2x2.jsonl`, 3 seeds, order × orientation, warm) shows a *better* interim exists — **flip orientation, do NOT permute**:

| order \ orientation | precision | covariance |
|---|---|---|
| pathfinder (unpermuted) | R̂ 2/3 (1.0048–1.0142), g/e 264–427, ε 0.0009 | **R̂ 3/3 (1.0011–1.0017), g/e 21.3–26.8, ε 0.86–0.89** |
| kernel (permuted) | R̂ 0/3 (1.0313–1.0518), g/e 465–600 | R̂ 3/3 (1.0017–1.0055), g/e 56.4–103.6, ε 0.77–0.83 |

`pathfinder:covariance` (flip, no permute) is the **fastest** GEV arm (21–27 g/e) and **converges** (R̂ 3/3). It is a fully viable — and on GEV superior — interim state. So "no neutral interim state" is too strong; it should be "no *converging, correctly-ordered* interim state," which is a different (and weaker) claim.

## 4. The permutation is a measured GEV efficiency regression with an unproven correctness benefit

Holding orientation = covariance in the 2×2 isolates the permutation:
- pathfinder (unpermuted): **21.3 / 21.5 / 26.8** g/e, R̂ 3/3
- kernel (permuted): **96.5 / 56.4 / 103.6** g/e, R̂ 3/3

**The permutation is 2–4× slower on GEV with no overlap (max 26.8 < min 56.4) and no R̂ benefit (both 3/3).** It is throwing away a large fraction of the flip's own gain (which alone, precision→covariance, cuts GEV g/e ~13× at the unpermuted order, 278.7→21.5 median, and ~6× at the permuted order, 567.2→96.5 median). Its correctness benefit is **not demonstrated** by any row in the set: both orderings converge on GEV, and there is no posterior-accuracy (mean/variance vs reference) comparison between the two orderings anywhere in the data I read. The claim that "the metric is applied to the wrong coordinates" is a code-level assertion; the statistics available here do not establish that the corrected ordering produces a more accurate posterior, and on GEV they show it produces a *slower* one.

**Caveat:** the 2×2 uses a GEV with `d=7` while the R2 rebaseline GEV has `d=3` — these are not the same GEV configuration, so cross-file g/e comparison is invalid. The within-2×2 order comparison (both d=7) is valid and is what I cite.

## 5. R1 (T1__r1_sweep.jsonl): "3 of 4 vs 2 of 4" — CONFIRMED, but confounded and all-cold

Applying `g_r1` (C·D=10240 ≥ 2000; unresolvable if floor > 0.005; else pass iff b² < 0.01) **seed by seed**, "model passes" = both seeds pass at some W:

| model | int:shipped (prec+ma) | int:covraw (cov+raw) |
|---|---|---|
| 8schools | pass W=100,200 (fails 50) | **pass W=50,100,200** |
| logit | **fail all W** (b² 0.039–0.193) | **pass W=50,100,200** (b² 0.0010–0.0058) |
| gev | pass W=100,200 (fails 50, seed2 b² 0.087) | **pass W=50,100,200** |
| radon | fail 50; **unresolvable 100,200** (b² 12–106) | fail 50; unresolvable 100; **200 = fail seed1 (0.0234) / pass seed2 (0.0071)** |

**Shipped passes 2/4 (8schools, gev); corrected passes 3/4 (8schools, logit, gev). CONFIRMED.** Weakest case: radon — the corrected arm does **not** pass on both seeds at any W (200 is 1 fail / 1 pass; 100 is unresolvable).

**Attribution (the key caveat):** the two arms differ in **two** factors — orientation (precision→covariance) **and** adapter iterate (ma→raw). So the R1 "3/4 vs 2/4" is **NOT attributable to the metric alone.** Also all 48 rows are `cold`, rep 0 — no warm wall, so no R1 wall-clock claim is supportable.

**Single arm that would settle attribution:** add **(orientation=covariance, iterate=ma)** on **logit_1000x25, W=100, seeds 1,2** (the model with the largest R1 gap). If that passes, the orientation alone flips the biggest R1 gap and the iterate is not load-bearing; if it does not, the R1 gain is the raw iterate's, not the metric's. (Equivalent test on 8schools, but logit is where the gap — and thus the signal — is largest.)

**Floor robustness:** if the floor were 2× larger, 13 cells reclassify (mostly → unresolvable), but the model-level 3/4 vs 2/4 **does not change** (unresolvable is not a pass, and each passing model still passes at some W on both seeds). The R1 headline is robust to a 2× floor.

## 6. Noise floor: roughly calibrated, with one design defect

`bias()`: `floor = max(mcse_mean²/den, mcse_mean_sq²/den2) · 2·log(2d)`.
- `mcse` is arviz_stats (ESS-based, autocorrelation-aware) — so the per-quantity noise is conservative, not naive. Good.
- `2·log(2d)` is a Bonferroni over d parameters. For radon (d=175) this is ≈11.7; for 8schools (d=10) ≈6.0. It is slightly above the i.i.d. expectation (≈2·log d) but below what highly-autocorrelated runs would actually produce, so it is **roughly calibrated, possibly mildly optimistic** for very-correlated chains.
- **Design defect (NON-BLOCKING):** `g_r1` returns "unresolvable" whenever floor > 0.005 **without looking at b²**. So shipped radon at b² = 105 / 63 / 12 — four orders of magnitude above its floor — is labeled "unresolvable" (neutral) rather than "fail." The unresolvable verdict should require b² to be within a few floors; b² ≫ floor must stay a fail. As written, the gate can mask a catastrophic bias on high-d models.

## Verdict on the decision

The **orientation flip (precision → covariance) is statistically justified** and the R2 evidence for it is clean (iterate held at ma; only order+orientation move). It is the dominant, large effect: fixes GEV R̂ (0/5 → 3/5), halves GEV posterior-mean error, and is a non-overlapping efficiency win on 3/4 models. The accepted cost (GEV divergences 0–4 → 8–35 per 40k draws) is real but small in absolute terms, is a mechanical consequence of the ~800× larger correctly-scaled step, and is **not** the cause of the two remaining GEV R̂ failures (R̂ does not track divergence count).

What is **not** justified as shipped: (a) the "improves on 4 of 4" efficiency claim — it is 3/4, and GEV is a wash/regression; (b) the **unconditional permutation** — a measured 2–4× GEV efficiency regression with no demonstrated correctness benefit; and (c) the "no neutral interim state" rationale — refuted by `pathfinder:covariance`, a viable, GEV-superior interim.

**BLOCKING**
1. The branch ships the permutation **unconditionally as a correctness fix**, but the data show it is a 2–4× GEV efficiency regression (2×2: 21–27 vs 56–104 g/e, both R̂ 3/3) and contain no posterior-accuracy evidence that the corrected ordering is more accurate. Either demonstrate the correctness benefit (see "single measurement") or make the permutation conditional/justified; do not ship it as an unconditional free correctness fix.

**NON-BLOCKING**
1. "Improves gradients/ESS on 4 of 4 real models" should read "3 of 4 (8schools, logit, radon, non-overlapping); GEV is a wash (cov better on 2/5 seeds) / regression in the 2×2."
2. "No neutral interim state" should be softened: `pathfinder:covariance` (flip, no permute) is a converging, GEV-superior interim; the true claim is "no correctly-ordered converging interim with the old orientation."
3. `g_r1` masks catastrophic bias when floor > 0.005 (b²=105 → "unresolvable"). Require b² within a few floors for the unresolvable label.
4. R1 "3/4 vs 2/4" is confounded with the adapter iterate and all-cold; label it as such and add the (covariance, ma) logit arm for attribution.
5. GEV configuration is inconsistent across files (2×2 d=7 vs R2 d=3) — state which GEV each table refers to.

**NIT**
1. G3 `worst_shift_reference_sigmas` reaches 3.34 (covariance seed 44) — the "mean within 1σ" framing is not met by either arm; the flip's mean-error case rests on the sd-scale shift (0.020–0.028), not on the reference-sigma metric.

**RECOMMEND: hold** — merge the orientation flip (justified), but gate the unconditional permutation on a correctness (posterior-accuracy) demonstration; as shipped it bundles a measured GEV efficiency regression behind an unproven correctness claim and a refuted "no neutral interim" rationale.

**Was the flip decision statistically justified?** Yes for the orientation flip (precision→covariance); no for the bundled unconditional permutation. The single measurement that would most change my answer: **a posterior-accuracy comparison on radon (d=175), permuted vs unpermuted, both with the covariance orientation, scoring worst mean and variance error against the CPU NUTS reference** — if the permuted arm is more accurate there, the permutation is a justified correctness fix (accept the GEV slowdown); if it is not more accurate, the permutation is a pure regression and should be dropped or made conditional.

---

## Re-review after the order A/B

New data: `~/pf/int/results/runs/ORDER__radon_noncentered.jsonl` (10 rows; 5 seeds × 2 arms; orientation pinned to `covariance`, only `imm_order` differs: `kernel` = permuted to sorted-key order, `pathfinder` = left in value_vars order; radon noncentered, d=175, 32×1250, W=2000, scored per parameter vs the CPU NUTS reference in unconstrained space). All numbers below recomputed by me; they reproduce the lead's headlines.

| arm (order) | worst mean error (sd) median (range) | max R̂ median (range) | min-variable ESS median | grads per min-ESS median (range) |
|---|---|---|---|---|
| pathfinder (unpermuted) | 0.01659 (0.0124–0.0211) | 1.0076 (1.0073–1.0124) | 3,286 | **18.42** (17.14–21.38) |
| kernel (permuted) | 0.02143 (0.0170–0.0302) | 1.0063 (1.0051–1.0084) | 4,418 | **3.92** (3.58–4.32) |

Per-seed throughput (pathfinder ÷ kernel grads/min-ESS): 5.03 / 4.43 / 5.97 / 4.37 / 3.99 → **the permutation is 4.0–6.0× faster on radon (median 4.7×), on all 5 seeds.** Per-seed worst-mean-error (kernel − pathfinder): +0.0113 / +0.0091 / +0.0045 / +0.0048 / +0.0035 sd → **the permutation is slightly less accurate on radon, on all 5 seeds** (reference-sigma maxima 3.80 vs 3.25, same direction). Min-variable ESS is higher for the permuted arm (4,418 vs 3,286 median), consistent with the throughput.

I also re-isolated the permutation on GEV from the 2×2 (orientation=covariance): pathfinder 21.3/21.5/26.8 g/e (R̂ 3/3) vs kernel 96.5/56.4/103.6 g/e (R̂ 3/3) → **the permutation is a 2.6–4.5× SLOWDOWN on GEV, on all 3 seeds.** This is the reverse of "accidental speed-up on GEV."

### (1) Does the permutation stay BLOCKING?
No — I withdraw the BLOCKING status, on two grounds now established:
- The "efficiency regression" that motivated the block was **GEV-specific** and is **reversed on the decisive high-d model**: the permutation is a 4.7× throughput *gain* on radon (all 5 seeds), a 2.6–4.5× *loss* on GEV (all 3 seeds). It is not a general regression.
- The "no demonstrated accuracy benefit" is **confirmed and slightly strengthened**: on radon the permutation is not neutral but *consistently* marginally less accurate (all 5 seeds, +0.0035–0.0113 sd worst-mean-error). So it is not an accuracy fix anywhere I measured.

What is **not** refuted, and is the reason it stays default-on: on scale-heterogeneous geometry the ordering is correctness-critical (the 1e4-spread fixture: unpermuted orientation-arm reads R̂ 12 / ESS 32 vs 1.002 / 30,967 permuted). That is a safety property, independent of the radon/GEV throughput side effects.

### (2) Is the proposed framing a defensible basis to merge unconditional?
Mostly yes, with **two factual corrections**:
- GEV: the permutation is an accidental **slowdown** (2.6–4.5×), not a speed-up. I measured kernel 56–104 g/e vs pathfinder 21–27 g/e, all 3 seeds.
- radon: it is not pure throughput — it carries a small but seed-consistent **accuracy cost** (worst-mean-error 0.017 → 0.021 sd; reference-sigma max 3.25 → 3.80).

Defensible one-line characterization: *a representation-ordering (coordinate-coupling) correction — correctness-critical on scale-heterogeneous geometry, with geometry-dependent throughput side effects (4.7× faster on radon d=175; 2.6–4.5× slower on GEV) and a small radon accuracy cost; not an accuracy improvement.*

**I do not want `imm_order` defaulted to `pathfinder`.** That would ship the catastrophic ordering on scale-heterogeneous models (the fixture is the existence proof). The GEV slowdown is the price of that safety and is one model's side effect. Keep the default `kernel` (permuted); the unpermuted order is a diagnostic/measurement knob, not a safe default.

### (3) What the report must add so this is not mistaken for an accuracy fix
1. State plainly that the permutation is a **coordinate-coupling correction, not a posterior-accuracy improvement**. Cite the fixture (R̂ 12.1 / ESS 32 unpermuted-precision vs 1.002 / 30,967 permuted-covariance) as the reason it is default-on, and cite the radon order A/B (accuracy 0.0166 → 0.0214 sd, all 5 seeds) as the reason it is *not* an accuracy win.
2. Report the geometry-dependent throughput explicitly: radon 18.42 → 3.92 grads/min-ESS (4.7×, all 5 seeds) vs GEV 21.5 → 96.5 g/e (2.6–4.5× slower, all 3 seeds). A reader must see that the sign of the efficiency effect depends on the model.
3. Attribute the GEV posterior-mean-error halving (0.0366 → 0.0205 sd) to the **orientation flip**, not the permutation — the order A/B shows the permutation does not improve (and slightly worsens) accuracy.
4. Note the radon accuracy cost so it is not rediscovered as a regression later: worst-mean-error and reference-sigma both slightly worse for the permuted arm, consistent across all 5 seeds, small in magnitude (max over d=175, so indicative not precise).

**RECOMMEND: merge** (moved from hold). The orientation flip is justified and the permutation is the correct, safe default (correctness on heterogeneous geometry) — both should ship. The change from my first review is that the permutation is no longer a blocker: it is a default-on coordinate-coupling correction whose efficiency effect is model-dependent and whose accuracy effect is null-to-slightly-negative, and it must be documented as such, not as "the metric is now accurate."
