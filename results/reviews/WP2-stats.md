# WP2 — statistical-validity review (vendored ChEES adapter, raw vs ma `log T` iterate)

Reviewer: `wp2-stats-review-x` (independent, worker model). Subject: `wp/wp2` @ `b02a6e8`,
checkout `/home/akbar/pf/wp2`. All numbers below were printed from
`/home/akbar/pf/wp2/results/runs/WP2__lgrowth.jsonl` (32 rows, git_sha `3ada873`) or from
`results/WP2/SUMMARY.md` in this session.

## File structure — read this first, it changes several answers

Each `(model, arm)` has **four rows = two device runs × (rep 0 cold, rep 1 warm), all at seed 43**:
one run on `cpu:0`, one on `NVIDIA RTX A2000:0`. The SUMMARY says so in prose (line 27: "GPU rows (16)
are the decision rows; the CPU rows (16) are a cross-device sanity set"), but the **row keys
`(model, arm, seed, rep)` are not unique** — the device field is the only disambiguator. Consequence:
every "2 seeds" description of this matrix (including the task's own final message and the lead's
ledger entry) is wrong; it is **1 seed × 2 devices**. Reproducibility is to ~10% per the SUMMARY's own
line 65, so seed spread is not estimated anywhere in this file.

## Measurement-integrity context (the `r1_bias.py` timer bug)

**Not applicable to these rows.** WP2's harness is `tools/wp2_lgrowth.py`, which does not call
`nominal_length` or any per-warmup-step diagnostic (0 matches for `nominal_length` in `tools/`). Its
`frozen_L_nominal` is the scalar `traj_len_adj` the vendored adapter computes at freeze
(`pfcheeshmc/utils.py:348`), read after the timed region, and `stage_s.warmup` brackets only the sampler
call (`wp2_lgrowth.py:97-111`). The rows predate `wp/wp3` `9d5c30d` by a day, but that fix was in a
different tool. **WP2's wall numbers are not inflated by that bug.** The wall confound that actually
exists here is the **device mix**, below.

## 1. Frozen L / ε by model, device and run — and the accept/kill rule

GPU rows (decision rows per the SUMMARY):

| model | arm | frozen ε | frozen L | T = ε·L |
|---|---|---|---|---|
| 8schools | A prec,ma | 0.19021 | 41.34 | 7.86 |
| 8schools | B cov,ma | 0.53111 | 21.46 | 11.40 |
| 8schools | C cov,raw | 0.51518 | **22.32** | 11.50 |
| 8schools | D cov,raw,t0 | 0.53500 | 20.24 | 10.83 |
| logistic | A prec,ma | 0.00208 | 130.20 | 0.271 |
| logistic | B cov,ma | 0.64763 | **27.50** | 17.81 |
| logistic | C cov,raw | 0.64852 | **26.04** | 16.89 |
| logistic | D cov,raw,t0 | 0.64879 | 24.23 | 15.73 |

CPU rows (same seed 43, `cpu:0`): 8schools A 44.53 / B 21.40 / C 21.67 / D 19.78; logistic A 116.32 /
B 26.80 / C 25.01 / D 24.90.

**Kill rule** ("raw's frozen L worse than ma at W=2,000 on two or more models"): raw is **worse on
8 Schools on both devices** (22.32 vs 21.46 GPU, +4.0%; 21.67 vs 21.40 CPU, +1.3%) and **better on
logistic on both devices** (26.04 vs 27.50 GPU, −5.3%; 25.01 vs 26.80 CPU, −6.7%). Worse on **one** of
two models → **kill criterion NOT met → ACCEPT as an option: CONFIRMED.** The direction is consistent
across devices, which is the best available substitute for a second seed; the magnitude (1–7%) is not
separable from the ~10% reproducibility floor the SUMMARY itself states, so "raw is worse on 8 Schools"
is a direction, not a measured regression.

## 2. Is 5.5% better / 3.7% worse separable from seed noise?

**No.** There is one seed. The within-file noise floor is visible in the plan-reproduction column
(SUMMARY line 37: the same variant reads 21.7 on CPU and 22.3 on GPU, both seed 43 — a 2.8% device
delta on the same quantity). Against a 2.8% device noise floor, a 3.7% cross-device regression on 8
Schools and a 5.5% gain on logistic are **both inside the noise floor**, even though each is consistent
in sign across devices. The honest verdict on the accept/kill data is therefore:

- "worse on one model, better on one" — **CONFIRMED, direction consistent across devices**;
- "the regression is 3.7%" / "the gain is 5.5%" — **NOT ESTABLISHED** (1 seed, floor ≈ 3–10%);
- the SUMMARY's own wording ("one seed cannot separate 5% from noise") is the correct one, and its
  "ACCEPT as an option, default unchanged, flip goes to G3" follows.

## 3. The warmup-wall "2.1× saving" — real, but attributed to the wrong factor

Same-device, warm (rep 1) `stage_s.warmup` seconds:

| model, device | A prec,ma | B cov,ma | C cov,raw | D cov,raw,t0 |
|---|---|---|---|---|
| 8schools GPU | 4.08 | 3.48 | 3.51 | 3.57 |
| 8schools CPU | 2.19 | 1.97 | 1.96 | 2.00 |
| logistic GPU | **16.72** | **7.30** | 7.86 | 7.92 |
| logistic CPU | 78.76 | 27.42 | 31.43 | 31.96 |

- The SUMMARY's headline "logistic **A 16.7 s → C 7.9 s (2.1×)**" is warm-vs-warm, GPU-vs-GPU, and the
  ratio is real (16.72/7.86 = 2.13×). **But it is an A→C comparison, i.e. metric AND iterate together.**
- **The iterate-specific comparison B→C is not a saving: 7.30 → 7.86 (GPU, raw 7.7% slower) and
  27.42 → 31.43 (CPU, raw 14.6% slower).** 8 Schools is neutral (3.48→3.51, 1.97→1.96).
- The mechanism is in the trace columns: raw starts its warmup at a *larger* L (logistic L@100 = 25.4
  CPU / 42.1 GPU versus 8.3 for ma), so the first ~1,000 warmup steps integrate more leapfrogs; it
  converges to a similar plateau but has already spent the extra work.
- **SUMMARY line 59 — "raw saves 2.1× logistic warmup wall (16.7 → 7.9 s)" — is a misattribution.**
  Arm B (covariance metric, iterate unchanged) already reaches 7.30 s; the raw iterate costs +7% on top.
  This is material because line 59 lists the wall saving among "the data for" the default-flip decision
  (G3), which goes to the owner. The flip's wall argument as written is backwards.
- **The wp3 timer bug does not rescue or hurt this claim** (see integrity section): no per-step
  diagnostic in this harness, so none of the above is inflated by it.

## 4. "Metric sets the frozen point, iterate sets the speed" — CONFIRMED by the rows

Frozen trajectory time T = ε·L:

| model, device | A→B (metric) | B→C (iterate) |
|---|---|---|
| 8schools GPU | 7.86 → 11.40 (**1.45×**) | 11.40 → 11.50 (1.01×) |
| 8schools CPU | 7.88 → 11.22 (1.42×) | 11.22 → 11.11 (0.99×) |
| logistic GPU | 0.271 → 17.81 (**65.7×**) | 17.81 → 16.89 (0.95×) |
| logistic CPU | 0.241 → 17.55 (**72.8×**) | 17.55 → 16.05 (0.91×) |

The metric moves the frozen point by 1.4×–73×; the iterate moves it by 1–9%. The rows fully support
"the plan mis-attributes the gain to the iterate" — with the refinement from §3 that the iterate's
contribution to *warmup cost* is a small **penalty** at W=2,000, not a saving.

## 5. Equivalence test strength — strongest available; 9 passed in 14.87 s (observed)

`tests/test_chees_vendored.py::test_ma_mode_is_bit_for_bit_upstream` (line 80) drives the vendored and
installed adapters for **64 updates and asserts every state field is bit-for-bit equal at every step**
(`_assert_states_equal` at lines 67–77, called in the loop at line 99, using `jnp.equal` — exact, not
tolerance-based). This is the form that makes "vendor-and-fix" safe: a state-ordering difference (value
consumed this step vs stored for next) would diverge at step 1 and fail. Run result, observed:
**9 passed in 14.87 s** (forced CPU). `iterate="raw"` divergence is pinned by
`test_raw_iterate_grows_L_faster_in_the_early_phase`, harmonic pooling (divergence exclusion) by
`test_harmonic_pooling_excludes_divergent_chains`, metric-shape guard by `test_metric_shape_guard_raises`.

## 6. SUMMARY claims the rows do not support

1. **Line 59: "raw saves 2.1× logistic warmup wall"** — REFUTED as an iterate claim (§3): the saving is
   the metric's; raw costs +7% (GPU) / +14% (CPU) relative to B. Material for the owner-facing G3 data.
2. **Line 37: "the same variant reads 21.7 on CPU seed 43, so the delta is seed, not device"** — confused
   direction. 21.7 (CPU) vs 22.3 (GPU) at the same seed is the **device** delta; the **seed** delta is
   22.3 (this run, GPU seed 43) vs the plan's 19 (GPU seed 42). The SUMMARY contradicts itself here,
   since line 65 correctly attributes the 116-vs-130 spread to device.
3. Every number in the reproduction table (lines 29–37) **matches the rows** as printed above (A
   0.00208/130.20 GPU, 0.00207/116.32 CPU; B 0.64763/27.50; C 0.64852/26.04; D 0.64879/24.23; 8schools
   A 0.19021/41.34, B 0.53111/21.46, C 0.51518/22.32, D 0.53500/20.24). The "7 of 8 within 8%" claim
   holds as printed.
4. Data hygiene: duplicate `(model, arm, seed, rep)` keys (one per device) — the file is not uniquely
   addressable by the standard row key; a consumer that dedupes or takes the first hit silently picks a
   device. Disambiguated by `device`, so non-blocking, but a uniqueness guard on
   `(model, arm, seed, rep, device)` belongs in `wp2_lgrowth.py`.

## BLOCKING

- **B1.** Fix SUMMARY line 59 before this branch's numbers are cited for the G3 default-flip decision:
  replace "raw saves 2.1× logistic warmup wall (16.7 → 7.9 s)" with the same-device decomposition
  (A→B metric 16.72→7.30 = 2.29×; B→C iterate 7.30→7.86 = raw 7.7% slower GPU, 14.6% slower CPU).
  This is a one-line docs change, not a re-measurement.

## NON-BLOCKING

- 1 seed only: no claim about the 3.7%/5.5% magnitudes should be quoted without "direction consistent
  across devices, magnitude inside the ~10% reproducibility floor".
- Row-key uniqueness: add the `device` field to the dedup key or guard.
- Line 37's seed/device confusion (Q6.2).

## NIT

- L@500 "seed-sensitive intermediate" (line 33) is really a convergence-phase value; the frozen L and
  the warmup-wall number are the decision-relevant endpoints, and both are stable.
- The plan's raw-target L@500 = 19 is not reproduced (34.8); the SUMMARY correctly logs it as an
  intermediate-curve mismatch rather than an endpoint failure.

## RECOMMEND

**RECOMMEND: merge** — the vendored adapter is bit-for-bit proven (Q5), the accept/kill verdict
("ACCEPT as an option, default stays `ma`") is correctly supported (Q1–Q2), and the metric-vs-iterate
attribution the branch draws for the frozen point is confirmed (Q4). **Condition: B1 first** — the wall
saving quoted to the owner is the metric's, and the iterate's own wall contribution is a 7–15% cost,
which the flip rationale must not invert.
