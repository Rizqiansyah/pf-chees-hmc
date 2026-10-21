# Run 2 — REPORT

Branch `pf-r2` (base `pf-r1` `cca5407`), never pushed; `main` untouched at `ecd9ba1`. Orchestrated by
the lead took over after the previous session stalled at R2-WP0, with fast/steady tasks (LEDGER, handover entry). Every
number below is from a committed row file under `results/r2/runs/`; generated tables are in
`results/r2/tables/`; decisions are evaluated clause by clause in `results/r2/LEDGER.md`.

Shapes: **R1** = C=1,024 chains × D=10 draws (owner's objective); **R2** = C=32 × D=1,250 at W=2,000.
Gate: `g_r1` (`tools/r1_bias.py`) — pass iff b² < 0.01 with floor ≤ 0.005; "not resolvable" when the
floor is above 0.005 and b² ≤ max(0.01, 3·floor). Pooled-seed rows (G-R1-pool) are a labelled
diagnostic, never a gate.

## 0. Headline

**Owner's question: does R1 on the GPU beat 32-core CPU NUTS at equal accuracy?** Not on this box
for these models. Both sides here pass `g_r1` on all 3 seeds. The RTX A2000 (6 GB) running the
R1 default path is slower than CPU NUTS on 8 Schools (0.56×), GEV (0.51×) and radon (0.77×), and
1.1× faster on logistic (R2-T4). The GPU's sampling compute (warmup + draws) alone is 1.1–1.9× faster
than the whole CPU run. The loss is a fixed per-call cost of 2.8–10.8 s (init + Pathfinder + ε0). On
radon it is led by an 8 s Pathfinder stage, caused by the `mc` default's 122,500-row pool. Doubling to
C=2,048 is nearly free (+0–27% wall).

**What runs 1 and 2 changed (M4x, same harness on both trees):**
- At R1 the pre-run-1 code passes only 8 Schools (W=100). The current default passes all four:
  8 Schools and GEV at W=25, logistic at W=50, radon at W=300.
- At R2 the current default is 1.3–2.2× faster in wall time and 1.5–4.6× cheaper in grads/ESS on
  every model. GEV is the one regression: about 2× fewer ESS per second and about 3× more divergences.

**Decisions:**
- **D1 = NO.** `iterate` stays `ma`: `raw` passes at smaller W but costs more gradients on 3 of 4
  models.
- **D2 = NO.** `imm_source` stays `mc`, missed by one radon seed's R̂ of 1.0101. Otherwise the closed
  form matches at R2, matches at R1, and halves radon's memory.
- **D3: recommend the metric refresh stays OFF.** It is 1.30× dearer on logistic, never passes radon,
  and is 5.6–36× dearer at R2.

**The re-seed fix holds:** frozen L at C=1,024 is 0.6–1.6× that of the no-refresh run (run 1: 213–283).

**Two new run-1 defects:**
- **E13:** `iterate="raw"` was silently ignored without a refresh.
- **E14:** the WP1 R2 rows undercounted sampling gradients about 39×, which voids run 1's R2 headline.

## R2-T1 — R1 smallest passing W (all 3 seeds pass `g_r1`)

Source: `M2a__r1.jsonl` (no refresh), `M2b__r1.jsonl` (refresh, IT = raw, fixed re-seed); M2c skipped
(trigger not met). Grid W ∈ {25, 50, 100, 200, 300} unless noted. Default path otherwise
(`imm_source=mc`, covariance, `pf_draws=4000` with the pool rule). Full per-seed tables:
`tables/M2a_r1.md`, `tables/M2b_r1.md`, `tables/M2a_smallest.txt`, `tables/M2b_smallest.txt`.

| model | arm | smallest W | med grads/chain | per-seed b²/floor at that W | pooled b²/floor (diag.) | frozen ε | frozen L |
|---|---|---|---|---|---|---|---|
| 8schools | **ma (default)** | 25 | 540 | .00071/.0016 .00068/.0016 .0018/.0015 | .00037/.00052 | 0.262 | 28 |
| 8schools | raw | 25 | 861 | .00050/.00094 .00055/.00086 .00076/.00092 | .00041/.00029 | 0.282 | 44 |
| 8schools | refresh diag | 25 | 616 | .00052/.0019 .0012/.0019 .0012/.0019 | .00036/.00064 | 0.176 | 28 |
| 8schools | refresh diag, no re-seed (W=50 only) | 50 | 1,306 | | | 0.160 | 34 |
| 8schools | refresh fisher (W 50,100) | 50 | 1,165 | | | 0.204 | 30 |
| gev | **ma (default)** | 25 | 270 | .0014/.0011 .00033/.0014 .00021/.0011 | .00019/.00038 | 0.519 | 14 |
| gev | raw | 25 | 629 | .00081/.00063 .00023/.00087 .00004/.00057 | .00006/.00023 | 0.420 | 34 |
| gev | refresh diag | 25 | 324 | .0022/.0011 .00018/.0013 .00041/.0013 | .00040/.00040 | 0.240 | 18 |
| gev | refresh diag, no re-seed | 50 | 707 | | | 0.206 | 21 |
| gev | refresh fisher | 50 | 1,472 | | | 0.240 | 39 |
| logit_1000x25 | ma (default) | 50 | 319 | .0039/.0024 .0045/.0024 .0011/.0025 | .0023/.00080 | 0.574 | 9 |
| logit_1000x25 | **raw** | 25 | 297 | .00055/.0022 .0020/.0021 .00060/.0022 | .00021/.00072 | 0.540 | 17 |
| logit_1000x25 | refresh diag | 25 | 387 | .0024/.0025 .0016/.0023 .0017/.0023 | .0011/.00080 | 0.185 | 30 |
| logit_1000x25 | refresh diag, no re-seed | 50 | 507 | | | 0.146 | 16 |
| logit_1000x25 | refresh fisher | 50 | 1,186 | | | 0.229 | 56 |
| radon_noncentered | **ma (default)** | 300 | 1,770 | .0034/.0031 .0034/.0039 .0047/.0039 | .0018/.0012 | 0.838 | 16 |
| radon_noncentered | raw | 100 | 2,037 | .0025/.0027 .0021/.0030 .0017/.0037 | .00087/.0010 | 0.697 | 38 |
| radon_noncentered | refresh diag | none ≤ 300 | — | W≥50: all "not resolvable" (floors .0055–.0068; b² .0063–.0116) | .0022/.0017 … .0036/.0020 | 0.15–0.18 | 15–22 |
| radon_noncentered | refresh diag, no re-seed | none (W=50) | — | not resolvable ×3 | .0064/.0017 | 0.178 | 22 |
| radon_noncentered | refresh fisher | none ≤ 100 | — | mixed pass / not resolvable | .0027/.0015, .0024/.0018 | 0.25–0.32 | 22–39 |

Radon is **gateable at R1 without the refresh** (ma W=300, raw W=100) — run 1 never ran W=300 and its
"raw" rows ran ma (E13). The refresh's diagonal metric discards Pathfinder's dense correlation
(d=175), roughly doubling radon's noise floor, so its cells are unresolvable rather than failing.
**Fixed re-seed holds at C=1,024**: refresh L / no-refresh L at W=50 is 0.6 / 0.7 / 1.6 / 0.65
(run 1: 213–283 absolute).

## R2-T2 — M1: closed-form vs Monte-Carlo Pathfinder covariance (D2)

Source `M1__{r2,r1,mem}.jsonl`; tables `tables/M1_*.md`.

| model | R2 med grads/ESS closed / mc (ratio) | R2 seeds R̂≤1.01 closed / mc | R1 W=50 g_r1 closed / mc |
|---|---|---|---|
| 8schools | 57.39 / 55.76 (1.03) | 3 / 3 | pass×3 / pass×3 |
| gev | 146.9 / 141.5 (1.04) | 2 / 2 | pass×3 / pass×3 |
| logit_1000x25 | 77.34 / 75.86 (1.02) | 3 / 3 | pass×3 / pass×3 |
| radon_noncentered | 296.3 / 302.5 (0.98) | **2 / 3** (closed seed 2 R̂ = 1.0101) | fail×3 / fail×3 |

Radon R1 peak memory (separate process, seed 1): **653.5 MiB closed (pool 4,096 rows) vs 1,242 MiB mc
(pool 122,500 rows)**. D2 = **NO** on clause (b2) alone (one radon seed, R̂ 1.0101 vs 1.01). R2-WP2
test 1 also showed the run-1 `lbfgs_lowrank` arm is the same matrix as `pf_closed` (to 1e-16).

## R2-T3 — M3: R2 matrix (5 seeds) and GEV G3 (D1)

Source `M3__r2.jsonl`; full per-seed table `tables/M3_r2.md` (R̂, bulk/tail ESS, E-BFMI, divergences,
b²); mean shift in sd units `tables/M3_shift_bound.md`.

| model | arm | seeds R̂≤1.01 | med grads/ESS | min bulk ESS (range) | min tail ESS (range) | E-BFMI min | divs/1k sampling | worst mean shift, sd units (≤√b²) |
|---|---|---|---|---|---|---|---|---|
| 8schools | ma | 5/5 | 54.86 | 20,160–22,530 | 17,870–19,870 | 0.96 | 0–0.05 | ≤ 0.018 |
| 8schools | raw | 5/5 | 64.24 | 19,830–21,910 | 15,320–19,790 | 0.95 | 0–0.05 | ≤ 0.021 |
| 8schools | raw+refresh | 3/5 | 305.2 | 449–4,313 | 1,149–7,711 | 0.79 | 0 | ≤ 0.038 |
| gev | ma | 3/5 | 141.5 | 1,350–15,270 | 2,321–13,760 | 1.00 | 0.43–1.18 | ≤ 0.029 |
| gev | raw | 4/5 | 254.9 | 1,845–18,310 | 3,439–11,990 | 0.93 | 0.55–14.7 | ≤ 0.025 |
| gev | raw+refresh | 2/5 | 804.0 | 871–33,860 | 1,555–20,040 | 0.62 | 0.03–2.0 | ≤ 0.038 |
| logit_1000x25 | ma | 5/5 | 77.33 | 16,570–18,530 | 17,680–19,020 | 0.96 | 0 | ≤ 0.021 |
| logit_1000x25 | raw | 5/5 | 80.12 | 13,870–21,160 | 16,740–19,990 | 0.95 | 0 | ≤ 0.026 |
| logit_1000x25 | raw+refresh | 1/5 | 636.4 | 2,224–3,279 | 4,775–7,453 | 0.88 | 0 | ≤ 0.039 |
| radon_noncentered | ma | 5/5 | 297.2 | 4,056–4,841 | 5,182–6,392 | 0.60 | 0 | ≤ 0.031 |
| radon_noncentered | raw | 5/5 | 324.2 | 3,899–4,604 | 4,132–6,307 | 0.59 | 0 | ≤ 0.029 |
| radon_noncentered | raw+refresh | 1/5 | 10,750 | 65–3,244 | 55–3,522 | 0.27 | 0 | **≤ 0.205** (R̂ 1.37–1.48 on 3 seeds) |

The sd-unit shift is √b²_max in unconstrained space: exact where the arg-max moment is first-order
(all GEV ma/raw seeds), an upper bound otherwise.

**GEV G3 (A1.3; informational, nothing changed on it).** Source `M3g__gev_meanshift.jsonl`, the run-1
G3 scorer (`results/r2/drivers/g3_meanshift.py`, a copy of `tools/gev_g3b.py` with configurable
arms). Seeds 1–5 reproduce the M3 rows exactly (same divergences and R̂). Clauses: R̂ ≤ 1.01;
sampling divergences ≤ 0.10/1k (the run-1 shipped maximum); mean shift "within 1σ" in G3's z form
(shift ÷ run MCSE), plus the same shift in reference-sd units.

| arm | seed | R̂ ≤ 1.01 | divs/1k ≤ 0.10 | z ≤ 1 | worst shift, sd units | frozen ε / L |
|---|---|---|---|---|---|---|
| ma (default) | 1 | 1.0085 ✓ | 1.100 ✗ | 1.35 ✗ | 0.0101 | 0.978 / 8 |
| ma | 2 | 1.0125 ✗ | 0.425 ✗ | 1.76 ✗ | 0.0284 | 0.857 / 7 |
| ma | 3 | 1.0078 ✓ | 1.175 ✗ | 2.00 ✗ | 0.0144 | 0.806 / 9 |
| ma | 4 | 1.0307 ✗ | 0.525 ✗ | 1.04 ✗ | 0.0095 | 0.882 / 5 |
| ma | 5 | 1.0015 ✓ | 1.200 ✗ | 2.81 ✗ | 0.0195 | 0.817 / 17 |
| raw | 1 | 1.0174 ✗ | 1.025 ✗ | 1.84 ✗ | 0.0180 | 0.990 / 6 |
| raw | 2 | 1.0082 ✓ | 0.650 ✗ | 2.33 ✗ | 0.0250 | 0.862 / 9 |
| raw | 3 | 1.0089 ✓ | 1.050 ✗ | 2.33 ✗ | 0.0185 | 0.808 / 9 |
| raw | 4 | 1.0045 ✓ | 0.550 ✗ | 1.85 ✗ | 0.0134 | 0.794 / 12 |
| raw | 5 | 1.0018 ✓ | 14.700 ✗ | 1.82 ✗ | 0.0118 | 0.707 / **269** |
| raw+refresh | 1 | 1.0296 ✗ | 0.050 ✓ | 1.34 ✗ | 0.0215 | 0.575 / 8 |
| raw+refresh | 2 | 1.0015 ✓ | 0.025 ✓ | 3.61 ✗ | 0.0246 | 0.404 / 30 |
| raw+refresh | 3 | 1.0148 ✗ | 0.075 ✓ | 0.95 ✓ | 0.0195 | 0.568 / 9 |
| raw+refresh | 4 | 1.0018 ✓ | 2.000 ✗ | 1.77 ✗ | 0.0095 | 0.450 / **282** |
| raw+refresh | 5 | 1.0012 ✓ | 0.825 ✗ | 3.93 ✗ | 0.0212 | 0.354 / **363** |

No arm meets G3 as written on any seed except refresh seed 3. In reference-sd units every shift is
0.0095–0.028 sd, so the means are far inside 1σ. G3's z form fails because the run MCSE at 40,000
draws is about 0.01 sd. That is the A1.3 owner question: the clause is unchanged here, and both forms
are reported. Divergences are the binding G3 failure for the default path, at 0.43–1.20 per 1k against
0.10. `raw` can also run away in L on GEV (seed 5: L 269, 14.7 divs/1k).

## R2-T4 — Time to pass: 32-core CPU NUTS vs GPU R1 (headline)

CPU: `M5__cpu_nuts.jsonl` (PyMC NUTS, 32 chains on 32 cores, warm C cache, no GPU job running);
cheapest (tune + draws) with `g_r1` pass on all 3 seeds. GPU: `M4__*.jsonl`, RTX A2000 6 GB,
C=1,024 × D=10, the default path at its M2a smallest passing W (plus the best arm where different),
warm = median of rep-1 rows over 3 seeds, cold = rep-0 seed 1, peak from a separate single-cell process.
Speed-up = CPU wall / GPU warm wall (> 1 means the GPU is faster).

| model | CPU config | CPU wall (median) | CPU core-s | GPU config | GPU warm wall | GPU cold wall | GPU peak | warm speed-up | b² CPU / GPU (seeds 1–3) |
|---|---|---|---|---|---|---|---|---|---|
| 8schools | tune 200 + draws 250 | 3.54 s | 15.7 | default, W=25 | 6.31 s | 35.9 s | 224 MiB | **0.56×** | .0027 .0010 .0012 / .0007 .0007 .0018 |
| gev | tune 200 + draws 100 | 5.36 s | 19.5 | default, W=25 | 10.49 s | 46.6 s | 288 MiB | **0.51×** | .00048 .00016 .00051 / .0014 .00033 .00021 |
| logit_1000x25 | tune 200 + draws 250 | 9.27 s | 198.7 | default, W=50 · raw, W=25 | 8.42 s · 8.35 s | 52.2 s · 51.7 s | 582 MiB | **1.10× · 1.11×** | .0028 .00071 .0016 / .0039 .0045 .0011 · .00055 .0020 .00060 |
| radon_noncentered | tune 200 + draws 800 † | 19.13 s | 273.5 | default, W=300 · raw, W=100 | 24.93 s · 24.96 s | 85.2 s · 85.6 s | 1,242 MiB | **0.77× · 0.77×** | .00066 .00063 .0013 / .0032 .0059 .0053 · .0025 .0021 .0016 |

† The plan grid (draws ≤ 400) has no radon pass: all cells are "not resolvable", with b² 0.001–0.003
and floors > 0.005. M5x extended draws to {800, 1,600} (DEVIATED). This is the cheapest of the tested
grid; draws between 400 and 800 were not tried.

Where the GPU time goes (warm, median s; `stage_s`):

| model | init | Pathfinder | ε0 | warmup | sampling | fixed (init+PF+ε0) | compute (warmup+sampling) | CPU wall ÷ GPU compute |
|---|---|---|---|---|---|---|---|---|
| 8schools | 0.53 | 1.15 | 1.07 | 1.69 | 1.52 | 2.75 | 3.21 | 1.10 |
| gev | 1.55 | 1.89 | 1.48 | 2.68 | 2.09 | 4.92 | 4.77 | 1.12 |
| logit_1000x25 | 0.51 | 1.38 | 1.14 | 3.22 | 1.73 | 3.03 | 4.95 | 1.87 |
| radon_noncentered | 0.93 | 8.03 | 1.79 | 7.36 | 2.55 | 10.75 | 9.91 | 1.93 |

C=2,048 at the same W (seed 1, warm) costs 6.49, 10.51, 10.21 and 31.68 s, which is +3%, +0%, +21%
and +27% for twice the draws.

## Before vs after run 1 (owner request, M4x)

Source `M4x__r2.jsonl` and `M4x__r1.jsonl` (`tables/M4x_r1.md`). "Legacy" is the pre-run-1 tree
`ecd9ba1` (`--legacy ~/pf/legacy`, its own defaults: precision orientation, permuted metric on
gev/radon, `ma`). "Current" is the `pf-r2` default path. The same harness and the same exact gradient
counter score both (sampling grads / (1250·L) = 0.50–0.55 in every row, so the E14 undercount is gone).

**R2 (C=32 × D=1,250, W=2,000), 3 seeds, warm (rep-1) walls:**

| model | warm wall legacy → current | speed-up | med grads/ESS legacy → current | min(bulk,tail) ESS per s, median legacy → current | max R̂ legacy / current | divs/1k legacy / current |
|---|---|---|---|---|---|---|
| 8schools | 10.6 → 8.4 s | 1.27× | 97.6 → 55.8 (1.75×) | 1,872 → 2,310 (1.23×) | ≤1.002 / ≤1.001 | 0 / ≤0.03 |
| gev | 20.5 → 12.0 s | 1.71× | 208.2 → 141.9 (1.47×) | **607 → 282 (0.46×)** | ≤1.008 / ≤1.012 | 0.17–0.33 / **0.42–1.18** |
| logit_1000x25 | 36.1 → 16.3 s | 2.21× | 350.7 → 76.0 (4.61×) | 371 → 1,109 (2.99×) | 1.002 / ≤1.002 | 0 / 0 |
| radon_noncentered | 45.2 → 29.5 s | 1.53× | 782.5 → 291.5 (2.68×) | 93 → 157 (1.69×) | ≤1.009 / ≤1.009 | 0 / 0 |

GEV is the one regression at R2. The current covariance metric is about 1.7× faster in wall time, but
its worst bulk ESS fell from 3,069–15,667 to 2,192–3,705, so ESS per second fell by about 2×. Its
divergences also rose about 3×. On `grads_per_ess_worst` (the harness metric) it still improves
(1.47×). The two metrics disagree because that metric's worst-case ESS is not min(bulk, tail).
Both are reported.

**R1 (C=1,024 × D=10), legacy with `--pf-draws 1`, smallest W passing all 3 seeds:**

| model | legacy | current default |
|---|---|---|
| 8schools | W=100, 1,283 grads | W=25, 540 grads |
| gev | none ≤ 300 (pass/fail/pass at W=200 and 300; pooled b² 0.0076–0.028) | W=25, 270 grads |
| logit_1000x25 | none ≤ 300 (pooled b² 0.012 at W=300) | W=50, 319 grads |
| radon_noncentered | none ≤ 300 (pooled b² 65 → 0.031 from W=25 to W=300) | W=300, 1,770 grads |

At the owner's R1 target, the run-1/run-2 fixes turn three never-passing models into passing ones,
and cut 8 Schools' cost 2.4×.

## R2-T5 — Decisions

| decision | rule clauses (Part B §3) | evaluation | deciding rows | outcome |
|---|---|---|---|---|
| **D2** `imm_source` → `pf_closed` | (a) WP2 tests 1–3; (b) R2 grads/ESS ≤ 1.10× and R̂-count not lower; (c) R1 verdicts ≥; (d) radon peak ≤ | (a) pass; (b) cost pass (0.98–1.04×), **R̂ count fails on radon 2 vs 3**; (c) pass (identical); (d) pass (653 vs 1,242 MiB) | `M1__r2.jsonl` radon `m1:closed` seed 2 (R̂ 1.0101) | **NO** — default stays `mc` |
| **D1** `iterate` → `raw` | (a) raw smallest W ≤ ma on 4/4, < on ≥2; (b) raw grads at its W ≤ ma's at its W on ≥3/4; (c) R2 grads/ESS ≤ 1.15× every model and R̂ count not lower | (a) pass (25≤25, 25≤25, 25<50, 100<300); (b) **fail, 1/4** (861>540, 629>270, 297≤319, 2,037>1,770); (c) **fail** (8schools 1.17×, gev 1.80×; R̂ counts not lower) | `M2a__r1.jsonl`, `M3__r2.jsonl` | **NO** — default stays `ma` |
| **D3** metric refresh (recommendation only) | on iff lowers grads at smallest passing W on ≥2/4 and raises > 1.2× on none; or is the only arm passing radon ≤ 300 | lowers on 8schools (616<861), gev (324<629); **raises 1.30× on logit, radon none**; not the only radon pass | `M2b__r1.jsonl` vs `M2a__r1.jsonl` raw | **recommend OFF** (default unchanged); M3 shows it harmful at R2 (5.6–36× cost, radon R̂ to 1.48) |

## R2-T6 — Errata and coverage

Run-1 errata E1–E12 applied in `results/REPORT.md` and `results/REVIEW-HANDOFF.md` (R2-WP4, merged
`25d2618`): E1, E2, E3a/b, E4, E5, E7 corrected inline; **E3c narrowed** (the plan's "radon covraw
fails 6/6" was wrong — W=200 seed 2 passes, b² 0.00709 / floor 0.00413); **E6 deleted** (no row file
exists); E8–E12 tagged void. New run-1 defects found in run 2 (LEDGER):

- **E13** — single-phase `_run_chees` ignored `iterate`, so all 88 run-1 rows that requested `raw`
  without a refresh ran `ma` (incl. T1 `covraw`, `prodpool*`, `r2_poolcap*`, `WP2__lgrowth`). The run-1
  R1 passes "with (covariance, raw)" were (covariance, ma) — the current default. Fixed `5982ee7`.
- **E14** — `WP1__rebaseline*.jsonl` and `WP1__covarm.jsonl` count ≈1/39 of sampling gradients
  (`grads_per_chain_sampling / (1250·L) = 0.013` vs 0.50–0.56 in every later row), so run-1's R2
  grads/ESS headline (35.5/71.2/46.1/182.3 vs 58.2/73.4/151.7/501.9) is void as absolute numbers.

__COVERAGE__

## Open questions for the owner (≤ 5)

1. **Flip D2 anyway?** `pf_closed` missed only by one radon R̂ (1.0101 vs 1.01). It is cost-neutral
   at R2, has identical R1 verdicts, halves radon's R1 peak (653 vs 1,242 MiB) and removes the O(d³)
   pool. It should also shrink radon's 8 s Pathfinder stage, the largest GPU overhead.
   *Settles it:* 5 more radon R2 seeds per arm (the R̂ count over 8), plus an M4 warm timing of radon
   with `--imm-source pf_closed`.
2. **What does a "call" look like in your workload?** The GPU loses on per-call fixed cost, not on
   sampling: 2.8–4.9 s on the small models, 10.8 s on radon, even warm. If you run many fits per
   compiled model, or larger C, the balance changes.
   *Settles it:* profile init, Pathfinder and ε0 (run-1 WP6.3, out of scope here), and time a batch
   of fits in one process.
3. **How should G3 read for GEV (A1.3)?** Mean shifts are 0.010–0.028 sd, but the z form (÷ run
   MCSE ≈ 0.01 sd) fails on every seed. The covariance metric also raises GEV divergences to
   0.4–1.2/1k (legacy 0.17–0.33) and halves its ESS per second.
   *Settles it:* a GEV R2 arm at `target_accept=0.9` (divergences, ESS/s), after you choose the G3 form.
4. **Does the crossover favour the GPU at your model's size?** The CPU cost grows fastest with d
   (radon: 274 core-s), and the GPU's compute-only advantage is largest there (1.9×).
   *Settles it:* the M4/M5 pair on your model.
5. **Per-model iterate for radon?** `ma` W=300 and `raw` W=100 tie in wall time (24.9 s). `raw`
   passes with 3× fewer warmup steps but runs away in L on GEV (L 269, 14.7 divs/1k on one seed),
   so it stays opt-in. *Settles it:* nothing further is needed unless you want a per-model recommendation.

## Caveats

- `grads_per_chain_total` = warmup (both phases) + sampling integration steps per chain; it excludes the
  32 ε-search probes (C-independent) and the fisher refresh-pool gradients (≤ 20/chain), and
  Pathfinder's shared gradients (R2-WP3b SUMMARY item 6).
- GPU walls include fixed per-call stages (init, Pathfinder, ε0) that do not scale with C·D; they are
  reported per stage so a batch-of-calls user can discount them.
- CPU NUTS draws=50 cells are `g_r1`-ineligible at 32 chains (C·D = 1,600 < 2,000) by construction.
- M2b/M2c/M3/M4 order deviated from A8 only in scheduling (M2b after M2a, then M3, M5, M4); code trees
  were identical (`a0f644d…/516dd5e…`) across every measurement step from M1 to M4x.
- ~7 GPU-hours idle overnight (lead watcher bug, LEDGER); no data affected.
