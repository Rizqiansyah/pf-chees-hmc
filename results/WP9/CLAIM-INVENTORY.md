# WP9 claim-retraction inventory

Pre-work for REPORT table T6. Read-only pass over `README.md`, `docs/ladder-results.md`,
`docs/gpu-upgrade-plan.md`, `results/LEDGER.md`, plan `docs/review-2026-10/pf-chees-hmc-implementation-plan.md`
(at `pf-r1` @ `9f892b2`). 40 rows; ranked shipped-default / README-headline first.

## What is settled vs still measured

**Settled by the WP1 2×2 fixture (ordering × orientation)** — 4-coordinate fixture, posterior
variances 1e2 / 3 / 1 / 1e-2, value_vars `[delta,beta,zeta,alpha]` vs kernel order
`[alpha,beta,delta,zeta]`, 32 chains, 500 warmup, 1,000 draws, seeds 5–7, CPU:

| cell | max R̂ | min bulk ESS |
|---|---|---|
| permuted + covariance | 1.0020–1.0022 | 30,121–33,495 |
| permuted + precision | 12.10–13.08 | 32–38 |
| unpermuted + covariance | 1.0069–1.1451 | 166–3,176 |
| unpermuted + precision (= the shipped default) | 10.37–11.71 | 33 |

Orientation is a **correctness-class failure of the shipped default**, not a tuning preference.
The two factors have distinguishable signatures: an ordering error kills specific coordinates; an
orientation error kills every wide coordinate. This fixture supersedes the LEDGER
`CONTRADICTION` entry that §0.1 "is refuted for the kernel path" — the momentum-draw probe was
necessary but not sufficient, because end-to-end dual-averaging + ChEES adaptation on the
wrong-oriented metric does not recover. Record trail: the original kernel-path contradiction is
commit `699be6d` in `results/LEDGER.md`; its formal retraction, resting on the 2×2 fixture, is
commit `a03dc0f` on `pf-r1` (LEDGER entry "RETRACTION of entry 699be6d").

**Still measured (do not cite as done):** the WP1 real-model 5-seed confirmation, in progress —
4 models × 5 seeds, arms `legacy` (unpermuted+precision), `wp1` permuted+precision, `wp1`
permuted+covariance, plus the two low-rank `imm_source` arms. Rows marked `PENDING R-run` name the
missing number.

**Current code state for T6:** as of `4460dc7` the shipped default is still `precision`
(inv(cov)) — the failing orientation — so every historical number in the repo's docs came from it
unless the model's metric happened to be near-isotropic. Orientation is now an explicit
`imm_orientation` knob, so the R-run and every re-measure below can select the arm directly.
Recommended actions that would flip the default (rows 2, 3, 4, 14, 20) do so only on G3:
≥ 5 seeds, divergence rate not increased, posterior means within 1σ — those are the numbers the
in-progress R-run must still produce.

**Protected by 8 Schools' identity permutation:** only 8 Schools — its value-vars order equals
sorted-key order, so no coordinate misassignment; its historical numbers came from the failing
*orientation only* (fixture cell 4). **Not protected:** logistic, GEV and both radon
parameterisations (permutation + orientation; `order_check.py`: GEV and radon DIFFERENT).
Magnitude note: the fixture's 10⁴:1 spread makes the failure catastrophic; damage scales with
model anisotropy, which is why historical 8 Schools (cond 22) read R̂ ≈ 1.002 and logistic read
ε 0.0021 / L 136 — same failure, milder models. The R-run quantifies per model.

## Inventory

| # | claim (≤15 words) | file:line | what it was based on | verdict | refuting/confirming evidence | recommended action |
|---|---|---|---|---|---|---|
| 1 | Fisher low-rank: "1.22×, 1.96×, 0.19×", "R-hat unchanged" | README.md:78 | E4 A/B at 32 GPU ch (ladder E4 table, `~/pf-runs/e4-*.jsonl`) | CONFOUNDED | LEDGER `9f892b2`: arms never shared orientation (mc feeds M⁻¹, low-rank arms feed ≈Σ); baseline arm is the fixture's failing cell; "R-hat unchanged" contradicted by fixture cell 4 | Delete the ratios; re-measure after WP1 with all arms on one explicit convention; cite the orientation mismatch as the mechanism |
| 2 | "GPU is about 3.6x and 2.6x at 128, ~1.2x at 32; CPU 9.5 s vs 15.0 s" | README.md:74 | E1 grid + equal-draw walls, all GPU rows on shipped (failing) arm | CONFOUNDED | 2×2 fixture cell 4 (unpermuted+precision R̂ 10.37–11.71): every GPU ESS/s and wall in the basis is on the failing orientation; §2c: corrected arm gives logistic L 136→26, 8 Schools L 43→22 (orientation verdict per `a03dc0f` retracting `699be6d`) | Re-run equal-draw cells on the corrected arm (select via the `imm_orientation` knob from `4460dc7`); missing number for any default flip (G3): per-model GPU wall + ESS/s + divs/1k + mean shift, arm permuted+covariance, ≥ 5 seeds, R-run |
| 3 | "Warmup 2,000 is enough; 10,000 was not" + table | README.md:82–91 | 5-seed warmup sweep on 8 Schools + GEV, shipped arm | CONFOUNDED | 2×2 fixture cell 4: both models' quality columns (R̂, tail-ESS) measured on the failing orientation; GEV additionally permuted (verdict per `a03dc0f`) | Re-measure the sweep on the corrected arm; missing number: per-model R̂/tail-ESS at W ∈ {1000, 2000, 10000}, arm permuted+covariance, 5 seeds |
| 4 | "Do not go as low as 1,000" (GEV L collapse 6.2, R̂ 1.068, tail-ESS 583) | README.md:93 | GEV W=1,000 cell, shipped (failing, permuted) metric | CONFOUNDED | GEV permutation (`order_check.py`) + fixture cell 4; plan §0.3: the T-growth rate limit makes the frozen-L signature adapter-dependent (verdict per `a03dc0f`) | Retract the "do not go below 1,000" guidance; re-derive the floor via the WP3 bias gate (b²_max < 0.01), not split-R̂; missing number for any warmup-default change (G3): smallest W passing G-R1 on GEV, corrected arm, ≥ 5 seeds |
| 5 | "More chains also means less needed warmup" | README.md:93 (final sentence) | ChEES pooling argument; ladder warmup-by-chains grid | SUPERSEDED | Plan A2 lists it a superseded belief; the grid it leans on is itself void (short-chain split-R̂ artifact, plan §2b; failing metric) | Delete the sentence; replace with the bias-based W finding when the R-run/WP3 sweep lands |
| 6 | "run over a lot of short chains… potential to sample much faster" | README.md:14 | no row id; aspirational paragraph | CANNOT TELL | No measurement basis found in repo or row files | Replace with a bias-based statement after WP3 self-test (plan §2e: 8 Schools C=1024 D=10 W=100 b²_max 0.0008) — verify through the public-API harness first |
| 7 | "48.3 s cold / 39.8 s cached / 22 s warm; cache recovered 8.5 s of 26 s" | README.md:76 | 8 Schools fresh-process A/B; cache A/B (`db551ad`) | UNAFFECTED | Compile cost is metric-independent; 4-cell fresh-process cache A/B re-verified recovery (6.2–6.7 s at a smaller config) | Keep the claim; refresh the three absolute walls after the R-run |
| 8 | "progressbar=False does not change compile cost; 6–7 s, ~640 compiles either way" | README.md:80 | 4-cell fresh-process A/B | UNAFFECTED | `db551ad`: compile counts 642–652 with bars on/off; metric-independent | Keep |
| 9 | "ChEES adapts ε and L only — it does not adapt the mass matrix" | README.md:9 | source reading of `chees_adaptation.base` | UNAFFECTED | True of the private `base()` path the package calls; WP8 `f62a22f` (reviews pending) qualifies the wording around BlackJAX's unused public diagonal option | Apply WP8's wording on merge |
| 10 | "45,614 ESS/s at 32 chains and 134,898 at 128; per-transition flat 32→256 (+2%)" | README.md:72 | pre-`a9d1a3e` chain sweep | CONFOUNDED | Ladder caveats (lines 58–64) already supersede the quoted numbers (41,568 / 133,824 post-RNG-fix); those successors sit on the failing arm; "flat to 256" mixes held-fixed quantities (plan A2: per-step number includes O(C) init/host costs) | Rewrite with post-R-run numbers and state what is held fixed (draws/chain vs total draws) |
| 11 | E1 grid: per-model ESS/s table (0.99x … 0.59x) | docs/ladder-results.md:15–21 | 115-cell E1 sweep, `~/pf-runs/ladder-*.jsonl`, shipped arm | CONFOUNDED | 2×2 fixture cell 4: all cells on the failing orientation; GEV + both radon rows additionally permuted | Full E1 re-measure in WP9 on the corrected arm (missing number: the whole table, permuted+covariance, warm, 4 seeds median) |
| 12 | "GPU advantage tracks steps-per-draw; 128 ch = 3.2x/2.8x/1.65x of 32 ch" | docs/ladder-results.md:29–35 | E1 grid | CONFOUNDED | Same basis as row 11; throughput ratios move when the metric changes (ε/L re-price per §2c) | Re-derive ratios from the row-11 re-measure; missing number: per-model ESS/s at 32 vs 128 chains, corrected arm |
| 13 | "Our sampler fails on the centered funnel (R̂ 4.33, L frozen ~4.8)" | docs/ladder-results.md:37–43 | radon centered 32 ch, shipped (failing, permuted) metric | UNAFFECTED | Funnels are out of scope and radon centered is never a gate (plan §8); the magnitude is metric-contaminated, so the row only supports "out of scope" | Re-qualify as an out-of-scope observation; drop "failure" framing; do not re-run |
| 14 | "Four chains is not a usable GPU configuration… unconverged output" | docs/ladder-results.md:45–50 | 4-chain sweep, shipped arm (retroactive justification of the default change) | CONFOUNDED | 2×2 fixture cell 4 (failing orientation kills every wide coordinate → inflates 4-chain R̂ on GEV/radon; speed half (6–25x slower) is metric-independent throughput (verdict per `a03dc0f`) | Keep the speed half; re-measure 4-chain R̂ on the corrected arm before restating the quality half; missing number: 4-chain max R̂, corrected arm, GEV + radon NC |
| 15 | "Gate G1 is satisfied by measurement (cond 358 / 2,207 / 8,837)" | docs/ladder-results.md:52–56 | posterior draw covariance | UNAFFECTED | Condition number of the posterior is metric-independent of the WP1 defects | Keep |
| 16 | E4 table: mc / lbfgs / fisher ESS/s and step counts (incl. "37 → 514") | docs/ladder-results.md:85–89 | E4 A/B, `~/pf-runs/e4-*.jsonl` | CONFOUNDED | LEDGER `9f892b2` convention mismatch between arms; baseline on fixture-failing cell; plan §2d: PF metric 100× under-dispersed on radon (separate, WP4's) | Void the table; plan designates WP4 (ensemble metric) as the successor; missing number: per-arm ESS/s, all arms convention-aligned, R-run |
| 17 | "Fisher's units are 2,000x below the posterior's own scale" | docs/ladder-results.md:99–101 | eigen-spectra of the three arm arrays compared | SUPERSEDED | LEDGER `9f892b2`: the mc array is precision while the low-rank arrays are ≈covariance — "2,000×" is a convention artifact (plan WP9: "it is precision against covariance"); 2×2 fixture settles which array is canonical | Retract as stated; if re-quoted, compare same-convention arrays only |
| 18 | "Rescaling made every model worse (50,626 → 13,628 …)" | docs/ladder-results.md:103–107 | rescale A/B under the failing baseline | CONFOUNDED | Plan WP9 designates retraction; both arms of that A/B sat on the failing orientation, so the "joint adaptation" reading is unverified | Re-run the rescale experiment after WP1; missing number: rescaled-vs-not ESS/s, corrected arm, same 4 models |
| 19 | "E4 ships as an option; worth 1.2–2.0x, loses 5x; tell = step jump" | docs/ladder-results.md:113–117 | rows 16–18 | CONFOUNDED | Same as row 16; the "tens → hundreds" tell was observed under the failing orientation (logistic L 136 is the fixture-failure signature, §2c) | Keep the option; delete the ratios and the tell; re-derive guidance from WP1+WP4 numbers |
| 20 | "At equal sample counts the CPU wins: 15 of 16 cells" | docs/ladder-results.md:162–165 | equal-draw sweep, GPU arm on shipped metric | CONFOUNDED | 2×2 fixture cell 4: every GPU wall/ESS in the basis is on the failing orientation; §2c shows the corrected arm collapses logistic L 136→26 (walls move; verdict per `a03dc0f` retracting `699be6d`) | Re-run the sweep on the corrected arm; missing number for any default flip (G3): GPU wall + ESS1 + divs/1k per cell, permuted+covariance, both models, ≥ 5 seeds |
| 21 | "The gap is fixed cost, not sampling speed" | docs/ladder-results.md:167–170 | wall decomposition of the same sweep | UNAFFECTED | Attribution is structural (2,000 sequential adaptation steps + per-C re-trace), independent of the metric defects | Keep the attribution; re-price the magnitudes with row 20 |
| 22 | "Short chains cost quality… GPU R-hat rises monotonically with chains" | docs/ladder-results.md:171–173 | 4,000-draw equal-draw cells, split-R̂ | SUPERSEDED | Plan §2b chop test: the same signature (R̂ rising as draws/chain falls) is reproduced by chopping one converged run; iid control reads 1.0021 at 1024×39; plan §2e: 1024×10 reaches b²_max < 0.01 | Retract; re-qualify any short-chain quality statement to the bias harness + nested R̂ |
| 23 | "Warmup 1,000 is a real lever only on long chains (radon R̂ 2.16 at 4 ch)" | docs/ladder-results.md:175–178 | radon equal-draw cells, W=1,000, shipped metric | CONFOUNDED | radon on the failing orientation (fixture cell 4) and permutation; 4-chain R̂ additionally artifact-prone | Re-measure W=1,000 vs 2,000 on radon with the corrected arm; missing number: radon NC max R̂ + wall at W ∈ {1000, 2000}, corrected arm |
| 24 | "CPU divergences are the counterweight (GPU 0–0.05 vs CPU 1.3–2.3 /1k)" | docs/ladder-results.md:179–181 | equal-draw sweep sample_stats | CONFOUNDED | GPU divergence counts came from the failing arm (ε/L set by it); the CPU half is fine | Re-verify GPU divs/1k after WP1; missing number: divs/1k per cell, corrected arm |
| 25 | per-gradient table: 23.0 / 116.3 / 138.7 µs + floor controls | docs/ladder-results.md:205–210 | DCE-fixed gradient probe (`~/pf-runs/baseline.jsonl`) | UNAFFECTED | Model-evaluation cost; metric-independent; floor controls included | Keep |
| 26 | "Chains are effectively free on small models (23–25 µs at 1/32/128)" | docs/ladder-results.md:212–213 | same probe | UNAFFECTED | Gradient-cost fact; the ESS/s corollary in the same sentence is confounded (row 11) | Keep the cost fact; delete or re-derive the ESS/s corollary |
| 27 | "Float32 buys nothing on the graphs we ship; keep it parked" | docs/ladder-results.md:215–218 | real-graph f64 vs f32 at 32 chains | UNAFFECTED | 23.0 vs 22.5 (8 Schools), 138.7 vs 141.4 (radon) — metric-independent; plan §8 concurs (parked) | Keep |
| 28 | "83% of a step is overhead on 8 Schools; 55% on radon" | docs/ladder-results.md:221–222 | production wall vs per-gradient floor, shipped metric | CONFOUNDED | Production walls and the L values they contain were realized under the failing arm (fixture cell 4) | Re-profile after WP1 (WP6.3 owns the split); missing number: per-step ms at corrected ε/L, both models |
| 29 | stage timers: "warm total 11.86 s / 43.03 s; warmup 35–48%" | docs/ladder-results.md:232–246 | `~/pf-o1-stages.py` runs, shipped metric | CONFOUNDED | 2×2 fixture cell 4: every stage wall is on the failing arm | Re-price all stage walls in the R-run; keep the stage list |
| 30 | finite-start repair: 1.75→0.24 s / 4.24→0.39 s; "radon gains 3.4 s"; 5-seed "1.23 s faster (10%)" | docs/ladder-results.md:305–312, 460–470 | before/after A/B, `cf18539` (5 seeds) | UNAFFECTED | Both arms of each A/B used the same (failing) metric, so the deltas are valid comparisons; 5-seed confirmation flat on R̂/ESS | Keep; note absolute totals re-priced by the R-run |
| 31 | init-policy dispersion: "stratified beats iid where free; over-dispersion buys quality with compute" (radon) | docs/ladder-results.md:322–365 (lead-corrected; the task cited 449-461, which is the E5 status paragraph) | `tools/probe_init_policy.py` on radon NC, shipped (failing, permuted) metric | CONFOUNDED | Radon permutation + fixture cell 4; this is the evidence base of the SHIPPED stratified-pick default (`9fdd15d`) | Dedicated re-run with the corrected metric at short warmup — the WP1 R-run covers imm arms only, NOT pick policy; missing number: per-policy max R̂ + min ESS at W ∈ {250, 500, 1000}, corrected arm |
| 32 | stop rule: "costs more than it saves (+6.7–9.0 s); not a default, not worth the rewrite" | docs/ladder-results.md:365–403 | fresh-process A/B base vs rule, same metric in both arms | UNAFFECTED | A/B: the metric cancels; the 7–9 s block-boundary overhead is a property of the blocked-scan design, not the metric; plan §8/WP9 park it anyway | Keep the verdict and the parking; re-open only per the plan's condition (min W varies >4× across models) |
| 33 | MEADS+GHMC: "not shippable; 4.8x fewer gradients/ESS; radon degenerates (ε 0.0/NaN)" | docs/ladder-results.md:423–429 | MEADS pilot, same starts in all arms, `~/pf-runs/meads-pilot.jsonl` | CONFOUNDED | 8 Schools half is a clean A/B (same metric both arms) and interpretable; radon rows ran on the failing, permuted metric; plan: re-test at equal gradients after WP1/WP4 | Keep the 8 Schools reading; re-run radon after WP1/WP4; missing number: radon NC MEADS ε/L/R̂ on the corrected metric |
| 34 | "Short chains do not substitute for long ones (R̂ 1.0013 → 1.0404, ESS/s 2,185 → 940)" | docs/ladder-results.md:509–510 | fixed-40,000-draw chain sweep, `~/pf-runs/chain-scaling*.jsonl` | SUPERSEDED | Plan §2b chop test reproduces every "degraded" R̂ by chopping one converged run (docs' 1.040 vs chop 1.050 etc.); plan §2e: 1024×10 passes b² < 0.01 at W=100 on 8 Schools | Retract the quality sentence; the wall/cost half is row 35 |
| 35 | "c/step rises 3.17 → 13.69 ms from 32 → 1,024 chains" | docs/ladder-results.md:553–555 (table 485–491) | same sweep; wall/(W + D/C) | CONFOUNDED | Plan A2: the number includes O(C) init and host costs; walls additionally on the failing arm | Replace with WP5's device-cost-per-step measurement (T4); missing number: per-step device cost vs C, corrected arm |
| 36 | "More warmup does not repair what more chains broke; a start needs draws-per-chain to burn off" | docs/ladder-results.md:521–522 | warmup×chains grid (10,000 draws), `~/pf-runs/warmup-by-chains.jsonl` | SUPERSEDED | Plan A2 lists both beliefs superseded; the grid's R̂ values are the short-chain artifact (plan §2b) measured on the failing metric | Retract both sentences |
| 37 | radon grid: "R̂ 2.2397 (W=100) → 1.0381 (W=2000); still above 1.01 at 10,000 draws" | docs/ladder-results.md:524–526 | same grid, radon NC, shipped (failing, permuted) metric | CONFOUNDED | radon on fixture-failing orientation + permutation | Re-measure the W-ladder on radon with the corrected arm; missing number: radon NC max R̂ at W ∈ {100…2000}, corrected arm |
| 38 | "why there is no smaller [warmup] default" (logistic W=500: R̂ 1.0433, 6x ESS loss, L 15.4) | docs/ladder-results.md:528–541 | 5-seed W=500 vs 2,000 A/B, shipped arm | SUPERSEDED | Plan A2: "2,000 steps are needed" was measured under the defective metric; logistic L 15.4 is the failing-orientation signature (§2c: corrected L ≈ 26) compounded by the §0.3 adapter rate limit (verdict per `a03dc0f` retracting `699be6d`); the 8 Schools A/B delta itself is valid (same metric both arms) | Retract the conclusion, keep the 8 Schools delta; re-run the W-ladder on the corrected arm; missing number: smallest W passing G-R1 on logistic, corrected arm |
| 39 | "radon's ceiling is 64 chains; memory source NOT ESTABLISHED" | docs/ladder-results.md:549–551 | OOM sweep history | SUPERSEDED | Plan §0.6: init-stage memory confirmed — PF pool OOM (797 MiB) then stratified scoring OOM (500 MiB) at C=1024, d=25; WP5 fixes both; radon C=128 re-test still open in the plan | Re-qualify: state the confirmed pool/scoring mechanism; re-test radon C=128 in WP5 |
| 40 | "LAPS: wrong regime" (D7) | docs/gpu-upgrade-plan.md:262 | D7 evaluation, pre-run | SUPERSEDED | Plan A2: "LAPS is the wrong regime" is superseded — the owner's regime is now R1, which is LAPS's stated regime | Re-open the D7 verdict in the lit plan (WP10+); do not cite D7.1 |

Dropped beyond the 40-row cap: 4 rows, all subordinate detail inside a section that already has a
surviving row with the same verdict —
- "cutoff 2 → 10 helped radon noncentered modestly" (docs/ladder-results.md:110–111, CONFOUNDED;
  covered by rows 16/18, same fisher-arm convention defect);
- "kernel-only scan 443 µs; +full HMCInfo 491 µs; carry is not the overhead"
  (docs/ladder-results.md:255–257, UNAFFECTED; O1 arm A/B, same thread as row 30);
- roofline ceilings "f64 195.6 GFLOP/s, f32 14,771 (75x); ~130–155 µs floor"
  (docs/ladder-results.md:270–285, UNAFFECTED; hardware, same thread as rows 25–27);
- "float32 is the largest lever measured here (3.3x at radon shape)"
  (docs/ladder-results.md:296–300, SUPERSEDED; refuted in-file by the real-graph table at
  215–218, same thread as row 27).

## The 5 most dangerous rows

1. **README.md:93 — "Do not go as low as 1,000" warmup guidance (row 4).** A reader tuning
   warmup down hits a documented wall that exists only because of the shipped metric defect; a
   reader trusting "2,000 is enough" pays 2–3× the warmup wall the corrected arm needs (plan §2e:
   8 Schools passes at W=100) — they ship the slowest configuration thinking it is the safe one.
2. **README.md:78 / docs/ladder-results.md:113–117 — the Fisher 1.22×/1.96×/0.19× and the
   "tens → hundreds of steps" tell (rows 1, 19).** A reader switching to `fisher_lowrank` on this
   evidence adopts a metric whose comparison changed estimator AND orientation at once (LEDGER
   `9f892b2`) with a baseline on the failing orientation — the 0.19× on radon looks like a warning
   label that may itself be void.
3. **docs/ladder-results.md:528–541 — "no smaller warmup default" (row 38).** This IS the shipped
   default (2,000 steps), and the default orientation is still the failing one as of `4460dc7`.
   The stated basis (logistic L 15.4 "where the geometry needs ~60") is a product of the failing
   orientation plus the §0.3 adapter rate limit (retraction: `a03dc0f`); acting on the claim
   freezes the project's biggest fixed cost at the wrong value.
4. **README.md:74 — "GPU 3.6x/2.6x, CPU finishes sooner 9.5 s vs 15.0 s" (row 2).** A reader
   picking a device for a run uses failing-arm GPU numbers on both sides of the ratio; the
   direction of the recommendation may flip per model once the corrected arm lands (§2c: logistic
   L 136 → 26).
5. **docs/ladder-results.md:322–365 (lead-corrected; the task cited 449-461, which is the E5 status paragraph) — the init-policy evidence behind the shipped stratified
   default `9fdd15d` (row 31).** A reader extending or trusting the default — especially in the
   short-W regime R1 actually uses — relies on radon cells measured with a permuted,
   wrong-oriented metric; the stratified-vs-iid gap may shrink or vanish, and R1's whole warmup
   budget depends on the short-W half of that table.

## Claims I could not classify, and what would settle each

1. **README.md:14 "lot of short chains … potential to sample much faster"** (row 6). No row id
   backs it; it is a promise, not a measurement. Settled by: the WP3 self-test's 8 Schools
   C=1024 D=10 cells through the public API (plan §2e: b²_max 0.0008 at W=100, 1,200
   gradients/chain) — until those land on the corrected arm, the sentence should be deleted, not
   reworded.
2. **"R-hat unchanged on the healthy models"** inside README.md:78 (folded into row 1). Whether
   the healthy-model R̂s (8 Schools, logistic) were actually flat across imm_source arms depends on
   which arms were run per model — the E4 table has "—" cells, i.e. missing arms. Settled by:
   reading arm coverage in `~/pf-runs/e4-*.jsonl` per model; moot after the row-1 re-measure,
   which must include R̂ per arm.
3. **The 4-chain quality half (row 14)** — how much of the historical 4-chain R̂ degradation
   (1.13–3.59) was the metric and how much was genuinely "too few chains for ChEES" cannot be
   separated from the existing rows. Settled by: one 4-chain cell per model on the corrected arm
   (add to the R-run if a slot allows; 32 chains is the R2 baseline, 4 chains is the open cell).
