STATIC REVIEW (no shell in this task): arithmetic below is the lead's, not independently executed.

**Reviewed:** `wp/wp1` @ `1cb74bf` (`/home/akbar/pf/wp1`), merged tree `/home/akbar/pf/int` @ `574f960`, `docs/review-2026-10/pf-chees-hmc-implementation-plan.md`, `results/WP1/SUMMARY.md`, `results/LEDGER.md`, `tools/r1_bias.py`, row files `WP1__rebaseline.jsonl` / `WP1__covarm.jsonl` / `T1__r1_sweep.jsonl` / `G3__gev_meanshift.jsonl`, log `/tmp/gev_g3.log`, driver `~/scratch/t1_sweep.sh`.

**Method and its limits.** `bash` returned `Tool bash not found` on three attempts, so nothing was executed: no pytest, no median recomputation, no fixture. What I *did* do is read files and count pattern matches. Items marked **[read directly]** are values or counts I extracted from a file in this session; items marked **[lead's]** are arithmetic I did not perform and am reproducing under attribution. This is not the second independent review A4 requires, and it cannot certify that any test passes or fails anywhere.

---

## 1. Are the acceptance criteria answered as written?

### 1.1 The plan's clause is about the combination, and the branch scored a different question

Plan `:532-564` — WP1's **Change** is `inverse_mass_matrix = cov` **and** the permutation (reinforced by §1 `:354`: "Fix = `inverse_mass_matrix = cov` → **INSUFFICIENT**; also fix the ordering"). The **Accept** clause (`:562-564`) is: tests pass; gradients/ESS not worse on ≥ 3 of 4 models; divergence rate does not rise at equal C.

`results/WP1/SUMMARY.md` §5 files the criterion as **"FAIL — 1 of 4"**, measured on `wp1:mc` = permuted + precision. That arm is not the change the plan asked for; it is a decomposition the branch invented (and inventing it was good science — it is what proves the two factors are not independently beneficial). But as written, the plan's criterion is evaluated on the arm the plan specified, `wp1:mc:covariance`, where SUMMARY §3c's own table reports gradients/ESS **better on 4 of 4** (35.45 / 71.21 / 46.08 / 182.3 vs 58.23 / 73.44 / 151.7 / 501.9) **[lead's]**.

Required wording change: in §5, split the row into **(i) plan criterion, arm as specified → PASS on gradients/ESS, FAIL on the divergence clause (GEV 0.225 → 0.475 per 1k)** and **(ii) decomposition: permutation without the orientation fix → FAIL 1 of 4, and breaks R̂ on the two models where the bug lives**. Today a reader taking §5 at face value concludes WP1 failed; the plan's own test says it passed one clause and failed the other.

### 1.2 Statistic, arm, and row-selection are otherwise correct

- Statistic: `grads_per_ess_worst` = gradients per worst-variable bulk ESS, matching "gradients per worst-variable ESS". Lower is better, and the tables state it.
- Rows: §3c states 5 seeds, REPEAT=2, warm medians decide, cell id `(arm, model, seed, rep)` — matches A5/A8. I could not recompute the medians (no shell), so "warm rows only" is taken on the task's word **[lead's]**.
- Duplicate hygiene is designed correctly in source: `write_cell` raises on a live `(wp, arm, model, seed, rep)` key, `OVERWRITE=1` must be explicit, VOID rows do not occupy a key, and the summariser prints an audit line before every table (SUMMARY §7) **[read directly: SUMMARY §7; not executed]**. 142 rows, 2 VOID, 140 live, 0 duplicate keys **[lead's]**.
- `imm_order="pathfinder"` (the knob that reproduces the bug) is guarded: `sampler.py:568-572` defaults to `kernel` and raises `ValueError` on anything not in `{kernel, pathfinder}` **[read directly]**. Good — a typo cannot silently select the unpermuted metric.

### 1.3 Claims the row files do not support

1. **"The corrected arm passes the bias gate at 50 warmup steps on every model measured so far"** (LEDGER 23:45, repeated in the run's status report) is contradicted by the radon rows **[read directly]**:
   - `int:covraw` radon NC, W=50, seed 1: **b² = 3.0003**, floor 0.00485 → fail by ~300×.
   - W=200, seed 1: **b² = 0.0234**, floor 0.00477 → fail. Seed 2: **b² = 0.00709**, floor 0.00413 → pass.
   Plan §6 `:816` requires b² < 0.01 **on every seed**, and `:634` defines T1's output as "the smallest W with b² < 0.01 on every seed". Radon therefore passes at **no** W measured. The correct statement is: *radon NC does not pass G-R1 at C=1,024 D=10 at W ≤ 200 in either arm; the covariance arm improves b² at W=200 from 12.1 (1-of-2 seeds, [lead's]) to 0.007–0.023 (2 seeds, split verdict).*
2. **"GEV covariance passes 4/5"** on R̂ is **3/5** **[read directly from `/tmp/gev_g3.log`]**: covariance seeds 43–47 read R̂ 1.0065, 1.0017, 1.0054, **1.0141**, **1.0117**. Precision reads 1.0465, 1.0471, 1.0294, 1.0688, 1.0380 → fails 5/5.
3. **G-R1 / G3 "means within 1 sigma" is currently unmet by both arms**, depending on which sigma is meant. The probe prints both: shift in posterior-sd units (precision 0.033–0.059, covariance 0.015–0.028) and shift in units of the run's own MCSE (`z`: precision 1.21–2.70, **covariance 1.77–3.35**) **[read directly]**. Read as "within 1 posterior sd", both arms pass with room; read as "within 1 MCSE", neither arm passes in any cell — and the covariance arm is *worse* on that yardstick despite being better in sd units. The report must name the yardstick; "means within 1 sigma" as written in the plan is ambiguous, and this task cannot resolve it.
4. **`pf_grads_total` is null in every row** (SUMMARY §7 flags it), and the R1 stage split shows why that matters **[read directly]**: radon covraw W=200 seed 1 spends `pathfinder 37.0 s` of a `wall 58.8 s`; radon W=50 seed 1 `pathfinder 60.9 s` of `103.9 s`; logistic covraw W=50 seed 1 `pathfinder 46.2 s` of `72.3 s`. "Gradients per chain" therefore excludes the stage that is 60–78% of wall in R1 on the big models. It is the same in both arms so arm *ordering* is unaffected, but T1's cost column must not be read as cost-to-result — that is WP5's territory and it belongs in the REPORT's caveats.
5. **The plan's WP1 smoke test is absent.** §549-550 asks for "frozen ε ∈ [0.3, 2.0] and nominal L ≤ 10, 200-step warmup, 32 chains". `tests/test_metric_orientation.py` has no such assertion (grep for `0.3`, `2.0`, `smoke` returns only the fixture's `sigma = [2.0, 3.0, 5.0, 7.0]` and the two failure asserts) **[read directly]**. Either add it or record the omission in §7 next to the `pf_grads_total` gap.
6. **Record integrity:** `results/LEDGER.md` contains **two** entries retracting `699be6d` — 18:05 and 23:15 **[read directly]**. They disagree about the mechanism: the earlier says "Mechanism: NOT ESTABLISHED … I am not asserting it", the later asserts the `ω² = eig(A H)` explanation. The later one is the run's position; add a supersession pointer at the top of the 18:05 entry so a reader who lands there first does not treat the mechanism as open.
7. **Evidence location:** the T1 and G3 row files live in `/home/akbar/pf/int/results/runs/` — a scratch merge tree, not a branch that will ever be merged **[read directly: directory listing]**. If that checkout is removed the primary evidence for the run's headline table disappears with it. Commit them to `pf-r1` (or copy into `results/runs/` in the main checkout) before anything else.

---

## 2. Critique of the lead's interpretation

### (a) Floor semantics — you have it stricter than the code, not backwards

Plan §6 `:816-817`: "b²_max < 0.01 … **on every seed** … with the noise floor printed. **A cell whose floor exceeds 0.005 is "not resolvable", not a pass.**" `tools/r1_bias.py:273-280` implements exactly that **[read directly]**:

```python
if c_times_d < 2000: return "not eligible (C*D < 2000)"
if floor > 0.005:    return f"not resolvable (floor {floor:.4f} > 0.005)"
return "pass" if b2_max < 0.01 else "fail"
```

Resolvability is keyed on the **floor's magnitude**, not on b² versus floor. Every T1 floor I read is ≤ 0.0049, so under the plan's rule all 48 cells are resolvable and "b² below the floor = unresolvable" is your own tightening, not the harness's and not the plan's. It is a defensible tightening — b² is a squared quantity and a value below the noise term is genuinely indistinguishable from zero — but it is a **third definition**, so either encode it (`g_r1` gains a `b2 < floor → indeterminate` branch, plus a `gate` field in the row) or stop using the phrase in REPORT and the LEDGER. Note the asymmetry it creates: with the strict reading, 8 Schools covraw W=100/200 and GEV W=100 stop being passes, which is why the phrasing matters.

Is the floor optimistic? Two ways, both small but real:
- It uses run MCSE only (`mcse_mean`, `mcse_mean_sq`); the reference's own MCSE never enters, although the reference is 8×10,000 draws, so the omission is maybe 10–20% of the variance term **[reasoned from the formula, not computed]**.
- At D=10 the MCSE itself is computed from an ESS estimated from 10 draws per chain, which is where ESS estimates are least trustworthy. A floor quoted to four decimals with no uncertainty band, from 2 seeds, is being read as if it were exact. Suggest: three seeds minimum for any cell whose b² is within a factor 2 of its floor, and treat that band as indeterminate under either definition.
- The `max_f` over leaves and the `2·ln(2d)` union factor make it conservative on multiplicity. That part is right.

### (b) Gradient counts — now exact, and the fix changed no number in R1

The scalars are the mean of exact per-chain sums and the lists are carried (`grads_per_chain_*_list`) **[read directly]**. Two observations you should put in the record:

1. In every R1 row I opened, all 1,024 entries of the list are **identical** (259×1024, 280×1024, 899×1024, 1448×1024, 220×1024). ChEES adapts one ensemble-wide step size and trajectory length, and the Halton jitter is per step rather than per chain, so chain counts coincide by construction. The `np.median` defect the WP3 review called BLOCKING therefore had **zero numerical effect on these rows** — it was latent, not realised. Say so, otherwise the correction reads as if the T1 table changed when it did not.
2. "Gradients" is defined as `sum(num_integration_steps)`, which is the plan's own definition (`:616`), so the column is internally consistent. Whether BlackJAX also evaluates the gradient at a trajectory's start (one extra per transition) is **NOT ESTABLISHED** in this review, and at the R1 cov-arm step counts (`steps_mean` 4.0–7.3 **[read directly]**) one extra gradient per transition is 14–25% — the same order as some of the arm differences being compared. A shell-enabled reviewer should count `ld_fn` calls in one cell with a wrapper; if it is `steps + 1`, add the constant in the row rather than in prose.

### (c) Split R̂ at D=10 — you did not overstate it

I grepped all 48 T1 rows for any `max_rhat` below 1.01: **zero matches** **[read directly]**. Every R1 cell exceeds the project's bar, including cells whose b² is below the floor (radon covraw 4.068 / 2.736 / 2.051; logistic covraw 2.005 / 1.983; radon shipped 6.6–8.9 **[read directly / lead's table]**). So "split R̂ cannot certify R1" is supported by the whole file, not a handful of cells.

Keep the direction of the caveat precise, though: this shows split R̂ has no power at D=10 (it fires on converged runs), i.e. it is a **false-positive machine** there. It does **not** show R̂ is worthless as a flag — the fixture's R̂ 12.1 is exactly what caught the orientation defect. Write it as: *at D=10 split R̂ may indicate catastrophe, it can never certify convergence, and G-R1 is the certifier.*

### (d) Is "ship the knob, hold the default, diagnose GEV" supported?

**Argument for the hedge.** Orientation is a default-semantics change; the plan makes defaults owner-gated (A10) and G3 demands ≥ 5 seeds, no divergence increase, and means within 1σ. GEV's divergence rise is measured and unexplained; GEV also prefers the *wrongly ordered* covariance metric (1,310 vs 343 ESS/s **[lead's]**), which means the model that could falsify the fix is the model nobody understands. Radon does not pass G-R1 in any arm yet. On that reading, holding is disciplined.

**Argument against — and it is the finding.** "Hold the default" is **not** status quo. The permutation is unconditional (`sampler.py:568-572`, default `kernel`; SUMMARY §1 "shipped, unconditional" **[read directly]**), so merging the branch with the orientation held changes the released sampler from *unpermuted + precision* to *permuted + precision*. Measured on GEV at 32×1,250/W=2,000, that arm fails R̂ ≤ 1.01 in **5 of 5 seeds** (1.0294–1.0688 **[read directly]**, and 1.0465 median in SUMMARY §3c), and radon at 1.0142, while the released code passes both (1.0023, 1.0069 **[lead's, SUMMARY §3c]**). The merge-as-proposed ships a known regression on exactly the two models where the bug bites, and it will be described as "default unchanged".

**Single weakest link in the chain: the false baseline.** Everything downstream — "diagnose GEV's divergences before flipping", "GEV is the counter-example" — assumes the interim state is the state being protected. It is not. Three coherent options, all better than the hedge: (1) merge the permutation **and** the covariance default together, accepting the GEV divergence rise and reporting it; (2) gate the permutation on the orientation, so with `precision` the array stays in tuple order and today's behaviour is bit-identical, which makes "hold the default" literally true; (3) merge as-is and state in README/REPORT that the released default fails R̂ on GEV and radon until the owner flips it. Option (2) is the honest way to buy time; option (3) is the one that must not be done silently.

**Second weakest: T1 cannot attribute its own result.** `~/scratch/t1_sweep.sh` **[read directly]** runs `shipped = precision + ma` against `covraw = covariance + raw`. Two factors move at once, so "the metric fix gives the R1 gain" is not separated from "the raw iterate gives it" — the same confound WP1 correctly refused to accept in R2. It also never passes `--legacy`, so today's released arm (unpermuted + precision) has **no R1 measurement at all**; what the report calls "the shipped arm" is the merged tree's new default. Missing cells that decide this: `covariance + ma` and `precision + raw` (isolate orientation), plus one `legacy` cell per model (the actual baseline), plus C=512 and W=400/800 to know whether radon fails from warmup length or from capacity.

**Third: seed count and cold-only rows.** `--repeat 1` with `--seeds 1,2` yields **48 rows, all `"cold": true`, zero warm rows** **[read directly: pattern counts]**, and 2 seeds where the plan asks 3 for exploration and 5 for decisions (`:632-633`). Quality columns (b², R̂) are unaffected by cold/warm, so the pass/fail analysis stands; every **wall** number taken from T1 violates A5's "decide on warm" and should be labelled cold in REPORT's T1 table.

---

## BLOCKING (static scope)

1. **The default-hold plan is a measured regression, not a neutral state.** Fix by option (1), (2) or (3) above; if (3), the regression must appear in `results/REPORT.md` §"Decisions" and in README, not only in this review.
2. **T1's radon claim is wrong in the record.** LEDGER 23:45 and the run's status messages state the corrected arm passes at W=50 across measured models; rows show radon b² 3.0003 (W=50) and 0.0234 (W=200, seed 1). Correct the LEDGER with a dated amendment and set T1's radon cell to "no passing W ≤ 200; seeds split at W=200 (0.0234 / 0.0071)".
3. **T1 arms are confounded (orientation × iterate) and lack the released-code baseline.** Add `covariance+ma`, `precision+raw`, and `legacy` before the T1 table is called a result.
4. **The primary evidence is in a scratch checkout** (`~/pf/int/results/runs/`). Commit `T1__r1_sweep.jsonl`, `G3__gev_meanshift.jsonl` and the probe tool to `pf-r1` so the report's tables have artifacts behind them.

## NON-BLOCKING

5. Split §5's acceptance row into "plan criterion" vs "decomposition check" (§1.1).
6. Resolve "means within 1 sigma": name the yardstick (posterior sd or MCSE) in the REPORT and the plan annotation; both arms fail the MCSE reading today (§1.3.3).
7. Encode the stricter floor reading in `g_r1` and in a row field, or drop the phrase (§2a).
8. State that the median→mean gradient fix changed no R1 number (lists are constant) (§2b.1).
9. Add the plan's WP1 smoke test (ε ∈ [0.3, 2.0], L ≤ 10) or record its omission (§1.3.5).
10. Supersession pointer between the two LEDGER retraction entries (§1.3.6).
11. Label T1 walls as cold, and note Pathfinder is 60–78% of R1 wall and excluded from the gradient objective (§1.3.4, §2d-third).
12. Correct "GEV covariance passes 4/5" to 3/5 (§1.3.2).

## NIT

13. `b2_floor` and `b2_max` are printed side by side but the row has no `gate` field; verdicts exist only in `/tmp/t1_sweep.log` and `/tmp/gev_g3.log`, which do not survive a reboot. Persist them.
14. `G3__gev_meanshift.jsonl` scores constrained names (`σ`, `ξ`) while G-R1 scores unconstrained — defensible for a mean-shift check, but say it in the probe docstring so the two b²/shift numbers are not compared across spaces.
15. E-BFMI 1.053 on `wp1:fisher_lowrank` GEV (SUMMARY §3c) is outside the usual range of the statistic; not decision-relevant for this branch, worth one look at the E-BFMI formula.

---

**RECOMMEND: hold** — the branch's code and tests look sound and the ordering fix is unambiguously right, but as framed the merge ships a default that fails R̂ on GEV in 5/5 seeds while being described as unchanged, and the run's headline T1 claim is contradicted by its own radon rows.

**Flip the default now? Yes, with the permutation, in the same merge — the minimum missing evidence is not a GEV diagnosis but a decision that the interim state matters more than the released one:** the covariance arm passes R̂ on 3/5 GEV seeds and 5/5 elsewhere, beats the alternatives on gradients/ESS 4/4, and the alternative to flipping is a default that fails 5/5 on GEV. Missing before this is *certified* rather than *argued*: a legacy (unpermuted+precision) R1 arm, the two single-factor arms to attribute the R1 gain, ≥ 3 seeds on any cell within 2× of its floor, and the owner's word on G3's "1 sigma" yardstick.

**A shell-enabled reviewer must still check:** the 17 tests actually pass on `wp/wp1` and fail 12-of-5 on `ecd9ba1` for the stated reasons; the warm-row medians in §3c reproduce from the JSONL; `test_permuting_factors_equals_permuting_the_expansion` really covers non-square `U` and repeated eigenvalues; the radon covraw failure is reproducible at a third seed; `summarise_rows.py`'s VOID exclusion works; and whether one extra `ld_fn` evaluation per transition makes the short-trajectory arms' gradient counts off by ~15–25%.
