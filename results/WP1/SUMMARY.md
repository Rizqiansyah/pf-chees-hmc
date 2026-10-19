# WP1 — Metric correctness: ordering (fixed) + orientation (made explicit and measured)

Parent: `pf-r1` @ `c3bdbfc` (= `ecd9ba1` + WP0). Branch `wp/wp1`.
Commits: `c6aea8a` → `6fa5cd1` (merge) → `60c1baa` (harness) → `4460dc7` (2×2 resolution, tests, guards).

## 1. Change

> **Default changed after this summary was written (2026-10-20, `08357db`).** The owner approved
> flipping the shipped default to `covariance`, in the same merge as the permutation, because
> keeping `precision` next to a fixed permutation would ship a state that fails R-hat <= 1.01 on
> GEV in 5 of 5 seeds. Consequences for reading this file: the arm label `wp1:mc` below means
> "permuted + precision", which is now a **non-default** arm, and every phrase "default unchanged /
> shipped default" in the tables and in section 6.1 describes the state at the time of measurement,
> not the state of the branch. Evidence for the flip and its one accepted cost (GEV divergences
> 0.025 -> 0.400 per 1k, seed-stable) are in `results/LEDGER.md`.


`pfcheeshmc/sampler.py`, all inside `_run_pathfinder` plus one module helper:

| Factor | What changed | Status |
|---|---|---|
| **Ordering** | `_kernel_permutation` + `_metric_to_kernel_order` reorder the metric from Pathfinder's tuple order into the kernel's dict order, applied once at the return, for `imm_source` = `mc`, `lbfgs_lowrank`, `fisher_lowrank`. `_kernel_flat(position)` is the flatten helper WP4 must use. | **shipped, unconditional** — this is the P0 |
| **Orientation** | `imm_orientation` = `"precision"` (now `covariance`; was `precision`, = today's `mc` behaviour) or `"covariance"`. Applied to **all three sources identically**, which changes the two low-rank arms for the first time. | **explicit, default unchanged** (owner's call, see §6.1) |
| A8 contract | `idata.attrs`: `frozen_step_size`, `frozen_L_nominal`, `warmup_grad_evals_per_chain`, `sampling_grad_evals_per_chain` (summed exactly from `num_integration_steps`), plus `keep_untransformed` pass-through on `sample()` / `sample_pf_chees_hmc` for WP3. | shipped |
| Measurement knob | `imm_order="pathfinder"` skips the reorder so the 2×2 runs in one tree with one key stream. Production must not set it. | test-only |

### The ordering bug, measured on this box (`tools/probes/order_check.py`)

| model | `value_vars` order | kernel (sorted-key) order | agrees? |
|---|---|---|---|
| 8 Schools | mu, tau_log__, theta | same | yes, by luck |
| GEV | μ, σ_log__, ξ_interval__ | μ, **ξ_interval__, σ_log__** | **no** |
| radon NC | sigma_y, sigma_alpha, sigma_beta, mu_alpha, mu_beta, alpha_raw[85], beta_raw[85] | alpha_raw, beta_raw, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y | **no** |

On a d=4 fixture (prior variances 1e4 and 1e-4) `diag(imm)`:

| arm | `mc` | `lbfgs_lowrank` / `fisher_lowrank` |
|---|---|---|
| `ecd9ba1` | `[9.9e-5, 1.0e4, 1.0e4, 9.7e3]` inverted **and** unpermuted | `[1.0e4, 1e-4, 1e-4, 1e-4]` ordering bug alone |
| `wp/wp1` default (`precision`) | `[1e4, 1e4, 1e4, 1e-4]` = `1/Cov` in kernel order | `[1e4, 1e4, 1e4, 1e-4]` — same array, so all three arms now agree |

(Kernel coordinate order is `a, a, a, z`; the fixture's variances are `z`=1e4, `a`=1e-4. Under
`imm_orientation="covariance"` the same three arms give `[1e-4, 1e-4, 1e-4, 1e4]`. The point of the
row is not which orientation is right — it is that the two low-rank arms no longer hand the kernel the
opposite of what `mc` hands it, and that whichever array it is, it is now indexed the way the kernel
indexes positions. Pinned by `test_every_imm_source_hands_the_kernel_the_same_ordered_array`.)

### Convention, pinned by measurement not by naming
The array the kernel calls `inverse_mass_matrix` (call it A) is the matrix multiplying the momentum in
the kinetic energy, `KE = 1/2 p·(A p)` (`mcmc/metrics.py:262-270`), so momentum ~ `N(0, A⁻¹)` (verified:
`test_kernel_convention_is_ke_half_p_A_p_and_momentum_cov_is_A_inverse`, 4,000 draws, 15%) and leapfrog
gives `q'' = -A·grad U`. Squared frequencies are therefore `eig(A·H)` for target Hessian H. The wheel's
own naming is inconsistent (`LowRankInverseMassMatrix`'s docstring calls the same array `M⁻¹` while
`chees_adaptation.py:335` calls it "Σ"); the consumers are consistent, and that is what the test pins.

## 2. Tests added

`tests/test_metric_orientation.py`: **17 tests, 58 s CPU-forced**. Against `ecd9ba1` via
`~/scratch/wp1_legacy_check.py` (prepends `~/pf/legacy`, asserts `pfcheeshmc.__file__` before
collecting): **12 fail**. The 5 that pass there are exactly the ones that must: two documenting the
broken orientation/ordering as expected failures, one pure algebra, one kernel-convention check, and
the default-orientation pin.

Grouped: ordering (`test_every_imm_source_hands_the_kernel_the_same_ordered_array` × 3 sources × 2
orientations, `test_permutation_is_tuple_to_dict_index`, `..._identity_when_names_are_sorted`,
`..._rejects_a_tree_that_does_not_match_the_model`, `test_kernel_flat_...`), orientation
(`test_covariance_orientation_mixes_the_anisotropic_target`,
`test_precision_orientation_does_not_mix_the_anisotropic_target`,
`test_ordering_error_collapses_one_coordinate`, `test_default_orientation_is_covariance_and_mixes`),
contract (`test_unknown_orientation_raises`, `test_permuting_factors_equals_permuting_the_expansion`).

Full suite on the branch: **55 passed in 123 s**, CPU-forced, at `4460dc7` (53 from this branch + 2 merged from `pf-r1`'s stub-restore test).

## 3. Measurements

### 3a. The 2×2 that settles ordering vs orientation (CPU, done)

Fixture: four scalars, analytic posterior variances 1e2 / 3 / 1 / 1e-2, created in order
`[delta, beta, zeta, alpha]` against kernel order `[alpha, beta, delta, zeta]`. 32 chains, 500 warmup,
1,000 draws, seeds 5/6/7, **one tree, one key stream, only the two factors varied**
(`~/scratch/wp1_2x2b.py`; `PF_GPU_LOCK` not required, `jax.devices() == [CpuDevice(id=0)]` printed).

| metric order | orientation | frozen ε | max R̂ | min bulk ESS | ESS/grad | per-variable ESS (alpha/beta/delta/zeta) |
|---|---|---|---|---|---|---|
| permuted | covariance | 1.11–1.21 | **1.0020–1.0022** | 30,121–33,495 | 10.7–13.5 | ~31k / 31k / 31k / 31k |
| permuted | precision | 0.013–0.014 | **12.10–13.08** | **32–38** | 0.010–0.015 | 32–38 / 33k / 33–34 / 32–33 |
| unpermuted | covariance | 0.236–0.246 | 1.0069–**1.1451** | 166–3,176 | 0.028–0.140 | 26k / 22k / 27k / **166–3,176** |
| unpermuted | precision | 0.013–0.014 | 10.37–11.71 | 33 | 0.010–0.014 | 33–34 / 33k / 33 / 33 |

Readings:
1. The **deciding cell** (permuted + `inv(cov)`, i.e. ordering correct and only orientation wrong)
   fails at R̂ 12–13 with ESS in the tens. So orientation is not a tuning preference on anisotropic
   targets: it fails the project's own G-A bar (R̂ ≤ 1.01) at a normal budget.
2. The **ordering factor in isolation** (unpermuted + covariance) is real but differently shaped:
   exactly one coordinate collapses (zeta, variance 1e2, ESS 166 while the others hold ~26k), and on
   one seed R̂ reaches 1.145. A wrong permutation misplaces specific coordinates; a wrong orientation
   flattens every wide coordinate. That signature difference is why both need permanent tests.
3. ε is a *consequence* of the orientation as expected: 1.11–1.21 for covariance, 0.013–0.014 for
   precision, and the realised integration time T = ε·L is 5–8 vs 0.06–0.11.

**Fixture-design trap (the reason a first attempt was inconclusive).** An earlier 3-scalar fixture
(variances 1e1/1e0/1e-2, names reverse-sorted) gave *two* good cells: reversal of a symmetric
variance ladder mimics inversion, so an ordering error and an orientation error are the same error
there, and the 2×2 could not attribute anything. The degenerate run is kept for the record:
permuted+precision R̂ 8.76–10.10 / ESS 33, permuted+covariance R̂ 1.002 / ESS 24–32k,
unpermuted+precision R̂ 1.0016–1.0020 / ESS 15–18k, unpermuted+covariance R̂ 9.44–9.85 / ESS 33
(`~/scratch/wp1_2x2.py`). Anyone repeating this work must check that the permutation is not an
involution on the variance spectrum before believing a 2×2.

### 3b. Moment recovery is not a sufficient arbiter (CPU, done)

Both orientations recover the analytic posterior within 4 MCSE on the degenerate fixture, *including*
the cell that reads R̂ 10. Pathfinder's starts are themselves drawn from that (Gaussian) posterior, so
the ensemble mean and variance are right whether or not any chain moves. Any future orientation or
init test must include a mixing statistic (R̂, ESS), never moments alone.

### 3c. R2 re-baseline matrix (GPU, 4 models × 32 chains × 1,250 draws × 2,000 warmup, seeds 43–47, REPEAT=2, warm rows decide)

Arms, labelled by the two factors so no label can hide a confound:

* `legacy:mc` = unpermuted + precision — what `ecd9ba1` ships.
* `wp1:mc` = **permuted + precision** — the permutation fix alone. Non-default since `08357db`: the branch's default is now permuted + covariance.
* `wp1:mc:covariance` = permuted + covariance — Part B's combination.
* `wp1:lbfgs_lowrank`, `wp1:fisher_lowrank` — permuted, default orientation.

Rows: `results/runs/WP1__rebaseline.jsonl` (all arms) and `results/runs/WP1__covarm.jsonl`
(covariance arm; kept separate because it was run as a detached job by the lead). Medians over
5 seeds, **warm** rows; `R-hat med (max)` is the median and the worst seed. Cell id = `(arm, model,
seed, rep)`; the two pre-patch radon rows are VOID-stamped and excluded (§7).

| arm | model | n | wall s | eps | L_nom | steps | grads/ESS | ESS/s | R-hat med (max) | min tail ESS | divs/1k | warmup divs | E-BFMI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `legacy:mc = unpermuted + precision (ecd9ba1)` | 8schools | 5 | 10.51 | 0.1852 | 42 | 21.3 | 58.23 | 2085 | 1.0017 (1.0018) | 20796 | 0.000 | 26 | 0.953 |
| `legacy:mc = unpermuted + precision (ecd9ba1)` | gev | 5 | 20.11 | 0.0008583 | 54 | 27 | 73.44 | 403 | 1.0023 (1.0075) | 9058 | 0.225 | 16 | 0.990 |
| `legacy:mc = unpermuted + precision (ecd9ba1)` | logit_1000x25 | 5 | 36.38 | 0.002036 | 129 | 64.7 | 151.7 | 417 | 1.0016 (1.0018) | 20161 | 0.000 | 41 | 0.943 |
| `legacy:mc = unpermuted + precision (ecd9ba1)` | radon_noncentered | 5 | 39.64 | 0.002512 | 75 | 37.6 | 501.9 | 98 | 1.0069 (1.0078) | 4654 | 0.000 | 124 | 0.612 |
| `wp1:mc = PERMUTED + precision (shipped default)` | 8schools | 5 | 10.13 | 0.1874 | 42 | 21.4 | 57.19 | 2166 | 1.0020 (1.0020) | 20393 | 0.000 | 26 | 0.913 |
| `wp1:mc = PERMUTED + precision (shipped default)` | gev | 5 | 15.97 | 0.0009083 | 8 | 4.1 | 295.8 | 44 | 1.0465 (1.0688) | 1432 | 0.025 | 13 | 0.925 |
| `wp1:mc = PERMUTED + precision (shipped default)` | logit_1000x25 | 5 | 35.86 | 0.002052 | 122 | 61 | 163.8 | 403 | 1.0021 (1.0025) | 19368 | 0.000 | 41 | 0.936 |
| `wp1:mc = PERMUTED + precision (shipped default)` | radon_noncentered | 5 | 86.21 | 0.0006925 | 268 | 134 | 2279 | 28 | 1.0142 (1.0165) | 3335 | 0.000 | 197 | 0.629 |
| `wp1:mc:covariance = permuted + covariance` | 8schools | 5 | 9.18 | 0.5244 | 22 | 11.5 | 35.45 | 2226 | 1.0014 (1.0017) | 19091 | 0.000 | 33 | 0.950 |
| `wp1:mc:covariance = permuted + covariance` | gev | 5 | 16.74 | 0.8331 | 10 | 5.1 | 71.21 | 303 | 1.0074 (1.0161) | 6296 | 0.475 | 59 | 0.964 |
| `wp1:mc:covariance = permuted + covariance` | logit_1000x25 | 5 | 16.4 | 0.6559 | 27 | 13.6 | 46.08 | 1048 | 1.0014 (1.0016) | 18750 | 0.000 | 32 | 0.986 |
| `wp1:mc:covariance = permuted + covariance` | radon_noncentered | 5 | 26.23 | 0.9713 | 27 | 13.7 | 182.3 | 171 | 1.0066 (1.0092) | 5685 | 0.000 | 45 | 0.602 |
| `wp1:lbfgs_lowrank (permuted)` | 8schools | 5 | 10.48 | 0.1852 | 43 | 21.5 | 57.41 | 2102 | 1.0017 (1.0023) | 20063 | 0.000 | 26 | 0.926 |
| `wp1:lbfgs_lowrank (permuted)` | gev | 5 | 16.55 | 0.0009285 | 7 | 4 | 238.9 | 48 | 1.0400 (1.0575) | 1207 | 0.050 | 13 | 0.938 |
| `wp1:lbfgs_lowrank (permuted)` | logit_1000x25 | 5 | 36.69 | 0.002042 | 128 | 64.1 | 161.1 | 404 | 1.0017 (1.0022) | 20006 | 0.000 | 31 | 0.935 |
| `wp1:lbfgs_lowrank (permuted)` | radon_noncentered | 5 | 86.06 | 0.0006945 | 267 | 133.5 | 2354 | 25 | 1.0146 (1.0150) | 2863 | 0.000 | 204 | 0.612 |
| `wp1:fisher_lowrank (permuted, 2 of 4 models)` | 8schools | 5 | 17.09 | 0.1534 | 178 | 89.3 | 221.1 | 1179 | 1.0016 (1.0025) | 17591 | 0.175 | 112 | 0.955 |
| `wp1:fisher_lowrank (permuted, 2 of 4 models)` | gev | 5 | 44.09 | 0.0006724 | 116 | 58.2 | 362.3 | 108 | 1.0076 (1.0222) | 5603 | 0.900 | 195 | 1.053 |
| `wp1:fisher_lowrank (permuted, 2 of 4 models)` | logit_1000x25 | 0 | NOT RUN | | | | | | | | | | |
| `wp1:fisher_lowrank (permuted, 2 of 4 models)` | radon_noncentered | 0 | NOT RUN | | | | | | | | | | |

Head-to-head on the plan's own acceptance metric (gradients per worst-variable bulk ESS, lower is
better):

| model | `legacy:mc` | `wp1:mc` (perm-only) | `wp1:mc:covariance` (perm+cov) |
|---|---|---|---|
| 8 Schools | 58.23 | 57.19 **better** | 35.45 **better** |
| GEV | 73.44 | 295.8 **WORSE (4.0x)** | 71.21 better |
| logistic | 151.7 | 163.8 **WORSE (+8%)** | 46.08 **better (3.3x)** |
| radon NC | 501.9 | 2279 **WORSE (4.5x)** | 182.3 **better (2.8x)** |

**The permutation on its own is not shippable**: it is better on 1 of 4 models and it breaks R-hat on
the two models whose coordinate order actually differs (GEV median 1.0465 / worst seed 1.0688; radon
1.0142 / 1.0165) where the shipped code passes at 1.0023 / 1.0069. Permutation plus covariance
orientation is better on 4 of 4 and is the combination the acceptance criterion was written against.
Mechanism: with `precision`, the wrongly-ordered metric was accidentally partially compensating the
orientation error; correcting one factor without the other removes the compensation. That is exactly
what fixture A of `results/WP1/2x2.md` predicts, and the GEV cells below show it on a ladder model.

### 3d. Spot checks already in hand

| cell (warm) | wall | ε | L_nominal | steps_mean | max R̂ | ESS/s worst | divs/1k |
|---|---|---|---|---|---|---|---|
| 8 Schools, wp1 default(=precision+permuted), W=200, D=200 | — | 0.163 | 30 | — | 1.017 (short run) | — | 0 |
| 8 Schools, permuted+covariance, W=2,000 (pre-arbiter rows, `WP1__rebaseline-pre-arbiter.jsonl`) | 9.15 s | 0.5124 | 22 | 11.4 | 1.0019 | 2,253 | 0.025 |
| 8 Schools, same, cold | 39.49 s | 0.5124 | 22 | 11.4 | 1.0019 | 522 | — |

The pre-arbiter rows were taken while the branch defaulted to `covariance`; they are superseded by
3c for gate purposes but they independently reproduce the review's §2c variant-B numbers for
8 Schools (ε 0.51 vs 0.5213, nominal L 22 vs 22) and the "~1 divergence per chain in warmup" note
(40–56 warmup divergences at 32 chains).

## 4. Frozen ε / L and the realised step distribution

`frozen_L_nominal` = `ceil(T/ε)` (unjittered); `steps_mean` = realised mean of `ceil(halton(i)·T/ε)`,
and the Halton jitter has mean 0.5, so `steps_mean ≈ L_nominal/2` is expected and is not a
discrepancy — 8 Schools pre-arbiter: L 22, steps 11.4; legacy committed: L 43, steps 21.4. Comparisons
of nominal L across arms are therefore valid, and `T = ε·L` is the quantity to compare across
orientations (§3a reading 3). Recovery detail and its `2**b − 2` trap: `60c1baa`.

## 5. Gate verdicts

| gate | verdict | evidence (cell ids are arm/model/seed/rep) |
|---|---|---|
| **G-ORDER** — WP1 tests permanent; any new metric code passes the non-sorted-name model | **PASS** | 17/17 on `wp/wp1`, 12 fail on `ecd9ba1`; all three `imm_source` values covered |
| Suite green | **PASS** | 55 passed in 123 s, CPU-forced |
| **Acceptance: gradients/ESS not worse on >= 3 of 4 models** (`wp1:mc` vs `legacy:mc`) | **FAIL — 1 of 4** | better only on 8schools (57.19 vs 58.23); GEV 295.8 vs 73.44, logistic 163.8 vs 151.7, radon 2279 vs 501.9. Cells `(wp1:mc, gev, *, 1)`, `(wp1:mc, radon_noncentered, *, 1)` |
| Same criterion for the **permuted + covariance** arm | **PASS — 4 of 4** | 35.45 / 71.21 / 46.08 / 182.3 vs 58.23 / 73.44 / 151.7 / 501.9; `WP1__covarm.jsonl`, all 20 warm cells |
| **G-A max R-hat <= 1.01** | `legacy:mc` PASS; `wp1:mc` **FAIL** on GEV (median 1.0465, worst 1.0688) and radon (1.0142/1.0165); `wp1:mc:covariance` PASS on medians (1.0014-1.0074), GEV worst seed 1.0161 | `WP1__rebaseline.jsonl`, `WP1__covarm.jsonl` |
| **G-B tail ESS >= 400** | PASS in every arm that ran (min 1,207, GEV `wp1:mc`) | same |
| **E-BFMI >= 0.3** | PASS (0.60-0.99; radon is the low one in all arms, as before) | same |
| **Divergence rate not higher at equal C** | `wp1:mc` PASS (GEV 0.225 -> 0.025 per 1k, others 0). `wp1:mc:covariance` **FAIL on GEV**: 0.225 -> 0.475 per 1k, warmup divergences 16 -> 59 | `WP1__covarm.jsonl` gev cells; LEDGER (main checkout) entry at `c2993cf` |
| G3 for a default flip (>= 5 seeds, means within 1 sigma, no divergence increase) | **NOT MET** | 5 seeds exist for both arms; the divergence clause fails on GEV; per-variable mean-shift vs the 0.011/0.029 sd noise floors has not been computed |
| `wp1:fisher_lowrank` | **NOT RUN (2 of 4 models)** | 8schools + gev complete (20 rows); logit and radon terminated by the lead: "deprioritised; feeds T6/WP9 retraction list only" |
| R1 / G-R1 | out of scope for WP1 (WP3's harness) |

### Recommendation on the orientation default (for the owner at the end of the run; not applied here)

Merge the permutation **and** the explicit `imm_orientation` knob, keep the default at `precision`, and
do **not** ship the permutation on its own: as measured it is worse on 3 of 4 ladder models and it
breaks the R-hat bar on GEV and radon, which the shipped code passes. The combination that should
eventually become the default is permuted + covariance: gradients per worst-variable ESS improve on 4
of 4 (35.45/71.21/46.08/182.3 vs 58.23/73.44/151.7/501.9), worst-variable ESS/s on 3 of 4, R-hat within
the bar on every median, and it reproduces the review's §2c directions (8 Schools eps 0.52 / L 22,
logistic eps 0.66 / L 26). The one thing blocking the flip is GEV: the corrected arm there loses ESS/s
(403 -> 303) and **raises divergences from 0.225 to 0.475 per 1k** (warmup 16 -> 59), which trips G3's
no-divergence-increase clause, and GEV is also the model where the correctly-ordered metric scored
*worse* than the swapped one (see §6.7). Missing G3 evidence for a flip, concretely: (a) a GEV
divergence diagnosis at equal C over >= 5 seeds — which coordinate trips the divergence, and whether
`xi_interval__`'s boundary is being hit because the covariance metric gives it a longer trajectory;
(b) per-variable posterior-mean shifts against the same-config different-seed noise floor (0.011 sd on
8 Schools, 0.029 sd on radon NC) rather than medians only; (c) confirmation on the owner's own model.
Until (a) exists, the default stays `precision` and `imm_orientation="covariance"` is opt-in.

## 6. Surprises and contradictions with Part B

1. **Orientation: Part B's §0.1 is right on the measurement that matters, and wrong as a "the array
   is inverted" claim.** The lead's momentum probe is correct and I reproduced it: with
   A = `inv(Σ)` the sampled momentum variance *is* Σ, which is what the argument's name promises
   (`M⁻¹`). That does not make the current choice well-conditioned: `q'' = -A·grad U` puts squared
   frequencies at `eig(A·H)`, so A = `Cov⁻¹` gives 1/var per coordinate and one global ε cannot serve
   a spread. Row evidence: §3a, permuted+precision R̂ 12.10–13.08 with ESS 32–38 against R̂ 1.0020
   with ESS 30k for permuted+covariance, at 32 chains / 500 warmup / 1,000 draws / 3 seeds. Per the
   ruling I have **not** changed the default; I record that the failure is at the level of the G-A
   gate, not of taste, and hand the flip to WP4/owner with these rows.
2. **The three `imm_source` arms disagreed with each other before this WP** (`mc` inverted, low-rank
   not). Every `mc`-vs-low-rank number in `docs/ladder-results.md` — including the Fisher 1.22×/1.96×/
   0.19× ratios and "rescaling made every model worse" — compared opposite orientations. This was not
   in Part B and is not fixed by the permutation; it is a WP9 retraction candidate that needs 3c.
3. **Pathfinder's metric is the identity on any model whose initial point is its mode.** On the
   prior-only `z ~ N(0,100)` / `a ~ N(0,0.01, 3)` fixture Part B specified, the PF pool covariance came
   out `[0.947, 0.908, 1.022, 1.026]` and `lbfgs_lowrank` exactly `[1,1,1,1]`: zero gradient at the
   start gives L-BFGS no curvature pairs, so Pathfinder stays at its identity initial scaling. Two
   consequences: Part B's proposed "compare diag(imm) to the PF draw variance within 20%" test passes
   on `ecd9ba1` for both defects (the tests here inject a pool of known covariance instead), and §2d's
   "PF is under-dispersed by 100×" is really "PF's metric is not informative at all" for such models —
   relevant to WP4's refresh schedule.
4. **A moment test cannot arbitrate anything about the metric** (§3b): the non-mixing cell still
   recovers mean and variance within 4 MCSE because the starts carry them.
5. **Tool traps found on this box, each cost a run:** `az.mcse(..., method="sd")` raises
   `TypeError: only 0-dimensional arrays can be converted to Python scalars` under numpy 2.4 →
   use `arviz_stats.mcse`; `halton_sequence(2**b − 1) == 0` (the sequence indexes `arg+1`), so the
   frozen-L recovery needs `2**b − 2`; `bj.vi.pathfinder.sample` returns `(samples, state)` and
   `approximate` is `(rng_key, logdensity_fn, initial_position)`; `Metric`'s field is
   `sample_momentum`; `ravel_pytree` of a vmapped pytree flattens completely, not per sample.
6. **`bench_ladder.load_notebook_model` left a `_tiny_idata` stub on
   `pfcheeshmc.sample_pf_chees_hmc`** (fixed on `pf-r1`; I merged it). A cell that hit it would have
   reported a fast wall time, zero divergences and no exception. The harness now asserts identity with
   `bl.REAL_SAMPLE_PF` before every call and records `entry_point` in `kwargs`.

7. **GEV, the real model where the orders differ, confirms the interaction and adds a counter-example
   (`results/WP1/2x2.md`, LEDGER `c2993cf`).** All four cells at 32 chains / 1,250 draws / 2,000 warmup /
   seeds 43-45: permuted+precision (today's default) R-hat 1.0313-1.0518 with min bulk ESS 581-839;
   unpermuted+precision (shipped `ecd9ba1`) 1.0048-1.0142 with 2,361-5,363; permuted+covariance
   1.0017-1.0055 with 5,825-16,627; and unpermuted+covariance 1.0011-1.0017 with 15,202-25,142 — the
   *best* cell on GEV is the one with the wrong order. So Part B's §0.1 is CONFIRMED as a conditioning
   defect (fixture A's deciding cell, and the 4-of-4 win on the real-model covariance arm) and the
   ordering fix is confirmed real on fixture A, but on GEV the two factors are not independently
   beneficial, and GEV's preference for the swapped metric is NOT ESTABLISHED. That unresolved point,
   not the ε scale, is what should keep the default at `precision` for now.

## 7. Library-first exceptions

- The permutation: uses `ravel_pytree`, no hand-written index arithmetic.
- `frozen_nominal_L` (sampler, and a copy in the harness so the legacy arm is measured with the
  identical instrument — `ecd9ba1` cannot carry new attributes). 8 lines calling the kernel's own
  closure.
- No hand-written R̂/ESS/EBFMI/MCSE: `bench_ladder.score` (arviz 0.23, comparable with the committed
  ladder grid) and `arviz_stats.mcse`.
- **Duplicate-cell guard added** (`write_cell` in `tools/wp1_rebaseline.py`): a row whose
  `(wp, arm, model, seed, rep)` key is already live makes the harness **raise** rather than append;
  `OVERWRITE=1` replaces and prints what it removed; `VOID`-stamped rows do not occupy a key, so a
  failed cell can be refilled without erasing its record. Chosen over silent overwrite because every
  table here is a median across seeds — a duplicated cell shifts a median and looks completely normal.
  It caught a real case: radon `legacy:mc` and `wp1:mc` each had 11 rows (the pre-patch `to_array()`
  failure plus its refill); both pre-patch rows are now VOID-stamped with the reason and the
  superseding `git_sha`, and `tools/summarise_rows.py` prints an audit line (`VOID excluded`,
  `duplicate keys: 0`) before every table. Final audit: 142 rows, 2 VOID, 140 live, 0 duplicate keys.
- **Contract gap:** `pf_grads_total` is `null` in every row. BlackJAX does not
  expose a Pathfinder fit's L-BFGS gradient count; getting it means instrumenting
  `bj.optimizers.lbfgs`, outside WP1's files. A8 lists the field, so it should be specified or dropped.
- Ownership: the A8 attrs and `keep_untransformed` are in `sample()` and the public entrypoint — not
  in `_run_pathfinder`'s pool/scoring region (WP5) and not in `utils.py` (WP2). `utils.py` untouched.

## 8. Exact commands

```bash
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; WT=~/pf/wp1; cd "$WT"

# tests on this tree (CPU-forced: activation would otherwise put pytest on the GPU)
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= env PYTHONPATH="$WT" \
    python -m pytest "$WT/tests/test_metric_orientation.py" -q --rootdir "$WT" -p no:cacheprovider
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= env PYTHONPATH="$WT" \
    python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider

# the same file against ecd9ba1 (must fail): prepends ~/pf/legacy, asserts pfcheeshmc.__file__
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= python ~/scratch/wp1_legacy_check.py

# the 2x2 (ordering x orientation), CPU, one tree:   ~2 min
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= env PYTHONPATH="$WT" \
    python ~/scratch/wp1_2x2b.py            # non-degenerate fixture
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= env PYTHONPATH="$WT" \
    python ~/scratch/wp1_2x2.py             # the degenerate one, for the record

# R2 matrix, one bounded process per (model, imm_source), GPU under the lock
MODEL=8schools IMM=mc ARM=wp1 SEEDS=43,44,45,46,47 REPEAT=2 \
  OUT="$WT/results/runs/WP1__rebaseline.jsonl" \
  "$WT/tools/gpu_run.sh" pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" \
  python "$WT/tools/wp1_rebaseline.py"
# baseline arm: PYTHONPATH="$HOME/pf/legacy" ARM=legacy (identical kwargs otherwise)
#   driver for all 16 processes in sequence: ~/scratch/wp1_matrix.sh, log /tmp/wp1_matrix.log

# read it back (--cold for rep 0)
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= \
    python "$WT/tools/summarise_rows.py" "$WT/results/runs/WP1__rebaseline.jsonl"
```
