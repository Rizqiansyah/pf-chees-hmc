# WP0refs statistical-validity review (reference posteriors) — run by the lead

The statistical-review task for this branch died on the tool layer (`NOT RUN: tool layer
unavailable`), so this review is executed directly. Every number below is computed now from the
on-disk `~/pf-runs/references/*.npz` files (scripts `~/scratch/wp0refs_stats{,2}.py`).

**The question:** the four CPU NUTS references are the *only* ground truth for the run's central
gate — `b² = (mean_run − mean_ref)² / Var_ref` per parameter (for x and x²), gated at `b²_max <
0.01` at C=1024, D=10. Are they good enough to be that ground truth?

## 1. Gate resolution — the reference side is negligible
`b² = 0.01` means a bias of `0.1·sd_ref`. For the reference to resolve it, its own MCSE must be
far below `0.1·sd`. Worst per-leaf reference MCSE/sd, computed now:

| model | d | worst reference MCSE/sd | worst variable | vs 0.1 gate |
|---|---|---|---|---|
| 8schools | 10 | **0.0049** | u__tau_log__ | 20× below |
| gev | 3 | **0.0051** | u__ξ_interval__ | 20× below |
| logit_1000x25 | 25 | **0.0071** | u__b[21] | 14× below |
| radon_noncentered | 175 | **0.0085** | u__sigma_beta_log__ | 12× below |

The reference is more than adequate: its MCSE is 12–20× smaller than the bias the gate targets.
**Consequence:** the gate's resolution floor is entirely on the *run* side (C=1024, D=10), which
`r1_bias.py` already prints as the noise floor. The references do **not** need more draws.

## 2. Single seed is sufficient
Because the reference MCSE/sd ≤ 0.0085 « 0.1, the single-seed reference contributes negligible
bias to `mean_run − mean_ref`. A second seed would move `mean_ref` by ~0.005–0.009 sd — invisible
against the run-side floor. **A second seed is a waste of CPU time; one seed is correct.**

## 3. Second moment (x²) is adequate on the heavy-tailed vars
The gate also scores x², so `Var_ref(x²)` must be well estimated from 80,000 draws:
- gev `ξ_interval`: worst `MCSE(E[x²])/Var(x²)` = **0.038**
- gev `σ_log`: **0.015**
- radon `alpha_raw`: **0.021**

All small (2–4%); the stored `mcse_x2__` is present for every variable. The second-moment side is
fine even for the funnel-adjacent and boundary variables.

## 4. GEV at target_accept 0.99 — defensible, keep it
`meta__target_accept`: GEV = **0.99**, the other three = 0.95 (confirmed from the files). GEV is
the only model that hit the divergence limit at 0.95 (128 → 18 divergences at 0.99). 18
divergences in 80,000 draws (0.0225%) is negligible bias for a mean/variance gate, whereas 128
near the boundary is not. **Position: keep 0.99 for GEV.** The comparability argument (all at 0.95)
is weak because GEV is the only model that needed the change; the 18/80,000 caveat is the right
one to carry.

## 5. Unconstrained space is the correct scoring space
`r1_bias.py` loads `u__*` and scores b² in unconstrained space. That is the sampler's native space
and where the metric lives; both run and reference are measured in the same space, so the
comparison is fair and Jacobian-free. For transformed parameters (log/interval) the unconstrained
scale makes the 0.01 threshold correspond to a different *constrained* bias than for identity
parameters, but since both sides share the space the comparison remains consistent. **Position:
correct and standard; no change.**

## 6. 8 Schools funnel — the reference is not stuck
Per-chain mean spread (a confidently-stuck reference would show a chain far from the grand mean):
- `mu`: per-chain means within 0.021 sd; max chain deviation **3.71 MCSE** from the grand mean
- `tau_log__`: within 0.020 sd; max deviation **2.59 MCSE**

Backed by R̂ 1.0002 and min bulk ESS 52,051, the reference is well-mixed, not stuck. **NIT:** `mu`'s
3.71 MCSE is the max over 8 chains and borderline (~iid expectation is ~2.1); it is not evidence of
a stuck chain, but if the owner wants maximal caution a 16-chain regen would settle it. Not blocking
given the ESS.

## BLOCKING
None.

## NON-BLOCKING
- 8 Schools `mu` max chain deviation 3.71 MCSE is borderline (max of 8). Optional 16-chain regen if
  the owner wants it; the ESS says it is well-mixed.
- The stored stat columns (`mean__/var__/mcse_*`) are display-only; the gate recomputes from
  samples. Confirm no consumer reads the stored stats (the code review confirmed `r1_bias.bias()`
  recomputes).

## NIT
- `mcse_x2__` is the MCSE of `E[x²]`, not of a variance — already documented in the manifest.

## Verdict
**RECOMMEND: merge.** The four references are statistically valid ground truth for the `b² < 0.01`
gate: the reference-side MCSE is 12–20× below the gate's bias threshold on every model and for the
second moment, a single seed is provably sufficient, GEV's 0.99 choice is the right one, and the 8
Schools funnel reference is well-mixed. The gate's resolution floor is on the run side, which the
harness already reports.
