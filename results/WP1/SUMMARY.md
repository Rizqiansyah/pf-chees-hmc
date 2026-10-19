# WP1 — Metric correctness: orientation + ordering

Parent: `pf-r1` @ `c3bdbfc` (which is `ecd9ba1` + WP0). Branch `wp/wp1`.

## 1. Change

`pfcheeshmc/sampler.py`, three edits, one choke point:

| What | Where |
|---|---|
| `inverse_mass_matrix = cov` (was `jnp.linalg.inv(cov)`), ridge kept | `_run_pathfinder` step 6 |
| `_kernel_permutation` / `_metric_to_kernel_order` — reorder the metric from Pathfinder's tuple order into the kernel's dict order, applied once at the return, covering `imm_source` = `mc`, `lbfgs_lowrank`, `fisher_lowrank` | new methods + step 8 |
| `_kernel_flat(position)` — the one true flatten for future metric code (WP4) | new method |
| A8 attrs (`frozen_step_size`, `frozen_L_nominal`, `warmup_grad_evals_per_chain`, `sampling_grad_evals_per_chain`) and a `keep_untransformed` pass-through on `sample()` / `sample_pf_chees_hmc` | `sample()` |

Commits: `c6aea8a` (fix + tests), `6fa5cd1` (merge `pf-r1`), `60c1baa` (harness + frozen-L fix).

### Why the covariance is the right orientation (verified in the installed wheel, three ways)
- `mcmc/metrics.py:262-270`: `velocity = linear_map(inverse_mass_matrix, momentum)`, `KE = 1/2 p·velocity`;
  with `_format_covariance(..., is_inv=True)` the momentum is drawn as `N(0, imm⁻¹)`. Leapfrog then gives
  `q'' = -imm·grad U`, so `imm ≈ Cov(q)` is what makes every coordinate oscillate at unit frequency.
- `adaptation/chees_adaptation.py:335` documents the same argument as "the diagonal inverse_mass_matrix
  **(Σ)**", and `:453` computes `v' = Σ p'`.
- `adaptation/mass_matrix.py:344-353` — BlackJAX's own Stan window adaptation — returns
  `beta_data*covariance + …`, i.e. the covariance, to the same slot.
- Naming caveat for the record: `mcmc/metrics.py`'s `LowRankInverseMassMatrix` docstring calls that same
  array `M⁻¹`. The library's *naming* is inconsistent; its consumers are not. Candidate upstream doc fix.

### The ordering bug, concretely
Pathfinder is fitted on `tuple(position[n] for n in model.value_vars)` (`:352`), so its draws, covariance
and L-BFGS factors are in `value_vars` order. The kernel state is a dict, and `ravel_pytree` sorts dict
keys (`mcmc/metrics.py:236-244`). Verified on this box with `tools/probes/order_check.py`:

| model | `value_vars` order | kernel (sorted) order | agrees? |
|---|---|---|---|
| 8 Schools | mu, tau_log__, theta | same | yes, by luck |
| GEV | μ, σ_log__, ξ_interval__ | μ, **ξ_interval__, σ_log__** | **no** |
| radon NC | sigma_y, sigma_alpha, sigma_beta, mu_alpha, mu_beta, alpha_raw[85], beta_raw[85] | alpha_raw, beta_raw, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y | **no** |

On the non-sorted fixture (d=4, prior variances 1e4 and 1e-4) the measured `diag(imm)`:

| arm | mc | lbfgs_lowrank / fisher_lowrank |
|---|---|---|
| `ecd9ba1` | `[9.9e-5, 1.0e4, 1.0e4, 9.7e3]` — inverted **and** unpermuted | `[1.0e4, 1e-4, 1e-4, 1e-4]` — ordering bug alone (the low-rank routes never took the inverse) |
| `wp/wp1` | `[1e-4, 1e-4, 1e-4, 1e4]` | `[1e-4, 1e-4, 1e-4, 1e4]` |
| correct | `[1e-4, 1e-4, 1e-4, 1e4]` (kernel order = a, a, a, z) | same |

## 2. Tests added

`tests/test_metric_orientation.py` — 10 tests, CPU-forced with the post-activation env form, 37 s.
Run against the parent commit with `~/scratch/wp1_legacy_check.py` (prepends `~/pf/legacy` to
`sys.path` and asserts `pfcheeshmc.__file__` before collecting): **8 of 10 fail on `ecd9ba1`, 10/10
pass here**. Full suite: **46 passed** (was 36).

| test | fails on parent? |
|---|---|
| `test_metric_is_the_covariance_in_kernel_order` | yes |
| `test_kinetic_energy_and_momentum_scale_every_coordinate` | yes (see §6.3) |
| `test_low_rank_metric_sources_are_kernel_ordered_too[lbfgs_lowrank]` | yes |
| `test_low_rank_metric_sources_are_kernel_ordered_too[fisher_lowrank]` | yes |
| `test_permutation_is_tuple_to_dict_index` | yes (no such helper) |
| `test_permutation_is_identity_when_names_are_sorted` | yes |
| `test_permutation_rejects_a_tree_that_does_not_match_the_model` | yes |
| `test_kernel_flat_matches_the_kernels_own_ravel` | yes |
| `test_frozen_step_size_is_in_posterior_units` (end-to-end, public entrypoint) | yes |
| `test_permuting_factors_equals_permuting_the_expansion` | no — pure algebra, guards the §6.2 deviation |

## 3. Measurements

R2 shape: 4 models × 32 chains × 1,250 draws × 2,000 warmup, seeds 43–47, REPEAT=2 (rep 0 cold,
rep 1 warm), decisions on **warm**. Rows: `results/runs/WP1__rebaseline.jsonl`
(`tools/wp1_rebaseline.py`, one process per model × imm_source, GPU under `/tmp/pf-gpu.lock`).

<!--TABLE-->

## 4. Frozen ε / L and the realised step distribution

Reading convention that §2c of the plan does not state: `frozen_L_nominal = ceil(T/ε)` is the
**unjittered** nominal length, while `steps_mean` is the realised mean of `ceil(halton(i)·T/ε)`, and
the Halton jitter has mean 0.5. So `steps_mean ≈ L_nominal/2` is expected, not a discrepancy — 8
Schools here is L=22 with steps 11.3. Legacy's "L=43, steps 21.4" follows the same 2:1 ratio, so
nominal-L comparisons across arms are valid.

<!--FROZEN-->

## 5. Gate verdicts

<!--GATES-->

## 6. Surprises and contradictions with Part B

1. **CONTRADICTION — the specified WP1 test instrument cannot detect either defect.** Part B asks for
   `diag(imm) ≈ kernel-order variance of the PF draws within 20%` on `z ~ N(0,100)` (created first) and
   `a ~ N(0,0.01, shape=3)` (created second). Measured on exactly that model: Pathfinder's own pool
   covariance is `[0.947, 0.908, 1.022, 1.026]` and its `lbfgs_lowrank` metric is exactly `[1, 1, 1, 1]`.
   Not a repo bug — a prior-only model has zero gradient at the initial point, so L-BFGS receives no
   usable curvature pairs and Pathfinder's approximation never leaves its identity initial scaling. A
   20%-tolerance test against the PF pool on that model passes on `ecd9ba1` for *both* defects.
   Kept as: one LEDGER entry, and the tests here inject a pool of known covariance so the real
   covariance/ridge/permutation code is exercised. Consequence for the run, worth telling WP4:
   **Pathfinder's metric is identity on any model whose initial point is its mode**, so "PF is
   under-dispersed" (§2d) is a stronger statement than "PF is under-dispersed by 100×".
2. **Deliberate deviation, with proof.** Part B says to permute the low-rank factors (`sigma[perm]`,
   `U[perm,:]`) before expansion. I permute the expanded dense matrix at the single choke point.
   Conjugation by a permutation matrix commutes with `D(I + U(Λ−I)Uᵀ)D`, so the two are the same map;
   `test_permuting_factors_equals_permuting_the_expansion` pins it to 1e-10 rather than carrying two
   code paths.
3. **A single-coordinate KE check is not enough.** An early version of the KE test asserted only the
   z coordinate; it *passed* on `ecd9ba1`, because an inverted and unpermuted diagonal metric has two
   errors that cancel at that index. The test now checks all four coordinates and the momentum scale
   of each (`p ~ N(0, imm⁻¹)`), which fails on the parent as intended.
4. **§2c reproduced independently.** 8 Schools with the covariance metric: ε 0.5213 (plan: 0.51),
   nominal L 22 (plan: 22), warmup divergences 1.25/chain (plan: "~1 per chain on every variant").
   This is a different code path (vendored-free, MA-fed, my own run), so it is a real cross-check.
5. **Confirmed on 8 Schools, at 5 seeds:** steps/draw 21.4 (legacy, committed in
   `docs/ladder-results.md`) → 11.3 (WP1) = **1.9× fewer leapfrogs per draw**, matching the plan's
   "about 2×" expectation. Warm wall is unchanged within seed noise (9.15 s vs the committed
   10.63 s median, −14%), because 8 Schools is dominated by per-step host machinery, not gradients —
   see §6.6.
6. **Where the saving is *not*.** On 8 Schools the fixed metric halves the gradients per draw but
   barely moves wall time and does not move worst-variable ESS/s (2,244 vs 2,098–2,260 committed).
   The gradient saving therefore only becomes throughput where the model evaluation is the cost
   (logistic, radon). The gate is written on gradients/ESS, and that is the right choice.

## 7. Library-first exceptions

- The permutation itself: allowed, and it uses `ravel_pytree` rather than hand-built index arithmetic.
- `frozen_nominal_L` (sampler) and `frozen_nominal_L` (harness): 8 lines that call the kernel's own
  `integration_steps_fn` at the all-ones Halton index. Duplicated in the harness so the legacy arm is
  measured with the identical instrument (ecd9ba1 cannot carry new attributes). Trap recorded: the
  argument must be `2**b − 2`, not `2**b − 1` — `(arg+1) = 2**b` is divisible by every `2**k` in the
  sequence, so the jitter is 0 and the reported L is 1.
- No hand-written R-hat/ESS/EBFMI: rows come from `tools/bench_ladder.py::score` (arviz 0.23), which
  keeps new rows comparable with the committed ladder grid.
- **Contract gap for the lead:** `pf_grads_total` is `null` in every row. BlackJAX does not
  expose the L-BFGS gradient count of a Pathfinder fit; getting it means instrumenting
  `bj.optimizers.lbfgs`, which is outside WP1's files. A8 lists the field, so it should either be
  specified or dropped.
- Ownership note: the A8 attrs and the `keep_untransformed` pass-through live in `sample()` and the
  public entrypoint — not in `_run_pathfinder`'s metric block, `utils.py` (WP2), or the pool/scoring
  and sampling-scan regions (WP5). No other WP's region was touched.

## 8. Exact commands

```bash
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; WT=~/pf/wp1; cd "$WT"

# tests on this tree (CPU-forced: activation would otherwise put pytest on the GPU)
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= \
    python -m pytest "$WT/tests/test_metric_orientation.py" -q --rootdir "$WT" -p no:cacheprovider
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= \
    python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider

# the same file against ecd9ba1 (must fail): ~/scratch/wp1_legacy_check.py prepends ~/pf/legacy
# and asserts pfcheeshmc.__file__ before collecting
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= python ~/scratch/wp1_legacy_check.py

# the matrix (one bounded process per cell group, GPU under the lock)
MODEL=8schools IMM=mc ARM=wp1 SEEDS=43,44,45,46,47 REPEAT=2 \
  OUT="$WT/results/runs/WP1__rebaseline.jsonl" \
  "$WT/tools/gpu_run.sh" pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" \
  python "$WT/tools/wp1_rebaseline.py"
# legacy arm: PYTHONPATH="$HOME/pf/legacy" ARM=legacy IMM=mc
#   driver that ran all 16 processes in sequence: ~/scratch/wp1_matrix.sh (log /tmp/wp1_matrix.log)

# read it back
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= \
    python "$WT/tools/summarise_rows.py" "$WT/results/runs/WP1__rebaseline.jsonl"      # --cold for rep 0
```
