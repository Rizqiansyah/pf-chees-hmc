# WP1 — lead's independent verification

Run by the lead because the two mandated reviewer tasks were not completed and wrote no review file. A static review pass runs separately and is static; this file carries the evidence that only execution can produce.

## 1. Merged tree is green

`pf-r1-int` = `pf-r1` + `wp/wp8` + `wp/wp0refs` + `wp/wp1` + `wp/wp2` + `wp/wp3`, CPU-forced:

```
99 passed in 149.33s
```

Before the seam fix at `1cb74bf` the same tree was **11 failed / 88 passed**, every failure `AttributeError: 'WarmupInfo' object has no attribute 'num_integration_steps'`. The branches were individually green (55 / 47 / 66 / 42) and the combination was broken — the reason the merged-tree rule now stands.

## 2. The G-ORDER tests have teeth against the pre-fix tree

Command (pinned import to `ecd9ba1`, so a legacy failure cannot be an import error):

```
pixi run --manifest-path "$MAIN/pixi.toml" env CUDA_VISIBLE_DEVICES= \
  python ~/scratch/legacy_teeth.py
```

Result: **12 failed, 5 passed in 51.83 s**, and the failures are the intended assertions. Examples:

- `test_covariance_orientation_mixes_the_anisotropic_target` — `max R-hat 10.3706, per-variable ESS {'delta': 32.6, 'beta': 37842.0, 'zeta': 32.7, 'alpha': 34.0}`: three of four coordinates frozen.
- All six `test_every_imm_source_hands_the_kernel_the_same_ordered_array[{mc,lbfgs_lowrank,fisher_lowrank}-{precision,covariance}]` fail, i.e. the ordering defect is caught in every arm, not only the default one.
- `test_permutation_is_tuple_to_dict_index`, `test_permutation_rejects_a_tree_that_does_not_match_the_model`, `test_unknown_orientation_raises`, `test_kernel_flat_matches_the_kernels_own_ravel` fail because the mechanism does not exist on `ecd9ba1`.

The 5 passing tests are the orientation-agnostic convention pins (kinetic-energy contract, momentum covariance), which is exactly the right set to pass on both trees.

## 3. Orientation arbiter, run personally on the merged tree

`~/scratch/orientation_arbiter.py` — 4 conjugate-Normal scalars, posterior variances 1e2 / 3 / 1 / 1e-2, `value_vars` order `[delta, beta, zeta, alpha]` against kernel order `[alpha, beta, delta, zeta]`, 32 chains, 500 warmup, 1,000 draws, seed 5:

| cell | max R̂ | min bulk ESS | ESS/gradient | worst mean error | ε | nominal L |
|---|---|---|---|---|---|---|
| permuted + covariance | 1.0021 | 30,967 | 12.45 | 0.0102 sd | 1.2134 | 4 |
| permuted + precision (shipped) | **12.1019** | **32** | 0.014 | 0.2319 sd | 0.0140 | 4 |
| unpermuted + covariance | 1.0069 | 3,176 | 0.140 | 0.0203 sd | 0.2425 | 45 |

Matches the task's reported fixture to the digit, which is what makes its `imm_source` claim credible enough to record.

## 4. Verdict

Merge `wp/wp1` for the **knobs, the permutation and the tests**. It changes no default behaviour, which is why the default flip remains separately gated. Two things this verification does *not* settle:

1. **GEV's divergence rise under the correct orientation** (0.225 → 0.475 per 1k) — the only real obstacle to the flip, and unexplained.
2. **GEV's wrongly-ordered cell beating the correct one** (1,310 vs 343 ESS/s) — logged, not explained, assigned to WP4.
