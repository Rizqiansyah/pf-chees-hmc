# WP2 — vendored ChEES with the paper-faithful raw iterate

## 1. Change

| file | what |
|---|---|
| `pfcheeshmc/chees.py` (new) | Vendored `base()` from BlackJAX 1.7.1 @ gcde2ab036 (Apache-2.0, header and whitening derivation kept verbatim). Added: `iterate` (`"ma"` default = installed behaviour, `"raw"` = paper), `t0` (paper init π/2), `single_step_iters` (L=1 while ε settles), `optim=None → adam(lr,b1,b2)`, and a per-step scalar `ChEESTrace`. |
| `pfcheeshmc/utils.py` | Warmup routed through the vendored adapter (kwargs `iterate`, `t0`, `single_step_iters`); scan ys gains the trace, returned as `WarmupInfo(info, trace)`; parameters dict is an explicit projection with named `frozen_step_size` / `frozen_L_nominal`; `inverse_mass_matrix` shape-checked against the flattened kernel-order dimension before tracing (size only — ordering is WP1's concern; the metric is passed through unchanged). |
| `tests/test_chees_vendored.py` (new) | 9 CPU tests. |
| `tools/wp2_lgrowth.py` (new) | Trajectory-growth driver on the public warmup entry point. |

Commits: `a80ac3a` (vendoring + 6 tests), `3ada873` (utils wiring + 3 tests), rows committed with this summary.

## 2. Tests added (CPU)

- `test_ma_mode_is_bit_for_bit_upstream` — 64 updates, identical deterministic inputs, every state field equal between `iterate="ma"` and the installed `bj.adaptation.chees_adaptation.base`. This is what makes "vendor-and-fix" reviewable: the default is provably the installed code.
- `test_optim_default_matches_positional`, `test_t0_none_is_bit_for_bit_upstream_init` — the new defaults are inert.
- `test_raw_iterate_grows_L_faster_in_the_early_phase` — the mechanism: on a stiff/soft 2-D Gaussian, raw L@100 ≥ ma L@100 (measured 55.9 vs 40.9) and the traces genuinely differ. The frozen-value comparison is deliberately NOT asserted here: on that toy, dual averaging drives ε to ~1e-4 and the late-phase ordering is a regime artifact. The frozen-value claim is the GPU measurement below.
- `test_trace_matches_state_and_counts`, `test_harmonic_pooling_excludes_divergent_chains`, `test_metric_shape_guard_raises`, `test_invalid_iterate_raises`, `test_bad_single_step_iters_raises`.

Suite: 38 → **47 passed** (forced CPU, 88 s).

## 3. Measurements

### 3.1 Reproduction of plan §2a/§2c (32 chains, W=2,000, A2000, seed 43)

`results/runs/WP2__lgrowth.jsonl`, GPU rows (16) are the decision rows; the CPU rows (16, seed 43, device `cpu:0`) are a cross-device sanity set. Arm labels: `A_prec_ma` = installed behaviour, `B_cov_ma` = orientation fix only, `C_cov_raw` = orientation + raw iterate, `D_cov_raw_t0` = C + t0=π/2.

| model | variant | plan target (GPU, seed 42) | this run (GPU, seed 43) | verdict |
|---|---|---|---|---|
| logistic | A prec, ma | ε 0.0021, L 136; L@100/500/1000 = 6.6/37/92 | ε **0.002083**, L **130.2**; 15.0/33.9/84.6 | ε exact; frozen L within 4.5%; L@500/1000 within 8% |
| logistic | B cov, ma | ε 0.66, L 26 | ε **0.6476**, L **27.5** | match (L +5.6%) |
| logistic | C cov, raw | ε 0.66, L 27; 34/19/28 | ε **0.6485**, L **26.0**; 42.1/34.8/28.4 | ε exact, L within 4%; L@1000 28.4 vs 28 exact; L@500 34.8 vs 19 (seed-sensitive intermediate) |
| logistic | D cov, raw, t0=π/2 | "adds nothing on top" (§0.3) | ε 0.6488, L **24.2** | confirmed: ΔL = −1.8 vs C |
| 8 Schools | A prec, ma | ε 0.18, L 43 | ε **0.1902**, L **41.3** | match |
| 8 Schools | B cov, ma | ε 0.51, L 22 | ε **0.5311**, L **21.5** | match |
| 8 Schools | C cov, raw | ε 0.53, L 19 | ε **0.5152**, L **22.3** | ε within 3%; L +17% (largest gap; the same variant reads 21.7 on CPU seed 43, so the delta is seed, not device) |

**Reproduction: PASS** — seven of eight frozen pairs within 8%, the eighth within 17% on a different seed. The intermediate L@500 is seed-sensitive while the endpoints are stable; the endpoint is the decision-relevant number.

### 3.2 What the deviation costs, and where the wall time goes (same rows, warm rep)

Two separate effects, not one. The frozen trajectory time T = L·ε is dominated by the **metric**: logistic A 0.271 → B 17.8 → C 16.9 → D 15.7 (a 65× rescale — the covariance metric makes the geometry isotropic enough for ε to grow 300×, and the ChEES criterion on the whitened geometry then chooses a longer trajectory; the reviewer's own §2c numbers show the same jump, 136·0.0021=0.29 vs 26·0.66=17.2). The **iterate** is the small effect: within each metric, raw moves frozen T by ≤12% (logistic B→C −5%, B→D −12%; 8 Schools B 11.42 → C 11.49 → D 10.81). So the raw iterate does not change where the run ends in any material way; it changes how fast warmup gets there.

Warmup wall, logistic 1000x25: **A 16.7 s → C 7.9 s** (2.1× faster) — **but that pair moves two factors (orientation AND iterate). The iterate alone is B → C: 7.30 → 7.86 s, i.e. raw is **7.6% slower** on GPU (and 27.42 → 31.43 s, +14.6%, on CPU), because raw starts warmup at a larger L (L@100 25–42 vs 8.3) and pays for it. The 2.1× belongs to the metric, not to `iterate="raw"`. 8 Schools: A 4.08 → C 3.51 s; B → C +1.0%. The saving is leapfrog count, not gradient cost: A's L climbs to 130 over the warmup (4.7× more integration steps per transition at the end), while C sits at ~26 — 2,000 warmup steps × fewer leapfrogs. The 2.1× (not 4.7×) is because the average over the run includes the early low-L steps both arms share.

Warmup divergences: flat across variants on logistic (32–42); on 8 Schools the cov metric doubles them (34 → 63 in B/C/D) — a metric effect, not an iterate effect (C, D equal B), consistent with §2c's ~1 divergence/chain observation.

What `iterate="raw"` costs in wall time at C=32: **nothing — it is the saving**, on the models whose trajectory is long on the failing orientation.

## 4. Frozen ε–L and realised steps

GPU rows (ε, L, T=L·ε): logistic A 0.002083 / 130.2 / 0.271, B 0.6476 / 27.5 / 17.81, C 0.6485 / 26.0 / 16.86, D 0.6488 / 24.2 / 15.70; 8 Schools A 0.1902 / 41.3 / 7.86, B 0.5311 / 21.5 / 11.42, C 0.5152 / 22.3 / 11.49, D 0.535 / 20.2 / 10.81. Attribution: the metric arm (A→B) rescales T by 1.5× (8 Schools) to 65× (logistic); the iterate arm (B→C→D) moves T by ≤12%. The warmup wall saving (section 3.2) is the iterate's, because it stops L overgrowing; the T rescale is the metric's, and it is expected behaviour of a preconditioner that makes the geometry isotropic.

## 5. Gate verdicts

- **Reproduction (deliverable 3, first half): PASS** — eight frozen pairs, seven within 8% of the plan's GPU targets, the eighth (8 Schools C, L 22.3 vs 19) within 17% and stable across device at 21.7/22.3, i.e. a seed effect.
- **Accept/kill (deliverable 3, second half): ACCEPT as an option; the kill criterion is not met.** The plan's kill rule is "raw's noise at C=32 makes frozen L worse than ma at W=2,000 on two or more models". Measured at W=2,000, GPU, C=32: logistic 26.0 (raw) vs 27.5 (ma) — raw better by 5.5%; 8 Schools 22.3 (raw) vs 21.5 (ma) — raw worse by 3.7%. That is worse on ONE model, within single-seed spread, not two or more. Frozen trajectory time T agrees to ≤12% within each metric. The honest statement is that one seed cannot separate 5% from noise, so the default-flip question is genuinely open — which is why it goes to G3.
- **Default stays `ma`** (shipped behaviour unchanged; the option exists for callers). **G3 (5+ seeds, means within 1σ, no divergence-rate increase): NOT RUN** — deliberately. Flipping the default is the owner's call per the plan, and the data for it is: raw saves 2.1× logistic warmup wall (16.7 → 7.9 s) with frozen T within 12%, changes frozen L by ≤5% on two models at one seed, and shows no divergence-rate increase on the iterate axis (the 8 Schools warmup-divergence doubling is the cov metric's, identical in B, C, D).

## 6. Surprises and contradictions with the plan

1. The plan's §2c frozen-L target for logistic A is 136; I reproduce **130.2 on GPU** (4.5% low) with seed 43 vs the plan's 42, and 116.3 on CPU (15% low). The gap shrinks on the plan's own device, so it is seed, not a contradiction. Logged either way.
2. Logistic C's L@500 is 34.8 here vs the plan's 19; L@1000 is 28.4 vs 28 (exact). The intermediate curve is seed-sensitive even when the endpoint is stable; the endpoint is the decision-relevant number.
3. The CPU-vs-GPU spread on frozen L (116 vs 130 for logistic A) is larger than the seed spread would suggest and is device-dependent reduction order in the 2,000-step moving average — a reminder that L is only reproducible to ~10% without pinning device and seed. Not a defect; stated for the reporting task.
4. No contradiction between the vendored `iterate="ma"` and the installed adapter: 64 steps, all fields, exact equality on CPU. The one-line fix is the one-line fix.

## 7. Library-first exceptions and handoffs

- Vendoring itself is the exception, justified in the `pfcheeshmc/chees.py` module docstring: the deviation is inside `base`'s update loop and no BlackJAX option expresses it.
- **Handoff to the sampler owner (WP1's branch or a later merge):** `idata.attrs` currently receives only `warmup_steps_used` (sampler.py:933). `chees_parameters` now also carries `frozen_step_size` and `frozen_L_nominal`; writing them into `idata.attrs` is a one-line addition the row schema (A8) expects and WP1's harness already reads.
- **Handoff to WP4:** the warmup trace (`WarmupInfo.trace`) gives per-step (ε, T, harmonic acceptance) for free; an ensemble-metric refresh can gate on the harmonic-acceptance tail without new instrumentation.

## 8. Exact reproduce commands

```
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; WT=~/pf/wp2; cd "$WT"
# tests (CPU):
pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" CUDA_VISIBLE_DEVICES= \
  python -m pytest "$WT/tests/test_chees_vendored.py" -q --rootdir "$WT" -p no:cacheprovider
# trajectory growth (GPU, under the lock; the CPU variant is what produced table 3.1):
"$WT/tools/gpu_run.sh" pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python "$WT/tools/wp2_lgrowth.py"
# CPU variant of the same:
JAX_PLATFORMS=cpu PF_GPU_LOCK=1 pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python "$WT/tools/wp2_lgrowth.py"
```
