# F2a SUMMARY — log-density with model inputs as runtime inputs

Task: fast-tier, two rounds. Branch `f2-a`, head 14cd7cf, fast-forwarded into `f2`. Task's suite line
(full suite on 14cd7cf): `262 passed, 5 warnings in 380.69s (0:06:20)`. `tests/test_lifted.py`
alone: `71 passed in 84.37s`. Files: `pfcheeshmc/lifted.py` (new), `pfcheeshmc/sampler.py` (wiring at
`:185-188` and `:1365-1366`), `tests/test_lifted.py` (new).

## DEVIATIONS (all accepted by the lead)
1. **Value-use filter also allows `Shape`, `Shape_i` and `Output`** (`lifted.py:146`).
   - With the plan's literal list, every `pm.Data` stays static: its `Shape_i` cannot be folded.
     So does an MvNormal's observed array, whose logp calls `pt.shape(value)`. Their values would
     then leak into the program.
   - Mutation M5 shows the leak.
2. **Erfcx guard band and series are wider than planned** (`lifted.py:82-128`).
   - JAX returns 0 on float64 [26.54326, 26.64175] (float32 [9.19455, 9.41928]), i.e.
     [erfcinv(tiny), sqrt(log(max))]. The band is derived from the dtype with a 0.1% margin.
   - The series has 9 terms: 3 terms give 5.3e-9 relative error.
   - Stock JAX's gradient is also wrong in the band: −1.128 against −8.05e-4.
3. **Probe 6's sign:** `get_jaxified_logp(negative_logp=False)` is `−model.logp()`; magnitudes agreed.
4. **Widening also covers folded one-element constants.**
   - For a shaped RV the literal reaches the logp as `ExpandDims(const)`, which PyMC's rewrites fold
     into a (1,) constant with the literal's autocast dtype.
   - These are matched by fingerprint (dtype, shape, bytes) against `constant_fold` of the RV's
     literal-built parameters, and widened the same way (`lifted.py:237-277`).
   - Without this, logit and radon would keep dtype-dependent keys (mutation M9).
   - If the match ever misses, the only effect is a cache miss, never a wrong value.
5. **Widening reads `op.dist_params(node)`**, so rng and size are never widened.

## Lead change in round 2: dtype widening (rule "model_inputs")
- **Problem:** PyTensor autocasts Python literals to the smallest exact dtype (0 and 10 → int8,
  1000 → int16, 2.5 → float32, 2.7 → float64). A lifted prior's dtype, and so its key, then depended
  on its value.
- **Fix:** before lifting, scalar distribution-parameter constants (plus the folded one-element forms)
  are widened with `clone_replace(..., rebuild_strict=False)`:
  - a float narrower than floatX → floatX;
  - an int param of a float-valued RV → floatX;
  - an int param of an int-valued RV → int64;
  - never narrowed.
- **Key now stable:** one key and identical HLO across (0,10), (0.0,10.0), (2.5,2.7), (0.5,3.0),
  (0.1,10.1), (1,1000), (100,40000) and (−3,12.5), for scalar, shaped, Poisson/Binomial and
  TruncatedNormal models.
- **Fallback chain:** `RULES = ("model_inputs", "model_inputs_native_dtype", "arrays_and_shared",
  "none")`. `pm.LKJCholeskyCov(n=3)` is a natural widening failure; it falls back to the native-dtype
  rule with a warning.
- **Upstream bug found and fixed on our path:** `pm.Uniform("u", -100, 100)` has logp NaN in PyMC
  5.28.5 on both backends. The int8 literals make `upper − lower` overflow. `lift_logp` gives −1.38629,
  equal to the float-literal model.
- **Numerics vs main:**
  - main constant-folds float32-representable normalisers in float32, so the widened logp differs by
    ≤ 1.4e-9 relative on the ladder models and ≤ 1.1e-7 on Multinomial;
  - gradients agree within 1e-7;
  - the widened side is the more accurate one.

## Spec items
- **Builder:** `lift_graph(model, inputs, outputs, rule)` (`:379`), `lift_logp` (`:415`), and the
  frozen dataclass `LiftedGraph` (`:158`) — DONE.
- **Rule C:**
  - candidates in `_candidates` (`:280`);
  - `_first_bad_client` looks through `DimShuffle`;
  - lifting is done before `mode.JAX.optimizer.rewrite`;
  - the order is the first toposort appearance, deterministic across `PYTHONHASHSEED`;
  - graphs with a shared RNG are refused.

  DONE.
- **Fallback chain**, with a `RuntimeWarning` naming the rule and the exception — DONE.
- **Key:** sha256 of the lowered StableHLO at `ShapeDtypeStruct`s (`:328`). The text has no `loc(`;
  one ulp in a static constant changes the key — DONE.
- **Registry:** key → first `fn`; unbounded for the process lifetime — DONE.
- **Erfcx (F2-1):** registered on import, and it applies inside `Composite` too — DONE.
- **Sampler wiring:** `self._lifted` plus a closure over its values; the value→RV transform goes
  through `lift_graph`; the `get_jaxified_*` imports are removed — DONE.

## Tests (71, 84 s) and mutations
- **Exactness:**
  - ladder and extra-model logp vs the C backend on the same widened graph, ≤ 1e-12;
  - closeness to today (`get_jaxified_logp` / `compile_logp`), ≤ 1e-7;
  - gradients ≤ 1e-7.
- **Extra models:** `pm.Data`, integer-written priors, MvNormal with a constant cov, a Potential with
  a data array, a Poisson GLM, TruncatedNormal with a free mu.
- **No leaks:** fresh builds from different seeds lower to identical HLO and keys (bypassing the
  registry); a shape change gives a new key.
- **Key stability:** prior-literal dtype does not change the key; the native rule keeps the old
  behaviour.
- **Uniform:** `Uniform(-100,100)` is finite.
- **Rewrites survive:** `log1p` and `square` survive lifting.
- **Fallbacks:** natural failure (shape-feeding shared variable; LKJ), forced at each level, and the
  last rule's failure propagates.
- **Erfcx:** values on [−20, 1e5] including the band at rtol 1e-12; float32 band; gradient.
- **Other:** registry identity; value→RV transform; end-to-end `sample` against the old closure.

| Mutation | Fails |
|---|---|
| M1 lift nothing | 6 identical-HLO tests ("a data or prior value leaked") |
| M2 lift every constant (rule A) | `log1p` and `square` rewrite tests |
| M3 erfcx without the guard | erfcx value and gradient tests |
| M4 no registry | shared-function-object test |
| M5 no `Shape`/`Shape_i` | `[mvnormal]`, `[pm_data]` |
| M6 never fall back | shape-feeding fallback test (`NotImplementedError` from `jnp.arange`) |
| M7 no Erfcx registration | `[truncnorm]` and the old-path-crash test |
| M8 no widening | 4 dtype-key tests and the Uniform test |
| M9 no folded matching | `[shaped]`: "7 different programs for 8 literal pairs" |
| M10 no never-narrow | `[float64-uint64-int64-None]` |
| M11 widening failure not caught | LKJ fallback test: "DID NOT WARN" |

Old-code evidence:
- TruncatedNormal with a free mu on stock `get_jaxified_logp` raises `NotImplementedError: No JAX
  implementation for Op erfcx. Implementation is available if tfp-nightly is installed`.
- `Uniform(-100,100)`: old `get_jaxified_logp=nan`.

## Measurements (task, CPU, host load about 1)

All four ladder models took rule `model_inputs`.

| model | n_lifted | kept static |
|---|---|---|
| 8schools | 7 | 1 × (Subtensor int64 (1,)): RV size vectors |
| gev | 8 | 1 × the same |
| logit_1000x25 | 4 | 2 × the same |
| radon_noncentered | 8 | 2 × the same |

| model | vmap(value_and_grad), C=1024, ms: today | lifted | `lift_logp` build, s | `get_jaxified_logp`, s |
|---|---|---|---|---|
| 8schools | 0.188 / 0.191 | 0.150 / 0.136 | 0.16–0.20 | 0.13 |
| gev | 3.159 / 3.172 | 2.959 / 3.065 | 0.41–0.62 | 0.33 |
| logit_1000x25 | 13.78 / 13.97 | 14.06 / 14.44 | 0.13–0.14 | 0.10 |
| radon_noncentered | 17.19 / 18.96 | 18.13 / 18.51 | 0.27–0.50 | 0.22 |

Sampler draws vs base 1b7d1c4 (seed 3, C=16, D=10, W=40):
- logit: bit-identical;
- 8schools: ≤ 6.7e-9;
- gev: ≤ 2.8e-8.

The cause is prior arithmetic moving from PyTensor's constant folding into XLA (ulp-level), amplified
by the trajectories. Accepted; the bias gate runs in F2f.

## Limitations
- **Merged equal constants:** a prior literal equal to another model constant is one graph node
  until it differs, so it becomes a different program (pinned by a test). The same coincidence can lift
  a formula constant next to an equal prior literal.
- **LKJ fallback:** the LKJ-style widening failure keeps value-dependent dtypes, with a warning.
- **Stale values:** `LiftedGraph.values` is a build-time snapshot. A new `PfCheesHMC` is built per
  `sample_pf_chees_hmc` call (`sampler.py:1741`). Instance reuse exists only in tools and tests.
  `pm.set_data` between calls on one instance is not seen; F2b re-reads.
- **UNVERIFIED:** radon draw identity; models with unknown static dims; throughput on a quiet host.

## JAX bug report (draft, not filed)
`jax.scipy.special.erfcx` returns exactly 0 for float64 x in [26.5433, 26.6417] and float32 x in
[9.1946, 9.4193], and its gradient is wrong there.
- **Cause:** `exp(x**2) * erfc(x)` is used up to `sqrt(log(finfo.max))` while `lax.erfc` is already
  flushed to 0 above `erfcinv(finfo.tiny)`.
- **Fix:** switch to the asymptotic branch at `erfcinv(finfo.tiny)`.
- **Repro:** `jax.scipy.special.erfcx(jnp.float64(26.6))` gives 0.0; SciPy gives 0.021195.
