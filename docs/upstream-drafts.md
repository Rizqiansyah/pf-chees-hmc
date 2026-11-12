# Upstream bug reports: drafts

**Both drafts are NOT FILED** (draft (b) is no longer needed: JAX 0.12 fixed the bug). They are pending actions for the maintainer, not history. Each reproducer below was run
on 2026-10-28 on this repository's pixi environment (CPU, `JAX_PLATFORMS=cpu`); the output shown is what it printed.

Package-side handling, so that nothing here blocks the sampler:
- the Uniform overflow is repaired on this package's path by widening scalar prior constants (`pfcheeshmc/lifted.py:_widen_model_inputs`);
- the `erfcx` band is replaced by an asymptotic series (`pfcheeshmc/lifted.py:_erfcx_guarded`).

See `docs/design.md`, sections 1 and 6.

---

## (a) PyMC: `pm.Uniform("u", -100, 100)` has a NaN log-probability (int8 overflow)

**Draft title:** `Uniform` with integer bounds wider than 127 gives `logp = nan`

**Versions:** pymc 5.28.5, pytensor 2.38.2, Python 3.13, floatX float64. The NaN is the same through the C backend and through
PyMC's JAX backend (`pymc.sampling.jax.get_jaxified_logp`).

**Reproducer:**

```python
import pymc as pm
from pymc.sampling.jax import get_jaxified_logp

with pm.Model() as m:
    pm.Uniform("u", -100, 100)
ip = m.initial_point()
print("C backend  :", m.compile_logp()(ip))
print("JAX backend:", get_jaxified_logp(m)([ip["u_interval__"]]))

with pm.Model() as m2:
    pm.Uniform("u", -100.0, 100.0)
print("float bounds:", m2.compile_logp()(m2.initial_point()))
```

**Output (confirmed):**

```
RuntimeWarning: overflow encountered in scalar subtract      (from pytensor/tensor/rewriting/math.py:2678)
C backend  : nan
JAX backend: nan
float bounds: -1.3862943611198906
```

**Cause.** PyTensor gives a Python integer literal the smallest integer dtype that holds it, so `-100` and `100` are `int8`.
The interval transform and the density compute `upper - lower` while both are `int8`, and `100 - (-100) = 200` wraps to `-56`
(`np.int8(100) - np.int8(-100)` is `-56`), so a logarithm of it is NaN. Checked on the width: `Uniform(-50, 50)` and
`Uniform(-63, 64)` (width 127) give `-1.386...`; `Uniform(-64, 64)` (width 128) gives `nan`. Float bounds are fine.

**Expected:** `log(1/200)` plus the transform's log-Jacobian, `-1.3862943611198906` at the initial point, as with float bounds.

**Suggested fix:** cast the bounds to floatX when the distribution is built (or compute the width after the cast) in
`Uniform.dist` / the interval transform.

**Not verified:** other PyMC versions; other integer-bounded distributions that subtract their bounds (UNVERIFIED).

---

## (b) JAX: `jax.scipy.special.erfcx` returns 0 on a band, in float64 and in float32

**Fixed upstream: no longer needed.** JAX 0.12.0.1 is right in the band, value and gradient (float64 relative error
3e-16, float32 8e-8; checked 2026-11-10). The draft is kept as the record of the bug in 0.11.2 and earlier.

**Draft title:** `erfcx` returns exactly 0 (and a wrong gradient) just below the point where it switches to its asymptotic branch

**Versions:** jax 0.11.2, CPU backend (the same code runs on the GPU backend; the GPU was not tried for this draft, UNVERIFIED).

**Reproducer:**

```python
import sys
import numpy as np
import scipy.special as sps

x64 = sys.argv[1] == "64"
import jax
jax.config.update("jax_enable_x64", x64)
import jax.numpy as jnp
import jax.scipy.special as jss

dt = jnp.float64 if x64 else jnp.float32
print("jax", jax.__version__, jax.default_backend(), "x64", x64)
for x in ([26.0, 26.54, 26.6, 26.64, 26.65] if x64 else [9.1, 9.19, 9.3, 9.42, 9.43]):
    print(x, float(jss.erfcx(dt(x))), "scipy", sps.erfcx(x))

lo, hi = (26.4, 26.8) if x64 else (9.0, 9.6)
xs = np.linspace(lo, hi, 400001).astype(np.float64 if x64 else np.float32)
v = np.asarray(jss.erfcx(jnp.asarray(xs, dt)))
bad = np.abs(v - sps.erfcx(xs.astype(np.float64))) > 1e-3 * np.abs(sps.erfcx(xs.astype(np.float64)))
print("wrong on [%.5f, %.5f]" % (xs[bad].min(), xs[bad].max()), "values there:", np.unique(v[bad]))
print("grad in the band:", float(jax.grad(jss.erfcx)(dt(26.6 if x64 else 9.3))))
```

**Output, `python repro.py 64` (confirmed):**

```
jax 0.11.2 cpu x64 True
26.0 0.021683584850562918 scipy 0.02168358485056291
26.54 0.021243027203158172 scipy 0.021243027203158175
26.6 0.0 scipy 0.02119517815916648
26.64 0.0 scipy 0.021163398286570656
26.65 0.02115546819320206 scipy 0.02115546819320206
wrong on [26.54326, 26.64175] values there: [0.]
grad in the band: -1.1283791670955126
```

**Output, `python repro.py 32` (confirmed):**

```
jax 0.11.2 cpu x64 False
9.1 0.06163109093904495 scipy 0.06163109560008546
9.19 0.06103450059890747 scipy 0.06103450322212491
9.3 0.0 scipy 0.06032075148210422
9.42 0.059560809284448624 scipy 0.05956081152457974
9.43 0.059498339891433716 scipy 0.05949834198199045
wrong on [9.19455, 9.41928] values there: [0.]
grad in the band: -1.128379225730896
```

So `erfcx` is exactly 0 on about [26.54, 26.64] in float64 and on [9.19455, 9.41928] in float32 (true values about 0.0212 and
0.0553 to 0.0603). The gradient is `-1.128` (that is `-2/sqrt(pi)`, the value for `erfcx = 0`) where the true derivative
is about `-8.05e-4` (`2 x erfcx(x) - 2/sqrt(pi)` with the true `erfcx`). Outside the band the values agree with SciPy
to the last digits shown.

**Cause (from reading the code, not from a debugger).** The edges match `[erfcinv(finfo.tiny), sqrt(log(finfo.max))]`
(float64: 26.5433 and 26.6417; float32: 9.1946 and 9.4193), which is what the draft attributes the band to: `exp(x**2) * erfc(x)`
is used up to `sqrt(log(finfo.max))` while `lax.erfc` is already flushed to 0 above `erfcinv(finfo.tiny)`. The edges
agree to the digits measured; the mechanism itself is UNVERIFIED against JAX's source in this draft.

**Suggested fix:** switch to the asymptotic branch at `erfcinv(finfo.tiny)` instead of at `sqrt(log(finfo.max))`.
This package's guard (`pfcheeshmc/lifted.py:_erfcx_guarded`) does that with a nine-term series, and on a 5.6-million-point
float32 grid from -20 to 1e5 it is within 1e-5 of SciPy everywhere (measured by the F2k task, probe kept at the tag
`archive/pre-cleanup`; not re-run here, UNVERIFIED today).

**Not verified:** the GPU backend, float16 and bfloat16 (JAX upcasts them to float32 inside `erfcx`, so they should share the
float32 band), and other JAX versions.
