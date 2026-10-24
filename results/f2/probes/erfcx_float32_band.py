"""F2k: JAX's erfcx under float32 against scipy.special.erfcx, raw and through `lifted._erfcx_guarded`.

    python erfcx_float32_band.py TREE

Run in a process where x64 is OFF (PYTENSOR_FLAGS=floatX=float32, or no PyTensor import before JAX is used).
The reference is SciPy in float64 at the same float32 inputs, rounded to float32 (so it overflows where float32 does).
Prints, per range, the largest relative error (finite reference only) and where JAX's own function returns 0 or a
non-finite value that the reference does not; then the same for the guarded function.
"""
import os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import scipy.special as sps
import jax, jax.numpy as jnp, jax.scipy.special as jss
from pfcheeshmc import lifted
print("x64", jax.config.jax_enable_x64)

grid = np.unique(np.concatenate([
    np.linspace(-20, 30, 5_000_001), np.geomspace(30, 1e5, 20_001),
    np.linspace(9.0, 9.6, 600_001), np.linspace(26.4, 26.8, 1_000)])).astype(np.float32)
ref = sps.erfcx(grid.astype(np.float64)).astype(np.float32)
for name, f in (("jax", jss.erfcx), ("guarded", lifted._erfcx_guarded)):
    got = np.asarray(jax.jit(f)(jnp.asarray(grid)))
    ok = np.isfinite(ref) & (ref > np.finfo(np.float32).tiny)
    rel = np.abs(got[ok] - ref[ok]) / ref[ok]
    bad = ok & ~(np.abs(got - ref) <= 1e-5 * np.abs(ref))
    print(f"{name:8s} n={grid.size} compared={ok.sum()} max rel err {rel.max():.3e} at x={grid[ok][rel.argmax()]:.6g}; "
          f"points off by more than 1e-5 relative: {bad.sum()}")
    if bad.any():
        xs = grid[bad]
        print(f"         first {xs.min():.6f} last {xs.max():.6f}; got at first {got[bad][0]:.4g} ref {ref[bad][0]:.4g}")
    for lo, hi in ((-20, -5), (-5, 0), (0, 5), (5, 9.19), (9.19, 9.43), (9.43, 30), (30, 1e5)):
        sel = ok & (grid >= lo) & (grid < hi)
        if sel.any():
            r = np.abs(got[sel] - ref[sel]) / ref[sel]
            print(f"         [{lo:>6}, {hi:>6})  max rel err {r.max():.3e}")
# the guard's band, as lifted computes it
print("band float32", lifted._band(np.dtype("float32")), "float64", lifted._band(np.dtype("float64")))
# float32 input at the float64 band (26.55-26.63) is far above float32's hi: asymptotic branch in JAX
sel = (grid > 26.5) & (grid < 26.7)
print("26.5-26.7 jax max rel err", np.max(np.abs(np.asarray(jax.jit(jss.erfcx)(jnp.asarray(grid[sel]))) - ref[sel]) / ref[sel]))
