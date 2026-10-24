"""F2k: why a float32 GEV Pathfinder pool has rows at -inf where float64 has none.

    python f32_pool_inf_rows.py TREE record OUT.npz     # floatX=float32: the pool rows whose log-density is not finite
    python f32_pool_inf_rows.py TREE evaluate IN.npz    # floatX=float64: the log-density at those rows, as float64

`record` runs `sample_pf_chees_hmc` on GEV (8 chains, 10 draws, warmup 20, seed 3), wraps `compiled.score_pool` and keeps
every pool row that scores non-finite. `evaluate` scores the same coordinates (float32 values, widened) in float64:
finite and below float32's range (-3.4e38) means the float32 value is an overflow, not a different point.
"""
import os, sys
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "tools")]
import numpy as np, jax, jax.numpy as jnp, pytensor
import r1_bias
from pfcheeshmc import compiled, lifted, sample_pf_chees_hmc

model = r1_bias.build_model("gev")
names = [v.name for v in model.value_vars]
if sys.argv[2] == "record":
    rows = []
    real = compiled.score_pool
    def spy(fn, names_, values, pool, idx):
        out = real(fn, names_, values, pool, idx)
        bad = np.flatnonzero(~np.isfinite(np.asarray(out)))
        rows.append(np.stack([np.asarray(p)[np.asarray(idx)[bad]] for p in pool], axis=1))
        return out
    compiled.score_pool = spy
    sample_pf_chees_hmc(model, draws=10, chains=8, random_seed=3, progressbar=False, chees_kwargs={"draws": 20})
    bad = np.concatenate(rows)
    print("floatX", pytensor.config.floatX, "pool rows with a non-finite log-density:", len(bad))
    np.savez(sys.argv[3], rows=bad.astype(np.float64))
else:
    x = np.load(sys.argv[3])["rows"]
    lg = lifted.lift_logp(model)
    lp = np.asarray(jax.jit(jax.vmap(lambda r: lg.fn(tuple(r[i] for i in range(r.shape[0])), lg.values)[0]))(jnp.asarray(x)))
    print("floatX", pytensor.config.floatX, "rows", len(x), "float64 log-density at them:", lp)
    print("coordinates", names, x)
