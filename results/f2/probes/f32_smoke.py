import os, sys, time, traceback
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "tools")]
import r1_bias, jax, numpy as np, pytensor
from pfcheeshmc import sample_pf_chees_hmc
print("floatX", pytensor.config.floatX, "x64", jax.config.jax_enable_x64, flush=True)
for name in sys.argv[2:]:
    m = r1_bias.build_model(name)
    try:
        t = time.perf_counter()
        idata = sample_pf_chees_hmc(m, draws=10, chains=64, random_seed=1, progressbar=False, chees_kwargs={"draws": 50})
        post = idata.posterior
        dts = sorted({str(post[v].dtype) for v in post.data_vars})
        fin = all(np.isfinite(post[v].values).all() for v in post.data_vars)
        div = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else None
        print(f"{name}: OK {time.perf_counter()-t:.1f}s dtypes={dts} finite={fin} divergences={div}", flush=True)
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:300]}", flush=True)
        tb = traceback.extract_tb(e.__traceback__)
        for fr in tb[-6:]:
            print(f"    {os.path.relpath(fr.filename)}:{fr.lineno} {fr.name}")
