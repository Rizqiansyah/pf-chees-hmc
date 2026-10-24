"""F2k: the non-default Pathfinder and warmup options under floatX=float32 (or float64), at a small shape.

    python f32_variants.py TREE MODEL [MODEL ...]

One sampler call per variant (16 chains, 10 draws, warmup 30); prints finished/finite/dtype/divergences, or the error.
Variants: imm_source pf_closed / lbfgs_lowrank / fisher_lowrank, imm_orientation precision, init_pick iid,
metric_refresh fisher_diag / sample_diag / dense_shrunk / diag_lowrank, warmup_stop, pm_init jitter+adapt_diag with
pm_chains 4, pf_start_scale 0.5, iterate raw.
"""
import os, sys, time, warnings
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "tools")]
import numpy as np, jax, pytensor
import r1_bias
from pfcheeshmc import sample_pf_chees_hmc

VARIANTS = {
    "default": ({}, {}),
    "pf_closed": ({"imm_source": "pf_closed"}, {}),
    "lbfgs_lowrank": ({"imm_source": "lbfgs_lowrank"}, {}),
    "fisher_lowrank": ({"imm_source": "fisher_lowrank"}, {}),
    "precision": ({"imm_orientation": "precision"}, {}),
    "init_pick iid": ({"init_pick": "iid"}, {}),
    "start_scale 0.5": ({"pf_start_scale": 0.5}, {}),
    "jitter pm_chains 4": ({"pm_init": "jitter+adapt_diag", "pm_chains": 4}, {}),
    "refresh fisher_diag": ({}, {"metric_refresh": "fisher_diag", "metric_refresh_warmup": 15}),
    "refresh sample_diag": ({}, {"metric_refresh": "sample_diag", "metric_refresh_warmup": 15}),
    "refresh dense_shrunk": ({}, {"metric_refresh": "dense_shrunk", "metric_refresh_warmup": 15}),
    "refresh diag_lowrank": ({}, {"metric_refresh": "diag_lowrank", "metric_refresh_warmup": 15}),
    "iterate raw": ({}, {"iterate": "raw"}),
    "warmup_stop": ({}, {"warmup_stop": {"block": 10, "window": 10, "rhat": 0.0}}),
}
print("floatX", pytensor.config.floatX, "x64", jax.config.jax_enable_x64)
for name in sys.argv[2:]:
    model = r1_bias.build_model(name)
    for label, (pf, ch) in VARIANTS.items():
        t = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                idata = sample_pf_chees_hmc(model, draws=10, chains=16, random_seed=1, progressbar=False,
                                            pathfinder_kwargs=pf, chees_kwargs={"draws": 30, **ch})
            post = idata.posterior
            fin = all(np.isfinite(post[v].values).all() for v in post.data_vars)
            dts = sorted({str(post[v].dtype) for v in post.data_vars})
            div = int(idata.sample_stats["diverging"].values.sum())
            dtype_warn = sum("Explicitly requested dtype" in str(w.message) for w in caught)
            print(f"{name:18s} {label:22s} OK {time.perf_counter()-t:5.1f}s finite={fin} dtypes={dts} div={div} dtype_warnings={dtype_warn}", flush=True)
        except Exception as e:
            print(f"{name:18s} {label:22s} FAIL {type(e).__name__}: {(str(e).splitlines() or [''])[0][:160]}", flush=True)
