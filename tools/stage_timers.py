"""O1: attribute the wall time that is NOT model evaluation.

Wraps each PfCheesHMC stage with a wall-clock timer (blocking on the GPU output before the timer
stops) and reports the per-stage split. Run per model at the 40,000-draw budget config; the first
config per process carries the compile (cold), so report two consecutive configs and read the
second.

Stage list (pfcheeshmc/sampler.py PfCheesHMC):
  pymc_init         _pymc_adapt_diag_start   pm.init_nuts(adapt_diag, chains, tune=500) on CPU
  finite_init       _ensure_finite_inits     per-chain repair
  pathfinder        _run_pathfinder          L-BFGS fit + draws + covariance
  step_size_tune    _run_step_size_tune      the W1 vmapped search
  chees_warmup      _run_chees               2000 lockstep adaptation steps
  dynamic_hmc       _run_dynamic_hmc         the sampling scan
  (residual)        everything else in sample() + InferenceData build
"""
import os
import numpy as np
import sys
import time

REPO = "/home/akbar/Documents/Python packages/pf-chees-hmc"
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)

import jax
import bench_ladder as bl
import pfcheeshmc.sampler as sm

TIMINGS = {}
STAGES = ("pymc_init", "finite_init", "pathfinder", "step_size_tune", "chees_warmup", "dynamic_hmc")
M = {
    "pymc_init": "_pymc_adapt_diag_start",
    "finite_init": "_ensure_finite_inits",
    "pathfinder": "_run_pathfinder",
    "step_size_tune": "_run_step_size_tune",
    "chees_warmup": "_run_chees",
    "dynamic_hmc": "_run_dynamic_hmc",
}


def wrap(name, fn):
    def inner(self, *a, **k):
        t0 = time.perf_counter()
        out = fn(self, *a, **k)
        first = out[0] if isinstance(out, tuple) else out
        jax.block_until_ready(first)
        TIMINGS[name] = TIMINGS.get(name, 0.0) + (time.perf_counter() - t0)
        return out
    return inner


for name, meth in M.items():
    setattr(sm.PfCheesHMC, meth, wrap(name, getattr(sm.PfCheesHMC, meth)))


def run(model_name, chains, draws, warmup, seed):
    model = (bl.load_notebook_model(bl.REPO / "examples" / "8 Schools example.ipynb")
             if model_name == "8schools" else bl.build_radon(True))
    TIMINGS.clear()
    t0 = time.perf_counter()
    idata = bl.REAL_SAMPLE_PF(model, draws=draws, chains=chains, random_seed=seed,
                              chees_kwargs={"draws": warmup})
    first = next(iter(idata.posterior.values()))
    jax.block_until_ready(first.values)
    total = time.perf_counter() - t0
    print(f"\n=== {model_name}  chains={chains} draws={draws} warmup={warmup} seed={seed}  "
          f"total={total:.2f}s")
    acc = 0.0
    for stage in STAGES:
        t = TIMINGS.get(stage, 0.0)
        acc += t
        print(f"  {stage:<15} {t:8.2f} s   {100 * t / total:5.1f}%")
    print(f"  {'residual':<15} {total - acc:8.2f} s   {100 * (total - acc) / total:5.1f}%   "
          f"(sample glue + InferenceData export)")
    import arviz as _az
    _post = idata.posterior
    _r = max(float(np.nanmax(_az.rhat(_post)[v].values)) for v in _post.data_vars)
    _e = min(float(np.nanmin(_az.ess(_post, method="bulk")[v].values)) for v in _post.data_vars)
    print(f"  {'quality':<15} max_rhat={_r:.4f} min_bulk_ess={_e:.0f} "
          f"ess_s_worstvar={_e / total:.0f}")
    return total


if __name__ == "__main__":
    model = os.environ.get("MODEL", "8schools")
    chains = int(os.environ.get("CHAINS", "32"))
    draws = int(os.environ.get("DRAWS", "1250"))
    warmup = int(os.environ.get("WARMUP", "2000"))
    seed = int(os.environ.get("SEED", "42"))
    run(model, chains, draws, warmup, seed)      # cold: carries compile
    for s in range(seed + 1, seed + 1 + int(os.environ.get("WARM_SEEDS", "1"))):
        run(model, chains, draws, warmup, s)     # warm: read these
