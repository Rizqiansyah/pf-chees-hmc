"""E1 ladder harness: GPU vs CPU throughput as models get larger and stiffer.

Usage (all configurable by environment):
    pixi run python tools/bench_ladder.py
Environment:
    MODELS=8schools,gev          which models (see MODELS below)
    ARMS=gpu,cpu                 which backends
    GPU_CHAINS=32,128            chain counts for the GPU arm
    CPU_CHAINS=4,16,32           chain counts for the CPU arm (cores = chains, 1 thread each)
    DRAWS=2000 WARMUP=2000 CPU_DRAWS=1000 SEEDS=42,43,44
    LADDER_OUT=~/pf-runs/ladder.jsonl

Reported per cell: wall (cold and warm flagged), bulk ESS/s total and worst-variable,
min tail-ESS, max R-hat, divergences per 1000 draws, mean leapfrog steps, the padding ratio
(mean over draws of max/mean across chains of num_integration_steps — warp-divergence cost),
and whether sample_stats.energy contains non-finite values.

CPU wall time includes tuning. GPU wall time includes Pathfinder, ChEES warmup and any
compilation, so cold and warm must never be mixed in a comparison.
"""
import json
import os
import time
from pathlib import Path

# Must be set before numpy/pytensor import: the CPU arm's fairness depends on one thread per
# chain, so that "cores = chains" actually means what it says (matches the CPU curve protocol).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# This repo lives under a path containing a space ("Python packages"). PyTensor puts the env prefix
# into blas__ldflags as an unquoted -Wl,-rpath,... token, and its linker then dies with
# 'invalid token "packages/pf-chees-hmc/.pixi/envs/default/lib"' for any model that compiles a Dot.
# Overriding the flags without the rpath fixes it; numpy still supplies BLAS at runtime.
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")

import arviz as az
import numpy as np

import pfcheeshmc
import pymc as pm

# Captured before load_notebook_model() stubs them for notebook loading.
REAL_SAMPLE_PF = pfcheeshmc.sample_pf_chees_hmc
REAL_PM_SAMPLE = pm.sample

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("LADDER_OUT", str(Path.home() / "pf-runs" / "ladder.jsonl")))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
DRAWS = int(os.environ.get("DRAWS", "2000"))
WARMUP = int(os.environ.get("WARMUP", "2000"))
CPU_DRAWS = int(os.environ.get("CPU_DRAWS", "1000"))
GPU_CHAINS = [int(c) for c in os.environ.get("GPU_CHAINS", "32,128").split(",")]
CPU_CHAINS = [int(c) for c in os.environ.get("CPU_CHAINS", "4,16,32").split(",")]
ARMS = os.environ.get("ARMS", "gpu,cpu").split(",")

SKIP_PREFIX = ("plt.", "az.plot", "sns.", "%", "display(", "pmat.", "arviz.plot", ".plot(")


# --------------------------------------------------------------------------- models
def _tiny_idata(*args, **kwargs):
    """Stand-in for any sampling call a notebook makes while we are only loading its model."""
    import xarray as xr
    rng = np.random.default_rng(0)
    ds = xr.Dataset({"f": (("chain", "draw"), rng.normal(size=(2, 10)))})
    return az.InferenceData(posterior=ds)


def load_notebook_model(nb_path: str):
    """Executes a notebook's code cells with plotting removed and sampling stubbed, returns the model.

    Sampling must be stubbed, not merely skipped: the committed examples contain real
    `sample_pf_chees_hmc` / `pm.sample` / `pm.sample_posterior_predictive` calls, and executing them
    here would silently run a full sample just to load a model. If a data cell were skipped the
    model would end up with no observed RVs, which would silently measure prior sampling, so that is
    a hard error.
    """
    import pymc as pm
    import pfcheeshmc

    pm.sample = _tiny_idata
    pm.sample_prior_predictive = _tiny_idata
    pm.sample_posterior_predictive = _tiny_idata
    pfcheeshmc.sample_pf_chees_hmc = _tiny_idata

    nb = json.loads(Path(nb_path).read_text())
    ns = {"pm": pm, "np": np, "az": az}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        lines = ["" if ln.strip().startswith(SKIP_PREFIX) else ln for ln in src.splitlines()]
        try:
            code = "\n".join(lines)
            exec(compile(code, str(nb_path), "exec"), ns)
        except SyntaxError:
            exec(compile(src, str(nb_path), "exec"), ns)
    model = ns.get("model")
    if model is None or not getattr(model, "observed_RVs", []):
        raise RuntimeError(f"{nb_path}: no model or no observed RVs — a data cell did not execute")
    return model


def build_gaussian(d, cond, n_obs=2000, seed=7):
    """Unknown mean of an ill-conditioned multivariate normal, estimated from n_obs draws.

    Conditioning of the posterior is inherited from Sigma, so this is a stiffness test with a
    well-posed answer. It needs many observations: with one observation the posterior is so tiny
    and so anisotropic that even CPU NUTS fails, which makes the cell useless as a comparison.
    """
    import pymc as pm

    rng = np.random.default_rng(seed)
    scales = np.exp(np.linspace(0.0, np.log(cond), d))
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    sigma = (q * scales) @ q.T
    sigma = sigma @ sigma.T
    sigma = sigma / np.max(np.diag(sigma))          # unit-ish marginal variances, cond preserved
    y = rng.multivariate_normal(np.zeros(d), sigma, size=n_obs)

    with pm.Model() as model:
        mu = pm.Normal("mu", shape=(d,))
        pm.MvNormal("y", mu=mu, cov=sigma, observed=y, shape=(n_obs, d))
    return model


def build_logistic(n_obs, d, seed=11):
    """Bernoulli logistic regression with correlated predictors."""
    import pymc as pm

    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d)) / np.sqrt(d)
    x = rng.normal(size=(n_obs, d)) @ (np.eye(d) + a)
    beta = rng.normal(size=d) * 0.5
    p = 1.0 / (1.0 + np.exp(-(x @ beta)))
    y = rng.binomial(1, p)

    with pm.Model() as model:
        b = pm.Normal("b", shape=(d,))
        pm.Bernoulli("y", logit_p=x @ b, observed=y, shape=(n_obs,))
    return model


def model_specs():
    return {
        "8schools": lambda: load_notebook_model(REPO / "examples" / "8 Schools example.ipynb"),
        "gev": lambda: load_notebook_model(REPO / "examples" / "GEV Port Pirie example.ipynb"),
        "gauss_d50": lambda: build_gaussian(50, 1e4),
        "gauss_d500": lambda: build_gaussian(500, 1e4),
        "logit_1000x25": lambda: build_logistic(1000, 25),
        "logit_10000x50": lambda: build_logistic(10000, 50),
    }


# --------------------------------------------------------------------------- metrics
def score(idata, wall, backend, model, chains, cold):
    ess = az.ess(idata.posterior, method="bulk", relative=False)
    etail = az.ess(idata.posterior, method="tail", relative=False)
    rh = az.rhat(idata.posterior)
    tot = float(sum(float(np.sum(v.values)) for v in ess.values()))
    row = {
        "backend": backend, "model": model, "chains": chains, "cold": cold,
        "draws": DRAWS if backend == "gpu" else CPU_DRAWS, "warmup": WARMUP,
        "wall_s": round(wall, 2), "ess_s_total": round(tot / wall, 0),
        "ess_s_worstvar": round(min(float(np.min(v.values)) for v in ess.values()) / wall, 0),
        "min_tail_ess": round(min(float(np.min(v.values)) for v in etail.values()), 0),
        "max_rhat": round(float(max(float(np.max(v.values)) for v in rh.values())), 4),
        "divs_per_1k_draws": round(float(np.sum(idata.sample_stats["diverging"].values))
                                   / (idata.posterior.sizes["draw"] * chains) * 1000, 3),
        "energy_finite": bool(np.all(np.isfinite(np.asarray(idata.sample_stats["energy"].values,
                                                          dtype=float))))
        if "energy" in idata.sample_stats else None,
    }
    if "num_integration_steps" in idata.sample_stats:
        steps = np.asarray(idata.sample_stats["num_integration_steps"].values, dtype=float)
        steps = steps.reshape(steps.shape[0], steps.shape[1], -1)[..., 0]   # (chain, draw)
        row["steps_mean"] = round(float(steps.mean()), 1)
        row["padding_ratio"] = round(float(np.mean(steps.max(axis=0) / np.maximum(steps.mean(axis=0), 1e-9))), 3)
    return row


def run_gpu(model_obj, label, chains, seed, cold):
    t0 = time.perf_counter()
    idata = REAL_SAMPLE_PF(
        model_obj, draws=DRAWS, chains=chains, random_seed=seed,
        chees_kwargs={"draws": WARMUP},
    )
    wall = time.perf_counter() - t0
    return score(idata, wall, "gpu", label, chains, cold)


def run_cpu(model_obj, label, chains, seed, cold):
    t0 = time.perf_counter()
    with model_obj:
        idata = REAL_PM_SAMPLE(draws=CPU_DRAWS, tune=CPU_DRAWS, chains=chains, cores=chains,
                               target_accept=0.8, random_seed=seed, progressbar=False)
    wall = time.perf_counter() - t0
    return score(idata, wall, "cpu_nuts", label, chains, cold)


def main():
    OUT.parent.mkdir(exist_ok=True, parents=True)
    specs = model_specs()
    for label in os.environ.get("MODELS", "8schools,gev").split(","):
        build = specs[label.strip()]
        for i, seed in enumerate(SEEDS):
            model_obj = build()
            cold = i == 0
            if "gpu" in ARMS:
                for chains in GPU_CHAINS:
                    row = run_gpu(model_obj, label, chains, seed, cold)
                    print(json.dumps(row), flush=True)
                    with open(OUT, "a") as fh:
                        fh.write(json.dumps(row) + "\n")
            if "cpu" in ARMS:
                for chains in CPU_CHAINS:
                    row = run_cpu(model_obj, label, chains, seed, cold)
                    print(json.dumps(row), flush=True)
                    with open(OUT, "a") as fh:
                        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
