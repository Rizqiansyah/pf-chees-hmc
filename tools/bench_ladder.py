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
the draw-covariance condition number (mass-matrix difficulty proxy, unit-dependent),
and whether sample_stats.energy contains non-finite values.

CPU wall time includes tuning. GPU wall time includes Pathfinder, ChEES warmup and any
compilation, so cold and warm must never be mixed in a comparison.

Capacity is part of the result, not an accident: at d=175 (radon) 128 chains does not fit the 6 GB
RTX A2000 — JAX fails allocating 699 MiB in a dot_general — while 32 chains runs comfortably. So a
chain-count sweep has to report where the device stops, not extrapolate past it.
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


# A synthetic ill-conditioned MvNormal family was here and is deleted on purpose: with a dense
# `cov` over many observations the jaxed logp allocated 4.63 GiB under 32 vmapped chains at d=50,
# and the cheap n_obs=1 variant was unusable as a comparison because CPU NUTS failed it too
# (R-hat 1.33). Stiffness comes from real models instead.

PDB_RAW = "https://raw.githubusercontent.com/stan-dev/posteriordb/master/posterior_database"
PDB_CACHE = Path.home() / ".cache" / "pf-ladder"


def pdb_data(name: str):
    """Fetch one PosteriorDB dataset (json.zip) and cache it under $HOME."""
    import io
    import urllib.request
    import zipfile

    PDB_CACHE.mkdir(parents=True, exist_ok=True)
    out = PDB_CACHE / f"{name}.json"
    if not out.exists():
        raw = urllib.request.urlopen(f"{PDB_RAW}/data/data/{name}.json.zip", timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            out.write_bytes(z.read(f"{name}.json"))
    return json.loads(out.read_text())


def build_radon(noncentered: bool):
    """Radon hierarchical slope/intercept model, posterior_database radon_mn pair (d=175, N=919).

    The two parameterisations differ only in how the county effects are drawn, which makes this the
    conditioning A/B: `centered` is the funnel where the group size is not identified, and
    Seyboldt et al. excluded every centered variant of it for non-convergence. Ported directly from
    the database's Stan source: half-normal(0,1) on the three sigmas, normal(0,10) on the two means,
    county intercepts and slopes normal(mu, sigma), y ~ normal(alpha[c] + floor*beta[c], sigma_y).
    """
    import pymc as pm

    d = pdb_data("radon_mn")
    idx = np.asarray(d["county_idx"]) - 1          # Stan is 1-based
    floor = np.asarray(d["floor_measure"], dtype=float)
    y = np.asarray(d["log_radon"], dtype=float)
    J = int(d["J"])

    with pm.Model() as model:
        sigma_y = pm.HalfNormal("sigma_y", 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)
        sigma_beta = pm.HalfNormal("sigma_beta", 1.0)
        mu_alpha = pm.Normal("mu_alpha", 0.0, 10.0)
        mu_beta = pm.Normal("mu_beta", 0.0, 10.0)
        if noncentered:
            alpha_raw = pm.Normal("alpha_raw", shape=(J,))
            beta_raw = pm.Normal("beta_raw", shape=(J,))
            alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_raw)
            beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_raw)
        else:
            alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=(J,))
            beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=(J,))
        pm.Normal("obs", mu=alpha[idx] + floor * beta[idx], sigma=sigma_y, observed=y)
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
        "logit_1000x25": lambda: build_logistic(1000, 25),
        "logit_10000x50": lambda: build_logistic(10000, 50),
        "radon_centered": lambda: build_radon(False),
        "radon_noncentered": lambda: build_radon(True),
    }


# --------------------------------------------------------------------------- metrics
def score(idata, wall, backend, model, chains, cold, drop_vars=()):
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
    # Conditioning proxy for the mass matrix: the sampler's inverse mass matrix is built from
    # Pathfinder draws, so the posterior draw covariance's condition number says what the metric
    # has to cope with. Unit-dependent by construction -- compare it across models in the same
    # units, never as an absolute. Deterministics are excluded: they are affine copies of sampled
    # variables here, and that exact collinearity shows up as a huge (fake) condition number.
    try:
        flat = np.concatenate(
            [np.asarray(idata.posterior[v].values).reshape(-1, int(np.prod(
                idata.posterior[v].values.shape[2:] or (1,))))
             for v in idata.posterior.data_vars if v not in set(drop_vars)], axis=1)
        if len(flat) > 20_000:
            flat = flat[:: max(len(flat) // 20_000, 1)]   # strided, so the sample spans all chains
        if flat.shape[1] > 1 and np.all(np.isfinite(flat)):
            ev = np.linalg.eigvalsh(np.cov(flat, rowvar=False))
            row["cov_cond"] = round(float(ev[-1] / max(ev[0], 1e-300)), 1)
    except Exception as exc:                                    # never lose a timing row to a diagnostic
        row["cov_cond_error"] = f"{type(exc).__name__}: {exc}"[:160]
    if backend == "gpu":
        row["imm"] = os.environ.get("IMM", "mc")
        if row["imm"] != "mc":
            row["rank"] = int(os.environ.get("MAXRANK", "10"))
            row["cutoff"] = float(os.environ.get("FISHER_CUTOFF", "2.0"))
    if "energy" in idata.sample_stats:
        # E-BFMI (Betanin et al. / Vehtari et al.), threshold 0.3. BlackJAX stores the *proposal*
        # state's energy (blackjax/mcmc/hmc.py:171), so a divergent transition writes +inf and the
        # plain arviz call returns NaN -- which is why GEV had no BFMI at all. Non-finite entries
        # are dropped before differencing; that creates one artificial adjacency per dropped draw,
        # which biases E-BFMI upward (looks healthier), so divergence counts are reported too.
        e = np.asarray(idata.sample_stats["energy"].values, dtype=float)
        e = e.reshape(e.shape[0], e.shape[1], -1)[..., 0]      # (chain, draw)
        row["energy_nonfinite"] = int((~np.isfinite(e)).sum())
        bfmi = []
        for c in range(e.shape[0]):
            ec = e[c][np.isfinite(e[c])]
            if ec.size > 10:
                var_e = float(np.var(ec))
                if var_e > 0:
                    bfmi.append(float(np.var(np.diff(ec)) / var_e))
        row["ebfmi"] = round(min(bfmi), 4) if bfmi else None
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
        pathfinder_kwargs={"imm_source": os.environ.get("IMM", "mc"),
                           "max_rank": int(os.environ.get("MAXRANK", "10")),
                           "fisher_gamma": float(os.environ.get("FISHER_GAMMA", "1e-5")),
                           "fisher_cutoff": float(os.environ.get("FISHER_CUTOFF", "2.0"))},
    )
    wall = time.perf_counter() - t0
    return score(idata, wall, "gpu", label, chains, cold,
                 drop_vars=tuple(d.name for d in model_obj.deterministics))


def run_cpu(model_obj, label, chains, seed, cold):
    t0 = time.perf_counter()
    with model_obj:
        idata = REAL_PM_SAMPLE(draws=CPU_DRAWS, tune=CPU_DRAWS, chains=chains, cores=chains,
                               target_accept=0.8, random_seed=seed, progressbar=False)
    wall = time.perf_counter() - t0
    return score(idata, wall, "cpu_nuts", label, chains, cold,
                 drop_vars=tuple(d.name for d in model_obj.deterministics))


def main():
    OUT.parent.mkdir(exist_ok=True, parents=True)
    specs = model_specs()

    def emit(row):
        print(json.dumps(row), flush=True)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    for label in os.environ.get("MODELS", "8schools,gev").split(","):
        build = specs[label.strip()]
        for i, seed in enumerate(SEEDS):
            model_obj = build()
            cold = i == 0
            # A cell that dies must not end a 90-minute sweep: the device ceiling (d=175 at 128
            # chains OOMs on 6 GB) is a result to record, not a crash to propagate.
            for arm, chain_list, runner in (("gpu", GPU_CHAINS, run_gpu),
                                            ("cpu", CPU_CHAINS, run_cpu)):
                if arm not in ARMS:
                    continue
                for chains in chain_list:
                    try:
                        emit(runner(model_obj, label, chains, seed, cold))
                    except Exception as exc:
                        emit({"backend": arm, "model": label, "chains": chains, "seed": seed,
                              "cold": cold, "error": f"{type(exc).__name__}: {exc}"[:200]})


if __name__ == "__main__":
    main()
