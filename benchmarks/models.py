"""The four ladder models, built in Python: `build_model(name)` for every name in `MODELS`.

    8schools            8 Schools, theta = mu + tau * (standard-normal offsets) (d=10)
    gev                 the Port Pirie annual-maximum sea levels, a GEV likelihood (d=3, a Deterministic return level)
    logit_1000x25       Bernoulli logistic regression, 1,000 observations, 25 correlated predictors
    radon_noncentered   the posteriordb radon_mn pair, county effects non-centered (d=175, N=919)

The graphs are those the two example notebooks and `bench_ladder` built before: `8schools` and `gev` are the model cells
of the 8 Schools and the GEV Port Pirie notebooks in `examples/`, transcribed (data, priors, names, shapes, creation
order); `logit_1000x25` and `radon_noncentered` are `bench_ladder.build_logistic` and `build_radon`, moved here. `tests/test_benchmark_models.py` holds the proof (value variables, `logp` and `dlogp` at 100 points, the
graph, and the draws of `sample_pf_chees_hmc`, against the notebook-built models of the tag `archive/pre-cleanup`).

Building a model sets no JAX option and no environment variable: not `jax_enable_x64` (PyTensor's JAX dispatch sets it
from floatX when it is imported), not the platform, not the compilation cache. A caller that wants float32 sets
``floatX`` before PyTensor is imported, and the models follow it.

`radon_noncentered` downloads the posteriordb `radon_mn` data on first use (~/.cache/pf-ladder). The data are not part
of this repository.
"""
import json
from pathlib import Path

import numpy as np
import pymc as pm

PDB_RAW = "https://raw.githubusercontent.com/stan-dev/posteriordb/master/posterior_database"
PDB_CACHE = Path.home() / ".cache" / "pf-ladder"


def build_8schools():
    """8 Schools: ``theta`` holds the standard-normal offsets of the schools, the school effect is ``mu + tau * theta``."""
    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        tau = pm.HalfCauchy("tau", 5.0)

        theta = pm.Normal("theta", mu=0, sigma=1, shape=J)
        theta_1 = mu + tau * theta
        obs = pm.Normal("obs", mu=theta_1, sigma=sigma, shape=J, observed=y)
    return model


def build_gev():
    """Port Pirie annual maximum sea levels (65 years), a GEV likelihood and the 10-year return level."""
    import pymc_extras as pmx

    # fmt: off
    data = np.array([4.03, 3.83, 3.65, 3.88, 4.01, 4.08, 4.18, 3.80,
                     4.36, 3.96, 3.98, 4.69, 3.85, 3.96, 3.85, 3.93,
                     3.75, 3.63, 3.57, 4.25, 3.97, 4.05, 4.24, 4.22,
                     3.73, 4.37, 4.06, 3.71, 3.96, 4.06, 4.55, 3.79,
                     3.89, 4.11, 3.85, 3.86, 3.86, 4.21, 4.01, 4.11,
                     4.24, 3.96, 4.21, 3.74, 3.85, 3.88, 3.66, 4.11,
                     3.71, 4.18, 3.90, 3.78, 3.91, 3.72, 4.00, 3.66,
                     3.62, 4.33, 4.55, 3.75, 4.08, 3.90, 3.88, 3.94,
                     4.33])
    # fmt: on
    p = 1 / 10  # return period
    with pm.Model() as model:
        # Priors
        μ = pm.Normal("μ", mu=3.8, sigma=0.2)
        σ = pm.HalfNormal("σ", sigma=0.3)
        ξ = pm.TruncatedNormal("ξ", mu=0, sigma=0.2, lower=-0.6, upper=0.6)

        # Likelihood
        gev = pmx.GenExtreme(
            "gev",
            mu=μ,
            sigma=σ,
            xi=ξ,
            observed=data
        )
        # Return level
        z_p = pm.Deterministic("z_p", μ - σ / ξ * (1 - (-np.log(1 - p)) ** (-ξ)))
    return model


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


_BUILDERS = {
    "8schools": build_8schools,
    "gev": build_gev,
    "logit_1000x25": lambda: build_logistic(1000, 25),
    "radon_noncentered": lambda: build_radon(True),
}
MODELS = tuple(_BUILDERS)


def build_model(name):
    """A fresh PyMC model of the ladder: one of `MODELS`. Every call builds a new model (new variables)."""
    if name not in _BUILDERS:
        raise KeyError(f"unknown ladder model {name!r}; the ladder is {MODELS}")
    return _BUILDERS[name]()
