"""Shared by the test modules: the extra models, the literal models, a point and the PyMC logp, the mixing instrument.

(Fixtures are in `conftest.py`.) Not a test module; no test module imports from another one.
"""
import arviz as az
import arviz_stats
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pfcheeshmc.sampler import sample_pf_chees_hmc

LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")


# ---------------------------------------------------------------- models drawn from a seed

def m_pm_data(seed):
    """pm.Data for the observed data; floats written as floats."""
    rng = np.random.default_rng(seed)
    with pm.Model() as m:
        y = pm.Data("y", rng.normal(rng.uniform(1, 3), 1.0, size=30))
        mu = pm.Normal("mu", float(rng.uniform(-1, 1)), float(rng.uniform(2, 6)))
        sigma = pm.HalfNormal("sigma", float(rng.uniform(1, 3)))
        pm.Normal("obs", mu, sigma, observed=y)
    return m


def m_int_priors(seed):
    """Integer-written priors, pm.Normal("mu", 0, 10): int8 constants, direct inputs of the RV node."""
    rng = np.random.default_rng(seed)
    with pm.Model() as m:
        mu = pm.Normal("mu", int(rng.integers(0, 5)), int(rng.integers(5, 100)))
        sd = pm.HalfNormal("sd", int(rng.integers(1, 10)))
        pm.Normal("obs", mu, sd, observed=rng.normal(size=25))
    return m


def m_mvnormal(seed):
    """MvNormal with a constant covariance matrix (a Blockwise Cholesky/solve consumer)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + 3.0 * np.eye(3)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0.0, 2.0, shape=3)
        pm.MvNormal("obs", mu=mu, cov=cov, observed=rng.normal(size=(12, 3)))
    return m


def m_potential(seed):
    """A pm.Potential that uses a data array (a weight vector)."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=5)
    with pm.Model() as m:
        x = pm.Normal("x", 0.0, 1.0, shape=5)
        pm.Potential("pot", -pt.sum(w * x**2))
    return m


def m_poisson(seed):
    """Poisson GLM with integer observations (int64 data) and a float design matrix."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 3))
    y = rng.poisson(np.exp(0.3 * X[:, 0] + 0.5))
    with pm.Model() as m:
        beta = pm.Normal("beta", 0.0, 1.0, shape=3)
        b0 = pm.Normal("b0", 0.0, 1.0)
        pm.Poisson("y", pt.exp(pt.dot(X, beta) + b0), observed=y)
    return m


def m_truncnorm(seed):
    """TruncatedNormal with a free mu: its logp has log(erfc(.)) in a parameter-dependent normaliser,
    which PyTensor rewrites to `erfcx`. The old JAX path has no `erfcx` without TFP."""
    rng = np.random.default_rng(seed)
    with pm.Model() as m:
        mu = pm.Normal("mu", float(rng.uniform(-0.5, 0.5)), 1.0)
        pm.TruncatedNormal("x", mu=mu, sigma=float(rng.uniform(0.8, 1.2)), lower=0.0,
                           observed=np.abs(rng.normal(size=20)) + 0.05)
    return m


EXTRA = {
    "pm_data": m_pm_data,
    "int_priors": m_int_priors,
    "mvnormal": m_mvnormal,
    "potential": m_potential,
    "poisson": m_poisson,
    "truncnorm": m_truncnorm,
}




def literal_models():
    """One model per kind of prior literal: shape () and shaped RVs (whose parameter reaches the logp as a
    folded one-element constant), float- and int-valued RVs, and a SymbolicRV."""
    def scalar(mu, sd):
        with pm.Model() as m:
            x = pm.Normal("mu", mu, sd)
            pm.Normal("obs", x, 1.0, observed=np.linspace(-1, 1, 10))
        return m

    def shaped(mu, sd):
        with pm.Model() as m:
            x = pm.Normal("x", mu, sd, shape=3)
            pm.Normal("obs", x.sum(), 1.0, observed=np.linspace(-1, 1, 10))
        return m

    def poisson(rate, n):
        with pm.Model() as m:
            lam = pm.Gamma("lam", rate, 1.0)
            pm.Poisson("c", lam, observed=np.array([1, 4, 2]))
            pm.Binomial("k", n=n, p=0.3, observed=np.array([3, 5, 7]))
        return m

    def truncated(mu, sd):
        with pm.Model() as m:
            a = pm.Normal("a", mu, sd)
            pm.TruncatedNormal("t", mu=a, sigma=1.0, lower=-0.37, observed=np.abs(np.linspace(-1, 1, 7)) + 0.1)
        return m

    return {"scalar": scalar, "shaped": shaped, "poisson": poisson, "truncated": truncated}


# Literal pairs that PyTensor autocasts to different dtypes: int8 -> float32 -> float64 -> int16 -> int32.
# Equal constants are merged by PyMC's own logp rewrites, so a prior literal that equals another constant
# of the model (mu=0 next to lower=0, or a formula constant) shares one graph node with it and gives a
# different program once it stops being equal. The models below keep their other literals clear of these.
LITERALS = [(0, 10), (0.0, 10.0), (2.5, 2.7), (0.5, 3.0), (0.1, 10.1), (1, 1000), (100, 40000), (-3, 12.5)]
# (rate of a Gamma, n of a Binomial): an int-valued RV's integer parameter stays an integer
LITERALS_INT_RV = [(2.5, 10), (0.1, 1000), (7, 40000), (-3, 12), (3.7, 5), (0.75, 99)]




def point(model, seed=0):
    ip = model.initial_point()
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(ip[v.name] + 0.3 * rng.standard_normal(np.shape(ip[v.name]))) for v in model.value_vars)


def pymc_logp(model, x):
    return float(model.compile_logp()({v.name: np.asarray(a) for v, a in zip(model.value_vars, x)}))


# PF-order variances for the injection tests: z wide, a tight.
PF_ORDER_VAR = np.array([1e4, 1e-4, 1e-4, 1e-4])
KERNEL_ORDER_VAR = np.array([1e-4, 1e-4, 1e-4, 1e4])   # sorted: a, a, a, z


def mixing_report(anisotropic_model, **pathfinder_kwargs):
    model, mean, var = anisotropic_model
    kwargs = {"draws": 256}
    kwargs.update(pathfinder_kwargs)
    idata = sample_pf_chees_hmc(model, draws=1000, chains=32, random_seed=5, progressbar=False,
                                pathfinder_kwargs=kwargs, chees_kwargs={"draws": 500})
    max_rhat = max(float(np.max(v.values)) for v in az.rhat(idata.posterior).values())
    ess = az.ess(idata.posterior, method="bulk", relative=False)
    per = {n: float(np.min(ess[n].values)) for n in var}
    steps = float(np.mean(idata.sample_stats["num_integration_steps"].values))
    min_ess = min(per.values())
    # arviz_stats, not arviz: arviz 0.23's _mcse_sd raises "only 0-dimensional arrays can be
    # converted to Python scalars" under numpy 2.4 on this box (verified). A6 says use arviz_stats.
    mcse_mean = arviz_stats.mcse(idata, method="mean")
    return idata, max_rhat, min_ess, per, steps, mcse_mean, mean, var


