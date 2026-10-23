"""Models built from seeds at fixed shapes: "same structure, other values" is a real rebuild.

`hier(data_seed, prior_seed)` is an 8-schools-shaped hierarchical model. `data_seed` moves the data (an
observed array and a `pm.Data`); `prior_seed` moves the prior values (a location and two scales, written
as the floats a user would write). `J` is the one shape.
"""
import numpy as np
import pymc as pm


def hier(data_seed, prior_seed=None, J=8):
    prior_seed = data_seed if prior_seed is None else prior_seed
    data, prior = np.random.default_rng(data_seed), np.random.default_rng(1000 + prior_seed)
    y = data.normal(data.uniform(4.0, 12.0), 10.0, size=J)
    se = data.uniform(8.0, 18.0, size=J)
    with pm.Model() as m:
        se_data = pm.Data("se", se)
        mu = pm.Normal("mu", float(prior.uniform(-1.0, 1.0)), float(prior.uniform(4.0, 8.0)))
        tau = pm.HalfCauchy("tau", float(prior.uniform(3.0, 7.0)))
        z = pm.Normal("z", 0.0, 1.0, shape=J)
        theta = pm.Deterministic("theta", mu + tau * z)
        pm.Normal("obs", theta, se_data, observed=y)
    return m


def dims_model(data_seed=1, prior_seed=None, J=5):
    """A model with `dims=`: the value variable `th` has the unknown static shape `(None,)`.

    `lifted._abstract` used to read its shape off `model.initial_point()`, which compiles a PyTensor function.
    `data_seed` moves a `pm.Data` and the observed array; `prior_seed` the two prior values.
    """
    prior_seed = data_seed if prior_seed is None else prior_seed
    data, prior = np.random.default_rng(data_seed), np.random.default_rng(2000 + prior_seed)
    with pm.Model(coords={"g": list(range(J))}) as m:
        x = pm.Data("x", data.normal(size=J), dims="g")
        th = pm.Normal("th", x, float(prior.uniform(0.5, 2.0)), dims="g")
        s = pm.HalfNormal("s", float(prior.uniform(2.0, 4.0)))
        pm.Normal("y", th, s, observed=data.normal(size=J), dims="g")
    return m
