"""`idata.attrs["diagnostics"]` and the warnings that read it: L-BFGS iterations against Pathfinder's cap, divergences by
the energy at the trajectory's end, and float32 matrix products in TF32 on an Ampere or later GPU.

Each count is checked against an independent one: the L-BFGS iterations against L-BFGS's own status (which BlackJAX drops
in `pathfinder.approximate`), the energy classes against `sample_stats`. Models: a quadratic whose L-BFGS path needs more
than two iterations, a Gaussian with a hard edge (proposals past it have log density -inf, energy +inf), and one whose
log density is NaN past a point. The TF32 check runs with the backend and the devices replaced.
"""
import json
import types
import warnings

import blackjax as bj
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pfcheeshmc import compiled, sample_pf_chees_hmc
from pfcheeshmc import sampler as sampler_module
from pfcheeshmc.sampler import PfCheesHMC, _divergence_energy, _tf32_message


def _stiff_quadratic():
    """A Gaussian posterior with scales 1e-2 to 1e2 whose mode is away from PyMC's start (the prior mean, 0), so L-BFGS
    has a path to run."""
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1e3, shape=3)
        pm.Normal("y", x, np.array([1e-2, 1.0, 1e2]), observed=np.array([0.5, -2.0, 300.0]))
    return model


def _run(model, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        idata = sample_pf_chees_hmc(model, draws=kw.pop("draws", 20), chains=8, random_seed=kw.pop("seed", 1),
                                    progressbar=False, chees_kwargs={"draws": kw.pop("warmup", 30)}, **kw)
    return idata, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


# ------------------------------------------------------------------------------------------------ L-BFGS iterations
@pytest.mark.parametrize("maxiter", [2, 5, 200])
def test_lbfgs_iterations_are_lbfgs_own_count(maxiter):
    """The count read off the path equals ``iter_num`` of L-BFGS's status, from BlackJAX's own minimiser on the same
    objective and start."""
    s = PfCheesHMC(_stiff_quadratic(), seed=1)
    ld = s._logdensity()
    x0 = (jnp.zeros(3),)
    _, got = compiled.pathfinder_fit(bj.vi.pathfinder.approximate, ld.fn, ld.names, ld.values, jax.random.key(0), x0,
                                     lbfgs=(("maxiter", maxiter),))
    objective = lambda flat: -compiled._logdensity(ld.fn, ld.names, ld.values)((flat,))
    (_, status), _ = bj.optimizers.lbfgs._minimize_lbfgs(objective, x0[0], maxiter, 10, 1e-8, 1e-5, 1000)
    assert int(got) == int(status.iter_num)
    if maxiter == 200:
        assert int(got) < maxiter           # converged before the cap: the warning's premise is not always true


def test_the_cap_warns_and_is_recorded():
    idata, warned = _run(_stiff_quadratic(), pathfinder_kwargs={"pf_maxiter": 2})
    diag = json.loads(idata.attrs["diagnostics"])
    assert diag["pathfinder"] == {"lbfgs_iterations": 2, "lbfgs_maxiter": 2}
    assert any("used all 2 of its iterations" in m for m in warned)


def test_no_warning_below_the_cap():
    idata, warned = _run(_stiff_quadratic(), pathfinder_kwargs={"pf_maxiter": 200})
    diag = json.loads(idata.attrs["diagnostics"])
    assert diag["pathfinder"]["lbfgs_maxiter"] == 200
    assert 0 < diag["pathfinder"]["lbfgs_iterations"] < 200
    assert not any("of its iterations" in m for m in warned)


def test_the_default_cap_is_blackjax_default():
    import inspect
    want = inspect.signature(bj.vi.pathfinder.approximate).parameters["maxiter"].default
    idata, _ = _run(_stiff_quadratic())
    assert json.loads(idata.attrs["diagnostics"])["pathfinder"]["lbfgs_maxiter"] == want


# ------------------------------------------------------------------------------------------- divergences by energy
def test_energy_classes():
    div = np.array([[True, True, True, True, False, True]])
    energy = np.array([[np.inf, np.nan, 3.0, -np.inf, np.nan, np.inf]])
    assert _divergence_energy(div, energy) == {"inf": 2, "-inf": 1, "nan": 1, "finite": 1}
    assert _divergence_energy(np.zeros((2, 3), bool), np.full((2, 3), np.nan)) == {"inf": 0, "-inf": 0, "nan": 0,
                                                                                    "finite": 0}


def _hard_edge():
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0)
        pm.Potential("edge", pt.switch(pt.gt(x, 0.8), -np.inf, 0.0))
    return model


def _nan_branch():
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0)
        pm.Potential("broken", pt.switch(pt.gt(x, 0.8), np.nan, 0.0))
    return model


def _counted_from_sample_stats(idata):
    div = idata.sample_stats["diverging"].values
    e = idata.sample_stats["energy"].values[div]
    return {"inf": int(np.isposinf(e).sum()), "-inf": int(np.isneginf(e).sum()), "nan": int(np.isnan(e).sum()),
            "finite": int(np.isfinite(e).sum())}, int(div.sum())


def test_a_hard_edge_diverges_at_plus_inf_without_the_nan_warning():
    idata, warned = _run(_hard_edge(), draws=60)
    got = json.loads(idata.attrs["diagnostics"])["divergence_energy"]
    want, n_div = _counted_from_sample_stats(idata)
    assert got == want and sum(got.values()) == n_div
    assert got["inf"] > 0 and got["nan"] == got["-inf"] == 0
    assert not any("NaN (or -inf) energy" in m for m in warned)


def test_a_nan_branch_warns():
    idata, warned = _run(_nan_branch(), draws=60)
    got = json.loads(idata.attrs["diagnostics"])["divergence_energy"]
    want, n_div = _counted_from_sample_stats(idata)
    assert got == want and got["nan"] > 0
    assert any(f"{got['nan'] + got['-inf']} of {n_div} divergent transitions ended at a NaN" in m for m in warned)


# ---------------------------------------------------------------------------------------------------------- TF32
@pytest.mark.parametrize("floatx, backend, caps, precision, warns", [
    ("float32", "gpu", ["8.0"], None, True),
    ("float32", "gpu", ["12.0"], None, True),
    ("float32", "gpu", ["7.5", "8.6"], None, True),       # any device of capability 8.0 or above
    ("float32", "gpu", ["7.5"], None, False),             # Turing and older have no TF32
    ("float32", "gpu", ["8.0"], "highest", False),
    ("float32", "gpu", ["8.0"], "float32", False),
    ("float64", "gpu", ["8.0"], None, False),
    ("float32", "cpu", [], None, False),
    ("float32", "gpu", [None], None, False),              # capability unknown: no claim
])
def test_tf32_rule(floatx, backend, caps, precision, warns):
    assert (_tf32_message(floatx, backend, caps, precision) is not None) == warns


def test_tf32_warning_reads_backend_devices_and_precision(monkeypatch):
    monkeypatch.setattr(sampler_module, "pytensor", types.SimpleNamespace(config=types.SimpleNamespace(floatX="float32")))
    monkeypatch.setattr(sampler_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(sampler_module.jax, "devices", lambda: [types.SimpleNamespace(compute_capability="8.0")])
    with pytest.warns(RuntimeWarning, match="JAX_DEFAULT_MATMUL_PRECISION=highest"):
        sampler_module._warn_if_tf32()
    with jax.default_matmul_precision("highest"), warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler_module._warn_if_tf32()


def test_no_tf32_warning_on_this_cpu_run():
    _, warned = _run(_stiff_quadratic(), draws=5)
    assert not any("TF32" in m for m in warned)
