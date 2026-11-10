"""pathfinder_kwargs pf_maxiter, pf_maxcor, pf_maxls, pf_gtol, pf_ftol, pf_elbo_samples reach BlackJAX's L-BFGS.

1. absent, the fit is called exactly as before (no ``lbfgs`` argument), so the default path is unchanged;
2. given, they arrive at `blackjax.vi.pathfinder.approximate` under its own names, and change the fit;
3. they are recorded in ``idata.attrs["config"]`` as ``pf_lbfgs``.
"""
import json
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import pytensor  # noqa: E402

pytensor.config.blas__ldflags = "-llapack -lblas -lcblas -lm"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.models import build_model  # noqa: E402

from pfcheeshmc import compiled, sample_pf_chees_hmc  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402


def _fit_calls(monkeypatch):
    calls = []
    real = compiled.pathfinder_fit

    def spy(*a, **k):
        calls.append(k)
        out = real(*a, **k)
        calls[-1] = (k, out)
        return out

    monkeypatch.setattr(compiled, "pathfinder_fit", spy)
    return calls


def test_default_call_is_unchanged(monkeypatch):
    calls = _fit_calls(monkeypatch)
    PfCheesHMC(build_model("gev"), seed=1)._run_pathfinder(chains=8, draws=32, seed=1, pm_tune=50)
    assert calls[0][0] == {}


def test_settings_reach_approximate_and_change_the_fit(monkeypatch):
    calls = _fit_calls(monkeypatch)
    s = PfCheesHMC(build_model("gev"), seed=1)
    s._run_pathfinder(chains=8, draws=32, seed=1, pm_tune=50)
    s._run_pathfinder(chains=8, draws=32, seed=1, pm_tune=50, pf_maxiter=2, pf_ftol=1e-3, pf_elbo_samples=20)
    (k0, (st0, _)), (k1, (st1, _)) = calls
    assert k1 == {"lbfgs": (("ftol", 1e-3), ("maxiter", 2), ("num_samples", 20))}
    assert not np.allclose(np.concatenate([np.ravel(x) for x in st0.position]),
                           np.concatenate([np.ravel(x) for x in st1.position]))


def test_recorded_in_config():
    with build_model("8schools"):
        idata = sample_pf_chees_hmc(draws=5, chains=8, random_seed=1, progressbar=False,
                                    pathfinder_kwargs={"draws": 50, "pm_tune": 50, "pf_maxiter": 100},
                                    chees_kwargs={"draws": 10})
    assert json.loads(idata.attrs["config"])["pf_lbfgs"] == {"maxiter": 100}
