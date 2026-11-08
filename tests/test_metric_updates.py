"""Metric updates during the ChEES warmup (``chees_kwargs["metric_updates"]``).

The Pathfinder metric is fitted once, before any chain has moved, and frozen. On a stiff hierarchical model
it under-reads one slow coordinate, and that coordinate then stays slow for the whole run. The option
re-estimates the metric from the chain positions at given warmup fractions and restarts the step size.
What must hold:

1. Off (the default), nothing changes: the draws are those of the previous code (pinned by the golden
   files of test_golden.py), and an empty tuple is the same as absent.
2. A Gaussian target started with a wrong metric gets its covariance back, dense and diagonal.
3. The updates add no program after the first call, and the warmup chunk program stays one program.
4. A bad estimate is never used, and bad options raise.
"""
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.compile_counter import count_compiles  # noqa: E402
from benchmarks.seeded_models import hier  # noqa: E402

from pfcheeshmc import compiled
from pfcheeshmc.sampler import sample_pf_chees_hmc
from pfcheeshmc.utils import chees_warmup_with_progress

SD = np.array([10.0, 1.0, 0.1])
RHO = 0.95


def _gauss(dense, num_steps=400, updates=(0.25, 0.5, 0.75), chains=256, shrinkage=None):
    """A 3-d Gaussian with scales SD (and correlation RHO between the first two, ``dense``), started
    from N(0, I) with the identity metric, which is wrong by a factor of 10,000 in variance."""
    cov = np.diag(SD ** 2)
    if dense:
        cov[0, 1] = cov[1, 0] = RHO * SD[0] * SD[1]
    prec = jnp.asarray(np.linalg.inv(cov))
    ld = lambda p: -0.5 * p["x"] @ prec @ p["x"]
    positions = {"x": jax.random.normal(jax.random.key(3), (chains, 3))}
    imm = jnp.eye(3) if dense else jnp.ones(3)
    (states, params), _ = chees_warmup_with_progress(
        ld, jax.random.key(0), positions, jnp.float64(0.05), 0.05, num_steps, imm,
        progressbar=False, metric_updates=updates, metric_shrinkage=shrinkage)
    return cov, states, params


@pytest.mark.parametrize("dense", [True, False])
def test_a_wrong_metric_is_recovered(dense):
    cov, _, params = _gauss(dense)
    m = np.asarray(params["inverse_mass_matrix"])
    log = params["metric_updates"]
    assert [u["step"] for u in log] == [100, 200, 300]
    assert all(u["used"] for u in log)
    var = np.diag(m) if dense else m
    # 256 chains: a variance estimate has a relative sd of ~9%; 0.6-1.6 is a wide band around the truth
    # that the identity metric (ratios 0.01, 1, 100) is far outside of
    np.testing.assert_array_less(0.6 * SD ** 2, var)
    np.testing.assert_array_less(var, 1.6 * SD ** 2)
    if dense:
        corr = m[0, 1] / np.sqrt(m[0, 0] * m[1, 1])
        assert abs(corr - RHO) < 0.05, corr


def test_without_updates_the_metric_is_the_one_passed_in():
    _, _, params = _gauss(True, updates=None)
    assert params["metric_updates"] == []
    np.testing.assert_array_equal(np.asarray(params["inverse_mass_matrix"]), np.eye(3))


def test_the_step_size_restarts_from_a_search_under_the_new_metric():
    """Under the recovered metric the target is roughly white, so the restarted step size is of order
    one; under the identity metric it was set by the 0.1 scale."""
    _, _, params = _gauss(False)
    eps = [u["step_size"] for u in params["metric_updates"]]
    assert all(np.isfinite(eps))
    assert eps[-1] > 0.3, eps


def test_full_shrinkage_keeps_the_old_metric():
    _, _, params = _gauss(False, shrinkage=1.0)
    np.testing.assert_allclose(np.asarray(params["inverse_mass_matrix"]), np.ones(3))


def test_an_estimate_that_is_not_positive_definite_is_not_used():
    """Two chains at the same point: the covariance is zero, the old metric is kept, the flag says so."""
    pos = {"x": jnp.zeros((2, 3))}
    new, used = compiled.ensemble_metric(pos, jnp.eye(3), 0.0)
    assert not bool(used)
    np.testing.assert_array_equal(np.asarray(new), np.eye(3))
    new, used = compiled.ensemble_metric(pos, jnp.ones(3), 0.0)
    assert not bool(used)
    # a non-finite chain is left out, not propagated
    pos = {"x": jnp.array([[0.0, 0, 0], [1.0, 2, 3], [jnp.nan, 0, 0], [2.0, 1, 0]])}
    new, used = compiled.ensemble_metric(pos, jnp.ones(3), 0.0)
    assert bool(used)
    np.testing.assert_allclose(np.asarray(new), np.var([[0, 0, 0], [1, 2, 3], [2, 1, 0]], axis=0, ddof=1))


def test_the_metric_is_estimated_in_kernel_order():
    """Two leaves: the kernel ravels a dict by sorted key, and so does the estimate."""
    k = jax.random.split(jax.random.key(1), 2)
    pos = {"b": 100.0 * jax.random.normal(k[0], (4000,)), "a": jax.random.normal(k[1], (4000, 2))}
    new, used = compiled.ensemble_metric(pos, jnp.ones(3), 0.0)
    v = np.asarray(new)
    assert bool(used) and v[2] > 1000 * max(v[0], v[1]), v


@pytest.mark.parametrize("bad", [dict(metric_updates=(0.0,)), dict(metric_updates=(1.0,)),
                                 dict(metric_updates=(0.5,), metric_shrinkage=1.5),
                                 dict(metric_updates=(0.5,), stop_rule={"block": 10})])
def test_bad_options_raise(bad):
    ld = lambda p: -0.5 * jnp.sum(p["x"] ** 2)
    with pytest.raises(ValueError):
        chees_warmup_with_progress(ld, jax.random.key(0), {"x": jnp.zeros((4, 2))}, jnp.float64(0.1), 0.05,
                                   20, jnp.ones(2), progressbar=False, **bad)


def _call(model, seed, **chees):
    with count_compiles() as cc:
        idata = sample_pf_chees_hmc(model, draws=10, chains=16, random_seed=seed, progressbar=False,
                                    chees_kwargs={"draws": 20, **chees})
    return idata, cc


def test_an_empty_update_list_is_the_default_path():
    a, _ = _call(hier(1, 1), 1)
    b, _ = _call(hier(1, 1), 1, metric_updates=())
    for v in a.posterior.data_vars:
        np.testing.assert_array_equal(a.posterior[v].values, b.posterior[v].values)


def test_updates_compile_nothing_on_a_second_call_and_are_recorded():
    _call(hier(1, 1), 1, metric_updates=(0.5,))
    idata, cc = _call(hier(2, 2), 2, metric_updates=(0.3, 0.6))
    assert cc.count == 0, str(cc)
    import json
    config = json.loads(idata.attrs["config"])
    assert config["metric_updates"] == [0.3, 0.6]
    assert [u["step"] for u in config["metric_update_log"]] == [6, 12]


def test_the_first_call_compiles_the_warmup_chunk_once():
    """Segments are the same chunk program run from an offset: one warmup_chunk program however many updates.
    A fresh shape (chains=24) so nothing is cached from the other tests."""
    with count_compiles() as cc:
        sample_pf_chees_hmc(hier(1, 1), draws=10, chains=24, random_seed=1, progressbar=False,
                            chees_kwargs={"draws": 20, "metric_updates": (0.25, 0.5, 0.75)})
    assert sum("warmup_chunk" in n for n in cc.names) == 1, cc.names
