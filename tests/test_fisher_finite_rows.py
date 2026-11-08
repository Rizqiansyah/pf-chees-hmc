"""imm_source="fisher_lowrank" with Pathfinder draws outside the support.

A model with a hard edge (log-density -inf beyond it) can have Pathfinder draws there; their gradients are NaN.
They must be left out of the Fisher estimate, not make the metric NaN (which then makes every leapfrog step NaN
and the step-size search halve to the float floor).
"""
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pfcheeshmc.sampler import PfCheesHMC


def _edge_model():
    with pm.Model() as m:
        x = pm.Normal("x", 0.0, 1.0, shape=3)
        # a hard edge near the bulk, written like pymc_extras' GenExtreme: the -inf branch is selected by a switch,
        # so beyond the edge the log-density is -inf but its gradient is NaN (0 x the NaN of the unused branch)
        # (mass pushed against the edge, like the upper bound pressing on the largest observation)
        pm.Potential("edge", pt.switch(pt.lt(x[0], 0.5), 3.0 * x[0] + 0.05 * pt.log(0.5 - x[0]), -np.inf))
    return m


def test_rows_with_a_non_finite_gradient_are_left_out():
    import jax.numpy as jnp
    import pytest
    from pfcheeshmc.sampler import _finite_gradient_rows
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(200, 3))); g = -x
    same = _finite_gradient_rows(x, g, 2)
    assert same[0] is x and same[1] is g and same[2] == {"fisher_rows": 200, "fisher_rows_used": 200}
    g_bad = g.at[[3, 50, 51], 0].set(jnp.nan).at[7, 2].set(jnp.inf)
    with pytest.warns(RuntimeWarning, match="4 of 200"):
        x2, g2, rec = _finite_gradient_rows(x, g_bad, 2)
    assert rec == {"fisher_rows": 200, "fisher_rows_used": 196} and x2.shape == (196, 3)
    np.testing.assert_array_equal(np.asarray(g2), np.delete(np.asarray(g), [3, 7, 50, 51], axis=0))
    with pytest.raises(ValueError, match="too few"):
        _finite_gradient_rows(x[:8], g_bad[:8].at[:, 0].set(jnp.nan), 2)


def test_the_fisher_metric_from_rows_with_nan_gradients_is_finite():
    """The estimator itself, fed the filtered rows, gives a finite metric; fed the raw rows, it does not."""
    import blackjax as bj
    import jax.numpy as jnp
    from pfcheeshmc.sampler import _finite_gradient_rows
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(300, 4)) * [1.0, 2.0, 0.5, 3.0]); g = -x / jnp.asarray([1.0, 4.0, 0.25, 9.0])
    g = g.at[:10, 1].set(jnp.nan)
    raw = bj.adaptation.metric_estimators.fisher_score_low_rank(x, g, 2, gamma=1e-5, cutoff=2.0)
    assert not np.all(np.isfinite(np.asarray(raw.sigma)))
    with pytest_warns():
        xf, gf, _ = _finite_gradient_rows(x, g, 2)
    lr = bj.adaptation.metric_estimators.fisher_score_low_rank(xf, gf, 2, gamma=1e-5, cutoff=2.0)
    assert np.all(np.isfinite(np.asarray(lr.sigma))) and np.all(np.isfinite(np.asarray(lr.U)))


def pytest_warns():
    import pytest
    return pytest.warns(RuntimeWarning)
