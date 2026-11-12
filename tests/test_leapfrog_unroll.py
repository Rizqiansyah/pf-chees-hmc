"""Several leapfrog steps per loop pass (``dynamic_hmc_kwargs={"leapfrog_unroll": k}``).

The same steps run in a different loop structure (n // k passes of k steps, then n % k single steps), so the draws
must be those of k = 1 (BlackJAX's kernel, the golden files), to rounding: the same accept decisions and step counts,
positions equal to 1e-12 of each coordinate's scale. With delayed rejection the retries' trajectories are unrolled too.
"""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

from pfcheeshmc import compiled, sample_pf_chees_hmc



def assert_same_run(a, b, msg):
    """Equal to rounding: XLA may fuse the steps of an unrolled pass differently, so a coordinate can differ in its
    last bits (seen: 2e-16 absolute on values of 1e-4 whose coordinate's scale is 0.1); decisions and step counts are
    compared exactly by the callers."""
    a, b = np.asarray(a), np.asarray(b)
    scale = np.max(np.abs(a), axis=tuple(range(a.ndim - 1)) if a.ndim > 1 else None)
    np.testing.assert_allclose(a, b, rtol=0, atol=float(np.max(1e-12 * scale)), err_msg=msg)


def _edge_logp(pos):
    x = pos["x"]
    return jnp.where(x > 0, 2.7 * jnp.log(jnp.abs(x) + 1e-300) - x, -jnp.inf)


def _gauss_logp(pos):
    x = pos["x"]
    return -0.5 * jnp.sum((x / jnp.array([0.1, 1.0, 10.0])) ** 2, axis=-1)


def _chunk(logp, start, imm, eps, L, unroll, dr=None, K=30):
    ld = compiled.plain_log_density(logp, ["x"])
    state = compiled.batch_init(ld.fn, ld.names, ld.values, {"x": start}, 0)
    s, (st, info) = compiled.sampling_chunk(ld.fn, ld.names, ld.values, state, imm, eps, L, jax.random.key(3),
                                            np.int32(0), np.int32(K), chunk=K, jitter_amount=1.0, dr_factor=dr,
                                            dr_stages=3, unroll=unroll)
    return np.asarray(st.position["x"]), info


@pytest.mark.parametrize("unroll", [2, 3, 4, 8])
def test_same_draws_as_one_step_per_pass(unroll):
    start = jax.random.normal(jax.random.key(0), (256, 3)) * jnp.array([0.1, 1.0, 10.0])
    imm = jnp.array([0.01, 1.0, 100.0])
    a, ia = _chunk(_gauss_logp, start, imm, 0.9, 7.3, 1)
    b, ib = _chunk(_gauss_logp, start, imm, 0.9, 7.3, unroll)
    assert np.array_equal(np.asarray(ia.num_integration_steps), np.asarray(ib.num_integration_steps))
    assert np.array_equal(np.asarray(ia.is_accepted), np.asarray(ib.is_accepted))
    assert_same_run(a, b, f"unroll={unroll}")


@pytest.mark.parametrize("unroll", [2, 4])
def test_same_draws_with_delayed_rejection(unroll):
    start = jnp.asarray(np.random.default_rng(0).gamma(3.7, size=4096))
    a, ia = _chunk(_edge_logp, start, jnp.ones(1), 4.0, 3.0, 1, dr=2)
    b, ib = _chunk(_edge_logp, start, jnp.ones(1), 4.0, 3.0, unroll, dr=2)
    assert int(np.sum(ia.retry_accepted)) > 100
    for f in ("retried", "retry_accepted", "retry_divergent", "retry_integration_steps"):
        assert np.array_equal(np.asarray(getattr(ia, f)), np.asarray(getattr(ib, f))), f
    assert_same_run(a, b, f"unroll={unroll} with delayed rejection")


def test_end_to_end_option_and_config():
    with pm.Model() as model:
        pm.Normal("x", 0.0, np.array([0.1, 1.0, 10.0]), shape=3)
    kw = dict(draws=20, chains=16, random_seed=1, progressbar=False, chees_kwargs={"draws": 30})
    a = sample_pf_chees_hmc(model, **kw)
    b = sample_pf_chees_hmc(model, **kw, dynamic_hmc_kwargs={"leapfrog_unroll": 4})
    assert_same_run(a.posterior["x"].values, b.posterior["x"].values, "end to end")
    assert "leapfrog_unroll" not in json.loads(a.attrs["config"])
    assert json.loads(b.attrs["config"])["leapfrog_unroll"] == 4


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "4"])
def test_option_is_checked(bad):
    with pm.Model() as model:
        pm.Normal("x")
    with pytest.raises(ValueError, match="leapfrog_unroll"):
        sample_pf_chees_hmc(model, draws=5, chains=8, random_seed=1, progressbar=False, chees_kwargs={"draws": 5},
                            dynamic_hmc_kwargs={"leapfrog_unroll": bad})
