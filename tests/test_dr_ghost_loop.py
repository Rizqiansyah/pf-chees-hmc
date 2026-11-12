"""Delayed rejection's ghosts built in one loop (`delayed_rejection_step(ghost_loop=True)`, the default) against the
Python recursion they replaced (``ghost_loop=False``): the same draws and retry fields, bit for bit, on the CPU.

The loop traces the integrator once per stage instead of 2^(k - 1) times; the step / a^(j - 1) and a^(j - 1) n it
integrates with become data instead of constants, and the acceptance arithmetic runs over the same tree in the same
order. Targets: the hard edge of test_delayed_rejection.py and Neal's funnel (non-trivial gradients); factors 2, 3, 4,
8 (3 is not a power of two: a division by it is not exact); two to four stages; the three retry rules.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st

from pfcheeshmc import compiled


def _edge(pos):
    x = pos["x"]
    return jnp.where(x > 0, 2.7 * jnp.log(jnp.abs(x) + 1e-300) - x, -jnp.inf)


def _funnel(pos):
    v, x = pos["v"], pos["x"]
    return -v ** 2 / 18.0 - 0.5 * jnp.sum(x ** 2, axis=-1) * jnp.exp(-v) - 1.5 * v


def _run(target, factor, stages, retry, ghost_loop):
    rng = np.random.default_rng(0)
    if target == "edge":
        ld = compiled.plain_log_density(_edge, ["x"])
        start, imm, eps = {"x": jnp.asarray(st.gamma(3.7).rvs(2048, random_state=rng))}, jnp.ones(1), 4.0
    else:
        ld = compiled.plain_log_density(_funnel, ["v", "x"])
        v = rng.normal(scale=3.0, size=2048)
        start = {"v": jnp.asarray(v), "x": jnp.asarray(rng.normal(size=(2048, 3)) * np.exp(v / 2)[:, None])}
        imm, eps = jnp.ones(4), 1.5
    state = compiled.batch_init(ld.fn, ld.names, ld.values, start, 0)
    s, (_, info) = compiled.sampling_chunk(ld.fn, ld.names, ld.values, state, imm, eps, 3.0, jax.random.key(5),
                                           np.int32(0), np.int32(15), chunk=15, jitter_amount=1.0, dr_factor=factor,
                                           dr_stages=stages, dr_retry=retry, dr_ghost_loop=ghost_loop)
    return s, info


@pytest.mark.parametrize("target, factor, stages, retry", [
    ("edge", 4, 2, "divergent"), ("edge", 4, 3, "divergent"), ("edge", 2, 4, "divergent"), ("edge", 8, 4, "divergent"),
    ("edge", 3, 3, "divergent"), ("funnel", 4, 3, "divergent"), ("funnel", 2, 4, ("energy_above", 2.0)),
    ("funnel", 3, 3, "probabilistic"), ("edge", 4, 3, "probabilistic"),
])
def test_loop_equals_recursion(target, factor, stages, retry):
    a, ia = _run(target, factor, stages, retry, False)
    b, ib = _run(target, factor, stages, retry, True)
    assert int(np.sum(ia.retried)) > 50
    for k in a.position:
        assert np.array_equal(np.asarray(a.position[k]), np.asarray(b.position[k])), k
    for f in ("retried", "retry_accepted", "retry_divergent", "retry_integration_steps"):
        assert np.array_equal(np.asarray(getattr(ia, f)), np.asarray(getattr(ib, f))), f
