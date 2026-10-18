"""Tests for `_find_finite_start` rejecting a feasible point with a NaN gradient.

The case that motivated this: GEV Port Pirie at `model.initial_point()` has log-density
-7.749 (finite) and a NaN gradient in every variable, because `GenExtreme.logp` differentiates
through an unselected `switch` branch containing (xi+1)/xi. The old guard checked the
log-density only, so it accepted that point and Pathfinder started its L-BFGS fit from a NaN
gradient. The pipeline survived only because its seed landed elsewhere.

These tests use a synthetic density with the same shape -- finite value, NaN derivative at the
point -- so the guard is exercised without loading a notebook.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

from pfcheeshmc.sampler import PfCheesHMC


def _x_squared_sinv(pos):
    """Finite value, NaN derivative at 0 -- the same shape as GEV's logp at xi = 0.

    d/dx[x^2 sin(1/x)] = 2x sin(1/x) - cos(1/x), and cos(1/0) is NaN. The `where` keeps the
    forward value at 0, and as in `GenExtreme.logp` the NaN leaks back through the unselected
    branch: 0 * NaN is NaN.
    """
    x = pos["x"]
    return -jnp.sum(jnp.where(x == 0.0, 0.0, x * x * jnp.sin(1.0 / x)))


def _sampler_with(ld_fn):
    with pm.Model() as model:
        pm.Normal("x", shape=(3,))
    s = PfCheesHMC(model, seed=7)
    s.ld_fn = ld_fn
    return s


def test_nan_gradient_point_is_not_accepted_as_is():
    """Sanity check on the fixture itself: value finite, gradient NaN at the start."""
    s = _sampler_with(_x_squared_sinv)
    start = {"x": jnp.zeros(3)}
    assert bool(jnp.isfinite(s.ld_fn(start)))
    grads = jax.grad(s.ld_fn)(start)
    assert not bool(jnp.all(jnp.isfinite(grads["x"])))


def test_repair_returns_a_point_with_finite_gradient():
    s = _sampler_with(_x_squared_sinv)
    fixed = s._find_finite_start({"x": jnp.zeros(3)}, jax.random.key(0))
    assert bool(jnp.all(jnp.isfinite(s.ld_fn(fixed))))
    assert bool(jnp.all(jnp.isfinite(jax.grad(s.ld_fn)(fixed)["x"])))
    assert bool(jnp.all(fixed["x"] != 0.0))


def test_good_start_is_left_essentially_alone():
    """A usable start must come back within the jitter radius, not be thrown across the space."""
    s = _sampler_with(lambda pos: -0.5 * jnp.sum(pos["x"] ** 2))
    start = {"x": jnp.array([1.0, -2.0, 3.0])}
    fixed = s._find_finite_start(start, jax.random.key(1))
    assert float(jnp.max(jnp.abs(fixed["x"] - start["x"]))) < 1e-3


def test_unusable_everywhere_raises():
    def ld(pos):                                   # -inf everywhere: no usable start exists
        return jnp.where(jnp.all(jnp.isfinite(pos["x"])), -jnp.inf, -jnp.inf)

    s = _sampler_with(ld)
    with pytest.raises(RuntimeError, match="finite log-density and finite gradient"):
        s._find_finite_start({"x": jnp.zeros(3)}, jax.random.key(2))


def _sampler_with_model(d=3):
    with pm.Model() as model:
        pm.Normal("x", shape=(d,))
    return PfCheesHMC(model, seed=7)


def test_all_usable_starts_take_the_batched_fast_path():
    """The point of _usable_mask: no per-chain eager repair when nothing is wrong."""
    s = _sampler_with_model()
    stack = {"x": jnp.array([[0.1, 0.2, 0.3], [-1.0, 2.0, 0.5]])}
    calls = []
    orig = s._find_finite_start
    s._find_finite_start = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    out = s._ensure_finite_inits(stack, jax.random.key(0))
    assert calls == [], f"expected zero per-chain repairs, got {len(calls)}"
    np.testing.assert_allclose(np.asarray(out["x"]), np.asarray(stack["x"]), rtol=0, atol=0)


def test_one_bad_chain_repairs_only_that_chain():
    """A singularity in one chain must trigger exactly one repair, not a sweep of all chains."""
    s = _sampler_with_model(d=3)

    def ld_with_spike(pos):
        x = pos["x"]
        safe = x.at[0].set(1.0)          # keep the gradient finite wherever the spike is active
        base = -0.5 * jnp.sum(safe * safe)
        return base + jnp.where(x[0] == 99.0, 1.0 / (x[0] - 99.0), 0.0)

    s.ld_fn = ld_with_spike
    stack = {"x": jnp.array([[99.0, 0.2, 0.3], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])}
    calls = []
    orig = s._find_finite_start
    s._find_finite_start = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    out = s._ensure_finite_inits(stack, jax.random.key(0))
    assert len(calls) == 1, f"expected exactly one repair, got {len(calls)}"
    assert float(out["x"][0, 0]) != 99.0
    np.testing.assert_allclose(np.asarray(out["x"][1:]), np.asarray(stack["x"][1:]), atol=0)


def test_batched_mask_agrees_with_the_per_chain_check():
    s = _sampler_with_model(d=3)
    s.ld_fn = lambda pos: -0.5 * jnp.sum(pos["x"] ** 2)
    stack = {"x": jnp.zeros((4, 3))}
    assert bool(jnp.all(s._usable_mask(stack))), "a zero point on a unit Normal is usable"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} PASS")
