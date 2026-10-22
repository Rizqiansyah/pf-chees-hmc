"""The sampler's jitted programs outside the two loops, defined once, at module level.

Why this module exists. A jitted function is cached by the identity of the Python function plus its
static arguments and the shapes of its dynamic ones. Two habits defeated that cache on every sampler
call: building the jitted function inside a method (a new function object per call), and closing over
per-run values (data, metric, step size), which JAX then compiles in as constants. Every site here is
instead one module-level function with

* static arguments: the lifted log-density ``fn`` (`pfcheeshmc.lifted`; structurally equal models share
  ONE function object through its registry), the value-variable ``names`` and the settings;
* dynamic arguments: everything that changes between runs: the lifted ``values`` (data and prior
  values), RNG keys, positions, the metric, ...

so JAX's own jit cache does all the caching, and a second call with a new seed, new data or new prior
values at the same shapes compiles nothing.

`LogDensity` bundles the three pieces a site needs, and is itself the plain ``position -> scalar``
callable the two loops still take (they close over ``values`` for now).
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable

import blackjax as bj
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.dynamic_hmc as dynamic_hmc

__all__ = [
    "LogDensity",
    "batch_init",
    "feasible_mask",
    "pathfinder_draw",
    "pathfinder_fit",
    "pool_gradients",
    "score_pool",
    "score_positions",
    "search_step_size",
    "transform_draws",
    "usable_mask",
]


def _position_sequence(position, names):
    """A position as the tuple ``lifted.LiftedGraph.fn`` takes (value-variable order)."""
    if isinstance(position, dict):
        return tuple(position[n] for n in names)
    if isinstance(position, (list, tuple)):
        return tuple(position)
    return tuple(jax.tree_util.tree_leaves(position))      # last resort: a stable leaf order


def _logdensity(fn, names, values) -> Callable:
    """``position -> scalar`` for a position that is a dict (by name) or a tuple (in ``names`` order)."""
    return lambda position: fn(_position_sequence(position, names), values)[0]


class LogDensity:
    """A model's log-density as the pieces the compiled sites take.

    fn
        ``fn(position_tuple, values) -> list`` (`LiftedGraph.fn`): static for jit; the same object for
        structurally equal models.
    names
        The value variables in ``fn``'s order: static for jit.
    values
        The lifted data and prior values: dynamic for jit.

    Calling it, ``ld(position)`` is the scalar log-density of a dict or tuple position, with ``values``
    closed over as constants (what the two loops do until they take ``values`` as an argument).
    """

    __slots__ = ("fn", "names", "values")

    def __init__(self, fn: Callable, names: tuple, values: tuple):
        self.fn, self.names, self.values = fn, tuple(names), tuple(values)

    def __call__(self, position):
        return self.fn(_position_sequence(position, self.names), self.values)[0]


_STATIC = ("fn", "names")


# --------------------------------------------------------------------------------------------------
# start screening and scoring
# --------------------------------------------------------------------------------------------------
@partial(jax.jit, static_argnames=_STATIC)
def usable_mask(fn, names, values, stack):
    """Per chain: finite log-density AND finite gradient. ``stack`` is a dict of ``(chains, ...)``."""
    ld = _logdensity(fn, names, values)

    def ok(x):
        g = jax.grad(ld)(x)
        return jnp.isfinite(ld(x)) & jnp.all(jnp.isfinite(ravel_pytree(g)[0]))

    return jax.vmap(ok)(stack)


@partial(jax.jit, static_argnames=_STATIC)
def feasible_mask(fn, names, values, stack):
    """Per row of ``stack`` (dict of ``(n, ...)``): is the log-density finite?"""
    ld = _logdensity(fn, names, values)
    return jax.vmap(lambda p: jnp.isfinite(ld(p)))(stack)


@partial(jax.jit, static_argnames=_STATIC)
def score_positions(fn, names, values, stack):
    """Log-density of every row of ``stack`` (dict of ``(n, ...)``)."""
    return jax.vmap(_logdensity(fn, names, values))(stack)


@partial(jax.jit, static_argnames=_STATIC)
def score_pool(fn, names, values, pool, idx):
    """Log-density of rows ``idx`` of ``pool``, a position pytree whose leaves are ``(rows, ...)``."""
    ld = _logdensity(fn, names, values)
    return jax.vmap(lambda i: ld(jax.tree_util.tree_map(lambda x: x[i], pool)))(idx)


@partial(jax.jit, static_argnames=_STATIC)
def pool_gradients(fn, names, values, template, draws):
    """Gradients of the log-density at the rows of ``draws``, flat, in the order of ``template``.

    ``draws`` is ``(n, d)``: each row is ``ravel_pytree(position)`` for a position shaped like
    ``template`` (one position, not a stack; only its structure and shapes are used).
    """
    ld = _logdensity(fn, names, values)
    _, unravel = ravel_pytree(template)
    return jax.vmap(lambda f: ravel_pytree(jax.grad(ld)(unravel(f)))[0])(draws)


# --------------------------------------------------------------------------------------------------
# Pathfinder
# --------------------------------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("approximate",) + _STATIC)
def pathfinder_fit(approximate, fn, names, values, key, initial_position):
    """``approximate(key, ld, initial_position)``: the L-BFGS path, whole, as one program.

    ``approximate`` is ``blackjax.vi.pathfinder.approximate``, looked up by the caller at call time (a
    static argument, so a replaced function is its own cache entry). maxiter, maxcor, num_samples and
    the rest keep BlackJAX's defaults, which are Python constants of the trace. Only the fitted
    `PathfinderState` is returned; the traversed path (``maxiter + 1`` states) is not kept.
    """
    state, _ = approximate(
        rng_key=key, initial_position=initial_position, logdensity_fn=_logdensity(fn, names, values))
    return state


@partial(jax.jit, static_argnames=("sample", "num_samples"))
def pathfinder_draw(sample, key, state, num_samples):
    """``num_samples`` draws from the fitted Pathfinder Gaussian (``blackjax.vi.pathfinder.sample``)."""
    draws, _ = sample(key, state, num_samples)
    return draws


# --------------------------------------------------------------------------------------------------
# step-size search
# --------------------------------------------------------------------------------------------------
@partial(jax.jit, static_argnames=_STATIC)
def search_step_size(fn, names, values, inverse_mass_matrix, key, position):
    """BlackJAX's doubling/halving search (target acceptance 0.651) from one position, for one metric."""
    ld = _logdensity(fn, names, values)
    return bj.adaptation.step_size.find_reasonable_step_size(
        key,
        lambda eps: bj.hmc(ld, eps, inverse_mass_matrix, num_integration_steps=1).step,
        bj.hmc(ld, 1e-2, inverse_mass_matrix, num_integration_steps=1).init(position),
        initial_step_size=1e-2,
        target_accept=0.651,
    )


# --------------------------------------------------------------------------------------------------
# warmup initial states, and the value -> problem-space transform
# --------------------------------------------------------------------------------------------------
@partial(jax.jit, static_argnames=_STATIC)
def batch_init(fn, names, values, positions, random_generator_arg):
    """Dynamic-HMC initial states for a dict of ``(chains, ...)`` positions."""
    ld = _logdensity(fn, names, values)
    return jax.vmap(lambda p: dynamic_hmc.init(p, ld, random_generator_arg))(positions)


@partial(jax.jit, static_argnames=("fn",))
def transform_draws(fn, values, raw):
    """``fn(value_arrays, values)`` over every (chain, draw): ``raw`` is a list of ``(C, D, ...)`` arrays."""
    return jax.vmap(jax.vmap(lambda *xs: fn(xs, values)))(*raw)
