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

`LogDensity` bundles the three pieces a site needs, and is also the plain ``position -> scalar`` callable.

The two loops are programs of the same kind (`warmup_chunk`, `sampling_chunk`): one jitted *chunk* of
``chunk`` steps, with a dynamic active count ``n``, that the host calls ``ceil(total / chunk)`` times
(`run_chunks`). So the warmup length W and the number of draws D are loop bounds on the host, never
shapes or constants of a program. Step ``j`` of a chunk is skipped (`lax.cond` on a scalar, outside the
per-chain vmap: no model evaluation) when ``j >= n``, and the per-step key is
``fold_in(stage_key, global_step_index)``, which does not depend on W, D or the chunk size.
"""
from __future__ import annotations

import functools
from functools import partial
from typing import Any, Callable, NamedTuple

import blackjax as bj
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
from jax.flatten_util import ravel_pytree

import blackjax.adaptation.chees_adaptation as chees_adaptation
import blackjax.mcmc.dynamic_hmc as dynamic_hmc
import blackjax.mcmc.hmc as bj_hmc
import blackjax.mcmc.integrators as bj_integrators
import blackjax.mcmc.metrics as bj_metrics
import blackjax.mcmc.trajectory as bj_trajectory
import blackjax.optimizers.dual_averaging as dual_averaging

__all__ = [
    "ChEESTrace",
    "HALTON_BITS",
    "JITTER_STREAM",
    "LogDensity",
    "HostStack",
    "SAMPLE_CHUNK",
    "SampledInfo",
    "SampledState",
    "TRANSFORM_CHUNK",
    "WARMUP_CHUNK",
    "WarmupScanInfo",
    "batch_init",
    "chees_harmonic_acceptance",
    "chees_harmonic_acceptance_np",
    "DR_BATCH",
    "delayed_rejection_step",
    "feasible_mask",
    "halton_jitter",
    "initial_point",
    "integration_steps",
    "jittered_initial_points",
    "pathfinder_draw",
    "pathfinder_fit",
    "plain_log_density",
    "pool_gradients",
    "run_chunks",
    "sampling_chunk",
    "score_pool",
    "score_positions",
    "search_step_size",
    "search_step_sizes",
    "transform_draws",
    "usable_mask",
    "warmup_chunk",
    "warmup_init",
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
@partial(jax.jit, static_argnames=("approximate", "lbfgs") + _STATIC)
def pathfinder_fit(approximate, fn, names, values, key, initial_position, lbfgs=()):
    """``approximate(key, ld, initial_position, **dict(lbfgs))``: the L-BFGS path, whole, as one program.

    ``approximate`` is ``blackjax.vi.pathfinder.approximate``, looked up by the caller at call time (a
    static argument, so a replaced function is its own cache entry). ``lbfgs`` is a sorted tuple of
    ``(name, value)`` pairs among ``num_samples``, ``maxiter``, ``maxcor``, ``maxls``, ``gtol`` and ``ftol``
    (static: they are Python constants of the trace, so another value is another program); the ones not
    given keep BlackJAX's defaults. Returns the fitted `PathfinderState` and the number of L-BFGS iterations that
    ran; the traversed path (``maxiter + 1`` states) is not kept.

    BlackJAX drops L-BFGS's own iteration count. Once L-BFGS stops, its scan repeats the last row of the path (the
    no-op branch of ``_minimize_lbfgs``), so the iterations that ran are the rows that differ from the row before
    (NaN equal to NaN). A count equal to ``maxiter`` means L-BFGS used every iteration it was given.
    """
    state, info = approximate(
        rng_key=key, initial_position=initial_position, logdensity_fn=_logdensity(fn, names, values),
        **dict(lbfgs))
    flat = jax.vmap(lambda p: ravel_pytree(p)[0])(info.path.position)
    a, b = flat[1:], flat[:-1]
    same = (a == b) | (jnp.isnan(a) & jnp.isnan(b))
    return state, jnp.sum(~jnp.all(same, axis=1))


@partial(jax.jit, static_argnames=("sample", "num_samples"))
def pathfinder_draw(sample, key, state, num_samples):
    """``num_samples`` draws from the fitted Pathfinder Gaussian (``blackjax.vi.pathfinder.sample``)."""
    draws, _ = sample(key, state, num_samples)
    return draws


# --------------------------------------------------------------------------------------------------
# step-size search
# --------------------------------------------------------------------------------------------------
def _search_step_size(ld, inverse_mass_matrix, key, position):
    """BlackJAX's doubling/halving search (target acceptance 0.651) from one position, for one metric."""
    return bj.adaptation.step_size.find_reasonable_step_size(
        key,
        lambda eps: bj.hmc(ld, eps, inverse_mass_matrix, num_integration_steps=1).step,
        bj.hmc(ld, 1e-2, inverse_mass_matrix, num_integration_steps=1).init(position),
        initial_step_size=1e-2,
        target_accept=0.651,
    )


@partial(jax.jit, static_argnames=_STATIC)
def search_step_size(fn, names, values, inverse_mass_matrix, key, position):
    """The step-size search from one position (the serial reference of `search_step_sizes`)."""
    return _search_step_size(_logdensity(fn, names, values), inverse_mass_matrix, key, position)


@partial(jax.jit, static_argnames=_STATIC + ("take",))
def search_step_sizes(fn, names, values, inverse_mass_matrix, key, stack, *, take):
    """The initial step size from the first ``take`` rows of ``stack``: ONE program for any number of feasible probes.

    ``stack`` is a dict of ``(n, ...)`` arrays (positions). Returns ``(eps0, eps, n_ok)``: the geometric median
    of the finite positive step sizes the search finds from those of the ``take`` probes that have a finite
    log-density (NaN if none), the ``take`` step sizes (the first ``n_ok`` are the feasible probes', in order),
    and how many probes are feasible.

    * A probe at a non-finite log-density has NaN acceptance, which the search reads as "halve forever", so
      those are left out. The feasible ones are moved to the front in their original order, and the slots
      behind them repeat the first feasible probe, so every row of the batch is a valid search and the
      result depends on the probes only through the feasible ones.
    * Probe ``i`` uses ``fold_in(key, i)``, which is ``split(key, n)[i]`` (`jax_threefry_partitionable`) and
      does not depend on ``n``.
    * ``jax.vmap`` of the search is the serial search per probe: JAX lowers a batched ``while`` predicate by
      evaluating the body for every element and keeping the previous carry for the finished ones.
    """
    probes = jax.tree_util.tree_map(lambda x: x[:take], stack)
    ld = _logdensity(fn, names, values)
    n = take
    feasible = jax.vmap(lambda p: jnp.isfinite(ld(p)))(probes)
    n_ok = jnp.sum(feasible)
    order = jnp.argsort(~feasible, stable=True)
    slot = order[jnp.where(jnp.arange(n) < n_ok, jnp.arange(n), 0)]
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(n))
    eps = jax.vmap(lambda k, p: _search_step_size(ld, inverse_mass_matrix, k, p))(
        keys, jax.tree_util.tree_map(lambda x: x[slot], probes))
    good = (jnp.arange(n) < n_ok) & jnp.isfinite(eps) & (eps > 0.0)
    # geometric median: the values are dyadic, so a linear median biases upward
    return jnp.exp(jnp.nanmedian(jnp.where(good, jnp.log(eps), jnp.nan))), eps, n_ok


# --------------------------------------------------------------------------------------------------
# initial points
# --------------------------------------------------------------------------------------------------
# `fold_in` index of the jitter stream within a root key made from `pm_seed`. `_run_pathfinder` splits the key
# made from `seed` in five (`fold_in` 0..4, `jax_threefry_partitionable`), and `pm_seed` defaults to `seed`.
JITTER_STREAM = 17


@partial(jax.jit, static_argnames=("fn",))
def initial_point(fn, values):
    """The model's un-jittered initial point: a tuple of arrays, one per value variable (`lifted.lift_initial_point`)."""
    return tuple(fn((), values))


@partial(jax.jit, static_argnames=_STATIC + ("chains",))
def jittered_initial_points(fn, names, values, base, key, max_retries, *, chains):
    """``chains`` copies of ``base`` plus a U(-1, 1) jitter per coordinate, redrawn while the log-density is not finite.

    PyMC's ``jitter+adapt_diag`` start (``pymc.sampling.mcmc._init_jitter``) in JAX: the jitter is added in the
    space of the value variables, and a chain whose point has a non-finite log-density gets a whole new jitter,
    up to ``max_retries`` times (``jitter_max_retries``), while the chains that are finite keep theirs.
    ``base`` is a tuple of arrays in ``names`` order (`initial_point`). Attempt ``a`` draws all chains' jitter at
    once, from ``fold_in(fold_in(key, a), variable index)``, and keeps the rows of the chains that still need one.
    Returns ``(points, finite)``: a tuple of ``(chains, ...)`` arrays, and per chain whether its point has a
    finite log-density at the end (False only after ``max_retries`` retries).
    """
    ld = _logdensity(fn, names, values)

    def draw(attempt):
        k = jax.random.fold_in(key, attempt)
        return tuple(
            b + jax.random.uniform(jax.random.fold_in(k, j), (chains,) + b.shape, b.dtype, -1.0, 1.0)
            for j, b in enumerate(base))

    def finite(points):
        return jax.vmap(lambda p: jnp.isfinite(ld(p)))(points)

    def retry(state):
        attempt, points, ok = state
        fresh = draw(attempt + 1)
        points = tuple(jnp.where(ok.reshape((-1,) + (1,) * (p.ndim - 1)), p, f) for p, f in zip(points, fresh))
        return attempt + 1, points, ok | finite(fresh)

    points = draw(0)
    _, points, ok = lax.while_loop(
        lambda state: (state[0] < max_retries) & ~jnp.all(state[2]), retry, (jnp.int32(0), points, finite(points)))
    return points, ok


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


# --------------------------------------------------------------------------------------------------
# the two loops: one jitted chunk program each, driven from the host
# --------------------------------------------------------------------------------------------------
# Halton bit depth of the trajectory jitter. It used to be ceil(log2(W + 1000)), which made W a constant
# of the warmup program. 30 gives the same value whenever i + 1 < 2**bits, and the old depth was always
# above that for the steps a run takes; blackjax's `halton_sequence` needs max_bits below the integer
# width (32 for int32, when x64 is off).
HALTON_BITS = 30

# Steps per chunk. A chunk program has a static length, so these are the loop bounds a run can change
# for free: any W (D) is `ceil(W / WARMUP_CHUNK)` calls of ONE program, the last with a smaller active
# count. They are read at call time (tests set them small to exercise the multi-chunk path).
#
# Why 50 (measured on the CPU, 8 Schools, W=200, D=500; the probe is at the tag `archive/pre-cleanup`): a chunk costs the host a dispatch and
# the copy of its outputs, about 0.05-0.2 ms with one chunk kept in flight (0.3 ms without), whatever the
# chunk length. At C=32, the cheapest realistic case (0.25 ms per step), a chunk of 50 steps is 12 ms of
# compute, so the overhead is under 1.5%; at C=1,024 (8 ms per step) it is under 0.1%, and chunk lengths
# from 5 to 500 are indistinguishable. Longer chunks buy nothing and cost more: the last chunk is padded
# (its skipped rows are written as zeros and copied back), which is 50 rows of waste at most, and a chunk
# is also the unit of progress-bar granularity.
WARMUP_CHUNK = 50
SAMPLE_CHUNK = 50
TRANSFORM_CHUNK = 50        # draws per call of `transform_draws` (the last call is padded)


class WarmupScanInfo(NamedTuple):
    """The only per-step fields of the warmup scan that a consumer reads.

    num_integration_steps -- exact gradient counts (A8: sum of steps, never an estimate)
    is_divergent          -- divergence counters and the harmonic-acceptance mask
    acceptance_rate       -- the ChEES step-size signal (harmonic mean over chains)
    """

    num_integration_steps: Any
    is_divergent: Any
    acceptance_rate: Any


class SampledState(NamedTuple):
    """The sampling scan's state output: only the leaves a consumer reads.

    `position` feeds the posterior, `logdensity` feeds sample_stats `lp`.
    `momentum` / `logdensity_grad` are not needed after the step and used to be
    stacked for every (draw, chain) anyway.
    """

    position: Any
    logdensity: Any


class SampledInfo(NamedTuple):
    """The sampling scan's info output: exactly the six sample_stats fields minus lp.

    Field names match blackjax's HMCInfo so `_to_inference_data` reads them unchanged.
    The dropped fields (momentum, proposal.*) duplicate the state leaves at ~2x their
    volume -- the whole point of the projection.
    """

    is_divergent: Any
    acceptance_rate: Any
    is_accepted: Any
    num_integration_steps: Any
    energy: Any
    # Delayed rejection only (`sampling_chunk(..., dr_factor=a)`); None otherwise, so the default stack is unchanged.
    # is_divergent stays the FIRST-stage divergence: the retry never hides the diagnostic.
    retried: Any = None                  # first stage divergent and rejected: a second proposal was made
    retry_accepted: Any = None           # ... and accepted
    retry_divergent: Any = None          # ... and the second proposal itself diverged
    retry_integration_steps: Any = None  # leapfrog steps the retry branch ran for this chain (0 if not run)


def chees_harmonic_acceptance(acc_vals, is_divergent):
    """The acceptance ChEES actually adapts on: harmonic mean over the non-divergent chains.

    Mirrors `blackjax.adaptation.chees_adaptation` (harmonic_mean, :358-363), including the
    non-finite -> 0 guard. The arithmetic mean of per-trajectory acceptances is systematically
    higher than this (Jensen), so displaying it hides exactly the heterogeneity the step-size
    adaptation reacts to -- it already caused a wrong diagnosis in this project.
    """
    hm = 1.0 / jnp.mean(1.0 / acc_vals, where=~is_divergent)
    return jnp.where(jnp.isfinite(hm), hm, 0.0)


def chees_harmonic_acceptance_np(acc_vals, is_divergent) -> float:
    """`chees_harmonic_acceptance` on the host (NumPy): what the progress bars read between chunks."""
    acc, ok = np.asarray(acc_vals, dtype=np.float64), ~np.asarray(is_divergent, dtype=bool)
    if not ok.any():
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = 1.0 / np.mean(1.0 / acc[ok])
    return float(hm) if np.isfinite(hm) else 0.0


def _next_random_arg(i):
    return i + 1


def halton_jitter(i, jitter_amount):
    """The trajectory-length jitter in (0, 1] for step index ``i`` (blackjax's Halton sequence)."""
    return dynamic_hmc.halton_sequence(i, HALTON_BITS) * jitter_amount + (1.0 - jitter_amount)


def integration_steps(random_generator_arg, trajectory_length_adjusted, jitter_amount=1.0):
    """``ceil(jitter * L)`` leapfrog steps, ``L = T / eps`` the nominal count (the kernel's step-count rule)."""
    return jnp.asarray(
        jnp.ceil(halton_jitter(random_generator_arg, jitter_amount) * trajectory_length_adjusted),
        dtype=int)


_ADAPTER = ("optim", "target_accept", "decay_rate", "jitter_amount", "max_leapfrog_steps")


class ChEESTrace(NamedTuple):
    """Per-step scalar trace of one ChEES update, built beside BlackJAX's adapter state.

    step
        1-based update index (the step number just executed).
    step_size
        The step size the state carries after this update.
    trajectory_length
        The trajectory length (time, T = L * eps) the state carries after this update -- what the kernel
        consumes next. trajectory_length / step_size is the nominal leapfrog count L at this step.
    harmonic_acceptance
        `chees_harmonic_acceptance` of this update's chains: the quantity ChEES adapts the step size
        against, not the arithmetic mean of per-chain acceptances.
    """

    step: Any
    step_size: Any
    trajectory_length: Any
    harmonic_acceptance: Any


def _adapter(optim, target_accept, decay_rate, jitter_amount, max_leapfrog_steps):
    """BlackJAX's ChEES adapter ``(init, update)``, its ``update`` returning ``(state, ChEESTrace)``.

    ``optim`` is an Adam learning rate (a stable static argument: an optax object is a new value on every
    call) or an optax transformation. The state is BlackJAX's, untouched; the trace is read off it.
    """
    if not hasattr(optim, "init"):
        optim = optax.adam(optim)
    init, update = chees_adaptation.base(
        lambda i: halton_jitter(i, jitter_amount), _next_random_arg, optim, target_accept, decay_rate,
        max_leapfrog_steps)

    def update_with_trace(state, proposed_positions, proposed_momentums, initial_positions, acceptance,
                          is_divergent, inverse_mass_matrix):
        new = update(state, proposed_positions, proposed_momentums, initial_positions, acceptance,
                     is_divergent, inverse_mass_matrix)
        return new, ChEESTrace(new.step, new.step_size, new.trajectory_length,
                               chees_harmonic_acceptance(acceptance, is_divergent))

    return init, update_with_trace


def _strong(tree):
    """Leaves as arrays of a definite dtype (no weak type): a carry that goes into a jitted chunk and
    comes back out must have one abstract type, or the second chunk would be a second program."""
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.result_type(x)), tree)


def _like(new, old):
    """``new`` with the dtypes of ``old``, leaf by leaf (the carry of a chunk keeps its type)."""
    return jax.tree_util.tree_map(lambda a, b: jnp.asarray(a).astype(jnp.asarray(b).dtype), new, old)


def _num_chains(tree) -> int:
    return int(jax.tree_util.tree_leaves(tree)[0].shape[0])


def _chunk_scan(step, carry, start, n, chunk):
    """``chunk`` steps of ``step(carry, global_index) -> (carry, outputs)``, the first ``n`` of them real.

    The predicate is a scalar outside any vmap, so XLA runs a real conditional: a skipped step costs no
    model evaluation, passes the carry through, and writes zeros to the (host-discarded) outputs.
    """
    spec = jax.eval_shape(lambda c: step(c, start)[1], carry)
    zeros = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), spec)

    def body(c, j):
        return lax.cond(j < n, lambda c: step(c, start + j), lambda c: (c, zeros), c)

    return lax.scan(body, carry, jnp.arange(chunk, dtype=jnp.int32))


@partial(jax.jit, static_argnames=_STATIC + _ADAPTER)
def warmup_init(fn, names, values, positions, initial_step_size, *, optim, target_accept, decay_rate,
                jitter_amount, max_leapfrog_steps):
    """The warmup's start: HMC states for a dict of ``(chains, ...)`` positions, and the adapter's state.

    ``initial_step_size`` is a value (dynamic).
    """
    ld = _logdensity(fn, names, values)
    init_adapt, _ = _adapter(optim, target_accept, decay_rate, jitter_amount, max_leapfrog_steps)
    states = jax.vmap(lambda p: dynamic_hmc.init(p, ld, 0))(positions)
    return _strong(states), _strong(init_adapt(0, initial_step_size))


@jax.jit
def ensemble_metric(positions, old, shrinkage):
    """The metric re-estimated from the chains: ``(1 - shrinkage) * cov + shrinkage * old``, and whether it was used.

    ``positions`` is the warmup carry's dict of ``(chains, ...)`` positions; each chain is raveled as the
    kernel ravels its state (sorted keys), so the result is in kernel order like ``old``. ``old`` decides
    the form: a 2-D metric gets the dense covariance, a 1-D one the variances. Chains with a non-finite
    coordinate are left out. ``old`` is returned unchanged (and the flag is False) unless the shrunk
    estimate is finite and positive definite (a Cholesky factor exists, or every variance is positive)
    and at least two chains are usable.
    """
    x = jax.vmap(lambda p: ravel_pytree(p)[0])(positions)
    ok = jnp.isfinite(x).all(axis=1)
    n = jnp.sum(ok)
    xs = jnp.where(ok[:, None], x, 0.0)
    xc = jnp.where(ok[:, None], xs - jnp.sum(xs, axis=0) / n, 0.0)
    old = jnp.asarray(old)
    if old.ndim == 2:
        est = (xc.T @ xc) / (n - 1)
        new = (1.0 - shrinkage) * est + shrinkage * old
        good = jnp.isfinite(jnp.linalg.cholesky(new)).all()
    else:
        est = jnp.sum(xc * xc, axis=0) / (n - 1)
        new = (1.0 - shrinkage) * est + shrinkage * old
        good = jnp.isfinite(new).all() & (new > 0.0).all()
    good = good & (n >= 2)
    return jnp.where(good, new, old).astype(old.dtype), good


@jax.jit
def restart_step_size(astate, step_size):
    """The adapter state with its step size restarted at ``step_size``: a fresh dual-averaging state there,
    and the step size's moving average set to it. The trajectory length, its moving average, the Adam state,
    the jitter index and the step counter carry on. Used after the metric changed under the warmup."""
    da_init, _, _ = dual_averaging.dual_averaging()
    step_size = jnp.asarray(step_size, dtype=astate.step_size.dtype)
    new = astate._replace(step_size=step_size, log_step_size_moving_average=jnp.log(step_size),
                          da_state=da_init(step_size))
    return _like(new, astate)


@partial(jax.jit, static_argnames=_STATIC + _ADAPTER + ("chunk", "stack_positions"))
def warmup_chunk(fn, names, values, carry, inverse_mass_matrix, key, start, n, *, chunk, stack_positions,
                 optim, target_accept, decay_rate, jitter_amount, max_leapfrog_steps):
    """``chunk`` ChEES warmup steps (the first ``n`` real) from ``carry = (HMC states, adapter state)``.

    Step ``j`` is global step ``start + j``: its key is ``fold_in(key, start + j)`` and its Halton jitter
    index is the one carried in the state, so a run split into chunks of any size is the same run.
    Returns ``(carry, ys)``, ``ys`` the per-step stack (``chunk`` rows; rows ``>= n`` are zeros):
    ``(WarmupScanInfo, ChEESTrace)``, preceded by the positions when ``stack_positions`` (the stop rule
    reads them; stacking them is the memory wall at C=1,024).
    """
    ld = _logdensity(fn, names, values)
    _, update_adapt = _adapter(optim, target_accept, decay_rate, jitter_amount, max_leapfrog_steps)
    kernel = dynamic_hmc.build_kernel(
        next_random_arg_fn=_next_random_arg,
        integration_steps_fn=partial(integration_steps, jitter_amount=jitter_amount))
    imm = jnp.asarray(inverse_mass_matrix)
    # ChEES whitens its trajectory-length criterion by 1/sqrt(imm), so it needs the diagonal of the
    # inverse mass matrix (variances). The HMC kernel itself still gets the full dense metric.
    imm_diag = jnp.diag(imm) if imm.ndim == 2 else imm
    num_chains = _num_chains(carry[0].position)

    def step(c, i):
        states, astate = c
        keys = jax.random.split(jax.random.fold_in(key, i), num_chains)
        _step = partial(
            kernel,
            logdensity_fn=ld,
            step_size=astate.step_size,
            inverse_mass_matrix=imm,
            # BlackJAX >=1.7: the trajectory-length adjustment is unpacked into
            # integration_steps_fn at call time, not passed as a named kwarg.
            integration_steps_params=(astate.trajectory_length / astate.step_size,),
        )
        new_states, info = jax.vmap(_step)(keys, states)

        # accept field name can be acceptance_probability or acceptance_rate depending on BJ version
        acc_vals = getattr(info, "acceptance_probability", None)
        if acc_vals is None:
            acc_vals = info.acceptance_rate

        # Scan output projection: the only fields a downstream consumer reads. Everything else
        # in HMCInfo (momentum, proposal.*) duplicates state leaves the scan does not need to
        # keep, and was the warmup memory wall at large C (~14 GB at d=175, W=2,000, C=1,024).
        proj = WarmupScanInfo(info.num_integration_steps, info.is_divergent, acc_vals)

        # Update adaptation state (BlackJAX's adapter state + the scalar trace)
        new_astate, trace = update_adapt(
            astate,
            info.proposal.position,
            info.proposal.momentum,
            states.position,
            acc_vals,              # accepts vector of acceptances
            info.is_divergent,
            imm_diag,              # 1.7 requires the diagonal metric here
        )
        new_c = _like((new_states, new_astate), c)
        if stack_positions:
            return new_c, (new_states.position, proj, trace)
        return new_c, (proj, trace)

    return _chunk_scan(step, carry, start, n, chunk)


# Delayed rejection: chains retried per batch (`delayed_rejection_step`). Retries are rare (they follow
# divergences), so a small batch keeps the padded waste small; more retries in one step take more batches.
DR_BATCH = 8
# The acceptance test of retry stage k uses fold_in(key, DR_STREAM + k), key being the chain's key for the
# iteration. BlackJAX's kernel splits that key in two (momentum, its own accept test), and with JAX's partitionable
# keys (the default) fold_in(key, i) is split(key, n)[i]: a small i would reuse one of those two numbers.
DR_STREAM = 0x44520000


@partial(jax.jit, static_argnames=_STATIC + ("chunk", "jitter_amount", "dr_factor", "dr_batch", "dr_stages"))
def sampling_chunk(fn, names, values, state, inverse_mass_matrix, step_size, traj_len_adj, key, start, n, *,
                   chunk, jitter_amount, dr_factor=None, dr_batch=None, dr_stages=2):
    """``chunk`` dynamic-HMC draws (the first ``n`` real) from ``state`` (a batch of `DynamicHMCState`).

    The frozen step size, metric and trajectory-length adjustment are arguments of the kernel call, as
    BlackJAX's dynamic kernel takes them (`blackjax/mcmc/dynamic_hmc.py:101-111`). Returns
    ``(state, (SampledState, SampledInfo))``, the second the per-draw stack (rows ``>= n`` are zeros).

    ``dr_factor`` (opt-in, an integer a >= 2): delayed rejection on divergence (`delayed_rejection_step`).
    None runs BlackJAX's kernel alone, the same program as before the option existed.
    """
    ld = _logdensity(fn, names, values)
    kernel = dynamic_hmc.build_kernel(
        next_random_arg_fn=_next_random_arg,
        integration_steps_fn=partial(integration_steps, jitter_amount=jitter_amount))
    imm = jnp.asarray(inverse_mass_matrix)
    num_chains = _num_chains(state.position)

    def step(s, i):
        keys = jax.random.split(jax.random.fold_in(key, i), num_chains)
        _step = partial(kernel, logdensity_fn=ld, step_size=step_size, inverse_mass_matrix=imm,
                        integration_steps_params=(traj_len_adj,))
        new_state, info = jax.vmap(_step)(keys, s)
        retry = ()
        if dr_factor is not None:
            new_state, retry = delayed_rejection_step(ld, imm, step_size, dr_factor, keys, s, new_state, info,
                                                      batch=dr_batch, stages=dr_stages)
        # Projection: keep only what _to_inference_data reads (position + logdensity for the
        # posterior/lp, and the five sample_stats fields). momentum, proposal.position,
        # proposal.momentum and proposal.logdensity_grad duplicate state leaves and are the
        # bulk of the scan volume.
        return _like(new_state, s), (
            SampledState(new_state.position, new_state.logdensity),
            SampledInfo(info.is_divergent, info.acceptance_rate, info.is_accepted,
                        info.num_integration_steps, info.energy, *retry))

    return _chunk_scan(step, state, start, n, chunk)


def _log1m_accept(delta_h):
    """log(1 - min(1, exp(-delta_h))), delta_h = H(end) - H(start): -inf where the move is always accepted."""
    return jnp.where(delta_h > 0, jnp.log(-jnp.expm1(-jnp.where(delta_h > 0, delta_h, 1.0))), -jnp.inf)


def _delta_h(energy, z0, z1):
    """H(z1) - H(z0), with NaN as +inf (BlackJAX's `safe_energy_diff` convention, sign reversed)."""
    d = energy(z1) - energy(z0)
    return jnp.where(jnp.isnan(d), jnp.inf, d)


def _log1m_exp(log_a):
    """log(1 - exp(log_a)) for log_a <= 0 (-inf at log_a = 0, 0 at log_a = -inf)."""
    safe = jnp.where(log_a < 0, log_a, -1.0)
    return jnp.where(log_a < 0, jnp.log(-jnp.expm1(safe)), -jnp.inf)


def delayed_rejection_step(ld, inverse_mass_matrix, step_size, factor, keys, old, new, info,
                           divergence_threshold=1000.0, batch=None, stages=2):
    """Later proposals of k-stage delayed-rejection HMC, on the chains whose earlier proposals diverged.

    Modi, Barnett & Carpenter (2023), "Delayed rejection Hamiltonian Monte Carlo for sampling multiscale
    distributions", Bayesian Analysis, eqs. (29)-(30), with the retry probability p_(j+1)(x) = 1{proposal j from
    x diverged}: proposal j + 1 is made only when proposal j diverged (and was rejected). Proposal j is
    F_j = flip . L^(a^(j-1) n)_(eps / a^(j-1)) (same integration time, step / a^(j-1)); F_1 is BlackJAX's. All are
    volume-preserving involutions, and n is the iteration's step count (a function of the iteration only), so F_j
    at any state is the same map. Proposal k, y = F_k(x), is accepted with probability

        min{1, pi(y) prod_(j<k) [1 - alpha_j(y)] p_(j+1)(y) / (pi(x) prod_(j<k) [1 - alpha_j(x)] p_(j+1)(x))},

    where the factors at y belong to the ghosts: the proposals a chain at y would have made (`ghost_terms`,
    recursively). With the indicator p, a stage-k proposal is accepted only if the reverse stage-j proposals from y
    also diverge, j < k; this keeps the kernel exact when retries are restricted to divergences. stages = 2 is the
    two-stage scheme (one retry at step / a).

    Cost: stage k costs a^(k-1) n leapfrog steps for the proposal plus its ghosts (stage 2: n; stage 3: 6 n at
    a = 4), per chain that reaches it, run `batch` chains at a time (`DR_BATCH`), so a step without retries costs
    nothing and one with r costs ceil(r / batch) batches per stage. On a GPU the wall time follows the sequential
    leapfrog steps, not the gradient count: a step with any retry takes about (a + 1) trajectories longer at
    stage 2 (a GEV regression, 1,024 chains, a = 4, retries on 10% of steps: gradients +0.06%, sampling time +30%).
    Returns (state, retry fields of `SampledInfo`): retried (stage 2 made), retry_accepted (some later stage
    accepted), retry_divergent (the last stage made diverged and nothing was accepted), retry_integration_steps.
    """
    batch = DR_BATCH if batch is None else batch
    metric = bj_metrics.default_metric(inverse_mass_matrix)
    integrate = bj_trajectory.static_integration(bj_integrators.velocity_verlet(ld, metric.kinetic_energy))
    energy = bj_trajectory.hmc_energy(metric.kinetic_energy)
    n = info.num_integration_steps

    def prop(z, j, n_c):
        """Proposal j from z."""
        return bj_hmc.flip_momentum(integrate(z, step_size / factor ** (j - 1), factor ** (j - 1) * n_c))

    def ghost_terms(z, m, n_c):
        """sum_(j<=m) log([1 - alpha_j(z)] p_(j+1)(z)) for a chain at z: -inf once some p is 0."""
        total = jnp.zeros((), dtype=energy(z).dtype)
        for j in range(1, m + 1):
            y = prop(z, j, n_c)
            dh = _delta_h(energy, z, y)
            ty = ghost_terms(y, j - 1, n_c) if j > 1 else 0.0
            reach = jnp.isfinite(total)
            log_a = jnp.where(jnp.isfinite(dh) & reach, jnp.minimum(0.0, -dh + ty - jnp.where(reach, total, 0.0)),
                              -jnp.inf)
            total = total + _log1m_exp(log_a) + jnp.where(dh > divergence_threshold, 0.0, -jnp.inf)
        return total

    def stage(k):
        def one(key, pos, lp, grad, momentum, s_x, n_c):
            x = bj_integrators.IntegratorState(pos, momentum, lp, grad)
            y = prop(x, k, n_c)
            dh = _delta_h(energy, x, y)
            log_a = jnp.minimum(0.0, -dh + ghost_terms(y, k - 1, n_c) - s_x)
            log_a = jnp.where(jnp.isfinite(dh), log_a, -jnp.inf)
            accept = jnp.log(jax.random.uniform(jax.random.fold_in(key, DR_STREAM + k))) < log_a
            # s_x for stage k + 1 if this one is rejected; its p is 1 (the chain only gets there if it diverged)
            return y, accept, dh > divergence_threshold, s_x + _log1m_exp(log_a)
        return one

    def run_stage(k, need, s_x, state):
        """Stage k for the chains in `need` (packed, `batch` at a time, gathered and scattered back)."""
        num = need.shape[0]
        idx = jnp.nonzero(need, size=num, fill_value=num)[0]
        count = jnp.sum(need)
        inputs = (keys, old.position, old.logdensity, old.logdensity_grad, info.momentum, s_x, n)
        zeros = jnp.zeros_like(need)
        one = stage(k)

        def body(carry):
            b, pos, lp, grad, take, div, s_next = carry
            sl = lax.dynamic_slice(idx, (b * batch,), (batch,))
            at = jnp.minimum(sl, num - 1)
            sel = jax.tree_util.tree_map(lambda u: u[at], inputs)
            y, accept, d, s_new = jax.vmap(one)(*sel)
            put = lambda base, upd: base.at[sl].set(upd, mode="drop")
            keep = lambda u, v: jnp.where(accept.reshape(accept.shape + (1,) * (u.ndim - 1)), u, v)
            pos = jax.tree_util.tree_map(lambda base, u: put(base, keep(u, base[at])), pos, y.position)
            lp = put(lp, keep(y.logdensity, lp[at]))
            grad = jax.tree_util.tree_map(lambda base, u: put(base, keep(u, base[at])), grad, y.logdensity_grad)
            return b + 1, pos, lp, grad, put(take, accept), put(div, d), put(s_next, s_new)

        init = (jnp.asarray(0, jnp.int32), state.position, state.logdensity, state.logdensity_grad, zeros, zeros,
                jnp.full_like(s_x, -jnp.inf))
        _, pos, lp, grad, take, div, s_next = lax.while_loop(lambda c: c[0] * batch < count, body, init)
        return state._replace(position=pos, logdensity=lp, logdensity_grad=grad), take, div, s_next

    def steps(k):
        """Leapfrog steps (in units of n) of stage k for one chain: its proposal and its ghosts."""
        def ghost(m):
            return sum(factor ** (j - 1) + ghost(j - 1) for j in range(1, m + 1))
        return factor ** (k - 1) + ghost(k - 1)

    need = info.is_divergent & ~info.is_accepted
    retried = need
    # stage 1 was rejected and diverged: s_x = log([1 - alpha_1(x)] p_2(x)) = log1p(-p1)
    s_x = jnp.log1p(-info.acceptance_rate)
    state, accepted, last_div = new, jnp.zeros_like(need), jnp.zeros_like(need)
    cost = jnp.zeros_like(n)
    for k in range(2, stages + 1):
        state, take, div, s_x = run_stage(k, need, s_x, state)
        cost = cost + jnp.where(need, steps(k) * n, 0)
        accepted = accepted | take
        last_div = jnp.where(need, div & ~take, last_div)
        need = need & ~take & div          # the next stage only where this one diverged
    return state, (retried, accepted, last_div, cost)


@functools.lru_cache(maxsize=64)
def _plain_fn(ld_fn, names):
    return lambda seq, values: [ld_fn(dict(zip(names, seq)))]


def plain_log_density(ld_fn: Callable, names) -> LogDensity:
    """A `LogDensity` for a plain ``position -> scalar`` callable, compiled in as a constant (as before).

    The wrapper is cached per callable, so the same callable is one static argument for jit, not a new
    program per call.
    """
    names = tuple(names)
    try:
        fn = _plain_fn(ld_fn, names)
    except TypeError:                                   # an unhashable callable: one program per call
        fn = lambda seq, values: [ld_fn(dict(zip(names, seq)))]       # noqa: E731
    return LogDensity(fn, names, ())


class HostStack:
    """Per-step outputs of the chunks, gathered on the host (NumPy) into arrays of ``total`` rows.

    A chunk returns ``chunk`` rows of which ``n`` are real. The rows are cut with NumPy slicing: a device
    slice with a W- or D-dependent bound is one XLA compile per bound.
    """

    def __init__(self, total: int):
        self.total = int(total)
        self._buf = None
        self._treedef = None

    def add(self, start: int, n: int, out) -> None:
        leaves, treedef = jax.tree_util.tree_flatten(out)
        host = [np.asarray(x) for x in leaves]
        if self._buf is None:
            self._treedef = treedef
            self._buf = [np.empty((self.total,) + x.shape[1:], x.dtype) for x in host]
        a, b = start, start + n
        if b > a:
            for buf, x in zip(self._buf, host):
                buf[a:b] = x[a - start:b - start]

    def result(self):
        return jax.tree_util.tree_unflatten(self._treedef, self._buf)


def run_chunks(call: Callable, carry, total: int, size: int, consume: Callable):
    """Drive ``total`` steps as chunks of ``size``: ``call(carry, start, n) -> (carry, outputs)``.

    ``consume(start, n, outputs)`` is handed each chunk's (device) outputs. One chunk is kept in flight:
    chunk k+1 is dispatched (its carry is chunk k's device output, so there is nothing to wait for)
    BEFORE chunk k's outputs are converted to NumPy, so the host transfer and the progress bar overlap
    the next chunk's compute. Returns the final carry. ``total == 0`` runs one empty chunk, so the
    output shapes are still known.
    """
    pending = None
    for start in range(0, total, size) if total > 0 else (0,):
        n = min(size, total - start)
        carry, out = call(carry, start, n)
        if pending is not None:
            consume(*pending)
        pending = (start, n, out)
    consume(*pending)
    return carry
