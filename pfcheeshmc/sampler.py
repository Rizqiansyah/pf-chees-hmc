import jax, jax.numpy as jnp
import blackjax as bj
from jax.flatten_util import ravel_pytree
from jax import lax
import optax

import pymc as pm
import pytensor
from pymc import modelcontext
from pymc.sampling.mcmc import all_continuous
from pymc.util import get_default_varnames
import arviz as az

from tqdm.auto import tqdm
from .utils import *
from . import compiled
from .compiled import LogDensity
from .compilation_cache import ensure_default_compilation_cache
from .lifted import lift_graph, lift_initial_point, lift_logp

from typing import Tuple, Optional, Dict, Any, NamedTuple
import numpy as np
from datetime import date
import io
import json
import re
import logging
import contextlib
import warnings

__all__ = ["sample_pf_chees_hmc"]

# The stratified start pick scores at most this many Pathfinder pool rows per chain, whatever the pool size
# (under the default pool, max(4C, 4d^2) rows, d > sqrt(C) would otherwise score 4d^2 of them).
START_SCORE_PER_CHAIN = 4

# PyMC's default (`pm.init_nuts(jitter_max_retries=...)`): redraws of the jitter of a chain whose point has a
# non-finite log-density.
JITTER_MAX_RETRIES = 10

# The keys each stage reads from its dict of options (`pathfinder_kwargs`, `step_size_tune_kwargs`, `chees_kwargs`,
# `dynamic_hmc_kwargs`). `PfCheesHMC.sample` adds ``chains`` to every dict, ``draws`` to the sampling one. A key outside
# the set raises a TypeError: an option that was removed, or a typo, is never ignored.
_COMMON_OPTIONS = frozenset({"chains", "seed"})
_PATHFINDER_OPTIONS = _COMMON_OPTIONS | frozenset({
    "draws", "ridge", "init_source", "imm_source", "pf_pool_cap", "score_chunk",
    "pm_init", "pm_tune", "pm_seed", "pm_chains",
    "pf_jitter_eps", "pf_max_backtracks", "pf_max_restarts", "pf_restart_radius",
    "fisher_draws", "max_rank", "fisher_gamma", "fisher_cutoff",
    "pf_maxiter", "pf_maxcor", "pf_maxls", "pf_gtol", "pf_ftol", "pf_elbo_samples"})
# pathfinder_kwargs key -> keyword of blackjax.vi.pathfinder.approximate
_PF_LBFGS_OPTIONS = {"pf_maxiter": "maxiter", "pf_maxcor": "maxcor", "pf_maxls": "maxls", "pf_gtol": "gtol",
                     "pf_ftol": "ftol", "pf_elbo_samples": "num_samples"}
_STEP_SIZE_OPTIONS = _COMMON_OPTIONS | frozenset({"eps0_probes"})
_CHEES_OPTIONS = _COMMON_OPTIONS | frozenset({
    "draws", "adam_lr", "target_accept", "max_leapfrog_steps", "warmup_stop", "progressbar",
    "metric_updates", "metric_shrinkage"})
_DYNAMIC_HMC_OPTIONS = _COMMON_OPTIONS | frozenset({"draws", "progressbar", "delayed_rejection"})
# Delayed rejection (opt-in): a retry rate above this share of transitions is a warning (see `_run_dynamic_hmc`).
DR_RETRY_WARN = 1e-3
_SAMPLE_OPTIONS = frozenset({"chains", "draws"})


def _check_options(where, given, allowed):
    """Raise a TypeError naming every key of ``given`` that is not in ``allowed``."""
    unknown = sorted(set(given) - set(allowed), key=str)
    if unknown:
        names = ", ".join(repr(k) for k in unknown)
        raise TypeError(
            f"{where} got an unknown option {names} (an option that was removed raises as well). "
            f"The options are: {', '.join(sorted(allowed))}")


def _warn_if_x64_contradicts_floatx():
    """Warn when ``jax_enable_x64`` is on under a floatX narrower than float64.

    PyTensor's JAX dispatch sets ``jax_enable_x64 = (floatX == "float64")`` when it is imported. Anything that
    turns it on afterwards (the example notebooks' first cell does) leaves the value variables float32 and the
    JAX defaults float64: a data array that PyMC left float64 makes the log-density float64 at a float32
    position, and L-BFGS fails inside optax's ``lax.cond`` ("cond branches must have equal output types"). The
    HMC loops draw their momentum in the default dtype, so they cannot carry float32 positions either.
    """
    floatx = pytensor.config.floatX
    if floatx != "float64" and jax.config.jax_enable_x64:
        warnings.warn(
            f"floatX is {floatx} but jax_enable_x64 is on: JAX will compute in float64 beside the {floatx} value "
            "variables, which the sampler does not support (it can fail with \"cond branches must have equal output "
            "types\" or \"carry input and carry output must have equal types\"). PyTensor's JAX backend sets "
            "jax_enable_x64 from floatX when it is imported; something turned it on afterwards (an example "
            "notebook's jax.config.update(\"jax_enable_x64\", True) does). Turn it off again with "
            "jax.config.update(\"jax_enable_x64\", False) before sampling.",
            RuntimeWarning, stacklevel=3)


# The sampling scan's per-draw projections are defined with the chunk program that returns them.
_SampledState, _SampledInfo = compiled.SampledState, compiled.SampledInfo


def default_pool_cap(chains, d):
    """Row count for the Pathfinder start/metric pool when the caller does not name one.

    One multiple of `chains` cannot size this pool, because it feeds two consumers with different
    needs, and both failure modes are measured:

    * Too many rows is a hard failure. The pool is materialised as one operand: 4,096,000 x 175 x 8 B
      = 5,734,400,000 B, exactly the allocation XLA refuses on a 6 GB card at C=1,024 with the
      production 4,000-draw pool (measured: it fits at d=8, not at d=25).
    * Too few rows is a correctness failure. The same pool is the sample the Monte-Carlo covariance is
      estimated from, so rows must stay large relative to d^2. Capping radon (d=175) at 4C=128 rows at
      C=32 read b2 2.34860 against a 0.07689 floor with split R-hat 1.2121 -- a rank-deficient metric,
      not merely a slower one (measured).

    `max(4C, 4d^2)` closes both without touching the regime that already works: at C=32, d=175 it is
    122,500 rows, essentially the historical 128,000, while at C=1,024, d=175 it cuts the pool from
    4.1M rows (5.73 GB) to 122,500 (43 MB).
    """
    c, dd = int(chains), int(d)
    if c < 1 or dd < 1:
        raise ValueError(f"default_pool_cap needs chains>=1 and d>=1, got chains={c} d={dd}")
    return max(4 * c, 4 * dd * dd)


def resolve_pool_cap(pf_pool_cap, chains, d, imm_source="mc"):
    """Row cap for the Pathfinder start/metric pool, or None for the historical uncapped pool.

    `pf_pool_cap=None` (the default) depends on who consumes the pool:

    * ``imm_source == "mc"``: the pool is also the sample the Monte-Carlo covariance is estimated
      from, so rows must stay large relative to d^2 -> :func:`default_pool_cap`, ``max(4C, 4d^2)``.
    * any other source: the metric comes from the fitted L-BFGS factors (``pf_closed``) or from its own
      estimator, and the pool only feeds the start picks, so ``4C`` rows are enough. This also removes
      the ~32 d^3-byte pool the d-dependent cap needs at large d (4 GB at d=500).

    ``0`` is the uncapped pool and an explicit integer pins the row count, for every source.
    """
    if pf_pool_cap is None:
        return default_pool_cap(chains, d) if imm_source == "mc" else 4 * int(chains)
    cap = int(pf_pool_cap)
    if cap == 0:
        return None
    if cap < chains:
        raise ValueError(f"pf_pool_cap={cap} must be >= chains={chains} (or 0 for uncapped)")
    return cap


def start_score_stride(total, chains):
    """Stride of the scored subset of a ``total``-row Pathfinder pool: at most START_SCORE_PER_CHAIN * chains rows.

    1 (the whole pool is scored) whenever ``total <= START_SCORE_PER_CHAIN * chains``, which includes the default
    pool, ``max(4C, 4d^2)`` rows, for d <= sqrt(C). Rounded up, so ``ceil(total / stride)`` never exceeds the cap.
    """
    return max(-(-int(total) // (START_SCORE_PER_CHAIN * max(int(chains), 1))), 1)


def _stratified_pool_idx(logp_subset, subset_idx, n_pick):
    """Equally spaced ranks by log-density, chosen from a scored subset of the draw pool.

    Deliberately not Pathfinder's PS-IR: that resamples WITH replacement, and the Pathfinder paper
    reports it collapsing to 4 distinct draws out of 100 on mcycle_gp-accel_gp -- it removes the very
    dispersion this is meant to guarantee. Measured on radon non-centered (32 chains, 500 draws):
    worst R-hat 1.0323 -> 1.0210 and min bulk ESS +6% at warmup 250, identical spread to iid and no
    extra leapfrog steps; indistinguishable from iid at warmup >= 1,000.

    Rows whose log-density is not finite are ranked after the finite ones and left out of the ranks: sorted first,
    the lowest rank would start a chain at ``-inf``, where it stays for the whole run (every proposal is rejected).
    A pool with no such row (every float64 run of the four ladder models) is ranked exactly as before. A Pathfinder
    pool can hold one: GEV's float32 fit put a row outside the support (``1 + xi z <= 0``, ``-inf`` in float64 too;
    ``tests/test_float32.py``). If no row is finite the ranks run over all of them, as before.
    """
    lp = np.asarray(logp_subset)
    finite = np.isfinite(lp)
    order = np.argsort(np.where(finite, lp, np.inf), kind="stable")
    m = int(finite.sum()) or int(order.shape[0])
    ranks = (np.arange(n_pick) * m) // n_pick
    return subset_idx[order[ranks]]


def _finite_gradient_rows(draws, grads, max_rank):
    """The rows of a Fisher estimate's (draws, gradients) whose gradient is finite.

    A Pathfinder draw beyond a hard edge of the support (log-density -inf) has a NaN gradient: written as a switch,
    as in pymc_extras' GenExtreme, the unused branch's NaN survives the masking (0 x NaN). One such row makes the
    whole metric NaN, every leapfrog step NaN, and the step-size search halve to the float floor. Those rows are
    left out. When every row is finite (the usual case) the arrays are returned as they are, so nothing changes,
    not even the compiled shapes. Returns (draws, grads, record) with record = {"fisher_rows", "fisher_rows_used"}.
    """
    ok = np.asarray(jnp.all(jnp.isfinite(grads), axis=1))
    record = {"fisher_rows": int(ok.shape[0]), "fisher_rows_used": int(ok.sum())}
    if ok.all():
        return draws, grads, record
    warnings.warn(
        f"imm_source='fisher_lowrank': {ok.shape[0] - int(ok.sum())} of {ok.shape[0]} Pathfinder draws have a "
        "non-finite gradient (outside the support) and are left out of the metric", RuntimeWarning, stacklevel=3)
    if ok.sum() < 2 * max_rank + 2:
        raise ValueError(
            f"imm_source='fisher_lowrank': only {int(ok.sum())} of {ok.shape[0]} Pathfinder draws have a finite "
            f"gradient (the rest lie outside the support), too few for rank {max_rank}; use imm_source='mc'")
    return draws[ok], grads[ok], record


def _frozen_nominal_L(integration_steps_fn, warmup_steps_used):
    """ceil(T/eps) at the frozen point, recovered from the kernel's own step-count function.

    `integration_steps_fn(arg)` returns ceil(halton(arg) * T/eps), and halton(arg) < 1 always, so a
    direct call underestimates. arg = 2**b - 2 (all-ones index; 2**b - 1 would give 0 because
    (arg+1) = 2**b is divisible by every 2**k in the sequence) gives jitter = 1 - 2**-b, exact enough
    identify ceil(T/eps) for any L far below 2**b. Bit width is guessed from the warmup length and
    checked over b-1..b+1, taking the maximum, so a change to utils' `max_sampling_steps` default
    cannot make the reported number too small.
    """
    base = max(int(warmup_steps_used) + 1000, 1024)   # 1000 = utils' max_sampling_steps default
    best = 0.0
    for bits in (base.bit_length() - 1, base.bit_length(), base.bit_length() + 1):
        if bits < 1:
            continue
        arg = jnp.asarray(2 ** bits - 2, dtype=int)   # all-ones index -> jitter 1-2^-bits (int64 with x64, else int32)
        try:
            best = max(best, float(integration_steps_fn(arg)))
        except Exception:
            continue
    return best


def _json_default(o):
    """json.dumps fallback for the config record: numpy / JAX scalars and arrays -> plain Python."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"config value {o!r} of type {type(o).__name__} is not JSON serialisable")


class PfCheesHMC:
    def __init__(self, model:pm.Model, seed:Optional[int]=None):
        self.model = model
        # Per-stage record of what each stage actually resolved (defaults filled in). Written by the
        # stage methods, composed into idata.attrs["config"] by sample(). Stages can be called on
        # their own (tests, tools), so every stage overwrites only its own entry.
        self._resolved = {}

        self.rvs = [rv.name for rv in model.value_vars]
        # The log-density with the model's data and prior values as runtime arguments (F2a). The
        # compiled sites (`pfcheeshmc.compiled`) take `fn` as a static and `_values` as a dynamic
        # argument; `_ld_fn` is the same function with the values closed over, for eager use.
        self._lifted = lift_logp(model)
        self._values = self._lifted.values
        self._ld_fn = self._default_ld_fn = lambda seq: self._lifted.fn(seq, self._values)[0]
        self._override = None
        self._seed = seed

    def _refresh_values(self):
        """Re-read the lifted values, so a ``pm.set_data`` since the last call is seen.

        Shapes and dtypes are part of the program, so if a shared variable was set to another shape the
        log-density is lifted again (a new structure, hence a new key). So it is when a shared variable
        that is baked into the program as a constant (kept static: a ``pm.Data`` that is a basic index,
        a shape, ...) has another value: ``fn`` would silently go on computing with the old one.
        """
        lg = self._lifted
        values = lg.current_values()
        if lg.is_stale() or any(v.shape != o.shape or v.dtype != o.dtype for v, o in zip(values, lg.values)):
            self._lifted = lg = lift_logp(self.model)
            values = lg.values
        self._values = values

    def _logdensity(self) -> LogDensity:
        """The log-density as the pieces the compiled sites take (see `pfcheeshmc.compiled`).

        A sampler whose ``ld_fn`` or ``_ld_fn`` was replaced on the instance (tests do this) is
        honoured by every site: the replacement is wrapped into a ``fn`` with no lifted values, so it is
        compiled in as a constant, as before. The wrapper is kept per replacement so that it is one
        static argument for jit, not a new one per call.
        """
        names = tuple(self.rvs)
        replaced = (vars(self).get("ld_fn"), self._ld_fn)
        if replaced[0] is None and replaced[1] is self._default_ld_fn:
            return LogDensity(self._lifted.fn, names, self._values)
        if self._override is None or any(a is not b for a, b in zip(self._override[0], replaced)):
            self._override = (replaced, lambda seq, values: [self.ld_fn(dict(zip(names, seq)))])
        return LogDensity(self._override[1], names, ())

    def ld_fn(self, position):
        """
        Logp wrapper: take a position pytree (dict/list/tuple), return scalar logp
        """
        if isinstance(position, dict):
            seq = tuple(position[name] for name in self.rvs)
        elif isinstance(position, (list, tuple)):
            seq = tuple(position)
        else:
            # last resort: pull leaves in a stable order
            seq = tuple(jax.tree_util.tree_leaves(position))
        return self._ld_fn(seq) 

    def dict_to_position(self, model, names):
        """The model's initial point as a list in ``names`` order (JAX; ``model.initial_point()`` if it cannot)."""
        try:
            lg = lift_initial_point(model)
            ip = dict(zip((v.name for v in model.value_vars), compiled.initial_point(lg.fn, lg.values)))
        except Exception:
            ip = model.initial_point()
        return [jnp.asarray(ip[n]) for n in names]

    def _kernel_flat(self, position):
        """Flatten a chain position exactly as the HMC kernel does.

        The kernel ravels the *state*, which is a dict, so JAX orders the coordinates by sorted
        key (blackjax/mcmc/metrics.py:236-244). Anything that builds a metric, or reads one, must
        flatten through here so it cannot silently use Pathfinder's tuple order instead.
        """
        return ravel_pytree(position)[0]

    def _kernel_permutation(self, ref_tree):
        """Index vector ``perm`` with ``perm[k]`` = tuple-order index of kernel coordinate ``k``.

        Two flatten orders exist in this pipeline and they are not the same order:

        * Pathfinder is fitted on ``tuple(position[n] for n in self.rvs)`` (:py:func:`ld_fn` and
          ``approximate(initial_position=...)``), so its draws, its covariance and its L-BFGS
          factors are in **model.value_vars order**.
        * The HMC state is a **dict**, and ``ravel_pytree`` sorts dict keys, so the kernel reads a
          metric in **sorted-name order**.

        When those differ (GEV: ``[mu, sigma_log__, xi_interval__]`` vs ``[mu, xi_interval__,
        sigma_log__]``; radon: the five scalars land inside the 170 raw effects) a metric built in
        tuple order is applied to the wrong coordinates, and nothing raises. This returns the
        permutation that fixes it; identity when the orders already agree.
        """
        if isinstance(ref_tree, dict):
            # The tuple/dict distinction is the whole point of this function: leaves() of a dict is
            # already sorted-key order, so passing a dict would compute a plausible-looking but
            # wrong permutation. Only the Pathfinder tuple is a valid argument.
            raise ValueError(
                "_kernel_permutation expects Pathfinder's tuple in model.value_vars order; a dict "
                "is already kernel-ordered and would silently yield a wrong permutation")
        leaves0 = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x[0], ref_tree))
        if len(leaves0) != len(self.rvs):
            raise ValueError(
                f"the position tree has {len(leaves0)} leaves but the model has "
                f"{len(self.rvs)} value_vars ({self.rvs}); cannot map Pathfinder's coordinate "
                "order onto the kernel's"
            )
        d = int(sum(int(jnp.size(x)) for x in leaves0))
        _, unravel_tuple = ravel_pytree(jax.tree_util.tree_map(lambda x: x[0], ref_tree))
        # Label each tuple-order slot with its own index, then ravel as a dict: entry k of the
        # result is the tuple-order index of the coordinate the kernel puts at position k.
        labelled = dict(zip(self.rvs, jax.tree_util.tree_leaves(unravel_tuple(jnp.arange(d)))))
        return ravel_pytree(labelled)[0].astype(int)

    def _metric_to_kernel_order(self, imm, perm):
        """Reorder a dense metric from Pathfinder's tuple order into the kernel's dict order.

        Applied to the *expanded* dense matrix for every ``imm_source``. For a low-rank factor
        ``D (I + U (lam-1) U^T) D`` this is identical to permuting the factors first
        (``sigma[perm]``, ``U[perm, :]``), because conjugation by the permutation matrix commutes
        with that expression; ``tests/test_metric_orientation.py`` pins the equivalence rather
        than carrying two code paths.
        """
        if jnp.ndim(imm) == 2:
            return imm[perm][:, perm]
        return imm[perm]
    
    @property
    def seed(self):
        return int(date.today().strftime("%Y%m%d")) if self._seed is None else self._seed
    
    @seed.setter
    def seed(self, value):
        self._seed = value
    
    def _find_finite_start(
        self,
        x0,
        key,
        *,
        jitter_eps: float = 1e-8,
        max_backtracks: int = 16,
        max_restarts: int = 64,
        restart_radius: float = 0.05,
    ):
        """
        Make a bad start usable without changing the model. A start must have a finite
        log-density AND a finite gradient. Order of repair:
        1) tiny relative jitter (fixes exact singularities like xi=0),
        2) backtrack toward the origin of the unconstrained space,
        3) random restarts around the origin with growing radius.
        Returns a pytree, or raises RuntimeError if nothing usable is found.
        """
        from jax.flatten_util import ravel_pytree

        def _is_usable(pos):
            """Finite log-density and finite gradient.

            The gradient test is not decoration. GEV Port Pirie at `model.initial_point()`
            (xi_interval__ = 0) has a finite log-density of -7.749 and a NaN gradient in all
            three variables: `GenExtreme.logp` evaluates both branches of
            `switch(isclose(xi, 0), gumbel, xi_branch)`, and the unselected xi-branch contains
            (xi+1)/xi -> inf multiplied by log1p(xi*z) -> 0, which is NaN. The forward value is
            fine; only the vjp poisons the shared leaves. The log-density is also genuinely not
            C^1 at xi = 0 -- PyTensor's own autodiff returns NaN for the xi derivative on the
            same graph -- so this is not a JAX-backend quirk to work around, and Pathfinder's
            L-BFGS fit must not start from a NaN gradient.

            Decided by `_usable_mask` on a one-row stack, the same jitted program that screened the starts
            (F2g): evaluated eagerly the log-density and its gradient compiled one XLA program per primitive
            (156 of them for GEV, 4.4 s on CPU) and dispatched them op by op on every call.
            """
            try:
                one_row = {n: np.asarray(x)[None] for n, x in zip(self.rvs, compiled._position_sequence(pos, self.rvs))}
                return bool(self._usable_mask(one_row)[0])
            except Exception:             # a check that cannot be jitted must not cost us the run: decide it eagerly
                if not bool(jnp.isfinite(self.ld_fn(pos))):
                    return False
                grads = jax.grad(self.ld_fn)(pos)
                return bool(jnp.all(jnp.isfinite(ravel_pytree(grads)[0])))

        def _interp(a, b, t):
            return jax.tree_util.tree_map(lambda ai, bi: bi + t * (ai - bi), a, b)

        flat0, unrav = ravel_pytree(x0)

        # (1) tiny relative jitter
        key, k0 = jax.random.split(key)
        rel = 1.0 + jnp.abs(flat0)
        jitter = jax.random.normal(k0, flat0.shape, dtype=flat0.dtype) * (jitter_eps * rel)
        x = unrav(flat0 + jitter)
        if _is_usable(x):
            return x

        # (2) backtrack toward the safe center (zeros)
        center = unrav(jnp.zeros_like(flat0))
        for _ in range(max_backtracks):
            x_try = _interp(x, center, 0.5)
            if _is_usable(x_try):
                return x_try
            x = x_try

        # (3) random restarts around the center
        radius = restart_radius
        for _ in range(max_restarts):
            key, k = jax.random.split(key)
            noise = jax.random.normal(k, flat0.shape, dtype=flat0.dtype) * radius
            x_try = unrav(ravel_pytree(center)[0] + noise)
            if _is_usable(x_try):
                return x_try
            radius *= 1.5

        raise RuntimeError("Failed to locate a start with finite log-density and finite gradient "
                           "for Pathfinder.")

    @contextlib.contextmanager
    def _silence_pymc_init(self):
        """
        Hide PyMC's 'Initializing NUTS using ...' line regardless of whether it
        comes from logging or direct console output.
        """
        # 1) Filter loggers that PyMC uses
        msg_re = re.compile(r"Initializing NUTS using")
        loggers = [logging.getLogger("pymc"),
                logging.getLogger("pymc.sampling.mcmc")]

        class _HideInitFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                # Keep everything EXCEPT the init line
                try:
                    msg = record.getMessage()
                except Exception:
                    msg = record.msg if isinstance(record.msg, str) else ""
                return msg_re.search(msg) is None

        filters = []
        old_levels = []
        for lg in loggers:
            flt = _HideInitFilter()
            lg.addFilter(flt)
            filters.append((lg, flt))
            old_levels.append((lg, lg.level))
            # Ensure INFO messages (where the line is) are not emitted
            lg.setLevel(logging.WARNING)

        # 2) Also silence anything written to stdout / stderr during the call
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                yield
        finally:
            for lg, flt in filters:
                lg.removeFilter(flt)
            for lg, lvl in old_levels:
                lg.setLevel(lvl)

    def _jax_initial_points(self, chains: int, *, jitter: bool, random_seed, jitter_max_retries: int):
        """PyMC's ``adapt_diag`` / ``jitter+adapt_diag`` initial points with no PyTensor compile.

        The un-jittered point is PyMC's own initial-point graph (`lifted.lift_initial_point`: the model's
        prior values and data are lifted, so a new prior compiles nothing) evaluated by a module-level jit.
        With ``jitter``, each chain adds U(-1, 1) per coordinate and redraws while its log-density is not
        finite (`compiled.jittered_initial_points`; the stream is ``pm_seed``'s, not PyMC's).
        Returns ``(dict{name: (chains, ...)} in self.rvs order, indices of the chains left non-finite)``.
        """
        lg = lift_initial_point(self.model)
        base = compiled.initial_point(lg.fn, lg.values)
        names = tuple(self.rvs)
        if len(base) != len(names):
            raise ValueError(f"{len(base)} initial values for {len(names)} value variables")
        if not jitter:
            points, bad = tuple(jnp.broadcast_to(b, (chains,) + b.shape) for b in base), np.zeros(0, dtype=int)
        else:
            seed = random_seed if random_seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
            key = jax.random.fold_in(jax.random.key(int(seed)), compiled.JITTER_STREAM)
            ld = self._logdensity()
            points, ok = compiled.jittered_initial_points(
                ld.fn, ld.names, ld.values, base, key, int(jitter_max_retries), chains=chains)
            bad = np.flatnonzero(~np.asarray(ok))
        return dict(zip(names, points)), bad

    def _pymc_adapt_diag_start(
        self,
        chains: int,
        *,
        init: str = "adapt_diag",   # or "jitter+adapt_diag"
        tune: int = 500,
        random_seed: Optional[int] = None,
        jitter_max_retries: int = JITTER_MAX_RETRIES,
    ):
        """
        PyMC's per-chain initial points for ``init`` (no posterior sampling). Returns dict{name: (chains, ...)}
        keyed by self.rvs.

        ``pm.init_nuts`` builds a whole NUTS step (PyTensor compiles of the log-density and its gradient on the
        C backend, 0.5-1.6 s) and the sampler keeps only its ``init_points``. For ``adapt_diag`` those are the
        initial values of the model (the test point, in the transformed space) for every chain; for
        ``jitter+adapt_diag`` the same plus a U(-1, 1) jitter per coordinate, redrawn while the log-density is
        not finite. That is computed in JAX instead (`_jax_initial_points`): ``adapt_diag`` points equal
        PyMC's, the jitter is the same distribution from ``random_seed``'s own JAX stream. ``tune`` is not
        used by either (``init_nuts`` reads it only for ``jitter+adapt_diag_grad``). Any other ``init``, a
        model with discrete variables, or an initial point that cannot be built in JAX (a random draw, or a
        graph that fails to lift: a warning names the exception) goes to ``pm.init_nuts`` as before.
        """
        print(f"Initializing ChEES-HMC using pathfinder+{init}...")
        kind = init.lower() if isinstance(init, str) else init
        if kind in ("adapt_diag", "jitter+adapt_diag") and all_continuous(self.model.value_vars):
            try:
                found, bad = self._jax_initial_points(
                    chains, jitter=kind == "jitter+adapt_diag", random_seed=random_seed,
                    jitter_max_retries=jitter_max_retries)
            except Exception as exc:
                reason = (str(exc).splitlines() or [""])[0]
                warnings.warn(
                    f"initial points in JAX failed with {type(exc).__name__}: {reason}; "
                    "falling back to pm.init_nuts", RuntimeWarning, stacklevel=2)
            else:
                if bad.size:     # as PyMC after its last retry: the error that names the non-finite term
                    self.model.check_start_vals({n: np.asarray(a[bad[0]]) for n, a in found.items()})
                return found
        with self._silence_pymc_init():
            init_points, _ = pm.init_nuts(
                init=init,
                chains=chains,
                tune=tune,
                model=self.model,
                random_seed=random_seed,
                progressbar=True,
                jitter_max_retries=jitter_max_retries,
            )
        # list[chain] of dicts -> dict of stacked arrays in self.rvs order
        return {
            name: jnp.stack([jnp.asarray(ip[name]) for ip in init_points], axis=0)
            for name in self.rvs
        }

    def _usable_mask(self, init_stack):
        """One jitted, vmapped check for finite log-density AND finite gradient, all chains at once.

        The per-chain repair is eager Python: 32 separate calls into the compiled jaxed logp cost 1.75 s on
        8 Schools and 4.24 s on radon, which is Python dispatch, not compute. Chains that already pass never
        enter it, and the repair decides each candidate with this same program on a one-row stack (F2g), so
        that no log-density or gradient is evaluated op by op.
        """
        ld = self._logdensity()
        return compiled.usable_mask(ld.fn, ld.names, ld.values, {n: init_stack[n] for n in self.rvs})

    def _ensure_finite_inits(
        self,
        init_stack: dict,
        key,
        *,
        jitter_eps=1e-8,
        max_backtracks=16,
        max_restarts=64,
        restart_radius=0.05,
    ):
        """
        For a dict{name: (chains, ...)} ensure every chain's point has a finite log-density and a
        finite gradient. Applies _find_finite_start to the rows that fail, each under its own key
        (``split(key, chains)[c]`` for row c). Returns a dict{name: (chains, ...)} ready for Pathfinder;
        when every row passes it is the arrays that came in.

        Note on jitter: points that already pass are returned untouched. Previously every chain was
        jittered by ~1e-8 relative, even when its start was fine. Same distribution, different
        realisation.
        """
        chains = next(iter(init_stack.values())).shape[0]
        try:
            usable = np.asarray(self._usable_mask(init_stack))        # the one host read of the whole check
        except Exception:                     # a batched check that cannot run must not cost us the run
            usable = np.zeros(chains, dtype=bool)
        if usable.all():
            return {n: jnp.asarray(init_stack[n]) for n in self.rvs}

        keys = jax.random.split(key, chains)
        repaired = {n: np.array(init_stack[n]) for n in self.rvs}      # a host copy: no program is compiled to put rows back
        for c in map(int, np.flatnonzero(~usable)):
            x0 = {n: init_stack[n][c] for n in self.rvs}
            try:
                x0_safe = self._find_finite_start(
                    x0, keys[c],
                    jitter_eps=jitter_eps,
                    max_backtracks=max_backtracks,
                    max_restarts=max_restarts,
                    restart_radius=restart_radius,
                )
            except RuntimeError:
                # last-resort: zero center
                x0_safe = self._find_finite_start(
                    {n: jnp.zeros_like(v[c]) for n, v in init_stack.items()},
                    keys[c],
                    jitter_eps=jitter_eps,
                    max_backtracks=max_backtracks,
                    max_restarts=max_restarts,
                    restart_radius=restart_radius,
                )
            for n in self.rvs:
                repaired[n][c] = np.asarray(x0_safe[n])

        return {n: jnp.asarray(repaired[n]) for n in self.rvs}

    def _run_pathfinder(self, **kwargs) -> Tuple[dict, jnp.ndarray]:
        """
        Combined strategy:
        • get PyMC adapt_diag starts,
        • repair any -inf starts generically,
        • seed Pathfinder from the *best* finite chain,
        • draw one PF batch (chains*draws), compute dense mass,
        • choose inits either from PF or from the (repaired) adapt_diag starts.

        pathfinder_kwargs you can pass:
        pm_init: "adapt_diag" | "jitter+adapt_diag" (default "adapt_diag")
        pm_chains: int in [1, chains] (default 1): how many PyMC points are built. They are shared
            cyclically, chain c taking point c mod pm_chains: 1 = every chain shares one point,
            chains = one point per chain. The points are screened and repaired once each, before they
            are shared
        pm_tune: int (default 500)
        pm_seed: int (default self.seed)
        init_source: "pf" | "adapt_diag" (default "pf")
        ridge: float covariance ridge (default 1e-6; the "mc" covariance only)
        imm_source: "mc" (default) | "pf_closed" | "lbfgs_lowrank" | "fisher_lowrank"
        pf_pool_cap: pool row cap; None -> see resolve_pool_cap (depends on imm_source), 0 = uncapped
        score_chunk: block size of the vmapped log-density scorings (default 1024)
        fisher_draws, max_rank, fisher_gamma, fisher_cutoff: the "fisher_lowrank" metric only
        pf_jitter_eps, pf_max_backtracks, pf_max_restarts, pf_restart_radius: the start repair
        pf_maxiter, pf_maxcor, pf_maxls, pf_gtol, pf_ftol, pf_elbo_samples: the L-BFGS path and its ELBO draws
            (blackjax.vi.pathfinder.approximate's maxiter, maxcor, maxls, gtol, ftol, num_samples); absent
            keeps BlackJAX's defaults (30, 10, 1000, 1e-8, 1e-5, 200). The defaults stop early on stiff
            posteriors: L-BFGS can end far from the mode, and the starts and the metric come from there
        chains, draws, seed, ridge: the number of chains, the Pathfinder draws per chain (the pool is
            ``chains * draws`` rows before the cap), the stage seed and the covariance ridge

        Any other key raises a TypeError that names it.
        """
        _check_options("pathfinder_kwargs", kwargs, _PATHFINDER_OPTIONS)
        # ---- Defaults
        chains = kwargs.get("chains", 32)
        draws  = kwargs.get("draws", 4000)
        seed   = kwargs.get("seed", self.seed)
        ridge  = kwargs.get("ridge", 1e-6)
        init_source = kwargs.get("init_source", "pf")  # "pf" or "adapt_diag"

        # PyMC init knobs
        pm_init = kwargs.get("pm_init", "adapt_diag")          # or "jitter+adapt_diag"
        pm_tune = kwargs.get("pm_tune", 500)
        pm_seed = kwargs.get("pm_seed", seed)
        # WP5: how many PyMC init points to build. pm.init_nuts runs a short NUTS metric
        # adaptation PER chain, so this is the slowest C-dependent work in the init path.
        # The points only seed Pathfinder (and serve as the init_source="adapt_diag"
        # fallback), so one point shared by all chains is enough; chain diversity comes
        # from the PF pool. Default 1 = the R1 fix; pass pm_chains=chains for the old
        # per-chain behaviour. Any count in between is shared cyclically: chain c takes
        # point c mod pm_chains.
        pm_chains = int(kwargs.get("pm_chains", 1))
        if not 1 <= pm_chains <= chains:
            raise ValueError(f"pm_chains must be in [1, chains={chains}], got {pm_chains}")

        # WP5: cap on the PF start pool (chains*draws rows). At C=1,024 with the default
        # draws=4,000 the pool is 4.1M rows -- 5.73 GB at d=175 -- and it feeds both the covariance
        # estimate and the start picks. None (the default) means resolve_pool_cap(...): 4C rows when
        # imm_source != "mc" (the pool only feeds start picks), default_pool_cap(chains, d) =
        # max(4C, 4d^2) when imm_source == "mc"; 0 is the historical uncapped pool; an integer pins
        # the row count.
        pf_pool_cap = kwargs.get("pf_pool_cap", None)
        # WP5: block size for the vmapped log-density scorings below (peak memory is
        # O(block), not O(pool or chains)).
        score_chunk = int(kwargs.get("score_chunk", 1024))
        if score_chunk < 1:
            raise ValueError(f"score_chunk must be >= 1, got {score_chunk}")

        # finite-start knobs
        pf_jitter_eps     = kwargs.get("pf_jitter_eps", 1e-8)
        pf_max_backtracks = kwargs.get("pf_max_backtracks", 16)
        pf_max_restarts   = kwargs.get("pf_max_restarts", 64)
        pf_restart_radius = kwargs.get("pf_restart_radius", 0.05)

        key = jax.random.key(seed)
        # five children, as ever: the key of the start pick is no longer drawn on (the pick is stratified), but
        # the other four keys are the same ones they were
        key_fix, key_pf, key_samp, _, key_choose = jax.random.split(key, 5)

        # 0) Ask PyMC for feasible-ish starts -- one point shared by all chains by default
        init_stack = self._pymc_adapt_diag_start(
            pm_chains, init=pm_init, tune=pm_tune, random_seed=pm_seed
        )

        # 1) Ensure every init is finite (handles xi=0 -> -inf cases). The check and any repair run once on
        #    the pm_chains PyMC points (each repaired row under its own key, split(key_fix, pm_chains)[row]),
        #    never C times on C copies of them. Then the points are shared by the chains: chain c starts from
        #    repaired row c mod pm_chains. That is every chain on row 0 when pm_chains = 1, and the identity
        #    when pm_chains = chains.
        init_stack = self._ensure_finite_inits(
            init_stack, key_fix,
            jitter_eps=pf_jitter_eps,
            max_backtracks=pf_max_backtracks,
            max_restarts=pf_max_restarts,
            restart_radius=pf_restart_radius,
        )
        if pm_chains < chains:
            # gathered on the host and put on the device: a handful of rows, and no XLA program is compiled
            # to copy them (the old broadcast_to compiled one; jnp.asarray compiles one per shape)
            rows = np.arange(chains) % pm_chains
            init_stack = {n: jax.device_put(np.take(np.asarray(a), rows, axis=0)) for n, a in init_stack.items()}

        # 2) Choose a single finite seed for PF optimizer: the chain with highest logp.
        # One vmapped, chunked scoring pass instead of one eager call per chain: the old
        # loop was C device syncs (the C-dependent wall of the init path at large C), and
        # the argmax of the finite scores is exactly what it computed.
        n_inits = int(next(iter(init_stack.values())).shape[0])
        ld = self._logdensity()
        lp_inits = jnp.empty(n_inits, dtype=jnp.float32)
        def _score_block(s0, s1):
            sub = {n: init_stack[n][s0:s1] for n in self.rvs}
            return compiled.score_positions(ld.fn, ld.names, ld.values, sub)
        first = _score_block(0, min(score_chunk, n_inits))
        lp_inits = jnp.empty(n_inits, dtype=first.dtype)
        lp_inits = lp_inits.at[: first.shape[0]].set(first)
        for s0 in range(score_chunk, n_inits, score_chunk):
            lp_inits = lp_inits.at[s0:s0 + score_chunk].set(
                _score_block(s0, min(s0 + score_chunk, n_inits)))
        lp_inits = jnp.where(jnp.isfinite(lp_inits), lp_inits, -jnp.inf)
        c_best = int(jnp.argmax(lp_inits))
        x0_pf = {n: init_stack[n][c_best] for n in self.rvs}

        # 3) Fit Pathfinder. The L-BFGS settings are passed only when given, so the default call is unchanged.
        lbfgs = tuple(sorted((_PF_LBFGS_OPTIONS[k], v) for k, v in kwargs.items() if k in _PF_LBFGS_OPTIONS))
        pf_state = compiled.pathfinder_fit(
            bj.vi.pathfinder.approximate, ld.fn, ld.names, ld.values,
            key_pf, tuple(x0_pf[n] for n in self.rvs), **({"lbfgs": lbfgs} if lbfgs else {}),
        )

        # 4) Draw the start pool, bounded by `pf_pool_cap` rows (see default_pool_cap for why the
        # default is dimension-aware rather than a multiple of `chains`). pf_pool_cap=0 is the
        # historical uncapped pool; an explicit integer pins it.
        d_flat = 0
        for n in self.rvs:
            k = 1
            # x0_pf[n] is ONE chain's position, so every axis is a coordinate: dropping a leading
            # axis here would read a d-vector as a scalar and under-size the pool by ~d.
            for s in jnp.asarray(x0_pf[n]).shape:
                k *= int(s)
            d_flat += k
        imm_source = kwargs.get("imm_source", "mc")
        cap = resolve_pool_cap(pf_pool_cap, chains, d_flat, imm_source)
        total = chains * draws
        if cap is not None:
            total = min(total, cap)
        samples_flat = compiled.pathfinder_draw(bj.vi.pathfinder.sample, key_samp, pf_state, total)
        # samples_flat is the whole pool in flat (total, ...) order; the chunk axis is just
        # how Pathfinder emitted the draws, so any row may seed any chain. (The old code
        # reshaped to (chains, draws) and back; the round trip was a no-op kept for a reader.)
        samples_total = samples_flat

        # 5) Choose *chain* initial positions
        if init_source == "pf":
            # Stratified pick: score a strided subset of at most START_SCORE_PER_CHAIN * C rows -- ranking the
            # whole pool would cost more than the benefit, which only exists at short warmup, and the default
            # pool is max(4C, 4d^2) rows (122,500 evaluations at d=175, 1,000,000 at d=500). With a pool of at
            # most 4C rows the stride is 1 and the whole pool is scored, chunked to keep the vmapped
            # log-density activation at O(score_chunk) rows (a single 65,536-row vmap was the second R1 OOM,
            # ~500 MiB at d=25).
            stride = start_score_stride(total, chains)
            subset_idx = jnp.arange(0, total, stride)
            def _score_pool(idx):
                return compiled.score_pool(ld.fn, ld.names, ld.values, samples_flat, idx)
            first = _score_pool(subset_idx[: min(score_chunk, subset_idx.shape[0])])
            lp_sub = jnp.empty(subset_idx.shape, dtype=first.dtype)
            lp_sub = lp_sub.at[: first.shape[0]].set(first)
            for s0 in range(score_chunk, subset_idx.shape[0], score_chunk):
                lp_sub = lp_sub.at[s0:s0 + score_chunk].set(_score_pool(
                    subset_idx[s0:s0 + score_chunk]))
            pick_global = _stratified_pool_idx(lp_sub, subset_idx, chains)
            # samples_flat is the whole pool in (chains, draws) chunk order; the chunk axis is just
            # how Pathfinder emitted the draws, so any row may seed any chain.
            pf_inits_tree = jax.tree_util.tree_map(lambda x: x[pick_global], samples_flat)
        elif init_source == "adapt_diag":
            # use the repaired adapt_diag starts as chain inits. With shared base points
            # (pm_chains < chains: chain c sits on repaired row c mod pm_chains, so several chains
            # share one) add a tiny per-chain relative jitter so the option does not degenerate to
            # chains that start on the same point -- identical starts read R-hat 1.000 while
            # stuck, which no diagnostic flags. The base points are guard-validated; a 1e-3
            # relative move of a finite-gradient point stays finite in practice.
            pf_inits_tree = init_stack
            if pm_chains < chains:
                jkey = jax.random.split(key_choose, len(self.rvs))
                # Relative AND absolute jitter: a purely relative move cannot budge a
                # coordinate that is exactly 0.0 (common in initial points), and zero
                # coordinates would keep those chains tied together.
                pf_inits_tree = {
                    n: init_stack[n] * (1.0 + 1e-3 * jax.random.normal(
                            jkey[i], init_stack[n].shape, dtype=init_stack[n].dtype))
                        + 1e-6 * jax.random.normal(
                            jax.random.fold_in(jkey[i], 1), init_stack[n].shape,
                            dtype=init_stack[n].dtype)
                    for i, n in enumerate(self.rvs)
                }
        else:
            raise ValueError("init_source must be 'pf' or 'adapt_diag'")

        # 6) Dense covariance from ALL PF samples. This is a COVARIANCE in Pathfinder's own tuple
        # order; it is handed to the kernel as it is (step 6c), after the reorder to kernel order in step 8.
        # (samples_total is samples_flat, already (total, ...); the old reshape round trip
        # through (chains, draws) was a no-op.)
        def _flatten_i(i):
            sliced = jax.tree_util.tree_map(lambda x: x[i], samples_total)
            flat, _ = ravel_pytree(sliced)
            return flat
        flat_mat = jax.vmap(_flatten_i)(jnp.arange(total))  # (total, D)

        X   = flat_mat - flat_mat.mean(axis=0, keepdims=True)
        covariance = (X.T @ X) / (total - 1) + ridge * jnp.eye(X.shape[1], dtype=X.dtype)

        # 6b) Optional alternative metric (E4). Default "mc" keeps the empirical covariance above.
        # Every source produces the same kind of object -- a COVARIANCE estimate (see 6c).
        fisher_record = {}
        if imm_source != "mc":
            if imm_source == "pf_closed":
                # The exact covariance Pathfinder's own sampler draws with: bfgs_sample builds
                # phi = mu + sqrt(alpha) * (Q (L - I) Q^T u + u) with L L^T = I + R gamma R^T and
                # Q R = beta / sqrt(alpha), whose covariance is diag(alpha) + beta gamma beta^T
                # (formula II.1 of zhang2022pathfinder). So the MC pool covariance above is a noisy
                # estimate of this matrix; here it is taken in closed form, from the one fit made in
                # step 3. It is in Pathfinder's flat (tuple, self.rvs) order and goes through the
                # same kernel-order choke point (8) as "mc". No ridge: it is
                # positive definite by construction, which is asserted instead of patched over.
                covariance = bj.optimizers.lbfgs.lbfgs_inverse_hessian_formula_1(
                    pf_state.alpha, pf_state.beta, pf_state.gamma)
                lam_min = float(jnp.linalg.eigvalsh(covariance).min())
                if not lam_min > 0.0:
                    raise ValueError(
                        f"imm_source='pf_closed': the closed-form Pathfinder covariance is not "
                        f"positive definite (min eigenvalue {lam_min:.3e}, dtype {covariance.dtype}, "
                        f"d={covariance.shape[0]}). No ridge is added by design; use "
                        "imm_source='mc' or enable float64.")
            elif imm_source == "lbfgs_lowrank":
                # Free: these are the L-BFGS factors Pathfinder already fitted (zhang2022pathfinder
                # formula II.1/II.3). No extra gradients, no extra solve.
                lr = bj.mcmc.metrics.lbfgs_inverse_hessian_to_low_rank_metric(
                    pf_state.alpha, pf_state.beta, pf_state.gamma)
            elif imm_source == "fisher_lowrank":
                # Algorithm 1 of seyboldt2026preconditioning, as shipped in BlackJAX. Needs
                # gradients of the log-density at the Pathfinder draws, so subsample: cost is
                # n_grad gradients, not `total`.
                n_grad = int(min(total, kwargs.get("fisher_draws", 4096)))
                gdraws = flat_mat[:n_grad]
                ggrads = compiled.pool_gradients(
                    ld.fn, ld.names, ld.values,
                    jax.tree_util.tree_map(lambda x: x[0], samples_total), gdraws)
                gdraws, ggrads, fisher_record = _finite_gradient_rows(gdraws, ggrads, int(kwargs.get("max_rank", 10)))
                lr = bj.adaptation.metric_estimators.fisher_score_low_rank(
                    gdraws, ggrads, int(kwargs.get("max_rank", 10)),
                    gamma=float(kwargs.get("fisher_gamma", 1e-5)),
                    cutoff=float(kwargs.get("fisher_cutoff", 2.0)))
            else:
                raise ValueError(
                    "imm_source must be 'mc', 'pf_closed', 'lbfgs_lowrank' or 'fisher_lowrank', "
                    f"got {imm_source!r}")
            if imm_source != "pf_closed":
                D = jnp.diag(lr.sigma)
                covariance = (D @ (jnp.eye(lr.sigma.shape[0], dtype=lr.sigma.dtype)
                                   + lr.U @ jnp.diag(lr.lam - 1.0) @ lr.U.T) @ D)

        # 6c) Orientation. The array the kernel calls `inverse_mass_matrix` (call it A) is the matrix
        # that multiplies the momentum in the kinetic energy, KE = 1/2 p.(A p) (mcmc/metrics.py:
        # 262-270), so momentum ~ N(0, A^-1) and leapfrog gives q'' = -A * grad U. For a target with
        # Hessian H, the squared frequencies are eig(A H): A = H^-1 = Cov(q) makes every coordinate
        # oscillate at unit frequency (dimensionless eps), while A = H makes them 1/var, so one global
        # eps cannot serve narrow and wide coordinates at once. So every metric source hands the kernel
        # a posterior COVARIANCE, as it is (never its inverse).
        #
        # Measured, on a 3-scalar conjugate Gaussian with posterior variances 1e1 / 1e0 / 1e-2 and
        # reverse-sorted names (32 chains, 500 warmup, 1,000 draws, seeds 5-7): the inverse (the
        # precision) froze at eps=0.0138 with max R-hat 10.10 and bulk ESS 33 / 41 / 27,221, while the
        # covariance gave eps=1.14, max R-hat 1.0024 and ESS 24,664 / 24,377 / 25,117. Both pass a 4-MCSE
        # moment check, because Pathfinder's starts already sample that Gaussian posterior: the ensemble
        # mean is right whether or not the chains move, so a moment test alone cannot arbitrate.
        inverse_mass_matrix = covariance
        self.last_covariance = covariance      # diagnostics only (WP1 A/B), never read by the sampler

        # 7) Return dict-of-arrays keyed by variable (leading axis = chains)
        if isinstance(pf_inits_tree, (list, tuple)):
            pf_samples = {name: arr for name, arr in zip(self.rvs, pf_inits_tree)}
        elif isinstance(pf_inits_tree, dict):
            pf_samples = {name: pf_inits_tree[name] for name in self.rvs}
        else:
            pf_samples = {self.rvs[0]: pf_inits_tree}

        # 8) Single choke point: hand the rest of the pipeline a metric in the kernel's order.
        # Everything downstream (find_reasonable_step_size, the ChEES whitening diagonal, and the
        # dynamic-HMC kernel) ravels the dict state, so the reorder happens here and nowhere else.
        perm = self._kernel_permutation(samples_flat)
        inverse_mass_matrix = self._metric_to_kernel_order(inverse_mass_matrix, perm)

        self._resolved["pathfinder"] = {
            "seed": int(seed),
            "imm_source": imm_source,
            # the ridge is added to the Monte-Carlo covariance only; every other source overwrites it
            "ridge": float(ridge) if imm_source == "mc" else 0.0,
            "pf_draws": int(draws),
            "pf_pool_cap": 0 if cap is None else int(cap),     # 0 = uncapped, as the kwarg
            "pf_pool_rows": int(total),
            "init_source": init_source,
            "pm_chains": pm_chains,
            "pm_init": pm_init,
            "pm_tune": int(pm_tune),
            "score_chunk": score_chunk,
            **fisher_record,      # "fisher_lowrank": rows of the Fisher estimate, and how many had a finite gradient
            "lbfgs": dict(lbfgs),          # only the settings given; the rest are BlackJAX's defaults
        }

        return pf_samples, inverse_mass_matrix



    def _run_step_size_tune(self, pf_samples, inverse_mass_matrix, **kwargs):
        """Initial step size for the ChEES warmup.

        BlackJAX's doubling/halving acceptance search
        (`blackjax.adaptation.step_size.find_reasonable_step_size`, Hoffman & Gelman 2014
        Algorithm 4, target acceptance 0.651) run from a batch of Pathfinder positions and
        reduced by geometric median.

        What changed is the execution, not the rule: the search used to be rebuilt from
        Python once per chain, which re-traced the kernel every doubling step and measured
        77.07 s / 32 chains = 2.4 s per chain in the instrumented run (~1.1 s per chain
        uninstrumented). Here it is traced once and run for all probes in one vmapped
        dispatch (`compiled.search_step_sizes`), so cost is independent of the number of
        sampling chains — the sampler itself is flat from 32 to 256 chains (+2% per
        transition), and the old stage made wall time grow linearly.

        Probes are positions, not chains: 32 of them are enough to estimate a global ε0,
        and ChEES re-tunes ε by dual averaging from iteration 1. Do not read ε0 as a mere
        warm start with no consequences — on the stiff GEV model a different ε0 changed the
        divergence count from 33 to 255 — it anchors dual averaging (`dual_averaging.py:94`)
        and sets the ChEES length ceiling (`chees_adaptation.py:499-501`).

        shortcut: `eps0_probes` is fixed at 32 and the search keeps BlackJAX's own
        float-limit stopping rule; raise the probe count if a model shows acceptance
        problems in the first ChEES iterations.
        """
        _check_options("step_size_tune_kwargs", kwargs, _STEP_SIZE_OPTIONS)
        ld = self._logdensity()
        num_probes = int(kwargs.get("eps0_probes", 32))
        seed = kwargs.get("seed", self.seed)
        self._resolved["step_size"] = {"eps0_probes": num_probes, "seed": int(seed)}

        take = min(num_probes, int(len(next(iter(pf_samples.values())))))

        # One dispatch: the probes with a finite log-density (a probe at a non-finite one yields nan
        # acceptance, which BlackJAX turns into "halve forever", step_size.py:239, dragging the median to
        # the float floor) are searched in one vmapped program of a fixed number of probes, so the number
        # of feasible ones does not change the program, and reduced by geometric median.
        eps0, _, _ = compiled.search_step_sizes(
            ld.fn, ld.names, ld.values, inverse_mass_matrix, jax.random.key(seed), pf_samples, take=take)
        eps0 = float(eps0)
        return eps0 if np.isfinite(eps0) else 1e-2      # nothing usable; ChEES dual averaging still runs

    def _run_chees(self, pf_samples, inverse_mass_matrix, init_step_size, **kwargs):
        """The ChEES warmup (`chees_warmup_with_progress`) from the Pathfinder starts and metric.

        chees_kwargs you can pass: ``draws`` (the warmup steps, default 2000), ``adam_lr`` (0.05),
        ``target_accept`` (0.651), ``max_leapfrog_steps`` (1000), ``warmup_stop`` (a stop rule, e.g.
        ``{"block": 250, "window": 100, "rhat": 1.01}``; absent by default, so the warmup runs exactly
        ``draws`` steps), ``metric_updates`` (warmup fractions at which the metric is re-estimated from
        the chains and the step size restarted, e.g. ``(0.25, 0.5, 0.75)``; absent by default, so the
        Pathfinder metric is kept), ``metric_shrinkage`` (weight of the old metric in each re-estimate,
        default ``5 / (chains + 5)``), ``progressbar``, ``seed`` and ``chains``. Any other key raises a
        TypeError that names it. See `chees_warmup_with_progress` for what an update does.
        """
        _check_options("chees_kwargs", kwargs, _CHEES_OPTIONS)
        #Default values
        kwargs.setdefault("chains", 32)
        kwargs.setdefault("draws", 2000)
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("adam_lr", 0.05)
        kwargs.setdefault("target_accept", bj.adaptation.chees_adaptation.OPTIMAL_TARGET_ACCEPTANCE_RATE)
        kwargs.setdefault("max_leapfrog_steps", 1000)

        # The Adam learning rate, not an optax object: it is a static argument of the warmup program, and
        # a fresh optax object would be a new value (hence a new program) on every call.
        optim = float(kwargs["adam_lr"])
        # What this stage resolved, for idata.attrs["config"].
        self._resolved["warmup"] = {
            "seed": int(kwargs["seed"]),
            "num_warmup": int(kwargs["draws"]),
            "adam_lr": float(kwargs["adam_lr"]),
            "target_accept": float(kwargs["target_accept"]),
            "max_leapfrog_steps": int(kwargs["max_leapfrog_steps"]),
            "warmup_stop": kwargs.get("warmup_stop", None),
            "metric_updates": (None if kwargs.get("metric_updates") is None
                               else [float(f) for f in kwargs["metric_updates"]]),
        }
        # Stage-scoped key. This site used to be `_, key_warm = jax.random.split(jax.random.key(seed))`,
        # which produced exactly the key the sampling stage derived, so sampling replayed the warmup
        # key stream's child. Stage numbers are part of the key now: 1 = warmup, 2 = sampling.
        key_warm = jax.random.fold_in(jax.random.key(kwargs["seed"]), 1)
        (chees_states, chees_parameters), chees_info = chees_warmup_with_progress(
            self._logdensity(),
            key_warm,
            pf_samples,            # dict: {"mu": (C,), "tau_log__": (C,), "theta": (C, J)}
            init_step_size,
            optim,
            kwargs["draws"],
            inverse_mass_matrix,   # your PF-based diag/dense (flattened if diag)
            target_acceptance_rate=kwargs['target_accept'],
            max_leapfrog_steps=kwargs['max_leapfrog_steps'],
            progressbar=bool(kwargs.get("progressbar", True)),
            # Warmup stopping rule, e.g. {"block": 250, "window": 100, "rhat": 1.01}.
            # Absent by default: warmup then runs exactly num_steps as before.
            stop_rule=kwargs.get("warmup_stop", None),
            metric_updates=kwargs.get("metric_updates", None),
            metric_shrinkage=kwargs.get("metric_shrinkage", None),
        )
        self._resolved["warmup"]["metric_update_log"] = chees_parameters.get("metric_updates", [])
        if hasattr(chees_info, "info"):
            # utils.WarmupInfo wraps the stacked HMCInfo together with
            # the per-step scalar trace. Everything downstream reads HMCInfo fields
            # (num_integration_steps, is_divergent), so unwrap once at the seam and keep the trace
            # for whoever wants the growth curve. Without it:
            # 'WarmupInfo' object has no attribute 'num_integration_steps'.
            self._warmup_trace = getattr(chees_info, "trace", None)
            chees_info = chees_info.info

        return (
            chees_states,
            chees_parameters,
            chees_info,
        )

    def _run_dynamic_hmc(self, chees_states, chees_parameters, inverse_mass_matrix, **kwargs):
        """Sample from the frozen kernel: host-driven chunks of ONE jitted program (`compiled.sampling_chunk`).

        The number of draws is the host's loop bound, so a different D compiles nothing. The frozen step
        size, metric and trajectory-length adjustment are arguments of the kernel call, and the per-step
        key is ``fold_in(stage_key, draw_index)``. Returns the final state and the per-draw stacks as
        NumPy arrays, ``(draws, chains, ...)``.
        """
        _check_options("dynamic_hmc_kwargs", kwargs, _DYNAMIC_HMC_OPTIONS)
        # Defaults
        kwargs.setdefault("chains", 32)
        kwargs.setdefault("draws", 1000)
        kwargs.setdefault("seed", self.seed)

        rng = jax.random.fold_in(jax.random.key(kwargs["seed"]), 2)  # 2 = sampling stage, see _run_chees
        num_chains = kwargs["chains"]
        num_draws = int(kwargs["draws"])
        # Opt-in delayed rejection on divergence (`compiled.delayed_rejection_step`): an integer a >= 2, the
        # second proposal's step-size divisor. A surgical tool for rare regions the tuned step was never meant
        # for, after the divergences have been diagnosed; not a substitute for a reparameterisation.
        # An integer a is {"factor": a, "stages": 2}; "stages" k >= 2 adds proposals at step / a^2, ... / a^(k-1).
        dr = kwargs.get("delayed_rejection")
        dr_cfg = None
        if dr is not None:
            spec = dr if isinstance(dr, dict) else {"factor": dr}
            if not isinstance(spec, dict) or set(spec) - {"factor", "stages"}:
                raise ValueError(f"delayed_rejection must be None, an integer >= 2 or a dict with 'factor' and "
                                 f"'stages', got {dr!r}")
            factor, stages = spec.get("factor"), spec.get("stages", 2)
            for name, v in (("factor", factor), ("stages", stages)):
                if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 2:
                    raise ValueError(f"delayed_rejection {name} must be an integer >= 2, got {dr!r}")
            dr_cfg = {"factor": int(factor), "stages": int(stages)}
        dr = dr_cfg
        self._resolved["sampling"] = {"seed": int(kwargs["seed"]), "chains": int(num_chains),
                                      "num_draws": int(num_draws), "delayed_rejection": dr}

        ld = self._logdensity()
        imm = jnp.asarray(inverse_mass_matrix)
        step_size = jnp.asarray(chees_parameters["step_size"])
        # The frozen nominal trajectory length L = T/eps (a value), and the jitter setting the warmup used.
        traj_len_adj = jnp.asarray(chees_parameters.get("traj_len_adj", chees_parameters["frozen_L_nominal"]))
        jitter_amount = float(chees_parameters.get("jitter_amount", 1.0))
        rng, key_sample = jax.random.split(rng)

        # ---------- progress bar: updated by the host between chunks, from the chunk's own outputs ----------
        progressbar = bool(kwargs.get("progressbar", True))
        hmc_bar = (tqdm(total=num_draws, desc="Sampling Dynamic HMC", position=0, leave=True)
                   if progressbar else None)
        div_total = [0]
        retry_total = [0, 0]   # retries, accepted retries
        stack = compiled.HostStack(num_draws)

        def consume(start, n, out):
            host = jax.tree_util.tree_map(np.asarray, out)
            stack.add(start, n, host)
            if hmc_bar is None or n == 0:
                return
            info = host[1]
            div_total[0] += int(np.sum(info.is_divergent[:n]))
            hmc_bar.n = start + n
            # Show the acceptance ChEES targeted, not the arithmetic mean of trajectories
            # (see utils.chees_harmonic_acceptance), at the chunk's last draw.
            acc_mean = compiled.chees_harmonic_acceptance_np(info.acceptance_rate[n - 1], info.is_divergent[n - 1])
            retry = ""
            if info.retried is not None:
                retry_total[0] += int(np.sum(info.retried[:n]))
                retry_total[1] += int(np.sum(info.retry_accepted[:n]))
                retry = f" (retried {retry_total[0]}, accepted {retry_total[1]})"
            hmc_bar.set_postfix_str(f"Divergence: {div_total[0]}{retry}, Acceptance Rate: {acc_mean:.3f}")
            hmc_bar.refresh()

        def call(state, start, n):
            return compiled.sampling_chunk(
                ld.fn, ld.names, ld.values, state, imm, step_size, traj_len_adj, key_sample,
                np.int32(start), np.int32(n), chunk=compiled.SAMPLE_CHUNK, jitter_amount=jitter_amount,
                dr_factor=None if dr is None else dr["factor"], dr_stages=2 if dr is None else dr["stages"])

        dhmc_state = compiled.run_chunks(call, chees_states, num_draws, compiled.SAMPLE_CHUNK, consume)
        if hmc_bar is not None:
            hmc_bar.close()
        states_hist, infos_hist = stack.result()
        if dr is not None:
            rate = float(np.mean(infos_hist.retried))
            if rate > DR_RETRY_WARN:
                warnings.warn(
                    f"delayed rejection retried {rate:.2%} of transitions (warning above {DR_RETRY_WARN:.2%}): the "
                    "first-stage step diverges in the bulk, not in a rare region. Diagnose the divergences "
                    "(reparameterise or fix the model) rather than rely on retries.", RuntimeWarning, stacklevel=3)
        return dhmc_state, states_hist, infos_hist



    def _lift_value_to_rv(self, *, keep_untransformed: bool = False, var_names=None):
        """(`LiftedGraph` of value variables -> requested variables, the requested variables)."""
        model = modelcontext(self.model)
        if var_names is None:
            var_names = model.unobserved_value_vars
        vars_to_sample = list(
            get_default_varnames(var_names, include_transformed=keep_untransformed)
        )
        return lift_graph(model, list(model.value_vars), vars_to_sample), vars_to_sample

    def _compile_value_to_rv_fn(self, *, keep_untransformed: bool = False, var_names=None):
        lg, vars_to_sample = self._lift_value_to_rv(
            keep_untransformed=keep_untransformed, var_names=var_names)
        return (lambda *xs: lg.fn(xs, lg.values)), vars_to_sample

    def _transform_value_samples_to_problem_space(
        self,
        states_hist,
        *,
        keep_untransformed: bool = False,
        postprocessing_backend: str = "cpu",
        postprocessing_chunks: int | None = None,
    ):
        """
        Returns dict[rv_name] -> array with shape (chains, draws, *rv.shape) in problem space.
        """
        # Lift the value->problem-space transformer
        lg, vars_to_sample = self._lift_value_to_rv(keep_untransformed=keep_untransformed)

        # Assemble raw_mcmc_samples just like PyMC does (list aligned to value_vars)
        raw_mcmc_samples = self._collect_raw_mcmc_samples(states_hist)

        # Mirror PyMC’s postprocessing: nested vmap over (chains, draws)
        # (optionally you could reproduce their device placement/chunking too)
        if postprocessing_chunks is None:
            result = self._transform_in_draw_chunks(lg, raw_mcmc_samples)
        else:
            from jax.experimental.maps import SerialLoop, xmap
            jax_fn = lambda *xs: lg.fn(xs, lg.values)
            loop = xmap(
                jax_fn,
                in_axes=["chain", "samples", ...],
                out_axes=["chain", "samples", ...],
                axis_resources={"samples": SerialLoop(postprocessing_chunks)},
            )
            f = xmap(loop, in_axes=[...], out_axes=[...])
            # optional: device placement like PyMC
            result = f(*jax.device_put(raw_mcmc_samples, jax.devices(postprocessing_backend)[0]))

        posterior = {v.name: arr for v, arr in zip(vars_to_sample, result)}
        return posterior


    @staticmethod
    def _transform_in_draw_chunks(lg, raw_mcmc_samples):
        """``lg.fn`` over every (chain, draw), in chunks of ``compiled.TRANSFORM_CHUNK`` draws.

        ``raw_mcmc_samples`` are host arrays ``(chains, draws, ...)``. The transform is one program for a
        ``(chains, TRANSFORM_CHUNK)`` batch, with the last chunk padded (by repeating its final draw, so
        the padding is a valid point of the model) and cut again on the host. D is only the number of
        calls, so a different D compiles nothing. One chunk is kept in flight, as in the loops.
        """
        S = int(compiled.TRANSFORM_CHUNK)
        D = int(raw_mcmc_samples[0].shape[1])
        out = None

        def pad(a, n):
            return a if n == S else np.pad(a, [(0, 0), (0, S - n)] + [(0, 0)] * (a.ndim - 2), mode="edge")

        def take(pending):
            nonlocal out
            lo, n, res = pending
            res = [np.asarray(r) for r in res]
            if out is None:
                out = [np.empty((r.shape[0], D) + r.shape[2:], r.dtype) for r in res]
            for o, r in zip(out, res):
                o[:, lo:lo + n] = r[:, :n]

        pending = None
        for lo in range(0, D, S):
            n = min(S, D - lo)
            res = compiled.transform_draws(lg.fn, lg.values, [pad(a[:, lo:lo + n], n) for a in raw_mcmc_samples])
            if pending is not None:
                take(pending)
            pending = (lo, n, res)
        if pending is not None:
            take(pending)
        return out

    def _collect_raw_mcmc_samples(self, states_hist):
        """
        Convert your scan-stacked BlackJAX states into PyMC-style raw_mcmc_samples:
        list aligned with model.value_vars, each array shaped (chains, draws, *param_shape).
        """
        # 1) Pull out position from the BlackJAX state history
        pos_hist = getattr(states_hist, "position", None)
        if pos_hist is None:
            # Some JAX/BlackJAX versions may pack namedtuple fields into tuples post-scan;
            # index 0 is position in HMCState-like structures.
            try:
                pos_hist = states_hist[0]
            except Exception as e:
                raise TypeError(
                    "Could not extract 'position' from states_hist. "
                    "Expected attribute '.position' or index [0]."
                ) from e

        # 2) Build list aligned with model.value_vars order
        if isinstance(pos_hist, dict):
            arrs = [pos_hist[name] for name in self.rvs]  # each (draws, chains, ...)
        elif isinstance(pos_hist, (list, tuple)):
            # Assume same order as self.rvs
            arrs = list(pos_hist)
        else:
            # Single-variable model
            arrs = [pos_hist]

        # 3) Reorder leading axes to (chains, draws, ...)
        def to_chains_draws(a):
            # Our scan stacked over draws first, then we vmapped over chains,
            # so leaves are typically (draws, chains, ...). Swap the first two axes. On the host, in
            # NumPy: a device swap has a D-dependent shape, which is one XLA compile per D.
            a = np.asarray(a)
            return np.swapaxes(a, 0, 1) if a.ndim >= 2 else a

        raw_mcmc_samples = [to_chains_draws(a) for a in arrs]
        return raw_mcmc_samples

    def _to_inference_data(
        self,
        states_hist,
        infos_hist=None,
        *,
        draw_axis: int = 0,
        chain_axis: int = 1,
        keep_untransformed: bool = False,
    ):
        """
        Build ArviZ InferenceData with:
        • posterior in PROBLEM space (incl. Deterministic),
        • sample_stats from the BlackJAX transition info + the sampled logdensity.

        Assumes states_hist leaves are shaped (draws, chains, ...) by default;
        override draw_axis/chain_axis if you stacked differently.
        """
        import numpy as np
        import arviz as az
        import jax.numpy as jnp

        # ---- helper: move (draw_axis, chain_axis) -> (chain, draw)
        def _swap_cd(x):
            na = x.ndim
            da = draw_axis % na
            ca = chain_axis % na
            if da == ca:
                raise ValueError("draw_axis and chain_axis must be different.")
            rest = [ax for ax in range(na) if ax not in (ca, da)]
            perm = [ca, da] + rest
            return np.transpose(np.asarray(x), perm)

        # ---- 1) Posterior in PROBLEM space (and deterministics)
        posterior_ps = self._transform_value_samples_to_problem_space(
            states_hist, keep_untransformed=keep_untransformed
        )  # dict[name] -> (chains, draws, ...)

        # Build coords/dims for parameter axes > 0
        coords, dims = {}, {}
        posterior_np = {}
        for name, arr in posterior_ps.items():
            arr_np = np.asarray(arr)  # already (chains, draws, ...)
            posterior_np[name] = arr_np
            if arr_np.ndim > 2:
                trailing = arr_np.shape[2:]
                dim_names = []
                for i, L in enumerate(trailing):
                    dname = f"{name}_dim_{i}"
                    coords[dname] = np.arange(L)
                    dim_names.append(dname)
                dims[name] = dim_names

        # ---- 2) Sample stats: explicit BlackJAX fields.
        # Do NOT flatten HMCInfo recursively: HMCInfo.momentum and HMCInfo.proposal
        # (an IntegratorState: position, momentum, logdensity, logdensity_grad) all
        # carry pytrees keyed by the model's own variable names, so flattened keys
        # collided last-writer-wins and sample_stats ended up holding logdensity
        # gradients under names like "mu".
        # lp is the SAMPLED state's logdensity (from states_hist); using
        # HMCInfo.proposal.logdensity instead would report the trajectory endpoint,
        # which is wrong on every rejected transition. Same choice PyMC makes.
        sample_stats = None
        if infos_hist is not None:
            lp_src = getattr(states_hist, "logdensity", states_hist[1])
            sample_stats = {
                "lp": _swap_cd(lp_src),
                "diverging": _swap_cd(infos_hist.is_divergent).astype(bool),
                "acceptance_rate": _swap_cd(infos_hist.acceptance_rate),
                "is_accepted": _swap_cd(infos_hist.is_accepted),
                "energy": _swap_cd(infos_hist.energy),
                "num_integration_steps": _swap_cd(infos_hist.num_integration_steps),
            }
            if getattr(infos_hist, "retried", None) is not None:
                # Delayed rejection: "diverging" stays the first-stage divergence; these say what the retry did.
                sample_stats.update({
                    "retried": _swap_cd(infos_hist.retried).astype(bool),
                    "retry_accepted": _swap_cd(infos_hist.retry_accepted).astype(bool),
                    "diverging_after_retry": _swap_cd(infos_hist.retry_divergent).astype(bool),
                    "retry_integration_steps": _swap_cd(infos_hist.retry_integration_steps),
                })

        # ---- 3) Build InferenceData
        idata = az.from_dict(
            posterior=posterior_np,
            coords=coords if coords else None,
            dims=dims if dims else None,
            sample_stats=sample_stats,
        )
        return idata


    def sample(
        self,
        chees_kwargs:dict={},
        step_size_tune_kwargs:dict={},
        pathfinder_kwargs:dict={},
        dynamic_hmc_kwargs:dict={},
        keep_untransformed: bool = False,
        **kwargs,
    ) -> az.InferenceData:
        # An option nobody reads is an error here, before the first stage runs (each stage checks its own as well).
        _check_options("sample", kwargs, _SAMPLE_OPTIONS)
        for where, given, allowed in (
                ("pathfinder_kwargs", pathfinder_kwargs, _PATHFINDER_OPTIONS),
                ("step_size_tune_kwargs", step_size_tune_kwargs, _STEP_SIZE_OPTIONS),
                ("chees_kwargs", chees_kwargs, _CHEES_OPTIONS),
                ("dynamic_hmc_kwargs", dynamic_hmc_kwargs, _DYNAMIC_HMC_OPTIONS)):
            _check_options(where, given, allowed)
        #Override chains in all kwargs
        kwargs.setdefault("chains", 32)
        if "chains" in kwargs:
            for d in (chees_kwargs, step_size_tune_kwargs, pathfinder_kwargs, dynamic_hmc_kwargs):
                d["chains"] = kwargs["chains"]
                
        #Override dynamic_hmc draws if given
        kwargs.setdefault("draws", 1000)
        if "draws" in kwargs:
            dynamic_hmc_kwargs["draws"] = kwargs["draws"]
        

        self._resolved = {}
        self._refresh_values()
        pf_samples, inverse_mass_matrix = self._run_pathfinder(**pathfinder_kwargs)
        eps0 = self._run_step_size_tune(pf_samples, inverse_mass_matrix, **step_size_tune_kwargs)
        chees_states, chees_parameters, chees_info = self._run_chees(
            pf_samples,
            inverse_mass_matrix,
            eps0,
            **chees_kwargs,
        )
        dhmc_state, states_hist, infos_hist = self._run_dynamic_hmc(
            chees_states,
            chees_parameters,
            # The metric the warmup ended with: Pathfinder's unless chees_kwargs["metric_updates"] replaced it.
            chees_parameters["inverse_mass_matrix"],
            **dynamic_hmc_kwargs,
        )

        idata = self._to_inference_data(
            states_hist,
            infos_hist,
            draw_axis=0,          # your scan stacks draws first
            chain_axis=1,         # and chains second (vmapped)
            keep_untransformed=keep_untransformed,
        )
        # How much warmup actually ran. Visible in the output on purpose: with a stop rule the
        # configured warmup length is an upper bound, and a reader must be able to see the difference.
        idata.attrs["warmup_steps_used"] = int(chees_parameters.get("warmup_steps_used", -1))

        # A8 artifact contract: a speed claim is only auditable if the frozen adaptation values and
        # the *exact* gradient counts travel with the draws. Gradient counts are summed from
        # num_integration_steps, never estimated from L x draws: the trajectory length is jittered
        # per step, so any product of frozen values is an approximation.
        import numpy as _np
        idata.attrs["frozen_step_size"] = float(chees_parameters["step_size"])
        # Nominal (unjittered) trajectory length L = T/eps at the frozen point. The kernel draws
        # ceil(jitter(i) * L) per step, and jitter is a Halton value in (0,1), so the largest jitter
        # available inside the scan's own bit width recovers ceil(L). Bit widths are probed over a
        # small range because max_bits is a property of utils' signature, not of this call.
        idata.attrs["frozen_L_nominal"] = _frozen_nominal_L(
            chees_parameters["integration_steps_fn"],
            int(idata.attrs["warmup_steps_used"]),
        )
        # The metric the draws were made with (kernel coordinate order: the sorted names, as ravel_pytree
        # orders a dict), so a divergence can be replayed with the sampler's own metric. Flattened, because
        # netCDF attributes are 1-D; a diagonal metric has length d, a dense one d * d.
        imm_frozen = _np.asarray(chees_parameters["inverse_mass_matrix"], dtype=_np.float64)
        idata.attrs["frozen_inverse_mass_matrix"] = imm_frozen.ravel()
        idata.attrs["frozen_inverse_mass_matrix_shape"] = _np.asarray(imm_frozen.shape, dtype=_np.int64)
        warm_n = _np.asarray(chees_info.num_integration_steps, dtype=_np.int64)     # (steps, chains)
        samp_n = _np.asarray(infos_hist.num_integration_steps, dtype=_np.int64)   # (draws, chains)
        idata.attrs["warmup_grad_evals_per_chain"] = int(warm_n.sum() // warm_n.shape[1])
        idata.attrs["sampling_grad_evals_per_chain"] = int(samp_n.sum() // samp_n.shape[1])
        if getattr(infos_hist, "retry_integration_steps", None) is not None:
            # The retried chains' leapfrog steps ((a + 1) n each; padded batch slots not counted).
            retry_n = _np.asarray(infos_hist.retry_integration_steps, dtype=_np.int64)
            idata.attrs["sampling_retry_grad_evals_per_chain"] = int(retry_n.sum() // retry_n.shape[1])
        # R2-WP2: the resolved configuration, one JSON string (key list: see _config_record).
        idata.attrs["config"] = self._config_record(idata, chees_parameters, eps0)
        # Which release made the draws (installed metadata; "unknown" for an uninstalled source tree). Not in the
        # golden files (tests/make_golden.py:_collect), so a version bump does not change them.
        from . import __version__ as _pkg_version
        idata.attrs["pfcheeshmc_version"] = _pkg_version
        return idata

    def _config_record(self, idata, chees_parameters, eps0) -> str:
        """JSON string for ``idata.attrs["config"]``: what the sampler RESOLVED, not what was asked.

        Every value is the one the code applied, defaults filled in, so a harness can compare it with
        the flags it passed and can never read "None" for a knob that had a default. Keys:

        Pathfinder / pool / metric (``_run_pathfinder``)
            ``imm_source``; ``ridge`` (added to the Monte-Carlo covariance only, so 0.0 for every other
            source); ``pf_draws``; ``pf_pool_cap`` (the cap applied, 0 = uncapped) and ``pf_pool_rows``
            (rows actually drawn); ``init_source``, ``pm_chains``, ``pm_init``, ``pm_tune``,
            ``score_chunk``.
        Step-size search (``_run_step_size_tune``)
            ``eps0_probes`` (number of probe positions) and ``eps0`` (the step size it returned) for
            the search before warmup.
        Warmup (``_run_chees``)
            ``num_warmup`` (configured warmup length) and ``warmup_steps_used`` (ran; differs only
            with a stop rule), ``adam_lr``, ``target_accept``, ``max_leapfrog_steps``,
            ``warmup_stop`` (null = no stop rule).
        Sampling (``_run_dynamic_hmc``)
            ``chains``, ``num_draws``; ``delayed_rejection`` (only when on).
        Frozen at the end of warmup
            ``frozen_step_size`` (eps) and ``frozen_L_nominal`` (L = T/eps).
        Seeds
            ``seed`` (the sampler's own) and ``stage_seeds`` for pathfinder / step_size / warmup /
            sampling, which differ from ``seed`` only if a stage's kwargs overrode it.
        ``jax_enable_x64`` is recorded because the metric dtype, and with it the pf_closed
        positive-definiteness check, depends on it.
        """
        r = self._resolved
        pf, ss, wu, sm = r["pathfinder"], r["step_size"], r["warmup"], r["sampling"]
        cfg = {
            "imm_source": pf["imm_source"],
            "ridge": pf["ridge"],
            "pf_draws": pf["pf_draws"],
            "pf_pool_cap": pf["pf_pool_cap"],
            "pf_pool_rows": pf["pf_pool_rows"],
            "init_source": pf["init_source"],
            "pm_chains": pf["pm_chains"],
            "pm_init": pf["pm_init"],
            "pm_tune": pf["pm_tune"],
            "score_chunk": pf["score_chunk"],
            "pf_lbfgs": pf["lbfgs"],
            "eps0_probes": ss["eps0_probes"],
            "eps0": float(eps0),
            "num_warmup": wu["num_warmup"],
            "warmup_steps_used": int(idata.attrs["warmup_steps_used"]),
            "adam_lr": wu["adam_lr"],
            "target_accept": wu["target_accept"],
            "max_leapfrog_steps": wu["max_leapfrog_steps"],
            "warmup_stop": wu["warmup_stop"],
            "metric_updates": wu["metric_updates"],
            "metric_update_log": wu["metric_update_log"],
            "chains": sm["chains"],
            "num_draws": sm["num_draws"],
            "frozen_step_size": idata.attrs["frozen_step_size"],
            "frozen_L_nominal": idata.attrs["frozen_L_nominal"],
            "seed": int(self.seed),
            "stage_seeds": {"pathfinder": pf["seed"], "step_size": ss["seed"],
                            "warmup": wu["seed"], "sampling": sm["seed"]},
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        }
        if sm["delayed_rejection"] is not None:   # recorded only when on, so a default run's record is unchanged
            cfg["delayed_rejection"] = sm["delayed_rejection"]
        return json.dumps(cfg, sort_keys=True, default=_json_default)
    


def sample_pf_chees_hmc(
    model: Optional[pm.Model] = None,
    *,
    draws: int = 1000,
    chains: int = 32,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
    keep_untransformed: bool = False,
    # pass-through buckets for your class:
    pathfinder_kwargs: Optional[Dict[str, Any]] = None,
    step_size_tune_kwargs: Optional[Dict[str, Any]] = None,
    chees_kwargs: Optional[Dict[str, Any]] = None,
    dynamic_hmc_kwargs: Optional[Dict[str, Any]] = None,
) -> az.InferenceData:
    """
    PyMC-style entrypoint for Pf-ChEES-HMC.

    Works inside a model context:
        with model:
            idata = sample_pf_chees_hmc(draws=2000, chains=32)

    Or outside:
        idata = sample_pf_chees_hmc(model, draws=2000, chains=4)

    Parameters
    ----------
    model : pm.Model | None
        If None, uses the active `with model:` context.
    draws : int
        Number of post-adaptation Dynamic HMC draws per chain (posterior draws).
    chains : int
        Number of chains.
    random_seed : int | None
        Seed for the PfCheesHMC instance.
    keep_untransformed : bool
        Return the posterior in unconstrained (value-var) space as well. Needed by the bias
        harness (`benchmarks/bias.py`), which compares moments against a NUTS reference in unconstrained space.
    pathfinder_kwargs, step_size_tune_kwargs, chees_kwargs, dynamic_hmc_kwargs : dict | None
        Forwarded to `PfCheesHMC.sample(...)`. ``pathfinder_kwargs["pm_chains"]`` (default 1, at most
        ``chains``) is how many PyMC start points are built; they are shared cyclically, chain c taking
        point ``c % pm_chains``, so 1 shares one point among all the chains and ``chains`` gives each its
        own. The other Pathfinder keys are listed in `PfCheesHMC._run_pathfinder`, the warmup keys in
        `PfCheesHMC._run_chees`, the sampling keys (``delayed_rejection``) in `PfCheesHMC._run_dynamic_hmc`. A key
        a stage does not read raises a TypeError that names it.

    Returns
    -------
    arviz.InferenceData
    """
    # Keep the compiled programs on disk for the next Python process, unless the user opted out or set a JAX cache of
    # their own (README, "Compile once, and the disk cache"); does nothing after the first call.
    ensure_default_compilation_cache()
    _warn_if_x64_contradicts_floatx()

    m = modelcontext(model)  # supports both inside/outside context usage

    sampler = PfCheesHMC(m, seed=random_seed)

    # normalize dicts
    pathfinder_kwargs = {} if pathfinder_kwargs is None else dict(pathfinder_kwargs)
    step_size_tune_kwargs = {} if step_size_tune_kwargs is None else dict(step_size_tune_kwargs)
    chees_kwargs = {} if chees_kwargs is None else dict(chees_kwargs)
    dynamic_hmc_kwargs = {} if dynamic_hmc_kwargs is None else dict(dynamic_hmc_kwargs)

    # One flag controls both bars. The bars are updated by the host between chunks, so on or off the loops
    # are the same compiled programs (no host callback is in either, which also keeps them eligible for
    # JAX's on-disk compilation cache). A per-stage value in chees_kwargs / dynamic_hmc_kwargs still wins.
    chees_kwargs.setdefault("progressbar", progressbar)
    dynamic_hmc_kwargs.setdefault("progressbar", progressbar)

    idata = sampler.sample(
        chains=chains,
        draws=draws,
        keep_untransformed=keep_untransformed,
        pathfinder_kwargs=pathfinder_kwargs,
        step_size_tune_kwargs=step_size_tune_kwargs,
        chees_kwargs=chees_kwargs,
        dynamic_hmc_kwargs=dynamic_hmc_kwargs,
    )
    return idata