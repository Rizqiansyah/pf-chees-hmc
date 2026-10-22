import jax, jax.numpy as jnp
import blackjax as bj
from jax.flatten_util import ravel_pytree
from jax import lax
import optax

import pymc as pm
from pymc import modelcontext
from pymc.util import get_default_varnames
import arviz as az

from tqdm.auto import tqdm
from .utils import *
from . import compiled
from .compiled import LogDensity
from .lifted import lift_graph, lift_logp
from .ensemble_metric import (
    estimate_covariance, reseed_trajectory_length, whitened_lambda_max, positions_to_draws,
    ENGLISH_NAMES as _METRIC_REFRESH_NAMES,
)

from typing import Tuple, Optional, Dict, Any, NamedTuple
import numpy as np
from datetime import date
import io
import json
import re
import logging
import contextlib

__all__ = ["sample_pf_chees_hmc"]


# The sampling scan's per-draw projections are defined with the chunk program that returns them.
_SampledState, _SampledInfo = compiled.SampledState, compiled.SampledInfo


def default_pool_cap(chains, d):
    """Row count for the Pathfinder start/metric pool when the caller does not name one.

    One multiple of `chains` cannot size this pool, because it feeds two consumers with different
    needs, and both failure modes are measured:

    * Too many rows is a hard failure. The pool is materialised as one operand: 4,096,000 x 175 x 8 B
      = 5,734,400,000 B, exactly the allocation XLA refuses on a 6 GB card at C=1,024 with the
      production 4,000-draw pool (`results/runs/T1__prodpool.jsonl`: it fits at d=8, not at d=25).
    * Too few rows is a correctness failure. The same pool is the sample the Monte-Carlo covariance is
      estimated from, so rows must stay large relative to d^2. Capping radon (d=175) at 4C=128 rows at
      C=32 read b2 2.34860 against a 0.07689 floor with split R-hat 1.2121 -- a rank-deficient metric,
      not merely a slower one (`results/runs/T1__r2_poolcap.jsonl`).

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


def _stratified_pool_idx(logp_subset, subset_idx, n_pick):
    """Equally spaced ranks by log-density, chosen from a scored subset of the draw pool.

    Deliberately not Pathfinder's PS-IR: that resamples WITH replacement, and the Pathfinder paper
    reports it collapsing to 4 distinct draws out of 100 on mcycle_gp-accel_gp -- it removes the very
    dispersion this is meant to guarantee. Measured on radon non-centered (32 chains, 500 draws):
    worst R-hat 1.0323 -> 1.0210 and min bulk ESS +6% at warmup 250, identical spread to iid and no
    extra leapfrog steps; indistinguishable from iid at warmup >= 1,000.
    """
    order = jnp.argsort(logp_subset)
    m = int(order.shape[0])
    ranks = (jnp.arange(n_pick) * m) // n_pick
    return subset_idx[order[ranks]]


def _concat_warmup_info(a, b):
    """Concatenate two warmup-info stacks (phase A then phase B) along the step axis.

    Returns a WarmupInfo whose ``info`` (the HMCInfo stack) and ``trace`` are concatenated so a
    downstream reader sees one warmup of length ``a_steps + b_steps``. ``positions`` is set to
    None: it is consumed only to build the refreshed estimator and is not read downstream, and
    keeping it would hold the (steps, C, ...) position stack for no reason.

    The stacks are host (NumPy) arrays and are joined with NumPy: a device concatenate has a W-dependent
    shape, which is one XLA compile per W.
    """
    from .utils import WarmupInfo
    ainfo = getattr(a, "info", a)
    binfo = getattr(b, "info", b)
    info = jax.tree_util.tree_map(
        lambda x, y: np.concatenate([np.asarray(x), np.asarray(y)], axis=0), ainfo, binfo)
    atr, btr = getattr(a, "trace", None), getattr(b, "trace", None)
    if atr is not None and btr is not None:
        # trace is a ChEESTrace named tuple (one (steps,) array per field), not a 2-D array,
        # so concatenate field-wise through the pytree, not with a single concatenate.
        trace = jax.tree_util.tree_map(
            lambda x, y: np.concatenate([np.asarray(x), np.asarray(y)], axis=0), atr, btr)
    else:
        trace = None
    return WarmupInfo(info=info, trace=trace, positions=None)


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
        arg = jnp.asarray(2 ** bits - 2, dtype=jnp.int64)   # all-ones index -> jitter 1-2^-bits
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

    @staticmethod
    def _pathfinder_mean(pf_state):
        """Mean of Pathfinder's Gaussian, as a pytree shaped like ``pf_state.position`` (tuple order).

        The ``mu`` that ``blackjax.optimizers.lbfgs.bfgs_sample`` draws around (lbfgs.py:429):
        ``position + alpha * grad + beta (gamma (beta^T grad))``, i.e. ``position + formula_1 @ grad``
        without forming the d x d matrix. Not the stored ``position``: the fit stops at the L-BFGS
        iterate and the Gaussian is centred one Newton step away from it.
        """
        pos, unravel = ravel_pytree(pf_state.position)
        g, _ = ravel_pytree(pf_state.grad_position)
        return unravel(pos + pf_state.alpha * g + pf_state.beta @ (pf_state.gamma @ (pf_state.beta.T @ g)))

    def _scale_starts(self, starts, pf_state, scale):
        """``x_i -> mu_pf + scale * (x_i - mu_pf)`` for every chain start; ``mu_pf`` = Pathfinder mean.

        Acts on chain starts only; the covariance handed to the kernel is not touched. ``starts`` is
        the tuple (``self.rvs`` order) the pool pick produces or the dict the adapt_diag fallback
        produces; the result has the same structure. A scale that carries a start to a point with a
        non-finite log-density or gradient raises instead of leaving a NaN chain.
        """
        mu = dict(zip(self.rvs, self._pathfinder_mean(pf_state)))
        if isinstance(starts, dict):
            out = {n: mu[n] + scale * (starts[n] - mu[n]) for n in self.rvs}
        else:
            out = tuple(mu[n] + scale * (x - mu[n]) for n, x in zip(self.rvs, starts))
        as_dict = out if isinstance(out, dict) else dict(zip(self.rvs, out))
        bad = int((~self._usable_mask(as_dict)).sum())
        if bad:
            raise ValueError(
                f"pf_start_scale={scale} puts {bad} of {next(iter(as_dict.values())).shape[0]} chain "
                "starts at a point with a non-finite log-density or gradient; use a smaller scale")
        return out

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
        center=None,
        restart_radius: float = 0.05,
    ):
        """
        Make a bad start usable without changing the model. A start must have a finite
        log-density AND a finite gradient. Order of repair:
        1) tiny relative jitter (fixes exact singularities like xi=0),
        2) backtrack toward `center` (default zeros in unconstrained space),
        3) random restarts around center with growing radius.
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
            """
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
        jitter = jax.random.normal(k0, flat0.shape) * (jitter_eps * rel)
        x = unrav(flat0 + jitter)
        if _is_usable(x):
            return x

        # (2) backtrack toward safe center (default zeros)
        if center is None:
            center = unrav(jnp.zeros_like(flat0))
        for _ in range(max_backtracks):
            x_try = _interp(x, center, 0.5)
            if _is_usable(x_try):
                return x_try
            x = x_try

        # (3) random restarts around center
        radius = restart_radius
        for _ in range(max_restarts):
            key, k = jax.random.split(key)
            noise = jax.random.normal(k, flat0.shape) * radius
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

    def _pymc_adapt_diag_start(
        self,
        chains: int,
        *,
        init: str = "adapt_diag",   # or "jitter+adapt_diag"
        tune: int = 500,
        random_seed: Optional[int] = None,
    ):
        """
        Run PyMC's NUTS initializer to get per-chain feasible initial points
        (no posterior sampling). Returns dict{name: (chains, ...)} keyed by self.rvs.
        """
        print(f"Initializing ChEES-HMC using pathfinder+{init}...")
        with self._silence_pymc_init():
            init_points, _ = pm.init_nuts(
                init=init,
                chains=chains,
                tune=tune,
                model=self.model,
                random_seed=random_seed,
                progressbar=True,
            )
        # list[chain] of dicts -> dict of stacked arrays in self.rvs order
        return {
            name: jnp.stack([jnp.asarray(ip[name]) for ip in init_points], axis=0)
            for name in self.rvs
        }

    def _usable_mask(self, init_stack):
        """One jitted, vmapped check for finite log-density AND finite gradient, all chains at once.

        The repair loop below is per-chain and eager: 32 separate calls into the compiled jaxed
        logp costs 1.75 s on 8 Schools and 4.24 s on radon, which is Python dispatch, not compute.
        Chains that already pass never enter that loop.
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
        center=None,
        restart_radius=0.05,
    ):
        """
        For a dict{name: (chains, ...)} ensure every chain's point has a finite log-density and a
        finite gradient. Applies _find_finite_start per-chain if needed.
        Returns a dict{name: (chains, ...)} ready for Pathfinder.

        Note on jitter: points that already pass are returned untouched. Previously every chain was
        jittered by ~1e-8 relative, even when its start was fine. Same distribution, different
        realisation.
        """
        chains = next(iter(init_stack.values())).shape[0]
        keys = jax.random.split(key, chains)
        fixed = {n: [] for n in self.rvs}
        try:
            usable = self._usable_mask(init_stack)
        except Exception:                     # a batched check that cannot run must not cost us the run
            usable = jnp.zeros(chains, dtype=bool)

        for c in range(chains):
            if bool(usable[c]):
                for n in self.rvs:
                    fixed[n].append(init_stack[n][c])
                continue
            x0 = {n: init_stack[n][c] for n in self.rvs}
            try:
                x0_safe = self._find_finite_start(
                    x0, keys[c],
                    jitter_eps=jitter_eps,
                    max_backtracks=max_backtracks,
                    max_restarts=max_restarts,
                    center=center,
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
                    center=center,
                    restart_radius=restart_radius,
                )
            for n in self.rvs:
                fixed[n].append(x0_safe[n])

        return {n: jnp.stack(fixed[n], axis=0) for n in self.rvs}

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
        pm_tune: int (default 500)
        pm_seed: int (default self.seed)
        init_source: "pf" | "adapt_diag" (default "pf")
        ridge: float covariance ridge (default 1e-6; the "mc" covariance only)
        imm_source: "mc" (default) | "pf_closed" | "lbfgs_lowrank" | "fisher_lowrank"
        pf_pool_cap: pool row cap; None -> see resolve_pool_cap (depends on imm_source), 0 = uncapped
        pf_start_scale: float > 0 (default 1.0). Chain starts become mu_pf + scale * (x - mu_pf), with
            mu_pf the Pathfinder mean; the covariance is not scaled and 1.0 skips the transform
        pf_jitter_eps, pf_max_backtracks, pf_max_restarts, pf_center, pf_restart_radius
        """
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
        # per-chain behaviour.
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
        pf_center         = kwargs.get("pf_center", None)
        pf_restart_radius = kwargs.get("pf_restart_radius", 0.05)

        key = jax.random.key(seed)
        key_fix, key_pf, key_samp, key_pick, key_choose = jax.random.split(key, 5)

        # 0) Ask PyMC for feasible-ish starts -- one point shared by all chains by default
        init_stack = self._pymc_adapt_diag_start(
            pm_chains, init=pm_init, tune=pm_tune, random_seed=pm_seed
        )

        # 1) Ensure every init is finite (handles xi=0 -> -inf cases). With a shared point the
        #    check and any repair run once on the unique points, then the result is broadcast --
        #    the same finite-gradient guard as before, without evaluating it C times on C
        #    identical rows.
        if pm_chains < chains:
            init_stack = self._ensure_finite_inits(
                init_stack, key_fix,   # split inside, into pm_chains subkeys
                jitter_eps=pf_jitter_eps,
                max_backtracks=pf_max_backtracks,
                max_restarts=pf_max_restarts,
                center=pf_center,
                restart_radius=pf_restart_radius,
            )
            init_stack = {
                n: jnp.broadcast_to(a, (chains,) + a.shape[1:])
                for n, a in init_stack.items()
            }
        else:
            init_stack = self._ensure_finite_inits(
                init_stack, key_fix,
                jitter_eps=pf_jitter_eps,
                max_backtracks=pf_max_backtracks,
                max_restarts=pf_max_restarts,
                center=pf_center,
                restart_radius=pf_restart_radius,
            )

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

        # 3) Fit Pathfinder
        pf_state = compiled.pathfinder_fit(
            bj.vi.pathfinder.approximate, ld.fn, ld.names, ld.values,
            key_pf, tuple(x0_pf[n] for n in self.rvs),
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
            pick = kwargs.get("init_pick", "stratified")
            if pick == "iid":
                pick_global = jax.random.randint(key_pick, shape=(chains,), minval=0, maxval=total)
            elif pick == "stratified":
                # Score a strided subset -- ranking the whole uncapped pool would cost more than
                # the benefit, which only exists at short warmup. With the pool capped at ~4C the
                # stride is 1 and the whole pool is scored, chunked to keep the vmapped
                # log-density activation at O(score_chunk) rows (a single 65,536-row vmap was
                # the second R1 OOM, ~500 MiB at d=25).
                stride = max(total // max(chains * 64, 1), 1)
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
            else:
                raise ValueError(f"init_pick must be 'iid' or 'stratified', got {pick!r}")
            # samples_flat is the whole pool in (chains, draws) chunk order; the chunk axis is just
            # how Pathfinder emitted the draws, so any row may seed any chain.
            #
            # superchains=K (WP3, nested R-hat): keep the first K picks and replicate each C//K times,
            # so the C chains form K equal-sized superchains that start at K *distinct* Pathfinder
            # points -- which is what makes between-superchain variation a real signal (arXiv
            # 2110.13017). Absent (default) leaves pick_global byte-identical to the previous pick.
            K = kwargs.get("superchains")
            if K is not None:
                K = int(K)
                if K < 1 or K > chains:
                    raise ValueError(f"superchains must be in [1, chains={chains}], got {K}")
                if chains % K:
                    raise ValueError(f"chains={chains} must be divisible by superchains={K}")
                pick_global = jnp.repeat(pick_global[:K], chains // K)
            pf_inits_tree = jax.tree_util.tree_map(lambda x: x[pick_global], samples_flat)
        elif init_source == "adapt_diag":
            # use the repaired adapt_diag starts as chain inits. With a shared base point
            # (pm_chains < chains) add a tiny per-chain relative jitter so the option does not
            # degenerate to C identical starts -- identical starts read R-hat 1.000 while
            # stuck, which no diagnostic flags. The base point is guard-validated; a 1e-3
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

        # 5b) Start dispersion. Rescale the picked starts about the Pathfinder mean; the covariance
        # (6, 6b) is never scaled. Exactly 1.0 skips the transform, so the default is bit-identical
        # to the code before this knob existed (x + 1.0 * (y - x) is not bitwise y in floating point).
        pf_start_scale = float(kwargs.get("pf_start_scale", 1.0))
        if not (pf_start_scale > 0.0 and jnp.isfinite(pf_start_scale)):
            raise ValueError(f"pf_start_scale must be a finite number > 0, got {pf_start_scale!r}")
        if pf_start_scale != 1.0:
            pf_inits_tree = self._scale_starts(pf_inits_tree, pf_state, pf_start_scale)

        # 6) Dense covariance from ALL PF samples. This is a COVARIANCE in Pathfinder's own tuple
        # order; whether it is handed to the kernel as-is or inverted is the `imm_orientation` choice
        # decided in step 6c, and the reorder to kernel order happens in step 8.
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
        # A low-rank source produces the same kind of object -- a COVARIANCE estimate -- so the
        # orientation is not its property, it is chosen once in 6c.
        if imm_source != "mc":
            if imm_source == "pf_closed":
                # The exact covariance Pathfinder's own sampler draws with: bfgs_sample builds
                # phi = mu + sqrt(alpha) * (Q (L - I) Q^T u + u) with L L^T = I + R gamma R^T and
                # Q R = beta / sqrt(alpha), whose covariance is diag(alpha) + beta gamma beta^T
                # (formula II.1 of zhang2022pathfinder). So the MC pool covariance above is a noisy
                # estimate of this matrix; here it is taken in closed form, from the one fit made in
                # step 3. It is in Pathfinder's flat (tuple, self.rvs) order and goes through the
                # same orientation (6c) and kernel-order choke point (8) as "mc". No ridge: it is
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
            # Note: rescaling this metric to the empirical covariance's geometric mean was tried and
            # is NOT implemented -- it made every measured model worse (8 Schools 50,626 -> 13,628
            # ESS/s, radon centered R-hat 2.10 -> 5.69). See docs/ladder-results.md. That comparison
            # is now known to be confounded: see 6c.

        # 6c) Orientation. The array the kernel calls `inverse_mass_matrix` (call it A) is the matrix
        # that multiplies the momentum in the kinetic energy, KE = 1/2 p.(A p) (mcmc/metrics.py:
        # 262-270), so momentum ~ N(0, A^-1) and leapfrog gives q'' = -A * grad U. For a target with
        # Hessian H, the squared frequencies are eig(A H): A = H^-1 = Cov(q) makes every coordinate
        # oscillate at unit frequency (dimensionless eps), while A = H makes them 1/var, so one global
        # eps cannot serve narrow and wide coordinates at once.
        #
        # Historically the pipeline was NOT internally consistent here: "mc" passed inv(covariance)
        # while the two low-rank sources passed their expansion un-inverted, i.e. "mc" ran in
        # "precision" and "lbfgs_lowrank"/"fisher_lowrank" in "covariance". Every published
        # mc-vs-low-rank comparison in docs/ladder-results.md therefore compared opposite
        # orientations. The knob now applies to all three arms identically, which makes the low-rank
        # sources behave consistently with `mc` for the first time (a behaviour change for them).
        #
        # The default is "covariance" (owner decision, 2026-10-20; the flip and the coordinate
        # permutation land in the same merge because holding "precision" next to a fixed
        # permutation would ship a state that fails R-hat <= 1.01 on GEV in 5 of 5 seeds).
        # Measured reason: the slot array is M^-1 in KE = 0.5 p^T A p, and leapfrog frequencies are
        # eig(A H) with H = Sigma^-1, so the well-conditioned value of A is the posterior
        # covariance. Accepted cost: GEV's divergence rate rises 0.025 -> 0.400 per 1k, seed-stable.
        # One measured caveat to carry into WP4 (which owns the metric), from CPU runs on a 3-scalar
        # conjugate Gaussian with posterior variances 1e1 / 1e0 / 1e-2 and reverse-sorted names
        # (32 chains, 500 warmup, 1,000 draws, seeds 5-7): "precision" froze at eps=0.0138 with
        # max R-hat 10.10 and bulk ESS 33 / 41 / 27,221, while "covariance" gave eps=1.14,
        # max R-hat 1.0024 and ESS 24,664 / 24,377 / 25,117 -- worst-variable ESS per gradient 0.012
        # vs 6.6-13.9. Both arms pass a 4-MCSE moment check, because Pathfinder's starts already
        # sample that Gaussian posterior: the ensemble mean is right whether or not the chains move,
        # so a moment test alone cannot arbitrate orientation. See results/WP1/SUMMARY.md section 6
        # and test_precision_orientation_does_not_mix_an_anisotropic_target.

        orientation = kwargs.get("imm_orientation", "covariance")
        if orientation == "precision":
            inverse_mass_matrix = jnp.linalg.inv(covariance)
        elif orientation == "covariance":
            inverse_mass_matrix = covariance
        else:
            raise ValueError(
                f"imm_orientation must be 'precision' or 'covariance', got {orientation!r}")
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
        # imm_order="pathfinder" skips the reorder. It exists only so the 2x2 (ordering x
        # orientation) can be measured in ONE tree with one key stream; nothing in a production run
        # should set it, and test_metric_orientation.py pins the default.
        order = kwargs.get("imm_order", "kernel")
        if order == "kernel":
            inverse_mass_matrix = self._metric_to_kernel_order(inverse_mass_matrix, perm)
        elif order != "pathfinder":
            raise ValueError(f"imm_order must be 'kernel' or 'pathfinder', got {order!r}")

        self._resolved["pathfinder"] = {
            "seed": int(seed),
            "imm_source": imm_source,
            "imm_orientation": orientation,
            "imm_order": order,
            # the ridge is added to the Monte-Carlo covariance only; every other source overwrites it
            "ridge": float(ridge) if imm_source == "mc" else 0.0,
            "pf_draws": int(draws),
            "pf_pool_cap": 0 if cap is None else int(cap),     # 0 = uncapped, as the kwarg
            "pf_pool_rows": int(total),
            "pf_start_scale": pf_start_scale,
            "init_source": init_source,
            "init_pick": kwargs.get("init_pick", "stratified"),
            "superchains": kwargs.get("superchains", None),
            "pm_chains": pm_chains,
            "pm_init": pm_init,
            "pm_tune": int(pm_tune),
            "score_chunk": score_chunk,
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
        uninstrumented). Here it is traced once and dispatched per probe, so cost is
        independent of the number of sampling chains — the sampler itself is flat from 32
        to 256 chains (+2% per transition), and the old stage made wall time grow linearly.

        Probes are positions, not chains: 32 of them are enough to estimate a global ε0,
        and ChEES re-tunes ε by dual averaging from iteration 1. Do not read ε0 as a mere
        warm start with no consequences — on the stiff GEV model a different ε0 changed the
        divergence count from 33 to 255 — it anchors dual averaging (`dual_averaging.py:94`)
        and sets the ChEES length ceiling (`chees_adaptation.py:499-501`).

        shortcut: `eps0_probes` is fixed at 32 and the search keeps BlackJAX's own
        float-limit stopping rule; raise the probe count if a model shows acceptance
        problems in the first ChEES iterations.
        """
        ld = self._logdensity()
        num_probes = int(kwargs.get("eps0_probes", 32))
        seed = kwargs.get("seed", self.seed)
        self._resolved["step_size"] = {"eps0_probes": num_probes, "seed": int(seed)}

        n_drawn = int(len(next(iter(pf_samples.values()))))
        take = min(num_probes, n_drawn)
        probes = [{k: jnp.asarray(v)[i] for k, v in pf_samples.items()} for i in range(take)]

        # A probe at a position with non-finite logp yields nan acceptance, which BlackJAX
        # turns into "halve forever" (step_size.py:239), dragging the median to the float floor.
        feasible = compiled.feasible_mask(
            ld.fn, ld.names, ld.values, {k: jnp.stack([p[k] for p in probes]) for k in probes[0]})
        keep = [bool(k) for k in feasible]
        probes = [p for p, k in zip(probes, keep) if k]
        if not probes:
            return 1e-2  # nothing usable; ChEES dual averaging still runs

        def search(key, pos_i):
            return compiled.search_step_size(
                ld.fn, ld.names, ld.values, inverse_mass_matrix, key, pos_i)

        # One trace, then one dispatch per probe. Per-probe rather than vmapped for cost
        # and clarity of failure modes, not correctness: `jax.vmap` over this search is also
        # correct, because JAX lowers a batched `while` predicate by re-evaluating it in the
        # body and selecting the previous carry for finished elements, so converged probes
        # are frozen (verified in jax/_src/lax/control_flow/loops.py:2210-2224 and by
        # comparing per-probe values vmap vs serial: identical).
        keys = jax.random.split(jax.random.key(seed), len(probes))
        eps_list = [float(search(keys[i], probes[i])) for i in range(len(probes))]
        eps_ok = [e for e in eps_list if jnp.isfinite(e) and e > 0.0]
        if not eps_ok:
            return 1e-2

        # geometric median: the values are dyadic, so a linear median biases upward
        return float(jnp.exp(jnp.median(jnp.log(jnp.asarray(eps_ok)))))

    def _run_chees(self, pf_samples, inverse_mass_matrix, init_step_size, **kwargs):
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
        refresh = kwargs.get("metric_refresh", "none")
        refreshing = refresh in _METRIC_REFRESH_NAMES
        # What this stage resolved, for idata.attrs["config"]. The three estimator defaults below
        # repeat the ones in _run_chees_two_phase (a refresh is only applied when `refresh` names an
        # estimator, so anything else is recorded as "none"); tests/test_pf_closed_form.py spies on
        # estimate_covariance so the two cannot drift apart unnoticed.
        self._resolved["warmup"] = {
            "seed": int(kwargs["seed"]),
            "num_warmup": int(kwargs["draws"]),
            "adam_lr": float(kwargs["adam_lr"]),
            "target_accept": float(kwargs["target_accept"]),
            "max_leapfrog_steps": int(kwargs["max_leapfrog_steps"]),
            "iterate": kwargs.get("iterate", "ma"),
            "warmup_stop": kwargs.get("warmup_stop", None),
            "metric_refresh": refresh if refreshing else "none",
            "metric_refresh_ridge": float(kwargs.get("metric_refresh_ridge", 1e-6)) if refreshing else None,
            "metric_refresh_shrink": float(kwargs.get("metric_refresh_shrink", 0.5)) if refreshing else None,
            "metric_refresh_rank": int(kwargs.get("metric_refresh_rank", 5)) if refreshing else None,
        }
        # WP4 two-phase metric refresh. ``metric_refresh`` names an ensemble estimator
        # (see pfcheeshmc/ensemble_metric.py); when set, phase A is a short fixed warmup on the
        # Pathfinder metric, the ensemble covariance is estimated from phase-A draws, and phase B
        # continues on the refreshed metric with a re-searched eps and re-seeded trajectory length.
        # ``"none"`` (default) keeps the single-phase path bit-for-bit.
        if refreshing:
            return self._run_chees_two_phase(
                pf_samples, inverse_mass_matrix, init_step_size, refresh, optim, **kwargs)
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
            # Until R2-WP2 this call did not forward `iterate`, so chees_kwargs={"iterate": "raw"}
            # without a metric refresh silently ran "ma" (only the two-phase path passed it on).
            iterate=kwargs.get("iterate", "ma"),
        )
        if hasattr(chees_info, "info"):
            # utils.WarmupInfo (the vendored-adapter path) wraps the stacked HMCInfo together with
            # the per-step scalar trace. Everything downstream reads HMCInfo fields
            # (num_integration_steps, is_divergent), so unwrap once at the seam and keep the trace
            # for whoever wants the growth curve. Found by the integration smoke cell on
            # pf-r1-int: 'WarmupInfo' object has no attribute 'num_integration_steps'.
            self._warmup_trace = getattr(chees_info, "trace", None)
            chees_info = chees_info.info

        return (
            chees_states,
            chees_parameters,
            chees_info,
        )

    def _run_chees_two_phase(self, pf_samples, imm_A, eps_A, refresh, optim, **kwargs):
        """WP4 two-phase warmup: short fixed warmup on the Pathfinder metric, then a refresh.

        Phase A runs ``metric_refresh_warmup`` steps of the ordinary ChEES warmup on ``imm_A``
        and collects its draws. The named ensemble estimator builds a COVARIANCE (kernel order)
        from those draws; eps is re-searched on the new metric (the same traced 32-probe search
        the shipped path uses, so the cost is independent of chain count) and the trajectory
        length is re-seeded at
        ``T0_B = (pi/2) * sqrt( lambda_max( L^-1 S L^-T ) )`` with ``B = L L^T`` the refreshed
        metric and ``S`` the sample covariance (ddof=1) of the same phase-A pool, then clipped to
        ``[eps_B, max_leapfrog_steps * eps_B]`` so ``1 <= L0 <= max_leapfrog_steps``. Phase B then
        continues the warmup to the budget from phase A's final positions, on the refreshed metric.

        Everything downstream of this method is unchanged: it returns the same
        ``(states, parameters), info`` contract as :meth:`_run_chees`, so ``sample()`` and the
        A8 attrs write the frozen values and gradient counts exactly as before.
        """
        W_total = int(kwargs["draws"])
        W_A = int(kwargs.get("metric_refresh_warmup", 50))
        W_A = max(2, min(W_A, W_total - 2))
        W_B = W_total - W_A

        # Stage-scoped key, same as the single-phase path; the two phases get distinct children
        # so a refresh never replays the phase-A key stream.
        key_warm = jax.random.fold_in(jax.random.key(kwargs["seed"]), 1)
        keyA, keyB = jax.random.split(key_warm)

        # Ensemble pool: the last `pool` phase-A steps, flattened in kernel order.
        pool = int(kwargs.get("metric_refresh_pool", min(W_A, 20)))
        pool = max(2, min(pool, W_A))

        # ---- Phase A: short fixed warmup on the Pathfinder metric, collect draws ----------
        (states_A, params_A), info_A = chees_warmup_with_progress(
            self._logdensity(), keyA, pf_samples, eps_A, optim, W_A, imm_A,
            target_acceptance_rate=kwargs["target_accept"],
            max_leapfrog_steps=kwargs["max_leapfrog_steps"],
            progressbar=False, stop_rule=None,
            stack_positions=True,      # phase A must keep its draws for the ensemble estimate
            positions_last=pool,       # ... of which only the last `pool` steps are read
            iterate=kwargs.get("iterate", "ma"))
        pos_hist = getattr(info_A, "positions", None)
        if pos_hist is None:
            # Unreachable on the current routing: the phase-A call above passes stack_positions=True,
            # which makes that one call stack its positions. Kept as a regression tripwire -- if the
            # routing is ever changed back to the plain default path the refresh would silently build its
            # ensemble from nothing, and a stale metric is worse than no refresh.
            raise RuntimeError(
                "two-phase metric refresh asked for phase-A positions (stack_positions=True) and got "
                "None. This is a routing regression in utils.chees_warmup_with_progress, not a "
                "configuration the user can set: the plain default path returns positions=None on "
                "purpose (that stack is WP5's C=1,024 memory win), and the refresh must not be "
                "routed through it.")

        # The pool is the stacked positions' last rows, already gathered on the host (NumPy).
        pos_pool = {n: jnp.asarray(p) for n, p in pos_hist.items()}
        draws = positions_to_draws(pos_pool, pool)                       # (pool*C, d)

        # Gradients are needed only by the Fisher estimator; chunked so peak memory is O(chunk).
        grads = None
        if refresh == "fisher_diag":
            grads = self._pool_gradients(pos_pool, draws, kwargs)

        # ---- The swap: build the refreshed covariance (kernel order), re-search eps, re-seed T
        imm_B = estimate_covariance(
            refresh, draws, grads,
            ridge=float(kwargs.get("metric_refresh_ridge", 1e-6)),
            shrink=float(kwargs.get("metric_refresh_shrink", 0.5)),
            max_rank=int(kwargs.get("metric_refresh_rank", 5)))
        # HMC state is a named tuple (index 0 = position dict); match the access the rest of
        # the sampler uses (states_hist[0] / .position), not a string key.
        pos_A = getattr(states_A, "position", None)
        if pos_A is None:
            pos_A = states_A[0]
        eps_B = self._run_step_size_tune(pos_A, imm_B, **kwargs)         # traced, C-independent
        # metric_refresh_reseed (default True) re-seeds the phase-B trajectory length at
        #     T0_B = (pi/2) * sqrt( lambda_max( L^-1 S L^-T ) ),  B = L L^T = imm_B,
        #     S = sample covariance (ddof=1) of the SAME `draws` the estimator above consumed.
        # Whitening S by the metric makes T0 invariant to the units of the coordinates, which the
        # run-1 formula (pi/2)/sqrt(lambda_min(imm_B)) was not -- that one read the precision of the
        # narrowest posterior coordinate in MODEL units, so a coordinate rescale moved T0 and frozen
        # L landed at 213-283 (plan Part B 1.1). False skips the re-seed entirely: phase B then
        # starts at L=1 (t0=None -> trajectory_length = step_size) and grows T with its own dynamics
        # on the refreshed metric, which is the control that isolates the re-seed's effect.
        reseed = bool(kwargs.get("metric_refresh_reseed", True))
        lmax_B = whitened_lambda_max(imm_B, draws) if reseed else None
        T0_B = None
        if reseed:
            # Clip on the step-size and leapfrog budget this warmup actually has, so the re-seeded
            # trajectory length L0 = T0_B / eps_B is inside [1, max_leapfrog_steps].
            # (Plain float min/max, not jnp.clip: that would round T0 to float32 when x64 is off.)
            # ``kwargs["max_leapfrog_steps"]`` is always set here: ``_run_chees`` does
            # ``kwargs.setdefault("max_leapfrog_steps", 1000)`` before routing to this method.
            eps_f = float(eps_B)
            max_L = int(kwargs["max_leapfrog_steps"])
            T0_B = float(min(max(reseed_trajectory_length(imm_B, draws), eps_f), max_L * eps_f))

        # ---- Phase B: continue to the budget on the refreshed metric -----------------------
        (states_B, params_B), info_B = chees_warmup_with_progress(
            self._logdensity(), keyB, pos_A, eps_B, optim, W_B, imm_B,
            target_acceptance_rate=kwargs["target_accept"],
            max_leapfrog_steps=kwargs["max_leapfrog_steps"],
            progressbar=bool(kwargs.get("progressbar", True)), stop_rule=None,
            iterate=kwargs.get("iterate", "ma"), t0=T0_B)

        # Merge the frozen bookkeeping so the A8 attrs and any row schema see one warmup.
        params_B["warmup_steps_used"] = W_A + W_B
        params_B["metric_refresh"] = refresh
        params_B["metric_refresh_warmup"] = W_A
        params_B["metric_refresh_pool"] = pool
        # eps bookkeeping for the A8 row schema and the "re-search moves eps" test:
        #   eps0            -- the step size the swap started from (phase-A input),
        #   phaseA_end_eps  -- what the ChEES adapter had settled at the end of phase A,
        #   eps_researched  -- the re-search on the NEW metric (must differ from phaseA_end).
        params_B["metric_refresh_eps0"] = float(eps_A)
        params_B["metric_refresh_phaseA_end_eps"] = float(params_A["step_size"])
        params_B["metric_refresh_eps_researched"] = float(eps_B)
        params_B["metric_refresh_reseed"] = reseed
        params_B["metric_refresh_T0"] = float(T0_B) if T0_B is not None else None
        # The unit-invariant quantity behind T0, before the (pi/2) factor and the clip. Kept because
        # T0 alone cannot distinguish "the pool is badly conditioned" from "the clip bit".
        params_B["metric_refresh_lmax_whitened"] = float(lmax_B) if lmax_B is not None else None
        # The third return must be the warmup HMCInfo stack for the WHOLE warmup (phase A + phase B)
        # so the exact A8 gradient count (sum of num_integration_steps) and the warmup divergence
        # count span both phases; returning phase B alone would under-count warmup gradients by W_A
        # steps. Build the combined WarmupInfo, then unwrap to the HMCInfo stack like the single-
        # phase path.
        combined = _concat_warmup_info(info_A, info_B)
        if hasattr(combined, "info"):
            self._warmup_trace = getattr(combined, "trace", None)
            combined = combined.info
        return (states_B, params_B, combined)

    def _pool_gradients(self, pos_pool, draws, kwargs):
        """Log-density gradients at the pooled phase-A positions, in kernel order, chunked.

        ``draws`` is the (N, d) kernel-order flat matrix (== positions_to_draws(pos_pool, pool)).
        The un-ravel is built from ONE position (a single chain at a single step), because a
        ravel_pytree of the whole (pool, C, ...) stack maps a full-stack vector, not one row --
        un-ravelling a single row with the stack's un-ravel produced a malformed position and a
        non-scalar log-density (caught: "Gradient only defined for scalar-output functions").
        """
        ld = self._logdensity()
        one = {n: pos_pool[n][0, 0] for n in pos_pool}
        chunk = int(kwargs.get("score_chunk", 1024))
        N = int(draws.shape[0])
        return jnp.concatenate([
            compiled.pool_gradients(ld.fn, ld.names, ld.values, one, draws[i:i + chunk])
            for i in range(0, N, chunk)], axis=0)
    
    def _run_dynamic_hmc(self, chees_states, chees_parameters, inverse_mass_matrix, **kwargs):
        """Sample from the frozen kernel: host-driven chunks of ONE jitted program (`compiled.sampling_chunk`).

        The number of draws is the host's loop bound, so a different D compiles nothing. The frozen step
        size, metric and trajectory-length adjustment are arguments of the kernel call, and the per-step
        key is ``fold_in(stage_key, draw_index)``. Returns the final state and the per-draw stacks as
        NumPy arrays, ``(draws, chains, ...)``.
        """
        # Defaults
        kwargs.setdefault("chains", 32)
        kwargs.setdefault("draws", 1000)
        kwargs.setdefault("seed", self.seed)

        rng = jax.random.fold_in(jax.random.key(kwargs["seed"]), 2)  # 2 = sampling stage, see _run_chees
        num_chains = kwargs["chains"]
        num_draws = int(kwargs["draws"])
        self._resolved["sampling"] = {"seed": int(kwargs["seed"]), "chains": int(num_chains),
                                      "num_draws": int(num_draws)}

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
            hmc_bar.set_postfix_str(f"Divergence: {div_total[0]}, Acceptance Rate: {acc_mean:.3f}")
            hmc_bar.refresh()

        def call(state, start, n):
            return compiled.sampling_chunk(
                ld.fn, ld.names, ld.values, state, imm, step_size, traj_len_adj, key_sample,
                np.int32(start), np.int32(n), chunk=compiled.SAMPLE_CHUNK, jitter_amount=jitter_amount)

        dhmc_state = compiled.run_chunks(call, chees_states, num_draws, compiled.SAMPLE_CHUNK, consume)
        if hmc_bar is not None:
            hmc_bar.close()
        states_hist, infos_hist = stack.result()
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
        # Snapshot: the two-phase refresh re-searches eps through the same method, with the CHEES
        # kwargs, and overwrites this entry. The re-search's own entry is read after warmup.
        eps0_record = dict(self._resolved["step_size"])
        chees_states, chees_parameters, chees_info = self._run_chees(
            pf_samples,
            inverse_mass_matrix,
            eps0,
            **chees_kwargs,
        )
        dhmc_state, states_hist, infos_hist = self._run_dynamic_hmc(
            chees_states,
            chees_parameters,
            inverse_mass_matrix,
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
        warm_n = _np.asarray(chees_info.num_integration_steps, dtype=_np.int64)     # (steps, chains)
        samp_n = _np.asarray(infos_hist.num_integration_steps, dtype=_np.int64)   # (draws, chains)
        idata.attrs["warmup_grad_evals_per_chain"] = int(warm_n.sum() // warm_n.shape[1])
        idata.attrs["sampling_grad_evals_per_chain"] = int(samp_n.sum() // samp_n.shape[1])
        # WP4 metric refresh: expose the swap bookkeeping in the output (and hence the row
        # schema) so a refresh is auditable -- which estimator, which warmup split it used,
        # and where the step size moved across the swap.
        for _k in ("metric_refresh", "metric_refresh_warmup", "metric_refresh_pool",
                  "metric_refresh_eps0", "metric_refresh_phaseA_end_eps",
                  "metric_refresh_eps_researched", "metric_refresh_reseed", "metric_refresh_T0",
                  "metric_refresh_lmax_whitened"):
            if _k in chees_parameters:
                idata.attrs[_k] = chees_parameters[_k]
        # R2-WP2: the resolved configuration, one JSON string (key list: see _config_record).
        idata.attrs["config"] = self._config_record(idata, chees_parameters, eps0, eps0_record)
        return idata

    def _config_record(self, idata, chees_parameters, eps0, eps0_record) -> str:
        """JSON string for ``idata.attrs["config"]``: what the sampler RESOLVED, not what was asked.

        Every value is the one the code applied, defaults filled in, so a harness can compare it with
        the flags it passed and can never read "None" for a knob that had a default. Keys:

        Pathfinder / pool / metric (``_run_pathfinder``)
            ``imm_source``, ``imm_orientation``, ``imm_order``; ``ridge`` (added to the Monte-Carlo
            covariance only, so 0.0 for every other source); ``pf_draws``; ``pf_pool_cap`` (the cap
            applied, 0 = uncapped) and ``pf_pool_rows`` (rows actually drawn); ``pf_start_scale``;
            ``init_source``,
            ``init_pick``, ``superchains`` (null = not set), ``pm_chains``, ``pm_init``, ``pm_tune``,
            ``score_chunk``.
        Step-size search (``_run_step_size_tune``)
            ``eps0_probes`` (number of probe positions) and ``eps0`` (the step size it returned) for
            the search before warmup; with a refresh, the re-search on the new metric reads its own
            ``metric_refresh_eps_probes`` from the CHEES kwargs (not the step-size kwargs).
        Warmup (``_run_chees``)
            ``num_warmup`` (configured warmup length) and ``warmup_steps_used`` (ran; differs only
            with a stop rule), ``iterate``, ``adam_lr``, ``target_accept``, ``max_leapfrog_steps``,
            ``warmup_stop`` (null = no stop rule).
        Metric refresh
            ``metric_refresh`` ("none" or the estimator name that ran) and, when one ran,
            ``metric_refresh_warmup`` (phase-A steps after clamping), ``metric_refresh_pool``,
            ``metric_refresh_reseed``, ``metric_refresh_ridge``, ``metric_refresh_shrink``,
            ``metric_refresh_rank``, ``metric_refresh_eps_probes``; all null when ``metric_refresh``
            is "none" (not applicable).
        Sampling (``_run_dynamic_hmc``)
            ``chains``, ``num_draws``.
        Frozen at the end of warmup
            ``frozen_step_size`` (eps) and ``frozen_L_nominal`` (L = T/eps).
        Seeds
            ``seed`` (the sampler's own) and ``stage_seeds`` for pathfinder / step_size / warmup /
            sampling, which differ from ``seed`` only if a stage's kwargs overrode it.
        ``jax_enable_x64`` is recorded because the metric dtype, and with it the pf_closed
        positive-definiteness check, depends on it.
        """
        r = self._resolved
        pf, ss, wu, sm = r["pathfinder"], eps0_record, r["warmup"], r["sampling"]
        refreshing = wu["metric_refresh"] != "none"
        cfg = {
            "imm_source": pf["imm_source"],
            "imm_orientation": pf["imm_orientation"],
            "imm_order": pf["imm_order"],
            "ridge": pf["ridge"],
            "pf_draws": pf["pf_draws"],
            "pf_pool_cap": pf["pf_pool_cap"],
            "pf_pool_rows": pf["pf_pool_rows"],
            "pf_start_scale": pf["pf_start_scale"],
            "init_source": pf["init_source"],
            "init_pick": pf["init_pick"],
            "superchains": pf["superchains"],
            "pm_chains": pf["pm_chains"],
            "pm_init": pf["pm_init"],
            "pm_tune": pf["pm_tune"],
            "score_chunk": pf["score_chunk"],
            "eps0_probes": ss["eps0_probes"],
            "eps0": float(eps0),
            "num_warmup": wu["num_warmup"],
            "warmup_steps_used": int(idata.attrs["warmup_steps_used"]),
            "iterate": wu["iterate"],
            "adam_lr": wu["adam_lr"],
            "target_accept": wu["target_accept"],
            "max_leapfrog_steps": wu["max_leapfrog_steps"],
            "warmup_stop": wu["warmup_stop"],
            "metric_refresh": wu["metric_refresh"],
            "metric_refresh_warmup": chees_parameters["metric_refresh_warmup"] if refreshing else None,
            "metric_refresh_pool": chees_parameters["metric_refresh_pool"] if refreshing else None,
            "metric_refresh_reseed": chees_parameters["metric_refresh_reseed"] if refreshing else None,
            "metric_refresh_ridge": wu["metric_refresh_ridge"],
            "metric_refresh_shrink": wu["metric_refresh_shrink"],
            "metric_refresh_rank": wu["metric_refresh_rank"],
            "metric_refresh_eps_probes": r["step_size"]["eps0_probes"] if refreshing else None,
            "chains": sm["chains"],
            "num_draws": sm["num_draws"],
            "frozen_step_size": idata.attrs["frozen_step_size"],
            "frozen_L_nominal": idata.attrs["frozen_L_nominal"],
            "seed": int(self.seed),
            "stage_seeds": {"pathfinder": pf["seed"], "step_size": ss["seed"],
                            "warmup": wu["seed"], "sampling": sm["seed"]},
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        }
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
        Return the posterior in unconstrained (value-var) space as well. Needed by the R1 bias
        harness (WP3), which compares moments against a NUTS reference in unconstrained space.
    pathfinder_kwargs, step_size_tune_kwargs, chees_kwargs, dynamic_hmc_kwargs : dict | None
        Forwarded to `PfCheesHMC.sample(...)`.

    Returns
    -------
    arviz.InferenceData
    """
    m = modelcontext(model)  # supports both inside/outside context usage

    sampler = PfCheesHMC(m, seed=random_seed)

    # normalize dicts
    pathfinder_kwargs = {} if pathfinder_kwargs is None else dict(pathfinder_kwargs)
    step_size_tune_kwargs = {} if step_size_tune_kwargs is None else dict(step_size_tune_kwargs)
    chees_kwargs = {} if chees_kwargs is None else dict(chees_kwargs)
    dynamic_hmc_kwargs = {} if dynamic_hmc_kwargs is None else dict(dynamic_hmc_kwargs)

    # One flag controls both bars. Off means no jax.debug.callback in either scan, which removes the
    # per-step host round-trip and makes both compiled scans eligible for JAX's on-disk compilation
    # cache. A per-stage value in chees_kwargs / dynamic_hmc_kwargs still wins.
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