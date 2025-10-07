import jax, jax.numpy as jnp
import blackjax as bj
from jax.flatten_util import ravel_pytree
from jax import lax
import optax

import pymc as pm
from pymc.sampling.jax import get_jaxified_graph
from pymc import modelcontext
from pymc.util import get_default_varnames
from pymc.sampling.jax import get_jaxified_logp
import arviz as az

from tqdm.auto import tqdm
from .utils import *

from typing import Tuple, Optional, Dict, Any
from datetime import date
import io
import re
import logging
import contextlib

__all__ = ["sample_pf_chees_hmc"]

class PfCheesHMC:
    def __init__(self, model:pm.Model, seed:Optional[int]=None):
        self.model = model

        self.rvs = [rv.name for rv in model.value_vars]
        self._ld_fn = get_jaxified_logp(model)
        self._seed = seed

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
        Make a bad (possibly -inf) start finite without changing the model:
        1) tiny relative jitter (fix exact singularities like xi=0),
        2) backtrack toward `center` (default zeros in unconstrained space),
        3) random restarts around center with growing radius.
        Returns a pytree with finite logp or raises RuntimeError.
        """
        from jax.flatten_util import ravel_pytree

        def _isfinite_logp(pos):
            return jnp.isfinite(self.ld_fn(pos))

        def _interp(a, b, t):
            return jax.tree_util.tree_map(lambda ai, bi: bi + t * (ai - bi), a, b)

        flat0, unrav = ravel_pytree(x0)

        # (1) tiny relative jitter
        key, k0 = jax.random.split(key)
        rel = 1.0 + jnp.abs(flat0)
        jitter = jax.random.normal(k0, flat0.shape) * (jitter_eps * rel)
        x = unrav(flat0 + jitter)
        if bool(_isfinite_logp(x)):
            return x

        # (2) backtrack toward safe center (default zeros)
        if center is None:
            center = unrav(jnp.zeros_like(flat0))
        for _ in range(max_backtracks):
            x_try = _interp(x, center, 0.5)
            if bool(_isfinite_logp(x_try)):
                return x_try
            x = x_try

        # (3) random restarts around center
        radius = restart_radius
        for _ in range(max_restarts):
            key, k = jax.random.split(key)
            noise = jax.random.normal(k, flat0.shape) * radius
            x_try = unrav(ravel_pytree(center)[0] + noise)
            if bool(_isfinite_logp(x_try)):
                return x_try
            radius *= 1.5

        raise RuntimeError("Failed to locate a finite-logp start for Pathfinder.")

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
        For a dict{name: (chains, ...)} ensure every chain's point is finite.
        Applies _find_finite_start per-chain if needed.
        Returns a dict{name: (chains, ...)} with all finite logp.
        """
        chains = next(iter(init_stack.values())).shape[0]
        keys = jax.random.split(key, chains)
        fixed = {n: [] for n in self.rvs}

        for c in range(chains):
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
        ridge: float covariance ridge (default 1e-6)
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

        # finite-start knobs
        pf_jitter_eps     = kwargs.get("pf_jitter_eps", 1e-8)
        pf_max_backtracks = kwargs.get("pf_max_backtracks", 16)
        pf_max_restarts   = kwargs.get("pf_max_restarts", 64)
        pf_center         = kwargs.get("pf_center", None)
        pf_restart_radius = kwargs.get("pf_restart_radius", 0.05)

        key = jax.random.key(seed)
        key_fix, key_pf, key_samp, key_pick, key_choose = jax.random.split(key, 5)

        # 0) Ask PyMC for feasible-ish starts
        init_stack = self._pymc_adapt_diag_start(
            chains, init=pm_init, tune=pm_tune, random_seed=pm_seed
        )

        # 1) Ensure every chain's init is finite (handles xi=0 -> -inf cases)
        init_stack = self._ensure_finite_inits(
            init_stack, key_fix,
            jitter_eps=pf_jitter_eps,
            max_backtracks=pf_max_backtracks,
            max_restarts=pf_max_restarts,
            center=pf_center,
            restart_radius=pf_restart_radius,
        )

        # 2) Choose a single finite seed for PF optimizer: the chain with highest logp
        def _score_chain(c):
            x = {n: init_stack[n][c] for n in self.rvs}
            return float(self.ld_fn(x))
        scores = [(_score_chain(c), c) for c in range(chains)]
        scores = [(s if jnp.isfinite(jnp.asarray(s)) else -jnp.inf, c) for s, c in scores]
        c_best = max(scores, key=lambda t: t[0])[1]
        x0_pf = {n: init_stack[n][c_best] for n in self.rvs}

        # 3) Fit Pathfinder
        pf_state, _ = bj.vi.pathfinder.approximate(
            rng_key=key_pf,
            initial_position=tuple(x0_pf[n] for n in self.rvs),
            logdensity_fn=self.ld_fn,
        )

        # 4) Draw ONE batch (chains*draws) and reshape to (chains, draws, ...)
        total = chains * draws
        samples_flat, _ = bj.vi.pathfinder.sample(key_samp, pf_state, total)
        samples_cd = jax.tree_util.tree_map(
            lambda x: x.reshape((chains, draws) + x.shape[1:]),
            samples_flat
        )

        # 5) Choose *chain* initial positions
        if init_source == "pf":
            # one PF draw per chain (random index per chain)
            pick_idx = jax.random.randint(key_pick, shape=(chains,), minval=0, maxval=draws)
            def _pick_one(x):  # x: (chains, draws, ...)
                return x[jnp.arange(chains), pick_idx, ...]
            pf_inits_tree = jax.tree_util.tree_map(_pick_one, samples_cd)  # (chains, ...)
        elif init_source == "adapt_diag":
            # use the repaired adapt_diag starts as chain inits
            pf_inits_tree = init_stack
        else:
            raise ValueError("init_source must be 'pf' or 'adapt_diag'")

        # 6) Dense covariance from ALL PF samples -> inverse mass
        samples_total = jax.tree_util.tree_map(
            lambda x: x.reshape((total,) + x.shape[2:]),
            samples_cd
        )
        def _flatten_i(i):
            sliced = jax.tree_util.tree_map(lambda x: x[i], samples_total)
            flat, _ = ravel_pytree(sliced)
            return flat
        flat_mat = jax.vmap(_flatten_i)(jnp.arange(total))  # (total, D)

        X   = flat_mat - flat_mat.mean(axis=0, keepdims=True)
        cov = (X.T @ X) / (total - 1) + ridge * jnp.eye(X.shape[1], dtype=X.dtype)
        inverse_mass_matrix = jnp.linalg.inv(cov)

        # 7) Return dict-of-arrays keyed by variable (leading axis = chains)
        if isinstance(pf_inits_tree, (list, tuple)):
            pf_samples = {name: arr for name, arr in zip(self.rvs, pf_inits_tree)}
        elif isinstance(pf_inits_tree, dict):
            pf_samples = {name: pf_inits_tree[name] for name in self.rvs}
        else:
            pf_samples = {self.rvs[0]: pf_inits_tree}

        return pf_samples, inverse_mass_matrix



    def _run_step_size_tune(self, pf_samples, inverse_mass_matrix, **kwargs):
        #Default values
        kwargs.setdefault("chains", 32)
        kwargs.setdefault("seed", self.seed)
        ld_fn = self.ld_fn
        num_chains = kwargs["chains"]
        rng = jax.random.key(kwargs["seed"])
        _, num_rng = jax.random.split(rng)
        keys_eps = jax.random.split(num_rng, 2 * num_chains)  # 1 for search, 1 for test-step

        # ==== Per-chain "reasonable" step size, then take the median as ε0 ====
        def eps_reasonable_for_pos(key, pos_dict, inv_mass):
            hmc = bj.hmc(ld_fn, step_size=1e-2, inverse_mass_matrix=inv_mass, num_integration_steps=1)
            state0 = hmc.init(pos_dict)
            return bj.adaptation.step_size.find_reasonable_step_size(
                key,
                lambda eps: bj.hmc(ld_fn, eps, inv_mass, num_integration_steps=1).step,
                state0,
                initial_step_size=1e-2,
                target_accept=0.651,
            )

        eps_list = []
        bar = tqdm(total=num_chains, desc="ε Tuning", leave=True)
        for i in range(num_chains):
            pos_i = {k: v[i] for k, v in pf_samples.items()}

            # run the built-in search (no internal hooks available)
            eps_i = eps_reasonable_for_pos(keys_eps[i], pos_i, inverse_mass_matrix)
            eps_list.append(eps_i)

            # optional: sanity-check acceptance at the found ε (1 leapfrog step)
            hmc_i = bj.hmc(ld_fn, float(eps_i), inverse_mass_matrix, num_integration_steps=1)
            state0_i = hmc_i.init(pos_i)
            _, info_i = hmc_i.step(keys_eps[num_chains + i], state0_i)
            acc_i = float(info_i.acceptance_rate)

            bar.update(1)
            bar.set_postfix_str(f"ε: {float(eps_i):.4f}, Acceptance rate: {acc_i:.2f}")
            
        eps0 = float(jnp.median(jnp.asarray(eps_list)))

        return eps0

    def _run_chees(self, pf_samples, inverse_mass_matrix, init_step_size, **kwargs):
        #Default values
        kwargs.setdefault("chains", 32)
        kwargs.setdefault("draws", 2000)
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("adam_lr", 0.05)
        kwargs.setdefault("target_accept", bj.adaptation.chees_adaptation.OPTIMAL_TARGET_ACCEPTANCE_RATE)

        optim = optax.adam(learning_rate=kwargs["adam_lr"])
        rng = jax.random.key(kwargs["seed"])
        _, key_warm = jax.random.split(rng)
        (chees_states, chees_parameters), chees_info = chees_warmup_with_progress(
            self.ld_fn,
            key_warm,
            pf_samples,            # dict: {"mu": (C,), "tau_log__": (C,), "theta": (C, J)}
            init_step_size,
            optim,
            kwargs["draws"],
            inverse_mass_matrix,   # your PF-based diag/dense (flattened if diag)
            target_acceptance_rate=kwargs['target_accept'],
        )

        return (
            chees_states,
            chees_parameters,
            chees_info,
        )
    
    def _run_dynamic_hmc(self, chees_states, chees_parameters, inverse_mass_matrix, **kwargs):
        # Defaults
        kwargs.setdefault("chains", 32)
        kwargs.setdefault("draws", 1000)
        kwargs.setdefault("seed", self.seed)

        rng = jax.random.key(kwargs["seed"])
        num_chains = kwargs["chains"]
        num_draws = kwargs["draws"]

        dyn = bj.mcmc.dynamic_hmc.as_top_level_api(
            self.ld_fn,
            step_size=chees_parameters["step_size"],
            inverse_mass_matrix=inverse_mass_matrix,
            next_random_arg_fn=chees_parameters["next_random_arg_fn"],
            integration_steps_fn=chees_parameters["integration_steps_fn"],
        )
        kernel_step = dyn.step

        # vmapped step over chains
        def _vmapped_step(state, keys):
            return jax.vmap(kernel_step)(keys, state)

        # ---------- discover real paths to fields (once, outside jit) ----------
        probe_keys = jax.random.split(jax.random.key(0), num_chains)
        _, info_probe = _vmapped_step(chees_states, probe_keys)  # one real step (we discard state)

        def _find_path(obj, want_bool=False, substr=""):
            """Find a tuple of attribute names whose leaf matches dtype/name criteria."""
            # namedtuple / dataclass-like
            if hasattr(obj, "_fields"):
                for name in obj._fields:
                    child = getattr(obj, name)
                    path = _find_path(child, want_bool, substr)
                    if path is not None:
                        return (name,) + path
                    # direct leaf
                    if hasattr(child, "dtype") and substr in name.lower():
                        if want_bool and child.dtype == jnp.bool_:
                            return (name,)
                        if (not want_bool) and jnp.issubdtype(child.dtype, jnp.floating):
                            return (name,)
            # mapping
            if isinstance(obj, dict):
                for name, child in obj.items():
                    path = _find_path(child, want_bool, substr)
                    if path is not None:
                        return (name,) + path
                    if hasattr(child, "dtype") and substr in str(name).lower():
                        if want_bool and child.dtype == jnp.bool_:
                            return (name,)
                        if (not want_bool) and jnp.issubdtype(child.dtype, jnp.floating):
                            return (name,)
            # list/tuple (use indices as names)
            if isinstance(obj, (list, tuple)):
                for i, child in enumerate(obj):
                    path = _find_path(child, want_bool, substr)
                    if path is not None:
                        return (i,) + path
                    if hasattr(child, "dtype") and substr in "":  # lists rarely carry names; skip
                        pass
            return None

        path_div = _find_path(info_probe, want_bool=True, substr="diverg")
        path_acc = _find_path(info_probe, want_bool=False, substr="accept")

        def _getter(path):
            if path is None:
                return None
            def g(x):
                for p in path:
                    x = x[p] if isinstance(p, int) else getattr(x, p)
                return x
            return g

        get_div = _getter(path_div)   # -> (chains,) bool or (chains, ... bool ...)
        get_acc = _getter(path_acc)   # -> (chains,) float or (chains, ...)

        # ---------- progress bar and host aggregator ----------
        update_every = 25
        hmc_bar = tqdm(total=num_draws, desc="Sampling Dynamic HMC", position=0, leave=True)
        _div_total = {"n": 0}

        def _host_update(i, window_div_sum, acc_mean):
            _div_total["n"] += int(window_div_sum)
            hmc_bar.n = int(i) + 1
            if jnp.isnan(acc_mean):
                hmc_bar.set_postfix_str(f"Divergence: {_div_total['n']}")
            else:
                hmc_bar.set_postfix_str(f"Divergence: {_div_total['n']}, Acceptance Rate: {float(acc_mean):.3f}")
            hmc_bar.refresh()

        vmapped_step_jit = jax.jit(_vmapped_step)

        def _one_step(carry, xs):
            state, window_div = carry                      # window_div has a fixed dtype
            i, key = xs
            keys = jax.random.split(key, num_chains)
            new_state, info = vmapped_step_jit(state, keys)

            # --- divergence per step ---
            if get_div is not None:
                div_leaf = get_div(info)
                if div_leaf.ndim > 1:
                    div_vec = jnp.any(div_leaf, axis=tuple(range(1, div_leaf.ndim)))
                else:
                    div_vec = div_leaf
                div_step = jnp.sum(div_vec.astype(jnp.int32))       # base int32
            else:
                div_step = jnp.asarray(0, dtype=jnp.int32)

            # *** critical: match carry dtype ***
            div_step = div_step.astype(window_div.dtype)
            window_div = window_div + div_step

            # --- acceptance (optional) ---
            if get_acc is not None:
                acc_leaf = get_acc(info)
                if acc_leaf.ndim > 1:
                    acc_mean = jnp.mean(acc_leaf, axis=tuple(range(1, acc_leaf.ndim)))
                    acc_mean = jnp.mean(acc_mean)
                else:
                    acc_mean = jnp.mean(acc_leaf)
            else:
                acc_mean = jnp.asarray(jnp.nan, dtype=jnp.float32)

            should_cb = jnp.logical_or(((i + 1) % update_every) == 0,
                                    (i + 1) == num_draws)

            def do_cb(_):
                jax.debug.callback(_host_update, i, window_div, acc_mean, ordered=True)
                # *** reset to 0 with same dtype as carry ***
                return jnp.zeros((), dtype=window_div.dtype)

            def dont_cb(_):
                # *** ensure dtype stays consistent in false branch too ***
                return window_div.astype(window_div.dtype)

            window_div_next = lax.cond(should_cb, do_cb, dont_cb, operand=None)
            return (new_state, window_div_next), (new_state, info)

        rng, key_sample = jax.random.split(rng)
        step_keys = jax.random.split(key_sample, num_draws)

        one_step_jit = jax.jit(_one_step)
        (dhmc_state, _), (states_hist, infos_hist) = jax.lax.scan(
            one_step_jit,
            # *** choose a single dtype for the carry counter up front ***
            (chees_states, jnp.zeros((), dtype=jnp.int32)),
            (jnp.arange(num_draws), step_keys),
        )
        return dhmc_state, states_hist, infos_hist



    def _compile_value_to_rv_fn(self, *, keep_untransformed: bool = False, var_names=None):
        model = modelcontext(self.model)
        if var_names is None:
            var_names = model.unobserved_value_vars
        vars_to_sample = list(
            get_default_varnames(var_names, include_transformed=keep_untransformed)
        )
        jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
        return jax_fn, vars_to_sample

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
        # Compile the value->problem-space transformer
        jax_fn, vars_to_sample = self._compile_value_to_rv_fn(
            keep_untransformed=keep_untransformed
        )

        # Assemble raw_mcmc_samples just like PyMC does (list aligned to value_vars)
        raw_mcmc_samples = self._collect_raw_mcmc_samples(states_hist)

        # Mirror PyMC’s postprocessing: nested vmap over (chains, draws)
        # (optionally you could reproduce their device placement/chunking too)
        if postprocessing_chunks is None:
            result = jax.vmap(jax.vmap(jax_fn))(*raw_mcmc_samples)
        else:
            from jax.experimental.maps import SerialLoop, xmap
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
            # so leaves are typically (draws, chains, ...). Swap the first two axes.
            return jnp.swapaxes(a, 0, 1) if a.ndim >= 2 else a

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
        • sample_stats auto-populated from infos_hist (no hard-coded field list).

        Assumes states_hist leaves are shaped (draws, chains, ...) by default;
        override draw_axis/chain_axis if you stacked differently.
        """
        import dataclasses
        import numpy as np
        import arviz as az
        import jax.numpy as jnp
        from collections.abc import Mapping

        # ---- helper: move (draw_axis, chain_axis) -> (chain, draw)
        def _swap_cd(x):
            na = x.ndim
            da = draw_axis % na
            ca = chain_axis % na
            if da == ca:
                raise ValueError("draw_axis and chain_axis must be different.")
            rest = [ax for ax in range(na) if ax not in (ca, da)]
            perm = [ca, da] + rest
            return np.asarray(jnp.transpose(x, perm))

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

        # ---- 2) Sample stats (auto-discovered)
        def _extract_arrays(obj, prefix="", out=None):
            """Recursively collect {name: array-like} from infos_hist."""
            if out is None:
                out = {}
            # dict / Mapping
            if isinstance(obj, Mapping):
                for k, v in obj.items():
                    _extract_arrays(v, f"{prefix}{k}.", out)
                return out
            # namedtuple / _asdict
            if hasattr(obj, "_asdict"):
                for k, v in obj._asdict().items():
                    _extract_arrays(v, f"{prefix}{k}.", out)
                return out
            # dataclass
            if dataclasses.is_dataclass(obj):
                for f in dataclasses.fields(obj):
                    _extract_arrays(getattr(obj, f.name), f"{prefix}{f.name}.", out)
                return out
            # tuple/list -> index
            if isinstance(obj, (tuple, list)):
                for i, v in enumerate(obj):
                    _extract_arrays(v, f"{prefix}{i}.", out)
                return out
            # base case: arrays
            if hasattr(obj, "shape"):
                name = prefix[:-1] if prefix.endswith(".") else prefix
                out[name or "value"] = obj
            return out

        sample_stats = None
        if infos_hist is not None:
            raw = _extract_arrays(infos_hist)

            # Reorder each to (chains, draws, ...) and cast dtypes/names
            sample_stats = {}
            # try to detect (chains, draws) from posterior to validate
            # (use the first posterior variable)
            _any_post = next(iter(posterior_np.values()))
            n_chains, n_draws = _any_post.shape[0], _any_post.shape[1]

            rename = {"logdensity": "lp", "is_divergent": "diverging"}

            for key, arr in raw.items():
                try:
                    arr_cd = _swap_cd(arr)  # (chains, draws, ...)
                except Exception:
                    # if swap fails (e.g., scalar or 1D), just convert as-is
                    arr_cd = np.asarray(arr)

                # sanity: ensure first two dims look like (chains, draws) when possible
                if arr_cd.ndim >= 2:
                    ok = (arr_cd.shape[0] == n_chains and arr_cd.shape[1] == n_draws)
                    # tolerate the opposite orientation and fix it
                    if not ok and arr_cd.shape[:2] == (n_draws, n_chains):
                        arr_cd = np.swapaxes(arr_cd, 0, 1)

                out_key = rename.get(key, key.split(".")[-1])  # strip prefix if nested
                if "diverg" in out_key and arr_cd.dtype != bool:
                    arr_cd = arr_cd.astype(bool)
                sample_stats[out_key] = arr_cd

            # nice alias if only 'diverging' found
            if "diverging" in sample_stats and "is_divergent" not in sample_stats:
                sample_stats["is_divergent"] = sample_stats["diverging"]

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
            inverse_mass_matrix,
            **dynamic_hmc_kwargs,
        )

        posterior = self._transform_value_samples_to_problem_space(
            states_hist,
            keep_untransformed=False,            # like sample_blackjax_nuts default
        )

        idata = self._to_inference_data(
            states_hist,
            infos_hist,
            draw_axis=0,          # your scan stacks draws first
            chain_axis=1,         # and chains second (vmapped)
            keep_untransformed=False,
        )
        return idata
    


def sample_pf_chees_hmc(
    model: Optional[pm.Model] = None,
    *,
    draws: int = 1000,
    chains: int = 4,
    random_seed: Optional[int] = None,
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
            idata = sample_pf_chees_hmc(draws=2000, chains=4)

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

    idata = sampler.sample(
        chains=chains,
        draws=draws,
        pathfinder_kwargs=pathfinder_kwargs,
        step_size_tune_kwargs=step_size_tune_kwargs,
        chees_kwargs=chees_kwargs,
        dynamic_hmc_kwargs=dynamic_hmc_kwargs,
    )
    return idata