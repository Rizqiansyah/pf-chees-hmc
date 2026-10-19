from functools import partial
import jax, jax.numpy as jnp
import time
from typing import Any, Callable, Dict, NamedTuple, Optional

import numpy as np
import arviz as az
import blackjax as bj
from jax import lax
from tqdm.auto import tqdm
import blackjax.mcmc.dynamic_hmc as dynamic_hmc

from . import chees as _chees_vendored


class WarmupInfo(NamedTuple):
    """What the warmup scan stacks, plus the per-step scalar trace.

    info
        The HMCInfo stack (draws, chains, ...) -- same as before the vendored
        adapter existed.
    trace
        A (steps, 4) float stack: (step, step_size, trajectory_length,
        harmonic_acceptance) at each warmup step, from the vendored adapter's
        ChEESTrace. trace.trajectory_length / trace.step_size is the nominal
        leapfrog count L at each step; that is the curve the trajectory-growth
        reproduction reads. harmonic_acceptance is the quantity ChEES adapts
        the step size on (harmonic mean over non-divergent chains), not the
        arithmetic mean the progress bar used to show.
    """

    info: Any
    trace: Any


__all__ = ["chees_warmup_with_progress", "chees_harmonic_acceptance", "block_max_rhat", "WarmupInfo"]


def chees_harmonic_acceptance(acc_vals, is_divergent):
    """The acceptance ChEES actually adapts on: harmonic mean over the non-divergent chains.

    Mirrors `blackjax.adaptation.chees_adaptation` (harmonic_mean, :358-363), including the
    non-finite -> 0 guard. The arithmetic mean of per-trajectory acceptances is systematically
    higher than this (Jensen), so displaying it hides exactly the heterogeneity the step-size
    adaptation reacts to -- it already caused a wrong diagnosis in this project.
    """
    hm = 1.0 / jnp.mean(1.0 / acc_vals, where=~is_divergent)
    return jnp.where(jnp.isfinite(hm), hm, 0.0)


def block_max_rhat(pos_window):
    """Max rank-normalised split-R-hat over a warmup window.

    `pos_window` is a pytree of (window, chains, ...) arrays -- what the warmup scan emits. All
    chains, draws and parameters go through ONE call to blackjax.diagnostics.rhat, which accepts the
    full (chains, draws, dim) array with chain_axis/sample_axis named and agrees with arviz to
    float32 noise (pinned in tests/test_warmup_stop.py).

    Calling it per parameter cost ~8 s of XLA compiles per warmup block: a stopped 8 Schools run
    spent 19.17 s in warmup for 500 steps, against 4.85 s for 2,000 steps unblocked.

    Reimplementing it was tried and is wrong. A hand-written "rank, normal-score, pool-standardise,
    split" read 1.000 where arviz read 1.059-1.086 on chains of differing width, because the library
    standardises ranks per chain rather than pooled -- so do not replace this with 15 lines of jnp.
    """
    arrs = []
    for leaf in jax.tree_util.tree_leaves(pos_window):
        a = jnp.asarray(leaf)
        if a.ndim < 2:
            raise ValueError("warmup window leaves must be (window, chains, ...)")
        a = jnp.swapaxes(a, 0, 1)                       # -> (chains, window, ...)
        arrs.append(a if a.ndim == 3 else a.reshape(a.shape[0], a.shape[1], -1))
    ref = arrs[0].shape[:2]
    for a in arrs:
        if a.shape[:2] != ref:
            raise ValueError(f"warmup window shape mismatch: {a.shape[:2]} vs {ref}")
    stacked = arrs[0] if len(arrs) == 1 else jnp.concatenate(arrs, axis=-1)
    return float(jnp.max(bj.diagnostics.rhat(stacked, chain_axis=0, sample_axis=1)))


def chees_warmup_with_progress(
    ld_fn,                      # logdensity function
    rng_key,
    positions,                 # dict {"mu": (C,), "tau_log__": (C,), "theta": (C,J)}
    initial_step_size,
    optim,
    num_steps,
    inverse_mass_matrix,
    *,
    target_acceptance_rate=0.651,
    decay_rate=0.5,
    jitter_amount=1.0,
    max_sampling_steps=1000,
    max_leapfrog_steps=1000,
    update_every=25,           # <-- throttle UI updates to keep overhead tiny
    progressbar=True,          # False removes jax.debug.callback from the compiled scan
    stop_rule=None,            # None = fixed num_steps, identical to the previous behaviour.
                             #     Measured net negative on all four ladder models (block
                             #     boundaries cost 6.7-9.0 s, more than any warmup saving);
                             #     kept opt-in for instrumentation, see docs/ladder-results.md
    iterate="ma",              # "ma" = installed BlackJAX behaviour (default); "raw" =
                             #     paper-faithful Adam on the raw log-T iterate, moving
                             #     average used only at the freeze. pfcheeshmc/chees.py.
    t0=None,                   # initial trajectory length in time units (paper: pi/2);
                             #     None keeps the installed initialisation (step_size)
    single_step_iters=0,       # run L=1 for this many steps while the step size settles
):
    # --- metric shape guard -------------------------------------------------
    # The metric is a flat kernel-order quantity: its dimension must equal the
    # number of unconstrained coordinates the chains carry. Ordering is another
    # task's concern (WP1); this only checks the SIZE, so a mismatched metric
    # raises here instead of broadcasting silently inside the whitened criterion.
    d_flat = sum(a.size // a.shape[0] for a in positions.values())
    imm_arr = jnp.asarray(inverse_mass_matrix)
    if imm_arr.ndim == 2:
        if tuple(imm_arr.shape) != (d_flat, d_flat):
            raise ValueError(
                f"inverse_mass_matrix is {imm_arr.shape} but the start positions "
                f"span {d_flat} flat coordinates; expected ({d_flat}, {d_flat})"
            )
    elif imm_arr.ndim == 1:
        if tuple(imm_arr.shape) != (d_flat,):
            raise ValueError(
                f"inverse_mass_matrix is {imm_arr.shape} but the start positions "
                f"span {d_flat} flat coordinates; expected ({d_flat},)"
            )
    else:
        raise ValueError(f"inverse_mass_matrix must be 1-D or 2-D, got ndim={imm_arr.ndim}")

    # --- Halton jitter (same idea as BJ’s ChEES) ---
    def halton_u(i):
        return dynamic_hmc.halton_sequence(
            i, int(np.ceil(np.log2(num_steps + max_sampling_steps)))
        )
    def jitter_gn(i):
        return halton_u(i) * jitter_amount + (1.0 - jitter_amount)

    # RNG arg increments each step
    next_random_arg_fn = lambda i: i + 1
    init_random_arg = 0

    def integration_steps_fn(random_generator_arg, trajectory_length_adjusted):
        # ceil(jitter * L/eps); must accept kw name 'trajectory_length_adjusted'
        return jnp.asarray(
            jnp.ceil(jitter_gn(random_generator_arg) * trajectory_length_adjusted),
            dtype=int,
        )

    # Dynamic-HMC step builder (same as BJ uses internally)
    step_fn = dynamic_hmc.build_kernel(
        next_random_arg_fn=next_random_arg_fn,
        integration_steps_fn=integration_steps_fn,
    )

    # Low-level ChEES adapter (init, update). Vendored in pfcheeshmc/chees.py:
    # iterate="ma" (default) is bit-for-bit the installed BlackJAX base();
    # update additionally returns a scalar per-step trace (see WarmupInfo).
    init_adapt, update_adapt = _chees_vendored.base(
        jitter_gn, next_random_arg_fn, optim, target_acceptance_rate, decay_rate, max_leapfrog_steps,
        iterate=iterate, t0=t0, single_step_iters=single_step_iters,
    )

    # Batch-initialize states for all chains
    num_chains_local = next(iter(positions.values())).shape[0]
    batch_init = jax.jit(jax.vmap(lambda p: dynamic_hmc.init(p, ld_fn, init_random_arg)))
    init_states = batch_init(positions)
    adaptation_state = init_adapt(init_random_arg, initial_step_size)

    # ChEES whitens its trajectory-length criterion by 1/sqrt(imm), so it needs
    # the diagonal of the inverse mass matrix (variances). The HMC kernel itself
    # still gets the full dense Pathfinder metric.
    imm_diag = (
        jnp.diag(inverse_mass_matrix)
        if jnp.ndim(inverse_mass_matrix) == 2
        else inverse_mass_matrix
    )

    # --- one progress bar, like your dynamic-HMC bar ------------------------
    # With progressbar=False no host callback enters the jaxpr at all. That is not cosmetic:
    # a scan containing jax.debug.callback cannot be written to JAX's on-disk compilation cache,
    # so leaving the bar on costs every fresh process a full recompile of the warmup scan.
    bar = tqdm(total=num_steps, desc="ChEES Adaptation", position=0, leave=True) if progressbar else None
    _div_total = {"n": 0}  # host-side mutable counter

    def _host_update(i, div_step, acc_mean, step_size_now):
        _div_total["n"] += int(div_step)
        bar.n = int(i) + 1
        bar.set_postfix_str(
            f"Divergence: {_div_total['n']}, Acceptance rate: {float(acc_mean):.2f}, ε: {float(step_size_now):.3g}"
        )
        bar.refresh()

    # One ChEES step (vectorized over chains) with throttled host callback
    def one_step(carry, xs):
        i, key = xs
        states, astate = carry
        keys = jax.random.split(key, num_chains_local)

        _step = partial(
            step_fn,
            logdensity_fn=ld_fn,
            step_size=astate.step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            # BlackJAX >=1.7: the trajectory-length adjustment is unpacked into
            # integration_steps_fn at call time, not passed as a named kwarg.
            integration_steps_params=(astate.trajectory_length / astate.step_size,),
        )
        new_states, info = jax.vmap(_step)(keys, states)

        # accept field name can be acceptance_probability or acceptance_rate depending on BJ version
        acc_vals = getattr(info, "acceptance_probability", None)
        if acc_vals is None:
            acc_vals = info.acceptance_rate

        # diagnostics for this iteration (across chains)
        div_step  = jnp.sum(info.is_divergent)
        acc_mean  = chees_harmonic_acceptance(acc_vals, info.is_divergent)

        # Update adaptation state (vendored adapter: state + scalar trace)
        new_astate, trace = update_adapt(
            astate,
            info.proposal.position,
            info.proposal.momentum,
            states.position,
            acc_vals,              # accepts vector of acceptances
            info.is_divergent,
            imm_diag,              # 1.7 requires the diagonal metric here
        )

        # JAX-safe throttling of host callback. Removed at trace time when the bar is off, so the
        # compiled scan is cacheable and carries no callback overhead.
        if progressbar:
            should_cb = jnp.logical_or(((i + 1) % update_every) == 0,
                                       (i + 1) == num_steps)

            def do_cb(_):
                jax.debug.callback(
                    _host_update, i, div_step, acc_mean, new_astate.step_size, ordered=True
                )
                return ()
            def dont_cb(_):  # no-op
                return ()

            lax.cond(should_cb, do_cb, dont_cb, operand=None)

        return (new_states, new_astate), (new_states.position, info, trace)   # carry, ys

    one_step_jit = jax.jit(one_step)

    def _scan_block(carry, start, n):
        """Run n warmup steps whose *global* step indices start at `start`.

        The Halton trajectory jitter and the progress bar are functions of the step index, so a
        resumed block has to be told where it sits in the run. Restarting the index at 0 would make
        every block replay the same jitter sequence.
        """
        ks = jax.random.split(jax.random.fold_in(rng_key, start), n)
        return jax.lax.scan(one_step_jit, carry, (jnp.arange(start, start + n), ks))

    def _stop_met(pos_win, info_win, astate):
        """Rank-normalised split-R-hat over a window, plus step-size quietness.

        R-hat is the load-bearing part. Adapter stability on its own would have declared the radon
        centered funnel converged while its trajectory was frozen at 4.4 leapfrog steps -- the adapter
        was perfectly happy and the sampler was not exploring.

        The harmonic-mean acceptance is deliberately NOT part of the default gate. Measured on a
        unit Normal (8 chains, 250-step window) it reads 0.0000-0.0126 for a well-tuned sampler:
        harmonic mean is dominated by the worst chain, and a single fully-rejected proposal -- which
        happens routinely -- drives it to zero. It is a per-step adaptation signal, not a windowed
        stability statistic. Supply accept_band=(lo, hi) to opt back in.

        Window length matters: with 50 draws per chain split-R-hat sits at 1.035-1.058 forever on
        that same trivial target; at 250 draws it reads 1.0031-1.0099. Short-window R-hat is a
        property of the estimator, not of the sampler.
        """
        rhat_limit = float(stop_rule.get("rhat", 1.01))   # the project's G-A convergence bar
        eps_limit = float(stop_rule.get("eps_stability", 0.5))
        band = stop_rule.get("accept_band")   # opt-in only; see the note below

        rhat = block_max_rhat(pos_win)

        acc = jnp.asarray(info_win.acceptance_rate)[-window:]
        div = jnp.asarray(info_win.is_divergent)[-window:]
        hm = float(chees_harmonic_acceptance(acc, div))
        eps_gap = float(jnp.abs(astate.log_step_size_moving_average - jnp.log(astate.step_size)))
        met = (rhat < rhat_limit) and (eps_gap < eps_limit)
        if band is not None:
            met = met and (band[0] <= hm <= band[1])
        if stop_rule.get("debug"):
            print(f"    stop_check step={done}: rhat={rhat:.4f} (limit {rhat_limit}) "
                  f"accept_hm={hm:.4f} eps_gap={eps_gap:.4f} (limit {eps_limit}) "
                  f"-> {'STOP' if met else 'continue'}", flush=True)
        return met, {"rhat": rhat, "accept": hm, "eps_gap": eps_gap}

    if stop_rule:
        block = int(stop_rule.get("block", 250))
        # Default window = the whole block. A 50-draw window made split-R-hat read 1.04 on a unit
        # Normal no matter how long warmup ran; 250 draws per chain reads 1.003-1.010.
        window = int(stop_rule.get("window", block))
        carry = (init_states, adaptation_state)
        done = 0
        stopped_at = None
        while done < num_steps:
            n = min(block, num_steps - done)
            _t = time.perf_counter()
            carry, (pos_blk, info_blk, trace_blk) = _scan_block(carry, done, n)
            _t_scan = time.perf_counter() - _t
            done += n
            if bar is not None:
                bar.n = done
                bar.refresh()
            if done < num_steps and _stop_met(
                    jax.tree_util.tree_map(lambda p: p[-window:], pos_blk), info_blk, carry[1])[0]:
                stopped_at = done
            if stop_rule.get("debug"):
                print(f"    block {done:5d}: scan {_t_scan:6.2f} s", flush=True)
            if stopped_at is not None:
                break
        (last_states, last_adapt_state), (pos_hist, warm_info, trace_hist) = carry, (pos_blk, info_blk, trace_blk)
    else:
        # Default path: one scan, same key stream as before the stop rule existed.
        keys = jax.random.split(rng_key, num_steps)
        (last_states, last_adapt_state), (pos_hist, warm_info, trace_hist) = jax.lax.scan(
            one_step_jit, (init_states, adaptation_state), (jnp.arange(num_steps), keys))
        stopped_at = None

    if bar is not None:
        bar.close()

    # Assemble parameters like BJ’s run(). Explicit projection: the frozen values a
    # reader or row schema can rely on are named keys, not positions in a dict.
    # The metric is passed through UNCHANGED -- its orientation and ordering are
    # another task's decision (WP1); nothing here inverts or permutes it.
    traj_len_adj = jnp.exp(
        last_adapt_state.log_trajectory_length_moving_average
        - last_adapt_state.log_step_size_moving_average
    )
    parameters = {
        "step_size": jnp.exp(last_adapt_state.log_step_size_moving_average),
        "frozen_step_size": jnp.exp(last_adapt_state.log_step_size_moving_average),
        "inverse_mass_matrix": inverse_mass_matrix,
        "next_random_arg_fn": next_random_arg_fn,
        "integration_steps_fn": lambda arg: integration_steps_fn(arg, traj_len_adj),
        # Nominal leapfrog count L = T/eps before the Halton jitter, i.e. the
        # frozen trajectory length in integration steps. This is the quantity the
        # trajectory-growth literature quotes ("L 136 -> 27").
        "frozen_L_nominal": traj_len_adj,
        # Number of warmup steps actually executed: equals num_steps unless a stop rule fired.
        "warmup_steps_used": int(stopped_at) if stopped_at else int(num_steps),
    }
    # warm_info is the (info, trace) stack. The sampler's `_run_chees` returns it
    # as `chees_info`, which sample() currently does not read -- so the added
    # trace costs nothing downstream, and a future caller can read the whole
    # warmup curve (step, eps, T, harmonic acceptance) without re-running.
    return (last_states, parameters), WarmupInfo(info=warm_info, trace=trace_hist)