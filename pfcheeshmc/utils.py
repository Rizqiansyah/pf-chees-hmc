import jax, jax.numpy as jnp
import time
from typing import Any, Callable, Dict, NamedTuple, Optional

import numpy as np
import arviz as az
import blackjax as bj
from tqdm.auto import tqdm

from . import compiled as _compiled
# The warmup scan's per-step projection and the acceptance the bars show live with the chunk program
# that produces them (`compiled.warmup_chunk`); they are re-exported here, where they have always been.
from .compiled import WarmupScanInfo, chees_harmonic_acceptance


class WarmupInfo(NamedTuple):
    """What the warmup scan stacks, plus the per-step scalar trace.

    info
        The (projected) warmup scan output, (steps, chains, ...). With the stop rule
        off this is a WarmupScanInfo stack -- num_integration_steps, is_divergent,
        acceptance_rate -- the only fields a consumer reads (the sampler's
        warmup_grad_evals_per_chain, the divergence counters, the stop-rule check).
        Stacking the full HMCInfo would also stack proposal.position/momentum/
        logdensity_grad, ~4x the volume of the states, which is what made the warmup
        scan the device-memory wall at C=1,024 (measured: ~14 GB at d=175, W=2,000).
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


__all__ = [
    "chees_warmup_with_progress",
    "chees_harmonic_acceptance",
    "block_max_rhat",
    "WarmupInfo",
    "WarmupScanInfo",
]


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
    ld_fn,                      # logdensity function, or a compiled.LogDensity (carries fn/names/values)
    rng_key,
    positions,                 # dict {"mu": (C,), "tau_log__": (C,), "theta": (C,J)}
    initial_step_size,
    optim,                     # the Adam learning rate (a stable static for the chunk program), or an
                             #     optax transformation (a new static value per object)
    num_steps,
    inverse_mass_matrix,
    *,
    target_acceptance_rate=0.651,
    decay_rate=0.5,
    jitter_amount=1.0,
    max_sampling_steps=1000,   # unused: the Halton depth is fixed (compiled.HALTON_BITS)
    max_leapfrog_steps=1000,
    update_every=25,           # unused: the bar is updated between chunks
    progressbar=True,          # a tqdm bar updated by the host between chunks; no host callback exists
                             #     in any compiled program, so True and False run the same programs
    stop_rule=None,            # None = fixed num_steps, identical to the previous behaviour.
                             #     Measured net negative on all four ladder models (block
                             #     boundaries cost 6.7-9.0 s, more than any warmup saving);
                             #     kept opt-in for instrumentation
    single_step_iters=0,       # run L=1 for this many steps while the step size settles
):
    """ChEES warmup, as host-driven chunks of ONE jitted program (`compiled.warmup_chunk`).

    The warmup length is the host's loop bound, not a shape or a constant of any program, so a different
    ``num_steps`` compiles nothing. Per-step keys are ``fold_in(rng_key, global_step_index)``; the Halton
    jitter index runs in the carried state. Both are independent of ``num_steps`` and of the chunk size.

    Returns ``(last_states, parameters), WarmupInfo``; ``WarmupInfo.info`` / ``.trace`` are NumPy stacks
    with one row per step run.
    """
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

    num_steps = int(num_steps)
    ld = ld_fn if isinstance(ld_fn, _compiled.LogDensity) else _compiled.plain_log_density(ld_fn, tuple(positions))

    # Settings the chunk programs branch on (static for jit); everything else is an argument.
    if optim is None:
        optim = 0.05                       # the vendored adapter's own default learning rate
    settings = dict(
        optim=optim if hasattr(optim, "init") else float(optim),
        target_accept=float(target_acceptance_rate), decay_rate=float(decay_rate),
        jitter_amount=float(jitter_amount), max_leapfrog_steps=int(max_leapfrog_steps),
        single_step_iters=int(single_step_iters))

    # Start: HMC states for every chain, and the adapter's state. The step size is a value.
    eps0 = jnp.asarray(initial_step_size, dtype=jnp.result_type(initial_step_size))
    init_states, adaptation_state = _compiled.warmup_init(
        ld.fn, ld.names, ld.values, positions, eps0, **settings)

    def chunk_call(size, stack):
        def call(carry, start, n):
            return _compiled.warmup_chunk(
                ld.fn, ld.names, ld.values, carry, imm_arr, rng_key, np.int32(start), np.int32(n),
                chunk=size, stack_positions=stack, **settings)
        return call

    # --- one progress bar, like your dynamic-HMC bar ------------------------
    # The host updates it between chunks from the chunk's own outputs (divergences, the harmonic
    # acceptance and the step size), so the bar costs the compiled programs nothing, and is not a
    # reason for a second program: no host callback is traced into any of them.
    bar = tqdm(total=num_steps, desc="ChEES Adaptation", position=0, leave=True) if progressbar else None
    div_total = [0]

    def tick(done, info, trace):
        """``info`` / ``trace``: the chunk's real rows, on the host."""
        if bar is None:
            return
        div_total[0] += int(np.sum(info.is_divergent))
        bar.n = done
        bar.set_postfix_str(
            f"Divergence: {div_total[0]}, Acceptance rate: {float(trace.harmonic_acceptance[-1]):.2f}, "
            f"ε: {float(trace.step_size[-1]):.3g}")
        bar.refresh()

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

    carry = (init_states, adaptation_state)
    if stop_rule:
        # The stop rule's blocks are chunks: the block length is a setting (static), so a rule compiles
        # once, and the last, shorter block is the same program with a smaller active count. After a
        # block the rule reads the position window and the adapter state, so this path is synchronous.
        block = int(stop_rule.get("block", 250))
        # Default window = the whole block. A 50-draw window made split-R-hat read 1.04 on a unit
        # Normal no matter how long warmup ran; 250 draws per chain reads 1.003-1.010.
        window = int(stop_rule.get("window", block))
        call = chunk_call(block, True)
        done = 0
        stopped_at = None
        while done < num_steps:
            n = min(block, num_steps - done)
            _t = time.perf_counter()
            carry, (pos_blk, info_blk, trace_blk) = call(carry, done, n)
            jax.block_until_ready(carry)
            _t_scan = time.perf_counter() - _t
            done += n
            if bar is not None:
                tick(done, *[jax.tree_util.tree_map(lambda x: np.asarray(x)[:n], t)
                             for t in (info_blk, trace_blk)])
            if done < num_steps and _stop_met(
                    jax.tree_util.tree_map(lambda p: p[-window:], pos_blk), info_blk, carry[1])[0]:
                stopped_at = done
            if stop_rule.get("debug"):
                print(f"    block {done:5d}: scan {_t_scan:6.2f} s", flush=True)
            if stopped_at is not None:
                break
        # The last block's stacks only, as before.
        warm_info, trace_hist = jax.tree_util.tree_map(
            lambda x: np.asarray(x)[:n], (info_blk, trace_blk))
    else:
        # Default path: chunks of WARMUP_CHUNK steps. The positions are not stacked: that stack is the
        # memory wall at C=1,024, and only the stop rule reads it.
        rest = _compiled.HostStack(num_steps)

        def consume(start, n, out):
            host = jax.tree_util.tree_map(np.asarray, out)
            rest.add(start, n, host)
            tick(start + n, *[jax.tree_util.tree_map(lambda x: x[:n], t) for t in host])

        carry = _compiled.run_chunks(chunk_call(_compiled.WARMUP_CHUNK, False), carry,
                                     num_steps, _compiled.WARMUP_CHUNK, consume)
        warm_info, trace_hist = rest.result()
        stopped_at = None
    last_states, last_adapt_state = carry

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
        "next_random_arg_fn": _compiled._next_random_arg,
        "integration_steps_fn": lambda arg: _compiled.integration_steps(arg, traj_len_adj, float(jitter_amount)),
        # Nominal leapfrog count L = T/eps before the Halton jitter, i.e. the
        # frozen trajectory length in integration steps. This is the quantity the
        # trajectory-growth literature quotes ("L 136 -> 27").
        "frozen_L_nominal": traj_len_adj,
        # What the sampling stage's chunk program takes as arguments in place of the closure above:
        # the trajectory-length adjustment (a value) and the jitter setting.
        "traj_len_adj": traj_len_adj,
        "jitter_amount": float(jitter_amount),
        # Number of warmup steps actually executed: equals num_steps unless a stop rule fired.
        "warmup_steps_used": int(stopped_at) if stopped_at else int(num_steps),
    }
    # warm_info is the projected per-step stack (WarmupScanInfo): exact gradient counts,
    # divergences, and the acceptance signal -- NOT the full HMCInfo, whose proposal.*
    # leaves are ~4x the states and were the warmup memory wall at C=1,024 (WP5).
    return (last_states, parameters), WarmupInfo(info=warm_info, trace=trace_hist)
