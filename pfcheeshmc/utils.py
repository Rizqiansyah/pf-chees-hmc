from functools import partial
import jax, jax.numpy as jnp
import numpy as np
import arviz as az
import blackjax as bj
from jax import lax
from tqdm.auto import tqdm
import blackjax.mcmc.dynamic_hmc as dynamic_hmc

__all__ = ["chees_warmup_with_progress"]

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
    update_every=25,           # <-- throttle UI updates to keep overhead tiny
):
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

    # Low-level ChEES adapter (init, update)
    init_adapt, update_adapt = bj.adaptation.chees_adaptation.base(
        jitter_gn, next_random_arg_fn, optim, target_acceptance_rate, decay_rate
    )

    # Batch-initialize states for all chains
    num_chains_local = next(iter(positions.values())).shape[0]
    batch_init = jax.jit(jax.vmap(lambda p: dynamic_hmc.init(p, ld_fn, init_random_arg)))
    init_states = batch_init(positions)
    adaptation_state = init_adapt(init_random_arg, initial_step_size)

    # --- one progress bar, like your dynamic-HMC bar ------------------------
    bar = tqdm(total=num_steps, desc="ChEES Adaptation", position=0, leave=True)
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
            trajectory_length_adjusted=astate.trajectory_length / astate.step_size,
        )
        new_states, info = jax.vmap(_step)(keys, states)

        # accept field name can be acceptance_probability or acceptance_rate depending on BJ version
        acc_vals = getattr(info, "acceptance_probability", None)
        if acc_vals is None:
            acc_vals = info.acceptance_rate

        # diagnostics for this iteration (across chains)
        div_step  = jnp.sum(info.is_divergent)
        acc_mean  = jnp.mean(acc_vals)

        # Update adaptation state
        new_astate = update_adapt(
            astate,
            info.proposal.position,
            info.proposal.momentum,
            states.position,
            acc_vals,              # accepts vector of acceptances
            info.is_divergent,
        )

        # JAX-safe throttling of host callback
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

        return (new_states, new_astate), info  # (carry, y)

    one_step_jit = jax.jit(one_step)

    # keys for each warmup step
    keys = jax.random.split(rng_key, num_steps)

    # Run the scan (no BlackJAX progress wrapper; our bar is the only one)
    (last_states, last_adapt_state), warm_info = jax.lax.scan(
        one_step_jit,
        (init_states, adaptation_state),
        (jnp.arange(num_steps), keys),
    )

    bar.close()

    # Assemble parameters like BJ’s run()
    traj_len_adj = jnp.exp(
        last_adapt_state.log_trajectory_length_moving_average
        - last_adapt_state.log_step_size_moving_average
    )
    parameters = {
        "step_size": jnp.exp(last_adapt_state.log_step_size_moving_average),
        "inverse_mass_matrix": inverse_mass_matrix,
        "next_random_arg_fn": next_random_arg_fn,
        "integration_steps_fn": lambda arg: integration_steps_fn(arg, traj_len_adj),
    }
    return (last_states, parameters), warm_info