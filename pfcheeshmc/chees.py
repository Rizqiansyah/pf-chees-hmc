# Copyright 2020- The Blackjax developers (the original, blackjax/adaptation/chees_adaptation.py)
# Modifications Copyright 2025-2026 Rizqiansyah (listed in the docstring below)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Vendored ChEES warmup adapter (BlackJAX `chees_adaptation.base`).

Source
------
BlackJAX 1.7.1, commit gcde2ab036 (the wheel this project pins), file
`blackjax/adaptation/chees_adaptation.py`, function `base` and its module-level
helpers (`weighted_empirical_mean`, `ChEESAdaptationState`, the three constants).
BlackJAX is Apache-2.0; the upstream header and the long whitened-criterion
derivation comment are kept verbatim so a reviewer can diff this file against the
installed one.

Why vendored (one-line justification for the library-first exception, A6)
--------------------------------------------------------------------------
The adapter returns a per-step scalar trace (step, step_size, trajectory_length, harmonic_acceptance) beside its
state, and can build its Adam optimiser from a learning rate. Neither is an option of `base`, and the only other
levers are a source edit of site-packages (forbidden) or a monkeypatch at call time (not shippable). So: vendor
`base`.

Deliberate differences from the vendored upstream `base` (and nothing else)
-----------------------------------------------------------------------------
1. `single_step_iters`: for the first N updates the consumed trajectory length
   is pinned to one leapfrog (L=1, which is already the state's clip minimum),
   i.e. the step size settles before the trajectory grows. SNAPER runs ~100
   such steps. 0 (default) is a Python-level no-op, not even a jnp.where.
2. `optim=None` builds `optax.adam(adam_lr, b1=adam_b1, b2=adam_b2)`; callers
   may still pass an optimiser positionally exactly as upstream.
3. `update` returns `(new_state, ChEESTrace)` -- a per-step scalar trace
   (step, step_size, trajectory_length, harmonic_acceptance) so a caller can
   plot or gate on the warmup without reconstructing it from the scan output.
   The state itself is untouched by this.

Everything else is preserved exactly: harmonic-mean acceptance target 0.651
with the non-finite -> 0 guard, divergence exclusion from BOTH the dual
averaging and the criterion weighting, the LOG_UPDATE_CLIP on the log-space
update, the `isfinite` guard on the Adam step, the T clip to
[new_step_size, max_leapfrog_steps * new_step_size], the `_whiten_criterion`
ablation seam, and the Halton jitter contract (jitter_generator is applied
inside the gradient as upstream does).

Note on the slow-direction length floor: in the installed wheel it lives in the
PUBLIC `chees_adaptation(...)` builder only (gated on `mass_matrix_estimation`,
applied at consumption), not in `base`. This project calls `base` directly, so
the floor was never active on this path; vendoring `base` changes nothing here.
"""

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax

import blackjax.mcmc.dynamic_hmc as dynamic_hmc  # noqa: F401  (parity with upstream imports)
import blackjax.optimizers.dual_averaging as dual_averaging
from blackjax.types import Array, ArrayLikeTree

# Upstream constants, verbatim (chees_adaptation.py:19-25).
# optimal tuning for HMC, see https://arxiv.org/abs/1001.4460
OPTIMAL_TARGET_ACCEPTANCE_RATE = 0.651
# Clip the final log-space update like the original implementation in TFP (~log(2)/2 ≈ 0.35).
LOG_UPDATE_CLIP = 0.35
# Small constant to avoid division by zero or log of zero
EPS_FLOAT = 1e-20

__all__ = [
    "ChEESAdaptationState",
    "ChEESTrace",
    "base",
    "weighted_empirical_mean",
    "OPTIMAL_TARGET_ACCEPTANCE_RATE",
]


class ChEESAdaptationState(NamedTuple):
    """State of the ChEES-HMC adaptation scheme. (Upstream, unchanged.)

    step_size
        Value of the step_size parameter of the HMC algorithm.
    log_step_size_moving_average
        Running moving average of the log step_size parameter.
    trajectory_length
        Value of the num_integration_steps * step_size parameter of
        the HMC algorithm.
    log_trajectory_length_moving_average
        Running moving average of the log num_integration_steps / step_size
        parameter.
    optim_state
        Optax optimizing state for used to maximize the ChEES criterion.
    random_generator_arg
        Utility array for generating a pseudo or quasi-random sequence of
        numbers.
    step
        Current iteration number.

    """

    step_size: float
    log_step_size_moving_average: float
    trajectory_length: float
    log_trajectory_length_moving_average: float
    da_state: dual_averaging.DualAveragingState
    optim_state: optax.OptState
    random_generator_arg: Array
    step: int


class ChEESTrace(NamedTuple):
    """Per-step scalar trace of one ChEES update (new, not in upstream).

    step
        1-based update index (the step number just executed).
    step_size
        The step size the state carries after this update.
    trajectory_length
        The trajectory length (time, T = L * eps) the state carries after this
        update -- i.e. what the kernel consumes next. trajectory_length /
        step_size is the nominal leapfrog count L at this step.
    harmonic_acceptance
        The harmonic mean acceptance over the NON-DIVERGENT chains that this
        update's dual averaging ran on (0.0 if non-finite). This is the
        quantity ChEES actually adapts step size against, not the arithmetic
        mean of per-chain acceptances.
    """

    step: int
    step_size: float
    trajectory_length: float
    harmonic_acceptance: float


def weighted_empirical_mean(x, w):
    # x: (num_chains, dim), w: (num_chains,)
    # Upstream, verbatim.
    x_safe = jnp.where(jnp.isfinite(x), x, 0.0)
    w = jnp.where(jnp.isfinite(x).all(axis=-1), w, 0.0)

    w_exp = w.reshape((w.shape[0],) + (1,) * (x.ndim - 1))
    num = jnp.sum(w_exp * x_safe, axis=0)
    den = jnp.sum(w_exp, axis=0) + EPS_FLOAT
    return jax.lax.stop_gradient(num / den)


def base(
    jitter_generator: Callable,
    next_random_arg_fn: Callable,
    optim: Optional[optax.GradientTransformation] = None,
    target_acceptance_rate: float = OPTIMAL_TARGET_ACCEPTANCE_RATE,
    decay_rate: float = 0.5,
    max_leapfrog_steps: int = 1000,
    *,
    single_step_iters: int = 0,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_lr: float = 0.05,
    _whiten_criterion: bool = True,
) -> tuple[Callable, Callable]:
    """ChEES criterion (trajectory length) + dual averaging (step size) for the
    jittered HMC kernel, :cite:p:`hoffman2021adaptive`.

    Same contract as upstream `blackjax.adaptation.chees_adaptation.base` --
    returns `(init, update)` where `update` additionally returns a scalar
    `ChEESTrace` (see the module docstring for the full list of differences).

    Parameters are upstream's, plus:
      single_step_iters: run with L=1 for this many updates before the
          trajectory is allowed to grow (SNAPER uses ~100).
      adam_b1/adam_b2/adam_lr: used only when `optim` is None.
    """
    if single_step_iters < 0:
        raise ValueError("single_step_iters must be >= 0")
    if optim is None:
        optim = optax.adam(adam_lr, b1=adam_b1, b2=adam_b2)

    da_init, da_update, _ = dual_averaging.dual_averaging()

    def compute_parameters(
        proposed_positions: ArrayLikeTree,
        proposed_momentums: ArrayLikeTree,
        initial_positions: ArrayLikeTree,
        acceptance_probabilities: Array,
        is_divergent: Array,
        initial_adaptation_state: ChEESAdaptationState,
        inverse_mass_matrix: Array,
    ) -> tuple[ChEESAdaptationState, Array]:
        """Compute values for the parameters based on statistics collected from
        multiple chains. (Upstream docstring.)
        """
        (
            step_size,
            log_step_size_ma,
            trajectory_length,
            log_trajectory_length_ma,
            da_state,
            optim_state,
            random_generator_arg,
            step,
        ) = initial_adaptation_state

        harmonic_mean = 1.0 / jnp.mean(
            1.0 / acceptance_probabilities, where=~is_divergent
        )
        # Replace inf/nan harmonic mean as zero to avoid issues in dual averaging
        harmonic_mean = jnp.where(jnp.isfinite(harmonic_mean), harmonic_mean, 0.0)
        da_state_ = da_update(da_state, target_acceptance_rate - harmonic_mean)
        step_size_ = jnp.exp(da_state_.log_x)
        new_step_size, new_da_state, new_log_step_size = jax.lax.cond(
            jnp.isfinite(step_size_),
            lambda _: (step_size_, da_state_, da_state_.log_x),
            lambda _: (step_size, da_state, da_state.log_x),
            None,
        )
        update_weight = step ** (-decay_rate)
        new_log_step_size_ma = (
            1.0 - update_weight
        ) * log_step_size_ma + update_weight * new_log_step_size

        w = jnp.where(~is_divergent, acceptance_probabilities, 0.0)
        proposals_mean = jax.tree.map(
            lambda p: weighted_empirical_mean(p, w), proposed_positions
        )
        # The above weighted mean is presumably better than the simple mean:
        # proposals_mean = jax.tree.map(
        #     lambda p: jnp.nanmean(p, axis=0), proposed_positions
        # )
        initials_mean = jax.tree.map(
            lambda p: jnp.nanmean(p, axis=0), initial_positions
        )
        proposals_centered = jax.tree.map(
            lambda p, pm: p - pm, proposed_positions, proposals_mean
        )
        initials_centered = jax.tree.map(
            lambda p, pm: p - pm, initial_positions, initials_mean
        )

        vmap_flatten_op = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])
        proposals_matrix = vmap_flatten_op(proposals_centered)
        initials_matrix = vmap_flatten_op(initials_centered)
        momentums_matrix = vmap_flatten_op(proposed_momentums)

        # --- Metric-aware (whitened) ChEES criterion ------------------------
        # ChEES's trajectory-length gradient is a pathwise-derivative estimator
        # of d/dL E[||x' - E[x']||^2], built from the product
        #     (||Δx'||^2 - ||Δx||^2) * <Δx', dx'/dL>
        # where dx'/dL is proportional to the trajectory's *velocity* at the
        # endpoint. Under Hamiltonian dynamics with mass matrix M = Σ^-1,
        # velocity v = M^-1 p = Σp -- this is blackjax's own convention, e.g.
        # metrics.py's `linear_map(inverse_mass_matrix, momentum)` kinetic-
        # energy term and the NUTS U-turn check. The pre-existing code below
        # implicitly assumes M = I (so v == p): that's exactly what TFP's own
        # implementation comment flags at this spot ("implicitly assumes an
        # identity mass matrix") -- there is no reference implementation for
        # the general-M case to copy, hence the derivation here.
        #
        # For a general diagonal M, the *norm* term must be measured in the
        # geometry M induces -- the whitened space x̃ = Σ^{-1/2}x in which the
        # preconditioned sampler behaves isotropically -- otherwise a handful
        # of large-variance dimensions dominate ||Δx||^2 and swamp the
        # trajectory-length signal coming from the well-scaled ones. That is
        # edit 1: proposals/initials get whitened by Σ^{-1/2} before the dot
        # products that build the norm difference.
        #
        # The <Δx', v'> term needs the SAME whitening transform applied to
        # velocity: velocity is a tangent vector living in position-space (not
        # momentum's dual/cotangent space), so under x̃ = Σ^{-1/2}x it
        # transforms the same way position differences do, i.e.
        # ṽ = Σ^{-1/2}v -- NOT the contragredient transform momentum itself
        # would get. That is edit 2, and skipping it is the "silent-bug trap":
        # reusing the edit-1-whitened Δx' in `jnp.dot(pm, mm)` without also
        # whitening the velocity leaves a spurious Σ^{-1/2} factor on the
        # momentum term. Composing both edits explicitly:
        #     v'  = Σ  p'                (velocity, metrics.py convention)
        #     ṽ' = Σ^{-1/2} v' = Σ^{1/2} p'   (velocity whitened like position)
        #     <x̃', ṽ'> = Σ_i (Δx'_i / √Σ_i) · (√Σ_i · p'_i) = Σ_i Δx'_i p'_i
        #              = <Δx', p'>   (the RAW expression, for *any* diagonal Σ)
        # This exact cancellation is not a coincidence: Δx' and p' are
        # canonically conjugate (position, momentum), so their pairing is
        # metric-invariant -- unlike ||Δx||^2 alone. The code below still
        # computes both quantities via the explicit whitening transforms
        # (rather than hand-simplifying to the raw dot product) so that (a)
        # this derivation is directly checkable against the code and (b) with
        # Σ = ones every multiply/divide is an IEEE-754 no-op, so the whitened
        # path is bit-for-bit identical to the original expression -- see
        # ``test_whitened_criterion_reduces_to_raw_when_identity``.
        #
        # `_whiten_criterion=False` (private ablation seam, see `base`'s
        # docstring) skips both edits and always uses the raw, pre-existing
        # expression regardless of `inverse_mass_matrix` -- this is the
        # "naive" arm (kernel uses the estimated diagonal metric, but the
        # criterion does not) the validation study compares against.
        if _whiten_criterion:
            inv_sqrt_imm = 1.0 / jnp.sqrt(inverse_mass_matrix)
            proposals_matrix_w = proposals_matrix * inv_sqrt_imm
            initials_matrix_w = initials_matrix * inv_sqrt_imm
            velocity_matrix = momentums_matrix * inverse_mass_matrix  # v' = Σp'
            velocity_matrix_w = velocity_matrix * inv_sqrt_imm  # ṽ' = Σ^{-1/2}v'
        else:
            proposals_matrix_w = proposals_matrix
            initials_matrix_w = initials_matrix
            velocity_matrix_w = momentums_matrix

        trajectory_gradients = (
            jitter_generator(random_generator_arg)
            * trajectory_length  # this effectively make this gradient w.r.t. log_trajectory_length
            * jax.vmap(
                lambda pm, im, mm: (jnp.dot(pm, pm) - jnp.dot(im, im)) * jnp.dot(pm, mm)
            )(proposals_matrix_w, initials_matrix_w, velocity_matrix_w)
        )

        trajectory_gradient = jnp.sum(
            acceptance_probabilities * trajectory_gradients,
            where=~is_divergent,
        ) / jnp.sum(acceptance_probabilities + EPS_FLOAT, where=~is_divergent)

        # Upstream lines 473-474: `log_trajectory_length = jnp.log(trajectory_length)`
        # followed by `optim.update(trajectory_gradient, optim_state,
        # log_trajectory_length)`. Because upstream sets
        # trajectory_length = exp(log_trajectory_length_moving_average), Adam's
        # operating point here is the t^-0.5-weighted moving average, not the
        # raw iterate (Hoffman et al. 2021 Algorithm 1 differs here; this
        # adapter keeps upstream's behaviour).
        log_trajectory_length = jnp.log(trajectory_length)
        updates, optim_state_ = optim.update(
            trajectory_gradient, optim_state, log_trajectory_length
        )

        updates = jax.tree.map(
            lambda u: jnp.clip(u, -LOG_UPDATE_CLIP, LOG_UPDATE_CLIP), updates
        )
        log_trajectory_length_ = optax.apply_updates(log_trajectory_length, updates)
        new_log_trajectory_length, new_optim_state = jax.lax.cond(
            jnp.isfinite(
                jax.flatten_util.ravel_pytree(log_trajectory_length_)[0]
            ).all(),
            lambda _: (log_trajectory_length_, optim_state_),
            lambda _: (log_trajectory_length, optim_state),
            None,
        )
        new_log_trajectory_length_ma = (
            1.0 - update_weight
        ) * log_trajectory_length_ma + update_weight * new_log_trajectory_length

        # Upstream's behaviour: the kernel consumes the moving average.
        new_trajectory_length = jnp.exp(new_log_trajectory_length_ma)

        # clip new trajectory length to avoid too large trajectories, also the
        # minimum trajectory length is one integrator step
        new_trajectory_length = jnp.clip(
            new_trajectory_length,
            max=max_leapfrog_steps * new_step_size,
            min=new_step_size,
        )
        if single_step_iters > 0:
            # Settle the step size first: L=1 (the clip minimum) for the first
            # single_step_iters updates.
            new_trajectory_length = jnp.where(
                step <= single_step_iters, new_step_size, new_trajectory_length
            )
        state = ChEESAdaptationState(
            new_step_size,
            new_log_step_size_ma,
            new_trajectory_length,
            new_log_trajectory_length_ma,
            new_da_state,
            new_optim_state,
            next_random_arg_fn(random_generator_arg),
            step + 1,
        )
        return state, harmonic_mean

    def init(random_generator_arg: Array, step_size: float):
        return ChEESAdaptationState(
            step_size=step_size,
            log_step_size_moving_average=0.0,
            trajectory_length=step_size,
            log_trajectory_length_moving_average=0.0,
            da_state=da_init(step_size),
            optim_state=optim.init(step_size),
            random_generator_arg=random_generator_arg,
            step=1,
        )

    def update(
        adaptation_state: ChEESAdaptationState,
        proposed_positions: ArrayLikeTree,
        proposed_momentums: ArrayLikeTree,
        initial_positions: ArrayLikeTree,
        acceptance_probabilities: Array,
        is_divergent: Array,
        inverse_mass_matrix: Array,
    ) -> tuple[ChEESAdaptationState, ChEESTrace]:
        """Upstream `update` plus a scalar trace (see ChEESTrace)."""
        new_state, harmonic_mean = compute_parameters(
            proposed_positions,
            proposed_momentums,
            initial_positions,
            acceptance_probabilities,
            is_divergent,
            adaptation_state,
            inverse_mass_matrix,
        )
        trace = ChEESTrace(
            step=new_state.step,
            step_size=new_state.step_size,
            trajectory_length=new_state.trajectory_length,
            harmonic_acceptance=harmonic_mean,
        )
        return new_state, trace

    return init, update
