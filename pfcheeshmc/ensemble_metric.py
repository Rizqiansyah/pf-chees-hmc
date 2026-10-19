"""WP4 -- ensemble mass-matrix estimators and the two-phase metric refresh.

Settled convention for this run (do not relitigate; see ``results/LEDGER.md``, the entry
"RETRACTION of 699be6d"): the array the HMC kernel consumes in the ``inverse_mass_matrix``
slot is ``M^-1`` in ``KE = 1/2 p^T A p``, and the *correct* value to put in the slot is the
posterior **COVARIANCE** (leapfrog frequencies are ``eig(A H)`` with target Hessian
``H = Sigma^-1``). Every estimator here therefore returns a **covariance**, in the kernel's
coordinate order, and is handed to the warmup unchanged (``imm_orientation="covariance"``).

Library-first (plan A6): the four estimators are thin wrappers over the BlackJAX 1.7.1 fork's
``blackjax.adaptation.metric_estimators`` where one exists, so the only hand-written numerics
are (i) the diagonal-shrinkage convex combination for the dense estimator and (ii) the explicit
ridge + positive-definite assertion, which the fork does not apply.

All estimators take draws ``(N, d)`` already flattened in **kernel order** and return a dense
``(d, d)`` covariance. They are pure (no sampler state) so they are testable in eager mode on
CPU; the caller is responsible for flattening positions through the kernel's own ravel.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "positions_to_draws",
    "estimate_covariance",
    "reseed_trajectory_length",
    "ENGLISH_NAMES",
]


# --------------------------------------------------------------------------- flatten
def positions_to_draws(pos_hist, steps: int) -> jnp.ndarray:
    """Flatten a stacked warmup-position tree into ``(steps * C, d)`` in kernel order.

    ``pos_hist`` is the ``lax.scan`` output of ``chees_warmup_with_progress`` with
    ``stop_rule=None``: a dict keyed by variable, each leaf shaped ``(steps, C, ...)``.
    Flattening is done with the SAME ``ravel_pytree`` the kernel uses, so the coordinate
    order is the kernel's (sorted-key) order -- not Pathfinder's tuple order. Mixing the
    two is exactly the P0 this run exists to fix.
    """
    # Each leaf is (steps, C, ...). ``ravel_pytree`` would concatenate ALL leaves into one flat
    # vector and lose the (steps, C, d) structure, so flatten per-leaf instead. ``tree_leaves``
    # on a dict visits keys in sorted order -- the SAME order the HMC kernel ravels the state,
    # which is the whole point of doing the flattening this way (kernel order, not tuple order).
    leaves = jax.tree_util.tree_leaves(pos_hist)
    if not leaves:
        raise ValueError("empty position tree")
    s0 = int(leaves[0].shape[0])
    if s0 != int(steps):
        raise ValueError(
            f"pos_hist leaves have {s0} steps on the leading axis but {steps} were expected; "
            "with stop_rule set, positions hold only the last block")
    cols = [leaf.reshape(s0, int(leaf.shape[1]), -1) for leaf in leaves]     # each (steps, C, k)
    flat = jnp.concatenate(cols, axis=2)                                     # (steps, C, d)
    return flat.reshape(s0 * int(flat.shape[1]), int(flat.shape[2]))         # (steps*C, d)


def _estimate(name: str, draws: jnp.ndarray, grads: Optional[jnp.ndarray],
              ridge: float, shrink: float, max_rank: int) -> jnp.ndarray:
    from blackjax.adaptation import metric_estimators as ME   # imported lazily: the fork module

    d = draws.shape[-1]
    eye = jnp.eye(d, dtype=draws.dtype)

    if name == "sample_diag":
        # Library: Bessel-corrected per-coordinate sample variance = diagonal of the sample
        # covariance.
        cov = jnp.diag(ME.welford_diagonal(draws))
    elif name == "fisher_diag":
        # Library: sigma^2 = sqrt(Var[x] / Var[grad]) per coordinate, clamped to nutpie's
        # [1e-20, 1e20]. This IS the covariance diagonal in this run's convention.
        if grads is None:
            raise ValueError("the 'fisher_diag' estimator requires gradients")
        cov = jnp.diag(ME.fisher_score_diagonal(draws, grads))
    elif name == "dense_shrunk":
        # Library dense sample covariance, shrunk toward its own diagonal. The convex
        # combination is the only hand-written step (A6 exception): it controls the
        # condition number of a dense estimate built from a small phase-A pool.
        cov_full = ME.welford_dense(draws)
        cov = (1.0 - shrink) * cov_full + shrink * jnp.diag(jnp.diag(cov_full))
    elif name == "diag_lowrank":
        # Library low-rank metric from the SVD of standardised draws; expanded to a dense
        # covariance with the SAME expression WP1 uses for the lbfgs/fisher low-rank arms
        # (D (I + U (lam-1) U^T) D), so orientation is consistent across the codebase.
        lr = ME.draws_singular_value_low_rank(draws, max_rank)
        D = jnp.diag(lr.sigma)
        cov = D @ (eye + lr.U @ jnp.diag(lr.lam - 1.0) @ lr.U.T) @ D
    else:
        raise ValueError(f"unknown estimator {name!r}; expected one of {sorted(ENGLISH_NAMES)}")

    return _regularize_pd(cov, ridge, eye)


def _regularize_pd(cov: jnp.ndarray, ridge: float, eye: jnp.ndarray) -> jnp.ndarray:
    """Add an explicit ridge and fail loudly (not with NaN) if the result is not PD.

    ``metrics._format_covariance`` Cholesky-factors its input, so a non-PD matrix would
    otherwise become NaN downstream. Symmetrise first (the shrinkage/low-rank steps are only
    numerically symmetric), then assert the smallest eigenvalue is positive on the host.
    """
    cov = 0.5 * (cov + cov.T) + ridge * eye
    eigs = np.asarray(jax.device_get(jnp.linalg.eigvalsh(cov)), dtype=np.float64)
    if not np.all(np.isfinite(eigs)) or float(eigs.min()) <= 0.0:
        raise ValueError(
            f"ensemble metric is not positive definite after ridge={ridge}: "
            f"eigs min={eigs.min():.3e}, max={eigs.max():.3e}, finite={np.all(np.isfinite(eigs))}")
    return cov


ENGLISH_NAMES = {
    "sample_diag": "sample_diag",
    "fisher_diag": "fisher_diag",
    "dense_shrunk": "dense_shrunk",
    "diag_lowrank": "diag_lowrank",
}


def estimate_covariance(name: str, draws: jnp.ndarray, grads: Optional[jnp.ndarray] = None,
                        ridge: float = 1e-6, shrink: float = 0.5, max_rank: int = 5) -> jnp.ndarray:
    """Return a kernel-order ``(d, d)`` covariance for the given estimator name.

    ``draws`` is ``(N, d)`` in kernel order. ``grads`` is ``(N, d)`` in kernel order and is
    required only for ``fisher_diag``. See the module docstring for the orientation convention.
    """
    draws = jnp.asarray(draws, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    if draws.ndim != 2:
        raise ValueError(f"draws must be (N, d), got shape {draws.shape}")
    if draws.shape[0] < 2:
        raise ValueError(f"need >= 2 draws for a covariance, got {draws.shape[0]}")
    if grads is not None:
        grads = jnp.asarray(grads, dtype=draws.dtype)
        if grads.shape != draws.shape:
            raise ValueError(f"grads shape {grads.shape} != draws shape {draws.shape}")
    return _estimate(name, draws, grads, float(ridge), float(shrink), int(max_rank))


def reseed_trajectory_length(cov: jnp.ndarray) -> float:
    """T0 = (pi/2) * sqrt(l_max(M^-1)) for a covariance metric, without forming the inverse.

    The brief defines the reseed in terms of the largest eigenvalue of the metric's *inverse*.
    For a covariance ``Sigma``, ``M^-1 = Sigma^-1`` and its largest eigenvalue is
    ``1 / l_min(Sigma)``. Computing it from ``eigvalsh`` of the (small, dense, PD) covariance
    avoids an explicit matrix inverse and a conditioning loss.
    """
    import math
    eigs = np.asarray(jax.device_get(jnp.linalg.eigvalsh(jnp.asarray(cov))), dtype=np.float64)
    lmin = float(eigs.min())
    if lmin <= 0.0:
        raise ValueError(f"reseed requires a PD covariance; l_min={lmin:.3e}")
    return math.pi / 2.0 * math.sqrt(1.0 / lmin)
