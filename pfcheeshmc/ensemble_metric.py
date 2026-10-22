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

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "positions_to_draws",
    "estimate_covariance",
    "reseed_trajectory_length",
    "whitened_lambda_max",
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
    eye = jnp.eye(draws.shape[-1], dtype=draws.dtype)
    return _regularize_pd(_raw_covariance(name, draws, grads, shrink, max_rank), ridge, eye)


# Module level, with the estimator name and rank static: the library estimators run `lax.scan`s whose
# bodies are closures, which compiled afresh on every call when this ran eagerly. Draws, gradients and
# the shrinkage are arguments, so a second refresh with other draws compiles nothing.
@partial(jax.jit, static_argnames=("name", "max_rank"))
def _raw_covariance(name: str, draws: jnp.ndarray, grads: Optional[jnp.ndarray],
                    shrink: float, max_rank: int) -> jnp.ndarray:
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

    return cov


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


def _sample_covariance(draws: jnp.ndarray) -> jnp.ndarray:
    """Bessel-corrected (ddof=1) dense sample covariance of ``(N, d)`` draws."""
    n = int(draws.shape[0])
    if n < 2:
        raise ValueError(f"need >= 2 draws for a covariance, got {n}")
    x = jnp.asarray(draws, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    x = x - x.mean(axis=0, keepdims=True)
    s = x.T @ x / float(n - 1)
    return 0.5 * (s + s.T)


def _require_pd(mat: jnp.ndarray, what: str) -> np.ndarray:
    """Symmetrise, then fail with a ValueError naming PD (not with a NaN or a Cholesky error)."""
    sym = 0.5 * (mat + mat.T)
    eigs = np.asarray(jax.device_get(jnp.linalg.eigvalsh(sym)), dtype=np.float64)
    if not np.all(np.isfinite(eigs)) or float(eigs.min()) <= 0.0:
        raise ValueError(
            f"{what} must be positive definite; eigs min={eigs.min():.3e}, "
            f"max={eigs.max():.3e}, finite={bool(np.all(np.isfinite(eigs)))} (not PD)")
    return eigs


def whitened_lambda_max(cov_B: jnp.ndarray, draws: jnp.ndarray) -> float:
    """``lambda_max( L^-1 S L^-T )`` -- the re-seed's unit-invariant quantity.

    ``B = L L^T`` is the refreshed metric (a dense covariance in kernel order, Cholesky) and ``S``
    is the Bessel-corrected sample covariance (ddof=1) of the SAME pool draws the estimator used,
    shape ``(N, d)``. ``L^-1`` is applied with ``solve_triangular`` twice and never as an explicit
    inverse.

    Why this and not ``sqrt(1/lambda_min(B))``: under the covariance metric ``M^-1 = B`` the leapfrog
    frequencies are ``sqrt(eig(B H))`` for target Hessian ``H``, and those are invariant to rescaling
    a coordinate. ``lambda_min(B)`` is the precision of the narrowest coordinate in MODEL units, so
    it moves when units move -- that is run-1's defect (plan Part B 1.1), which produced frozen L of
    213-283 on radon and GEV. Whitening S by B makes the quantity a pure number: 1 when ``B == S``,
    ``1 + |rho|`` for a diagonal B against a correlated pool.
    """
    from jax.scipy.linalg import solve_triangular

    B = jnp.asarray(cov_B, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"cov_B must be a square (d, d) matrix, got shape {B.shape}")
    d = int(B.shape[0])
    D = jnp.asarray(draws, dtype=B.dtype)
    if D.ndim != 2 or D.shape[1] != d:
        raise ValueError(f"draws must be (N, {d}), got shape {D.shape}")

    _require_pd(B, "the refreshed metric B (imm_B)")
    S = _sample_covariance(D)
    L = jnp.linalg.cholesky(B)
    A = solve_triangular(L, S, lower=True)                 # L^-1 S
    W = solve_triangular(L, A.T, lower=True).T            # (L^-1 S^T) ^T = L^-1 S L^-T
    W = 0.5 * (W + W.T)
    eigs = np.asarray(jax.device_get(jnp.linalg.eigvalsh(W)), dtype=np.float64)
    if not np.all(np.isfinite(eigs)):
        raise ValueError(f"whitened lambda_max is not finite: eigs={eigs}")
    return float(eigs.max())


def reseed_trajectory_length(cov_B: jnp.ndarray, draws: jnp.ndarray) -> float:
    """``T0_B = (pi/2) * sqrt( lambda_max( L^-1 S L^-T ) )``.

    The plan's formula, verbatim. ``B = L L^T`` is the refreshed metric (``imm_B``, a dense
    covariance in kernel order, Cholesky); ``S`` is the sample covariance (ddof=1) of the SAME
    phase-A pool draws the estimator used, shape ``(N, d)``; ``lambda_max`` comes from
    ``jnp.linalg.eigvalsh`` of the symmetrised ``d x d`` matrix with ``L^-1`` applied by
    ``jax.scipy.linalg.solve_triangular`` (no explicit inverse).

    The caller applies the clip ``T0_B = clip(T0_B, eps_B, max_leapfrog_steps * eps_B)`` because
    ``eps_B`` and ``max_leapfrog_steps`` are warmup settings, not properties of the metric; this
    function stays a pure function of ``(B, draws)``.

    ChEES optimum, for orientation: on a unit Gaussian the criterion is ``4d sin^2 T``, which peaks
    at ``T = pi/2``. So the optimum tracks the largest standard deviation in the metric's whitened
    frame -- which is exactly what ``lambda_max(L^-1 S L^-T)`` measures.
    """
    import math

    return math.pi / 2.0 * math.sqrt(max(whitened_lambda_max(cov_B, draws), 0.0))
