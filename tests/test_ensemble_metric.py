"""WP4 -- ensemble metric estimators and the two-phase metric refresh.

Orientation convention (settled this run, see results/LEDGER.md "RETRACTION of 699be6d"): every
estimator returns the posterior COVARIANCE in the kernel's coordinate order, which is the value
the HMC kernel's ``inverse_mass_matrix`` slot must hold. These tests pin that, plus the ridge/PD
guard, the kernel-order flattening, and -- as a teeth test -- that the step-size re-search after
the swap actually moves eps (a silent no-op would leave it exactly equal to the phase-A value).
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import pymc as pm

jax.config.update("jax_enable_x64", True)

from pfcheeshmc import ensemble_metric as EM
from pfcheeshmc.ensemble_metric import estimate_covariance, positions_to_draws, reseed_trajectory_length

ES = ["sample_diag", "fisher_diag", "dense_shrunk", "diag_lowrank"]


# --------------------------------------------------------------------------- fixtures
def _anisotropic_draws(n=4000, seed=0):
    """Kernel-order draws with per-coordinate variances 100 / 3 / 1 / 0.01 (a 1e4 spread)."""
    rng = np.random.default_rng(seed)
    sd = np.array([10.0, np.sqrt(3.0), 1.0, 0.1])
    draws = jnp.asarray(rng.normal(0.0, 1.0, size=(n, 4)) * sd, dtype=jnp.float64)
    # score of a standard-Normal posterior: grad_i = -x_i / var_i
    grads = jnp.asarray(-draws / (sd ** 2), dtype=jnp.float64)
    return draws, grads, sd ** 2


# --------------------------------------------------------------------------- estimators
@pytest.mark.parametrize("name", ES)
def test_estimator_returns_symmetric_pd_covariance(name):
    draws, grads, var = _anisotropic_draws()
    cov = np.asarray(estimate_covariance(name, draws, grads if name == "fisher_diag" else None))
    d = draws.shape[-1]
    assert cov.shape == (d, d), f"{name}: shape {cov.shape}"
    assert np.max(np.abs(cov - cov.T)) < 1e-8 * max(1.0, np.abs(cov).max()), f"{name}: not symmetric"
    eigs = np.linalg.eigvalsh(cov)
    assert np.all(np.isfinite(eigs)), f"{name}: non-finite eigenvalues {eigs}"
    assert eigs.min() > 0.0, f"{name}: not positive definite, min eig {eigs.min()}"
    # The diagonal should be the per-coordinate variance (the estimator's job), within 20%.
    diag = np.diag(cov)
    rel = np.abs(diag - var) / var
    assert rel.max() < 0.2, f"{name}: diag {diag} vs true var {var} (max rel err {rel.max()})"


@pytest.mark.parametrize("name", ["sample_diag", "dense_shrunk", "diag_lowrank"])
def test_degenerate_input_yields_ridge_not_nan(name):
    """All-identical draws give a zero sample covariance; the ridge must make it PD, not NaN."""
    degen = jnp.zeros((16, 3), dtype=jnp.float64)
    cov = np.asarray(estimate_covariance(name, degen, ridge=1e-4))
    assert np.all(np.isfinite(cov)), f"{name}: NaN/inf on degenerate input"
    eigs = np.linalg.eigvalsh(cov)
    assert eigs.min() > 0.0, f"{name}: degenerate input not made PD, min eig {eigs.min()}"
    # The only signal left is the ridge, so the matrix should be ~ ridge * I.
    assert np.allclose(cov, 1e-4 * np.eye(3), atol=1e-6), f"{name}: {cov}"


def test_fisher_requires_gradients():
    draws, _, _ = _anisotropic_draws(n=200)
    with pytest.raises(ValueError, match="requires gradients"):
        estimate_covariance("fisher_diag", draws, grads=None)


def test_unknown_estimator_raises():
    draws, grads, _ = _anisotropic_draws(n=200)
    with pytest.raises(ValueError, match="unknown estimator"):
        estimate_covariance("nope", draws, grads)


def test_reseed_matches_closed_form():
    """T0 = (pi/2)*sqrt(l_max(Sigma^-1)) = (pi/2)/sqrt(l_min(Sigma))."""
    cov = jnp.asarray(np.diag([100.0, 3.0, 1.0, 0.01]), dtype=jnp.float64)
    got = reseed_trajectory_length(cov)
    want = np.pi / 2.0 * np.sqrt(1.0 / 0.01)
    assert abs(got - want) / want < 1e-6, f"T0 {got} != {want}"


def test_reseed_rejects_non_pd():
    with pytest.raises(ValueError, match="PD"):
        reseed_trajectory_length(jnp.asarray(np.diag([1.0, 0.0, 1.0]), dtype=jnp.float64))


# --------------------------------------------------------------------------- kernel order
def test_positions_to_draws_uses_kernel_sorted_order():
    """A stacked position tree with NON-sorted keys must flatten to sorted (kernel) order.

    variances 1e2 / 1e-2 are assigned to names whose sorted order differs from the insertion
    order; the output columns must follow the sorted-key order, not the dict insertion order.
    """
    # insertion (tuple) order: [beta, alpha] with var 1e-2 and 1e2; sorted order: [alpha, beta]
    pos = {"beta": jnp.ones((5, 4), dtype=jnp.float64) * 0.0,
           "alpha": jnp.ones((5, 4), dtype=jnp.float64) * 0.0}
    rng = np.random.default_rng(1)
    pos["beta"] = jnp.asarray(rng.normal(0, 0.1, size=(5, 4)), dtype=jnp.float64)     # var 1e-2
    pos["alpha"] = jnp.asarray(rng.normal(0, 10.0, size=(5, 4)), dtype=jnp.float64)    # var 1e2
    draws = np.asarray(positions_to_draws(pos, 5))
    assert draws.shape == (20, 2), f"shape {draws.shape}"
    var = np.var(draws, axis=0)
    # sorted keys: ['alpha', 'beta'] -> column 0 = alpha (wide), column 1 = beta (narrow)
    assert var[0] > 10.0, f"column 0 should be the wide 'alpha' var, got {var[0]}"
    assert var[1] < 0.1, f"column 1 should be the narrow 'beta' var, got {var[1]}"


# --------------------------------------------------------------------------- two-phase swap
def _correlated_model(seed=3):
    """A 2-D posterior with a real off-diagonal covariance, so a dense PF metric and a diagonal
    ensemble metric are NOT the same matrix (the swap must change the conditioning)."""
    rng = np.random.default_rng(seed)
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0)
        pm.Normal("y", 0.6 * x, 1.0)
    return model


def test_reshare_is_not_a_silent_noop():
    """The metric refresh must take the two-phase path, run both phases, re-seed T, and re-search
    eps to a value different from the phase-A endpoint. A silent no-op (single phase, or a
    re-search that returns the old eps exactly) fails at least one of these."""
    from pfcheeshmc.sampler import PfCheesHMC

    model = _correlated_model()
    sampler = PfCheesHMC(model, seed=1)
    idata = sampler.sample(
        chains=8, draws=50, random_seed=1, progressbar=False,
        pathfinder_kwargs={"draws": 128},
        chees_kwargs={"draws": 40, "metric_refresh": "sample_diag",
                      "metric_refresh_warmup": 20, "metric_refresh_pool": 20},
    )
    a = idata.attrs
    # (a) the two-phase path was taken
    assert a.get("metric_refresh") == "sample_diag", f"metric_refresh attr {a.get('metric_refresh')!r}"
    # (b) both phases ran
    assert a.get("warmup_steps_used") == 40, f"warmup_steps_used {a.get('warmup_steps_used')} != 40"
    assert a.get("metric_refresh_warmup") == 20
    # (c) the trajectory length was re-seeded to a finite positive T
    assert np.isfinite(a.get("metric_refresh_T0")) and a["metric_refresh_T0"] > 0.0
    # (d) the re-search returned a finite positive eps
    eps_r = a.get("metric_refresh_eps_researched")
    assert np.isfinite(eps_r) and eps_r > 0.0, f"re-searched eps {eps_r}"
    # (e) the re-search MOVED eps off the phase-A endpoint (a no-op would leave them exactly equal)
    eps_a = a.get("metric_refresh_phaseA_end_eps")
    assert eps_r != eps_a, f"re-search did not move eps (researched {eps_r} == phaseA {eps_a})"
    # (f) the frozen values and exact gradient counts are still present (A8 contract intact)
    assert np.isfinite(a.get("frozen_step_size")) and a["frozen_step_size"] > 0.0
    assert a.get("warmup_grad_evals_per_chain", -1) > 0
