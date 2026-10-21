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


def _spd_pool(d=6, n=20_000, seed=7):
    """A random SPD covariance and N draws from the matching Gaussian, in kernel order."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) / np.sqrt(d)
    sigma = A @ A.T + 0.25 * np.eye(d)                     # SPD by construction
    chol = np.linalg.cholesky(sigma)
    draws = rng.normal(size=(n, d)) @ chol.T
    return sigma, jnp.asarray(draws, dtype=jnp.float64)


def test_reseed_is_invariant_to_coordinate_units():
    """R2-WP1 test 1 (plan Part B 1.1): rescaling a coordinate must not move T0.

    The run-1 formula ``(pi/2)/sqrt(lambda_min(B))`` is the precision of the narrowest coordinate in
    MODEL units, so scaling coordinate 0 by 10 and coordinate 3 by 0.01 moved T0 by orders of
    magnitude -- that is the defect that produced frozen L 213-283. The whitened quantity
    ``lambda_max(L^-1 S L^-T)`` is invariant, because B and S scale together.

    ridge=0 is deliberate: the default 1e-6 ridge is an absolute floor, so on a coordinate whose
    variance has been scaled to ~1e-4 it is a ~1% perturbation and would mask the invariance being
    tested. The clip lives in the sampler, not here, for the same reason.
    """
    _, draws = _spd_pool(d=6, n=20_000)
    b = np.asarray(estimate_covariance("sample_diag", draws, ridge=0.0))
    t0 = EM.reseed_trajectory_length(b, draws)

    scale = np.ones(6)
    scale[0] = 10.0
    scale[3] = 0.01
    draws_r = jnp.asarray(np.asarray(draws) * scale, dtype=jnp.float64)
    b_r = np.asarray(estimate_covariance("sample_diag", draws_r, ridge=0.0))
    t0_r = EM.reseed_trajectory_length(b_r, draws_r)

    assert t0 > 0.0 and np.isfinite(t0_r)
    rel = abs(t0_r - t0) / t0
    assert rel < 1e-9, f"T0 moved with coordinate units: {t0:.12g} -> {t0_r:.12g} (rel {rel:.3e})"


def test_reseed_is_pi_over_two_when_metric_equals_pool_covariance():
    """R2-WP1 test 2: with B = S exactly, the whitened matrix is I, so T0 = pi/2.

    Uses the dense unshrunk estimator path (``dense_shrunk`` with shrink=0, ridge=0) so B is the
    sample covariance itself. Run-1's formula returned pi/2/sqrt(lambda_min(S)) here, which is not
    pi/2 unless the pool happens to have unit variance in every coordinate -- so the pool below has
    variances spanning ~1e2 to make the old formula fail loudly rather than by rounding.
    """
    rng = np.random.default_rng(11)
    sd = np.array([3.0, 7.0, 0.5, 20.0, 1.3])
    draws = jnp.asarray(rng.normal(size=(20_000, 5)) * sd, dtype=jnp.float64)
    b = np.asarray(estimate_covariance("dense_shrunk", draws, shrink=0.0, ridge=0.0))
    t0 = EM.reseed_trajectory_length(b, draws)
    want = np.pi / 2.0
    rel = abs(t0 - want) / want
    assert rel < 1e-8, f"T0 {t0:.12g} != pi/2 {want:.12g} (rel {rel:.3e})"


def test_reseed_tracks_pool_correlation_against_a_diagonal_metric():
    """R2-WP1 test 3: d=2, B = diag(S) -> whitened matrix is the correlation matrix.

    Its largest eigenvalue is 1 + |rho_hat|, so T0 = (pi/2) * sqrt(1 + |rho_hat|) exactly. This pins
    the part of the formula the unit-invariance test cannot see: correlation the metric does not
    carry must still be paid for in the re-seed.
    """
    rng = np.random.default_rng(13)
    n = 60_000
    rho = 0.55
    z = rng.normal(size=(n, 2))
    x = np.column_stack([z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]]) * np.array([4.0, 9.0])
    draws = jnp.asarray(x, dtype=jnp.float64)
    b = np.asarray(estimate_covariance("sample_diag", draws, ridge=0.0))     # diag(S)
    s = np.cov(np.asarray(draws), rowvar=False, ddof=1)
    rho_hat = s[0, 1] / np.sqrt(s[0, 0] * s[1, 1])

    t0 = EM.reseed_trajectory_length(b, draws)
    want = np.pi / 2.0 * np.sqrt(1.0 + abs(rho_hat))
    rel = abs(t0 - want) / want
    assert rel < 1e-8, f"T0 {t0:.12g} != (pi/2)sqrt(1+|rho|) {want:.12g} (rho_hat {rho_hat:.6f}, rel {rel:.3e})"


def test_reseed_matches_generalized_eigenproblem_for_a_dense_unrelated_metric():
    """Beyond the plan's tests 1-3 (B = S, B = diag(S)): a dense B that is neither.

    lambda_max(L^-1 S L^-T) is the largest generalized eigenvalue of S v = lambda B v, so SciPy's
    ``eigh(S, B)`` is an independent route to the same number.
    """
    from scipy.linalg import eigh

    rng = np.random.default_rng(17)
    d = 7
    a = rng.normal(size=(d, d))
    b = a @ a.T + 0.5 * np.eye(d)                                   # dense SPD metric, not S
    draws = np.random.default_rng(18).normal(size=(5000, d)) @ rng.normal(size=(d, d))
    s_mat = np.cov(draws, rowvar=False, ddof=1)
    want = np.pi / 2.0 * np.sqrt(eigh(s_mat, b, eigvals_only=True).max())
    got = EM.reseed_trajectory_length(jnp.asarray(b, dtype=jnp.float64),
                                      jnp.asarray(draws, dtype=jnp.float64))
    assert abs(got - want) / want < 1e-9, f"T0 {got:.12g} != generalized-eigh {want:.12g}"


def test_reseed_rejects_non_pd():
    """R2-WP1 test 4 (kept): a non-PD metric must raise, not return NaN or a negative sqrt."""
    draws = jnp.asarray(np.random.default_rng(5).normal(size=(50, 2)), dtype=jnp.float64)
    with pytest.raises(ValueError, match="PD"):
        EM.reseed_trajectory_length(jnp.asarray(np.diag([1.0, 0.0, 1.0]), dtype=jnp.float64),
                                    jnp.asarray(np.random.default_rng(5).normal(size=(50, 3)),
                                                dtype=jnp.float64))
    with pytest.raises(ValueError, match="PD"):
        EM.reseed_trajectory_length(jnp.asarray(np.diag([1.0, 0.0]), dtype=jnp.float64), draws)


def test_reseed_signature_is_metric_and_draws():
    """R2-WP1: the signature changed to (cov_B, draws). A one-argument call must not work.

    Pinned because the units defect was invisible at the call site -- a caller could not tell that
    the metric alone was not enough information.
    """
    import inspect
    sig = inspect.signature(EM.reseed_trajectory_length)
    assert list(sig.parameters) == ["cov_B", "draws"], f"signature is {sig}"


def test_reseed_rejects_shape_mismatch():
    draws = jnp.asarray(np.random.default_rng(9).normal(size=(40, 3)), dtype=jnp.float64)
    with pytest.raises(ValueError, match="draws must be"):
        EM.reseed_trajectory_length(jnp.asarray(np.eye(2), dtype=jnp.float64), draws)


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
    # (c2) R2-WP1: the unit-invariant quantity reaches the attrs, and T0 is exactly its (pi/2)sqrt
    #      image clipped to [eps_B, max_leapfrog_steps * eps_B] (max_leapfrog_steps defaults to 1000).
    lmax = a.get("metric_refresh_lmax_whitened")
    assert lmax is not None and np.isfinite(lmax) and lmax > 0.0, f"lmax_whitened attr {lmax!r}"
    eps_b = a["metric_refresh_eps_researched"]
    want_t0 = min(max(np.pi / 2.0 * np.sqrt(lmax), eps_b), 1000 * eps_b)
    assert abs(a["metric_refresh_T0"] - want_t0) <= 1e-12 * want_t0, (a["metric_refresh_T0"], want_t0)
    # (d) the re-search returned a finite positive eps
    eps_r = a.get("metric_refresh_eps_researched")
    assert np.isfinite(eps_r) and eps_r > 0.0, f"re-searched eps {eps_r}"
    # (e) the re-search MOVED eps off the phase-A endpoint (a no-op would leave them exactly equal)
    eps_a = a.get("metric_refresh_phaseA_end_eps")
    assert eps_r != eps_a, f"re-search did not move eps (researched {eps_r} == phaseA {eps_a})"
    # (f) the frozen values and exact gradient counts are still present (A8 contract intact)
    assert np.isfinite(a.get("frozen_step_size")) and a["frozen_step_size"] > 0.0
    assert a.get("warmup_grad_evals_per_chain", -1) > 0
