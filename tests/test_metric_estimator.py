"""The Fisher-divergence metric at a metric update (``chees_kwargs["metric_estimator"]``, Seyboldt, Carlson & Carpenter
2026, as BlackJAX 1.7 estimates it).

What must hold:

1. The estimator is BlackJAX's on the usable chains: rows with a non-finite position or gradient are left out by
   centring the others and zeroing them, which must give BlackJAX's estimate of the usable rows alone.
2. For a Gaussian target the estimate is exact from any sample that spans it: the diagonal one for a diagonal
   covariance (the gradient is -x / s^2 coordinate by coordinate, so Var[x] / Var[grad] = s^4 in the sample itself);
   the low-rank one for a dense covariance whose correlation is outside the estimator's cutoff band.
3. In the warmup, a wrong metric is recovered; the option is recorded in the config; "cov" (the default) is the
   previous code (golden files); bad options raise.
"""
import json
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
import blackjax as bj
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pfcheeshmc import compiled  # noqa: E402
from pfcheeshmc.sampler import sample_pf_chees_hmc  # noqa: E402
from pfcheeshmc.utils import chees_warmup_with_progress  # noqa: E402

SD = np.array([10.0, 1.0, 0.1])


def _gauss_sample(cov, n=512, seed=0):
    x = np.random.default_rng(seed).multivariate_normal(np.zeros(len(cov)), cov, size=n)
    g = -x @ np.linalg.inv(cov)            # grad log p, row by row
    return jnp.asarray(x), jnp.asarray(g)


def _fisher(x, g, old, estimator, shrinkage=0.0, max_rank=10):
    return compiled.ensemble_metric_fisher({"x": x}, {"x": g}, jnp.asarray(old), shrinkage,
                                           estimator=estimator, max_rank=max_rank)


# ------------------------------------------------------------------------------------------------ the estimator
def test_diagonal_is_exact_for_a_diagonal_gaussian():
    x, g = _gauss_sample(np.diag(SD ** 2))
    est, used = _fisher(x, g, jnp.ones(3), "fisher")
    assert bool(used)
    np.testing.assert_allclose(np.asarray(est), SD ** 2, rtol=1e-12)
    dense, used = _fisher(x, g, jnp.eye(3), "fisher")            # a dense old metric gets the diagonal as a matrix
    assert bool(used)
    np.testing.assert_allclose(np.asarray(dense), np.diag(SD ** 2), rtol=1e-12, atol=0)


def _correlated(rho):
    cov = np.diag(SD ** 2)
    cov[0, 1] = cov[1, 0] = rho * SD[0] * SD[1]
    return cov


def test_low_rank_is_exact_for_a_strongly_correlated_gaussian():
    """The Fisher-optimal dense metric of a Gaussian is its covariance, from any sample: with g = -P x (P the precision),
    Cov[g] = P Cov[x] P, and the geometric mean of Cov[x] and Cov[g]^-1 is P^-1. The low-rank estimator keeps a direction
    only if its eigenvalue in the scaled coordinates (sigma^2 = sqrt(Var[x] / Var[g])) is outside [1/2, 2]. At correlation
    0.95 the two correlated directions are at (1 +- 0.95) / sqrt(1 - 0.95^2) = 6.2 and 0.16, and the third coordinate,
    independent, is at 1: so the estimate is the covariance, up to the estimator's gamma regularisation."""
    cov = _correlated(0.95)
    x, g = _gauss_sample(cov, n=1024)
    est, used = _fisher(x, g, jnp.eye(3), "fisher_lowrank")
    assert bool(used)
    np.testing.assert_allclose(np.asarray(est), cov, rtol=1e-6, atol=1e-9)


def test_low_rank_masks_directions_inside_the_band():
    """At correlation 0.3 the scaled eigenvalues are about 1.36 and 0.73, inside [1/2, 2]: both are masked to 1, and the
    estimate is the diagonal sigma^2 = sqrt(Var[x] / Var[g]) of the sample (about SD^2 sqrt(1 - rho^2) for the
    correlated pair: Var[g] -> diag(P))."""
    rho = 0.3
    cov = _correlated(rho)
    x, g = _gauss_sample(cov, n=1024)
    est, _ = _fisher(x, g, jnp.eye(3), "fisher_lowrank")
    est = np.asarray(est)
    sigma2 = np.sqrt(np.var(np.asarray(x), axis=0) / np.var(np.asarray(g), axis=0))
    np.testing.assert_allclose(np.diag(est), sigma2, rtol=1e-12)
    np.testing.assert_allclose(sigma2, SD ** 2 * np.sqrt([1 - rho ** 2, 1 - rho ** 2, 1.0]), rtol=0.05)
    assert np.abs(est - np.diag(np.diag(est))).max() < 1e-12 * np.abs(est).max()


def test_the_estimate_is_blackjax_on_the_usable_rows():
    cov = np.diag(SD ** 2)
    cov[0, 1] = cov[1, 0] = 0.9 * SD[0] * SD[1]
    x, g = _gauss_sample(cov, n=300, seed=4)
    xb, gb = np.asarray(x).copy(), np.asarray(g).copy()
    xb[[3, 77]] = np.nan                     # a non-finite position
    gb[[10, 150, 299], 1] = np.inf           # a non-finite gradient
    keep = np.isfinite(xb).all(1) & np.isfinite(gb).all(1)
    xk, gk = jnp.asarray(xb[keep]), jnp.asarray(gb[keep])

    got, _ = _fisher(jnp.asarray(xb), jnp.asarray(gb), jnp.ones(3), "fisher")
    want = bj.adaptation.metric_estimators.fisher_score_diagonal(xk, gk)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12)

    got, _ = _fisher(jnp.asarray(xb), jnp.asarray(gb), jnp.eye(3), "fisher_lowrank", max_rank=2)
    lr = bj.adaptation.metric_estimators.fisher_score_low_rank(xk, gk, 2)
    want = np.diag(lr.sigma) @ (np.eye(3) + lr.U @ np.diag(lr.lam - 1.0) @ lr.U.T) @ np.diag(lr.sigma)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-9, atol=1e-12)


def test_shrinkage_and_the_guard():
    x, g = _gauss_sample(np.diag(SD ** 2))
    old = jnp.full(3, 4.0)
    est, used = _fisher(x, g, old, "fisher", shrinkage=0.25)
    np.testing.assert_allclose(np.asarray(est), 0.75 * SD ** 2 + 0.25 * 4.0, rtol=1e-12)
    # one usable chain: the old metric, flagged unused
    xb = np.full((8, 3), np.nan)
    xb[0] = 1.0
    est, used = _fisher(jnp.asarray(xb), jnp.ones((8, 3)), old, "fisher")
    assert not bool(used) and np.array_equal(np.asarray(est), np.asarray(old))
    # zero gradient variance (a flat direction): sigma^2 = sqrt(var / 1e-10) is finite and positive, so it is used;
    # a non-finite estimate is not
    est, used = _fisher(x, jnp.zeros_like(g).at[:, 0].set(jnp.inf), old, "fisher")
    assert not bool(used) and np.array_equal(np.asarray(est), np.asarray(old))


# ---------------------------------------------------------------------------------------------------- the warmup
def _warmup(estimator, dense, chains=256, num_steps=400):
    cov = np.diag(SD ** 2)
    if dense:
        cov[0, 1] = cov[1, 0] = 0.95 * SD[0] * SD[1]
    prec = jnp.asarray(np.linalg.inv(cov))
    ld = lambda p: -0.5 * p["x"] @ prec @ p["x"]
    positions = {"x": jax.random.normal(jax.random.key(3), (chains, 3))}
    imm = jnp.eye(3) if dense else jnp.ones(3)
    (_, params), _ = chees_warmup_with_progress(
        ld, jax.random.key(0), positions, jnp.float64(0.05), 0.05, num_steps, imm, progressbar=False,
        metric_updates=(0.25, 0.5, 0.75), metric_estimator=estimator)
    return cov, params


@pytest.mark.parametrize("estimator, dense", [("fisher", False), ("fisher", True), ("fisher_lowrank", True)])
def test_a_wrong_metric_is_recovered(estimator, dense):
    """Started with the identity metric, 10,000 times wrong in variance. On a Gaussian the estimate is exact from the
    chains' positions at the update, stationary or not, for the diagonal estimator of a diagonal Gaussian and the
    low-rank one of a correlated Gaussian (test_low_rank_is_exact_for_a_strongly_correlated_gaussian); what is left is
    the shrinkage towards the old metric (5 / 261 per update, three updates). The diagonal estimator of a correlated
    Gaussian is a sample quantity: sqrt(Var[x] / Var[g]) -> sqrt(diag(cov) / diag(P)), within Monte Carlo error of
    256 chains."""
    cov, params = _warmup(estimator, dense)
    assert all(u["used"] for u in params["metric_updates"])
    m = np.asarray(params["inverse_mass_matrix"])
    if estimator == "fisher_lowrank":
        np.testing.assert_allclose(m, cov, rtol=1e-3, atol=1e-6)
    else:
        want = np.sqrt(np.diag(cov) / np.diag(np.linalg.inv(cov)))
        np.testing.assert_allclose(np.diag(m) if m.ndim == 2 else m, want, rtol=1e-3 if not dense else 0.1)


def test_recorded_in_config_and_default_absent():
    with pm.Model() as model:
        pm.Normal("x", 0.0, SD, shape=3)
    kw = dict(draws=5, chains=16, random_seed=1, progressbar=False)
    idata = sample_pf_chees_hmc(model, **kw, chees_kwargs={"draws": 20, "metric_updates": (0.5,),
                                                           "metric_estimator": "fisher_lowrank", "metric_max_rank": 2})
    cfg = json.loads(idata.attrs["config"])
    assert cfg["metric_estimator"] == "fisher_lowrank" and cfg["metric_max_rank"] == 2
    idata = sample_pf_chees_hmc(model, **kw, chees_kwargs={"draws": 20, "metric_updates": (0.5,)})
    cfg = json.loads(idata.attrs["config"])
    assert "metric_estimator" not in cfg and "metric_max_rank" not in cfg


@pytest.mark.parametrize("kw, match", [
    ({"metric_updates": (0.5,), "metric_estimator": "nutpie"}, "metric_estimator must be one of"),
    ({"metric_estimator": "fisher"}, "needs metric_updates"),
    ({"metric_updates": (0.5,), "metric_estimator": "fisher_lowrank", "metric_max_rank": 0}, "metric_max_rank"),
])
def test_bad_options_raise(kw, match):
    ld = lambda p: -0.5 * jnp.sum(p["x"] ** 2)
    with pytest.raises(ValueError, match=match):
        chees_warmup_with_progress(ld, jax.random.key(0), {"x": jnp.zeros((8, 3))}, jnp.float64(0.1), 0.05, 20,
                                   jnp.eye(3), progressbar=False, **kw)


def test_low_rank_needs_a_dense_metric():
    ld = lambda p: -0.5 * jnp.sum(p["x"] ** 2)
    with pytest.raises(ValueError, match="needs a dense metric"):
        chees_warmup_with_progress(ld, jax.random.key(0), {"x": jnp.zeros((8, 3))}, jnp.float64(0.1), 0.05, 20,
                                   jnp.ones(3), progressbar=False, metric_updates=(0.5,),
                                   metric_estimator="fisher_lowrank")
