"""R2-WP2: closed-form Pathfinder covariance, pool sizing, start scaling, and the config record.

Pathfinder makes ONE L-BFGS fit, and `bj.vi.pathfinder.sample` draws from a Gaussian whose covariance
is exactly `diag(alpha) + beta gamma beta^T` (`lbfgs_inverse_hessian_formula_1`). The Monte-Carlo
pool covariance the sampler used to build is a noisy estimate of that matrix. These tests pin:

1. the algebra (formula_1 == the dense expansion of the low-rank adapter, so the existing
   `lbfgs_lowrank` arm computes the same matrix);
2. that formula_1 is the covariance of Pathfinder's own draws, and `_pathfinder_mean` their mean;
3. G-ORDER for `pf_closed` (kernel coordinate order, both orientations) with teeth;
4. the pool rows each `imm_source` draws;
5. `pf_start_scale`;
6. `idata.attrs["config"]`: resolved values, not requests.

Models come from `tools/r1_bias.build_model`, which undoes `bench_ladder`'s global sampling stubs.
"""
import functools
import json
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest
from jax.flatten_util import ravel_pytree

import blackjax as bj
import pytensor

# This repo lives under a path with a space; PyTensor's default blas__ldflags carries the env prefix as
# an unquoted -Wl,-rpath token that its linker rejects for any model with a dot product (logit_1000x25).
# tools/bench_ladder.py works around it with PYTENSOR_FLAGS, but that is read at pytensor import, which
# pytest has already done by the time build_model imports bench_ladder. Set the same flags on the
# live config (numpy still supplies BLAS at runtime).
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas -lm"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import r1_bias as rb  # noqa: E402

import pfcheeshmc.sampler as sampler_mod  # noqa: E402
from pfcheeshmc import compiled  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC, default_pool_cap, resolve_pool_cap  # noqa: E402

# Shared with the run-1 G-ORDER file: the non-sorted-names fixtures and the mixing instrument.
from test_metric_orientation import (  # noqa: E402,F401  (fixtures are used by name)
    PF_ORDER_VAR, _mixing_report, anisotropic_model, odd_order_model)

MODELS = ["8schools", "gev", "logit_1000x25", "radon_noncentered"]
SMALL_MODELS = ["8schools", "gev"]
formula_1 = bj.optimizers.lbfgs.lbfgs_inverse_hessian_formula_1


# ------------------------------------------------------------------------------ fits
@functools.lru_cache(maxsize=None)
def _fit(model_name):
    """(sampler, Pathfinder state, metric the sampler handed on, kernel permutation).

    One `_run_pathfinder` call per model, with the real fit, `imm_source="pf_closed"` and the default
    kernel ordering. The state is captured by spying on `approximate`, so what is tested is the fit the
    sampler actually made.
    """
    model = rb.build_model(model_name)
    s = PfCheesHMC(model, seed=1)
    box = {}
    real = compiled.pathfinder_fit

    def spy(*a, **k):
        box["state"] = out = real(*a, **k)
        return out

    compiled.pathfinder_fit = spy
    try:
        pos, imm = s._run_pathfinder(chains=8, draws=32, seed=1, imm_source="pf_closed", pm_tune=50)
    finally:
        compiled.pathfinder_fit = real
    st = box["state"]
    draws, _ = bj.vi.pathfinder.sample(jax.random.key(0), st, 4)
    return s, st, np.asarray(imm), np.asarray(s._kernel_permutation(draws))


# -------------------------------------------------------------------- 1. exact algebra
@pytest.mark.parametrize("model_name", MODELS)
def test_formula_1_equals_the_dense_low_rank_expansion(model_name):
    """formula_1(alpha, beta, gamma) == D (I + U (lam - 1) U^T) D to 1e-8 (Frobenius, relative).

    The right-hand side is exactly what the `lbfgs_lowrank` arm hands the kernel (sampler.py builds
    it from `lbfgs_inverse_hessian_to_low_rank_metric`), so this settles what that arm is: the same
    closed-form Pathfinder covariance, expressed through an eigendecomposition of its low-rank part.
    """
    _, st, imm, _ = _fit(model_name)
    A = formula_1(st.alpha, st.beta, st.gamma)
    lr = bj.mcmc.metrics.lbfgs_inverse_hessian_to_low_rank_metric(st.alpha, st.beta, st.gamma)
    D = jnp.diag(lr.sigma)
    B = D @ (jnp.eye(lr.sigma.shape[0]) + lr.U @ jnp.diag(lr.lam - 1.0) @ lr.U.T) @ D
    rel = float(jnp.linalg.norm(A - B) / jnp.linalg.norm(A))
    assert rel < 1e-8, f"{model_name}: ||formula_1 - low-rank expansion||_F / ||formula_1||_F = {rel:.3e}"
    assert float(jnp.linalg.eigvalsh(A).min()) > 0.0, f"{model_name}: formula_1 is not PD"


@pytest.mark.parametrize("model_name", MODELS)
def test_pf_closed_hands_the_kernel_formula_1_in_kernel_order(model_name):
    """The metric the sampler returns is formula_1 of the captured fit, permuted tuple -> dict order.

    No ridge, no Monte-Carlo estimate, orientation "covariance" (the default): the returned array is
    exactly P formula_1 P^T. GEV and radon have non-sorted value-var names, so the permutation is not
    the identity there and a missing reorder cannot pass.
    """
    s, st, imm, perm = _fit(model_name)
    A = np.asarray(formula_1(st.alpha, st.beta, st.gamma))
    np.testing.assert_allclose(imm, A[perm][:, perm], rtol=1e-12, atol=1e-14)
    if model_name in ("gev", "radon_noncentered"):
        assert perm.tolist() != list(range(len(perm))), "fixture lost its point: names already sorted"


# ------------------------------------------------------------------ 2. MC consistency
@pytest.mark.parametrize("model_name", SMALL_MODELS)
def test_formula_1_is_the_covariance_of_pathfinders_own_draws(model_name):
    """N = 50,000 draws from `bj.vi.pathfinder.sample`: sample covariance ~ formula_1, mean ~ mu.

    Bound for the relative Frobenius error is the plan's 5 sqrt(d/N) (1 + tr(S)/||S||_F). For a
    Gaussian the exact value is E||S_hat - S||_F^2 = (tr(S)^2 + ||S||_F^2)/N; the failure message
    quotes the ratio to it. The mean check is the one that validates `_pathfinder_mean`, which
    `pf_start_scale` centres on.
    """
    s, st, _, _ = _fit(model_name)
    N = 50_000
    draws, _ = bj.vi.pathfinder.sample(jax.random.key(7), st, N)
    flat = np.asarray(jax.vmap(lambda t: ravel_pytree(t)[0])(draws))
    A = np.asarray(formula_1(st.alpha, st.beta, st.gamma))
    d = A.shape[0]
    fro = np.linalg.norm(A)
    rel = np.linalg.norm(np.cov(flat, rowvar=False) - A) / fro
    bound = 5.0 * np.sqrt(d / N) * (1.0 + np.trace(A) / fro)
    exact_se = np.sqrt((np.trace(A) ** 2 + fro ** 2) / N) / fro
    assert rel < bound, (f"{model_name}: rel Frobenius error {rel:.3e} >= bound {bound:.3e} "
                         f"({rel / exact_se:.2f}x the exact Wishart standard error)")
    mu = np.asarray(ravel_pytree(PfCheesHMC._pathfinder_mean(st))[0])
    z = (flat.mean(axis=0) - mu) / np.sqrt(np.diag(A) / N)
    assert np.abs(z).max() < 5.0, f"{model_name}: draw mean is {np.abs(z).max():.1f} SE from mu"
    # and mu is the plan's `position + formula_1 @ grad`
    pos = np.asarray(ravel_pytree(st.position)[0])
    grad = np.asarray(ravel_pytree(st.grad_position)[0])
    np.testing.assert_allclose(mu, pos + A @ grad, rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------------ 3. G-ORDER
def _pf_order_covariance():
    """A dense SPD covariance in Pathfinder order [z, a0, a1, a2] with a 10^4 spread of variances."""
    D = np.diag([1e1, 1e-1, 2e-1, 3e-1])
    M = np.eye(4) + 0.3 * (np.ones((4, 4)) - np.eye(4))
    return D @ M @ D


def _assert_kernel_order(model, monkeypatch, orientation):
    """Inject a known covariance as formula_1's output and check the array the kernel is handed."""
    C = _pf_order_covariance()
    monkeypatch.setattr(bj.optimizers.lbfgs, "lbfgs_inverse_hessian_formula_1",
                        lambda alpha, beta, gamma: jnp.asarray(C))
    s = PfCheesHMC(model, seed=0)
    imm = np.asarray(s._run_pathfinder(chains=8, draws=16, imm_source="pf_closed",
                                       imm_orientation=orientation)[1])
    perm = np.array([1, 2, 3, 0])           # kernel order is sorted names: a0, a1, a2, z
    ref = C if orientation == "covariance" else np.linalg.inv(C)
    np.testing.assert_allclose(imm, ref[perm][:, perm], rtol=1e-8, atol=0.0,
                               err_msg=f"pf_closed/{orientation}: not in kernel order")


@pytest.mark.parametrize("orientation", ["covariance", "precision"])
def test_pf_closed_g_order_on_the_non_sorted_names_fixture(odd_order_model, monkeypatch, orientation):
    """G-ORDER: the closed-form covariance passes the same choke point and orientation step as mc."""
    _assert_kernel_order(odd_order_model, monkeypatch, orientation)


@pytest.mark.parametrize("orientation", ["covariance", "precision"])
def test_g_order_check_fails_when_the_choke_point_is_the_identity(odd_order_model, monkeypatch, orientation):
    """Teeth: with `_metric_to_kernel_order` mutated to identity the check above must go red."""
    monkeypatch.setattr(PfCheesHMC, "_metric_to_kernel_order", lambda self, imm, perm: imm)
    with pytest.raises(AssertionError, match="not in kernel order"):
        _assert_kernel_order(odd_order_model, monkeypatch, orientation)


def test_pf_closed_mixes_the_anisotropic_target(anisotropic_model):
    """The run-1 anisotropic fixture, end to end with pf_closed: R-hat < 1.01 and ESS per gradient > 1."""
    _, max_rhat, min_ess, per, steps, *_ = _mixing_report(anisotropic_model, imm_source="pf_closed")
    assert max_rhat < 1.01, f"max R-hat {max_rhat:.4f}, per-variable ESS {per}"
    assert min_ess / (steps * 1000) > 1.0, f"ESS per gradient {min_ess / (steps * 1000):.3f}"


def test_pf_closed_refuses_a_non_positive_definite_covariance(odd_order_model, monkeypatch):
    bad = jnp.diag(jnp.array([1.0, 1.0, 1.0, -1e-3]))
    monkeypatch.setattr(bj.optimizers.lbfgs, "lbfgs_inverse_hessian_formula_1", lambda *a: bad)
    with pytest.raises(ValueError, match="not positive definite"):
        PfCheesHMC(odd_order_model, seed=0)._run_pathfinder(chains=8, draws=16, imm_source="pf_closed")


# -------------------------------------------------------------------------- 4. pool rows
def _wide_model(d=40):
    with pm.Model() as model:
        pm.Normal("x", shape=(d,))
    return model


def _spy_pool_rows(monkeypatch):
    seen = []
    real = compiled.pathfinder_draw

    def spy(sample, rng_key, state, num_samples):
        seen.append(int(num_samples))
        return real(sample, rng_key, state, num_samples)

    monkeypatch.setattr(compiled, "pathfinder_draw", spy)
    return seen


def test_pool_rows_at_c1024_pf_closed_is_4c_and_mc_is_max_4c_4d2(monkeypatch):
    """d = 40 makes 4d^2 = 6,400 exceed 4C = 4,096, so the two rules cannot be confused."""
    s = PfCheesHMC(_wide_model(40), seed=1)
    seen = _spy_pool_rows(monkeypatch)
    for source, expected in (("pf_closed", 4096), ("lbfgs_lowrank", 4096), ("mc", 6400)):
        seen.clear()
        s._run_pathfinder(chains=1024, draws=4000, seed=1, imm_source=source, pm_tune=50)
        assert seen == [expected], f"{source}: pool rows {seen}, expected {expected}"
        assert s._resolved["pathfinder"]["pf_pool_rows"] == expected
    assert default_pool_cap(1024, 40) == 6400


@pytest.mark.parametrize("source", ["mc", "pf_closed", "lbfgs_lowrank", "fisher_lowrank"])
def test_pool_cap_zero_and_explicit_integers_behave_as_before(monkeypatch, source):
    s = PfCheesHMC(_wide_model(4), seed=1)
    seen = _spy_pool_rows(monkeypatch)
    kw = dict(chains=8, draws=50, seed=1, imm_source=source, pm_tune=50, max_rank=1)
    s._run_pathfinder(pf_pool_cap=0, **kw)
    s._run_pathfinder(pf_pool_cap=100, **kw)
    assert seen == [8 * 50, 100], f"{source}: {seen}"


def test_resolve_pool_cap_table():
    d, c = 40, 1024
    assert resolve_pool_cap(None, c, d, "mc") == max(4 * c, 4 * d * d)
    assert resolve_pool_cap(None, c, d) == default_pool_cap(c, d)          # mc is the default
    for source in ("pf_closed", "lbfgs_lowrank", "fisher_lowrank"):
        assert resolve_pool_cap(None, c, 500, source) == 4 * c             # no d^2 term off the mc path
    for source in ("mc", "pf_closed"):
        assert resolve_pool_cap(0, c, d, source) is None
        assert resolve_pool_cap(5000, c, d, source) == 5000
        with pytest.raises(ValueError, match="pf_pool_cap"):
            resolve_pool_cap(c - 1, c, d, source)


# -------------------------------------------------------------------- 5. pf_start_scale
def _starts_and_mu(model, **kw):
    """(starts as a flat (C, d) array in Pathfinder order, mu_pf from the captured fit, metric)."""
    s = PfCheesHMC(model, seed=3)
    box = {}
    real = compiled.pathfinder_fit

    def spy(*a, **k):
        box["state"] = out = real(*a, **k)
        return out

    compiled.pathfinder_fit = spy
    try:
        pos, imm = s._run_pathfinder(chains=16, draws=50, seed=3, pm_tune=50, **kw)
    finally:
        compiled.pathfinder_fit = real
    st = box["state"]
    flat = np.stack([np.concatenate([np.ravel(np.asarray(pos[n])[c]) for n in s.rvs])
                     for c in range(16)])
    # the plan's mu: position + formula_1 @ grad, from the library formula (not _pathfinder_mean)
    A = np.asarray(formula_1(st.alpha, st.beta, st.gamma))
    mu = np.asarray(ravel_pytree(st.position)[0]) + A @ np.asarray(ravel_pytree(st.grad_position)[0])
    return flat, mu, np.asarray(imm)


@pytest.fixture(scope="module")
def scale_model():
    with pm.Model() as model:
        pm.Normal("a", 1.0, 2.0)
        pm.Normal("b", 0.0, 0.1, shape=3)
    return model


@pytest.mark.parametrize("source", ["mc", "pf_closed"])
def test_start_scale_one_is_bit_identical_and_two_doubles_the_dispersion(scale_model, source):
    x_default, mu, imm_default = _starts_and_mu(scale_model, imm_source=source)
    x_one, _, imm_one = _starts_and_mu(scale_model, imm_source=source, pf_start_scale=1.0)
    assert np.array_equal(x_default, x_one), "pf_start_scale=1.0 changed the starts"
    assert np.array_equal(imm_default, imm_one)
    x_two, mu2, imm_two = _starts_and_mu(scale_model, imm_source=source, pf_start_scale=2.0)
    np.testing.assert_allclose(mu2, mu, rtol=0, atol=0)
    np.testing.assert_allclose(x_two - mu, 2.0 * (x_one - mu), rtol=1e-12, atol=1e-12)
    sd = lambda x: np.sqrt(np.mean((x - mu) ** 2, axis=0))                 # sd of the starts about mu_pf
    np.testing.assert_allclose(sd(x_two), 2.0 * sd(x_one), rtol=1e-12)
    assert np.array_equal(imm_two, imm_default), "pf_start_scale must not touch the covariance"


def test_start_scale_applies_to_the_adapt_diag_starts_too(scale_model):
    kw = dict(init_source="adapt_diag", pm_init="jitter+adapt_diag", pm_chains=1)
    x1, mu, _ = _starts_and_mu(scale_model, **kw)
    x3, _, _ = _starts_and_mu(scale_model, pf_start_scale=3.0, **kw)
    np.testing.assert_allclose(x3 - mu, 3.0 * (x1 - mu), rtol=1e-12, atol=1e-12)


def test_start_scale_validation_and_nonfinite_guard(scale_model, monkeypatch):
    s = PfCheesHMC(scale_model, seed=3)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="pf_start_scale"):
            s._run_pathfinder(chains=4, draws=10, pm_tune=50, pf_start_scale=bad)
    real_mask = PfCheesHMC._usable_mask
    # only the scaled starts (4 rows) are declared unusable; the 1-row shared init point stays fine
    monkeypatch.setattr(PfCheesHMC, "_usable_mask", lambda self, init: (
        real_mask(self, init) if next(iter(init.values())).shape[0] == 1
        else jnp.zeros(next(iter(init.values())).shape[0], dtype=bool)))
    with pytest.raises(ValueError, match="non-finite"):
        s._run_pathfinder(chains=4, draws=10, pm_tune=50, pf_start_scale=2.0)


# ------------------------------------------------------------------------ 6. config record
@pytest.fixture(scope="module")
def tiny_run():
    """Two real runs on a 3-d Gaussian: all defaults, and every warmup knob moved off its default.

    `chees_warmup_with_progress` and `estimate_covariance` are spied so the record can be compared
    with what was actually applied.
    """
    model = _wide_model(3)
    calls = {"warmup": [], "estimate": []}
    real_w, real_e = sampler_mod.chees_warmup_with_progress, sampler_mod.estimate_covariance

    def spy_w(*a, **k):
        calls["warmup"].append(dict(k))
        return real_w(*a, **k)

    def spy_e(*a, **k):
        calls["estimate"].append(dict(k))
        return real_e(*a, **k)

    sampler_mod.chees_warmup_with_progress, sampler_mod.estimate_covariance = spy_w, spy_e
    try:
        base = sampler_mod.sample_pf_chees_hmc(
            model, draws=12, chains=4, random_seed=5, progressbar=False,
            chees_kwargs={"draws": 20}, pathfinder_kwargs={"draws": 50})
        n_base = len(calls["warmup"])
        moved = sampler_mod.sample_pf_chees_hmc(
            model, draws=12, chains=4, random_seed=5, progressbar=False,
            chees_kwargs={"draws": 20, "iterate": "raw", "adam_lr": 0.02, "target_accept": 0.7,
                          "max_leapfrog_steps": 50, "metric_refresh": "sample_diag",
                          "metric_refresh_warmup": 8},
            step_size_tune_kwargs={"eps0_probes": 4},
            pathfinder_kwargs={"draws": 50, "imm_source": "pf_closed", "imm_orientation": "precision",
                               "pf_start_scale": 0.5, "pf_pool_cap": 20})
    finally:
        sampler_mod.chees_warmup_with_progress, sampler_mod.estimate_covariance = real_w, real_e
    return (base, moved, calls["warmup"][:n_base], calls["warmup"][n_base:], calls["estimate"])


ALWAYS_SET = ["imm_source", "imm_orientation", "imm_order", "ridge", "pf_draws", "pf_pool_cap",
              "pf_pool_rows", "pf_start_scale", "init_source", "init_pick", "pm_chains", "pm_init",
              "pm_tune", "score_chunk", "eps0_probes", "eps0", "num_warmup", "warmup_steps_used",
              "iterate", "adam_lr", "target_accept", "max_leapfrog_steps", "metric_refresh", "chains",
              "num_draws", "frozen_step_size", "frozen_L_nominal", "seed", "stage_seeds",
              "jax_enable_x64"]
NULL_WHEN_NOT_USED = ["superchains", "warmup_stop"]
REFRESH_ONLY = ["metric_refresh_warmup", "metric_refresh_pool", "metric_refresh_reseed",
                "metric_refresh_ridge", "metric_refresh_shrink", "metric_refresh_rank", "metric_refresh_eps_probes"]


def test_config_is_a_json_string_with_every_resolved_default_filled_in(tiny_run):
    base, _, base_warm, _, _ = tiny_run
    cfg = json.loads(base.attrs["config"])
    for k in ALWAYS_SET:
        assert cfg.get(k) is not None, f"config[{k!r}] is {cfg.get(k)!r}: a default was not resolved"
    assert set(NULL_WHEN_NOT_USED + REFRESH_ONLY) <= set(cfg)
    assert all(cfg[k] is None for k in NULL_WHEN_NOT_USED + REFRESH_ONLY)
    assert cfg["metric_refresh"] == "none"
    # the sampler defaults, as applied
    assert (cfg["imm_source"], cfg["imm_orientation"], cfg["imm_order"]) == ("mc", "covariance", "kernel")
    assert cfg["iterate"] == "ma" and cfg["adam_lr"] == 0.05 and cfg["max_leapfrog_steps"] == 1000
    assert cfg["target_accept"] == bj.adaptation.chees_adaptation.OPTIMAL_TARGET_ACCEPTANCE_RATE
    assert cfg["pf_start_scale"] == 1.0 and cfg["ridge"] == 1e-6 and cfg["eps0_probes"] == 32
    # pool: d = 3, C = 4 -> mc default max(4C, 4d^2) = 36, which min(C * draws = 200, .) leaves at 36
    assert cfg["pf_pool_cap"] == cfg["pf_pool_rows"] == default_pool_cap(4, 3) == 36
    assert (cfg["chains"], cfg["num_draws"], cfg["num_warmup"], cfg["warmup_steps_used"]) == (4, 12, 20, 20)
    assert cfg["frozen_step_size"] == base.attrs["frozen_step_size"]
    assert cfg["frozen_L_nominal"] == base.attrs["frozen_L_nominal"]
    # and what chees_warmup_with_progress was actually called with
    (w,) = base_warm
    assert w["iterate"] == cfg["iterate"] and w["max_leapfrog_steps"] == cfg["max_leapfrog_steps"]
    assert w["target_acceptance_rate"] == cfg["target_accept"]


def test_config_records_moved_knobs_and_the_refresh_settings_as_applied(tiny_run):
    _, moved, _, moved_warm, estimate_calls = tiny_run
    cfg = json.loads(moved.attrs["config"])
    assert (cfg["imm_source"], cfg["imm_orientation"], cfg["ridge"]) == ("pf_closed", "precision", 0.0)
    assert cfg["pf_start_scale"] == 0.5 and cfg["pf_pool_cap"] == cfg["pf_pool_rows"] == 20
    assert cfg["iterate"] == "raw" and cfg["adam_lr"] == 0.02 and cfg["target_accept"] == 0.7
    assert cfg["max_leapfrog_steps"] == 50 and cfg["eps0_probes"] == 4
    assert cfg["metric_refresh"] == "sample_diag" and cfg["metric_refresh_warmup"] == 8
    assert cfg["metric_refresh_reseed"] is True and cfg["metric_refresh_eps_probes"] == 32
    assert all(cfg[k] is not None for k in REFRESH_ONLY)
    # both phases of the two-phase warmup were run with the recorded iterate and L cap
    assert len(moved_warm) == 2
    assert all(w["iterate"] == "raw" and w["max_leapfrog_steps"] == 50 for w in moved_warm)
    # the refresh-estimator defaults in the record are the ones estimate_covariance really received
    (e,) = estimate_calls
    assert (e["ridge"], e["shrink"], e["max_rank"]) == (
        cfg["metric_refresh_ridge"], cfg["metric_refresh_shrink"], cfg["metric_refresh_rank"])


def test_iterate_raw_reaches_the_single_phase_warmup(tiny_run):
    """Regression for the commit that made `_run_chees` forward `iterate` when no refresh runs."""
    model = _wide_model(3)
    seen = []
    real = sampler_mod.chees_warmup_with_progress
    sampler_mod.chees_warmup_with_progress = lambda *a, **k: (seen.append(k.get("iterate")), real(*a, **k))[1]
    try:
        idata = sampler_mod.sample_pf_chees_hmc(
            model, draws=5, chains=4, random_seed=5, progressbar=False,
            chees_kwargs={"draws": 10, "iterate": "raw"}, pathfinder_kwargs={"draws": 50})
    finally:
        sampler_mod.chees_warmup_with_progress = real
    assert seen == ["raw"] and json.loads(idata.attrs["config"])["iterate"] == "raw"
