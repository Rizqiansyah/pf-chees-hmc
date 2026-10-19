"""G-ORDER: the mass matrix must be the covariance, in the kernel's coordinate order.

Two independent defects lived in `PfCheesHMC._run_pathfinder` until WP1, and neither raises:

1. **Inverted orientation.** BlackJAX's `inverse_mass_matrix` is the matrix that multiplies the
   momentum in the kinetic energy (`mcmc/metrics.py:262-270`), so leapfrog gives q'' = -imm*grad U
   and a well-conditioned trajectory needs imm ~ Cov(q). The wheel says so itself:
   `adaptation/chees_adaptation.py:335` calls the argument "the diagonal inverse_mass_matrix
   (Sigma)", `:453` computes `v' = Sigma p'`, and `adaptation/mass_matrix.py:344-353` (BlackJAX's
   own Stan window adaptation) returns the regularised *covariance*. The code returned
   `jnp.linalg.inv(cov)`.
2. **Wrong coordinate order.** The metric was built by ravelling Pathfinder's position tree, a
   *tuple* in `model.value_vars` order, while the kernel ravels the *dict* state, which JAX
   flattens in sorted-key order (`mcmc/metrics.py:236-244`). 8 Schools agrees by luck. GEV does not
   (`[mu, sigma_log__, xi_interval__]` vs `[mu, xi_interval__, sigma_log__]`), and on radon the five
   scalars shift inside the 170 raw effects. A dense metric applied through the wrong permutation
   is a different, unit-incoherent preconditioner, and every downstream number is still printed.

Why the tests inject Pathfinder's draw pool instead of running Pathfinder. Part B §WP1 proposed
comparing `diag(imm)` to "the kernel-order variance of the PF draws within 20%". Measured on the
specified toy model (`z ~ N(0,100)`, `a ~ N(0,0.01)`, no data): Pathfinder's pool covariance is
`diag = [0.947, 0.908, 1.022, 1.026]` and its `lbfgs_lowrank` metric is exactly `[1, 1, 1, 1]` --
identity. That is not a bug in this repo: a prior-only model has zero gradient at its mode, so
L-BFGS receives no usable curvature pairs and Pathfinder's approximation stays at its identity
initial scaling. A test on that model passes on the broken code for *both* defects, and a stricter
tolerance would only test Pathfinder's fit quality. So the orientation and ordering tests below
replace `pathfinder.sample` with a pool of known covariance and exercise the real reordering and
orientation code; the identity claim itself is a recorded finding (SUMMARY §6), not a silent
workaround. Every test here fails on `ecd9ba1` and passes on `wp/wp1`.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

import blackjax as bj
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from jax.flatten_util import ravel_pytree

from pfcheeshmc.sampler import PfCheesHMC, sample_pf_chees_hmc

# The fixture's true prior variances, in *Pathfinder's* (value_vars) order: z, then a[0..2].
PF_ORDER_VAR = np.array([1e4, 1e-4, 1e-4, 1e-4])
# The same numbers in *kernel* (sorted-key) order: a[0], a[1], a[2], z.
KERNEL_ORDER_VAR = np.array([1e-4, 1e-4, 1e-4, 1e4])
Z_INDEX_IN_KERNEL_ORDER = 3


@pytest.fixture(scope="module")
def odd_order_model():
    """z (sd 100, created first) and a (sd 0.01, created second): value_vars != sorted keys.

    Deliberately not sorted: a test on an already-sorted model would pass on the broken code. The
    two scales are 8 orders of magnitude apart so an inverted or permuted metric cannot hide inside
    a tolerance.
    """
    with pm.Model() as model:
        pm.Normal("z", 0.0, 100.0)
        pm.Normal("a", 0.0, 0.01, shape=3)
    names = [rv.name for rv in model.value_vars]
    assert names == ["z", "a"], f"fixture lost its point: value_vars is {names}"
    assert sorted(names) == ["a", "z"]
    return model


def _inject_pool_with_known_covariance(monkeypatch, var_by_leaf, seed=1234):
    """Replace Pathfinder's sampler with draws whose covariance is exactly `var_by_leaf`.

    Only the draw generator is replaced; `_run_pathfinder` still computes the covariance, the ridge
    and the reordering itself, which is the code under test.
    """
    rng = np.random.default_rng(seed)

    def fake_sample(key, state, num_samples):
        leaves = jax.tree_util.tree_leaves(state.position)
        draws = [jnp.asarray(rng.normal(0.0, np.sqrt(v), size=(num_samples,) + jnp.shape(leaf)))
                 for v, leaf in zip(var_by_leaf, leaves)]
        return tuple(draws), state      # tuple order == value_vars order, as Pathfinder emits

    monkeypatch.setattr(bj.vi.pathfinder, "sample", fake_sample)


def _inject_low_rank_metric(monkeypatch, sigma, rank=1):
    """Replace both low-rank builders with a known metric expressed in Pathfinder's order."""
    d = len(sigma)
    metric = LowRankInverseMassMatrix(
        sigma=jnp.asarray(sigma, dtype=float),
        U=jnp.zeros((d, rank)),
        lam=jnp.ones((rank,)),          # lam == 1 -> expansion is exactly diag(sigma**2)
    )
    monkeypatch.setattr(bj.mcmc.metrics, "lbfgs_inverse_hessian_to_low_rank_metric",
                        lambda *a, **k: metric)

    def fake_fisher(gdraws, ggrads, max_rank, **kwargs):
        # The plan's claim: gdraws/ggrads are internally consistent in tuple order, so only the
        # *result* may be permuted. Pin that here, not just the output.
        assert jnp.ndim(gdraws) == 2 and gdraws.shape[1] == d
        assert jnp.ndim(ggrads) == 2 and ggrads.shape[1] == d
        return metric

    monkeypatch.setattr(bj.adaptation.metric_estimators, "fisher_score_low_rank", fake_fisher)
    return np.asarray(sigma) ** 2


def _imm(model, **pathfinder_kwargs):
    sampler = PfCheesHMC(model, seed=0)
    return np.asarray(sampler._run_pathfinder(chains=8, draws=128, **pathfinder_kwargs)[1])


def test_metric_is_the_covariance_in_kernel_order(odd_order_model, monkeypatch):
    _inject_pool_with_known_covariance(monkeypatch, PF_ORDER_VAR)
    diag = np.diag(_imm(odd_order_model))
    rel = np.abs(diag - KERNEL_ORDER_VAR) / KERNEL_ORDER_VAR
    assert rel.max() < 0.2, (
        f"diag(imm)={diag}, expected {KERNEL_ORDER_VAR} (max rel err {rel.max():.3f}). "
        "Large entries at indices 0-2 instead of 3 = the inverse was returned (orientation). "
        "Correct values but the 1e4 at index 0 = tuple order was never permuted (ordering)."
    )


def test_kinetic_energy_and_momentum_scale_every_coordinate(odd_order_model, monkeypatch):
    """Part B's toy check, on a pool whose covariance is known: KE = 1/2 p . (imm p).

    Every coordinate is checked, not just z. With an inverted *and* unpermuted diagonal metric the two
    errors cancel at one index -- that is exactly how a single-index version of this test passed on
    ecd9ba1 -- so a one-coordinate check proves nothing.
    """
    _inject_pool_with_known_covariance(monkeypatch, PF_ORDER_VAR)
    metric = bj.mcmc.metrics.default_metric(_imm(odd_order_model))
    base = {"z": jnp.asarray(0.0), "a": jnp.zeros(3)}
    ke = []
    for idx in range(4):
        pos = dict(base)
        if idx == Z_INDEX_IN_KERNEL_ORDER:
            pos["z"] = jnp.asarray(1.0)
        else:
            pos["a"] = jnp.zeros(3).at[idx].set(1.0)
        ke.append(float(metric.kinetic_energy(pos)))
    expected_ke = 0.5 * KERNEL_ORDER_VAR
    assert np.max(np.abs(np.asarray(ke) - expected_ke) / expected_ke) < 0.2, (
        f"KE per kernel coordinate ={ke}, expected {expected_ke}"
    )

    # Momentum must be drawn at the posterior scale: p ~ N(0, imm^-1), so the wide coordinate (z)
    # gets the *small* momentum spread. Reversed scales mean the metric is inverted.
    pos = {"z": jnp.asarray(3.0), "a": jnp.zeros(3)}
    mom = jax.vmap(lambda k: metric.sample_momentum(k, pos))(
        jax.random.split(jax.random.key(0), 2000))
    # tree_leaves of a dict pytree is sorted-key order, i.e. exactly the kernel's column order.
    flat = jnp.concatenate([jnp.reshape(x, (2000, -1))
                            for x in jax.tree_util.tree_leaves(mom)], axis=1)
    sds = np.array([float(col.std()) for col in flat.T])
    expected_sd = 1.0 / np.sqrt(KERNEL_ORDER_VAR)
    assert np.max(np.abs(sds - expected_sd) / expected_sd) < 0.2, (
        f"momentum sds in kernel order ={sds}, expected {expected_sd}"
    )


@pytest.mark.parametrize("imm_source", ["lbfgs_lowrank", "fisher_lowrank"])
def test_low_rank_metric_sources_are_kernel_ordered_too(odd_order_model, monkeypatch, imm_source):
    """The low-rank routes share the ordering bug: their factors come out in tuple order."""
    expected_kernel_diag = KERNEL_ORDER_VAR.copy()
    _inject_low_rank_metric(monkeypatch, np.sqrt(PF_ORDER_VAR))
    diag = np.diag(_imm(odd_order_model, imm_source=imm_source, max_rank=1))
    assert np.allclose(diag, expected_kernel_diag, rtol=1e-6), (
        f"{imm_source}: diag(imm)={diag}, expected {expected_kernel_diag}"
    )


def test_permutation_is_tuple_to_dict_index(odd_order_model):
    sampler = PfCheesHMC(odd_order_model, seed=0)
    ip = odd_order_model.initial_point()
    init_tuple = tuple(jnp.zeros(jnp.shape(ip[n]))
                       for n in (rv.name for rv in odd_order_model.value_vars))
    pf_state, _ = bj.vi.pathfinder.approximate(jax.random.key(0), sampler.ld_fn, init_tuple)
    draws, _ = bj.vi.pathfinder.sample(jax.random.key(1), pf_state, 4)
    perm = np.asarray(sampler._kernel_permutation(draws))
    # Kernel slot 0 is a[0], which sits at tuple index 1; kernel slot 3 is z, tuple index 0.
    assert perm.tolist() == [1, 2, 3, 0], f"expected kernel->tuple indices [1,2,3,0], got {perm}"


def test_permutation_is_identity_when_names_are_sorted():
    with pm.Model() as model:
        pm.Normal("alpha", 0.0, 1.0)
        pm.Normal("beta", 0.0, 1.0, shape=2)
    sampler = PfCheesHMC(model, seed=0)
    ip = model.initial_point()
    init_tuple = tuple(jnp.zeros(jnp.shape(ip[n]))
                       for n in (rv.name for rv in model.value_vars))
    pf_state, _ = bj.vi.pathfinder.approximate(jax.random.key(0), sampler.ld_fn, init_tuple)
    draws, _ = bj.vi.pathfinder.sample(jax.random.key(1), pf_state, 4)
    assert np.asarray(sampler._kernel_permutation(draws)).tolist() == [0, 1, 2]


def test_permutation_rejects_a_tree_that_does_not_match_the_model(odd_order_model):
    """A silent leaf-count mismatch is exactly the failure class this whole file exists to stop."""
    sampler = PfCheesHMC(odd_order_model, seed=0)
    with pytest.raises(ValueError, match="value_vars"):
        sampler._kernel_permutation((jnp.zeros((4, 1)), jnp.zeros((4, 3)), jnp.zeros((4,))))


def test_kernel_flat_matches_the_kernels_own_ravel(odd_order_model):
    sampler = PfCheesHMC(odd_order_model, seed=0)
    pos = {"z": jnp.asarray(3.0), "a": jnp.arange(3.0)}
    flat, _ = ravel_pytree(pos)                     # what the kernel does internally
    assert jnp.allclose(sampler._kernel_flat(pos), flat)
    assert float(sampler._kernel_flat(pos)[Z_INDEX_IN_KERNEL_ORDER]) == 3.0


def test_permuting_factors_equals_permuting_the_expansion():
    """WP1 permutes the expanded dense matrix; Part B's wording permutes sigma/U first.

    Same map (conjugation by a permutation matrix commutes with D(I + U(L-1)U^T)D), pinned here so
    the single code path cannot drift from the spec.
    """
    key = jax.random.key(1)
    d = 4
    sigma = jnp.array([2.0, 3.0, 5.0, 7.0])
    q, _ = jnp.linalg.qr(jax.random.normal(key, (d, 2)))
    lam = jnp.array([4.0, 0.5])

    def expand(sig, u):
        Dm = jnp.diag(sig)
        return Dm @ (jnp.eye(d) + u @ jnp.diag(lam - 1.0) @ u.T) @ Dm

    p = np.array([1, 2, 3, 0])
    assert jnp.allclose(expand(sigma, q)[p][:, p], expand(sigma[p], q[p, :]), atol=1e-10)


def test_frozen_step_size_is_in_posterior_units(odd_order_model, monkeypatch):
    """Smoke, end to end through the public entrypoint: eps is O(1) and the trajectory is short.

    The pool is injected so the metric the pipeline receives is the true covariance (Part B's
    intent); Pathfinder's own pool on this model is identity and would test nothing. With the
    correct metric every coordinate is a unit-frequency oscillator, so eps must land in posterior-sd
    units. Under the inverted metric eps has to absorb the 1e8 scale ratio instead: it collapses and
    L explodes. Measured bounds are in SUMMARY §3 (ecd9ba1: eps ~ 1e-4, L ~ 400).
    """
    _inject_pool_with_known_covariance(monkeypatch, PF_ORDER_VAR)
    idata = sample_pf_chees_hmc(
        odd_order_model, draws=50, chains=32, random_seed=7, progressbar=False,
        chees_kwargs={"draws": 200},
    )
    eps = float(idata.attrs["frozen_step_size"])
    L = float(idata.attrs["frozen_L_nominal"])
    assert 0.05 <= eps <= 2.0, f"frozen eps={eps}: not in posterior-sd units, metric is mis-scaled"
    assert 1 <= L <= 40, f"frozen nominal L={L}: a 4-dim Gaussian should not need hundreds of steps"
    for key in ("warmup_grad_evals_per_chain", "sampling_grad_evals_per_chain", "warmup_steps_used"):
        assert key in idata.attrs, f"A8 contract field {key!r} missing from idata.attrs"
    assert int(idata.attrs["warmup_grad_evals_per_chain"]) >= 200
