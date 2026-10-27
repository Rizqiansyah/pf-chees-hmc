"""G-ORDER: the metric must be applied in the kernel's coordinate order, and it is the posterior covariance.

Two separate questions, answered differently here:

**Ordering (a bug, unconditional).** The metric is built by ravelling Pathfinder's position tree, a
*tuple* in `model.value_vars` order, while the kernel ravels the *dict* state and JAX sorts dict keys
(`mcmc/metrics.py:236-244`). 8 Schools agrees by luck; GEV is `[mu, sigma_log__, xi_interval__]` vs
`[mu, xi_interval__, sigma_log__]`; radon shifts five scalars inside the 170 raw effects. A dense
metric applied through the wrong permutation is a different preconditioner, and nothing raises. Every
ordering test below fails on `ecd9ba1`.

**Orientation (a convention, then a measurement).** What the wheel fixes is the *convention*: the array
(call it A) is the matrix multiplying the momentum in the kinetic energy, `KE = 1/2 p.(A p)`
(`mcmc/metrics.py:262-270`), so `p ~ N(0, A^-1)` and `q'' = -A grad U`; with target Hessian H the squared
frequencies are `eig(A H)`, which is why `A = Cov(q)` makes every coordinate unit-frequency and
`A = Cov(q)^-1` makes them 1/var. That convention is pinned by `test_kernel_convention_...`. The sampler
hands the kernel the covariance, for every `imm_source` (the inverse, and the Pathfinder-order metric, were
options until C2 removed them: both reproduced known bugs), and
`test_the_default_metric_mixes_the_anisotropic_target` asks whether that recovers a known posterior at a
practical budget.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

import blackjax as bj
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from jax.flatten_util import ravel_pytree

from _helpers import PF_ORDER_VAR, mixing_report as _mixing_report
from pfcheeshmc.sampler import PfCheesHMC


def _scalar(x):
    """A 0-dim arviz_stats result to float, whatever shape it comes back in."""
    import numpy as _np
    a = _np.asarray(getattr(x, "values", x))
    return float(a.reshape(-1)[0]) if a.size == 1 else float(a)

Z_INDEX_IN_KERNEL_ORDER = 3


def _inject_pool_with_known_covariance(monkeypatch, var_by_leaf, seed=1234):
    """Replace Pathfinder's sampler with draws whose covariance is exactly `var_by_leaf`.

    Only the draw generator is replaced; `_run_pathfinder` still computes the covariance, the ridge,
    the orientation and the reordering, which is the code under test. Needed because Pathfinder's own
    pool on a prior-only model is identity (zero gradient at the initial point -> no curvature pairs),
    so a pool-comparison test on such a model passes on the broken code.
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
        # Part B's claim: gdraws/ggrads are internally consistent in tuple order, so only the
        # *result* may be permuted. Pinned here, not just the output.
        assert jnp.ndim(gdraws) == 2 and gdraws.shape[1] == d
        assert jnp.ndim(ggrads) == 2 and ggrads.shape[1] == d
        return metric

    monkeypatch.setattr(bj.adaptation.metric_estimators, "fisher_score_low_rank", fake_fisher)
    return np.asarray(sigma) ** 2




def _imm(model, **pathfinder_kwargs):
    sampler = PfCheesHMC(model, seed=0)
    return np.asarray(sampler._run_pathfinder(chains=8, draws=128, **pathfinder_kwargs)[1])


# ------------------------------------------------------------------- ordering (unconditional)
@pytest.mark.parametrize("imm_source", ["mc", "lbfgs_lowrank", "fisher_lowrank"])
def test_every_imm_source_hands_the_kernel_the_same_ordered_array(odd_order_model, monkeypatch, imm_source):
    """The array must arrive in the kernel's coordinate order.

    Expected array is derived from the injected covariance: covariance -> P C P^T.
    """
    _inject_pool_with_known_covariance(monkeypatch, PF_ORDER_VAR)
    if imm_source != "mc":
        _inject_low_rank_metric(monkeypatch, np.sqrt(PF_ORDER_VAR))
    # pf_pool_cap=0 pins the pool to chains*draws = 1,024 rows. The property under test is coordinate
    # ORDER, which is pool-size independent, but the fixture's injected pool draws exactly
    # `num_samples` rows from a known-variance normal, so the estimator's diagonal has relative SE
    # ~sqrt(2/N). Since `cc75289` the default pool is max(4C, 4d^2) = 64 rows on this d=4 fixture
    # (~18% SE), which is at the tolerance below -- the arm would then fail on estimator noise, not on
    # a permutation defect, and a test that cannot resolve its own signal has no teeth. Tolerance is
    # NOT loosened; the pool is pinned instead.
    diag = np.diag(_imm(odd_order_model, imm_source=imm_source, max_rank=1, pf_pool_cap=0))
    expected = KERNEL_ORDER_VAR
    rel = np.abs(diag - expected) / expected
    assert rel.max() < 0.2, (
        f"{imm_source}: diag(imm)={diag}, expected {expected} "
        f"(max rel err {rel.max():.3f}). Correct values but the 1e4 at index 0 = the tuple->dict "
        "permutation was never applied."
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
    """A silent leaf-count mismatch is the failure class this file exists to stop."""
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

    Same map (conjugation by a permutation matrix commutes with D(I + U(L-1)U^T)D), pinned so the
    single code path cannot drift from the spec.
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


def test_kernel_convention_is_ke_half_p_A_p_and_momentum_cov_is_A_inverse(odd_order_model, monkeypatch):
    """Pin the convention, orientation-agnostic: the passed array multiplies p in KE.

    This is the measurement behind the LEDGER entry that challenged Part B §0.1: with A =
    inv(Cov) the sampled momentum variance came out equal to Cov. That fixes what A *means*; it does
    not say which A is the better preconditioner. The frequency consequence (q'' = -A grad U, so
    squared frequencies are eig(A H)) is what the moment test below is sensitive to.
    """
    _inject_pool_with_known_covariance(monkeypatch, PF_ORDER_VAR)
    A = _imm(odd_order_model)
    metric = bj.mcmc.metrics.default_metric(A)
    base = {"z": jnp.asarray(0.0), "a": jnp.zeros(3)}
    for idx in range(4):
        pos = dict(base)
        if idx == Z_INDEX_IN_KERNEL_ORDER:
            pos["z"] = jnp.asarray(1.0)
        else:
            pos["a"] = jnp.zeros(3).at[idx].set(1.0)
        ke = float(metric.kinetic_energy(pos))
        assert abs(ke - 0.5 * A[idx, idx]) < 1e-8 * max(1.0, abs(ke)), f"KE on kernel coord {idx}: {ke}"

    pos = {"z": jnp.asarray(0.0), "a": jnp.zeros(3)}
    mom = jax.vmap(lambda k: metric.sample_momentum(k, pos))(
        jax.random.split(jax.random.key(0), 4000))
    flat = jnp.concatenate([jnp.reshape(x, (4000, -1))
                            for x in jax.tree_util.tree_leaves(mom)], axis=1)
    var = np.array([float(col.var()) for col in flat.T])
    expected = 1.0 / np.diag(A)              # momentum ~ N(0, A^-1)
    assert np.max(np.abs(var - expected) / expected) < 0.15, f"momentum variance {var} vs {expected}"


# ------------------------------------------------------------------- orientation (measured)
def test_the_default_metric_mixes_the_anisotropic_target(anisotropic_model):
    """The covariance, in kernel order: measured at R-hat 1.002 on this fixture.

    Mean within 4 MCSE of the analytic posterior for every coordinate, max R-hat under the 1.01
    bar, and worst-variable bulk ESS per gradient above 1.0. Squared frequencies are eig(A H) with
    A the array the kernel is handed and H the target Hessian, so A = Cov(q) puts every coordinate
    at unit frequency and one global eps serves a 1e4 variance spread. (The inverse of the covariance,
    which was an option until C2, froze at eps 0.013 with max R-hat 12-13 on it: a moment test alone
    cannot tell the two apart, which is why this one asks for mixing.)
    """
    idata, max_rhat, min_ess, per, steps, mcse_mean, mean, var = _mixing_report(anisotropic_model)
    for name in mean:
        got = float(idata.posterior[name].mean())
        assert abs(got - mean[name]) <= 4 * float(np.asarray(mcse_mean[name]).reshape(-1)[0]), (
            f"{name}: mean {got:.4g} != {mean[name]:.4g}")
    assert max_rhat < 1.01, f"max R-hat {max_rhat:.4f}, per-variable ESS {per}"
    assert min_ess / (steps * 1000) > 1.0, f"ESS per gradient {min_ess / (steps * 1000):.3f}"
    assert min_ess > 10_000, f"min bulk ESS {min_ess:.0f}, per-var {per}"
