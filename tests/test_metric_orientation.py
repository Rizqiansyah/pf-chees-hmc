"""G-ORDER: the metric must be applied in the kernel's coordinate order, and the orientation is a
measured choice, not an assumption.

Two separate questions were tangled in Part B §WP1, and they are answered differently here:

**Ordering (a bug, unconditional).** The metric is built by ravelling Pathfinder's position tree, a
*tuple* in `model.value_vars` order, while the kernel ravels the *dict* state and JAX sorts dict keys
(`mcmc/metrics.py:236-244`). 8 Schools agrees by luck; GEV is `[mu, sigma_log__, xi_interval__]` vs
`[mu, xi_interval__, sigma_log__]`; radon shifts five scalars inside the 170 raw effects. A dense
metric applied through the wrong permutation is a different preconditioner, and nothing raises. Every
ordering test below fails on `ecd9ba1`.

**Orientation (a measurement, not an assertion).** Both orientations are valid HMC and preserve the
target, so no array-inspection test can decide them; `test_posterior_moments_recovered` decides it by
asking whether the sampler recovers a known posterior at a practical budget. What the wheel fixes is
only the *convention*: the array (call it A) is the matrix multiplying the momentum in the kinetic
energy, `KE = 1/2 p.(A p)` (`mcmc/metrics.py:262-270`), so `p ~ N(0, A^-1)` and `q'' = -A grad U`;
with target Hessian H the squared frequencies are `eig(A H)`, which is why `A = Cov(q)` makes every
coordinate unit-frequency and `A = Cov(q)^-1` makes them 1/var. That convention is pinned by
`test_kernel_convention_...`, which is orientation-agnostic.

One finding this file forced: `imm_source="mc"` inverted its covariance while `lbfgs_lowrank` and
`fisher_lowrank` passed their expansion un-inverted, so the three sources ran in *opposite*
orientations. Every published `mc`-vs-low-rank comparison in `docs/ladder-results.md` is confounded by
it, independent of the permutation. `imm_orientation` now makes the choice explicit and applies it to
all three sources.
"""
import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
import pymc as pm
import pytest

import blackjax as bj
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from jax.flatten_util import ravel_pytree

import arviz_stats

import pfcheeshmc
from pfcheeshmc.sampler import PfCheesHMC, sample_pf_chees_hmc


def _scalar(x):
    """A 0-dim arviz_stats result to float, whatever shape it comes back in."""
    import numpy as _np
    a = _np.asarray(getattr(x, "values", x))
    return float(a.reshape(-1)[0]) if a.size == 1 else float(a)

Z_INDEX_IN_KERNEL_ORDER = 3
ORIENTATIONS = ["precision", "covariance"]


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def odd_order_model():
    """z (sd 100, created first) and a (sd 0.01, created second): value_vars != sorted keys.

    Deliberately not sorted: a test on an already-sorted model would pass on the broken code. The
    two scales are 8 orders of magnitude apart so a mis-permutation cannot hide inside a tolerance.
    """
    with pm.Model() as model:
        pm.Normal("z", 0.0, 100.0)
        pm.Normal("a", 0.0, 0.01, shape=3)
    names = [rv.name for rv in model.value_vars]
    assert names == ["z", "a"], f"fixture lost its point: value_vars is {names}"
    assert sorted(names) == ["a", "z"]
    return model


@pytest.fixture(scope="module")
def anisotropic_model():
    """Four scalars, posterior variances 1e2 / 3 / 1 / 1e-2, created in a NON-reversing order.

    Conjugate Normal-Normal, so the analytic posterior is exact: var = 1/(1/tau^2 + n/sigma^2),
    mean = var*(n/sigma^2)*ybar. Tests compare by name, never by position.

    The fixture design matters. An earlier version used three scalars with variances 1e1 / 1e0 /
    1e-2 created in reverse-sorted order, and it is degenerate for the question these tests ask: the
    permutation is a reversal, and a symmetric variance ladder means "read an unpermuted covariance"
    and "read a permuted precision" are nearly the same array, so two of the 2x2 cells came back
    good and ordering and orientation looked interchangeable. Here value_vars order is
    [delta, beta, zeta, alpha] against kernel order [alpha, beta, delta, zeta], variances
    (1, 1e-2, 3, 1e2): no symmetry to hide behind, and exactly one 2x2 cell is well-conditioned
    (see results/WP1/SUMMARY.md section 6).
    """
    rng = np.random.default_rng(3)
    specs = [("delta", 3.0, 3), ("beta", 1.0, 100), ("zeta", 10.0, 1), ("alpha", 1.0, 1)]
    y = {name: rng.normal(0.0, sigma, n) for name, sigma, n in specs}
    tau = 1e6
    with pm.Model() as model:
        for name, sigma, n in specs:
            v = pm.Normal(name, 0.0, tau)
            pm.Normal(f"obs_{name}", v, sigma, observed=y[name])
    mean, var = {}, {}
    for name, sigma, n in specs:
        prec = 1.0 / tau ** 2 + n / sigma ** 2
        var[name] = 1.0 / prec
        mean[name] = var[name] * (n / sigma ** 2) * float(y[name].mean())
    assert [rv.name for rv in model.value_vars] == ["delta", "beta", "zeta", "alpha"]
    assert sorted(rv.name for rv in model.value_vars) == ["alpha", "beta", "delta", "zeta"]
    return model, mean, var


def _inject_pool_with_known_covariance(monkeypatch, var_by_leaf, seed=1234):
    """Replace Pathfinder's sampler with draws whose covariance is exactly `var_by_leaf`.

    Only the draw generator is replaced; `_run_pathfinder` still computes the covariance, the ridge,
    the orientation and the reordering, which is the code under test. Needed because Pathfinder's own
    pool on a prior-only model is identity (zero gradient at the initial point -> no curvature pairs),
    so a pool-comparison test on such a model passes on the broken code. See results/WP1/SUMMARY.md.
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


# PF-order variances for the injection tests: z wide, a tight.
PF_ORDER_VAR = np.array([1e4, 1e-4, 1e-4, 1e-4])
KERNEL_ORDER_VAR = np.array([1e-4, 1e-4, 1e-4, 1e4])   # sorted: a, a, a, z


def _imm(model, **pathfinder_kwargs):
    sampler = PfCheesHMC(model, seed=0)
    return np.asarray(sampler._run_pathfinder(chains=8, draws=128, **pathfinder_kwargs)[1])


# ------------------------------------------------------------------- ordering (unconditional)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("imm_source", ["mc", "lbfgs_lowrank", "fisher_lowrank"])
def test_every_imm_source_hands_the_kernel_the_same_ordered_array(odd_order_model, monkeypatch, imm_source, orientation):
    """Whatever the orientation, the array must arrive in the kernel's coordinate order.

    Expected array is derived from the injected covariance by the arm's own definition: covariance
    -> P C P^T, precision -> P C^-1 P^T. Nothing here prefers one orientation.
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
    diag = np.diag(_imm(odd_order_model, imm_source=imm_source, max_rank=1,
                        imm_orientation=orientation, pf_pool_cap=0))
    expected = KERNEL_ORDER_VAR if orientation == "covariance" else 1.0 / KERNEL_ORDER_VAR
    rel = np.abs(diag - expected) / expected
    assert rel.max() < 0.2, (
        f"{imm_source}/{orientation}: diag(imm)={diag}, expected {expected} "
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


def test_unknown_orientation_raises(odd_order_model):
    with pytest.raises(ValueError, match="imm_orientation"):
        _imm(odd_order_model, imm_orientation="whatever")


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
    A = _imm(odd_order_model, imm_orientation="covariance")
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
def _mixing_report(anisotropic_model, **pathfinder_kwargs):
    model, mean, var = anisotropic_model
    kwargs = {"draws": 256}
    kwargs.update(pathfinder_kwargs)
    idata = sample_pf_chees_hmc(model, draws=1000, chains=32, random_seed=5, progressbar=False,
                                pathfinder_kwargs=kwargs, chees_kwargs={"draws": 500})
    max_rhat = max(float(np.max(v.values)) for v in az.rhat(idata.posterior).values())
    ess = az.ess(idata.posterior, method="bulk", relative=False)
    per = {n: float(np.min(ess[n].values)) for n in var}
    steps = float(np.mean(idata.sample_stats["num_integration_steps"].values))
    min_ess = min(per.values())
    # arviz_stats, not arviz: arviz 0.23's _mcse_sd raises "only 0-dimensional arrays can be
    # converted to Python scalars" under numpy 2.4 on this box (verified). A6 says use arviz_stats.
    mcse_mean = arviz_stats.mcse(idata, method="mean")
    return idata, max_rhat, min_ess, per, steps, mcse_mean, mean, var


def test_covariance_orientation_mixes_the_anisotropic_target(anisotropic_model):
    """permuted + covariance: the well-conditioned cell of the 2x2, measured at R-hat 1.002.

    Mean within 4 MCSE of the analytic posterior for every coordinate, max R-hat under the 1.01
    bar, and worst-variable bulk ESS per gradient above 1.0. Squared frequencies are eig(A H) with
    A the array the kernel is handed and H the target Hessian, so A = Cov(q) puts every coordinate
    at unit frequency and one global eps serves a 1e4 variance spread.
    """
    idata, max_rhat, min_ess, per, steps, mcse_mean, mean, var = _mixing_report(
        anisotropic_model, imm_orientation="covariance")
    for name in mean:
        got = float(idata.posterior[name].mean())
        assert abs(got - mean[name]) <= 4 * float(np.asarray(mcse_mean[name]).reshape(-1)[0]), (
            f"{name}: mean {got:.4g} != {mean[name]:.4g}")
    assert max_rhat < 1.01, f"max R-hat {max_rhat:.4f}, per-variable ESS {per}"
    assert min_ess / (steps * 1000) > 1.0, f"ESS per gradient {min_ess / (steps * 1000):.3f}"


def test_precision_orientation_does_not_mix_the_anisotropic_target(anisotropic_model):
    """permuted + inv(cov): the deciding cell. Ordering is correct here, only the orientation differs.

    Measured (this fixture, 32 chains, 500 warmup, 1,000 draws, seeds 5-7): eps 0.0130-0.0140,
    max R-hat 12.10-13.08, bulk ESS 32-38 on alpha, delta and zeta and ~33,000 on beta. With
    A = Cov^-1 the frequency of a coordinate is 1/var, so the tightest coordinate caps the single
    global eps and every wider coordinate stops moving. This is why the orientation is not merely a
    tuning preference: at a practical budget it fails the project's own G-A bar (R-hat <= 1.01).

    Asserted as a failure so the finding cannot be silently undone. If it ever passes, the
    orientation no longer matters on anisotropic targets and WP1's conclusion must be re-examined.
    """
    _, max_rhat, min_ess, per, *_ = _mixing_report(
        anisotropic_model, imm_orientation="precision")
    assert max_rhat > 2.0 or min_ess < 200.0, (
        f"precision arm now mixes (max R-hat {max_rhat:.4f}, ESS {per}); re-examine WP1 section 6")


def test_ordering_error_collapses_one_coordinate(anisotropic_model):
    """unpermuted + covariance: the ordering factor in isolation, which is the WP1 P0.

    Same orientation as the good cell, metric left in Pathfinder's tuple order. Measured:
    max R-hat 1.0069-1.1451 and bulk ESS 166-3,176 on zeta (variance 1e2) while the other three
    coordinates stay near 30,000. The signature is the opposite of the orientation failure: a wrong
    permutation misplaces specific coordinates, a wrong orientation flattens every wide one. Both
    are silent, and both survive a moment test, which is why G-ORDER is a permanent gate.
    """
    _, max_rhat, min_ess, per, *_ = _mixing_report(
        anisotropic_model, imm_orientation="covariance", imm_order="pathfinder")
    assert max_rhat > 1.01 or min_ess < 10_000, (
        f"unpermuted metric now mixes (max R-hat {max_rhat:.4f}, ESS {per}); "
        "the ordering fixture has become insensitive, not that the bug is fixed")


def test_default_orientation_is_covariance_and_mixes(anisotropic_model):
    """The shipped default is the covariance orientation (owner decision, 2026-10-20).

    Pinned by behaviour, not by a string: the default arm must mix an anisotropic target. Evidence
    behind the flip, all in results/LEDGER.md -- gradients/ESS better on 4 of 4 real models,
    posterior-mean error halved on GEV (0.059 -> 0.028 sd), R-hat <= 1.01 on GEV in 3 of 5 seeds
    where precision gives 0 of 5, and G-R1 at C=1,024 passing on 3 of 4 models vs 2 of 4. Accepted
    cost: GEV divergences 0.025 -> 0.400 per 1k, seed-stable, mechanism still open.
    If this test goes red the default moved again; change it with the G3 trail, not alone.
    """
    _, max_rhat, min_ess, per, *_ = _mixing_report(anisotropic_model)   # no imm_orientation passed
    assert max_rhat < 1.05, f"default arm is not mixing: max R-hat {max_rhat:.4f}, per-var ESS {per}"
    assert min_ess > 10_000, f"default arm is not mixing: min bulk ESS {min_ess:.0f}, per-var {per}"
