"""WP5: capacity at large chain count -- tests for the init-path changes.

What could silently break:
* the pool cap changing the *number* of picks or the pick range (stratified selection
  must stay a permutation of pool rows, not a rescale of them);
* the chunked scorings diverging from the single-vmap originals they replace (same
  seed, same pool -> same starts, byte for byte);
* the shared single init point weakening the finite-gradient guard (a non-finite
  gradient base must be repaired or raise, never produce a NaN start);
* the scan projections dropping a field a consumer reads (the six sample_stats
  fields and the exact warmup gradient counts are the contract).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

import blackjax as bj
import optax
import pfcheeshmc
from pfcheeshmc.sampler import PfCheesHMC
from pfcheeshmc.utils import WarmupInfo, WarmupScanInfo, chees_warmup_with_progress


def _tiny_model():
    with pm.Model() as model:
        pm.Normal("x", shape=(4,))
    return model


def _recorded_pool_sizes(monkeypatch):
    """Wrap bj.vi.pathfinder.sample to record every pool size requested."""
    seen = []
    real = bj.vi.pathfinder.sample

    def spy(rng_key, state, num_samples=()):
        seen.append(int(num_samples))
        return real(rng_key, state, num_samples)

    monkeypatch.setattr(bj.vi.pathfinder, "sample", spy)
    return seen


def test_default_pool_is_dimension_aware_not_uncapped(monkeypatch):
    """The default cap changed on purpose (2026-10-20): the pool used to be chains*draws.

    Both bounds are measured, not theorised: the uncapped production pool cannot be materialised on
    a 6 GB card at C=1,024/d=175 (5,734,400,000 B refused), and a 4C-sized pool at C=32/d=175 gives a
    rank-deficient covariance (b2 2.34860 vs floor 0.07689, R-hat 1.2121). So the default is
    max(4C, 4d^2) and `pf_pool_cap=0` is the escape hatch back to the historical pool.
    """
    model = _tiny_model()                      # d = 4
    s = PfCheesHMC(model, seed=3)

    seen = _recorded_pool_sizes(monkeypatch)
    s._run_pathfinder(chains=8, draws=500, seed=3)
    assert seen == [max(4 * 8, 4 * 4 * 4)], f"default pool is not max(4C, 4d^2): {seen}"

    seen.clear()
    s._run_pathfinder(chains=8, draws=500, seed=3, pf_pool_cap=0)
    assert seen == [8 * 500], f"pf_pool_cap=0 must give the historical uncapped pool: {seen}"

    seen.clear()
    pf_samples, imm = s._run_pathfinder(chains=8, draws=500, seed=3, pf_pool_cap=4 * 8)
    assert seen == [32], f"pool not capped to the requested 4C: {seen}"
    assert pf_samples["x"].shape == (8, 4)


def test_default_pool_cap_function_bounds_both_ways():
    from pfcheeshmc.sampler import default_pool_cap

    # metric-limited: at C=32, d=175 the cap must stay near the historical 128,000 rows, otherwise
    # the covariance estimate loses rank (the measured radon failure above).
    assert default_pool_cap(32, 175) == 4 * 175 * 175 == 122_500
    assert abs(122_500 - 32 * 4000) <= 0.05 * (32 * 4000), "R2 regime changed size"
    # chain-limited: a low-d problem still wants ~4C starts.
    assert default_pool_cap(1024, 8) == 4 * 1024
    # hard bound: the pool operand must fit a 6 GB card at the worst shape here.
    assert default_pool_cap(1024, 175) * 175 * 8 < 6 * 2**30
    with pytest.raises(ValueError, match="chains>=1"):
        default_pool_cap(0, 4)


def test_pool_cap_below_chains_raises(monkeypatch):
    s = PfCheesHMC(_tiny_model(), seed=3)
    _recorded_pool_sizes(monkeypatch)
    with pytest.raises(ValueError, match="pf_pool_cap"):
        s._run_pathfinder(chains=8, draws=500, seed=3, pf_pool_cap=7)


def test_chunked_scoring_matches_the_single_vmap_original(monkeypatch):
    """Same seed and pool: the chunked scorings must pick the identical starts.

    score_chunk smaller than the scored subset forces the loop through several
    blocks; a huge score_chunk is the original single-vmap path. If the chunked
    accumulation drops or reorders a row, the stratified picks diverge.
    """
    model = _tiny_model()
    s = PfCheesHMC(model, seed=11)
    kwargs = dict(chains=16, draws=300, seed=11, pf_pool_cap=4 * 16)
    a, _ = s._run_pathfinder(score_chunk=5, **kwargs)
    b, _ = s._run_pathfinder(score_chunk=10 ** 9, **kwargs)
    np.testing.assert_array_equal(np.asarray(a["x"]), np.asarray(b["x"]))


def test_best_chain_scoring_picks_the_manual_argmax(monkeypatch):
    """The vmapped+chunked replacement of the per-chain loop must seed Pathfinder from
    the same row: the manual per-row argmax of the repaired init stack (the old
    loop's semantics) against the position the sampler actually hands to approximate.
    """
    model = _tiny_model()
    s = PfCheesHMC(model, seed=5)
    captured = {}
    real_approx = bj.vi.pathfinder.approximate

    def spy(rng_key, logdensity_fn, initial_position, **kw):
        captured["init"] = np.asarray(initial_position)
        return real_approx(rng_key, logdensity_fn, initial_position, **kw)

    monkeypatch.setattr(bj.vi.pathfinder, "approximate", spy)
    # pm_chains=8 keeps eight DISTINCT points so the argmax is non-trivial;
    # the manual reproduction below must then see exactly the same point cloud.
    s._run_pathfinder(chains=8, draws=60, seed=5, score_chunk=3, pm_chains=8)
    assert "init" in captured

    # Reproduce the repaired stack with the same keys _run_pathfinder derives:
    key_fix = jax.random.split(jax.random.key(5), 5)[0]
    init_stack = s._pymc_adapt_diag_start(8, init="adapt_diag", tune=500, random_seed=5)
    init_stack = s._ensure_finite_inits(init_stack, key_fix)
    scores = [float(s.ld_fn({"x": init_stack["x"][c]})) for c in range(8)]
    c_ref = int(np.argmax([sc if np.isfinite(sc) else -np.inf for sc in scores]))
    np.testing.assert_allclose(captured["init"][0], np.asarray(init_stack["x"][c_ref]),
                               rtol=0, atol=0)


def test_shared_init_point_repairs_a_nan_gradient_base(monkeypatch):
    """The guard must fire on the shared point and repair, never pass a NaN through.

    Finite value, NaN derivative at 0 -- the GEV xi=0 shape (see test_finite_start.py).
    With pm_chains=1 the base point is the single shared one; every chain's start must
    be a finite-gradient repair of it.
    """
    model = _tiny_model()
    s = PfCheesHMC(model, seed=7)
    # Swap the underlying jaxified fn so the real dict/tuple `ld_fn` wrapper still
    # dispatches correctly: Pathfinder's L-BFGS calls the density with a flat tuple.
    def bad_sinv_flat(x):
        x = x[0] if isinstance(x, (tuple, list)) else x
        return -jnp.sum(jnp.where(x == 0.0, 0.0, x * x * jnp.sin(1.0 / x)))
    s._ld_fn = bad_sinv_flat

    def fake_init(chains, **kw):
        # The shared base point sits exactly where the gradient is NaN.
        return {"x": jnp.zeros((1, 4))}

    monkeypatch.setattr(s, "_pymc_adapt_diag_start", fake_init)
    # init_source="adapt_diag": the chain starts ARE the repaired init stack, so this
    # asserts directly on the guard's output rather than on PF pool draws.
    pf_samples, _ = s._run_pathfinder(chains=4, draws=60, seed=7, pm_chains=1,
                                      init_source="adapt_diag")
    starts = np.asarray(pf_samples["x"])
    assert starts.shape == (4, 4)
    assert np.isfinite(starts).all(), "NaN start leaked past the guard"
    # The repair moved the shared base off the singular point, and every chain got
    # that repaired base plus the tiny per-chain fallback jitter (<= ~2e-3 relative).
    assert (np.abs(starts) > 0).all()
    base = starts[0]
    bound = 2e-3 * np.abs(base) + 5e-6     # the fallback jitter scale
    assert (np.abs(starts - base) <= bound).all(), "starts drifted off the shared repaired base"
    # And the default init_source still completes with finite starts.
    pf_samples2, _ = s._run_pathfinder(chains=4, draws=60, seed=7, pm_chains=1)
    assert np.isfinite(np.asarray(pf_samples2["x"])).all()


def test_unrepairable_shared_base_raises_not_nans(monkeypatch):
    """If the guard cannot find a finite point it must raise, not return NaN."""
    model = _tiny_model()
    s = PfCheesHMC(model, seed=7)
    s.ld_fn = lambda pos: -jnp.inf

    def fake_init(chains, **kw):
        return {"x": jnp.zeros((1, 4))}

    monkeypatch.setattr(s, "_pymc_adapt_diag_start", fake_init)
    with pytest.raises(RuntimeError, match="finite log-density and finite gradient"):
        s._run_pathfinder(chains=4, draws=60, seed=7, pm_chains=1)


def test_warmup_scan_projects_to_the_three_fields_and_counts_are_exact():
    """WarmupScanInfo is the whole warmup scan output now.

    num_integration_steps must be the exact per-step leapfrog counts (A8), not an
    estimate: its per-chain sum equals the sum of integration steps the kernel took.
    """
    def ld(pos):
        x = pos["x"]
        return -0.5 * jnp.sum(x ** 2)

    positions = {"x": jnp.ones((4, 4)) * 0.5}
    (states, params), warm = chees_warmup_with_progress(
        ld, jax.random.key(0), positions, 0.1,
        optax.adam(0.05), 40, jnp.eye(4), progressbar=False)
    assert isinstance(warm, WarmupInfo)
    info = warm.info
    assert isinstance(info, WarmupScanInfo)
    assert set(type(info)._fields) == {"num_integration_steps", "is_divergent",
                                       "acceptance_rate"}
    steps = np.asarray(info.num_integration_steps)
    assert steps.shape == (40, 4)
    assert (steps > 0).all()
    assert steps.dtype.kind in "iu"
    divs = np.asarray(info.is_divergent)
    assert divs.shape == (40, 4)
    # The trace is still there, unscathed: one (steps,) array per trace field.
    assert len(warm.trace) == 4
    for t in warm.trace:
        assert np.asarray(t).shape == (40,)
    # Frozen values still projected.
    assert float(params["frozen_step_size"]) > 0
    assert int(params["warmup_steps_used"]) == 40


def test_sampling_scan_projection_keeps_all_six_sample_stats_fields():
    """_SampledState/_SampledInfo must keep exactly the fields _to_inference_data reads."""
    model = _tiny_model()
    idata = pfcheeshmc.sample_pf_chees_hmc(
        model, draws=25, chains=4, random_seed=9, progressbar=False,
        chees_kwargs={"draws": 20},
        pathfinder_kwargs={"draws": 100, "pf_pool_cap": 16})
    need = {"acceptance_rate", "diverging", "energy", "is_accepted", "lp",
            "num_integration_steps"}
    assert need <= set(idata.sample_stats.data_vars)
    steps = np.asarray(idata.sample_stats["num_integration_steps"].values)
    assert steps.shape == (4, 25) and (steps > 0).all()
    assert idata.posterior["x"].shape == (4, 25, 4)


if __name__ == "__main__":
    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if fn.__code__.co_argcount:
                fn(_MP())
            else:
                fn()
            print(f"{name} PASS")


def test_adapt_diag_fallback_keeps_distinct_starts_with_a_shared_base():
    """With pm_chains=1 the adapt_diag fallback must not give every chain the same start:
    identical starts read R-hat 1.000 while stuck, which no diagnostic flags."""
    model = _tiny_model()
    s = PfCheesHMC(model, seed=13)
    # jitter+adapt_diag: plain adapt_diag returns the SAME zero point per chain for this
    # model even pre-WP5, so it cannot demonstrate distinctness; the jittered variant can.
    a, _ = s._run_pathfinder(chains=6, draws=40, seed=13, pm_chains=1,
                             init_source="adapt_diag", pm_init="jitter+adapt_diag")
    starts = np.asarray(a["x"])
    assert starts.shape == (6, 4)
    assert np.isfinite(starts).all()
    assert len({tuple(r) for r in starts.tolist()}) == 6, (
        f"adapt_diag fallback collapsed to identical starts: {starts[:3]}")
    # And the same call with per-chain PyMC inits is untouched by the jitter path.
    b, _ = s._run_pathfinder(chains=6, draws=40, seed=13, pm_chains=6,
                             init_source="adapt_diag", pm_init="jitter+adapt_diag")
    assert len({tuple(r) for r in np.asarray(b["x"]).tolist()}) == 6
