"""F2b: a second sampler call compiles NOTHING (CPU), and the warmup length W and the number of draws D are free.

What a wrong answer here would silently cost: every `sample()` call recompiles XLA programs (15.9-17.4 s of
a 19.9-21.9 s call for 8 Schools at C=1,024) because jitted closures are rebuilt per call and per-run values
(data, priors, metric, step size) are compiled in as constants. Nothing in the draws would show it, so the
only honest test counts compiles: `benchmarks/compile_counter.count_compiles` listens to JAX's backend-compile event.

F2b-1 converted the non-loop sites (`pfcheeshmc.compiled`): the start screening
and scoring, the Pathfinder fit and draws, the step-size search, the pool gradients, the warmup init, and the
value -> problem-space transform. F2b-2 converted the two loops: the warmup (`compiled.warmup_chunk`) and the
sampling loop (`compiled.sampling_chunk`) are chunk programs driven from the host with a dynamic active
count, so W and D are loop bounds on the host. The correctness side (determinism, chunk-size invariance, a
known posterior, bookkeeping, the progress bars) is in `test_chunked_loops.py`.

The draws are pinned, bit for bit, by the golden files of `test_golden.py`.
"""
import contextlib
import pathlib
import sys
import types

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.compile_counter import count_compiles, count_pytensor_functions  # noqa: E402
from benchmarks.seeded_models import dims_model, hier  # noqa: E402

from pfcheeshmc import compiled
from pfcheeshmc.lifted import lift_logp
from pfcheeshmc.sampler import PfCheesHMC, sample_pf_chees_hmc

REPO = pathlib.Path(__file__).resolve().parents[1]
C, W, D = 16, 20, 10
SITES = ("pathfinder", "eps0", "usable_mask", "pool_gradients", "batch_init", "transform")


# ------------------------------------------------------------------------------------ helpers
@contextlib.contextmanager
def _site(counts, name):
    with count_compiles() as cc:
        yield
    counts[name] = cc


def run_sites(model, seed, chains=C, **pathfinder_kw):
    """Every converted non-loop site once, on a fresh sampler. Returns {site: CompileCount}."""
    s = PfCheesHMC(model, seed=seed)
    counts = {}
    with _site(counts, "pathfinder"):
        pf, imm = s._run_pathfinder(chains=chains, seed=seed, score_chunk=128, **pathfinder_kw)
    with _site(counts, "eps0"):
        s._run_step_size_tune(pf, imm, seed=seed)
    with _site(counts, "usable_mask"):
        s._usable_mask(pf)
    ld = s._logdensity()
    pool = {n: jnp.stack([pf[n]] * 3) for n in pf}                      # (3, chains, ...)
    draws = jnp.concatenate([a.reshape(a.shape[0] * a.shape[1], -1) for a in jax.tree_util.tree_leaves(pool)], axis=1)
    one = {n: pool[n][0, 0] for n in pool}
    with _site(counts, "pool_gradients"):                                # chunks of 40 and a rest
        for i in range(0, draws.shape[0], 40):
            compiled.pool_gradients(ld.fn, ld.names, ld.values, one, draws[i:i + 40])
    with _site(counts, "batch_init"):
        compiled.batch_init(ld.fn, ld.names, ld.values, pf, 0)
    hist = types.SimpleNamespace(position={n: jnp.stack([pf[n]] * D) for n in pf})    # (draws, chains, ...)
    with _site(counts, "transform"):
        s._transform_value_samples_to_problem_space(hist)
    return counts


def sample_call(model, seed, progressbar=False, pathfinder_kwargs=None, chains=C, warmup=W, draws=D, **chees):
    with count_compiles() as cc:
        idata = sample_pf_chees_hmc(
            model, draws=draws, chains=chains, random_seed=seed, progressbar=progressbar,
            chees_kwargs={"draws": warmup, **chees}, pathfinder_kwargs=pathfinder_kwargs)
    return idata, cc


def assert_no_compiles(counts, where=""):
    bad = {site: cc.names for site, cc in counts.items() if cc.count}
    assert not bad, f"{where}: a second invocation compiled programs: {bad}"


# the second invocation is a new seed / new data values / new prior values / all three, same shapes
VARIANTS = {
    "new seed": (lambda: hier(1, 1), 2),
    "new data": (lambda: hier(2, 1), 1),
    "new priors": (lambda: hier(1, 2), 1),
    "new everything": (lambda: hier(3, 3), 4),
}


@pytest.fixture(scope="module")
def warm():
    """Every site run once on the base model, so JAX's caches hold its programs."""
    return run_sites(hier(1, 1), seed=1)


@pytest.fixture(scope="module")
def variant_counts(warm):
    return {name: run_sites(build(), seed) for name, (build, seed) in VARIANTS.items()}


# ------------------------------------------------------------------------------- the counter
def test_counter_counts_a_compile_once_and_none_on_a_cache_hit():
    f = jax.jit(lambda x: jnp.sin(x) * 2.0)
    x = jnp.arange(7.0)                                   # a shape no other test uses
    x2, x3 = x + 1.0, jnp.arange(9.0)
    with count_compiles() as first:
        f(x)
    with count_compiles() as again:
        f(x2)
    with count_compiles() as new_shape:
        f(x3)
    with count_compiles() as eager:                       # eager primitives compile too, and are counted
        jnp.cos(jnp.arange(13.0))
    assert first.count == 1 and first.seconds > 0 and len(first.names) == 1
    assert again.count == 0 and again.names == []
    assert new_shape.count == 1
    assert eager.count >= 1


def test_counter_unregisters_its_listener_also_after_an_exception():
    x, y = jnp.zeros(15), jnp.zeros(17)
    with count_compiles() as done:
        jax.jit(lambda a: a * 3.25)(x)
    with pytest.raises(RuntimeError, match="boom"):
        with count_compiles() as interrupted:
            jax.jit(lambda a: a * 4.25)(x)
            raise RuntimeError("boom")
    seen = (done.count, interrupted.count)
    jax.jit(lambda a: a * 5.25)(y)                         # a compile after both contexts have exited
    assert seen == (1, 1) and (done.count, interrupted.count) == seen


def test_nested_counters_both_see_a_compile():
    x = jnp.zeros(11)
    with count_compiles() as outer:
        with count_compiles() as inner:
            jax.jit(lambda a: a + 3.5)(x)
    assert inner.count == outer.count == 1


# ------------------------------------------------------------------------------ preconditions
def test_variants_share_a_structure_and_differ_in_values():
    base = lift_logp(hier(1, 1))
    for name, (build, _) in VARIANTS.items():
        other = lift_logp(build())
        assert other.key == base.key and other.fn is base.fn, name
    assert not all(np.array_equal(a, b) for a, b in zip(base.values, lift_logp(hier(2, 1)).values)), "data"
    assert not all(np.array_equal(a, b) for a, b in zip(base.values, lift_logp(hier(1, 2)).values)), "priors"


# ----------------------------------------------------------------- 1. per site, per variant
@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_second_invocation_of_a_site_compiles_nothing(variant_counts, variant, site):
    cc = variant_counts[variant][site]
    assert cc.count == 0, f"{site} after {variant}: {cc}"


# --------------------------------------------------------------------- 2. the whole sample()
# After one call, every one of these changes compiles NOTHING: a new seed, new data, new priors, a larger W,
# a larger D, and the progress bar on or off (it is host-side, so both run the same programs).
CALL_VARIANTS = {
    "new seed": dict(model=lambda: hier(1, 1), seed=2),
    "new data": dict(model=lambda: hier(2, 1), seed=1),
    "new priors": dict(model=lambda: hier(1, 2), seed=1),
    "new everything": dict(model=lambda: hier(5, 6), seed=3),
    "larger W (20 -> 37)": dict(model=lambda: hier(1, 1), seed=1, warmup=37),
    "smaller W (20 -> 5)": dict(model=lambda: hier(1, 1), seed=1, warmup=5),
    "larger D (10 -> 23)": dict(model=lambda: hier(1, 1), seed=1, draws=23),
    "smaller D (10 -> 3)": dict(model=lambda: hier(1, 1), seed=1, draws=3),
    "W and D beyond a chunk": dict(model=lambda: hier(1, 1), seed=1, warmup=137, draws=211),
    "progressbar on": dict(model=lambda: hier(1, 1), seed=1, progressbar=True),
}


@pytest.fixture(scope="module")
def warm_call():
    sample_call(hier(1, 1), seed=1)                  # the programs of the base call (progressbar off)


@pytest.mark.parametrize("variant", list(CALL_VARIANTS))
def test_second_sample_call_compiles_nothing(warm_call, variant):
    kw = dict(CALL_VARIANTS[variant])
    idata, cc = sample_call(kw.pop("model")(), **kw)
    assert cc.count == 0, f"{variant}: {cc}"
    assert idata.posterior["mu"].shape == (C, kw.get("draws", D))
    assert idata.attrs["warmup_steps_used"] == kw.get("warmup", W)


# F2d / eps0: the PyTensor side. A warm call used to build 3 PyTensor functions (`pm.init_nuts`: the log-density
# and gradient of a NUTS step, and an initial-point function per chain count) and, for a model with `dims=`,
# one more per lift (`lifted._abstract` read the unknown static shapes off `model.initial_point()`).
PYTENSOR_CASES = {
    "new seed, data and priors": (lambda: hier(1, 1), lambda: hier(2, 3), None),
    "jitter+adapt_diag": (lambda: hier(1, 1), lambda: hier(3, 2), {"pm_init": "jitter+adapt_diag"}),
    "pm_chains=C": (lambda: hier(1, 1), lambda: hier(2, 2), {"pm_chains": C, "pm_init": "jitter+adapt_diag"}),
    "a model with dims=": (lambda: dims_model(1, 1), lambda: dims_model(2, 3), None),
}


@pytest.mark.parametrize("case", list(PYTENSOR_CASES))
def test_a_warm_second_sample_call_builds_no_pytensor_function_and_compiles_no_xla_program(case):
    first, second, pathfinder_kwargs = PYTENSOR_CASES[case]
    sample_call(first(), seed=1, pathfinder_kwargs=pathfinder_kwargs)
    with count_pytensor_functions() as pf, count_compiles() as cc:
        idata = sample_pf_chees_hmc(
            second(), draws=D, chains=C, random_seed=2, progressbar=False, chees_kwargs={"draws": W},
            pathfinder_kwargs=pathfinder_kwargs)
    assert pf.count == 0 and cc.count == 0, f"{case}: {pf}; {cc}"
    assert idata.posterior["s" if "dims" in case else "mu"].shape[:2] == (C, D)


def test_the_first_sample_call_of_a_dims_model_builds_no_pytensor_function_either():
    """Not only the warm one: nothing in the sampler asks PyTensor for a function (the model's own build aside)."""
    m = dims_model(5, 5)
    with count_pytensor_functions() as pf:
        sample_pf_chees_hmc(m, draws=D, chains=C, random_seed=1, progressbar=False, chees_kwargs={"draws": W})
    assert pf.count == 0, str(pf)


@pytest.mark.parametrize("first", [False, True])
def test_progressbar_on_and_off_run_the_same_programs(first):
    """The bar is updated by the host between chunks: no callback is traced into any program."""
    sample_call(hier(1, 1), seed=1, progressbar=first, warmup=21, draws=11)
    _, cc = sample_call(hier(1, 1), seed=2, progressbar=not first, warmup=21, draws=11)
    assert cc.count == 0, str(cc)


@pytest.fixture
def small_chunks(monkeypatch):
    """Chunks small enough that W and D span several of them (and the last one is partial)."""
    monkeypatch.setattr(compiled, "WARMUP_CHUNK", 6)
    monkeypatch.setattr(compiled, "SAMPLE_CHUNK", 4)
    monkeypatch.setattr(compiled, "TRANSFORM_CHUNK", 5)


def test_multi_chunk_regime_compiles_each_loop_once_and_then_nothing(small_chunks):
    """W = 3 chunks and D = 3 chunks first; then 2 chunks + 3 steps and 2 chunks + 3 draws, and other counts."""
    _, first = sample_call(hier(1, 1), seed=1, warmup=3 * 6, draws=3 * 4)
    for program in ("warmup_chunk", "sampling_chunk", "transform_draws"):
        assert first.names.count(f"jit({program})") == 1, (program, str(first))
    for warmup, draws in [(2 * 6 + 3, 2 * 4 + 3), (6 + 1, 4 + 1), (1, 1), (7 * 6, 9 * 4 + 2)]:
        _, cc = sample_call(hier(2, 2), seed=2, warmup=warmup, draws=draws)
        assert cc.count == 0, f"W={warmup} D={draws}: {cc}"


# ------------------------------------------------------------------------------ 3. shapes
def test_shape_change_recompiles_the_affected_sites_once_and_changing_back_compiles_nothing(warm):
    new_c = run_sites(hier(1, 1), seed=1, chains=24)
    changed = [s for s in ("pathfinder", "eps0", "usable_mask") if new_c[s].count]
    assert changed == ["pathfinder", "eps0", "usable_mask"], \
        f"C is a shape of these sites; they must compile for a new C: {changed}"      # the counter can see them
    assert_no_compiles(run_sites(hier(1, 1), seed=1, chains=24), "C=24 a second time")
    assert_no_compiles(run_sites(hier(2, 2), seed=2, chains=C), "back to the original C")
    smaller = run_sites(hier(1, 1, J=6), seed=1)
    assert smaller["pathfinder"].count and smaller["transform"].count, "a model with other shapes must compile"
    assert_no_compiles(run_sites(hier(2, 2, J=6), seed=2), "J=6 with other values")
    assert_no_compiles(run_sites(hier(3, 3), seed=3), "back to the original J")


def test_a_shape_change_of_the_whole_call_compiles_one_set_of_programs_and_changing_back_none(warm_call):
    """C is a shape of the loop programs and the transform: C=20 compiles each of them exactly once."""
    _, new_c = sample_call(hier(1, 1), seed=1, chains=20)
    for program in ("warmup_init", "warmup_chunk", "sampling_chunk", "transform_draws"):
        assert new_c.names.count(f"jit({program})") == 1, (program, str(new_c))
    _, again = sample_call(hier(2, 2), seed=2, chains=20)
    assert again.count == 0, str(again)
    _, back = sample_call(hier(3, 3), seed=3, chains=C)
    assert back.count == 0, str(back)


# ------------------------------------------------------------------------------- 5. paths
IMM_SOURCES = ["pf_closed", "lbfgs_lowrank", "fisher_lowrank"]


@pytest.mark.parametrize("imm_source", IMM_SOURCES)
def test_imm_sources_compile_nothing_on_a_second_invocation(imm_source):
    kw = dict(imm_source=imm_source, max_rank=2)
    run_sites(hier(1, 1), seed=1, **kw)
    assert_no_compiles(run_sites(hier(2, 3), seed=2, **kw), imm_source)


def test_stop_rule_path_second_call_compiles_nothing_also_for_another_W():
    """The block and window are settings (one program per rule); the last, shorter block is the same program."""
    rule = {"block": 8, "window": 8, "rhat": 0.0}            # never fires: every block runs
    sample_call(hier(1, 1), seed=1, warmup=20, warmup_stop=rule)
    idata, cc = sample_call(hier(2, 2), seed=2, warmup=28, warmup_stop=rule)
    assert cc.count == 0, str(cc)
    assert idata.attrs["warmup_steps_used"] == 28
    fires = {"block": 8, "window": 8, "rhat": 100.0, "eps_stability": 100.0}      # fires after the first block
    sample_call(hier(1, 1), seed=1, warmup=40, warmup_stop=fires)
    idata, cc = sample_call(hier(2, 2), seed=2, warmup=30, warmup_stop=fires)
    assert cc.count == 0, str(cc)
    assert idata.attrs["warmup_steps_used"] == 8


# --------------------------------------------------------- pm.set_data and replaced log-densities
def test_set_data_between_calls_is_seen_and_compiles_nothing():
    s = PfCheesHMC(hier(1, 1), seed=1)
    pos = {n: jnp.full(np.shape(v), 0.3) for n, v in s.model.initial_point().items() if n in s.rvs}
    stack = {n: jnp.stack([pos[n]] * 3) + jnp.arange(3.0).reshape((3,) + (1,) * np.ndim(pos[n]))
             for n in s.rvs}
    ld = s._logdensity()
    before = compiled.score_positions(ld.fn, ld.names, ld.values, stack)
    new_se = np.linspace(9.0, 16.0, 8)
    with s.model:
        pm.set_data({"se": new_se})
    s._refresh_values()
    ld = s._logdensity()
    with count_compiles() as cc:
        after = compiled.score_positions(ld.fn, ld.names, ld.values, stack)
    assert cc.count == 0, str(cc)
    assert not np.allclose(before, after), "the new data was not seen"
    ref = PfCheesHMC(_hier_with_se(new_se), seed=1)
    assert float(s.ld_fn(pos)) == pytest.approx(float(ref.ld_fn(pos)), rel=1e-12)


def _hier_with_se(se):
    base = hier(1, 1)
    with base:
        pm.set_data({"se": se})
    return base


def test_set_data_to_another_shape_lifts_again():
    rng = np.random.default_rng(0)
    with pm.Model(coords={"i": range(5)}) as m:
        x = pm.Data("x", rng.normal(size=5), dims="i")
        mu = pm.Normal("mu")
        pm.Normal("obs", mu, 1.0, observed=x, dims="i")
    s = PfCheesHMC(m, seed=0)
    key_before = s._lifted.key
    new_x = rng.normal(size=7)
    with m:
        pm.set_data({"x": new_x}, coords={"i": range(7)})
    s._refresh_values()
    assert s._lifted.key != key_before
    fresh = PfCheesHMC(m, seed=0)
    pos = {"mu": jnp.asarray(0.4)}
    assert float(s.ld_fn(pos)) == pytest.approx(float(fresh.ld_fn(pos)), rel=1e-12)
    expect = float(-0.5 * np.sum((new_x - 0.4) ** 2) - 7 * 0.5 * np.log(2 * np.pi)
                   - 0.5 * 0.4 ** 2 - 0.5 * np.log(2 * np.pi))
    assert float(s.ld_fn(pos)) == pytest.approx(expect, rel=1e-12)


def _static_index_model(idx=2):
    """A ``pm.Data`` integer that is a basic index: ``Subtensor`` needs it at trace time, so it is baked in."""
    with pm.Model() as m:
        i = pm.Data("idx", np.array(idx, dtype="int64"))
        theta = pm.Normal("theta", 0.0, 1.0, shape=5)
        pm.Deterministic("sel", theta[i])
        pm.Normal("obs", theta[i], 1.0, observed=10.0)
    return m


def test_a_shared_variable_baked_into_the_program_is_never_silently_stale():
    m = _static_index_model(2)
    s = PfCheesHMC(m, seed=0)
    first = s._lifted
    assert any("Subtensor" in client or "ScalarFromTensor" in client for client, _, _ in first.kept_static), \
        f"the index must stay static for this test to mean anything: {first.kept_static}"
    assert [v.name for v, _ in first.static_shared] == ["idx"] and not first.is_stale()
    pos = {"theta": jnp.arange(5.0)}
    before = float(s.ld_fn(pos))
    with m:
        pm.set_data({"idx": np.array(3, dtype="int64")})
    assert first.is_stale()                                  # `values` cannot see it: nothing lifted changed
    s._refresh_values()
    fresh = PfCheesHMC(_static_index_model(3), seed=0)
    assert not s._lifted.is_stale() and s._lifted.key != first.key
    assert float(s.ld_fn(pos)) == pytest.approx(float(fresh.ld_fn(pos)), rel=1e-12)
    assert float(s._logdensity()(pos)) == pytest.approx(float(fresh.ld_fn(pos)), rel=1e-12)
    assert float(s.ld_fn(pos)) != before
    with m:
        pm.set_data({"idx": np.array(2, dtype="int64")})     # and back: the registry entry is found again
    s._refresh_values()
    assert s._lifted.fn is first.fn and float(s.ld_fn(pos)) == pytest.approx(before, rel=1e-12)


def test_is_stale_covers_exactly_what_is_baked_in():
    from pfcheeshmc.lifted import lift_graph

    m = hier(1, 1)
    lifted_rule = lift_logp(m)                               # `se` is lifted: set_data is seen as a new value
    baked = lift_graph(m, list(m.value_vars), [m.logp()], rule="none")    # today's behaviour: all baked in
    assert lifted_rule.static_shared == () and [v.name for v, _ in baked.static_shared] == ["se"]
    assert not lifted_rule.is_stale() and not baked.is_stale()
    with m:
        pm.set_data({"se": np.linspace(9.0, 16.0, 8)})
    assert not lifted_rule.is_stale() and baked.is_stale()
    assert any(not np.array_equal(a, b) for a, b in zip(lifted_rule.current_values(), lifted_rule.values))
    with m:
        pm.set_data({"se": baked.static_shared[0][1]})        # back to the baked-in value
    assert not baked.is_stale()


def test_sample_after_set_data_of_a_static_shared_variable_uses_the_new_value():
    m = _static_index_model(2)
    s = PfCheesHMC(m, seed=1)

    def run():
        return s.sample(chains=8, draws=20, chees_kwargs={"draws": 40, "progressbar": False},
                        dynamic_hmc_kwargs={"progressbar": False}, pathfinder_kwargs={},
                        step_size_tune_kwargs={})

    # obs = 10 with prior N(0, 1): the indexed coordinate has posterior N(5, 0.7), the others N(0, 1)
    mean = lambda idata, k: float(idata.posterior["theta"].values[..., k].mean())
    a = run()
    assert mean(a, 2) > 3.0 and mean(a, 3) < 1.5
    with m:
        pm.set_data({"idx": np.array(3, dtype="int64")})
    b = run()                                               # the same sampler instance
    assert mean(b, 3) > 3.0 and mean(b, 2) < 1.5, (mean(b, 2), mean(b, 3))
    # the value -> problem-space transform is lifted afresh each call, so the Deterministic follows too
    np.testing.assert_array_equal(b.posterior["sel"].values, b.posterior["theta"].values[..., 3])


def test_a_replaced_ld_fn_is_honoured_by_the_compiled_sites_and_is_one_static_argument():
    s = PfCheesHMC(hier(1, 1), seed=0)
    default_mask = s._usable_mask({n: jnp.zeros((4,) + np.shape(v)) for n, v in
                                   s.model.initial_point().items() if n in s.rvs})
    assert bool(jnp.all(default_mask))
    # not finite wherever mu is above 100: the replacement must reach the jitted site, not be bypassed
    s.ld_fn = lambda pos: -0.5 * jnp.sum(pos["z"] ** 2) + jnp.where(pos["mu"] > 100.0, -jnp.inf, 0.0)
    ip = {n: v for n, v in s.model.initial_point().items() if n in s.rvs}
    stack = {n: jnp.zeros((4,) + np.shape(v)) for n, v in ip.items()}
    stack["mu"] = jnp.array([0.0, 200.0, 1.0, 300.0])
    assert np.array_equal(np.asarray(s._usable_mask(stack)), [True, False, True, False])
    stack["mu"] = jnp.array([500.0, 0.0, 0.0, 1.0])
    with count_compiles() as cc:
        again = s._usable_mask(stack)
    assert cc.count == 0 and np.array_equal(np.asarray(again), [False, True, True, True])
    s.ld_fn = lambda pos: -jnp.inf                        # another replacement is another program
    assert not bool(jnp.any(s._usable_mask(stack)))
