"""F2b-1: a second sampler call compiles nothing outside the warmup and sampling loops (CPU).

What a wrong answer here would silently cost: every `sample()` call recompiles XLA programs (15.9-17.4 s of
a 19.9-21.9 s call for 8 Schools at C=1,024) because jitted closures are rebuilt per call and per-run values
(data, priors, metric, step size) are compiled in as constants. Nothing in the draws would show it, so the
only honest test counts compiles: `compile_counter.count_compiles` listens to JAX's backend-compile event.

Sites (`pfcheeshmc.compiled`, plus the ensemble estimators): the start screening and scoring, the
Pathfinder fit and draws, the step-size search, the pool gradients, the warmup `batch_init`, and the
value -> problem-space transform. Their loops are F2b-2's: `LOOPS` below is the number of loop programs a
second call still compiles, and F2b-2 takes it to 0.

Equivalence with the F2 head before this task (`BASE_SHA`) is NOT bit-for-bit, and cannot be:
  * Pathfinder, the start scoring and the transform used to run op by op (eager); each is now one fused
    XLA program, which changes floating-point results by an ulp (measured: draws eager vs jit differ by
    <= 1.1e-15, the L-BFGS fit by <= 2.2e-16 on 8 Schools and 7e-15 on logit);
  * the log-density used to see data and priors as constants, which XLA folds and simplifies; as
    arguments they are not (8 Schools start scores differ by 7e-15, GEV's by 1.4e-14, between the two).
The ulp is then carried through chaotic trajectories, so draws agree to about 1e-10 relative at this size.
The test pins the discrete statistics exactly and the continuous ones tightly.
"""
import concurrent.futures
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import types

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

from compile_counter import count_compiles
from seeded_models import hier

from pfcheeshmc import compiled
from pfcheeshmc.ensemble_metric import ENGLISH_NAMES, estimate_covariance, positions_to_draws
from pfcheeshmc.lifted import lift_logp
from pfcheeshmc.sampler import PfCheesHMC, sample_pf_chees_hmc

REPO = pathlib.Path(__file__).resolve().parents[1]
C, W, D = 16, 20, 10
LOOPS = 2            # warmup scan + sampling scan: what a second call still compiles until F2b-2
LOOP_NAME = "jit(scan)"
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
    pool = {n: jnp.stack([pf[n]] * 3) for n in pf}                      # (3, chains, ...)
    with _site(counts, "pool_gradients"):
        s._pool_gradients(pool, positions_to_draws(pool, 3), {"score_chunk": 40})   # chunks of 40 and a rest
    ld = s._logdensity()
    with _site(counts, "batch_init"):
        compiled.batch_init(ld.fn, ld.names, ld.values, pf, 0)
    hist = types.SimpleNamespace(position={n: jnp.stack([pf[n]] * D) for n in pf})    # (draws, chains, ...)
    with _site(counts, "transform"):
        s._transform_value_samples_to_problem_space(hist)
    return counts


def sample_call(model, seed, progressbar=False, pathfinder_kwargs=None, **chees):
    with count_compiles() as cc:
        idata = sample_pf_chees_hmc(
            model, draws=D, chains=C, random_seed=seed, progressbar=progressbar,
            chees_kwargs={"draws": W, **chees}, pathfinder_kwargs=pathfinder_kwargs)
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
@pytest.mark.parametrize("progressbar", [False, True])
def test_second_sample_call_compiles_only_the_loop_programs(progressbar):
    sample_call(hier(1, 1), seed=1, progressbar=progressbar)
    idata, cc = sample_call(hier(5, 6), seed=2, progressbar=progressbar)       # new seed, data and priors
    assert sorted(cc.names) == [LOOP_NAME] * LOOPS, str(cc)
    assert idata.posterior["mu"].shape == (C, D)


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


# ------------------------------------------------------------------------------- 5. paths
IMM_SOURCES = ["pf_closed", "lbfgs_lowrank", "fisher_lowrank"]


@pytest.mark.parametrize("imm_source", IMM_SOURCES)
def test_imm_sources_compile_nothing_on_a_second_invocation(imm_source):
    kw = dict(imm_source=imm_source, max_rank=2)
    run_sites(hier(1, 1), seed=1, **kw)
    assert_no_compiles(run_sites(hier(2, 3), seed=2, **kw), imm_source)


def test_superchains_compile_nothing_on_a_second_invocation():
    kw = dict(superchains=4)
    run_sites(hier(1, 1), seed=1, **kw)
    assert_no_compiles(run_sites(hier(2, 3), seed=2, **kw), "superchains")


@pytest.mark.parametrize("estimator", sorted(ENGLISH_NAMES))
def test_every_ensemble_estimator_compiles_nothing_on_a_second_call(estimator):
    d = 6
    key = jax.random.key(0)
    draws = jax.random.normal(key, (40, d), dtype=jnp.float64)
    grads = -draws + 0.1 * jax.random.normal(jax.random.key(1), (40, d), dtype=jnp.float64)
    estimate_covariance(estimator, draws, grads)
    draws2, grads2 = 1.5 * draws + 0.2, 1.5 * grads - 0.1
    with count_compiles() as cc:
        estimate_covariance(estimator, draws2, grads2)
    assert cc.count == 0, str(cc)


def test_metric_refresh_path_second_call_compiles_only_the_loop_programs():
    refresh = dict(metric_refresh="fisher_diag", metric_refresh_warmup=10)
    sample_call(hier(1, 1), seed=1, **refresh)
    idata, cc = sample_call(hier(5, 6), seed=2, **refresh)
    # phase A (position-stacking scan), phase B, and the sampling scan
    assert sorted(cc.names) == [LOOP_NAME] * (LOOPS + 1), str(cc)
    assert idata.attrs["metric_refresh"] == "fisher_diag"


def test_superchain_path_second_call_compiles_only_the_loop_programs():
    kw = dict(pathfinder_kwargs={"superchains": 4})
    sample_call(hier(1, 1), seed=1, **kw)
    _, cc = sample_call(hier(5, 6), seed=2, **kw)
    assert sorted(cc.names) == [LOOP_NAME] * LOOPS, str(cc)


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


# ---------------------------------------------------------------------- 4. equivalence
BASE_SHA = "1eda92c"            # the f2 head before F2b-1
EQUIV_CASES = {                  # name -> (model, extra args for compile_equiv_run.py)
    "8schools": ("8schools", []),
    "gev": ("gev", []),
    "logit_1000x25": ("logit_1000x25", []),
    "radon_noncentered": ("radon_noncentered", []),
    "8schools+refresh": ("8schools", ["--chees", json.dumps(
        {"metric_refresh": "fisher_diag", "metric_refresh_warmup": 10})]),
    "8schools+pf_closed+superchains": ("8schools", ["--pathfinder", json.dumps(
        {"imm_source": "pf_closed", "superchains": 4})]),
}
EQUIV_RTOL = 1e-7        # relative to (1 + |x|); observed worst 1.2e-10 at C=8, W=20, D=10, x64


def _base_tree(tmp):
    git = ["git", "-C", str(REPO)]
    try:
        subprocess.run([*git, "cat-file", "-e", f"{BASE_SHA}^{{commit}}"], check=True, capture_output=True)
        blob = subprocess.run([*git, "archive", BASE_SHA], check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {BASE_SHA} is not available here (git archive failed)")
    dest = tmp / "base"
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest)
    return dest


def _run_in_tree(tree, model, out, extra):
    env = dict(os.environ, PYTHONPATH=str(tree), CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="2", PYTENSOR_FLAGS="blas__ldflags=-llapack -lblas -lcblas -lm")
    # The example notebooks the ladder models are built from set JAX_PLATFORM_NAME=gpu in os.environ
    # when another test has executed one in this process; a CPU-only child must not inherit that.
    env.pop("JAX_PLATFORM_NAME", None)
    cmd = [sys.executable, str(pathlib.Path(__file__).with_name("compile_equiv_run.py")),
           str(tree), model, str(out), *extra]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return np.load(out)


@pytest.fixture(scope="module")
def equivalence(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("equiv")
    base = _base_tree(tmp)
    jobs = {(case, tree_name): (tree, model, tmp / f"{tree_name}_{case.replace('+', '_')}.npz", extra)
            for case, (model, extra) in EQUIV_CASES.items()
            for tree_name, tree in (("old", base), ("new", REPO))}
    workers = max(2, min(len(jobs), (os.cpu_count() or 4) // 3))
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {key: pool.submit(_run_in_tree, *args) for key, args in jobs.items()}
        return {key: f.result() for key, f in futures.items()}


@pytest.mark.parametrize("case", list(EQUIV_CASES))
def test_draws_match_the_f2_head_before_this_task(equivalence, case):
    old, new = equivalence[(case, "old")], equivalence[(case, "new")]
    assert sorted(old.files) == sorted(new.files)
    worst = 0.0
    for key in old.files:
        a, b = old[key], new[key]
        if key == "config":
            continue
        assert a.shape == b.shape, key
        if a.dtype.kind in "biu":                       # steps, divergences, accepts: a flip is a different run
            assert np.array_equal(a, b), key
            continue
        rel = np.abs(a - b) / (1.0 + np.abs(a))
        worst = max(worst, float(np.nanmax(rel)))
        assert np.all(rel <= EQUIV_RTOL), f"{key}: worst relative difference {np.nanmax(rel):.3e}"
    old_cfg, new_cfg = json.loads(str(old["config"])), json.loads(str(new["config"]))
    assert old_cfg.keys() == new_cfg.keys()
    for k, v in old_cfg.items():                        # every resolved setting; floats only up to the ulp
        if isinstance(v, float):
            assert new_cfg[k] == pytest.approx(v, rel=EQUIV_RTOL), k
        else:
            assert new_cfg[k] == v, k
