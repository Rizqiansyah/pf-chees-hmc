"""F2k: the four ladder models run end to end under ``PYTENSOR_FLAGS=floatX=float32``, and float64 is as it was.

float32 is PyMC's own switch. PyTensor's JAX dispatch sets ``jax_enable_x64 = (floatX == "float64")`` when it is
imported, so every float32 run is a fresh process (`float32_run.py`, with the variable in its environment).

What a wrong answer here costs:
  * the smoke test of 2026-10-24 failed on all four models with "cond branches must have equal output types", inside
    BlackJAX's L-BFGS. The value that optax's line search keeps in its state has the dtype of the position (float32);
    a fresh evaluation has the dtype of the log-density, and that was float64: the two example notebooks (8 Schools,
    GEV) run ``jax.config.update("jax_enable_x64", True)`` in their first cell, `tools/r1_bias.build_model` executes
    them, and 8 Schools' data are float64 NumPy arrays that PyMC does not cast. The first model turned x64 on for the
    whole process, so the other three failed behind it. Nothing in the sampler produces the mismatch (a process whose
    x64 was off all along ran logit_1000x25 on the code before this task). `build_model` now puts x64 back to
    ``floatX == "float64"``, and the sampler says so when the two disagree;
  * a float32 run that finishes with a NaN, or whose posterior is float64 (something converted it back);
  * a warning that tells a float32 user to switch x64 on ("Explicitly requested dtype ... is not available");
  * a float64 run whose draws move by one bit. The code before this task is a `git archive` (`BASE_SHA`), run in fresh
    processes.
"""
import concurrent.futures
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest
import scipy.special as sps

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
BASE_SHA = "62c38a3"             # the f2 head before F2k
LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")
SOURCES = "pf,adapt_diag"        # init_source values: the Pathfinder pool pick, and the (jittered) PyMC start
IDENTITY_MODELS = ("8schools", "gev")


def _env(floatx):
    """The child's environment: the caller's ``PYTENSOR_FLAGS`` with ``floatX`` (and the BLAS link line) appended."""
    flags = [f for f in os.environ.get("PYTENSOR_FLAGS", "").split(",") if f and not f.startswith("floatX=")]
    if not any(f.startswith("blas__ldflags=") for f in flags):
        flags.append("blas__ldflags=-llapack -lblas -lcblas -lm")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2",
               PYTENSOR_FLAGS=",".join([*flags, f"floatX={floatx}"]), PFCHEESHMC_COMPILATION_CACHE="0")
    env.pop("JAX_PLATFORM_NAME", None)        # set by the example notebooks when another test ran one in-process
    return env


def _child(tree, name, out, floatx, *opts):
    done = subprocess.run([sys.executable, str(HERE / "float32_run.py"), str(tree), name, str(out), *opts],
                          env=dict(_env(floatx), PYTHONPATH=str(tree)), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    z = np.load(out)
    return json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}


class Jobs:
    """Children started at once (a few threads), their results read by label."""

    def __init__(self, tmp, jobs, workers=4):
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)
        self.futures = {label: self.pool.submit(_child, tree, name, tmp / f"{label}.npz", floatx, *opts)
                        for label, (tree, name, floatx, opts) in jobs.items()}

    def __getitem__(self, label):
        return self.futures[label].result()

    def close(self):
        self.pool.shutdown(wait=True)


@pytest.fixture(scope="module")
def float32_jobs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("float32")
    jobs = {name: (REPO, name, "float32", ("--init-sources", SOURCES)) for name in LADDER}
    jobs["mixed"] = (REPO, "8schools", "float32", ("--init-sources", "pf", "--x64-after-build"))
    run = Jobs(tmp, jobs)
    yield run
    run.close()


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory):
    git = ["git", "-C", str(REPO)]
    try:
        subprocess.run([*git, "cat-file", "-e", f"{BASE_SHA}^{{commit}}"], check=True, capture_output=True)
        blob = subprocess.run([*git, "archive", BASE_SHA], check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {BASE_SHA} is not available here (git archive failed)")
    dest = tmp_path_factory.mktemp("float32_base") / "base"
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest, filter="data")
    return dest


@pytest.fixture(scope="module")
def float64_jobs(tmp_path_factory, base_tree):
    tmp = tmp_path_factory.mktemp("float64")
    jobs = {f"{label}:{name}": (tree, name, "float64", ("--init-sources", SOURCES))
            for name in IDENTITY_MODELS for label, tree in (("old", base_tree), ("new", REPO))}
    run = Jobs(tmp, jobs)
    yield run
    run.close()


# ------------------------------------------------------------------------------------ float32, end to end
@pytest.mark.parametrize("name", LADDER)
def test_the_ladder_model_samples_under_float32(float32_jobs, name):
    """Finishes, finite draws, float32 posterior, x64 off, and no warning that advises switching x64 on."""
    meta, arrays = float32_jobs[name]
    assert meta["floatX"] == "float32" and meta["x64_start"] is False
    assert meta["x64_built"] is False, "building the model turned jax_enable_x64 on (an example notebook's cell)"
    assert meta["x64_end"] is False
    assert meta["errors"] == {}, meta["errors"]
    assert meta["warnings"] == [], meta["warnings"][:2]
    assert meta["x64_warnings"] == []
    for source in SOURCES.split(","):
        posterior = {k: v for k, v in arrays.items() if k.startswith(f"{source}/posterior/")}
        stats = {k: v for k, v in arrays.items() if k.startswith(f"{source}/stats/")}
        assert posterior and stats, source
        for key, value in posterior.items():
            assert value.dtype == np.float32, (key, value.dtype)
            assert value.shape[:2] == (8, 10), (key, value.shape)
            assert np.isfinite(value).all(), key
        assert arrays[f"{source}/stats/lp"].dtype == np.float32 and np.isfinite(arrays[f"{source}/stats/lp"]).all()
        assert set(meta["dtypes"][source].values()) <= {"float32", "bool", "int32"}, meta["dtypes"][source]


def test_the_two_notebook_models_leave_x64_off_under_float32(float32_jobs):
    """The leaf mismatch of the smoke test. `build_model` runs the notebook cell that turns x64 on; after it, x64 must
    be what floatX says (off), or the log-density of a float32 position is float64 and L-BFGS fails in `lax.cond`."""
    for name in ("8schools", "gev"):
        meta, _ = float32_jobs[name]
        assert meta["x64_built"] is False and meta["errors"] == {}, (name, meta["x64_built"], meta["errors"])


def test_x64_on_under_float32_is_said_not_left_to_a_jax_type_error(float32_jobs):
    """A session that turns x64 on itself (an example notebook's first cell) under floatX=float32: the sampler warns
    before it samples, and names the switch. (Whether the call then finishes is JAX's and BlackJAX's business.)"""
    meta, _ = float32_jobs["mixed"]
    assert len(meta["x64_warnings"]) == 1 and meta["x64_end"] is True
    text = meta["x64_warnings"][0]
    assert "floatX is float32" in text and 'jax.config.update("jax_enable_x64", False)' in text


# ------------------------------------------------------------------------------------ float64, bit for bit
@pytest.mark.parametrize("name", IDENTITY_MODELS)
def test_float64_draws_are_bit_identical_to_the_code_before(float64_jobs, name):
    """Both init sources, one call each: every posterior and sample_stats array equal to the last bit, same dtypes,
    same resolved config. (8 Schools has the notebook's float64 data; GEV the start repair and the erfcx.)"""
    old_meta, old = float64_jobs[f"old:{name}"]
    new_meta, new = float64_jobs[f"new:{name}"]
    assert sorted(old) == sorted(new) and len(old) >= 20
    for source in SOURCES.split(","):
        assert any(k.startswith(f"{source}/posterior/") for k in old) and any(k.startswith(f"{source}/stats/") for k in old)
    for key in old:
        assert old[key].dtype == new[key].dtype and np.array_equal(old[key], new[key], equal_nan=True), key
    assert old_meta["dtypes"] == new_meta["dtypes"]
    assert {s: json.loads(c) for s, c in new_meta["config"].items()} == {s: json.loads(c) for s, c in old_meta["config"].items()}
    for key in new:
        if "/posterior/" in key:
            assert new[key].dtype == np.float64 and np.isfinite(new[key]).all(), key
    # the float32 guard in `build_model` must not touch a float64 session: the notebook's x64 stays on
    assert new_meta["floatX"] == "float64" and new_meta["x64_built"] is True and new_meta["x64_end"] is True
    assert new_meta["x64_warnings"] == []


# ------------------------------------------------------------------------------------ in-process pieces
def test_the_halton_probe_of_the_frozen_length_needs_no_int64():
    """`_frozen_nominal_L` asked for an int64 array, which JAX truncates (and warns) when x64 is off."""
    from pfcheeshmc import compiled
    from pfcheeshmc.sampler import _frozen_nominal_L

    with jax.enable_x64(False), warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _frozen_nominal_L(lambda arg: compiled.integration_steps(arg, jnp.float32(7.5)), 100)
    with jax.enable_x64(True):
        want = _frozen_nominal_L(lambda arg: compiled.integration_steps(arg, jnp.float64(7.5)), 100)
    assert got == want == 8.0


def test_the_start_repair_keeps_the_dtype_of_the_start():
    """The jitter and restart noise of `_find_finite_start` took JAX's default float: with x64 on, a float32 start made
    a float64 flat vector and `ravel_pytree`'s unravel refused it."""
    from pfcheeshmc.sampler import PfCheesHMC

    with pm.Model() as model:
        pm.Normal("x", 0.0, 1.0, shape=3)
    sampler = PfCheesHMC(model)
    with jax.enable_x64(True):
        repaired = sampler._find_finite_start({"x": jnp.zeros(3, dtype=jnp.float32)}, jax.random.key(0), jitter_eps=1e-3)
    assert repaired["x"].dtype == jnp.float32


def test_the_stratified_pick_never_starts_a_chain_at_a_non_finite_density():
    """float32 GEV: the Pathfinder pool had a row outside the support (``1 + xi z <= 0``: ``-inf`` in float64 too; the
    float64 fit of the same seed had no such row), rank 0 of the sort took it, and chain 0 stayed on it for all of its
    draws (``lp`` -inf, every proposal rejected, counted as divergent). A pool of finite rows is ranked as before."""
    from pfcheeshmc.sampler import _stratified_pool_idx

    subset = jnp.arange(8) * 10
    lp = np.array([2.0, -np.inf, 5.0, 1.0, np.nan, -3.0, 4.0, 0.0])
    assert np.asarray(_stratified_pool_idx(jnp.asarray(lp), subset, 3)).tolist() == [50, 30, 60]
    assert np.asarray(_stratified_pool_idx(jnp.asarray(lp), subset, 6)).tolist() == [50, 70, 30, 0, 60, 20]
    rng = np.random.default_rng(0)
    finite = rng.normal(size=64)
    old = np.asarray(jnp.arange(0, 640, 10)[jnp.argsort(jnp.asarray(finite))[(jnp.arange(16) * 64) // 16]])
    assert np.array_equal(np.asarray(_stratified_pool_idx(jnp.asarray(finite), jnp.arange(0, 640, 10), 16)), old)
    nowhere = np.full(8, -np.inf)
    assert np.asarray(_stratified_pool_idx(jnp.asarray(nowhere), subset, 4)).shape == (4,)


def test_erfcx_in_float32_matches_scipy_over_its_domain():
    """JAX's own erfcx is 0 (or NaN) for float32 in [9.19455, 9.41928]; the guard repairs it, and the band that
    `lifted._band` gives for float32 is that one. Away from the band JAX is within 4e-6 of SciPy."""
    from pfcheeshmc import lifted

    x = np.unique(np.concatenate([
        np.linspace(-9.0, 30.0, 400001), np.linspace(9.0, 9.6, 60001), np.geomspace(30.0, 1e5, 2001)])).astype(np.float32)
    want = sps.erfcx(x.astype(np.float64))
    with jax.enable_x64(False):
        got = np.asarray(jax.jit(lifted._erfcx_guarded)(jnp.asarray(x)))
        stock = np.asarray(jax.jit(jax.scipy.special.erfcx)(jnp.asarray(x)))
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=4e-6, atol=0)
    wrong = ~np.isclose(stock, want, rtol=1e-5)
    assert wrong.any() and 9.18 < x[wrong].min() and x[wrong].max() < 9.43, "premise: stock JAX is wrong in the band"
    lo, hi = lifted._band(np.dtype("float32"))
    assert lo < x[wrong].min() and x[wrong].max() < hi
