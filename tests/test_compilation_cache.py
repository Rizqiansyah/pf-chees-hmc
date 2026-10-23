"""F2e-a: the opt-in on-disk compilation cache, across fresh Python processes (CPU).

What a wrong answer here would silently cost: every new Python process compiles every program again (about 10 s
of a 12 s first call for 8 Schools at 16 chains), although none of them depends on the data or the priors. The
only honest test is a second PROCESS, as JAX's in-memory caches hide everything within one. The cache is never
enabled in the pytest process: other tests count compiles, and a cache hit is still a compile event
(``/jax/core/compile/backend_compile_duration`` wraps cache retrieval too, `jax/_src/pxla.py:1533`), so the
cache events are what is counted here:

    /jax/compilation_cache/compile_requests_use_cache   one per XLA compile request
    /jax/compilation_cache/cache_hits                   served from the directory
    /jax/compilation_cache/cache_misses                 compiled, and written to the directory

Process 1 writes the cache for one model; process 2 rebuilds the model with NEW data AND NEW prior values at the
same shapes (`seeded_models.hier`) and must be served entirely from it.
"""
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RUNNER = HERE / "compilation_cache_run.py"
THRESHOLDS = {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0", "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1"}


def child_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("JAX_") or k == "JAX_PLATFORMS"}
    env.update(PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2",
               PYTENSOR_FLAGS="blas__ldflags=-llapack -lblas -lcblas -lm", **extra)
    return env


def run(tmp, label, via, cache_dir, seed, **env):
    out = tmp / f"{label}.npz"
    done = subprocess.run([sys.executable, str(RUNNER), str(REPO), via, str(cache_dir), str(seed), str(out)],
                          env=child_env(**env), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    report["draws"] = dict(np.load(out))
    return report


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Four independent two-process chains, run side by side:
    helper   `enable_compilation_cache(dir)`, then the same call with other data and priors;
    control  the same second call with no cache directory at all;
    env      JAX's own environment variables (the zero-code route), all three set;
    dir_only the environment variable for the directory alone."""
    tmp = tmp_path_factory.mktemp("cache")
    helper_dir, env_dir, dir_only_dir = (tmp / n for n in ("helper_cache", "env_cache", "dir_only_cache"))
    env_all = {"JAX_COMPILATION_CACHE_DIR": str(env_dir), **THRESHOLDS}
    dir_only = {"JAX_COMPILATION_CACHE_DIR": str(dir_only_dir)}
    chains = {
        "helper": lambda: [run(tmp, "helper1", "helper", helper_dir, 1), run(tmp, "helper2", "helper", helper_dir, 2)],
        "control": lambda: [run(tmp, "control2", "none", tmp / "nowhere", 2)],
        "env": lambda: [run(tmp, "env1", "none", env_dir, 1, **env_all), run(tmp, "env2", "none", env_dir, 2, **env_all)],
        "dir_only": lambda: [run(tmp, "dir1", "none", dir_only_dir, 1, **dir_only),
                             run(tmp, "dir2", "none", dir_only_dir, 2, **dir_only)],
    }
    with concurrent.futures.ThreadPoolExecutor(len(chains)) as pool:
        futures = {name: pool.submit(chain) for name, chain in chains.items()}
        return {name: f.result() for name, f in futures.items()}


def counts(report):
    e = report["events"]
    return {k: e.get(k, 0) for k in ("compile_requests_use_cache", "cache_hits", "cache_misses")}


def test_first_process_compiles_and_writes_every_program(runs):
    first, _ = runs["helper"]
    c = counts(first)
    assert c["compile_requests_use_cache"] > 50, "the call compiles nothing: the premise of the test is gone"
    # a program that two sites share is compiled once and read back once within the process
    assert c["cache_misses"] >= c["compile_requests_use_cache"] - 5 and c["cache_hits"] <= 5, c
    assert first["files_before"] == 0 and first["files_after"] >= c["cache_misses"] - 2


def test_second_process_with_new_data_and_priors_is_served_entirely_from_the_cache(runs):
    first, second = runs["helper"]
    c = counts(second)
    assert c["compile_requests_use_cache"] == counts(first)["compile_requests_use_cache"] > 50
    # Every request is a hit but one: `jit(log)` (0.04 s) is compiled while the packages are imported, before
    # any call can set the directory. And nothing is compiled, so nothing is written.
    assert c["cache_hits"] >= c["compile_requests_use_cache"] - 1, c
    assert c["cache_misses"] == 0, c
    assert second["files_after"] == second["files_before"] == first["files_after"], "no new entry was written"
    assert second["seconds"]["compile_time_saved_sec"] > 0
    # a hit is still a "backend compile" event, but it is a read: the total is far below the first process's
    assert second["seconds"]["backend_compile_duration"] < 0.3 * first["seconds"]["backend_compile_duration"]


def test_second_process_draws_are_finite_and_are_those_of_its_own_data(runs):
    first, second = runs["helper"]
    for name in ("mu", "tau"):
        a, b = first["draws"][name], second["draws"][name]
        assert a.shape == b.shape and np.all(np.isfinite(b)), name
        assert not np.allclose(a, b), f"{name}: other data gave the same draws"


def test_control_without_the_switch_gets_no_cache_hits(runs):
    (control,) = runs["control"]
    c = counts(control)
    assert c["compile_requests_use_cache"] > 50
    assert c["cache_hits"] == 0 and c["cache_misses"] == 0 and control["files_after"] == 0, c
    assert control["seconds"]["backend_compile_duration"] > 3 * runs["helper"][1]["seconds"]["backend_compile_duration"]
    # the same call, with the cache: the same draws (the cache returns the executable the compile would give)
    for name in ("mu", "tau"):
        np.testing.assert_array_equal(control["draws"][name], runs["helper"][1]["draws"][name])


def test_jax_environment_variables_are_the_zero_code_route(runs):
    first, second = runs["env"]
    c = counts(second)
    assert counts(first)["cache_misses"] > 50
    assert c["cache_hits"] == c["compile_requests_use_cache"] > 50 and c["cache_misses"] == 0, c   # set before the import: all


def test_the_directory_alone_is_not_enough_because_of_the_one_second_threshold(runs):
    """Why the helper sets the thresholds: with JAX's defaults only the programs that took >= 1 s are kept."""
    first, second = runs["dir_only"]
    c = counts(second)
    assert c["compile_requests_use_cache"] > 50
    assert c["cache_hits"] <= 0.1 * c["compile_requests_use_cache"], c            # a handful of programs reach 1 s
    assert first["files_after"] <= 0.1 * c["compile_requests_use_cache"], first["files_after"]


def test_the_helper_sets_the_three_settings_and_nothing_else(tmp_path):
    code = (
        "import sys, jax; sys.path.insert(0, sys.argv[1]); import pathlib, pfcheeshmc\n"
        "before = dict(jax.config.values)\n"
        "pfcheeshmc.enable_compilation_cache(pathlib.Path(sys.argv[2]))\n"
        "after = dict(jax.config.values)\n"
        "print({k: [before[k], after[k]] for k in before if before[k] != after[k]})\n"
    )
    done = subprocess.run([sys.executable, "-c", code, str(REPO), str(tmp_path / "c")], env=child_env(),
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-1500:]
    changed = eval(done.stdout.strip().splitlines()[-1])
    assert changed == {
        "jax_compilation_cache_dir": [None, str(tmp_path / "c")],
        "jax_persistent_cache_min_compile_time_secs": [1.0, 0.0],
        "jax_persistent_cache_min_entry_size_bytes": [0, -1],
    }, changed


def test_the_pytest_process_does_not_have_the_cache_on():
    import jax
    assert jax.config.jax_compilation_cache_dir is None, "other tests count compiles; never enable the cache here"
