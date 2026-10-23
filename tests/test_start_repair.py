"""F2g-1: the start repair decides "usable" with the jitted `_usable_mask`, not with an eager `ld_fn` and `grad`.

What a wrong answer here costs: GEV's start is at xi = 0, where the gradient is NaN by design, so every GEV call
repairs its one shared start. Evaluated eagerly, the log-density and its gradient compiled one XLA program per
primitive (156 programs, 4.4 s on CPU, about 9 s on the GPU, in a cold call) and were dispatched op by op in every
warm call (0.21 s of GEV's 0.55 s on the GPU). Any model whose start has a non-finite log-density or gradient paid it.

The decision, the repaired point and so the draws must not change. The old code is a `git archive` of the F2 head
before this task (`BASE_SHA`); `start_repair_run.py` runs `_ensure_finite_inits` on one case in a fresh process in
either tree, so the compile counts are cold counts (in this process the primitives of other tests are compiled
already).
"""
import concurrent.futures
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

from pfcheeshmc.sampler import PfCheesHMC

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
BASE_SHA = "087b54e"             # the f2 head before F2g
CASES = ("gev", "gev_c4_clean", "gev_c4_bad", "fence", "last_resort", "clean_64")
# Cold-call compiles of `_ensure_finite_inits` on GEV: 162 before this task (156 of them primitives of the eager
# check), 16 now (the `usable_mask` program and the repair's own key, jitter and ravel ops).
COLD_COMPILE_BOUND = 40


def _env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2",
               PYTENSOR_FLAGS="blas__ldflags=-llapack -lblas -lcblas -lm")
    env.pop("JAX_PLATFORM_NAME", None)        # set by the example notebooks when another test ran one in-process
    return env


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


def _run_case(tree, case, out):
    done = subprocess.run([sys.executable, str(HERE / "start_repair_run.py"), str(tree), case, str(out)],
                          env=dict(_env(), PYTHONPATH=str(tree)), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    report["arrays"] = dict(np.load(out))
    return report


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("repair")
    base = _base_tree(tmp)
    jobs = {(case, name): (tree, case, tmp / f"{name}_{case}.npz")
            for case in CASES for name, tree in (("old", base), ("new", REPO))}
    with concurrent.futures.ThreadPoolExecutor(max(2, min(len(jobs), (os.cpu_count() or 4) // 3))) as pool:
        futures = {key: pool.submit(_run_case, *args) for key, args in jobs.items()}
        return {key: f.result() for key, f in futures.items()}


# ------------------------------------------------------------------------ compiles
def test_cold_repair_of_gevs_start_compiles_a_handful_of_programs(cases):
    names = cases[("gev", "new")]["compiles"]["first"]
    assert names.count("jit(usable_mask)") == 1, "the one-row check must reuse the program that screened the start"
    assert len(names) <= COLD_COMPILE_BOUND, f"{len(names)} programs: {sorted(names)}"
    assert cases[("gev", "new")]["compiles"]["second"] == []
    # the premise: the old check was the cost, and the start did need repair
    old = cases[("gev", "old")]["compiles"]["first"]
    assert len(old) > 4 * COLD_COMPILE_BOUND, f"the old code compiled {len(old)} programs: the premise is gone"
    first, start = cases[("gev", "new")]["arrays"]["first/ξ_interval__"], cases[("gev", "new")]["arrays"]["input/ξ_interval__"]
    assert start[0] == 0.0 and first[0] != 0.0, "GEV's start (xi = 0) was not repaired"


@pytest.mark.parametrize("case", ["gev_c4_clean", "clean_64"])
def test_nothing_to_repair_compiles_only_the_check(cases, case):
    """No key split, no row slices, no stack: the screening program is the only compile, and the rows come back."""
    assert cases[(case, "new")]["compiles"]["first"] == ["jit(usable_mask)"]
    assert len(cases[(case, "old")]["compiles"]["first"]) > 2, "the old loop sliced and stacked every row"


def test_a_repair_with_several_rows_compiles_the_one_row_check_once(cases):
    """C > 1 rows: the screen is a (C, ...) program and the repair's check a (1, ...) one, once per shape."""
    for case in ("fence", "gev_c4_bad"):
        names = cases[(case, "new")]["compiles"]["first"]
        assert names.count("jit(usable_mask)") == 2 and len(names) <= COLD_COMPILE_BOUND, (case, names)
        assert cases[(case, "new")]["compiles"]["second"] == []


# ------------------------------------------------------------------------ the same point
@pytest.mark.parametrize("case", CASES)
def test_the_repaired_stack_is_bit_identical_to_the_old_code(cases, case):
    old, new = cases[(case, "old")]["arrays"], cases[(case, "new")]["arrays"]
    assert sorted(old) == sorted(new)
    for key in old:
        assert old[key].dtype == new[key].dtype and np.array_equal(old[key], new[key], equal_nan=True), key


def test_the_cases_reach_the_backtrack_the_restart_and_the_last_resort(cases):
    """A repaired point shows which branch of `_find_finite_start` made it."""
    fence = cases[("fence", "new")]["arrays"]["first/x"]
    np.testing.assert_allclose(fence[0], [3.0, 0.25], rtol=1e-6)              # 12 -> 6 (fenced) -> 3: backtrack
    np.testing.assert_array_equal(fence[1], [0.5, 0.7])                       # usable: untouched
    for row in (2, 3):                                                       # fenced strip around x1 = 0: restart
        assert abs(fence[row, 1]) >= 0.01 and np.all(np.abs(fence[row]) < 1.0), fence[row]
    assert not np.array_equal(fence[2], fence[3]), "one key per row"
    assert np.all(np.abs(cases[("last_resort", "new")]["arrays"]["first/x"]) < 1e-6)   # the search from the origin


def test_gev_rows_on_the_singularity_are_repaired_and_the_others_are_not(cases):
    arrays = cases[("gev_c4_bad", "new")]["arrays"]
    xi_in, xi_out = arrays["input/ξ_interval__"], arrays["first/ξ_interval__"]
    assert xi_in[1] == xi_in[3] == 0.0 and xi_out[1] != 0.0 and xi_out[3] != 0.0 and xi_out[1] != xi_out[3]
    np.testing.assert_array_equal(xi_out[[0, 2]], xi_in[[0, 2]])
    for name in ("μ", "σ_log__"):
        np.testing.assert_array_equal(arrays[f"first/{name}"][[0, 2]], arrays[f"input/{name}"][[0, 2]])


# ------------------------------------------------------------------------ the same draws
# pm_chains = chains: four identical starts on the singularity, each repaired under its own key. (1 < pm_chains <
# chains failed on 087b54e too, and is tested in `tests/test_pm_chains.py` since F2h.)
DRAW_CASES = {"gev": [], "gev+pm_chains=C": ["--chains", "4", "--pathfinder", json.dumps({"pm_chains": 4})]}


@pytest.fixture(scope="module")
def draws(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("repair_draws")
    base = _base_tree(tmp)

    def run(tree, name, extra):
        out = tmp / f"{tree.name}_{name.replace('+', '_').replace('=', '')}.npz"
        done = subprocess.run([sys.executable, str(HERE / "compile_equiv_run.py"), str(tree), "gev", str(out), *extra],
                              env=dict(_env(), PYTHONPATH=str(tree)), capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-2000:]
        return dict(np.load(out))

    jobs = {(name, label): (tree, name, extra) for name, extra in DRAW_CASES.items()
            for label, tree in (("old", base), ("new", REPO))}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futures = {key: pool.submit(run, *args) for key, args in jobs.items()}
        return {key: f.result() for key, f in futures.items()}


@pytest.mark.parametrize("name", list(DRAW_CASES))
def test_gev_draws_for_a_fixed_seed_are_bit_identical_to_the_old_code(draws, name):
    old, new = draws[(name, "old")], draws[(name, "new")]
    assert sorted(old) == sorted(new) and len(old) > 10
    for key in old:
        if key == "config":
            assert json.loads(str(new[key])) == json.loads(str(old[key]))
        else:
            assert old[key].dtype == new[key].dtype and np.array_equal(old[key], new[key], equal_nan=True), key


# ------------------------------------------------------------------------ in-process behaviour
def _sampler(d=2):
    with pm.Model() as model:
        pm.Normal("x", shape=(d,))
    return PfCheesHMC(model, seed=3)


def test_nothing_to_repair_returns_the_stack_untouched():
    s = _sampler()
    stack = {"x": jnp.asarray(np.random.default_rng(0).normal(size=(5, 2)))}
    out = s._ensure_finite_inits(stack, jax.random.key(0))
    assert out["x"] is stack["x"]


def test_an_instance_override_of_ld_fn_is_what_the_repair_asks():
    """The default log-density is finite everywhere; the override is -inf beyond 100. Only the override sends the
    start at 300 to 75 (300 -> 150 is still fenced, 150 -> 75 is not)."""
    s = _sampler()
    s.ld_fn = lambda pos: -0.5 * jnp.sum(pos["x"] ** 2) + jnp.where(pos["x"][0] > 100.0, -jnp.inf, 0.0)
    start = {"x": jnp.array([300.0, 1.0])}
    fixed = s._find_finite_start(start, jax.random.key(0))
    np.testing.assert_allclose(np.asarray(fixed["x"]), [75.0, 0.25], rtol=1e-6)
    # a replacement of `_ld_fn` is honoured the same way
    t = _sampler()
    t._ld_fn = lambda seq: -0.5 * jnp.sum(seq[0] ** 2) + jnp.where(seq[0][0] > 100.0, -jnp.inf, 0.0)
    np.testing.assert_allclose(np.asarray(t._find_finite_start(start, jax.random.key(0))["x"]), [75.0, 0.25], rtol=1e-6)


def test_a_check_that_cannot_be_jitted_is_decided_eagerly_with_the_same_answer(monkeypatch):
    s = _sampler()
    s.ld_fn = lambda pos: -0.5 * jnp.sum(pos["x"] ** 2) + jnp.where(pos["x"][0] > 100.0, -jnp.inf, 0.0)
    stack = {"x": jnp.array([[300.0, 1.0], [1.0, 1.0]])}
    jitted = s._ensure_finite_inits(stack, jax.random.key(0))

    def cannot_run(self, init_stack):
        raise RuntimeError("the jitted check cannot run")
    monkeypatch.setattr(PfCheesHMC, "_usable_mask", cannot_run)
    eager = s._ensure_finite_inits(stack, jax.random.key(0))
    # with no batched check every row counts as unusable, so the usable row 1 is jittered (as it always was)
    np.testing.assert_array_equal(np.asarray(eager["x"][0]), np.asarray(jitted["x"][0]))
    np.testing.assert_allclose(np.asarray(eager["x"][0]), [75.0, 0.25], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(eager["x"][1]), [1.0, 1.0], atol=1e-6)


def test_only_failing_rows_are_repaired_and_each_gets_its_own_key():
    s = _sampler()
    s.ld_fn = lambda pos: -0.5 * jnp.sum(pos["x"] ** 2) + jnp.where(jnp.abs(pos["x"][0]) > 100.0, -jnp.inf, 0.0)
    stack = {"x": jnp.array([[0.1, 0.2], [300.0, 1.0], [0.3, 0.4], [300.0, 1.0]])}
    key = jax.random.key(5)
    calls = []
    original = s._find_finite_start
    s._find_finite_start = lambda x0, k, **kw: (calls.append(k), original(x0, k, **kw))[1]
    out = np.asarray(s._ensure_finite_inits(stack, key)["x"])
    keys = jax.random.split(key, 4)
    assert len(calls) == 2
    for used, row in zip(calls, (1, 3)):
        assert jax.random.key_data(used).tolist() == jax.random.key_data(keys[row]).tolist(), f"row {row}"
    np.testing.assert_array_equal(out[[0, 2]], np.asarray(stack["x"])[[0, 2]])
    assert np.all(np.abs(out[[1, 3], 0]) < 100.0) and out[1, 1] != out[3, 1]
