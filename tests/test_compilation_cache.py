"""F2e-a, F2g-3: the on-disk compilation cache -- the explicit helper, and the default that is on -- across fresh
Python processes (CPU).

What a wrong answer here would silently cost: every new Python process compiles every program again (about 10 s
of a 12 s first call for 8 Schools at 16 chains), although none of them depends on the data or the priors. The
only honest test is a second PROCESS, as JAX's in-memory caches hide everything within one. The cache is never
enabled in the pytest process (`tests/conftest.py` sets the opt-out): other tests count compiles, and a cache
hit is still a compile event (``/jax/core/compile/backend_compile_duration`` wraps cache retrieval too,
`jax/_src/pxla.py:1533`), so the cache events are what is counted here:

    /jax/compilation_cache/compile_requests_use_cache   one per XLA compile request
    /jax/compilation_cache/cache_hits                   served from the directory
    /jax/compilation_cache/cache_misses                 compiled, and written to the directory

Process 1 writes the cache for one model; process 2 rebuilds the model with NEW data AND NEW prior values at the
same shapes (`seeded_models.hier`) and must be served entirely from it.

The default is tested the same way, with HOME (or XDG_CACHE_HOME) pointed into a temporary directory and the
opt-out removed from the children's environment, so nothing touches the real home directory. The rules of the
default (opt-out, a directory of the user's own, thresholds, permissions, activation after other programs were
compiled) are checked in one child of plain `jax.jit` programs, `compilation_cache_policy_run.py`.
"""
import concurrent.futures
import json
import os
import pathlib
import stat
import subprocess
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RUNNER = HERE / "compilation_cache_run.py"
OPT_OUT = "PFCHEESHMC_COMPILATION_CACHE"
THRESHOLDS = {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0", "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1"}


def child_env(default_cache=False, **extra):
    """The children's environment: no JAX settings but the platform, and (tests/conftest.py) the default cache OFF.

    ``default_cache=True`` removes the opt-out and XDG_CACHE_HOME, so a child that does not name HOME and
    XDG_CACHE_HOME in ``extra`` would write to the real ~/.cache: always name HOME."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("JAX_") or k == "JAX_PLATFORMS"}
    if default_cache:
        assert "HOME" in extra
        env.pop(OPT_OUT, None)
        env.pop("XDG_CACHE_HOME", None)
    env.update(PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2",
               PYTENSOR_FLAGS="blas__ldflags=-llapack -lblas -lcblas -lm", **extra)
    return env


def run(tmp, label, via, cache_dir, seed, default_cache=False, **env):
    out = tmp / f"{label}.npz"
    done = subprocess.run([sys.executable, str(RUNNER), str(REPO), via, str(cache_dir), str(seed), str(out)],
                          env=child_env(default_cache, **env), capture_output=True, text=True)
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
    homes = {n: tmp / f"home_{n}" for n in ("default", "xdg", "theirs", "optout", "early")}
    for home in homes.values():
        home.mkdir()
    default_dir = lambda home: home / ".cache" / "pfcheeshmc" / "jax"      # noqa: E731
    default = dict(default_cache=True)
    chains = {
        "helper": lambda: [run(tmp, "helper1", "helper", helper_dir, 1), run(tmp, "helper2", "helper", helper_dir, 2)],
        "control": lambda: [run(tmp, "control2", "none", tmp / "nowhere", 2)],
        "env": lambda: [run(tmp, "env1", "none", env_dir, 1, **env_all), run(tmp, "env2", "none", env_dir, 2, **env_all)],
        "dir_only": lambda: [run(tmp, "dir1", "none", dir_only_dir, 1, **dir_only),
                             run(tmp, "dir2", "none", dir_only_dir, 2, **dir_only)],
        # the default: nothing is set up by the child; the sampler call switches the cache on in HOME/.cache
        "default": lambda: [run(tmp, "default1", "none", default_dir(homes["default"]), 1, HOME=str(homes["default"]), **default),
                            run(tmp, "default2", "none", default_dir(homes["default"]), 2, HOME=str(homes["default"]), **default)],
        "xdg": lambda: [run(tmp, "xdg1", "none", tmp / "xdg_cache" / "pfcheeshmc" / "jax", 1, HOME=str(homes["xdg"]),
                            XDG_CACHE_HOME=str(tmp / "xdg_cache"), **default)],
        # a directory of the user's own: ours is not created and none of their settings moves
        "theirs": lambda: [run(tmp, "theirs1", "none", tmp / "theirs_cache", 1, HOME=str(homes["theirs"]),
                               JAX_COMPILATION_CACHE_DIR=str(tmp / "theirs_cache"), **default)],
        "optout": lambda: [run(tmp, "optout2", "none", default_dir(homes["optout"]), 2, HOME=str(homes["optout"]),
                               **{OPT_OUT: "0"}, **default)],
        "early": lambda: [run(tmp, "early1", "early", default_dir(homes["early"]), 1, HOME=str(homes["early"]), **default)],
    }
    with concurrent.futures.ThreadPoolExecutor(len(chains)) as pool:
        futures = {name: pool.submit(chain) for name, chain in chains.items()}
        results = {name: f.result() for name, f in futures.items()}
    results["homes"] = homes
    results["tmp"] = tmp
    return results


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
    assert os.environ.get(OPT_OUT) == "0", "tests/conftest.py opts the whole test session out of the default cache"
    assert jax.config.jax_compilation_cache_dir is None, "other tests count compiles; never enable the cache here"


# ------------------------------------------------------------------------------------ the default (F2g-3)
def test_the_default_is_on_it_creates_a_private_directory_under_home_and_writes_the_programs(runs):
    first, _ = runs["default"]
    c = counts(first)
    assert c["compile_requests_use_cache"] > 50
    assert c["cache_misses"] >= c["compile_requests_use_cache"] - 5, c
    assert first["files_before"] == 0 and first["files_after"] >= c["cache_misses"] - 2
    assert first["mode"] == 0o700, oct(first["mode"])
    parent = runs["homes"]["default"] / ".cache" / "pfcheeshmc"
    assert stat.S_IMODE(parent.stat().st_mode) == 0o700, "the whole tree we create is private"
    assert first["settings"]["jax_compilation_cache_dir"] == str(parent / "jax")
    assert first["settings"]["jax_persistent_cache_min_compile_time_secs"] == 0.0
    assert first["settings"]["jax_persistent_cache_min_entry_size_bytes"] == -1
    assert first["settings"]["jax_compilation_cache_max_size"] == -1, "no size limit is set for the user"


def test_a_second_process_with_new_data_hits_the_default_cache(runs):
    first, second = runs["default"]
    c = counts(second)
    assert c["compile_requests_use_cache"] == counts(first)["compile_requests_use_cache"] > 50
    assert c["cache_hits"] >= c["compile_requests_use_cache"] - 1 and c["cache_misses"] == 0, c
    assert second["files_after"] == second["files_before"] == first["files_after"]
    assert second["seconds"]["backend_compile_duration"] < 0.3 * first["seconds"]["backend_compile_duration"]


def test_draws_are_bit_equal_with_the_default_cache_on_and_off(runs):
    (off,), (_, on) = runs["optout"], runs["default"]
    for name in ("mu", "tau"):
        np.testing.assert_array_equal(off["draws"][name], on["draws"][name])
        np.testing.assert_array_equal(off["draws"][name], runs["control"][0]["draws"][name])


def test_the_opt_out_means_no_directory_no_hits_and_no_writes(runs):
    (off,) = runs["optout"]
    c = counts(off)
    assert c["compile_requests_use_cache"] > 50
    assert c["cache_hits"] == 0 and c["cache_misses"] == 0 and off["files_after"] == 0, c
    assert not (runs["homes"]["optout"] / ".cache" / "pfcheeshmc").exists(), "the opt-out must not even create the directory"
    assert off["settings"]["jax_compilation_cache_dir"] is None


def test_xdg_cache_home_moves_the_default_and_home_is_left_alone(runs):
    (first,) = runs["xdg"]
    assert first["files_after"] > 50 and first["mode"] == 0o700
    assert not (runs["homes"]["xdg"] / ".cache" / "pfcheeshmc").exists()


def test_a_cache_directory_of_the_users_own_is_respected_and_none_of_their_settings_moves(runs):
    (first,) = runs["theirs"]
    c = counts(first)
    assert not (runs["homes"]["theirs"] / ".cache" / "pfcheeshmc").exists(), "our default directory must not be created"
    assert first["files_after"] >= 1                             # their directory is used (JAX's own thresholds: a handful)
    assert first["files_after"] <= 0.1 * c["compile_requests_use_cache"], "the thresholds are theirs: left at JAX's defaults"
    assert first["settings"] == {
        "jax_compilation_cache_dir": str(runs["tmp"] / "theirs_cache"),
        "jax_persistent_cache_min_compile_time_secs": 1.0, "jax_persistent_cache_min_entry_size_bytes": 0,
        "jax_compilation_cache_max_size": -1, "jax_enable_compilation_cache": True}


def test_programs_compiled_before_the_first_sampler_call_do_not_stop_the_cache(runs):
    (first,) = runs["early"]
    c = counts(first)
    assert c["cache_misses"] >= c["compile_requests_use_cache"] - 10 and first["files_after"] > 50, (c, first["files_after"])


def test_importing_the_package_changes_no_cache_setting_and_creates_no_directory(tmp_path):
    code = (
        "import sys, jax; before = dict(jax.config.values)\n"
        "sys.path.insert(0, sys.argv[1]); import pfcheeshmc, pfcheeshmc.sampler, pfcheeshmc.compilation_cache\n"
        "after = dict(jax.config.values)\n"
        "print({k: [before[k], after[k]] for k in before if before[k] != after[k]})\n"
    )
    done = subprocess.run([sys.executable, "-c", code, str(REPO)], env=child_env(True, HOME=str(tmp_path)),
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-1500:]
    changed = eval(done.stdout.strip().splitlines()[-1])            # (importing PyTensor's JAX backend enables x64)
    assert not [k for k in changed if "cache" in k], changed
    assert not (tmp_path / ".cache" / "pfcheeshmc").exists(), "importing created the cache directory"


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("policy")
    done = subprocess.run([sys.executable, str(HERE / "compilation_cache_policy_run.py"), str(REPO), str(scratch)],
                          env=child_env(), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2500:]
    return json.loads(done.stdout.strip().splitlines()[-1]) | {"scratch": scratch}


def test_policy_the_default_directory_thresholds_and_idempotence(policy):
    r = policy["fresh"]
    assert r["on"] and r["warned"] == []
    assert r["dir"] == r["function_dir"] and r["dir"].endswith("/.cache/pfcheeshmc/jax")
    assert r["settings"] == {
        "jax_compilation_cache_dir": r["dir"], "jax_persistent_cache_min_compile_time_secs": 0.0,
        "jax_persistent_cache_min_entry_size_bytes": -1, "jax_compilation_cache_max_size": -1,
        "jax_enable_compilation_cache": True}
    assert r["mode"] == {"leaf": 0o700, "parent": 0o700}
    assert r["again"] == {"on": True, "warned": [], "same_settings": True}


@pytest.mark.parametrize("value", ["0", "false", "No", " OFF "])
def test_policy_the_opt_out_switch_leaves_everything_alone(policy, value):
    r = policy[f"optout[{value}]"]
    assert r["on"] is False and r["warned"] == [] and not r["home_has_cache"]
    assert r["settings"]["jax_compilation_cache_dir"] is None
    assert r["settings"]["jax_persistent_cache_min_compile_time_secs"] == 1.0


@pytest.mark.parametrize("value", ["1", "", "yes"])
def test_policy_any_other_value_of_the_switch_is_not_an_opt_out(policy, value):
    assert policy[f"not_optout[{value}]"]["on"] is True


def test_policy_a_directory_the_user_set_is_theirs_with_every_setting(policy):
    r = policy["user_dir"]
    assert r["on"] is False and r["unchanged"] and r["warned"] == []
    assert not r["home_has_cache"] and not r["theirs_created"], "we neither create ours nor touch theirs"
    assert r["settings"]["jax_persistent_cache_min_compile_time_secs"] == 1.0
    assert r["settings"]["jax_persistent_cache_min_entry_size_bytes"] == 0


def test_policy_a_user_who_disabled_jaxs_cache_keeps_it_disabled(policy):
    r = policy["user_disabled"]
    assert r["on"] is False and not r["home_has_cache"]
    assert r["settings"]["jax_compilation_cache_dir"] is None and r["settings"]["jax_enable_compilation_cache"] is False


def test_policy_thresholds_and_size_limit_the_user_set_are_kept(policy):
    r = policy["user_thresholds"]
    assert r["on"] and r["settings"]["jax_persistent_cache_min_compile_time_secs"] == 5.0
    assert r["settings"]["jax_persistent_cache_min_entry_size_bytes"] == 100
    r = policy["user_max_size"]
    assert r["on"] and r["settings"]["jax_compilation_cache_max_size"] == 123456789


def test_policy_a_directory_others_can_write_is_not_used_and_says_so(policy):
    r = policy["unsafe_mode"]
    assert r["on"] is False and r["dir_setting"] is None
    assert len(r["warned"]) == 1 and r["warned"][0].startswith("RuntimeWarning") and "written by group or others" in r["warned"][0]
    r = policy["readable_by_others"]                 # reading is not writing: a 0755 directory of ours is fine, and kept
    assert r["on"] and r["warned"] == [] and r["mode_kept"] == 0o755


def test_policy_a_directory_that_cannot_be_created_warns_and_never_raises(policy):
    r = policy["cannot_create"]
    assert r["on"] is False and r["dir_setting"] is None
    assert len(r["warned"]) == 1 and "cannot be created" in r["warned"][0]


def test_policy_xdg_cache_home_must_be_absolute(policy):
    scratch = policy["scratch"]
    assert policy["xdg"]["dir_setting"] == str(scratch / "xdg_cache" / "pfcheeshmc" / "jax") and not policy["xdg"]["home_has_cache"]
    for label in ("relative", "empty"):
        r = policy[f"xdg_{label}"]
        assert r["dir_setting"] == f"{r['home']}/.cache/pfcheeshmc/jax", label


def test_policy_the_directory_can_be_set_after_other_programs_were_compiled(policy):
    """JAX decides at its first compile whether a cache is in use from the platform and `jax_enable_compilation_cache`,
    not from the directory, and looks for the directory at every compile until it finds one: no reset is needed."""
    r = policy["after_other_programs"]
    assert r["on"] and r["misses_before"] == 0 and r["misses_after"] >= 1 and r["entries"] >= 1, r


def test_policy_an_explicit_directory_chosen_after_the_default_was_in_use_takes_over(policy):
    r = policy["explicit_after_default"]
    assert r["entries_default_before"] >= 1
    assert r["entries_default_after"] == r["entries_default_before"] and r["entries_mine"] >= 2, r
    assert r["settings"]["jax_persistent_cache_min_compile_time_secs"] == 0.0


def test_policy_the_explicit_helper_works_whatever_the_opt_out_says(policy):
    r = policy["explicit_with_optout"]
    assert r["entries_mine"] >= 1 and r["default_on"] is False
