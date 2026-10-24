"""C1: the ladder models of `benchmarks/models.py` are the models the example notebooks built, and building one sets no JAX option.

Until C1, `8schools` and `gev` were built by executing the cells of the two example notebooks (`tools/bench_ladder.py`'s
loader, with `pm.sample` stubbed), and `logit_1000x25` and `radon_noncentered` by two functions of the same file. The
notebook cells set ``JAX_PLATFORM_NAME=gpu`` and ``jax_enable_x64`` for the whole process. The models are Python now, and
the NUTS references the bias gate scores against were made on the notebook-built ones, so a Python-built model that
differs in one detail (a prior's dtype, the order of the names) would no longer fit them. The old models are the tag
`archive/pre-cleanup` (b34e66e), a `git archive` of it, in child processes (`benchmark_models_run.py`):

  * `structure`: in one process, both builds of each model: the same value variables (name, shape, dtype, order), the same
    free, observed and deterministic variables, the same dims, coordinates and initial point, ``equal_computations`` on the
    ``logp`` graph (whole and term by term), on the ``dlogp`` graph and on the deterministics, and ``compile_logp`` and
    ``compile_dlogp`` bit-equal at 100 random points;
  * `draws`: in two fresh processes, one per tree, `sample_pf_chees_hmc` on a fresh build of each model, seeds 1 and 2
    (8 chains, 10 draws, 25 warmup draws): every posterior and sample_stats array bit-identical, and the resolved config;
  * `settings`: ``build_model`` of every model changes no ``jax_*`` option and no environment variable (x64, the platform,
    the compilation cache).

Each is run at floatX float64 and float32 (PyTensor's JAX dispatch sets ``jax_enable_x64`` from floatX when it is imported,
so floatX is chosen in the child's ``PYTENSOR_FLAGS``). The test skips where git or the tag is not available.
"""
import concurrent.futures
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from benchmarks.models import MODELS, build_model  # noqa: E402
from test_api_prune import without_removed  # noqa: E402

TAG = "archive/pre-cleanup"
LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")
FLOATX = ("float64", "float32")
STRUCTURE_CHECKS = ("value_vars", "free_RVs", "observed_RVs", "deterministics", "potentials", "basic_RVs", "unobserved_RVs",
                    "named_vars", "dims_and_coords", "initial_point", "logp_graph", "logp_terms_graph", "dlogp_graph",
                    "deterministics_graph")


def _env(floatx):
    """The child's environment: floatX (and the BLAS link line, for the path with a space) in ``PYTENSOR_FLAGS``."""
    flags = [f for f in os.environ.get("PYTENSOR_FLAGS", "").split(",") if f and not f.startswith("floatX=")]
    if not any(f.startswith("blas__ldflags=") for f in flags):
        flags.append("blas__ldflags=-llapack -lblas -lcblas -lm")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2",
               PYTENSOR_FLAGS=",".join([*flags, f"floatX={floatx}"]), PFCHEESHMC_COMPILATION_CACHE="0")
    env.pop("JAX_PLATFORM_NAME", None)
    return env


def _child(floatx, tree, *args):
    """Runs ``benchmark_models_run.py`` with PYTHONPATH=tree and returns what it wrote: the first argument that is a
    ``.json`` or ``.npz`` path (an empty string is a flag that is not given)."""
    args = [a for a in args if a != ""]
    out = next(a for a in args if str(a).endswith((".json", ".npz")))
    done = subprocess.run([sys.executable, str(HERE / "benchmark_models_run.py"), *map(str, args)],
                          env=dict(_env(floatx), PYTHONPATH=str(tree)), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    if str(out).endswith(".json"):
        return json.loads(pathlib.Path(out).read_text())
    z = np.load(out)
    return json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}


@pytest.fixture(scope="module")
def tag_tree(tmp_path_factory):
    git = ["git", "-C", str(REPO)]
    try:
        subprocess.run([*git, "cat-file", "-e", f"{TAG}^{{commit}}"], check=True, capture_output=True)
        blob = subprocess.run([*git, "archive", TAG], check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"the tag {TAG} is not available here (git archive failed)")
    dest = tmp_path_factory.mktemp("benchmark_models_tag") / "tag"
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest, filter="data")
    return dest


@pytest.fixture(scope="module")
def runs(tag_tree, tmp_path_factory):
    """Every child, started at once (a few threads): label -> future of the child's result."""
    tmp = tmp_path_factory.mktemp("benchmark_models")
    jobs = {"settings:old:float64": ("float64", tag_tree, "settings", tag_tree, tmp / "settings_old.json")}
    for fx in FLOATX:
        jobs[f"structure:{fx}"] = (fx, REPO, "structure", tag_tree, REPO, tmp / f"structure_{fx}.json")
        jobs[f"draws:old:{fx}"] = (fx, tag_tree, "draws", tag_tree, tmp / f"draws_old_{fx}.npz")
        jobs[f"draws:new:{fx}"] = (fx, REPO, "draws", REPO, tmp / f"draws_new_{fx}.npz")
        for flag in ("", "--import-sampler"):
            jobs[f"settings:new:{fx}{flag}"] = (fx, REPO, "settings", REPO, tmp / f"settings_new_{fx}{flag}.json", flag)
    pool = concurrent.futures.ThreadPoolExecutor(4)
    futures = {label: pool.submit(_child, fx, tree, *args) for label, (fx, tree, *args) in jobs.items()}
    yield futures
    pool.shutdown(wait=True)


# ------------------------------------------------------------------------------------ the same models
@pytest.mark.parametrize("floatx", FLOATX)
@pytest.mark.parametrize("name", LADDER)
def test_the_python_model_is_the_notebook_model(runs, floatx, name):
    """Value variables, variable sets, initial point, the logp / dlogp / deterministics graphs, and logp and dlogp at 100
    random points (bit-equal; mostly finite, so the equality is not one of -inf)."""
    report = runs[f"structure:{floatx}"].result()
    assert report["floatX"] == floatx
    checks = report["models"][name]
    assert [c for c in STRUCTURE_CHECKS if checks[c] is not True] == [], checks
    assert checks["points_bit_equal"] == 100, checks
    assert checks["points_finite"] >= 50, checks


@pytest.mark.parametrize("floatx", FLOATX)
@pytest.mark.parametrize("name", LADDER)
def test_the_draws_are_bit_identical_to_the_notebook_models(runs, floatx, name):
    """Seeds 1 and 2, 8 chains, 10 draws, 25 warmup: every posterior and sample_stats array and the resolved config (less
    the keys of the options C2 removed, which the tag still records), equal to the last bit between the tag
    (notebook-built models) and this tree (`benchmarks.models`)."""
    old_meta, old = runs[f"draws:old:{floatx}"].result()
    new_meta, new = runs[f"draws:new:{floatx}"].result()
    assert old_meta["floatX"] == new_meta["floatX"] == floatx and old_meta["x64"] == new_meta["x64"] == (floatx == "float64")
    keys = sorted(k for k in old if k.startswith(f"{name}/"))
    assert keys == sorted(k for k in new if k.startswith(f"{name}/")) and len(keys) >= 16
    for seed in (1, 2):
        for label in ("posterior", "stats", "config"):
            assert any(k.startswith(f"{name}/seed{seed}/{label}") for k in keys), (seed, label)
    for key in keys:
        if key.endswith("/config"):                              # the resolved config
            assert without_removed(json.loads(str(old[key]))) == json.loads(str(new[key])), key
            continue
        assert old[key].dtype == new[key].dtype, key
        if old[key].dtype.kind in "US":
            assert str(old[key]) == str(new[key]), key
        else:
            assert np.array_equal(old[key], new[key], equal_nan=True), key
        if "/posterior/" in key:
            assert np.isfinite(new[key]).all(), key
            if new[key].dtype.kind == "f":
                assert new[key].dtype == np.dtype(floatx), (key, new[key].dtype)
    first = next(k for k in keys if "seed1/posterior/" in k)
    assert not np.array_equal(new[first], new[first.replace("seed1", "seed2")]), "the two seeds gave the same draws"


# ------------------------------------------------------------------------------------ no JAX setting
@pytest.mark.parametrize("flag", ["", "--import-sampler"], ids=["before-sampler-import", "after-sampler-import"])
@pytest.mark.parametrize("floatx", FLOATX)
def test_build_model_never_changes_a_jax_option_or_an_environment_variable(runs, floatx, flag):
    """Building the four models changes no ``jax_*`` option (x64, the platform, the compilation cache) and no environment
    variable, at either floatX, before or after `pfcheeshmc` (and PyTensor's JAX dispatch) is imported."""
    report = runs[f"settings:new:{floatx}{flag}"].result()
    assert report["floatX"] == floatx and report["n_options"] > 50
    assert report["changed"] == {}, report["changed"]
    assert report["x64_before"] == report["x64_after"] == str(bool(flag) and floatx == "float64")


def test_the_check_can_fail_the_notebook_loader_changed_them(runs):
    """The premise of the test above, on the tag: its `build_model` turned x64 on, set ``JAX_PLATFORM_NAME=gpu`` and left
    threading variables behind."""
    changed = runs["settings:old:float64"].result()["changed"]
    assert changed["jax:jax_enable_x64"] == ["False", "True"], changed
    assert changed["env:JAX_PLATFORM_NAME"] == [None, "gpu"], changed


# ------------------------------------------------------------------------------------ the module itself
def test_the_ladder_is_the_four_models():
    assert MODELS == LADDER
    with pytest.raises(KeyError, match="no_such_model"):
        build_model("no_such_model")


def test_build_model_replaces_no_sampling_entry_point():
    """Loading a notebook used to replace ``pm.sample`` and ``sample_pf_chees_hmc`` with stubs that returned a fake
    InferenceData. Nothing is replaced now, and every call builds a new model."""
    import pymc as pm
    import pfcheeshmc

    before = (pm.sample, pm.sample_prior_predictive, pm.sample_posterior_predictive, pfcheeshmc.sample_pf_chees_hmc)
    first = {name: build_model(name) for name in ("8schools", "gev", "logit_1000x25")}
    second = {name: build_model(name) for name in first}
    assert before == (pm.sample, pm.sample_prior_predictive, pm.sample_posterior_predictive, pfcheeshmc.sample_pf_chees_hmc)
    for name, model in first.items():
        assert model is not second[name] and len(model.observed_RVs) == 1, name
