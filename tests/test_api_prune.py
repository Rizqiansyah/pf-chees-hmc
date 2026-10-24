"""C2: the experiment options are gone, the default path is the one it was, and an option that does not exist raises.

Removed (the PLAN's "Q3" list): ``imm_orientation`` and ``imm_order`` (the inverted metric and the Pathfinder-order
permutation, which reproduced known bugs), the seven ``metric_refresh*`` options and the two-phase warmup with them
(``pfcheeshmc/ensemble_metric.py`` is gone), ``superchains`` (and the nested R-hat that read them), ``iterate``,
``init_pick``, ``pf_start_scale`` and ``pf_center`` (and the ``center`` argument of the start repair).

  * `test_every_removed_option_raises_a_type_error_that_names_it`: each one, in the dict of the stage that used to read it,
    through `PfCheesHMC.sample` (before any stage runs) and through the stage method; a misspelt name raises the same.
  * `test_the_draws_are_bit_identical_to_the_tag`: `sample_pf_chees_hmc` in two fresh processes, one in a `git archive` of the
    tag `archive/pre-cleanup` (b34e66e) and one in this tree (`api_prune_run.py`): the four ladder models at seeds 1 and 2 on
    the default path, and the options the default path keeps (every `imm_source`, `pm_chains` 4 and C, `init_source`,
    `pm_init`, the pool cap, the start-repair knobs, `warmup_stop`, ...). 16 chains, 5 draws, 20 warmup draws. Every
    posterior and sample_stats array and every numeric attribute must be equal to the last bit; the resolved config must
    be equal once the keys of the removed options (which the tag still records) are left out. The test skips where git or
    the tag is not available.
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
import pymc as pm
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402

TAG = "archive/pre-cleanup"
LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")
# the keys of idata.attrs["config"] that record a removed option (the tag writes them, this tree does not)
REMOVED_CONFIG_KEYS = (
    "imm_orientation", "imm_order", "pf_start_scale", "init_pick", "superchains", "iterate", "metric_refresh",
    "metric_refresh_warmup", "metric_refresh_pool", "metric_refresh_reseed", "metric_refresh_ridge",
    "metric_refresh_shrink", "metric_refresh_rank", "metric_refresh_eps_probes")

def without_removed(config):
    """A resolved config of a tree before C2 (the tag, or an older one) minus the keys of the removed options."""
    return {k: v for k, v in config.items() if k not in REMOVED_CONFIG_KEYS}


# removed option -> the dict of the stage that read it, and a value that was valid there
REMOVED = {
    "imm_orientation": ("pathfinder_kwargs", "precision"),
    "imm_order": ("pathfinder_kwargs", "pathfinder"),
    "superchains": ("pathfinder_kwargs", 4),
    "init_pick": ("pathfinder_kwargs", "iid"),
    "pf_start_scale": ("pathfinder_kwargs", 0.5),
    "pf_center": ("pathfinder_kwargs", None),
    "iterate": ("chees_kwargs", "raw"),
    "metric_refresh": ("chees_kwargs", "dense"),
    "metric_refresh_warmup": ("chees_kwargs", 50),
    "metric_refresh_pool": ("chees_kwargs", 20),
    "metric_refresh_reseed": ("chees_kwargs", False),
    "metric_refresh_ridge": ("chees_kwargs", 1e-6),
    "metric_refresh_shrink": ("chees_kwargs", 0.5),
    "metric_refresh_rank": ("chees_kwargs", 5),
}
STAGE_METHOD = {"pathfinder_kwargs": "_run_pathfinder", "chees_kwargs": "_run_chees"}


# ------------------------------------------------------------------------------------ the options are rejected
@pytest.fixture(scope="module")
def sampler():
    with pm.Model() as model:
        pm.Normal("x", shape=2)
    return PfCheesHMC(model, seed=1)


@pytest.mark.parametrize("option", sorted(REMOVED))
def test_every_removed_option_raises_a_type_error_that_names_it(sampler, option):
    """`sample` refuses it before a stage runs; the stage method, called on its own, refuses it too."""
    where, value = REMOVED[option]
    with pytest.raises(TypeError, match=rf"{where} got an unknown option '{option}'"):
        sampler.sample(chains=4, draws=2, **{where: {option: value}})
    stage = getattr(sampler, STAGE_METHOD[where])
    args = (None, None, 0.1) if where == "chees_kwargs" else ()
    with pytest.raises(TypeError, match=rf"unknown option '{option}'"):
        stage(*args, **{option: value})


@pytest.mark.parametrize("where", ["pathfinder_kwargs", "step_size_tune_kwargs", "chees_kwargs", "dynamic_hmc_kwargs"])
def test_a_misspelt_option_raises_in_every_stage_and_names_them_all(sampler, where):
    with pytest.raises(TypeError) as err:
        sampler.sample(chains=4, draws=2, **{where: {"pm_chain": 2, "adam_rl": 0.1}})
    assert f"{where} got an unknown option 'adam_rl', 'pm_chain'" in str(err.value)
    assert "The options are:" in str(err.value)


def test_sample_refuses_a_keyword_it_does_not_read(sampler):
    with pytest.raises(TypeError, match="sample got an unknown option 'imm_orientation'"):
        sampler.sample(chains=4, draws=2, imm_orientation="precision")


def test_the_public_function_has_no_open_keywords():
    import inspect
    from pfcheeshmc import sample_pf_chees_hmc
    kinds = {p.kind for p in inspect.signature(sample_pf_chees_hmc).parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds
    with pytest.raises(TypeError, match="imm_orientation"):
        sample_pf_chees_hmc(draws=2, chains=4, imm_orientation="precision")


def test_the_removed_modules_and_helpers_are_gone():
    import pfcheeshmc.sampler as sampler_mod
    import pfcheeshmc.utils as utils_mod
    assert not (REPO / "pfcheeshmc" / "ensemble_metric.py").exists()
    for name in ("_run_chees_two_phase", "_pool_gradients", "_scale_starts", "_pathfinder_mean"):
        assert not hasattr(PfCheesHMC, name), name
    for name in ("_concat_warmup_info", "estimate_covariance", "reseed_trajectory_length", "whitened_lambda_max"):
        assert not hasattr(sampler_mod, name), name
    assert "positions" not in utils_mod.WarmupInfo._fields


# ------------------------------------------------------------------------------------ the default path is unchanged
def _env():
    flags = [f for f in os.environ.get("PYTENSOR_FLAGS", "").split(",") if f and not f.startswith("floatX=")]
    if not any(f.startswith("blas__ldflags=") for f in flags):
        flags.append("blas__ldflags=-llapack -lblas -lcblas -lm")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2",
               PYTENSOR_FLAGS=",".join(flags), PFCHEESHMC_COMPILATION_CACHE="0")
    env.pop("JAX_PLATFORM_NAME", None)
    return env


def _child(tree, group, out):
    done = subprocess.run(
        [sys.executable, str(HERE / "api_prune_run.py"), str(tree), str(REPO / "benchmarks" / "models.py"), group, str(out)],
        env=dict(_env(), PYTHONPATH=str(tree)), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
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
    dest = tmp_path_factory.mktemp("api_prune_tag") / "tag"
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest, filter="data")
    return dest


@pytest.fixture(scope="module")
def runs(tag_tree, tmp_path_factory):
    """The four children (two groups, two trees), started at once: label -> future of (meta, arrays)."""
    tmp = tmp_path_factory.mktemp("api_prune")
    pool = concurrent.futures.ThreadPoolExecutor(4)
    futures = {(which, group): pool.submit(_child, tree, group, tmp / f"{which}_{group}.npz")
               for which, tree in (("tag", tag_tree), ("new", REPO)) for group in ("default", "kept")}
    yield futures
    pool.shutdown(wait=True)


def _config(array, drop=()):
    cfg = json.loads(str(array))
    return {k: v for k, v in cfg.items() if k not in drop}


def _compare(old, new):
    """(arrays compared, the keys that differ): bit equality, dtype and shape included; the config minus the removed keys."""
    assert sorted(old) == sorted(new)
    different = []
    for key in sorted(old):
        a, b = old[key], new[key]
        if key.endswith("/config"):
            same = _config(a, REMOVED_CONFIG_KEYS) == _config(b, ())
        elif a.dtype.kind in "US":
            same = str(a) == str(b)
        else:
            same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        if not same:
            different.append(key)
    return len(old), different


@pytest.mark.parametrize("group", ["default", "kept"])
def test_the_draws_are_bit_identical_to_the_tag(runs, group):
    old_meta, old = runs[("tag", group)].result()
    new_meta, new = runs[("new", group)].result()
    assert old_meta == new_meta and new_meta["x64"] is True
    n, different = _compare(old, new)
    assert different == [], f"{len(different)} of {n} differ: {different[:6]}"
    if group == "default":
        assert [c for c in new_meta["cases"]] == [f"{m}/seed{s}" for m in LADDER for s in (1, 2)]
    else:
        assert {f"8schools/imm_source={s}" for s in ("mc", "lbfgs_lowrank", "fisher_lowrank", "pf_closed")} <= set(
            new_meta["cases"])
        assert {"8schools/pm_chains=4", "8schools/pm_chains=4+jitter",
                                                   "8schools/pm_chains=C+jitter"} <= set(new_meta["cases"])
    assert n >= 15 * len(new_meta["cases"])


def test_the_cases_are_not_all_the_same_draw(runs):
    """The comparison is not between constants: the seeds, the models and the metric sources give different draws."""
    _, new = runs[("new", "default")].result()
    assert not np.array_equal(new["8schools/seed1/posterior/mu"], new["8schools/seed2/posterior/mu"])
    _, kept = runs[("new", "kept")].result()
    mu = {c: kept[f"8schools/{c}/posterior/mu"] for c in (
        "imm_source=mc", "imm_source=lbfgs_lowrank", "imm_source=fisher_lowrank", "imm_source=pf_closed",
        "pm_chains=4+jitter", "pm_chains=C+jitter", "init_source=adapt_diag", "warmup_stop_fires")}
    assert len({m.tobytes() for m in mu.values()}) == len(mu), "two cases gave the same draws"
    assert all(np.isfinite(m).all() for m in mu.values())
    assert kept["8schools/warmup_stop_fires/attrs/warmup_steps_used"] == 10     # the stop rule path ran and stopped


def test_the_tag_records_the_options_this_tree_removed(runs):
    """The config comparison drops these keys from the tag's record; they must be there to drop, and not in this tree's."""
    _, old = runs[("tag", "default")].result()
    _, new = runs[("new", "default")].result()
    old_cfg, new_cfg = _config(old["8schools/seed1/config"], ()), _config(new["8schools/seed1/config"], ())
    assert set(REMOVED_CONFIG_KEYS) <= set(old_cfg)
    assert not set(REMOVED_CONFIG_KEYS) & set(new_cfg)
    assert {k: old_cfg[k] for k in ("imm_orientation", "imm_order", "init_pick", "iterate", "metric_refresh")} == {
        "imm_orientation": "covariance", "imm_order": "kernel", "init_pick": "stratified", "iterate": "ma",
        "metric_refresh": "none"}
