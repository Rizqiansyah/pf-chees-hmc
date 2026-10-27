"""C2: the experiment options are gone, the default path is the one it was, and an option that does not exist raises.

Removed (the PLAN's "Q3" list): ``imm_orientation`` and ``imm_order`` (the inverted metric and the Pathfinder-order
permutation, which reproduced known bugs), the seven ``metric_refresh*`` options and the two-phase warmup with them
(``pfcheeshmc/ensemble_metric.py`` is gone), ``superchains`` (and the nested R-hat that read them), ``iterate``,
``init_pick``, ``pf_start_scale`` and ``pf_center`` (and the ``center`` argument of the start repair).

  * `test_every_removed_option_raises_a_type_error_that_names_it`: each one, in the dict of the stage that used to read it,
    through `PfCheesHMC.sample` (before any stage runs) and through the stage method; a misspelt name raises the same.
The draws of the default path and of the options it keeps are pinned by `tests/test_golden.py`, whose files equal what the
tag `archive/pre-cleanup` (b34e66e) produced.
"""
import pathlib
import sys

import pymc as pm
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402

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
