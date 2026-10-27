"""C1/C3: the ladder models of `benchmarks/models.py`: the ladder is the four models, and building one sets no JAX option.

The proof that the Python-built models are the notebook-built ones of the tag `archive/pre-cleanup` (value variables, the
logp / dlogp graphs, and the draws of `sample_pf_chees_hmc`) was a set of `git archive` children; the draws of all four are
now pinned by `tests/test_golden.py`, whose files equal the tag's. float32 behaviour (x64 stays off) is in `test_float32.py`.
"""
import os
import pathlib
import sys

import jax
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.models import MODELS, build_model  # noqa: E402

LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")


def test_the_ladder_is_the_four_models():
    assert MODELS == LADDER
    with pytest.raises(KeyError, match="no_such_model"):
        build_model("no_such_model")


def test_build_model_changes_no_jax_option_and_no_environment_variable():
    """The notebook cells the old loader executed set ``JAX_PLATFORM_NAME=gpu`` and ``jax_enable_x64`` for the process."""
    def snapshot():
        return {"jax": {n: repr(getattr(jax.config, n)) for n in sorted(dir(jax.config)) if n.startswith("jax_")},
                "env": dict(os.environ)}

    before = snapshot()
    for name in LADDER:
        build_model(name)
    after = snapshot()
    assert len(before["jax"]) > 50
    changed = {f"{section}:{k}": [before[section].get(k), after[section].get(k)]
               for section in before for k in sorted({*before[section], *after[section]})
               if before[section].get(k) != after[section].get(k)}
    assert changed == {}


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
