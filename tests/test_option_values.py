"""Option values are checked before any stage runs (`sampler._check_option_values`), with the stages' own checks, so a
bad value is not found after Pathfinder and the warmup; and a stage's warnings point at the caller's line."""
import warnings

import numpy as np
import pymc as pm
import pytest

from pfcheeshmc import sample_pf_chees_hmc
from pfcheeshmc.sampler import PfCheesHMC


@pytest.mark.parametrize("kw, match", [
    ({"dynamic_hmc_kwargs": {"delayed_rejection": 1}}, "delayed_rejection factor"),
    ({"dynamic_hmc_kwargs": {"delayed_rejection": {"factor": 4, "retry": "sometimes"}}}, "retry must be one of"),
    ({"dynamic_hmc_kwargs": {"leapfrog_unroll": 0}}, "leapfrog_unroll"),
    ({"chees_kwargs": {"metric_updates": (0.5,), "metric_estimator": "nutpie"}}, "metric_estimator must be one of"),
    ({"chees_kwargs": {"metric_estimator": "fisher"}}, "needs metric_updates"),
    ({"chees_kwargs": {"metric_updates": (0.5,), "metric_max_rank": 0}}, "metric_max_rank"),
    ({"chees_kwargs": {"metric_updates": (1.5,)}}, "metric_updates fractions"),
    ({"chees_kwargs": {"metric_updates": (0.5,), "metric_shrinkage": 2.0}}, "metric_shrinkage"),
    ({"chees_kwargs": {"metric_updates": (0.5,), "warmup_stop": {"block": 10}}}, "cannot be combined"),
    ({"chees_kwargs": {"acceptance_floor": 0.9}}, "acceptance_floor"),
    ({"chees_kwargs": {"trajectory_length": -1.0}}, "trajectory_length"),
])
def test_a_bad_value_raises_before_any_stage_runs(kw, match, monkeypatch):
    def stage(*a, **k):
        raise AssertionError("a stage ran before the option values were checked")
    monkeypatch.setattr(PfCheesHMC, "_run_pathfinder", stage)
    with pm.Model() as model:
        pm.Normal("x")
    with pytest.raises(ValueError, match=match):
        sample_pf_chees_hmc(model, draws=5, chains=8, random_seed=1, progressbar=False, **kw)


def test_a_stage_warning_points_at_the_callers_line():
    """Pathfinder's L-BFGS cap warning (raised in a stage method) names the line that called the sampler."""
    with pm.Model() as model:                      # mode away from the start, so L-BFGS has a path to run
        x = pm.Normal("x", 0.0, 1e3, shape=3)
        pm.Normal("y", x, np.array([1e-2, 1.0, 1e2]), observed=np.array([0.5, -2.0, 300.0]))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sample_pf_chees_hmc(model, draws=5, chains=8, random_seed=1, progressbar=False, chees_kwargs={"draws": 5},
                            pathfinder_kwargs={"pf_maxiter": 2})
    capped = [r for r in rec if "used all 2 of its iterations" in str(r.message)]
    assert capped and capped[0].filename == __file__, [(str(r.message)[:40], r.filename) for r in rec]
