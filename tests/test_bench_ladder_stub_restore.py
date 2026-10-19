"""load_notebook_model stubs pm.sample and pfcheeshmc.sample_pf_chees_hmc while it execs the
notebook (the committed examples really sample). Those stubs return a tiny fake InferenceData, so
if they leak, any later call in the same process silently produces a plausible but void result.
This pins the restore. CPU only, no sampling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


def test_notebook_load_restores_sampling_entry_points():
    import pfcheeshmc
    import pymc as pm
    import bench_ladder as bl

    real_pf, real_pm = bl.REAL_SAMPLE_PF, bl.REAL_PM_SAMPLE
    assert pfcheeshmc.sample_pf_chees_hmc is real_pf, "stubbed before the load: fixture order changed"
    bl.model_specs()["8schools"]()
    assert pfcheeshmc.sample_pf_chees_hmc is real_pf, "load_notebook_model leaked the sample_pf_chees_hmc stub"
    assert pm.sample is real_pm, "load_notebook_model leaked the pm.sample stub"


def test_loaded_model_has_observed_rvs():
    import bench_ladder as bl

    model = bl.model_specs()["8schools"]()
    assert len(model.observed_RVs) == 1
