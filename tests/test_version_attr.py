"""idata.attrs["pfcheeshmc_version"] names the release that made the draws, and the golden files leave it out (a
version bump must not change them)."""
import numpy as np
import pymc as pm

import pfcheeshmc


def test_the_version_is_recorded_and_left_out_of_the_golden_files():
    with pm.Model() as model:
        pm.Normal("x", shape=(2,))
    idata = pfcheeshmc.sample_pf_chees_hmc(model, draws=2, chains=2, random_seed=1,
                                           chees_kwargs={"draws": 2}, pathfinder_kwargs={"draws": 8})
    assert idata.attrs["pfcheeshmc_version"] == pfcheeshmc.__version__ != "unknown"

    import make_golden
    out = {}
    make_golden._collect(idata, "", out)
    assert not any("pfcheeshmc_version" in k for k in out)
    assert "attrs/frozen_step_size" in out and np.isfinite(out["attrs/frozen_step_size"])
