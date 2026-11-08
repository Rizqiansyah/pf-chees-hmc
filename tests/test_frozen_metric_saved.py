"""The metric the draws were made with travels with them (``idata.attrs["frozen_inverse_mass_matrix"]``, flattened,
and its shape), so a divergence can be replayed with the sampler's own metric. It must be the warmup's final metric,
in kernel order, and survive a netCDF round trip."""
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
import arviz as az
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.seeded_models import hier  # noqa: E402

from pfcheeshmc.sampler import sample_pf_chees_hmc


def test_the_frozen_metric_is_saved_and_round_trips(tmp_path):
    idata = sample_pf_chees_hmc(hier(1, 1), draws=10, chains=16, random_seed=1, progressbar=False,
                                chees_kwargs={"draws": 40})
    m = np.asarray(idata.attrs["frozen_inverse_mass_matrix"])
    shape = tuple(int(s) for s in idata.attrs["frozen_inverse_mass_matrix_shape"])
    d = sum(int(np.size(v)) for v in hier(1, 1).initial_point().values())   # unconstrained dimensions
    assert shape in [(d,), (d, d)]
    assert m.shape == (int(np.prod(shape)),) and np.all(np.isfinite(m))
    if len(shape) == 2:
        M = m.reshape(shape)
        np.testing.assert_allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)
    else:
        assert np.all(m > 0)
    az.to_netcdf(idata, tmp_path / "x.nc")
    back = az.from_netcdf(tmp_path / "x.nc")
    np.testing.assert_array_equal(np.asarray(back.attrs["frozen_inverse_mass_matrix"]), m)
