#!/usr/bin/env python
"""Re-derive plan §2b and add the R1 gate to it: split R-hat vs nested R-hat on the SAME data.

The file is one CONVERGED 32-chain x 2,000-draw 8 Schools run (full-run split R-hat 1.0012). Cutting
it into 1024x39, 512x20, 512x78 and 128x313 reproduces every "degraded" R-hat the repo reported for
many-short-chain runs, which is why split R-hat cannot be the R1 gate (plan §3 of the handoff, §2b).

Here each chop is also scored with nested R-hat, which compares superchains instead of chains, plus
the bias statistic against the WP0.2 reference. If nested R-hat stays near 1 while split R-hat rises,
the harness has the discrimination the repo was missing.

Two caveats recorded on purpose:
  1. a chop cannot give a superchain a shared start, so this exercise only shows nested R-hat does
     NOT false-alarm on short chains. The positive side (does it catch non-convergence when starts
     are shared, as the method requires?) is answered by the K=8 sweep cells, not by this script.
  2. this .nc stores PROBLEM-space variables (mu, theta, tau), while G-R1 is defined in unconstrained
     value-var space, so b2 is not computed here. Comparing tau against the reference's tau_log__
     would be a units error, not a bias measurement.
"""
import os
import sys

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

SRC = os.path.expanduser("~/pf-runs/8schools_pfchees_2000x32.nc")
CHOPS = [(1024, 39), (512, 78), (512, 20), (128, 313), (32, 2000)]
K = 8  # paper: K between 8 and 256 for KM = 2048 chains; 4-16 for KM = 128 (arXiv 2110.13017 v3)


def main():
    import arviz as az
    import arviz_stats as AS
    import xarray as xr
    import r1_bias as rb

    import arviz as _az

    idata = _az.from_netcdf(SRC)
    names = [str(v) for v in idata.posterior.data_vars]      # problem space: mu, theta, tau
    post = {n: np.asarray(idata.posterior[n].values, dtype="float64") for n in names}
    print(f"{SRC}: chains={post[names[0]].shape[0]} draws={post[names[0]].shape[1]} vars={names} "
          f"(problem space)")

    print(f"\n{'chop':>12} {'K':>4} {'split R-hat':>12} {'nested R-hat':>13}   verdict")
    for chains, draws in CHOPS:
        have = post[names[0]].shape[0] * post[names[0]].shape[1]
        assert have >= chains * draws, (have, chains, draws)
        flat = {n: np.asarray(v).reshape(-1, *np.shape(v)[2:])[:chains * draws]
                .reshape(chains, draws, -1) for n, v in post.items()}
        split = float(max(float(np.max(v.values)) for v in
                          az.rhat(az.convert_to_dataset(flat)).values()))
        ids = [i // (chains // K) for i in range(chains)]
        nested = float(np.asarray(AS.rhat_nested(
            np.concatenate([flat[n] for n in sorted(flat)], axis=2),
            superchain_ids=ids, chain_axis=0, draw_axis=1)).max())
        verdict = ("split says NOT converged, nested says converged"
                   if split > 1.01 >= nested else
                   "both say converged" if split <= 1.01 and nested <= 1.01 else
                   "both say not converged")
        print(f"{chains}x{draws:>5} {K:>4} {split:>12.4f} {nested:>13.4f}   {verdict}")


if __name__ == "__main__":
    main()
