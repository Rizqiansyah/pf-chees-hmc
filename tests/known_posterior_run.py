"""A sampler run on a conjugate Normal-Normal model with a known posterior, in a fresh process, against any tree.

    python known_posterior_run.py TREE OUT.npz [--chains C] [--warmup W] [--draws D] [--seed S] [--chees JSON]

TREE provides `pfcheeshmc/` (this checkout, or a `git archive` of an earlier commit). Writes the draws of
`theta`, `(chains, draws, 10)`. `analytic()` is the exact posterior, for the tests to compare against.
Everything runs in float64 (PyTensor's JAX dispatch enables x64).
"""
import argparse
import json
import os
import sys

import numpy as np

D_THETA = 10
PRIOR_MU = np.linspace(-1.0, 1.0, D_THETA)
PRIOR_SD = np.linspace(0.5, 2.0, D_THETA)
OBS = np.linspace(1.0, -1.0, D_THETA)
OBS_SE = np.linspace(1.0, 0.4, D_THETA)


def analytic():
    """Posterior mean and variance of each coordinate: theta_j ~ N(mu_j, s_j^2), y_j ~ N(theta_j, se_j^2)."""
    precision = 1.0 / PRIOR_SD ** 2 + 1.0 / OBS_SE ** 2
    var = 1.0 / precision
    return var * (PRIOR_MU / PRIOR_SD ** 2 + OBS / OBS_SE ** 2), var


def build():
    import pymc as pm

    with pm.Model() as model:
        theta = pm.Normal("theta", PRIOR_MU, PRIOR_SD, shape=D_THETA)
        pm.Normal("y", theta, OBS_SE, observed=OBS)
    return model


if __name__ == "__main__":
    os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("out")
    ap.add_argument("--chains", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chees", default="{}", help="extra chees_kwargs, JSON")
    args = ap.parse_args()
    sys.path.insert(0, args.tree)
    from pfcheeshmc import sample_pf_chees_hmc

    idata = sample_pf_chees_hmc(
        build(), draws=args.draws, chains=args.chains, random_seed=args.seed, progressbar=False,
        chees_kwargs={"draws": args.warmup, **json.loads(args.chees)})
    np.savez(args.out, theta=np.asarray(idata.posterior["theta"].values),
             div=np.asarray(idata.sample_stats["diverging"].values),
             steps=np.asarray(idata.sample_stats["num_integration_steps"].values))
