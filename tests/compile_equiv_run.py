"""One sampler run for the F2b-1 equivalence test, in a fresh process, against any source tree.

    python compile_equiv_run.py TREE MODEL OUT.npz [--chains C] [--warmup W] [--draws D] [--x64]

TREE is a source tree that provides `pfcheeshmc/` and `tools/` (the checkout, or a `git archive` of an
earlier commit). Writes the unconstrained draws, the sample statistics and `idata.attrs["config"]`.
It must not import anything that exists only in the newer tree.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--chains", type=int, default=8)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--draws", type=int, default=10)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--x64", action="store_true")
ap.add_argument("--chees", default="{}", help="extra chees_kwargs, JSON")
ap.add_argument("--pathfinder", default="{}", help="pathfinder_kwargs, JSON")
args = ap.parse_args()

sys.path.insert(0, os.path.join(args.tree, "tools"))
sys.path.insert(0, args.tree)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", args.x64)

import numpy as np  # noqa: E402
import r1_bias  # noqa: E402
from pfcheeshmc import sample_pf_chees_hmc  # noqa: E402

model = r1_bias.build_model(args.model)
idata = sample_pf_chees_hmc(
    model, draws=args.draws, chains=args.chains, random_seed=args.seed, progressbar=False,
    keep_untransformed=True, chees_kwargs={"draws": args.warmup, **json.loads(args.chees)},
    pathfinder_kwargs=json.loads(args.pathfinder))

arrays = {f"posterior/{k}": np.asarray(v.values) for k, v in idata.posterior.data_vars.items()}
arrays.update({f"stats/{k}": np.asarray(v.values) for k, v in idata.sample_stats.data_vars.items()})
arrays["config"] = np.array(idata.attrs["config"])
np.savez(args.out, **arrays)
