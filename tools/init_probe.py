#!/usr/bin/env python
"""F2g: what the init stage's start repair compiles and costs, per call (CPU or GPU).

    python tools/init_probe.py TREE MODEL CHAINS [--calls 3]

TREE is the source tree to measure: this checkout, or a `git archive` of an earlier commit. MODEL is a ladder name
or `fence`: a 2-d standard normal with a fence of -inf log-density around x1 = 0, where PyMC's initial point is,
so the start is repaired by random restarts (a non-ladder model that needs repair; GEV is the ladder's one).

Each call runs `sample_pf_chees_hmc` with 10 draws and 25 warmup draws (one seed per call), and reports, for the
methods of the init stage, how many times each ran, the XLA programs it compiled and its seconds. Call 1 is cold.
`_find_finite_start` is nested in `_ensure_finite_inits`, and `_usable_mask` in both.

    GEV, 64 chains, CPU:   _ensure_finite_inits   F2 head 087b54e  162 compiles 5.3 s cold, 0.12 s warm
                                                  F2g               17 compiles 1.3 s cold, 0.00 s warm

The on-disk compilation cache is off (`PFCHEESHMC_COMPILATION_CACHE=0` unless set), so cold means cold.
"""
import argparse
import collections
import os
import sys
import time

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
os.environ.setdefault("PFCHEESHMC_COMPILATION_CACHE", "0")

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("model")
ap.add_argument("chains", type=int)
ap.add_argument("--calls", type=int, default=3)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [args.tree, os.path.join(args.tree, "tools"), os.path.join(HERE, "..", "tests")]

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pymc as pm  # noqa: E402
import pytensor.tensor as pt  # noqa: E402
import r1_bias  # noqa: E402
import pfcheeshmc.sampler as sm  # noqa: E402
from compile_counter import count_compiles  # noqa: E402

METHODS = ("_pymc_adapt_diag_start", "_ensure_finite_inits", "_usable_mask", "_find_finite_start")
REC = {}


def instrument(method):
    original = getattr(sm.PfCheesHMC, method)

    def wrapped(self, *a, **k):
        with count_compiles() as c:
            t0 = time.perf_counter()
            out = original(self, *a, **k)
            jax.block_until_ready(out)
            dt = time.perf_counter() - t0
        rec = REC.setdefault(method, [0, 0, 0.0, collections.Counter()])
        rec[0] += 1
        rec[1] += c.count
        rec[2] += dt
        rec[3].update(c.names)
        return out

    setattr(sm.PfCheesHMC, method, wrapped)


for method in METHODS:
    instrument(method)


def fence_model():
    with pm.Model() as model:
        x = pm.Normal("x", shape=(2,))
        pm.Potential("fence", pt.switch(pt.lt(pt.abs(x[1]), 0.01), -np.inf, 0.0))
    return model


model = fence_model() if args.model == "fence" else r1_bias.build_model(args.model)
for call in range(1, args.calls + 1):
    REC.clear()
    t0 = time.perf_counter()
    sm.sample_pf_chees_hmc(model, draws=10, chains=args.chains, random_seed=call, progressbar=False,
                           chees_kwargs={"draws": 25})
    print(f"call {call} ({'cold' if call == 1 else 'warm'}), wall {time.perf_counter() - t0:.2f} s")
    for method in METHODS:
        runs, compiles, seconds, names = REC.get(method, [0, 0, 0.0, collections.Counter()])
        print(f"    {method:24s} ran {runs}  XLA compiles {compiles:4d}  {seconds:6.2f} s  {names.most_common(3)}")
