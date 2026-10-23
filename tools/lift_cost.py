#!/usr/bin/env python
"""F2e: what the three lifts of one sampler call cost on a freshly built model, and where the time goes (CPU or GPU).

    python tools/lift_cost.py TREE MODEL [MODEL ...] [--reps 4]

TREE is the source tree to measure (this checkout, or a `git archive` of an earlier commit: it provides
`pfcheeshmc/` and `tools/r1_bias.py`). MODEL is a ladder name. For each model, `--reps` fresh builds (as a user
who rebuilds the model for every dataset, or `tools/warm_stages.py`, does) are lifted at the three sites of a
call, in the order of a call:

    logp           lift_logp                      PfCheesHMC.__init__
    initial_point  lift_initial_point             the init stage
    transform      lift_graph(value vars -> RVs)  PfCheesHMC._lift_value_to_rv

Build 1 is the first of its structure in the process (everything is built); builds 2.. rebuild the model. Phases,
summed over the three sites (seconds):

    model.logp   PyMC building the log-density graph (not ours; the transform and initial-point sites do not call it)
    widen        lifted._widen_model_inputs
    rewrite      PyTensor's JAX rewrite (mode.JAX.optimizer.rewrite), the funcify, and the lowering of the key
    signature    lifted._signature + equal_computations (the lift memo; absent on a tree without one)
"""
import argparse
import collections
import os
import sys
import time

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("models", nargs="+")
ap.add_argument("--reps", type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, os.path.join(args.tree, "tools"))
sys.path.insert(0, args.tree)

import r1_bias  # noqa: E402
from pymc.util import get_default_varnames  # noqa: E402

from pfcheeshmc import lifted  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402

T = collections.defaultdict(float)
N = collections.Counter()


def timed(label, module, name):
    f = getattr(module, name, None)
    if f is None:
        return

    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[label] += time.perf_counter() - t0
            N[label] += 1

    setattr(module, name, wrapper)


class _Mode:
    """lifted uses `mode.JAX.optimizer.rewrite` and nothing else of `pytensor.compile.mode`."""
    real = lifted.mode

    class _Optimizer:
        def rewrite(self, fgraph, *a, **k):
            t0 = time.perf_counter()
            try:
                return _Mode.real.JAX.optimizer.rewrite(fgraph, *a, **k)
            finally:
                T["rewrite"] += time.perf_counter() - t0
                N["rewrite"] += 1

    class JAX:
        optimizer = None


_Mode.JAX.optimizer = _Mode._Optimizer()
lifted.mode = _Mode
timed("widen", lifted, "_widen_model_inputs")
timed("funcify", lifted, "jax_funcify")
timed("lowering", lifted, "_key")
timed("signature", lifted, "_signature")
timed("equal_computations", lifted, "equal_computations")


def sites(model):
    out = {}
    t0 = time.perf_counter()
    sampler = PfCheesHMC(model)
    out["logp"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    lifted.lift_initial_point(model)
    out["initial_point"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sampler._lift_value_to_rv()
    out["transform"] = time.perf_counter() - t0
    return out


for name in args.models:
    for rep in range(1, args.reps + 1):
        model = r1_bias.build_model(name)
        t0 = time.perf_counter()
        model.logp()
        logp_s = time.perf_counter() - t0
        T.clear()
        N.clear()
        per_site = sites(model)
        phases = {"rewrite": T["rewrite"], "funcify": T["funcify"], "lowering": T["lowering"], "widen": T["widen"],
                  "signature+equal": T["signature"] + T["equal_computations"]}
        print(f"{name:18s} build {rep}: lifts {sum(per_site.values()):6.3f}s  "
              + " ".join(f"{k} {v:5.3f}" for k, v in per_site.items())
              + f"  | model.logp {logp_s:5.3f}  " + " ".join(f"{k} {v:5.3f}" for k, v in phases.items())
              + f"  | rewrites {N['rewrite']}")
