"""Child process of tests/test_start_repair.py: `_ensure_finite_inits` on one case, in a fresh process, in any tree.

    python start_repair_run.py CASE OUT.npz

Run from the repository root on the path. Everything runs in float64: PyTensor's JAX dispatch switches `jax_enable_x64`
on at import.

Prints one JSON line: the XLA programs the call compiled (`first`: a cold process; `second`: the same call with
another key) and the stack's shape. The repaired stack of each call goes to OUT.npz (`first/<name>`,
`second/<name>`). CASE:

    gev           the ladder's GEV: one shared start at xi = 0, whose gradient is NaN (pm_chains=1)
    gev_c4_clean  GEV, four jittered starts, none of them on the singularity (pm_chains=4)
    gev_c4_bad    the same, rows 1 and 3 put back on xi = 0
    fence         a non-ladder model with a fence of -inf log-density and four starts:
                    row 0 fenced off at x0 = 12 (the backtrack finds a usable point on its 2nd halving),
                    row 1 usable, row 2 on a fenced strip around x1 = 0 (every backtrack fails: random restarts),
                    row 3 as row 2, under another key
    last_resort   a unit-ball log-density and a start far outside it, with no backtracks and no restarts allowed: the
                  search from the start fails and the search from the origin gives the answer
    clean_64      64 usable starts: nothing to repair
"""
import json
import os
import sys

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
case, out = sys.argv[1:3]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pymc as pm  # noqa: E402
import pytensor.tensor as pt  # noqa: E402
from benchmarks.compile_counter import count_compiles  # noqa: E402
from benchmarks.models import build_model  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402

KW = {}


def gev(chains, bad_rows=()):
    s = PfCheesHMC(build_model("gev"), seed=1)
    kwargs = dict(init="adapt_diag", tune=500, random_seed=1) if chains == 1 else \
        dict(init="jitter+adapt_diag", tune=500, random_seed=1)
    stack = s._pymc_adapt_diag_start(chains, **kwargs)
    if bad_rows:
        on_the_singularity = s._pymc_adapt_diag_start(1, init="adapt_diag", tune=500, random_seed=1)
        stack = {n: jnp.asarray(np.where(np.isin(np.arange(chains), bad_rows).reshape((-1,) + (1,) * (a.ndim - 1)),
                                         np.asarray(on_the_singularity[n]), np.asarray(a)))
                 for n, a in stack.items()}
    return s, stack


def fence_model():
    with pm.Model() as model:
        x = pm.Normal("x", shape=(2,))
        # -inf where x0 is beyond 5 or x1 is within 0.01 of 0; the switch is on a comparison, so the gradient
        # of the fence is 0 wherever it is finite
        pm.Potential("fence", pt.switch(pt.gt(pt.abs(x[0]), 5.0) | pt.lt(pt.abs(x[1]), 0.01), -np.inf, 0.0))
    return model


if case == "gev":
    s, stack = gev(1)
elif case == "gev_c4_clean":
    s, stack = gev(4)
elif case == "gev_c4_bad":
    s, stack = gev(4, bad_rows=(1, 3))
elif case == "fence":
    s = PfCheesHMC(fence_model(), seed=1)
    stack = {"x": jnp.array([[12.0, 1.0], [0.5, 0.7], [1.0, 0.0], [1.0, 0.0]])}
elif case == "last_resort":
    with pm.Model() as model:
        x = pm.Normal("x", shape=(2,))
        pm.Potential("ball", pt.switch(pt.sum(x ** 2) > 1.0, -np.inf, 0.0))
    s = PfCheesHMC(model, seed=1)
    stack = {"x": jnp.array([[50.0, 50.0]])}
    KW.update(max_backtracks=0, max_restarts=0)
elif case == "clean_64":
    with pm.Model() as model:
        pm.Normal("x", shape=(3,))
    s = PfCheesHMC(model, seed=1)
    stack = {"x": jnp.asarray(np.random.default_rng(0).normal(size=(64, 3)))}
else:
    raise SystemExit(f"unknown case {case!r}")

key_fix = jax.random.split(jax.random.key(1), 5)[0]           # what `_run_pathfinder` hands over
arrays, compiled_programs = {}, {}
for call, key in (("first", key_fix), ("second", jax.random.key(7))):
    with count_compiles() as cc:
        repaired = s._ensure_finite_inits(stack, key, **KW)
        jax.block_until_ready(repaired)
    compiled_programs[call] = cc.names
    arrays.update({f"{call}/{n}": np.asarray(v) for n, v in repaired.items()})
arrays.update({f"input/{n}": np.asarray(v) for n, v in stack.items()})
np.savez(out, **arrays)
print(json.dumps({"compiles": compiled_programs, "shapes": {n: list(v.shape) for n, v in stack.items()}}))
