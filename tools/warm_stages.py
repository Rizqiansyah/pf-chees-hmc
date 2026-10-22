#!/usr/bin/env python
"""F2b: per-stage wall time and XLA compile counts of repeated `sample_pf_chees_hmc` calls (CPU or GPU).

    python tools/warm_stages.py TREE MODEL CHAINS DRAWS WARMUP [--calls 3] [--json OUT]
                                [--progressbar] [--chees JSON] [--pathfinder JSON]

TREE is the source tree to measure: this checkout, or a `git archive` of an earlier commit (it provides
`pfcheeshmc/` and `tools/r1_bias.py`). MODEL is a ladder name, or `hier`: the seeded 8-Schools-shaped model
of `tests/seeded_models.py`, rebuilt with NEW data and prior values on every call. Call 1 is cold; calls 2..
are warm, with a new seed each.

Timing is `time.perf_counter` around the PfCheesHMC stage methods, blocking on their output (the same
instrumentation as `tools/r1_bias.instrument`, without the GPU lock `--timed` requires). Stages:
  init        _pymc_adapt_diag_start + _ensure_finite_inits
  pathfinder  _run_pathfinder minus init
  eps0        _run_step_size_tune
  warmup      _run_chees (loop programs included)
  sampling    _run_dynamic_hmc (loop program included)
  transform   _transform_value_samples_to_problem_space
Compiles are counted with `tests/compile_counter.py` (JAX's backend-compile event) over each stage and the
whole call; the program names of a warm call are printed.
"""
import argparse
import collections
import json
import os
import sys
import time

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("model")
ap.add_argument("chains", type=int)
ap.add_argument("draws", type=int)
ap.add_argument("warmup", type=int)
ap.add_argument("--calls", type=int, default=3)
ap.add_argument("--json")
ap.add_argument("--progressbar", action="store_true")
ap.add_argument("--chees", default="{}", help="extra chees_kwargs, JSON")
ap.add_argument("--pathfinder", default="{}", help="pathfinder_kwargs, JSON")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(args.tree, "tools"))
sys.path.insert(0, args.tree)
sys.path.insert(1, os.path.join(HERE, "..", "tests"))

import jax  # noqa: E402
import pfcheeshmc.sampler as sm  # noqa: E402
import r1_bias  # noqa: E402
from compile_counter import count_compiles  # noqa: E402
from seeded_models import hier  # noqa: E402

METHODS = {
    "_pymc_adapt_diag_start": "init", "_ensure_finite_inits": "init2", "_run_pathfinder": "pathfinder+init",
    "_run_step_size_tune": "eps0", "_run_chees": "warmup", "_run_dynamic_hmc": "sampling",
    "_transform_value_samples_to_problem_space": "transform",
}
REC = collections.defaultdict(lambda: {"s": 0.0, "n": 0})


def instrument(method, label):
    original = getattr(sm.PfCheesHMC, method)

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        with count_compiles() as cc:
            out = original(self, *a, **k)
            try:
                jax.block_until_ready(out[0] if isinstance(out, tuple) else out)
            except Exception:
                pass
        REC[label]["s"] += time.perf_counter() - t0
        REC[label]["n"] += cc.count
        return out

    setattr(sm.PfCheesHMC, method, timed)


for method, label in METHODS.items():
    instrument(method, label)


def build(call):
    return hier(100 + call, 200 + call) if args.model == "hier" else r1_bias.build_model(args.model)


results = []
for call in range(1, args.calls + 1):
    REC.clear()
    model = build(call)
    t0 = time.perf_counter()
    with count_compiles() as total:
        idata = sm.sample_pf_chees_hmc(
            model, draws=args.draws, chains=args.chains, random_seed=call, progressbar=args.progressbar,
            chees_kwargs={"draws": args.warmup, **json.loads(args.chees)},
            pathfinder_kwargs=json.loads(args.pathfinder))
        jax.block_until_ready(next(iter(idata.posterior.values())).values)
    wall = time.perf_counter() - t0
    init_s, init_n = REC["init"]["s"] + REC["init2"]["s"], REC["init"]["n"] + REC["init2"]["n"]
    stage = {
        "init": (init_s, init_n),
        "pathfinder": (REC["pathfinder+init"]["s"] - init_s, REC["pathfinder+init"]["n"] - init_n),
        **{k: (REC[k]["s"], REC[k]["n"]) for k in ("eps0", "warmup", "sampling", "transform")},
    }
    stage["rest"] = (wall - sum(s for s, _ in stage.values()), total.count - sum(n for _, n in stage.values()))
    results.append({"call": call, "wall": wall, "compiles": total.count, "compile_s": total.seconds,
                    "stage": stage, "names": sorted(total.names)})
    print(f"call {call}: wall {wall:6.2f}s  compiles {total.count} ({total.seconds:.2f}s of compile)")
    for k, (s, n) in stage.items():
        print(f"    {k:11s} {s:7.2f}s  compiles {n}")
    if call > 1:
        print("    programs:", sorted(total.names))
if args.json:
    with open(args.json, "w") as f:
        json.dump(results, f, indent=1)
