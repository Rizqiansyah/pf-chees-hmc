#!/usr/bin/env python
"""F2f: CPU NUTS wall per `pm.sample` call in ONE process, for the R2-T4 workflows (CPU only).

    python results/f2/drivers/cpu_nuts_calls.py MODEL TUNE DRAWS --data-seeds 11,12 --json OUT

Calls, in order (32 chains on 32 cores, the M5 keyword set `tools/t3_cpu_nuts.sample_kwargs`):
  first       a clone of the model with its own data (`new_data.with_new_data(m, 0)`), first call in the process
  repeat      the same model object again
  rebuilt     a new clone with the same data (a user who rebuilds the model every run)
  new_data_k  a clone with new observed data at the same shapes, one per --data-seeds entry
Every call is a clone, so all share one value-variable order. PyTensor's C cache is whatever
PYTENSOR_FLAGS points at: the driver runs this once with an empty compiledir (cold) and once with the
default one (warm). Use data seeds never used before in that compiledir, or "new data" is a cache hit.
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
MAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(MAIN, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_ladder as bl  # noqa: E402
import r1_bias  # noqa: E402
import t3_cpu_nuts  # noqa: E402
from new_data import with_new_data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("tune", type=int)
ap.add_argument("draws", type=int)
ap.add_argument("--data-seeds", default="11,12")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--json")
args = ap.parse_args()

base = r1_bias.build_model(args.model)
first = with_new_data(base, 0)
calls = [("first", first), ("repeat", first), ("rebuilt", None)]
calls += [(f"new_data_{s}", int(s)) for s in args.data_seeds.split(",")]
rows = []
for label, m in calls:
    if m is None:
        m = with_new_data(base, 0)
    elif isinstance(m, int):
        m = with_new_data(base, m)
    kw = t3_cpu_nuts.sample_kwargs(args.draws, args.tune, args.seed)
    load = os.getloadavg()[0]
    t0 = time.perf_counter()
    with m:
        bl.REAL_PM_SAMPLE(**kw)
    wall = time.perf_counter() - t0
    rows.append({"call": label, "wall": wall, "load1": load})
    print(f"{args.model} {label:12s} {wall:7.2f}s  load {load:.1f}", flush=True)
if args.json:
    with open(args.json, "w") as f:
        json.dump({"model": args.model, "tune": args.tune, "draws": args.draws,
                   "pytensor_flags": os.environ.get("PYTENSOR_FLAGS"), "rows": rows}, f, indent=1)
