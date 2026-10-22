#!/usr/bin/env python
"""F2b-2: what a chunk costs, to choose the chunk sizes of the warmup and sampling loops (CPU or GPU).

    python tools/chunk_overhead.py TREE MODEL CHAINS WARMUP DRAWS [--warmup-chunks 10,25,50,...]
                                   [--sample-chunks 25,50,...] [--reps 3] [--serial] [--json OUT]

TREE is the source tree (this checkout); MODEL is a ladder name or `hier`. Each loop is `ceil(total/chunk)`
calls of one jitted chunk program (`pfcheeshmc.compiled.warmup_chunk` / `sampling_chunk`). For each chunk
size the tool runs `sample_pf_chees_hmc` once (the compile), then `--reps` warm calls, and reports the
minimum stage time of the warmup, the sampling loop and the value -> RV transform. With the totals held
fixed, the extra time of a smaller chunk is the per-chunk overhead: host dispatch, the device -> host copy
of the chunk's outputs, the progress-bar-free bookkeeping, and the skipped rows of a partial last chunk.

`--serial` replaces `compiled.run_chunks` by a loop that converts each chunk BEFORE dispatching the next,
to measure what keeping one chunk in flight buys. `--warmup-chunks` / `--sample-chunks` default to a
ladder; the other loop stays at its module default while one is varied.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("model")
ap.add_argument("chains", type=int)
ap.add_argument("warmup", type=int)
ap.add_argument("draws", type=int)
ap.add_argument("--warmup-chunks", default="10,25,50,100,200")
ap.add_argument("--sample-chunks", default="10,25,50,100,250,1000")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--serial", action="store_true")
ap.add_argument("--json")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(args.tree, "tools"))
sys.path.insert(0, args.tree)
sys.path.insert(1, os.path.join(HERE, "..", "tests"))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pfcheeshmc.compiled as compiled  # noqa: E402
import pfcheeshmc.sampler as sm  # noqa: E402
import r1_bias  # noqa: E402
from seeded_models import hier  # noqa: E402

STAGES = {"_run_chees": "warmup", "_run_dynamic_hmc": "sampling",
          "_transform_value_samples_to_problem_space": "transform"}
REC = {}


def instrument(method, label):
    original = getattr(sm.PfCheesHMC, method)

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        out = original(self, *a, **k)
        REC[label] = REC.get(label, 0.0) + time.perf_counter() - t0
        return out

    setattr(sm.PfCheesHMC, method, timed)


for method, label in STAGES.items():
    instrument(method, label)


def serial_run_chunks(call, carry, total, size, consume):
    for start in (range(0, total, size) if total > 0 else (0,)):
        n = min(size, total - start)
        carry, out = call(carry, start, n)
        consume(start, n, out)
    return carry


if args.serial:
    compiled.run_chunks = serial_run_chunks


def one_call(seed):
    model = hier(100 + seed, 200 + seed) if args.model == "hier" else r1_bias.build_model(args.model)
    REC.clear()
    idata = sm.sample_pf_chees_hmc(model, draws=args.draws, chains=args.chains, random_seed=seed,
                                   progressbar=False, chees_kwargs={"draws": args.warmup})
    jax.block_until_ready(next(iter(idata.posterior.values())).values)
    return dict(REC)


def measure(label, **chunks):
    for name, value in chunks.items():
        setattr(compiled, name, value)
    one_call(0)                                                    # compile
    runs = [one_call(1 + i) for i in range(args.reps)]
    best = {k: min(r[k] for r in runs) for k in runs[0]}
    print(f"{label:>22s}  warmup {best['warmup']:7.3f}s  sampling {best['sampling']:7.3f}s  "
          f"transform {best['transform']:7.3f}s", flush=True)
    return {"label": label, **chunks, **best}


defaults = (compiled.WARMUP_CHUNK, compiled.SAMPLE_CHUNK, compiled.TRANSFORM_CHUNK)
rows = []
print(f"{args.model}  C={args.chains}  W={args.warmup}  D={args.draws}  reps={args.reps}  "
      f"{'serial' if args.serial else 'overlapped'}  (min stage time of warm calls)")
for s in [int(x) for x in args.warmup_chunks.split(",") if x]:
    rows.append(measure(f"WARMUP_CHUNK={s}", WARMUP_CHUNK=s, SAMPLE_CHUNK=defaults[1], TRANSFORM_CHUNK=defaults[2]))
for s in [int(x) for x in args.sample_chunks.split(",") if x]:
    rows.append(measure(f"SAMPLE_CHUNK={s}", WARMUP_CHUNK=defaults[0], SAMPLE_CHUNK=s, TRANSFORM_CHUNK=defaults[2]))
if args.json:
    with open(args.json, "w") as f:
        json.dump(rows, f, indent=1)
