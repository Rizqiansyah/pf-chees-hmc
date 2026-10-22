#!/usr/bin/env python
"""F2c: start-dispersion ratio of the Pathfinder start picks against the NUTS reference (REPORT run-1 section 3j).

    python tools/start_dispersion.py MODEL [--chains 1024] [--seed 1] [--calls 2] [--json OUT] [--starts OUT.npz]

Run it under the tree to measure (`PYTHONPATH=<tree>`: `pfcheeshmc` comes from there, this file and `r1_bias`
from wherever it is run). Method, as in the run-1 REPORT: `_run_pathfinder`'s start picks (`pf_samples`, one
row per chain, unconstrained space) are flattened in kernel order (sorted variable names: `r1_bias.flatten_run`)
and the ratio is, per coordinate, the sd across the picks over the sd of the reference draws
(`r1_bias.load_reference`, flattened the same way and labelled alike, with a label assert so that a wrong
pairing raises). Reported: the median ratio, the minimum, and the number of coordinates at or below 0.5 (and
0.05).

Also timed: `_run_pathfinder` per call (call 1 is cold, later ones warm, each a new seed), the part of it that
is the PyMC init (`_pymc_adapt_diag_start` and `_ensure_finite_inits`), and the start scoring (`compiled.score_pool`:
rows and seconds). `--starts` saves the picks of the last call, to compare two trees bit for bit.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--chains", type=int, default=1024)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--calls", type=int, default=2)
ap.add_argument("--json")
ap.add_argument("--starts")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jax  # noqa: E402
import numpy as np  # noqa: E402
import r1_bias  # noqa: E402
from pfcheeshmc import compiled  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402

model = r1_bias.build_model(args.model)
names = [v.name for v in model.value_vars]
ref_flat, ref_labels = r1_bias.flatten_ref(r1_bias.load_reference(model, names, require_stats=False))

TIMES = {"init": 0.0, "score_s": 0.0, "score_rows": 0}
for method in ("_pymc_adapt_diag_start", "_ensure_finite_inits"):
    original = getattr(PfCheesHMC, method)

    def timed(self, *a, __original=original, **k):
        t0 = time.perf_counter()
        out = __original(self, *a, **k)
        jax.block_until_ready(out)
        TIMES["init"] += time.perf_counter() - t0
        return out

    setattr(PfCheesHMC, method, timed)

_score_pool = compiled.score_pool


def timed_score_pool(*a, **k):
    t0 = time.perf_counter()
    out = jax.block_until_ready(_score_pool(*a, **k))
    TIMES["score_s"] += time.perf_counter() - t0
    TIMES["score_rows"] += int(out.shape[0])
    return out


compiled.score_pool = timed_score_pool

rows = []
for call in range(1, args.calls + 1):
    seed = args.seed + call - 1
    for key in TIMES:
        TIMES[key] = 0
    sampler = PfCheesHMC(model, seed=seed)
    t0 = time.perf_counter()
    pf, imm = sampler._run_pathfinder(chains=args.chains, seed=seed)
    jax.block_until_ready((pf, imm))
    total = time.perf_counter() - t0
    rows.append({"call": call, "seed": seed, "run_pathfinder_s": total, "init_s": TIMES["init"],
                 "pathfinder_stage_s": total - TIMES["init"], "score_s": TIMES["score_s"],
                 "score_rows": TIMES["score_rows"]})
    print(f"call {call} (seed {seed}): _run_pathfinder {total:.2f}s = init {TIMES['init']:.2f}s + "
          f"pathfinder stage {total - TIMES['init']:.2f}s, of which scoring {TIMES['score_s']:.2f}s on "
          f"{TIMES['score_rows']} rows")

# the dispersion of the last call's picks
starts = {n: np.asarray(a) for n, a in pf.items()}
run_flat, run_labels = r1_bias.flatten_run(
    {n: a.reshape((args.chains, 1) + a.shape[1:]) for n, a in starts.items()}, args.chains, 1)
assert run_labels == ref_labels, "start picks and reference are not in the same coordinate order"
ratio = run_flat.std(0, ddof=1) / ref_flat.std(0, ddof=1)
out = {
    "model": args.model, "chains": args.chains, "seed": args.seed + args.calls - 1, "d": int(ratio.size),
    "median_ratio": float(np.median(ratio)), "min_ratio": float(ratio.min()),
    "n_coords_below_half": int((ratio <= 0.5).sum()), "n_coords_below_twentieth": int((ratio <= 0.05).sum()),
    "ratio": ratio.tolist(), "labels": run_labels, "timings": rows,
}
print(f"{args.model} C={args.chains} d={ratio.size}: median ratio {out['median_ratio']:.4f}, min {out['min_ratio']:.4f}, "
      f"coordinates <= 0.5: {out['n_coords_below_half']}, <= 0.05: {out['n_coords_below_twentieth']}")
if args.json:
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
if args.starts:
    np.savez(args.starts, **starts)
