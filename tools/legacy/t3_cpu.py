# =============================================================================
# QUARANTINED — DO NOT RUN. Kept byte-for-byte below this block for provenance.
#
# This is run 1's T3 CPU-NUTS driver. Run-2 plan Part B 1.2 voids its conclusions:
#   * the noise floor it emits is MCSE/sd, NOT squared, and is then compared against a
#     squared b2 -- so cells that are resolvable passes read as "not resolvable". The
#     "CPU NUTS cannot resolve b2 at these shapes" claim (run-1 REPORT T3) is void.
#   * it scored CONSTRAINED space, dropping tau_log__, xi_interval__, sigma_log__ and
#     radon's three sigma_*_log__ -- the hardest coordinates.
# Superseded by tools/t3_cpu_nuts.py (R2-WP3), which scores the value variables through
# r1_bias.score_ensemble and gates with r1_bias.g_r1, the same instrument as the GPU rows.
# It was also copied from the pf-r1-int branch, so its sys.path/OUT assumptions no longer
# resolve from this directory -- it cannot run as-is even if someone ignores this block.
# =============================================================================
"""T3: how long does CPU NUTS take to reach the same bias bar the GPU R1 regime is gated on?

Convention, stated up front because it differs from G-R1: CPU NUTS is scored in CONSTRAINED space
against the reference's `x__` arrays, because `pm.sample(keep_untransformed=True)` in PyMC 5.28 puts
nothing unconstrained in the InferenceData (verified on this box: posterior carries only `sigma`).
Both sides of the comparison here therefore use the space the user reads, and the GPU rows are
re-scored the same way from their constrained idata so the two are directly comparable.

Rows: results/runs/T3__cpu_nuts.jsonl. One process per (model, config) keeps the wall clock honest.
"""
import json
import os
import sys
import time

if __name__ == "__main__":
    raise SystemExit(
        "QUARANTINED (run-2 plan Part B 1.2): run-1 T3 driver has a non-squared noise floor and\n"
        "scored constrained space only, so its verdicts are void. Use tools/t3_cpu_nuts.py.\n"
        "This file is kept for provenance and must not be run.")

import numpy as np
import arviz as _az

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common
import bench_ladder as bl

MODEL = os.environ["T3_MODEL"]
CHAINS = int(os.environ.get("T3_CHAINS", 32))
DRAWS = int(os.environ["T3_DRAWS"])
TUNE = int(os.environ.get("T3_TUNE", 1000))
TA = float(os.environ.get("T3_TARGET_ACCEPT", 0.9))
SEED = int(os.environ.get("T3_SEED", 42))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results/runs/T3__cpu_nuts.jsonl")

ref = np.load(os.path.expanduser(f"~/pf-runs/references/{MODEL}.npz"), allow_pickle=True)
# The references store constrained arrays ONLY for transformed RVs (8schools x__ has just `tau`;
# logit has none at all, since `b` is untransformed), so taking x__ alone scores 1-3 variables per
# model and skips logistic entirely -- that is what the first T3 run did, and it is not a valid
# comparison. Untransformed RVs are identical in both spaces, so `u__` fills the gap by name, with
# x__ preferred where both exist. Coverage is asserted below: a missing variable must abort the run.
xkeys = {k[3:]: np.asarray(ref[k], dtype="float64") for k in ref.files if k.startswith("u__")}
for k in ref.files:
    if k.startswith("x__"):
        xkeys[k[3:]] = np.asarray(ref[k], dtype="float64")
rm = {n: a.mean(axis=(0, 1)) for n, a in xkeys.items()}
rv = {n: a.var(axis=(0, 1), ddof=1) for n, a in xkeys.items()}
rm2 = {n: (a ** 2).mean(axis=(0, 1)) for n, a in xkeys.items()}
rv2 = {n: (a ** 2).var(axis=(0, 1), ddof=1) for n, a in xkeys.items()}

model = bl.model_specs()[MODEL]()
drop = tuple(d.name for d in model.deterministics)          # same convention as bench_ladder.run_cpu
t0 = time.perf_counter()
with model:
    idata = bl.REAL_PM_SAMPLE(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                              target_accept=TA, random_seed=SEED, progressbar=False,
                              idata_kwargs={"log_likelihood": False})
wall = time.perf_counter() - t0

want = (set(rm) - set(drop)) & set(idata.posterior.data_vars)
names = sorted(want)
missing = sorted((set(rm) - set(drop)) - set(idata.posterior.data_vars))
# A row that scores a subset of the parameters is not a bias measurement of the model.
uncovered = sorted(set(idata.posterior.data_vars) - set(drop) - want)
if uncovered:
    raise RuntimeError(f"reference does not cover posterior variables {uncovered}; refusing a partial b2")
b2, b2_2, mc2, mc2_2 = {}, {}, [], []
for n in names:
    arr = np.asarray(idata.posterior[n].values, dtype="float64").reshape(CHAINS * DRAWS, -1)
    b2[n] = float(np.max((arr.mean(axis=0) - rm[n]) ** 2 / np.maximum(rv[n], 1e-300)))
    a2 = (arr ** 2).mean(axis=0)
    b2_2[n] = float(np.max((a2 - rm2[n]) ** 2 / np.maximum(rv2[n], 1e-300)))
    n_eff = max(float(np.min(np.asarray(_az.ess(idata.posterior[[n]], method="mean").to_array()))), 1.0)  # ESS, not raw draws (arviz 0.23 returns a Dataset: use .to_array())
    mc2.append(float(np.max(np.sqrt(np.var(arr, axis=0, ddof=1) / n_eff)
                            / np.sqrt(np.maximum(rv[n], 1e-300)))))
    mc2_2.append(float(np.max(np.sqrt(np.var(arr ** 2, axis=0, ddof=1) / n_eff)
                              / np.sqrt(np.maximum(rv2[n], 1e-300)))))
d = sum(int(np.size(rm[n])) for n in names)
floor = max(mc2 + mc2_2) * 2 * np.log(2 * max(d, 1))

row = {"table": "T3", "wp": "T3", "model": MODEL, "arm": "cpu:nuts", "device": "cpu",
       "space": "constrained", "chains": CHAINS, "draws_per_chain": DRAWS, "tune": TUNE,
       "target_accept": TA, "seed": SEED, "cores": CHAINS,
       "wall_s": round(wall, 2), "core_seconds": round(wall * CHAINS, 1),
       "b2_max": float(max(list(b2.values()) + list(b2_2.values()))),
       "b2_by_var": b2, "b2_sq_by_var": b2_2, "noise_floor": float(floor),
       "passes_0.01": bool(max(list(b2.values()) + list(b2_2.values())) < 0.01),
       "resolvable": bool(floor < 0.01),
       "scored_vars": names, "missing_vars": missing,
       "divs_per_1k": float(np.asarray(idata.sample_stats["diverging"].values, dtype="bool").mean() * 1000),
       "total_draws": CHAINS * DRAWS, "git_sha": _common.git_sha()}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "a") as fh:
    fh.write(json.dumps(row) + "\n")
print(f"[{MODEL} C={CHAINS} D={DRAWS} tune={TUNE}] wall={row['wall_s']}s core_s={row['core_seconds']} "
      f"b2={row['b2_max']:.4g} floor={floor:.4g} pass={row['passes_0.01']} resolvable={row['resolvable']} "
      f"missing={missing}", flush=True)
