"""GEV G3 probe: does the corrected metric orientation move posterior means, and is the divergence rise seed-stable?

The default-flip gate (G3) needs three things the arm table does not carry:
  1. per-variable posterior means within ~1 sigma of the CPU NUTS reference, both arms, >= 5 seeds;
  2. the reference/MCSE-based noise floor, so "within 1 sigma" is not hand-waved;
  3. whether the divergence-rate rise (0.225 -> 0.475 per 1k) is seed-stable or a one-seed artifact.

Arms differ ONLY in pathfinder_kwargs["imm_orientation"]; the coordinate ordering is the fixed one in
both, so this isolates the orientation question. Rows: results/runs/G3__gev_meanshift.jsonl.
"""
import json
import os
import sys

import numpy as np
import arviz
import arviz_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common
import bench_ladder as bl
import pfcheeshmc

MODEL = os.environ.get("MODEL", "gev")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "43,44,45,46,47").split(",")]
CHAINS, DRAWS, WARM = int(os.environ.get("CHAINS", 32)), int(os.environ.get("DRAWS", 1250)), int(os.environ.get("WARMUP", 2000))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   f"results/runs/G3__{MODEL}_meanshift.jsonl")
REF = os.path.expanduser(f"~/pf-runs/references/{MODEL}.npz")
ARMS = [("shipped orientation (precision)", "precision"), ("candidate (covariance)", "covariance")]

ref = np.load(REF, allow_pickle=True)
# The idata carries the model's own RV names (mu, sigma, xi on GEV), so score the CONSTRAINED
# reference arrays, which match by name. Unconstrained-vs-constrained matters for b2 (WP3 does it
# in u__ space); for a mean-shift sanity gate the parameterisation the user reads is the useful one.
# Constrained names match the idata for transformed RVs (x__sigma -> sigma); an untransformed RV
# (GEV's mu) exists only under u__, so take the union and let the posterior-name intersection drop
# anything else. Duplicate names prefer x__.
keys = sorted({k for k in ref.files if k.startswith(("x__", "u__"))},
              key=lambda k: (k[3:], k.startswith("u__")))
ref_mean, ref_sd, floor_sd = {}, {}, {}
for k in keys:
    a = np.asarray(ref[k], dtype="float64")           # (chains, draws, ...)
    n = k[3:]
    m = a.mean(axis=(0, 1))
    ref_mean[n] = m
    ref_sd[n] = np.sqrt(a.var(axis=(0, 1), ddof=1))
    floor_sd[n] = np.sqrt(m.size) * np.sqrt(a.var(axis=(0, 1), ddof=1) / a.shape[0] / a.shape[1]) \
        / np.maximum(np.sqrt(a.var(axis=(0, 1), ddof=1)), 1e-12)     # ref MCSE in sd units, sqrt(d) Bonferroni-ish
print(f"model={MODEL} chains={CHAINS} draws={DRAWS} warmup={WARM} seeds={SEEDS}", flush=True)
print("reference noise floor (sd units, per var):",
      {n: round(float(np.max(v)), 4) for n, v in floor_sd.items()}, flush=True)

for label, orient in ARMS:
    for seed in SEEDS:
        model = bl.model_specs()[MODEL]()
        assert pfcheeshmc.sample_pf_chees_hmc is bl.REAL_SAMPLE_PF, "sampler is stubbed: row void"
        idata = pfcheeshmc.sample_pf_chees_hmc(
            model, draws=DRAWS, chains=CHAINS, random_seed=seed, progressbar=False,
            pathfinder_kwargs={"imm_orientation": orient},
            chees_kwargs={"draws": WARM})
        div = np.asarray(idata.sample_stats["diverging"].values, dtype="bool")
        mcse = arviz_stats.mcse(idata, method="mean")
        shift, zs = {}, {}
        for n in sorted(set(ref_mean) & set(idata.posterior.data_vars)):
            got = np.mean(np.asarray(idata.posterior[n].values, dtype="float64"), axis=(0, 1))
            sd = np.maximum(ref_sd[n], 1e-12)
            err = np.abs(np.ravel(got - ref_mean[n])) / np.ravel(sd)
            sig = np.maximum(np.ravel(np.asarray(mcse[n].values, dtype="float64")) / np.ravel(sd), 1e-12)
            shift[n] = [round(float(x), 4) for x in err]
            zs[n] = float(np.max(err / sig))
        row = {"probe": "G3_meanshift", "model": MODEL, "arm": orient, "label": label, "seed": seed,
               "chains": CHAINS, "draws": DRAWS, "warmup": WARM, "cold": None,
               "divs_per_1k": float(div.sum() / max(div.size, 1) * 1000.0),
               "max_rhat": float(np.max(np.asarray(arviz.rhat(idata.posterior).to_array()))),
               "worst_shift_sd": float(max(max(v) for v in shift.values())),
               "worst_shift_reference_sigmas": float(max(zs.values())),
               "shift_sd_by_var": shift, "z_by_var": zs,
               "ref_floor_sd": {n: round(float(np.max(v)), 4) for n, v in floor_sd.items()},
               "frozen_eps": idata.attrs.get("frozen_step_size"),
               "frozen_L_nominal": idata.attrs.get("frozen_L_nominal"),
               "git_sha": _common.git_sha(), "pfcheeshmc_file": pfcheeshmc.__file__}
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{orient:>12s} seed={seed}] divs/1k={row['divs_per_1k']:.3f} rhat={row['max_rhat']:.4f} "
              f"worst_shift={row['worst_shift_sd']:.4f} sd z={row['worst_shift_reference_sigmas']:.2f} "
              f"eps={row['frozen_eps']:.4f} L={row['frozen_L_nominal']:.0f}", flush=True)
