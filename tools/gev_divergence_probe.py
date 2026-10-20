#!/usr/bin/env python
"""GEV's start dispersion -- the measurable half of the divergence question (§7).

GEV is the one model where the corrected (covariance) orientation raises the warmup divergence rate
(0.23 -> 0.47 per 1k, seed-stable) and loses ESS/s. The leading candidate is a large Pathfinder
under-dispersion of one coordinate. This measures that candidate exactly: sd across the Pathfinder
starts versus sd of the CPU NUTS reference posterior, per coordinate, both flattened in kernel order
so a label mismatch raises instead of comparing two different variables.

What it CANNOT answer, and why: the per-coordinate *velocity* of the diverging transition is not
obtainable from the shipped tree. WP5 projected the warmup scan down to
`WarmupScanInfo(num_integration_steps, is_divergent, acceptance_rate)` (`pfcheeshmc/utils.py`) --
momentum and `proposal.*` are no longer returned, because they were ~4x the states and were the
warmup memory wall at C=1,024. Getting the velocity needs either the untrimmed HMCInfo (a memory
change to the hot path) or a `stack_positions` run plus a per-step momentum stack (neither exists).
So this probe settles whether the suspect coordinate really starts under-dispersed; the step-level
mechanism stays NOT ESTABLISHED, with the reason named.

Run: one process per (model, orientation) under tools/gpu_run.sh.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import r1_bias as rb  # noqa: E402
import _common as common  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402


def leaf_labels(starts):
    """One label per flat coordinate, in the order the sampler flattens (sorted value_vars)."""
    out = []
    for name in sorted(starts):
        a = np.asarray(starts[name])
        n = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
        if n == 1:
            out.append(name)
        else:
            for idx in np.ndindex(a.shape[1:]):
                out.append(f"{name}[{','.join(str(i) for i in idx)}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gev")
    ap.add_argument("--chains", type=int, default=32)
    ap.add_argument("--pf-draws", type=int, default=1,
                    help="1 = the R1/R2 start policy used by every bias row; 4000 = production pool")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--orientation", default="covariance", choices=["covariance", "precision"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    common.require_gpu_lock("start-dispersion probe")

    rows = []
    for model_name in a.models.split(","):
        model = rb.build_model(model_name)
        names = [v.name for v in model.value_vars]
        s = PfCheesHMC(model, seed=a.seed)
        starts, imm = s._run_pathfinder(chains=a.chains, draws=a.pf_draws,
                                        imm_orientation=a.orientation)
        labels = leaf_labels(starts)
        ref_arr, ref_labels = rb.flatten_ref(rb.load_reference(model_name, names))
        if ref_labels != labels:
            raise KeyError(f"start/reference leaf mismatch {labels[:4]} vs {ref_labels[:4]}")

        flat = np.concatenate([np.asarray(starts[n], dtype="float64")
                               .reshape(starts[next(iter(starts))].shape[0], -1)
                               for n in sorted(starts)], axis=1)
        sd_start = flat.std(axis=0, ddof=1) if flat.shape[0] > 1 else np.zeros(flat.shape[1])
        sd_ref = ref_arr.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sd_ref > 0, sd_start / sd_ref, np.nan)

        order = np.argsort(np.nan_to_num(ratio, nan=np.inf))
        out = {
            "model": model_name, "orientation": a.orientation, "pf_draws": a.pf_draws,
            "chains": a.chains, "seed": a.seed, "d": int(flat.shape[1]),
            "git_sha": common.git_sha(), "branch": common.branch(), "device": str(jax_dev()),
            "peak_mem_bytes": common.peak_mem_bytes(),
            "start_sd_over_ref_sd_coldest_first": {labels[i]: round(float(ratio[i]), 4) for i in order[:8]},
            "start_sd_over_ref_sd_widest_first": {labels[i]: round(float(ratio[i]), 4) for i in order[-8:][::-1]},
            "n_coords_below_half": int(np.sum(ratio < 0.5)),
            "n_coords_below_twentieth": int(np.sum(ratio < 0.05)),
            "median_ratio": round(float(np.nanmedian(ratio)), 4),
            "frozen_imm_diag_min": round(float(np.min(np.diag(np.asarray(imm, dtype="float64")))), 8)
            if np.asarray(imm).ndim == 2 else None,
        }
        rows.append(out)
        print(json.dumps(out, sort_keys=True), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(out, sort_keys=True) + "\n")
    if len(rows) > 1:
        coldest = {}
        for r in rows:
            k = next(iter(r["start_sd_over_ref_sd_coldest_first"]))
            coldest[r["model"]] = f"{k}={r['start_sd_over_ref_sd_coldest_first'][k]}"
        print("coldest coordinate per model: " + json.dumps(coldest), flush=True)


def jax_dev():
    import jax
    return str(jax.devices()[0])


if __name__ == "__main__":
    main()
