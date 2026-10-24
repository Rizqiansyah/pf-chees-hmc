"""E1 ladder harness: GPU vs CPU throughput as models get larger and stiffer.

Usage (all configurable by environment):
    pixi run python tools/bench_ladder.py
Environment:
    MODELS=8schools,gev          which models (the names in benchmarks/models.py)
    ARMS=gpu,cpu                 which backends
    GPU_CHAINS=32,128            chain counts for the GPU arm
    CPU_CHAINS=4,16,32           chain counts for the CPU arm (cores = chains, 1 thread each)
    DRAWS=2000 WARMUP=2000 CPU_DRAWS=1000 SEEDS=42,43,44
    LADDER_OUT=~/pf-runs/ladder.jsonl

Reported per cell: wall (cold and warm flagged), bulk ESS/s total and worst-variable,
min tail-ESS, max R-hat, divergences per 1000 draws, mean leapfrog steps, the padding ratio
(mean over draws of max/mean across chains of num_integration_steps — warp-divergence cost),
the draw-covariance condition number (mass-matrix difficulty proxy, unit-dependent),
and whether sample_stats.energy contains non-finite values.

CPU wall time includes tuning. GPU wall time includes Pathfinder, ChEES warmup and any
compilation, so cold and warm must never be mixed in a comparison.

Capacity is part of the result, not an accident: at d=175 (radon) 128 chains does not fit the 6 GB
RTX A2000 — JAX fails allocating 699 MiB in a dot_general — while 32 chains runs comfortably. So a
chain-count sweep has to report where the device stops, not extrapolate past it.
"""
import json
import os
import sys
import time
from pathlib import Path

# Must be set before numpy/pytensor import: the CPU arm's fairness depends on one thread per
# chain, so that "cores = chains" actually means what it says (matches the CPU curve protocol).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# This repo lives under a path containing a space ("Python packages"). PyTensor puts the env prefix
# into blas__ldflags as an unquoted -Wl,-rpath,... token, and its linker then dies with
# 'invalid token "packages/pf-chees-hmc/.pixi/envs/default/lib"' for any model that compiles a Dot.
# Overriding the flags without the rpath fixes it; numpy still supplies BLAS at runtime.
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
# Cold and warm rows must differ by what the process has compiled, not by a disk cache: the library's default
# on-disk compilation cache (F2g) is off here unless the environment says otherwise.
os.environ.setdefault("PFCHEESHMC_COMPILATION_CACHE", "0")

import arviz as az
import numpy as np

import pfcheeshmc
import pymc as pm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(REPO))                 # benchmarks/models.py
from benchmarks.models import MODELS, build_model  # noqa: E402

# The two sampling entry points the runners call. (The names date from when loading a notebook model replaced them with
# stubs; nothing replaces them now, and the tools that import them are unchanged.)
REAL_SAMPLE_PF = pfcheeshmc.sample_pf_chees_hmc
REAL_PM_SAMPLE = pm.sample

OUT = Path(os.environ.get("LADDER_OUT", str(Path.home() / "pf-runs" / "ladder.jsonl")))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
DRAWS = int(os.environ.get("DRAWS", "2000"))
WARMUP = int(os.environ.get("WARMUP", "2000"))
CPU_DRAWS = int(os.environ.get("CPU_DRAWS", "1000"))
GPU_CHAINS = [int(c) for c in os.environ.get("GPU_CHAINS", "32,128").split(",")]
CPU_CHAINS = [int(c) for c in os.environ.get("CPU_CHAINS", "4,16,32").split(",")]
ARMS = os.environ.get("ARMS", "gpu,cpu").split(",")


def _configs(spec):
    """Parse 'chains:draws_per_chain,...' into [(chains, draws)]. Empty -> caller's defaults."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        chains, _, draws = item.partition(":")
        out.append((int(chains), int(draws) if draws else DRAWS))
    return out


# Equal-total-draw budgets need (chains, draws) pairs, not two independent lists: 8x500 and
# 32x125 are the same 4,000 samples. When these are unset the old chains-list behaviour stands.
GPU_CONFIGS = _configs(os.environ.get("GPU_CONFIGS", ""))
CPU_CONFIGS = _configs(os.environ.get("CPU_CONFIGS", ""))
CPU_TUNE = int(os.environ.get("CPU_TUNE", "1000"))          # PyMC's default tune
REPEAT = int(os.environ.get("REPEAT", "1"))                 # 2 = cold row then warm row

# --------------------------------------------------------------------------- models
# The ladder models are in benchmarks/models.py. A synthetic ill-conditioned MvNormal family was here and is deleted on
# purpose: with a dense `cov` over many observations the jaxed logp allocated 4.63 GiB under 32 vmapped chains at d=50,
# and the cheap n_obs=1 variant was unusable as a comparison because CPU NUTS failed it too (R-hat 1.33). Stiffness
# comes from real models instead.


def model_specs():
    return {name: (lambda name=name: build_model(name)) for name in MODELS}


# --------------------------------------------------------------------------- metrics
def score(idata, wall, backend, model, chains, cold, drop_vars=(), draws=None, warmup=None):
    draws = DRAWS if draws is None else draws
    warmup = WARMUP if warmup is None else warmup
    ess = az.ess(idata.posterior, method="bulk", relative=False)
    etail = az.ess(idata.posterior, method="tail", relative=False)
    rh = az.rhat(idata.posterior)
    tot = float(sum(float(np.sum(v.values)) for v in ess.values()))
    row = {
        "backend": backend, "model": model, "chains": chains, "cold": cold,
        "draws": draws, "draws_total": draws * chains, "warmup": warmup,
        "wall_s": round(wall, 2), "ess_s_total": round(tot / wall, 0),
        "ess_s_worstvar": round(min(float(np.min(v.values)) for v in ess.values()) / wall, 0),
        "min_tail_ess": round(min(float(np.min(v.values)) for v in etail.values()), 0),
        "max_rhat": round(float(max(float(np.max(v.values)) for v in rh.values())), 4),
        "divs_per_1k_draws": round(float(np.sum(idata.sample_stats["diverging"].values))
                                   / (idata.posterior.sizes["draw"] * chains) * 1000, 3),
        "energy_finite": bool(np.all(np.isfinite(np.asarray(idata.sample_stats["energy"].values,
                                                          dtype=float))))
        if "energy" in idata.sample_stats else None,
    }
    # Conditioning proxy for the mass matrix: the sampler's inverse mass matrix is built from
    # Pathfinder draws, so the posterior draw covariance's condition number says what the metric
    # has to cope with. Unit-dependent by construction -- compare it across models in the same
    # units, never as an absolute. Deterministics are excluded: they are affine copies of sampled
    # variables here, and that exact collinearity shows up as a huge (fake) condition number.
    try:
        flat = np.concatenate(
            [np.asarray(idata.posterior[v].values).reshape(-1, int(np.prod(
                idata.posterior[v].values.shape[2:] or (1,))))
             for v in idata.posterior.data_vars if v not in set(drop_vars)], axis=1)
        if len(flat) > 20_000:
            flat = flat[:: max(len(flat) // 20_000, 1)]   # strided, so the sample spans all chains
        if flat.shape[1] > 1 and np.all(np.isfinite(flat)):
            ev = np.linalg.eigvalsh(np.cov(flat, rowvar=False))
            row["cov_cond"] = round(float(ev[-1] / max(ev[0], 1e-300)), 1)
    except Exception as exc:                                    # never lose a timing row to a diagnostic
        row["cov_cond_error"] = f"{type(exc).__name__}: {exc}"[:160]
    if backend == "gpu":
        row["imm"] = os.environ.get("IMM", "mc")
        if row["imm"] != "mc":
            row["rank"] = int(os.environ.get("MAXRANK", "10"))
            row["cutoff"] = float(os.environ.get("FISHER_CUTOFF", "2.0"))
    if "energy" in idata.sample_stats:
        # E-BFMI (Betanin et al. / Vehtari et al.), threshold 0.3. BlackJAX stores the *proposal*
        # state's energy (blackjax/mcmc/hmc.py:171), so a divergent transition writes +inf and the
        # plain arviz call returns NaN -- which is why GEV had no BFMI at all. Non-finite entries
        # are dropped before differencing; that creates one artificial adjacency per dropped draw,
        # which biases E-BFMI upward (looks healthier), so divergence counts are reported too.
        e = np.asarray(idata.sample_stats["energy"].values, dtype=float)
        e = e.reshape(e.shape[0], e.shape[1], -1)[..., 0]      # (chain, draw)
        row["energy_nonfinite"] = int((~np.isfinite(e)).sum())
        bfmi = []
        for c in range(e.shape[0]):
            ec = e[c][np.isfinite(e[c])]
            if ec.size > 10:
                var_e = float(np.var(ec))
                if var_e > 0:
                    bfmi.append(float(np.var(np.diff(ec)) / var_e))
        row["ebfmi"] = round(min(bfmi), 4) if bfmi else None
    if "num_integration_steps" in idata.sample_stats:
        steps = np.asarray(idata.sample_stats["num_integration_steps"].values, dtype=float)
        steps = steps.reshape(steps.shape[0], steps.shape[1], -1)[..., 0]   # (chain, draw)
        row["steps_mean"] = round(float(steps.mean()), 1)
        row["padding_ratio"] = round(float(np.mean(steps.max(axis=0) / np.maximum(steps.mean(axis=0), 1e-9))), 3)
    return row


def run_gpu(model_obj, label, chains, seed, cold, draws=None, warmup=None):
    draws = DRAWS if draws is None else draws
    warmup = WARMUP if warmup is None else warmup
    t0 = time.perf_counter()
    idata = REAL_SAMPLE_PF(
        model_obj, draws=draws, chains=chains, random_seed=seed,
        chees_kwargs={"draws": warmup},
        pathfinder_kwargs={"imm_source": os.environ.get("IMM", "mc"),
                           "max_rank": int(os.environ.get("MAXRANK", "10")),
                           "fisher_gamma": float(os.environ.get("FISHER_GAMMA", "1e-5")),
                           "fisher_cutoff": float(os.environ.get("FISHER_CUTOFF", "2.0"))},
    )
    wall = time.perf_counter() - t0
    return score(idata, wall, "gpu", label, chains, cold,
                 drop_vars=tuple(d.name for d in model_obj.deterministics),
                 draws=draws, warmup=warmup)


def run_cpu(model_obj, label, chains, seed, cold, draws=None, tune=None):
    draws = CPU_DRAWS if draws is None else draws
    tune = CPU_TUNE if tune is None else tune
    t0 = time.perf_counter()
    with model_obj:
        idata = REAL_PM_SAMPLE(draws=draws, tune=tune, chains=chains, cores=chains,
                               target_accept=0.8, random_seed=seed, progressbar=False)
    wall = time.perf_counter() - t0
    return score(idata, wall, "cpu_nuts", label, chains, cold,
                 drop_vars=tuple(d.name for d in model_obj.deterministics),
                 draws=draws, warmup=tune)


def main():
    OUT.parent.mkdir(exist_ok=True, parents=True)
    specs = model_specs()

    def emit(row):
        print(json.dumps(row), flush=True)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    for label in os.environ.get("MODELS", "8schools,gev").split(","):
        build = specs[label.strip()]
        for i, seed in enumerate(SEEDS):
            model_obj = build()
            for arm, cfgs, runner in (("gpu", GPU_CONFIGS or [(c, DRAWS) for c in GPU_CHAINS], run_gpu),
                                      ("cpu", CPU_CONFIGS or [(c, CPU_DRAWS) for c in CPU_CHAINS], run_cpu)):
                if arm not in ARMS:
                    continue
                for chains, draws in cfgs:
                    for rep in range(max(REPEAT, 1)):
                        # A cell that dies must not end a long sweep: the device ceiling (d=175 at
                        # 128 chains OOMs on 6 GB) is a result to record, not a crash to propagate.
                        try:
                            # rep 0 of the first seed carries the compile; later reps are pure cost.
                            emit(runner(model_obj, label, chains, seed, i == 0 and rep == 0,
                                        draws=draws))
                        except Exception as exc:
                            emit({"backend": arm, "model": label, "chains": chains, "seed": seed,
                                  "draws": draws, "cold": i == 0 and rep == 0,
                                  "error": f"{type(exc).__name__}: {exc}"[:200]})


if __name__ == "__main__":
    main()
