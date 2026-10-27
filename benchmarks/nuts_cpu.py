#!/usr/bin/env python
"""CPU NUTS time-to-pass, scored by the SAME instrument as the bias gate (`benchmarks/bias.py`).

The comparison the README's CPU numbers rest on: PyMC NUTS with 32 chains on 32 cores, at a given (tune, draws), scored
as the GPU rows are. There is no scorer of its own: the value-variable names come from `model.value_vars`, the arrays are
the unconstrained draws in the posterior group (`include_transformed=True` puts them there), the reference is
`bias.load_reference`, the statistic and floor are `bias.score_ensemble`, and the gate is `bias.g_r1`. A change to the
instrument therefore changes the GPU and CPU rows together.

Gradients per chain are exact: the sum of `n_steps` over `warmup_sample_stats` and `sample_stats`, divided by chains.
`discard_tuned_samples=False` is what makes the warmup group exist; without it the tuning gradients silently vanish from
the count.

One row per (model, tune, draws, seed), appended to `--out`: the cell, `b2_max`, `b2_floor`, `b2_argmax`, `g_r1`,
`wall_s`, `core_seconds` (process plus children CPU time), the gradient counts, divergences, the load average at the start
and the end. The pytensor C-module cache lives on disk, so a model never compiled before can inflate its first row's wall:
run the model once untimed first. One process per (model, tune, draws) keeps the clock honest; the script may loop over more.

Usage (CPU only; one thread per chain, so that "cores = chains" means what it says; never during a timed GPU batch):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      python benchmarks/nuts_cpu.py --models 8schools --tune 200 --draws 100 --seeds 1,2,3 --out ~/pf-runs/nuts_cpu.jsonl

Note C*D = 32 * draws, and G-R1 needs C*D >= 2,000: at draws=50 the verdict is always "not eligible (C*D < 2000)",
whatever b2 is.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

# One thread per chain is what makes "cores = chains" mean what it says (the CPU-curve protocol).
# These must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# This repo's path has a space ("Python packages"): PyTensor puts the environment prefix into blas__ldflags as an unquoted
# -Wl,-rpath token and its linker dies on it for any model that compiles a Dot.
_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "blas__ldflags" not in _flags:
    os.environ["PYTENSOR_FLAGS"] = ",".join(f for f in (_flags, "blas__ldflags=-llapack -lblas -lcblas -lm") if f)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pymc as pm  # noqa: E402

from benchmarks import bias  # noqa: E402
from benchmarks.models import build_model  # noqa: E402

SCHEMA = "pf-nuts/v1"
CHAINS = 32
TARGET_ACCEPT = 0.8          # PyMC's default, recorded in every row


# ---------------------------------------------------------------- sampling

def sample_kwargs(draws, tune, seed, chains=CHAINS, cores=None):
    """The exact keyword set handed to `pm.sample`. A function so a test can pin it."""
    return dict(draws=int(draws), tune=int(tune), chains=int(chains),
                cores=int(chains if cores is None else cores), target_accept=TARGET_ACCEPT,
                random_seed=int(seed), progressbar=False,
                discard_tuned_samples=False,                      # keeps warmup_sample_stats
                idata_kwargs={"include_transformed": True})       # keeps tau_log__ and friends


def _cpu_seconds():
    """Process CPU time including reaped children (pm.sample's chains are child processes)."""
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def timed_sample(model, **kw):
    """(idata, wall_s, core_seconds) for one `pm.sample` call."""
    c0, t0 = _cpu_seconds(), time.perf_counter()
    with model:
        idata = pm.sample(**kw)
    return idata, time.perf_counter() - t0, _cpu_seconds() - c0


# ---------------------------------------------------------------- scoring (no scorer of its own)

def unconstrained_from_idata(idata, names, chains, draws):
    """{value-var name: (chains, draws, leaves)} from the POSTERIOR group, post-tune draws only.

    With `discard_tuned_samples=False` PyMC 5.28 files the tuning draws under `warmup_posterior`,
    so `posterior` is already the sampling phase; the draw count is asserted rather than assumed.
    A missing name means `include_transformed` was off: it is an error, never a smaller score.
    """
    post = idata.posterior
    missing = [n for n in names if n not in post.data_vars]
    if missing:
        raise KeyError(f"value vars {missing} are not in the posterior (have "
                       f"{sorted(post.data_vars)}): was idata_kwargs include_transformed=True set?")
    unc = {}
    for n in names:
        a = np.asarray(post[n].values, dtype="float64")
        if a.shape[:2] != (chains, draws):
            raise ValueError(f"{n}: posterior is {a.shape[:2]} (chain, draw), expected "
                             f"({chains}, {draws}); tuned draws leaking into the score?")
        unc[n] = a.reshape(chains, draws, -1)
    return unc


def score_run(unc, ref, chains, draws):
    """`bias.score_ensemble` plus the `bias.g_r1` verdict. The scoring entry point."""
    sc = bias.score_ensemble(unc, ref, chains, draws)
    sc["g_r1"] = bias.g_r1(sc["b2_max"], sc["b2_floor"], chains * draws)
    return sc


def score_idata(idata, names, ref, chains, draws):
    """Unconstrained draws out of an InferenceData, then `score_run`."""
    return score_run(unconstrained_from_idata(idata, names, chains, draws), ref, chains, draws)


def grads_per_chain(idata, chains, tune, draws):
    """Exact per-chain gradient counts: sum of `n_steps` over warmup + sampling, / chains."""
    out = {}
    for key, group, n_draw in (("warmup", "warmup_sample_stats", tune),
                               ("sampling", "sample_stats", draws)):
        if group not in idata.groups():
            raise RuntimeError(f"idata has no {group}: the gradient count would silently omit "
                               "tuning (discard_tuned_samples must be False)")
        n = np.asarray(idata[group]["n_steps"].values, dtype="float64")       # (chain, draw)
        if n.shape != (chains, n_draw) or not np.isfinite(n).all():
            raise ValueError(f"{group}.n_steps is {n.shape} (finite: {np.isfinite(n).all()}), "
                             f"expected ({chains}, {n_draw}) and finite")
        out[key] = float(n.sum() / chains)
    out["total"] = out["warmup"] + out["sampling"]
    return out


# ---------------------------------------------------------------- one cell

def run_cell(model, model_name, names, ref, tune, draws, seed, chains=CHAINS, cores=None):
    """Sample, score and return one row (a failed cell is a row, not a gap)."""
    kw = sample_kwargs(draws, tune, seed, chains, cores)
    row = {"schema": SCHEMA, "model": model_name, "chains": chains, "draws": int(draws), "tune": int(tune),
           "seed": seed, "cores": kw["cores"], "target_accept": TARGET_ACCEPT, "space": "unconstrained",
           "git_sha": _git_sha(), "loadavg_start": list(os.getloadavg()), "loadavg_end": None,
           "wall_s": None, "core_seconds": None, "d": None, "b2_max": None, "b2_floor": None, "b2_argmax": None,
           "g_r1": None, "scored_vars": None, "grads_per_chain_warmup": None, "grads_per_chain_sampling": None,
           "grads_per_chain_total": None, "divs_per_1k": None, "warmup_divs": None, "error": None}
    t0 = time.perf_counter()
    try:
        idata, wall, core_s = timed_sample(model, **kw)
        row.update(wall_s=round(wall, 2), core_seconds=round(core_s, 1),
                   loadavg_end=list(os.getloadavg()))
        g = grads_per_chain(idata, chains, tune, draws)
        divs = np.asarray(idata.sample_stats["diverging"].values, dtype="bool")
        wdivs = np.asarray(idata.warmup_sample_stats["diverging"].values, dtype="bool")
        row.update(grads_per_chain_warmup=g["warmup"], grads_per_chain_sampling=g["sampling"],
                   grads_per_chain_total=g["total"],
                   divs_per_1k=float(divs.sum() / max(divs.size, 1) * 1000.0),
                   warmup_divs=int(wdivs.sum()))
        sc = score_idata(idata, names, ref, chains, draws)
        row.update(d=sc["d"], b2_max=sc["b2_max"], b2_floor=sc["b2_floor"],
                   b2_argmax=sc["b2_argmax"], g_r1=sc["g_r1"], scored_vars=sorted(names))
    except Exception:                       # noqa: BLE001
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        row.update(error=f"{tb[-1]}"[:400], g_r1="error",
                   wall_s=row["wall_s"] if row["wall_s"] is not None
                   else round(time.perf_counter() - t0, 2),
                   loadavg_end=list(os.getloadavg()))
    return row


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=REPO).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _ints(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", required=True, help="comma list, e.g. 8schools,gev")
    ap.add_argument("--tune", required=True, help="comma list of tuning-iteration counts")
    ap.add_argument("--draws", required=True, help="comma list of post-tune draws per chain")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", required=True, help="JSONL, appended")
    ap.add_argument("--chains", type=int, default=CHAINS,
                    help=f"{CHAINS} (= cores) is the protocol; other values are for debugging and are "
                         "recorded in the row")
    args = ap.parse_args(argv)

    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        model = build_model(model_name)
        names = [v.name for v in model.value_vars]
        bias.verify_reference(model_name)
        ref = bias.load_reference(model_name, names)
        for tune in _ints(args.tune):
            for draws in _ints(args.draws):
                for seed in _ints(args.seeds):
                    row = run_cell(model, model_name, names, ref, tune, draws, seed, chains=args.chains)
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    with open(args.out, "a") as fh:
                        fh.write(json.dumps(row, default=str) + "\n")
                    print(f"  [{model_name} tune={tune} draws={draws} seed={seed}] "
                          f"b2_max={row['b2_max']} floor={row['b2_floor']} G-R1={row['g_r1']} "
                          f"grads/chain={row['grads_per_chain_total']} "
                          f"divs/1k={row['divs_per_1k']} wall={row['wall_s']}s "
                          f"core_s={row['core_seconds']} err={row['error']}", flush=True)


if __name__ == "__main__":
    main()
