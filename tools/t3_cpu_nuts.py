#!/usr/bin/env python
"""M5 (R2-WP3a) -- CPU NUTS time-to-pass, scored by the SAME instrument as the GPU R1 rows.

Replaces run 1's `tools/legacy/t3_cpu.py` (quarantined, never run it), which got two things
wrong (run-2 plan Part B 1.2):

  * it compared b2 with an UNSQUARED floor (MCSE/sd instead of its square), so resolvable cells
    read as "not resolvable";
  * it scored CONSTRAINED `x__` keys, which drops tau_log__, xi_interval__, sigma_log__ and radon's
    three sigma_*_log__ -- the hardest coordinates.

Here there is no scorer of its own. The value-variable names come from `model.value_vars`, the
arrays are the unconstrained draws in the posterior group (`include_transformed=True` puts them
there), the reference is `r1_bias.load_reference`, the statistic and floor are
`r1_bias.score_ensemble`, and the gate is `r1_bias.g_r1`. A change to the instrument therefore
changes the GPU and CPU rows together.

Gradients per chain are exact: the sum of `n_steps` over `warmup_sample_stats` and `sample_stats`,
divided by chains. `discard_tuned_samples=False` is what makes the warmup group exist; without it
the tuning gradients silently vanish from the count.

Rows: pf-r2/v1 (pf-r1/v1 fields, null where NUTS has no equivalent, plus `step`, `pooled`,
`effective`, `tune`, `draws`, `core_seconds`, `target_accept`, `g_r1`, `loadavg_start/end`). One
process per (model, tune, draws) keeps the clock honest; the script may loop over more. The
pytensor C-module cache lives on disk, so a model never compiled before can inflate its first
row's wall: run the model once untimed first.

Usage (CPU only; never during a timed GPU batch -- tag overlap with --cpu-contended):
  pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python $WT/tools/t3_cpu_nuts.py \
      --models 8schools --tune 200 --draws 100 --seeds 1,2,3 --out results/r2/runs/M5__cpu_nuts.jsonl

Note C*D = 32 * draws, and G-R1 needs C*D >= 2,000: at draws=50 the verdict is always
"not eligible (C*D < 2000)", whatever b2 is.
"""

import argparse
import json
import os
import resource
import sys
import time

# One thread per chain is what makes "cores = chains" mean what it says (the CPU-curve protocol).
# These must be set before numpy is imported; bench_ladder's own setdefault is then too late.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import bench_ladder as bl  # noqa: E402  -- bl.REAL_PM_SAMPLE is pm.sample captured before any stub
import r1_bias as rb  # noqa: E402

SCHEMA = "pf-r2/v1"
STEP = "M5"
CHAINS = 32
TARGET_ACCEPT = 0.8          # PyMC's default, recorded in every row because the plan asks for it


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
    """(idata, wall_s, core_seconds) for one `bl.REAL_PM_SAMPLE` call."""
    if not getattr(bl.REAL_PM_SAMPLE, "__module__", "").startswith("pymc"):
        raise RuntimeError(f"bench_ladder.REAL_PM_SAMPLE is {bl.REAL_PM_SAMPLE!r}, not pm.sample: "
                           "a stub would produce a fake instant cell")
    c0, t0 = _cpu_seconds(), time.perf_counter()
    with model:
        idata = bl.REAL_PM_SAMPLE(**kw)
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


def score_run(unc, ref, chains, draws, nested_limit=1.01):
    """`r1_bias.score_ensemble` plus the `r1_bias.g_r1` verdict. The T3 scoring entry point."""
    sc = rb.score_ensemble(unc, ref, chains, draws, None, nested_limit)
    sc["g_r1"] = rb.g_r1(sc["b2_max"], sc["b2_floor"], chains * draws)
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
        out[key + "_list"] = n.sum(1).astype("int64").tolist()
    out["total"] = out["warmup"] + out["sampling"]
    return out


# ---------------------------------------------------------------- one cell

def run_cell(model, model_name, names, ref, tune, draws, seed, chains=CHAINS, cores=None,
             cpu_contended=False):
    """Sample, score and return one pf-r2/v1 row (a failed cell is a row, not a gap)."""
    from _common import git_sha, new_row

    kw = sample_kwargs(draws, tune, seed, chains, cores)
    row = new_row(wp=STEP, arm="cpu:nuts", model=model_name, chains=chains, draws_per_chain=draws,
                  warmup=tune, seed=seed, rep=0, cold=False, device="cpu",
                  cpu_contended=bool(cpu_contended), git_sha=git_sha(),
                  kwargs=json.dumps({"sampler_fn": "pm.sample", **{
                      k: v for k, v in kw.items() if k != "progressbar"}}, sort_keys=True))
    row.update(schema=SCHEMA, step=STEP, pooled=False, effective=None, tune=int(tune),
               draws=int(draws), target_accept=TARGET_ACCEPT, cores=kw["cores"],
               space="unconstrained", loadavg_start=list(os.getloadavg()), loadavg_end=None,
               core_seconds=None, grads_per_chain_total=None, g_r1=None, scored_vars=None)
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
                   grads_per_chain_warmup_list=g["warmup_list"],
                   grads_per_chain_sampling_list=g["sampling_list"],
                   steps_mean=float(g["sampling"] / draws),
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
                    help=f"plan fixes {CHAINS} (= cores); other values are for debugging and are "
                         "recorded in the row")
    ap.add_argument("--cpu-contended", action="store_true",
                    help="tag rows taken while another CPU-heavy job (or a GPU batch) was running")
    args = ap.parse_args(argv)

    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        model = rb.build_model(model_name)
        names = [v.name for v in model.value_vars]
        ref = rb.load_reference(model_name, names)
        for tune in _ints(args.tune):
            for draws in _ints(args.draws):
                for seed in _ints(args.seeds):
                    row = run_cell(model, model_name, names, ref, tune, draws, seed,
                                   chains=args.chains, cpu_contended=args.cpu_contended)
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
