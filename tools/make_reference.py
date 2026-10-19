"""WP0.2 -- reference posteriors for the pf-r1 run.

Usage (from the checkout directory, shared env per A3):

    MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; WT=/home/akbar/pf/wp0refs; cd "$WT"
    pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" \
        python "$WT/tools/make_reference.py" <model> [--chains 8] [--draws 10000] [--tune 2000]
        [--target 0.95] [--seed 42] [--out PATH] [--no-row]

Writes ~/pf-runs/references/<model>.npz, a flat dict keyed by variable name:

    x__<name>   constrained posterior samples, (chains, draws, ...)
    u__<name>   unconstrained value-var samples, (chains, draws, ...)
    mean__<name>, var__<name>          scalars (ddof=1 variance, over all chains)
    mcse_x__<name>, mcse_x2__<name>    Monte Carlo SE of the mean and of the variance
                                       (arviz_stats.mcse, methods 'mean' and 'var')
    meta__<key>  run metadata (seed, target_accept, divergences, gate verdict, ...)

The npz stays OUTSIDE git (A8). A JSONL row is appended to
results/runs/WP0__references.jsonl and the lead assembles
results/references/MANIFEST.json from those rows.

CPU-only: PyMC NUTS (pytensor), no JAX device allocation, no GPU lock needed.
Models load through bench_ladder because its notebook loader stubs pm.sample;
REAL_PM_SAMPLE is captured before the stub (tools/bench_ladder.py:50).

Gate (WP0.2): max R-hat < 1.01 AND bulk ESS > 10,000 per parameter, over every
posterior variable including the transformed ones.
"""

import argparse
import hashlib
import json
import os
import sys
import time

# bench_ladder first: it sets PYTENSOR_FLAGS for the Dot models before anything
# else imports pymc (A3 box trap).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_ladder as bl  # noqa: E402

import arviz as az  # noqa: E402
import arviz_stats  # noqa: E402
import numpy as np  # noqa: E402
import pymc as pm  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rhat_max(x):
    """Max split-R-hat over all elements of a (chains, draws, ...) array."""
    c, d = x.shape[:2]
    flat = x.reshape(c, d, -1)
    return float(np.nanmax([float(np.asarray(az.rhat(flat[:, :, i]))) for i in range(flat.shape[2])]))


def _ess_min(x):
    """Min bulk ESS over all elements of a (chains, draws, ...) array."""
    c, d = x.shape[:2]
    flat = x.reshape(c, d, -1)
    return float(np.nanmin([float(np.asarray(az.ess(flat[:, :, i], method="bulk"))) for i in range(flat.shape[2])]))


def _mcse(x, method):
    """arviz_stats.mcse on a raw (chains, draws, ...) array; scalar float64."""
    arr = arviz_stats.mcse(x, chain_axis=0, draw_axis=1, method=method)
    return float(np.ravel(np.asarray(arr))[0])


def make_reference(model_name, chains, draws, tune, target, seed, out):
    t0 = time.perf_counter()
    model = bl.model_specs()[model_name]()
    # Derive the variable list from the model, not from the group's data_vars:
    # the group may also carry coordinate-shaped entries (e.g. theta_dim_0).
    free = [rv.name for rv in model.free_RVs]
    det = [d.name for d in model.deterministics]
    u_names = [rv.name for rv in model.value_vars]
    c_names = [n for n in free + det if n not in set(u_names)]
    print(f"[{model_name}] value_vars (unconstrained): {u_names}", flush=True)
    print(f"[{model_name}] constrained vars: {c_names}", flush=True)
    names = u_names + c_names
    u_set = set(u_names)

    with model:
        idata = bl.REAL_PM_SAMPLE(
            draws=draws, tune=tune, chains=chains, cores=chains,
            target_accept=target, random_seed=seed, progressbar=False,
            idata_kwargs={"include_transformed": True},
        )
    wall = time.perf_counter() - t0

    post = idata.posterior  # xarray group: one DataArray per posterior variable

    diverging = idata.sample_stats["diverging"]
    div_total = int(np.asarray(diverging).sum())
    divs_per_1k = div_total / (chains * draws) * 1000.0

    # Gate over every posterior variable, including transformed ones. All stats
    # run on raw (chains, draws, ...) numpy: az.rhat / az.ess on a list of
    # var_names return a Dataset in arviz 0.23, which is not directly coercible.
    max_rhat, rhat_worst = -np.inf, None
    min_bulk_ess, ess_worst = np.inf, None
    data = {n: np.asarray(post[n].values) for n in names}
    for n in names:
        r = _rhat_max(data[n])
        if r > max_rhat:
            max_rhat, rhat_worst = r, n
        e = _ess_min(data[n])
        if e < min_bulk_ess:
            min_bulk_ess, ess_worst = e, n

    gate = "PASS" if (max_rhat < 1.01 and min_bulk_ess > 10000.0) else "FAIL"

    arr = {}
    for n in names:
        x = data[n]
        key = ("u__" if n in u_set else "x__") + n
        arr[key] = x
        arr["mean__" + n] = float(x.mean())
        arr["var__" + n] = float(x.var(ddof=1))
        arr["mcse_x__" + n] = _mcse(x, "mean")
        arr["mcse_x2__" + n] = _mcse(x ** 2, "mean")  # MCSE of the second moment

    meta = {
        "model": model_name,
        "seed": int(seed),
        "chains": int(chains),
        "draws": int(draws),
        "tune": int(tune),
        "target_accept": float(target),
        "divergences": div_total,
        "divs_per_1k": divs_per_1k,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "rhat_worst_var": rhat_worst,
        "ess_worst_var": ess_worst,
        "gate": gate,
        "wall_s": wall,
        "pymc": pm.__version__,
        "arviz": az.__version__,
        "arviz_stats": getattr(arviz_stats, "__version__", "installed"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for k, v in meta.items():
        arr[f"meta__{k}"] = np.asarray(v)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, **arr)
    sha = _sha256(out)
    meta["sha256"] = sha
    meta["path"] = out
    print(json.dumps(meta, indent=1, default=str), flush=True)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=sorted(bl.model_specs().keys()))
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--draws", type=int, default=10000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--no-row", action="store_true")
    a = p.parse_args()

    import _common

    out = a.out or os.path.expanduser(f"~/pf-runs/references/{a.model}.npz")
    _common.banner(model=a.model, cold=True, wp="WP0", arm="reference")
    meta = make_reference(a.model, a.chains, a.draws, a.tune, a.target, a.seed, out)

    row = _common.new_row(
        wp="WP0", arm="reference", model=a.model, chains=a.chains, draws_per_chain=a.draws,
        warmup=a.tune, seed=a.seed, rep=0, cold=None,
        kwargs={"target_accept": a.target},
        wall_s=meta["wall_s"], max_rhat=meta["max_rhat"], min_bulk_ess=meta["min_bulk_ess"],
        error=None if meta["gate"] == "PASS" else f"gate_{meta['gate'].lower()}",
    )
    row.update({
        "path": meta["path"], "sha256": meta["sha256"],
        "divs_per_1k": meta["divs_per_1k"],
        "target_accept": meta["target_accept"], "gate": meta["gate"],
    })
    if not a.no_row:
        row_file = os.path.join(os.getcwd(), "results", "runs", "WP0__references.jsonl")
        _common.write_row(row_file, row)
    print(f"[{a.model}] gate={meta['gate']} max_rhat={meta['max_rhat']:.4f} "
          f"min_bulk_ess={meta['min_bulk_ess']:.0f} divs={meta['divergences']} "
          f"wall={meta['wall_s']:.0f}s sha256={meta['sha256'][:12]}", flush=True)


if __name__ == "__main__":
    main()
