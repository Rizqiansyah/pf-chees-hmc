#!/usr/bin/env python
"""The R1 gate: score `sample_pf_chees_hmc` by BIAS against a NUTS reference, not by split R-hat.

In the many-short-chain regime (C ~ 1,024 chains, D ~ 10 draws) split R-hat is not a convergence signal: cutting ONE
converged 32x2,000 run into 1024x39, 512x20, 512x78 and 128x313 reproduces every "degraded" R-hat, so it measures the
estimator, not the sampler. The gate is bias against a long NUTS run, in the unconstrained space:

    b2(f) = (mean_ens f - mean_ref f)^2 / var_ref f     for f in {x_i, x_i^2}

and `b2_max` is its maximum over the leaves of every value variable and both moments. It has its own Monte-Carlo floor,
max_f (mcse_f^2 / var_ref_f) * 2 ln(2d), written with every cell. G-R1: pass = b2_max < 0.01 on every seed with
C*D >= 2,000. A cell whose floor is above 0.005 is NOT RESOLVABLE, not a pass.

One cell is a (model, W, seed) call of the public `sample_pf_chees_hmc` (`keep_untransformed=True` puts the value
variables in the posterior; nothing else is patched). One cell is one JSON row, appended to `--out`. The reference draws
are `~/pf-runs/references/<model>.npz` (or `$PF_REFERENCE_DIR`), outside git; their sha256 is checked against
`benchmarks/references.json` before a cell runs, so a gate is never scored against a file the manifest does not describe.
`--check-references-only` checks the files and exits.

    python benchmarks/bias.py --models 8schools --chains 1024 --draws 10 --warmups 25 --seeds 1,2,3
    python benchmarks/bias.py --models 8schools,gev --check-references-only
    benchmarks/gpu_run.sh pixi run python benchmarks/bias.py ...          (on a GPU the gate must hold the GPU lock)

The row (`pf-bias/v1`): the cell, `b2_max`, `b2_floor`, `b2_argmax`, `g_r1`, split R-hat (information only), the frozen
step size and trajectory length, gradients per chain, divergences per 1,000 draws, `wall_s` (`cold`: the first cell of the
process, which includes the compile), `effective` (what the sampler RESOLVED, `idata.attrs["config"]`), the reference's
sha256 and `draws_sha256`, the hash of the unconstrained draws, so that a change which must not move the draws can say so.

The on-disk compilation cache is OFF unless the environment says otherwise (`PFCHEESHMC_COMPILATION_CACHE`), so that a
`cold` row is cold. floatX is PyTensor's: `PYTENSOR_FLAGS=floatX=float32` before the run (it is opt-in, and the gate shows
what it costs).
"""
import argparse
import hashlib
import json
import os
import sys
import time

# This repo's path has a space ("Python packages"). PyTensor puts the environment prefix into blas__ldflags as an unquoted
# -Wl,-rpath token and its linker dies on it for any model that compiles a Dot. Without the rpath numpy supplies BLAS at run time.
_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "blas__ldflags" not in _flags:
    os.environ["PYTENSOR_FLAGS"] = ",".join(f for f in (_flags, "blas__ldflags=-llapack -lblas -lcblas -lm") if f)
os.environ.setdefault("PFCHEESHMC_COMPILATION_CACHE", "0")      # a cold cell must not read a disk cache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

SCHEMA = "pf-bias/v1"
MANIFEST = os.path.join(REPO, "benchmarks", "references.json")


# ---------------------------------------------------------------- references

def reference_path(model):
    """`$PF_REFERENCE_DIR/<model>.npz`, by default `~/pf-runs/references/<model>.npz`."""
    base = os.environ.get("PF_REFERENCE_DIR") or os.path.expanduser("~/pf-runs/references")
    return os.path.join(base, f"{model}.npz")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def verify_reference(model):
    """The reference file's sha256, which must be the manifest's. A missing file or a different hash raises."""
    path = reference_path(model)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no reference at {path}; the gate cannot be evaluated without it. Build it with "
            f"benchmarks/make_reference.py {model}. Do not substitute a shorter run.")
    entry = load_manifest()["models"].get(model)
    if entry is None:
        raise KeyError(f"{model} is not in {MANIFEST}")
    got = sha256_file(path)
    if got != entry["sha256"]:
        raise ValueError(f"{path} has sha256 {got}, the manifest says {entry['sha256']}: "
                         "not the file the manifest describes (rebuilt without updating it, or edited)")
    return got


def load_reference(model, names):
    """{variable name: (n_draws, n_leaves)} float64 for the model's value vars, from the reference npz.

    The file is flat: `u__<name>` is the unconstrained draws of value variable `name`, (chains, draws, *shape). The first two
    axes fold into rows and the rest into leaves; a name the file lacks is an error, never a smaller score (b2_max over a
    subset of leaves can pass while an unscored variable is badly biased).
    """
    path = reference_path(model)
    z = np.load(path)
    missing = [n for n in names if f"u__{n}" not in z.files]
    if missing:
        raise KeyError(f"{path} is missing unconstrained draws for {missing}; present: "
                       f"{sorted(k[3:] for k in z.files if k.startswith('u__'))}")
    out = {}
    for n in names:
        a = np.asarray(z[f"u__{n}"], dtype="float64")
        out[n] = a.reshape(-1, 1) if a.ndim == 1 else a.reshape(int(np.prod(a.shape[:2])), -1)
    return out


def value_var_names(model_name):
    """Unconstrained variable names of a ladder model, in value_vars order.

    Read from the model, not from a file, and deliberately not sorted: the kernel's dict-flatten order is sorted, and the
    difference between the two orders is the bug the start-ordering fix removed.
    """
    from benchmarks.models import build_model

    return [v.name for v in build_model(model_name).value_vars]


# ---------------------------------------------------------------- scoring

def flatten_run(by_name, chains, draws):
    """{name: (chains, draws, *shape)} -> ((chains*draws, leaves), labels), sorted by name.

    The shape is asserted, never inferred. A 2-D array is ambiguous -- it is either (chain, draw) for a SCALAR value
    variable or (rows, leaves) for a pre-flattened reference -- and guessing wrong silently under-samples exactly the
    scalar parameters (all three of GEV, 5 of radon's 7). Because b2_max is a max over leaves, an under-sampled scalar
    simply stops being the maximum: the number stays plausible and the gate stops covering those parameters.
    """
    cols, labels = [], []
    for name in sorted(by_name):
        a = np.asarray(by_name[name], dtype="float64")
        if a.ndim == 1:
            if a.size != chains * draws:
                raise ValueError(f"{name}: 1-D run array has {a.size} entries, "
                                 f"expected chains*draws = {chains * draws}")
            a = a.reshape(chains * draws, 1)
        else:
            if a.shape[0] != chains or a.shape[1] != draws:
                raise ValueError(f"{name}: expected (chains={chains}, draws={draws}, ...), "
                                 f"got {a.shape}")
            a = a.reshape(chains * draws, -1)
        cols.append(a)
        labels += [name] if a.shape[1] == 1 else [f"{name}[{j}]" for j in range(a.shape[1])]
    return np.concatenate(cols, axis=1), labels


def flatten_ref(by_name):
    """{name: (rows, leaves)} -> ((rows, total_leaves), labels), sorted by name. All variables carry the same rows."""
    cols, labels = [], []
    rows = None
    for name in sorted(by_name):
        a = np.asarray(by_name[name], dtype="float64")
        if a.ndim != 2:
            raise ValueError(f"reference {name}: expected (rows, leaves), got {a.shape}")
        if rows is None:
            rows = a.shape[0]
        elif a.shape[0] != rows:
            raise ValueError(f"reference row counts disagree: {name} has {a.shape[0]}, others {rows}")
        cols.append(a)
        labels += [name] if a.shape[1] == 1 else [f"{name}[{j}]" for j in range(a.shape[1])]
    return np.concatenate(cols, axis=1), labels


def argmax_name(labels, k):
    """Name the leaf behind bias()'s index, which runs over [x_1..x_d, x^2_1..x^2_d] (2d entries).

    Indexing `labels` (d entries) with it is an IndexError when k >= d and, worse, names the wrong parameter whenever the
    x^2 half wins for an early leaf.
    """
    d = len(labels)
    return f"{labels[k % d]} ({'x' if k < d else 'x^2'})"


def bias(ens, ref, mcse_mean, mcse_mean_sq):
    """b2_max over {x, x^2}, its argmax index, the MC noise floor and both halves. Pure numpy."""
    d = ens.shape[1]
    if ref.shape[1] != d:
        raise ValueError(f"reference has {ref.shape[1]} leaves, run has {d}")
    den = np.maximum(ref.var(0, ddof=1), 1e-300)
    den2 = np.maximum((ref ** 2).var(0, ddof=1), 1e-300)
    b2m = (ens.mean(0) - ref.mean(0)) ** 2 / den
    b2v = ((ens ** 2).mean(0) - (ref ** 2).mean(0)) ** 2 / den2
    floor = max(float(np.max(np.asarray(mcse_mean, dtype="float64") ** 2 / den)),
                float(np.max(np.asarray(mcse_mean_sq, dtype="float64") ** 2 / den2))) \
        * 2.0 * float(np.log(2.0 * d))
    allb = np.concatenate([b2m, b2v])
    return float(np.nanmax(allb)), int(np.nanargmax(allb)), floor, b2m, b2v


def g_r1(b2_max, floor, c_times_d):
    """G-R1 as a verdict string. Not resolvable is NOT a pass.

    A coarse floor does not neutralise a bias far above it: only a b² inside its noise band is unresolvable.
    """
    if c_times_d < 2000:
        return "not eligible (C*D < 2000)"
    if floor > 0.005:
        if b2_max > max(0.01, 3.0 * floor):
            return "fail"
        return f"not resolvable (floor {floor:.4f} > 0.005)"
    return "pass" if b2_max < 0.01 else "fail"


def score_ensemble(unc, ref, chains, draws):
    """b2 for one run. `unc` is {name: (chains, draws, leaves)}, `ref` is {name: (rows, leaves)}."""
    import arviz_stats as AS

    ens, labels = flatten_run(unc, chains, draws)
    reff, rlabels = flatten_ref(ref)
    if labels != rlabels:
        raise KeyError(f"run/reference leaf mismatch: {labels[:4]} vs {rlabels[:4]} "
                       f"(wrong space or missing value var?)")
    stacked = np.concatenate(
        [np.asarray(unc[n], dtype="float64").reshape(chains, draws, -1) for n in sorted(unc)],
        axis=2)                                                        # (C, D, d)
    d = stacked.shape[2]
    b2_max, k, floor, _, _ = bias(
        ens, reff,
        np.asarray(AS.mcse(stacked, method="mean", chain_axis=0, draw_axis=1)),
        np.asarray(AS.mcse(stacked ** 2, method="mean", chain_axis=0, draw_axis=1)))
    return {"labels": labels, "d": int(d), "b2_max": b2_max,
            "b2_argmax": argmax_name(labels, k), "b2_floor": floor}


def draws_digest(unc):
    """sha256 of the unconstrained draws ({name: array}, by sorted name, as float64): equal digests, equal draws."""
    h = hashlib.sha256()
    for n in sorted(unc):
        h.update(n.encode())
        h.update(np.ascontiguousarray(np.asarray(unc[n], dtype="float64")).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- one cell

def run_cell(model_name, chains, draws, W, seed, cold, ref, ref_sha, pathfinder_kwargs=None, chees_kwargs=None):
    """One (model, W, seed) cell -> one row. A cell that fails is a row with `error`, not a gap in the record."""
    import jax
    import pytensor
    from benchmarks.models import build_model
    from pfcheeshmc import sample_pf_chees_hmc

    pf_kw = dict(pathfinder_kwargs or {})
    ch_kw = dict(chees_kwargs or {})
    if "draws" in ch_kw:
        raise ValueError("chees_kwargs 'draws' is the warmup length W: pass --warmups")
    ch_kw["draws"] = W
    row = {"schema": SCHEMA, "model": model_name, "chains": chains, "draws_per_chain": draws, "warmup": W,
           "seed": seed, "cold": cold, "floatX": pytensor.config.floatX,
           "x64": bool(jax.config.jax_enable_x64), "device": [f"{d.device_kind}:{d.id}" for d in jax.devices()],
           "git_sha": _git_sha(), "kwargs": {"pathfinder_kwargs": pf_kw, "chees_kwargs": ch_kw},
           "reference_sha256": ref_sha, "error": None}
    t0 = time.perf_counter()
    try:
        model = build_model(model_name)
        names = [v.name for v in model.value_vars]
        t0 = time.perf_counter()                 # the wall clock is the sampler call: the build and the scoring are outside it
        idata = sample_pf_chees_hmc(
            model, chains=chains, draws=draws, random_seed=seed, progressbar=False, keep_untransformed=True,
            pathfinder_kwargs=pf_kw, chees_kwargs=ch_kw)
        row["wall_s"] = round(time.perf_counter() - t0, 2)
        unc = {}
        for n in names:
            a = np.asarray(idata.posterior[n].values, dtype="float64")
            unc[n] = a.reshape(a.shape[0], a.shape[1], -1)
        sc = score_ensemble(unc, ref, chains, draws)
        row.update(d=sc["d"], b2_max=sc["b2_max"], b2_floor=sc["b2_floor"], b2_argmax=sc["b2_argmax"],
                   g_r1=g_r1(sc["b2_max"], sc["b2_floor"], chains * draws), draws_sha256=draws_digest(unc))

        import arviz as az

        rh = az.rhat(idata.posterior)             # information only: it inflates on short chains that are not biased
        finite = [float(np.max(v.values)) for v in rh.values() if np.all(np.isfinite(v.values))]
        steps = np.asarray(idata.sample_stats["num_integration_steps"].values, dtype="int64")
        steps = steps.reshape(steps.shape[0], steps.shape[1], -1)[..., 0]                  # (C, D)
        divs = np.asarray(idata.sample_stats["diverging"].values, dtype="bool")
        attrs = dict(idata.attrs)
        row.update(max_rhat=round(max(finite), 4) if finite else None,
                   frozen_step_size=attrs.get("frozen_step_size"), frozen_L_nominal=attrs.get("frozen_L_nominal"),
                   warmup_steps_used=attrs.get("warmup_steps_used"),
                   steps_mean=float(steps.mean()),
                   grads_per_chain_warmup=attrs.get("warmup_grad_evals_per_chain"),
                   grads_per_chain_sampling=attrs.get("sampling_grad_evals_per_chain"),
                   divs_per_1k=float(divs.sum() / max(divs.size, 1) * 1000.0),
                   effective=json.loads(attrs["config"]))
    except Exception:                            # noqa: BLE001
        import traceback
        row.update(error=traceback.format_exc().strip().splitlines()[-1][:400],
                   wall_s=round(time.perf_counter() - t0, 2))
    return row


def _git_sha():
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=REPO).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def write_row(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def describe(row):
    return (f"[{row['model']} C={row['chains']} D={row['draws_per_chain']} W={row['warmup']} seed={row['seed']}] "
            + (f"error={row['error']}" if row["error"] else
               f"b2_max={row['b2_max']} floor={row['b2_floor']} argmax={row['b2_argmax']} G-R1={row['g_r1']} "
               f"eps={row['frozen_step_size']} L={row['frozen_L_nominal']} "
               f"grads/chain={row['grads_per_chain_warmup']}+{row['grads_per_chain_sampling']} "
               f"divs/1k={row['divs_per_1k']} splitRhat={row['max_rhat']} wall={row['wall_s']}s"))


# ---------------------------------------------------------------- command line

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", required=True, help="comma list of ladder models (benchmarks/models.py)")
    ap.add_argument("--chains", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=10, help="draws per chain")
    ap.add_argument("--warmups", help="comma list of warmup lengths W (chees_kwargs draws)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default=os.path.expanduser("~/pf-runs/bias.jsonl"), help="JSONL, appended")
    ap.add_argument("--pathfinder", default="{}", help="extra pathfinder_kwargs, JSON; the row records them")
    ap.add_argument("--chees", default="{}", help="extra chees_kwargs, JSON (not draws: that is --warmups)")
    ap.add_argument("--check-references-only", action="store_true",
                    help="check the reference files (sha256 against the manifest, the value variables, the shapes) and exit")
    return ap


def check_references(models):
    for model_name in models:
        try:
            names = value_var_names(model_name)
            verify_reference(model_name)
            ref = load_reference(model_name, names)
            leaves = sum(a.shape[1] for a in ref.values())
            print(f"[ref-guard] {model_name}: PASS  sha256 matches the manifest, {len(names)} value vars, {leaves} leaves, "
                  f"{next(iter(ref.values())).shape[0]} reference draws")
        except Exception as exc:                 # noqa: BLE001
            print(f"[ref-guard] {model_name}: FAIL  {type(exc).__name__}: {str(exc)[:300]}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.check_references_only:
        check_references(models)
        return
    if not args.warmups:
        raise SystemExit("--warmups is required")

    import jax
    import pytensor
    import pfcheeshmc  # noqa: F401  (its import sets PyTensor's JAX dispatch, which sets x64 from floatX)

    if pytensor.config.floatX != "float64":
        jax.config.update("jax_enable_x64", False)          # a float32 run is float32 end to end
    if jax.default_backend() != "cpu" and os.environ.get("PF_GPU_LOCK") != "1":
        raise SystemExit("the gate on a GPU must hold the GPU lock: benchmarks/gpu_run.sh <command>")
    pf_kw, ch_kw = json.loads(args.pathfinder), json.loads(args.chees)
    print(f"[bias] pfcheeshmc={pfcheeshmc.__file__} floatX={pytensor.config.floatX} x64={jax.config.jax_enable_x64} "
          f"devices={jax.devices()} C={args.chains} D={args.draws}", flush=True)

    cold = True
    for model_name in models:
        ref_sha = verify_reference(model_name)
        ref = load_reference(model_name, value_var_names(model_name))
        for W in [int(w) for w in args.warmups.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                row = run_cell(model_name, args.chains, args.draws, W, seed, cold, ref, ref_sha, pf_kw, ch_kw)
                cold = False
                write_row(args.out, row)
                print(describe(row), flush=True)


if __name__ == "__main__":
    main()
