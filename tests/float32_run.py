"""Child process of tests/test_float32.py: one ladder model, sampled in a fresh process.

    python float32_run.py TREE MODEL OUT.npz [--init-sources pf,adapt_diag] [--chains N] [--warmup W]
                          [--x64-after-build]

TREE is the source tree that provides `pfcheeshmc/` and `benchmarks/` (this checkout; `tests/make_golden.py` also runs it on the
tag `archive/pre-cleanup`, whose models are built by its own ``tools/r1_bias``).

The caller sets ``PYTENSOR_FLAGS`` (``floatX=float32``, or nothing for float64) in this process's environment: PyTensor's
JAX dispatch sets ``jax_enable_x64 = (floatX == "float64")`` when it is imported, so the type cannot be chosen from
inside a process that has imported it.

For each init source, one `sample_pf_chees_hmc` call on the model (``benchmarks.models.build_model``). OUT.npz holds every posterior (unconstrained
variables too) and sample_stats array as ``SOURCE/posterior/NAME`` and ``SOURCE/stats/NAME``, the numeric attributes as
``SOURCE/attrs/NAME`` and the config as ``SOURCE/config`` (these are the golden files), and ``meta`` as JSON:

    floatX, x64_start (after the imports), x64_built (after ``build_model``), x64_end,
    dtypes {SOURCE: {posterior/NAME: dtype}}, config {SOURCE: idata.attrs["config"]},
    warnings (the "Explicitly requested dtype" ones raised by the calls), x64_warnings (the sampler's warning that
    jax_enable_x64 contradicts floatX), errors {SOURCE: "ExceptionType: first line"} for a call that raised (the
    other sources still run)

``--x64-after-build`` switches ``jax_enable_x64`` on after the model is built: a session that turns x64 on itself
(an example notebook's first cell) under ``floatX=float32``.
"""
import json
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
tree, name, out = sys.argv[1:4]
opts = sys.argv[4:]
sys.path.insert(0, tree)


def option(flag, default):
    return opts[opts.index(flag) + 1] if flag in opts else default


import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytensor  # noqa: E402

from pfcheeshmc import sample_pf_chees_hmc  # noqa: E402

if os.path.isdir(os.path.join(tree, "benchmarks")):  # the tree decides, not an ImportError: PYTHONPATH can put THIS benchmarks/ on an older tree's path
    from benchmarks.models import build_model
else:                                                    # a tree before benchmarks/ (the tag): the notebooks' models, built by tools/r1_bias
    sys.path.insert(0, os.path.join(tree, "tools"))
    from r1_bias import build_model

meta = {"floatX": pytensor.config.floatX, "x64_start": bool(jax.config.jax_enable_x64), "dtypes": {},
        "warnings": [], "x64_warnings": [], "config": {}, "errors": {}}
model = build_model(name)
meta["x64_built"] = bool(jax.config.jax_enable_x64)
if "--x64-after-build" in opts:
    jax.config.update("jax_enable_x64", True)

arrays = {}
for source in option("--init-sources", "pf").split(","):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            idata = sample_pf_chees_hmc(
                model, draws=10, chains=int(option("--chains", 8)), random_seed=3, progressbar=False,
                keep_untransformed=True, pathfinder_kwargs={"init_source": source},
                chees_kwargs={"draws": int(option("--warmup", 20))})
        except Exception as exc:
            idata = None
            meta["errors"][source] = f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0][:200]}"
    messages = [str(w.message) for w in caught]
    meta["warnings"] += [m for m in messages if "Explicitly requested dtype" in m]
    meta["x64_warnings"] += [m for m in messages if "jax_enable_x64 is on" in m]
    if idata is None:
        continue
    meta["dtypes"][source] = {}
    for group, label in ((idata.posterior, "posterior"), (idata.sample_stats, "stats")):
        for k, v in group.data_vars.items():
            arrays[f"{source}/{label}/{k}"] = np.asarray(v.values)
            meta["dtypes"][source][f"{label}/{k}"] = str(v.dtype)
    meta["config"][source] = idata.attrs["config"]
    arrays[f"{source}/config"] = np.array(idata.attrs["config"])
    for k, v in idata.attrs.items():
        if k not in ("config", "pfcheeshmc_version", "diagnostics") and not k.startswith("frozen_inverse_mass_matrix"):   # as make_golden._collect
            arrays[f"{source}/attrs/{k}"] = np.asarray(v)
meta["x64_end"] = bool(jax.config.jax_enable_x64)
np.savez(out, meta=np.array(json.dumps(meta)), **arrays)
