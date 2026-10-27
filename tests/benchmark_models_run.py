"""Child process of tests/test_benchmark_models.py: the ladder models of `benchmarks/models.py`, against the notebook-built ones.

    python benchmark_models_run.py structure TAG_TREE NEW_TREE OUT.json
    python benchmark_models_run.py draws TREE OUT.npz
    python benchmark_models_run.py settings TREE OUT.json [--import-sampler]

The caller sets ``PYTENSOR_FLAGS`` (``floatX=float32``, or nothing for float64) in this process's environment: PyTensor's
JAX dispatch sets ``jax_enable_x64 = (floatX == "float64")`` when it is imported.

structure
    Both builds in one process, so the graphs can be compared: ``tools/r1_bias.build_model`` of TAG_TREE (a `git archive`
    of the tag `archive/pre-cleanup`, which executes the example notebooks) and ``benchmarks.models.build_model`` of
    NEW_TREE. For each ladder model, every check as ``{name: {check: result}}`` in OUT.json: the value variables (name,
    shape, dtype, in order), the free, observed and deterministic variables, the dims and coordinates, the initial point
    (names, dtypes and bits), ``equal_computations`` on the ``logp`` graph (as a scalar, and term by term), on the ``dlogp``
    graph and on the deterministics, and ``compile_logp`` / ``compile_dlogp`` at 100 random points (bit-equal; the number
    of finite ``logp`` values is reported so that the comparison cannot pass on nothing but -inf).
draws
    For each ladder model and seed 1 and 2: a fresh build and one `sample_pf_chees_hmc` call (8 chains, 10 draws, 25 warmup
    draws, the unconstrained variables kept). OUT.npz holds every posterior and sample_stats array as
    ``MODEL/seedS/posterior/NAME`` and ``MODEL/seedS/stats/NAME``, the resolved config as ``MODEL/seedS/config``, and
    ``meta`` (floatX, x64 and the dtypes) as JSON. TREE builds its models with ``benchmarks.models`` or, in a tree before
    it, with ``tools/r1_bias.build_model``.
settings
    Builds every ladder model in TREE and records the process's JAX options (every ``jax_*`` value of ``jax.config``) and its
    environment variables before and after: ``build_model`` must change none. ``--import-sampler`` imports `pfcheeshmc` (and
    so PyTensor's JAX dispatch, which sets x64 from floatX) before the first snapshot; without it, the snapshot precedes that
    import and a build must not make it either.
"""
import json
import os
import sys

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
mode = sys.argv[1]
LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")


def new_tree_builder(tree):
    sys.path.insert(0, tree)
    from benchmarks.models import MODELS, build_model

    assert tuple(MODELS) == LADDER, MODELS
    return build_model


def old_tree_builder(tree):
    """`tools/r1_bias.build_model` of a tree before `benchmarks/`: executes the example notebooks."""
    sys.path.insert(0, os.path.join(tree, "tools"))
    sys.path.insert(0, tree)
    import r1_bias

    return r1_bias.build_model


def any_tree_builder(tree):
    sys.path.insert(0, os.path.join(tree, "tools"))
    sys.path.insert(0, tree)
    if os.path.isdir(os.path.join(tree, "benchmarks")):   # the tree decides, not an ImportError: PYTHONPATH can put THIS benchmarks/ on an older tree's path
        from benchmarks.models import build_model
    else:
        from r1_bias import build_model
    return build_model


def structure(tag_tree, new_tree, out):
    import numpy as np
    import pytensor
    from pytensor.graph.basic import equal_computations

    # the notebooks' pymc_extras and pfcheeshmc imports resolve in the tag; `benchmarks` is the new tree's
    build_old = old_tree_builder(tag_tree)
    build_new = new_tree_builder(new_tree)
    report = {"floatX": pytensor.config.floatX, "models": {}}
    for name in LADDER:
        old, new = build_old(name), build_new(name)
        checks = {}

        def signature(m):
            return [(v.name, list(v.type.shape), str(v.type.dtype)) for v in m.value_vars]

        def names(m, attr):
            return [v.name for v in getattr(m, attr)]

        checks["value_vars"] = signature(old) == signature(new)
        checks["value_vars_n"] = len(old.value_vars)
        for attr in ("free_RVs", "observed_RVs", "deterministics", "potentials", "basic_RVs", "unobserved_RVs"):
            checks[attr] = names(old, attr) == names(new, attr)
        checks["named_vars"] = list(old.named_vars) == list(new.named_vars)
        checks["dims_and_coords"] = (old.named_vars_to_dims == new.named_vars_to_dims
                                     and dict(old.coords) == dict(new.coords))
        ip_old, ip_new = old.initial_point(), new.initial_point()
        checks["initial_point"] = (list(ip_old) == list(ip_new) and all(
            ip_old[k].dtype == ip_new[k].dtype and ip_old[k].shape == ip_new[k].shape
            and np.array_equal(ip_old[k], ip_new[k]) for k in ip_old))
        in_old, in_new = list(old.value_vars), list(new.value_vars)
        checks["logp_graph"] = bool(equal_computations([old.logp()], [new.logp()], in_xs=in_old, in_ys=in_new))
        terms_old, terms_new = old.logp(sum=False), new.logp(sum=False)
        checks["logp_terms_graph"] = len(terms_old) == len(terms_new) and bool(
            equal_computations(list(terms_old), list(terms_new), in_xs=in_old, in_ys=in_new))
        checks["dlogp_graph"] = bool(equal_computations([old.dlogp()], [new.dlogp()], in_xs=in_old, in_ys=in_new))
        checks["deterministics_graph"] = bool(equal_computations(
            list(old.deterministics), list(new.deterministics), in_xs=list(old.free_RVs), in_ys=list(new.free_RVs)))

        f_old, f_new = old.compile_logp(), new.compile_logp()
        g_old, g_new = old.compile_dlogp(), new.compile_dlogp()
        rng = np.random.default_rng(0)
        equal, finite = 0, 0
        for _ in range(100):
            point = {k: (v + 0.5 * rng.normal(size=v.shape)).astype(v.dtype) for k, v in ip_old.items()}
            a, b, ga, gb = f_old(point), f_new(point), g_old(point), g_new(point)
            equal += int(a.dtype == b.dtype and ga.dtype == gb.dtype and np.array_equal(a, b, equal_nan=True)
                         and np.array_equal(ga, gb, equal_nan=True))
            finite += int(np.isfinite(a) and np.isfinite(ga).all())
        checks["points_bit_equal"] = equal
        checks["points_finite"] = finite
        report["models"][name] = checks
    with open(out, "w") as f:
        json.dump(report, f)


def draws(tree, out):
    import jax
    import numpy as np
    import pytensor

    build_model = any_tree_builder(tree)
    from pfcheeshmc import sample_pf_chees_hmc

    arrays = {}
    meta = {"floatX": pytensor.config.floatX, "x64": bool(jax.config.jax_enable_x64), "dtypes": {}}
    for name in LADDER:
        for seed in (1, 2):
            idata = sample_pf_chees_hmc(
                build_model(name), draws=10, chains=8, random_seed=seed, progressbar=False, keep_untransformed=True,
                chees_kwargs={"draws": 25})
            key = f"{name}/seed{seed}"
            for group, label in ((idata.posterior, "posterior"), (idata.sample_stats, "stats")):
                for k, v in group.data_vars.items():
                    arrays[f"{key}/{label}/{k}"] = np.asarray(v.values)
                    meta["dtypes"][f"{key}/{label}/{k}"] = str(v.dtype)
            arrays[f"{key}/config"] = np.array(idata.attrs["config"])
    np.savez(out, meta=np.array(json.dumps(meta)), **arrays)


def settings(tree, out, import_sampler):
    import jax
    import pytensor

    build_model = any_tree_builder(tree)
    if import_sampler:
        import pfcheeshmc  # noqa: F401

    def snapshot():
        return {"jax": {n: repr(getattr(jax.config, n)) for n in sorted(dir(jax.config)) if n.startswith("jax_")},
                "env": dict(os.environ), "floatX": pytensor.config.floatX}

    before = snapshot()
    for name in LADDER:
        build_model(name)
    after = snapshot()
    changed = {f"{section}:{k}": [before[section].get(k), after[section].get(k)]
               for section in ("jax", "env") for k in sorted({*before[section], *after[section]})
               if before[section].get(k) != after[section].get(k)}
    with open(out, "w") as f:
        json.dump({"floatX": before["floatX"], "x64_before": before["jax"]["jax_enable_x64"],
                   "x64_after": after["jax"]["jax_enable_x64"], "n_options": len(before["jax"]),
                   "changed": changed}, f)


if mode == "structure":
    structure(*sys.argv[2:5])
elif mode == "draws":
    draws(*sys.argv[2:4])
else:
    settings(sys.argv[2], sys.argv[3], "--import-sampler" in sys.argv[4:])
