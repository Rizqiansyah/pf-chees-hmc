"""Writes the golden draws of `tests/test_golden.py`: ``tests/data/golden_<case>.npz``.

    python tests/make_golden.py [--tree TREE] [--out-dir DIR] [CASE ...]        (no CASE: all of them)

Each case is one `sample_pf_chees_hmc` call (or several in one process, or one stage method) on a small shape, and its file
holds every posterior and sample_stats array (the unconstrained variables too), the numeric attributes of the
InferenceData (``frozen_step_size``, ``frozen_L_nominal``, the gradient counts, ...) and the resolved config as JSON.
`tests/test_golden.py` runs the case again and requires every array to be equal to the last bit.

**A regeneration changes what the sampler is required to produce. It needs a reason, written in the commit message.**
Files made in this tree before a change must come out unchanged after a change that is meant to leave the draws alone.

TREE is the source tree whose `pfcheeshmc/` is run (default: this checkout). A tree before `benchmarks/` (the tag
`archive/pre-cleanup`, a `git archive` of it) builds the ladder models with its own ``tools/r1_bias.build_model``; the tree
decides which (``os.path.isdir(TREE/benchmarks)``), not an ImportError, because PYTHONPATH can put this checkout's
`benchmarks/` on an older tree's path. Only options the current sampler keeps are passed, so the cases run on both.
The float32 cases are `tests/float32_run.py` in a child process with ``PYTENSOR_FLAGS=floatX=float32`` (PyTensor's JAX dispatch
sets ``jax_enable_x64`` from floatX when it is imported, so the type cannot be chosen from inside a process that has imported
it); `tests/test_float32.py` compares the children it runs anyway with these files.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DATA = HERE / "data"
BLAS = "blas__ldflags=-llapack -lblas -lcblas -lm"
CHAINS, DRAWS, WARMUP = 8, 10, 20
LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")


def _tree_modules(tree):
    """Import ``pfcheeshmc`` and a ``build_model`` from TREE (put first on the path)."""
    tree = str(tree)
    sys.path[:0] = [tree, os.path.join(tree, "tools")]
    import pytensor
    if not pytensor.config.blas__ldflags:
        pytensor.config.blas__ldflags = BLAS.split("=", 1)[1]
    if os.path.isdir(os.path.join(tree, "benchmarks")):
        from benchmarks.models import build_model
    else:
        from r1_bias import build_model
    return build_model


def _shaped_prior(mu, sd):
    """A shaped prior with the literals ``mu`` and ``sd``: they are narrower than floatX, so the widening remakes the graph."""
    import numpy as np
    import pymc as pm
    with pm.Model() as m:
        x = pm.Normal("x", mu, sd, shape=3)
        pm.Normal("obs", x.sum(), 1.9, observed=np.linspace(-1, 1, 10))
    return m


def _fence_model():
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    with pm.Model() as model:
        x = pm.Normal("x", shape=(2,))
        pm.Potential("fence", pt.switch(pt.gt(pt.abs(x[0]), 5.0) | pt.lt(pt.abs(x[1]), 0.01), -np.inf, 0.0))
    return model


def _ball_model():
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    with pm.Model() as model:
        x = pm.Normal("x", shape=(2,))
        pm.Potential("ball", pt.switch(pt.sum(x ** 2) > 1.0, -np.inf, 0.0))
    return model


# case -> (floatX, calls); a call is (model name or builder key, seed, keyword arguments of sample_pf_chees_hmc beyond the common ones)
def _spec():
    pf = lambda **k: {"pathfinder_kwargs": k}
    cases = {name: ("float64", [(name, 1, {})]) for name in LADDER}
    cases.update({
        # 1 < pm_chains < chains (ragged cycle), and pm_chains = chains; GEV's shared start sits on xi = 0: the start repair runs
        "gev_pm_chains_3": ("float64", [("gev", 2, pf(pm_chains=3, pm_init="jitter+adapt_diag"))]),
        "gev_pm_chains_C": ("float64", [("gev", 2, pf(pm_chains=CHAINS, pm_init="jitter+adapt_diag"))]),
        "gev_adapt_diag": ("float64", [("gev", 1, pf(init_source="adapt_diag"))]),
        # the widening: a rebuilt model finds the first one's lift in the memo; shaped literals of float32 and int8 are widened
        "8schools_rebuilt": ("float64", [("8schools", 3, {}), ("8schools", 4, {})]),
        "shaped_literals": ("float64", [("shaped:0.5,3.0", 1, {}), ("shaped:0,10", 2, {})]),
        # several warmup chunks, several sampling chunks, a padded last transform chunk
        "8schools_chunked": ("float64", [("8schools", 1, {"_chunks": (7, 3, 4)})]),
        "8schools_warmup_stop_fires": ("float64", [("8schools", 1, {"chees_kwargs": {
            "warmup_stop": {"block": 10, "window": 10, "rhat": 1e9, "eps_stability": 1e9}}})]),
        "8schools_imm_source": ("float64", [("8schools", 1, pf(imm_source=s)) for s in ("mc", "lbfgs_lowrank", "fisher_lowrank", "pf_closed")]),
        "8schools_pf_pool_cap": ("float64", [("8schools", 1, pf(pf_pool_cap=64, draws=100))]),
        # float32: `tests/float32_run.py` in a child (seed 3, both init sources); compared in `tests/test_float32.py`
        **{f"float32_{name}": ("float32", None) for name in LADDER},
        # the start repair on its own: a backtrack, a restart, the search from the origin, and GEV's rows on the singularity
        "start_repair": ("float64", None),
    })
    return cases


CASES = tuple(_spec())


def _model(build_model, label):
    if label.startswith("shaped:"):
        mu, sd = (float(v) if "." in v else int(v) for v in label[7:].split(","))
        return _shaped_prior(mu, sd)
    return build_model(label)


def _collect(idata, prefix, out):
    for part, tag in ((idata.posterior, "posterior"), (idata.sample_stats, "stats")):
        for k, v in part.data_vars.items():
            out[f"{prefix}{tag}/{k}"] = v.values
    for k, v in idata.attrs.items():
        if k != "config":
            out[f"{prefix}attrs/{k}"] = v
    out[f"{prefix}config"] = idata.attrs["config"]


def _start_repair(build_model, out):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pfcheeshmc.sampler import PfCheesHMC

    key = jax.random.split(jax.random.key(1), 5)[0]            # what `_run_pathfinder` hands over

    def repair(name, sampler, stack, **kw):
        out[f"{name}/input"] = np.asarray(jnp.concatenate([a.reshape(a.shape[0], -1) for a in stack.values()], axis=1))
        fixed = sampler._ensure_finite_inits(stack, key, **kw)
        out[f"{name}/repaired"] = np.asarray(jnp.concatenate([a.reshape(a.shape[0], -1) for a in fixed.values()], axis=1))

    s = PfCheesHMC(build_model("gev"), seed=1)
    one = s._pymc_adapt_diag_start(1, init="adapt_diag", tune=500, random_seed=1)
    repair("gev", s, one)
    four = s._pymc_adapt_diag_start(4, init="jitter+adapt_diag", tune=500, random_seed=1)
    on_the_singularity = np.isin(np.arange(4), (1, 3))               # rows 1 and 3 put back on xi = 0
    repair("gev_c4_bad", s, {n: jnp.asarray(np.where(on_the_singularity.reshape((-1,) + (1,) * (a.ndim - 1)),
                                                     np.asarray(one[n]), np.asarray(a))) for n, a in four.items()})
    repair("fence", PfCheesHMC(_fence_model(), seed=1), {"x": jnp.array([[12.0, 1.0], [0.5, 0.7], [1.0, 0.0], [1.0, 0.0]])})
    repair("last_resort", PfCheesHMC(_ball_model(), seed=1), {"x": jnp.array([[50.0, 50.0]])}, max_backtracks=0, max_restarts=0)
    # the starts a pm_chains = 3 Pathfinder stage hands to the chains (GEV: the shared start is repaired first)
    starts, _ = PfCheesHMC(build_model("gev"), seed=3)._run_pathfinder(
        chains=CHAINS, draws=40, seed=3, pm_chains=3, pm_init="jitter+adapt_diag")
    for n, a in starts.items():
        out[f"pf_starts/{n}"] = np.asarray(a)


def compute(case, tree=REPO):
    """The arrays of one case, run in this process (whose floatX is what the case needs)."""
    build_model = _tree_modules(tree)
    import numpy as np
    import pytensor
    from pfcheeshmc import compiled, sample_pf_chees_hmc

    floatx, calls = _spec()[case]
    assert floatx == "float64", f"{case} is a float32 case: `compute_float32` runs it in a child"
    assert pytensor.config.floatX == floatx, f"{case} needs floatX={floatx}, this process has {pytensor.config.floatX}"
    out = {}
    if calls is None:
        _start_repair(build_model, out)
        return {k: np.asarray(v) for k, v in out.items()}
    saved = (compiled.WARMUP_CHUNK, compiled.SAMPLE_CHUNK, compiled.TRANSFORM_CHUNK)
    try:
        for i, (label, seed, extra) in enumerate(calls):
            extra = {k: dict(v) for k, v in extra.items() if k != "_chunks"}
            if "_chunks" in calls[i][2]:
                compiled.WARMUP_CHUNK, compiled.SAMPLE_CHUNK, compiled.TRANSFORM_CHUNK = calls[i][2]["_chunks"]
            extra.setdefault("chees_kwargs", {}).setdefault("draws", WARMUP)
            idata = sample_pf_chees_hmc(
                _model(build_model, label), draws=DRAWS, chains=CHAINS, random_seed=seed, progressbar=False,
                keep_untransformed=True, **extra)
            _collect(idata, f"call{i}/" if len(calls) > 1 else "", out)
    finally:
        compiled.WARMUP_CHUNK, compiled.SAMPLE_CHUNK, compiled.TRANSFORM_CHUNK = saved
    return {k: np.asarray(v) for k, v in out.items()}


def compute_float32(case, tree=REPO):
    """A float32 case: `float32_run.py` in a fresh process with ``floatX=float32`` in ``PYTENSOR_FLAGS`` (and a CPU-only
    environment); its arrays without ``meta``."""
    import tempfile
    import numpy as np
    name = case[len("float32_"):]
    flags = [f for f in os.environ.get("PYTENSOR_FLAGS", "").split(",") if f and not f.startswith("floatX=")]
    if not any(f.startswith("blas__ldflags=") for f in flags):
        flags.append(BLAS)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2", PFCHEESHMC_COMPILATION_CACHE="0",
               PYTENSOR_FLAGS=",".join([*flags, "floatX=float32"]), PYTHONPATH=str(tree))
    env.pop("JAX_PLATFORM_NAME", None)
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "out.npz"
        done = subprocess.run([sys.executable, str(HERE / "float32_run.py"), str(tree), name, str(out),
                               "--init-sources", "pf,adapt_diag"], env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(done.stderr[-3000:])
        z = np.load(out)
        return {k: z[k] for k in z.files if k != "meta"}


def _same(want, got):
    import numpy as np
    assert sorted(want) == sorted(got), (sorted(set(want) ^ set(got)))
    different = []
    for key in sorted(want):
        a, b = want[key], got[key]
        if key.endswith("config"):
            ok = json.loads(str(a)) == json.loads(str(b))
        else:
            ok = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        if not ok:
            different.append(key)
    return different


def assert_golden(case, got):
    """`got` (the arrays of a run of `case`) is the golden file of `case`, to the last bit."""
    import numpy as np
    z = np.load(DATA / f"golden_{case}.npz")
    want = {k: z[k] for k in z.files}
    different = _same(want, got)
    assert not different, (
        f"{len(different)} of {len(want)} arrays of {case} differ from tests/data/golden_{case}.npz: {different[:6]}. "
        "If the change is meant to alter the draws, regenerate with `python tests/make_golden.py` (all cases, or name them) "
        "and say in the commit message why the draws change; if not, the change is a bug.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--tree", default=str(REPO))
    ap.add_argument("--out-dir", default=str(DATA))
    args = ap.parse_args()
    import numpy as np
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = args.cases or list(CASES)
    for case in wanted:
        arrays = compute_float32(case, args.tree) if _spec()[case][0] == "float32" else compute(case, args.tree)
        np.savez_compressed(out_dir / f"golden_{case}.npz", **arrays)
        print(case, len(arrays), "arrays", flush=True)


if __name__ == "__main__":
    main()
