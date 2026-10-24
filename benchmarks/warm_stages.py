#!/usr/bin/env python
"""Per-stage wall time, XLA compiles and PyTensor functions of repeated `sample_pf_chees_hmc` calls (CPU or GPU).

    python benchmarks/warm_stages.py TREE MODEL CHAINS DRAWS WARMUP [--calls 3] [--json OUT]
                                [--progressbar] [--chees JSON] [--pathfinder JSON] [--cache-dir DIR] [--new-data S]

TREE is the source tree whose `pfcheeshmc/` is measured: this checkout (`.`), or a `git archive` of an earlier commit. The
models and the counters are always this checkout's `benchmarks/`. MODEL is a ladder name, or `hier`: the seeded
8-Schools-shaped model of `benchmarks/seeded_models.py`, rebuilt with NEW data and prior values on every call. Call 1 is
cold; calls 2.. are warm, with a new seed each. `--new-data S` gives each ladder call new observed data at the same shapes
(the "new data every run" workflow: `seeded_models.with_new_data`).

The on-disk compilation cache is OFF unless asked, so that call 1 is cold: the library switches a default cache
on in `~/.cache/pfcheeshmc/jax` at its first call, and a "cold" call that read it would be a warm one. The
tool sets `PFCHEESHMC_COMPILATION_CACHE=0` unless the environment already has a value. Asking for it:
  `--cache-dir DIR`                     `pfcheeshmc.enable_compilation_cache(DIR)` (TREE must have it): run the same
                                        command twice for call 1 with the cache cold, then warm, in a fresh process
                                        each time;
  `PFCHEESHMC_COMPILATION_CACHE=1`      the library's default cache (set HOME or XDG_CACHE_HOME to a scratch
                                        directory to start it empty);
  `JAX_COMPILATION_CACHE_DIR` (+ the two thresholds)  JAX's own settings, which the library then leaves alone.
The cache in use is printed in the header line; with one, each call reports its hits and writes.

Timing is `time.perf_counter` around the PfCheesHMC stage methods, blocking on their output. Stages:
  init        _pymc_adapt_diag_start + _ensure_finite_inits
  pathfinder  _run_pathfinder minus init
  eps0        _run_step_size_tune
  warmup      _run_chees (loop programs included)
  sampling    _run_dynamic_hmc (loop program included)
  transform   _transform_value_samples_to_problem_space
Compiles are counted with `benchmarks/compile_counter.py` over each stage and the whole call: XLA programs (JAX's
backend-compile event; the program names of a warm call are printed) and PyTensor functions (each is a
`FunctionMaker`: rewrites plus the C backend, e.g. `pm.init_nuts`). Cells are `seconds / XLA / PyTensor`.
"""
import argparse
import collections
import json
import os
import sys
import time

# This repo's path has a space ("Python packages"): PyTensor puts the environment prefix into blas__ldflags as an unquoted
# -Wl,-rpath token and its linker dies on it for any model that compiles a Dot.
_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "blas__ldflags" not in _flags:
    os.environ["PYTENSOR_FLAGS"] = ",".join(f for f in (_flags, "blas__ldflags=-llapack -lblas -lcblas -lm") if f)
os.environ.setdefault("PFCHEESHMC_COMPILATION_CACHE", "0")          # cold means cold: see the docstring

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("model")
ap.add_argument("chains", type=int)
ap.add_argument("draws", type=int)
ap.add_argument("warmup", type=int)
ap.add_argument("--calls", type=int, default=3)
ap.add_argument("--json")
ap.add_argument("--progressbar", action="store_true")
ap.add_argument("--chees", default="{}", help="extra chees_kwargs, JSON")
ap.add_argument("--pathfinder", default="{}", help="pathfinder_kwargs, JSON")
ap.add_argument("--cache-dir", help="enable the on-disk compilation cache in this directory")
ap.add_argument("--new-data", type=int, default=None, metavar="S",
                help="call k samples the model with new observed data at the same shapes (permutation seed S+k); "
                     "every call is a clone, so all share one structure (seeded_models.with_new_data)")
args = ap.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import pytensor  # noqa: E402
from benchmarks.compile_counter import count_compiles, count_pytensor_functions  # noqa: E402
from benchmarks.models import build_model  # noqa: E402
from benchmarks.seeded_models import hier, with_new_data  # noqa: E402

sys.path.insert(0, args.tree)                    # pfcheeshmc/ of the tree under measurement, imported from here on
import pfcheeshmc.sampler as sm  # noqa: E402

CACHE_EVENTS = collections.Counter()   # a cache hit is still a "backend compile" event: these tell them apart
import jax.monitoring  # noqa: E402
jax.monitoring.register_event_listener(lambda event, **kw: CACHE_EVENTS.update([event.rsplit("/", 1)[-1]]))
if args.cache_dir:
    import pfcheeshmc
    pfcheeshmc.enable_compilation_cache(args.cache_dir)
CACHE_OFF = os.environ.get("PFCHEESHMC_COMPILATION_CACHE", "").strip().lower() in ("0", "false", "no", "off")
print(f"floatX: {pytensor.config.floatX}  jax_enable_x64: {jax.config.jax_enable_x64}", flush=True)
print("disk cache:", args.cache_dir and f"--cache-dir {args.cache_dir}"
      or (jax.config.jax_compilation_cache_dir and f"JAX_COMPILATION_CACHE_DIR={jax.config.jax_compilation_cache_dir} "
          "(your environment: calls are not cold)")
      or ("off" if CACHE_OFF else "the library's default (~/.cache/pfcheeshmc/jax, or XDG_CACHE_HOME), from call 1"),
      flush=True)

METHODS = {
    "_pymc_adapt_diag_start": "init", "_ensure_finite_inits": "init2", "_run_pathfinder": "pathfinder+init",
    "_run_step_size_tune": "eps0", "_run_chees": "warmup", "_run_dynamic_hmc": "sampling",
    "_transform_value_samples_to_problem_space": "transform",
}
REC = collections.defaultdict(lambda: {"s": 0.0, "n": 0, "pt": 0})


def instrument(method, label):
    original = getattr(sm.PfCheesHMC, method)

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        with count_compiles() as cc, count_pytensor_functions() as pf:
            out = original(self, *a, **k)
            try:
                jax.block_until_ready(out[0] if isinstance(out, tuple) else out)
            except Exception:
                pass
        REC[label]["s"] += time.perf_counter() - t0
        REC[label]["n"] += cc.count
        REC[label]["pt"] += pf.count
        return out

    setattr(sm.PfCheesHMC, method, timed)


for method, label in METHODS.items():
    instrument(method, label)


def build(call):
    if args.model == "hier":
        return hier(100 + call, 200 + call)
    model = build_model(args.model)
    if args.new_data is None:
        return model
    return with_new_data(model, args.new_data + call)


results = []
for call in range(1, args.calls + 1):
    REC.clear()
    model = build(call)
    t0 = time.perf_counter()
    with count_compiles() as total, count_pytensor_functions() as total_pt:
        idata = sm.sample_pf_chees_hmc(
            model, draws=args.draws, chains=args.chains, random_seed=call, progressbar=args.progressbar,
            chees_kwargs={"draws": args.warmup, **json.loads(args.chees)},
            pathfinder_kwargs=json.loads(args.pathfinder))
        jax.block_until_ready(next(iter(idata.posterior.values())).values)
    wall = time.perf_counter() - t0
    init = tuple(REC["init"][f] + REC["init2"][f] for f in ("s", "n", "pt"))
    stage = {
        "init": init,
        "pathfinder": tuple(REC["pathfinder+init"][f] - i for f, i in zip(("s", "n", "pt"), init)),
        **{k: (REC[k]["s"], REC[k]["n"], REC[k]["pt"]) for k in ("eps0", "warmup", "sampling", "transform")},
    }
    stage["rest"] = (wall - sum(v[0] for v in stage.values()), total.count - sum(v[1] for v in stage.values()),
                     total_pt.count - sum(v[2] for v in stage.values()))
    results.append({"call": call, "wall": wall, "compiles": total.count, "compile_s": total.seconds,
                    "pytensor_functions": total_pt.count, "stage": stage, "names": sorted(total.names),
                    "floatX": pytensor.config.floatX})
    cache = (f"  disk cache: {CACHE_EVENTS['cache_hits']} hits, {CACHE_EVENTS['cache_misses']} written"
             if jax.config.jax_compilation_cache_dir else "")
    CACHE_EVENTS.clear()
    print(f"call {call}: wall {wall:6.2f}s  compiles {total.count} ({total.seconds:.2f}s of compile)  "
          f"pytensor functions {total_pt.count}{cache}")
    for k, (s, n, p) in stage.items():
        print(f"    {k:11s} {s:7.2f}s  compiles {n}  pytensor {p}")
    if call > 1:
        print("    programs:", sorted(total.names))
if args.json:
    with open(args.json, "w") as f:
        json.dump(results, f, indent=1)
