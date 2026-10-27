"""Child process of tests/test_pm_chains.py: the XLA programs one cold `_run_pathfinder` call compiles, in any tree.

    python pm_chains_run.py TREE OUT.json CHAINS PATHFINDER_KWARGS_JSON

TREE is a source tree that provides `pfcheeshmc/` and `benchmarks/` or `tools/` (the checkout, or a `git archive` of an
earlier commit). It must not import anything that exists only in the newer tree. The model is the ladder's GEV, whose one
shared start sits on xi = 0 (a NaN gradient), so the start repair runs. Writes ``{"compiles": [program names]}``
to OUT.json: the call runs in a fresh process, so the names are the cold ones.
"""
import json
import os
import sys

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
tree, out, chains, pathfinder = sys.argv[1:5]
sys.path.insert(0, os.path.join(tree, "tools"))
sys.path.insert(0, tree)
# benchmarks/ of THIS checkout (compile_counter), whatever TREE is
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "benchmarks"))

import jax  # noqa: E402
from compile_counter import count_compiles  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402

if os.path.isdir(os.path.join(tree, "benchmarks")):  # the tree decides, not an ImportError: PYTHONPATH can put THIS benchmarks/ on an older tree's path
    from benchmarks.models import build_model
else:                                                    # a tree before benchmarks/: its models are the notebooks', built by tools/r1_bias
    from r1_bias import build_model

sampler = PfCheesHMC(build_model("gev"), seed=3)
with count_compiles() as cc:
    starts, _ = sampler._run_pathfinder(chains=int(chains), draws=40, seed=3, **json.loads(pathfinder))
    jax.block_until_ready(starts)
with open(out, "w") as f:
    json.dump({"compiles": cc.names}, f)
