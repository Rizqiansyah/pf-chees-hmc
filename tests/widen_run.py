"""Child process of tests/test_widen.py: what `lifted._widen_model_inputs` makes, and the draws of a rebuilt model, in any tree.

    python widen_run.py graphs TREE OUT.json
    python widen_run.py draws TREE MODEL OUT.npz

TREE is a source tree that provides `pfcheeshmc/` and `benchmarks/` or `tools/` (the checkout, or a `git archive` of an earlier
commit).
Nothing here may need what only the newer tree has: `graphs` calls `_widen_model_inputs`, `initial_point_expression`
and the lifts only.

graphs
    For the ladder models, the extra models of test_lifted and its literal models at every literal pair (those cross the
    autocast boundaries: 0.5 is float32, 0.1 float64, 1000 int16), at the three sites of a lift (logp, initial point,
    value -> RV transform): the widened outputs as text (``debugprint``, which shows every node, type, constant and the
    sharing of nodes; variable names are left out, they do not enter a computation) and the constants replaced, each
    with its dtype, shape and bytes. Writes the hashes, and the head of the text to read a difference by. Also, for
    the ladder and the extra models: the lift of each site (key, ``n_lifted``, rule, the kept-static report), and
    whether a lift of a fresh build of the model returns the first one's function.
draws
    One sampler call on a model, then a second on a fresh build of it with another seed (the second lift finds the
    first one's memo), the arrays of both: unconstrained draws, sample statistics, ``config``.
"""
import hashlib
import io
import json
import os
import re
import sys

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
mode, tree = sys.argv[1:3]
sys.path.insert(0, os.path.join(tree, "tools"))
sys.path.insert(0, tree)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from pymc.util import get_default_varnames  # noqa: E402
from pytensor.printing import debugprint  # noqa: E402

from pfcheeshmc import lifted  # noqa: E402

if os.path.isdir(os.path.join(tree, "benchmarks")):  # the tree decides, not an ImportError: PYTHONPATH can put THIS benchmarks/ on an older tree's path
    from benchmarks.models import build_model
else:                                                    # a tree before benchmarks/: its models are the notebooks', built by tools/r1_bias
    from r1_bias import build_model


def sites(model):
    return {"logp": [model.logp()], "initial_point": lifted.initial_point_expression(model),
            "transform": list(get_default_varnames(model.unobserved_value_vars, include_transformed=False))}


def digest(text):
    text = re.sub(r" '[^'\n]*'$", "", text, flags=re.M)         # variable names: `clone_replace` drops some, `_rebuild` keeps them
    return {"sha": hashlib.sha256(text.encode()).hexdigest(), "n": len(text), "head": text[:600]}


def lift_sites(model):
    from pfcheeshmc.sampler import PfCheesHMC
    sampler = PfCheesHMC(model)
    transform, _ = sampler._lift_value_to_rv()
    return {"logp": sampler._lifted, "initial_point": lifted.lift_initial_point(model), "transform": transform}


def lifts(make_first, make_again):
    first, again = lift_sites(make_first()), lift_sites(make_again())
    return {site: {"key": lg.key, "n_lifted": lg.n_lifted, "rule": lg.rule_used,
                   "kept_static": [list(map(str, k)) for k in lg.kept_static],
                   "again_same_fn": again[site].fn is lg.fn, "again_same_key": again[site].key == lg.key}
            for site, lg in first.items()}


def graphs(out):
    import test_lifted as tl
    models = {name: (lambda name=name: build_model(name)) for name in tl.LADDER}
    models.update({f"extra:{name}": (lambda build=build: build(1)) for name, build in tl.EXTRA.items()})
    for kind, build in tl._literal_models().items():
        for pair in (tl.LITERALS_INT_RV if kind == "poisson" else tl.LITERALS):
            models[f"literal:{kind}:{pair}"] = lambda build=build, pair=pair: build(*pair)
    result = {}
    for name, make in models.items():
        model = make()
        for site, outputs in sites(model).items():
            widened, replaced = lifted._widen_model_inputs(model, outputs)
            text = io.StringIO()
            debugprint(widened, file=text, print_type=True)
            constants = "\n".join(
                f"{old.dtype} {old.data.shape} {np.asarray(old.data).tobytes().hex()} -> "
                f"{new.dtype} {new.data.shape} {np.asarray(new.data).tobytes().hex()}" for old, new in replaced.items())
            result[f"{name}|{site}"] = {"graph": digest(text.getvalue()), "replaced": digest(constants), "n": len(replaced)}
    for name in tl.LADDER:
        result[f"lift:{name}"] = lifts(lambda name=name: build_model(name), lambda name=name: build_model(name))
    for name, build in tl.EXTRA.items():
        result[f"lift:extra:{name}"] = lifts(lambda build=build: build(1), lambda build=build: build(2))
    with open(out, "w") as f:
        json.dump(result, f)


def draws(name, out):
    from pfcheeshmc import sample_pf_chees_hmc
    arrays = {}
    for call, seed in (("first", 3), ("second", 4)):
        idata = sample_pf_chees_hmc(
            build_model(name), draws=10, chains=8, random_seed=seed, progressbar=False,
            keep_untransformed=True, chees_kwargs={"draws": 20})
        arrays.update({f"{call}/posterior/{k}": np.asarray(v.values) for k, v in idata.posterior.data_vars.items()})
        arrays.update({f"{call}/stats/{k}": np.asarray(v.values) for k, v in idata.sample_stats.data_vars.items()})
        arrays[f"{call}/config"] = np.array(idata.attrs["config"])
    np.savez(out, **arrays)


if mode == "graphs":
    graphs(sys.argv[3])
else:
    draws(sys.argv[3], sys.argv[4])
