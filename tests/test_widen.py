"""F2i: `lifted._widen_model_inputs` remakes only what a widened constant reaches, and folds once, and only what can match.

On a model rebuilt around new data or prior values, the widening was the largest lift cost that is ours (`clone_replace`
cloned the whole graph twice and `constant_fold` ran once per parameter, at the three sites of a call). The cheaper
widening must give the SAME graph, so the key, the function and the draws cannot change, and nothing about a model's
values may be kept (there is no memo of it: the rebuilt model's own constants are widened each time).

What a wrong answer here would silently corrupt:
  * the logp: a node that is not remade although its input changed dtype, or a constant widened that should not be (or
    the reverse), so another program than the one `clone_replace` made;
  * the compile-once promise: a rebuilt model whose widened graph differs from the first one's (a dtype that depends on
    the literal's value) misses the lift memo and puts the rewrite, the funcify and the lowering back on the call;
  * the draws, which must be bit for bit those of the code before this task (`BASE_SHA`, a `git archive`, in fresh
    processes: `widen_run.py`).
"""
import concurrent.futures
import gc
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import weakref

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from pymc.util import get_default_varnames
from pytensor.graph.basic import equal_computations
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import toposort

# Same workaround as tests/test_lifted.py: PyMC's own C backend cannot link a Dot under a path with a space.
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas -lm"

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "tools"))
import r1_bias as rb  # noqa: E402
from test_lift_memo import point, pymc_logp  # noqa: E402
from test_lifted import EXTRA, LADDER, LITERALS, LITERALS_INT_RV, _literal_models  # noqa: E402

from pfcheeshmc import lifted  # noqa: E402
from pfcheeshmc.lifted import lift_logp  # noqa: E402

BASE_SHA = "44a0556"             # the f2 head before F2i
SITES = ("logp", "initial_point", "transform")


def sites(model):
    """The outputs of the three lifts of a sampler call, as `PfCheesHMC` and `lift_initial_point` make them."""
    return {"logp": [model.logp()], "initial_point": lifted.initial_point_expression(model),
            "transform": list(get_default_varnames(model.unobserved_value_vars, include_transformed=False))}


def models():
    """The ladder, the extra models of test_lifted, and its literal models at every literal pair (the pairs cross the
    autocast boundaries: 0.5 is float32, 0.1 float64, 1000 int16)."""
    cases = {name: (lambda name=name: rb.build_model(name)) for name in LADDER}
    cases.update({f"extra:{name}": (lambda build=build: build(1)) for name, build in EXTRA.items()})
    for kind, build in _literal_models().items():
        for pair in (LITERALS_INT_RV if kind == "poisson" else LITERALS):
            cases[f"literal:{kind}:{pair}"] = lambda build=build, pair=pair: build(*pair)
    return cases


CASES = models()


class Folds:
    """How many times `lifted._widen_model_inputs` called PyMC's `constant_fold`, and on how many expressions."""

    def __init__(self, monkeypatch):
        self.calls = self.expressions = 0
        real = lifted.constant_fold

        def counting(xs, *a, **k):
            self.calls += 1
            self.expressions += len(xs)
            return real(xs, *a, **k)

        monkeypatch.setattr(lifted, "constant_fold", counting)


# ------------------------------------------------------------------------------------ the graph
@pytest.mark.parametrize("name", list(CASES))
def test_widened_graph_is_the_one_clone_replace_makes_at_every_site(name):
    """`_rebuild` against the library's own `clone_replace(..., rebuild_strict=False)` on the same replacements."""
    model = CASES[name]()
    for site, outputs in sites(model).items():
        widened, replaced = lifted._widen_model_inputs(model, outputs)
        want = clone_replace(list(outputs), replace=replaced, rebuild_strict=False) if replaced else list(outputs)
        assert equal_computations(widened, want), (name, site)
        assert [o.type for o in widened] == [o.type for o in want], (name, site)


def below(outputs, keys):
    """The nodes of the graph of ``outputs`` that depend on one of ``keys``."""
    reached, nodes = set(keys), []
    for node in toposort(outputs):
        if any(i in reached for i in node.inputs):
            nodes.append(node)
            reached.update(node.outputs)
    return nodes


@pytest.mark.parametrize("name", ["8schools", "gev", "radon_noncentered", "literal:shaped:(0.5, 3.0)"])
def test_only_the_nodes_below_a_widened_constant_are_remade(name):
    """The nodes that no widened constant reaches are the old graph's own nodes; the ones below one are new, and the
    old graph is as it was (the model's logp is not touched)."""
    model = CASES[name]()
    outputs = [model.logp()]
    before = list(toposort(outputs))
    types_before = [i.type for node in before for i in node.inputs]
    widened, replaced = lifted._widen_model_inputs(model, outputs)
    assert replaced, "nothing was widened: the premise of the test is gone"
    remade = set(below(outputs, replaced))
    after = set(toposort(widened))
    assert remade and len(remade) < len(before), "the test needs a graph with nodes above and below the constants"
    assert all(node in after for node in before if node not in remade), "a node no widened constant reaches was cloned"
    assert not any(node in after for node in remade), "a node below a widened constant was kept"
    assert list(toposort(outputs)) == before and [i.type for node in before for i in node.inputs] == types_before
    assert any(k.dtype != v.dtype for k, v in replaced.items()), "no dtype changed"


# ------------------------------------------------------------------------------------ the fold
def test_the_parameters_are_folded_in_one_call_not_one_each(monkeypatch):
    model = rb.build_model("radon_noncentered")                 # four literal-built parameters
    folds = Folds(monkeypatch)
    lifted._widen_model_inputs(model, [model.logp()])
    assert (folds.calls, folds.expressions) == (1, 4)


def shaped_prior(mu, sd):
    """A shaped prior with the literals ``mu`` and ``sd``; the likelihood's own scale is a float64 literal, so these
    two are the only literals that can be narrower than floatX."""
    with pm.Model() as m:
        x = pm.Normal("x", mu, sd, shape=3)
        pm.Normal("obs", x.sum(), 1.9, observed=np.linspace(-1, 1, 10))
    return m


@pytest.mark.parametrize("model, site, calls", [
    (lambda: CASES["8schools"](), "transform", 0),      # no non-scalar constant in the transform: nothing can match
    (lambda: CASES["8schools"](), "logp", 1),
    (lambda: shaped_prior(0.1, 10.1), "logp", 0),       # float64 literals: not narrower than floatX, nothing to widen
    (lambda: shaped_prior(0.5, 3.0), "logp", 1),        # float32 literals: folded, matched and widened
    (lambda: shaped_prior(0, 10), "logp", 1),           # int8 literals
    (lambda: shaped_prior(0.5, 3.0), "transform", 0),
])
def test_nothing_is_folded_that_cannot_change_the_graph(monkeypatch, model, site, calls):
    model = model()
    folds = Folds(monkeypatch)
    lifted._widen_model_inputs(model, sites(model)[site])
    assert folds.calls == calls


def test_a_folded_parameter_is_matched_where_it_occurs_in_the_graph():
    """Shaped float32 and int8 literals reach the logp as one-element constants of their own, and are widened."""
    for pair, dtype in (((0.5, 3.0), "float32"), ((0, 10), "int8")):
        model = shaped_prior(*pair)
        widened, replaced = lifted._widen_model_inputs(model, [model.logp()])
        ones = [(k, v) for k, v in replaced.items() if k.ndim == 1]
        assert len(ones) == 2 and all(k.dtype == dtype and v.dtype == "float64" for k, v in ones), pair
        assert all(np.array_equal(k.data, v.data) for k, v in ones)
        assert sorted(float(k.data[0]) for k, _ in ones) == sorted(map(float, pair))


# ------------------------------------------------------------------------------------ across a boundary
@pytest.mark.parametrize("kind", ["scalar", "shaped", "poisson", "truncated"])
def test_a_prior_literal_across_an_autocast_boundary_is_a_memo_hit(kind):
    """The literals of one model crossing int8 / float32 / float64 / int16 / int32: the first build makes the function,
    every other literal pair reuses it (no rewrite: the memo's size does not grow), with its own values."""
    pairs = LITERALS_INT_RV if kind == "poisson" else LITERALS
    build = _literal_models()[kind]
    first = lift_logp(build(*pairs[0]))
    filed = sum(len(v) for v in lifted._MEMO.values())
    for pair in pairs[1:] + pairs[:2]:
        model = build(*pair)
        lg = lift_logp(model)
        assert sum(len(v) for v in lifted._MEMO.values()) == filed, f"{pair}: a build, not a hit"
        assert lg.fn is first.fn and lg.key == first.key and lg.rule_used == "model_inputs", pair
        assert [str(v.dtype) for v in lg.values] == [str(v.dtype) for v in first.values], pair
        x = point(model)
        # (-3 is not a Gamma rate: both logps are NaN there, and agree on it)
        np.testing.assert_allclose(float(lg.fn(x, lg.values)[0]), pymc_logp(model, x), rtol=1e-7, err_msg=str(pair))


# ------------------------------------------------------------------------------------ no new failure mode, no state
def test_a_failing_rebuild_falls_back_to_the_native_dtype_rule(monkeypatch):
    def boom(outputs, replace):
        raise TypeError("cannot remake a node")

    monkeypatch.setattr(lifted, "_rebuild", boom)
    model = CASES["literal:shaped:(0.5, 3.0)"]()
    with pytest.warns(RuntimeWarning, match="model_inputs.*cannot remake a node.*model_inputs_native_dtype"):
        lg = lift_logp(model)
    assert lg.rule_used == "model_inputs_native_dtype"
    x = point(model)
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(pymc_logp(model, x), rel=1e-7)


def containers():
    return {k: len(v) for k, v in vars(lifted).items() if isinstance(v, (dict, list, set))}


def test_the_widening_keeps_no_data_or_prior_values():
    """No module-level container grows, and the constants it makes are freed with the graph."""
    def build():
        rng = np.random.default_rng(0)
        with pm.Model() as m:
            mu = pm.Normal("mu", 0.5, 3.0, shape=3)
            pm.Normal("obs", mu.sum(), 1.9, observed=rng.normal(size=12))
        return m

    lifted._widen_model_inputs(build(), [build().logp()])              # whatever a first call sets up
    before = containers()
    model = build()
    widened, replaced = lifted._widen_model_inputs(model, [model.logp()])
    assert replaced and containers() == before
    refs = [weakref.ref(v) for v in replaced.values()]
    del widened, replaced, model
    gc.collect()
    assert all(r() is None for r in refs), "a widened constant is still referenced"


# ------------------------------------------------------------------------------------ against the code before
def _env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2",
               PYTENSOR_FLAGS="blas__ldflags=-llapack -lblas -lcblas -lm", PFCHEESHMC_COMPILATION_CACHE="0")
    env.pop("JAX_PLATFORM_NAME", None)        # set by the example notebooks when another test ran one in-process
    return env


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory):
    git = ["git", "-C", str(REPO)]
    try:
        subprocess.run([*git, "cat-file", "-e", f"{BASE_SHA}^{{commit}}"], check=True, capture_output=True)
        blob = subprocess.run([*git, "archive", BASE_SHA], check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {BASE_SHA} is not available here (git archive failed)")
    dest = tmp_path_factory.mktemp("widen_base") / "base"
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest)
    return dest


def _child(tree, *args):
    done = subprocess.run([sys.executable, str(HERE / "widen_run.py"), args[0], str(tree), *map(str, args[1:])],
                          env=dict(_env(), PYTHONPATH=str(tree)), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_widened_graphs_and_replacements_are_those_of_the_old_code(base_tree, tmp_path):
    """Every model of `CASES` at the three sites, in fresh processes: the same graphs (type, constant and sharing of
    every node) and the same constants replaced, byte for byte, as 44a0556 makes."""
    got = {}
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs = {label: pool.submit(_child, tree, "graphs", tmp_path / f"{label}.json")
                for label, tree in (("old", base_tree), ("new", REPO))}
        for label, job in jobs.items():
            job.result()
            got[label] = json.loads((tmp_path / f"{label}.json").read_text())
    old, new = got["old"], got["new"]
    assert sorted(old) == sorted(new) == sorted(f"{name}|{site}" for name in CASES for site in SITES)
    assert sum(v["n"] for v in old.values()) > 100, "few constants were replaced: the premise of the test is gone"
    for key in old:
        assert new[key]["n"] == old[key]["n"], key
        assert new[key]["replaced"] == old[key]["replaced"], key
        assert new[key]["graph"] == old[key]["graph"], key


@pytest.mark.parametrize("name", ["8schools", "gev"])
def test_draws_of_a_rebuilt_model_are_bit_identical_to_the_old_code(base_tree, tmp_path, name):
    """Two sampler calls in one process, the second on a rebuilt model (every lift of it finds the first one's memo):
    every array of both, in the new tree and in 44a0556. GEV has the folded one-element literals; 8schools the shaped
    prior with int literals."""
    out = {}
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs = {label: pool.submit(_child, tree, "draws", name, tmp_path / f"{label}.npz")
                for label, tree in (("old", base_tree), ("new", REPO))}
        for label, job in jobs.items():
            job.result()
            out[label] = dict(np.load(tmp_path / f"{label}.npz"))
    old, new = out["old"], out["new"]
    assert sorted(old) == sorted(new) and len(old) > 20
    for call in ("first", "second"):
        assert any(k.startswith(f"{call}/posterior/") for k in old) and any(k.startswith(f"{call}/stats/") for k in old)
    assert not all(np.array_equal(old[k], old[k.replace("first", "second")]) for k in old if k.startswith("first/posterior/"))
    for key in old:
        if key.endswith("/config"):
            assert json.loads(str(new[key])) == json.loads(str(old[key])), key
        else:
            assert old[key].dtype == new[key].dtype and np.array_equal(old[key], new[key], equal_nan=True), key
    for key in new:
        if "/posterior/" in key:
            assert np.isfinite(new[key]).all(), key
