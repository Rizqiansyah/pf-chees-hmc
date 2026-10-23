"""F2e-b: the lift memo -- `lifted._build` reuses the function made for a structurally equal graph (CPU).

What a wrong answer here would silently corrupt:
  * the logp: a memo hit hands back a function made for ANOTHER graph (a hash collision, an equality that
    ignores a constant or the position of a lifted input), or the values of the FIRST model;
  * the compile-once promise: a miss where there should be a hit puts the PyTensor rewrite, the funcify and
    the lowering (0.14-0.46 s per call, the largest fixed cost after the loops on small models) back on every
    call with a rebuilt model;
  * the keys: the lowered text depends on the shapes of the arguments, which the graph does not fix (a
    `pm.Data` has an unknown static size).
The rewrite is counted by putting a counting stand-in for `pytensor.compile.mode` in `lifted`, which uses
nothing of it but `mode.JAX.optimizer.rewrite`. Every test starts from an empty memo.

Resource note: every XLA executable a process holds costs it dozens of memory mappings, and Linux allows
65,530 per process (`vm.max_map_count`). Here the functions are evaluated under one `jax.jit` each rather than
op by op, and only the sampler tests compile whole sets of programs. The compiled programs are released when
the module is done, by `tests/conftest.py` for every module.
"""
import pathlib
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.variable import TensorConstant

# Same workaround as tests/test_lifted.py: PyMC's own C backend cannot link a Dot under a path with a space.
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas -lm"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import r1_bias as rb  # noqa: E402
from seeded_models import hier  # noqa: E402
from test_lifted import EXTRA  # noqa: E402

from pfcheeshmc import lifted  # noqa: E402
from pfcheeshmc.lifted import lift_graph, lift_initial_point, lift_logp  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC, sample_pf_chees_hmc  # noqa: E402

LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")
SITES = ("logp", "initial_point", "transform")


# ------------------------------------------------------------------------------------ helpers
@pytest.fixture(autouse=True)
def fresh_memo(monkeypatch):
    monkeypatch.setattr(lifted, "_MEMO", {})


class Counts:
    """How many times `lifted._build` ran the PyTensor JAX rewrite, the funcify and the lowering."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rewrite = self.funcify = self.lowering = 0


@pytest.fixture
def counts(monkeypatch):
    c = Counts()
    real_mode, real_funcify, real_key = lifted.mode, lifted.jax_funcify, lifted._key

    class Optimizer:
        def rewrite(self, fgraph, *a, **k):
            c.rewrite += 1
            return real_mode.JAX.optimizer.rewrite(fgraph, *a, **k)

    def funcify(*a, **k):
        c.funcify += 1
        return real_funcify(*a, **k)

    def key(*a, **k):
        c.lowering += 1
        return real_key(*a, **k)

    monkeypatch.setattr(lifted, "mode", types.SimpleNamespace(JAX=types.SimpleNamespace(optimizer=Optimizer())))
    monkeypatch.setattr(lifted, "jax_funcify", funcify)
    monkeypatch.setattr(lifted, "_key", key)
    return c


def lift_sites(model):
    """The three lifts of one sampler call, through the sampler's own code: `PfCheesHMC.__init__` (logp), the
    initial point and the value -> problem-space transform. {site: LiftedGraph}."""
    sampler = PfCheesHMC(model)
    transform, _ = sampler._lift_value_to_rv()
    return {"logp": sampler._lifted, "initial_point": lift_initial_point(model), "transform": transform}


def point(model, seed=0):
    ip = model.initial_point()
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(ip[v.name] + 0.3 * rng.standard_normal(np.shape(ip[v.name]))) for v in model.value_vars)


def site_args(model, site):
    return () if site == "initial_point" else point(model)


def run(fn, args, values):
    """``fn(args, values)`` as ONE jitted program (op by op, every primitive would be an executable of its own)."""
    return [np.asarray(o) for o in jax.jit(fn)(args, values)]


def evaluate(lg, model, site):
    return run(lg.fn, site_args(model, site), lg.values)


def build_without_memo(model, site, monkeypatch):
    """The function and values `_build` makes with the memo off (bypassing the registry too), for `site`."""
    inputs = [] if site == "initial_point" else list(model.value_vars)
    if site == "logp":
        outputs = [model.logp()]
    elif site == "initial_point":
        outputs = lifted.initial_point_expression(model)
    else:
        from pymc.util import get_default_varnames
        outputs = list(get_default_varnames(model.unobserved_value_vars, include_transformed=False))
    with monkeypatch.context() as mp:
        mp.setattr(lifted, "_recall", lambda inputs, outputs: (None, None))
        fn, values, *_ = lifted._build(model, inputs, outputs, "model_inputs")
    return fn, values


def pymc_logp(model, x):
    return float(model.compile_logp()({v.name: np.asarray(a) for v, a in zip(model.value_vars, x)}))


def perturb_values(model):
    """New data and prior values at the same shapes, in place: x * 1.01 + 0.003 for every float the lift rule
    makes a runtime input (a direct input of a random variable, an observed array, any array constant, a
    pm.Data). Integers (counts, indices) and the scalar constants of the model's own expressions stay: those
    are static by the rule (F2-2), and another value of one is another program. Returns the number of
    (arrays, scalars) changed. Call it on a fresh model, before its first logp."""
    inputs = set()
    for rv in model.basic_RVs:
        node = rv.owner
        inputs.update(i for i in (node.op.dist_params(node) if hasattr(node.op, "dist_params") else node.inputs)
                      if isinstance(i, TensorConstant))
    inputs.update(v for v in model.rvs_to_values.values() if isinstance(v, TensorConstant))
    roots = [*model.basic_RVs, *model.potentials, *model.deterministics, *model.rvs_to_values.values()]
    arrays = scalars = 0
    for var in ancestors(roots):
        if isinstance(var, SharedVariable) and not isinstance(var.type, RandomType):
            value = np.asarray(var.get_value())
            if value.dtype.kind == "f":
                var.set_value(value * 1.01 + 0.003)
                arrays += 1
        elif isinstance(var, TensorConstant) and (var in inputs or var.ndim > 0):
            value = np.asarray(var.data)
            if value.dtype.kind == "f":
                var.data = (value * 1.01 + 0.003).astype(value.dtype)
                arrays += value.ndim > 0
                scalars += value.ndim == 0
    return arrays, scalars


# ------------------------------------------------------------------------------------ hit
@pytest.mark.parametrize("name", LADDER)
def test_ladder_model_rebuilt_with_new_values_makes_no_rewrite_at_any_site(name, counts, monkeypatch):
    first_model = rb.build_model(name)
    first = lift_sites(first_model)
    assert counts.rewrite == counts.funcify >= 3 and counts.lowering >= 3     # the counters see a real build
    memos = sum(len(v) for v in lifted._MEMO.values())

    second_model = rb.build_model(name)
    arrays, scalars = perturb_values(second_model)
    assert arrays + scalars > 0, "nothing was perturbed: the premise of the test is gone"
    counts.reset()
    second = lift_sites(second_model)

    assert (counts.rewrite, counts.funcify, counts.lowering) == (0, 0, 0)
    assert sum(len(v) for v in lifted._MEMO.values()) == memos
    for site in SITES:
        assert second[site].fn is first[site].fn, site
        assert second[site].key == first[site].key, site
        assert second[site].rule_used == first[site].rule_used == "model_inputs", site
    # the values are the second model's own, not the first's...
    assert any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(first["logp"].values, second["logp"].values))
    # ...and the reused function computes the second model on them, bit for bit what a build for it computes
    for site in SITES:
        fn, values = build_without_memo(second_model, site, monkeypatch)
        want = run(fn, site_args(second_model, site), values)
        got = evaluate(second[site], second_model, site)
        assert len(got) == len(want) > 0
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w, err_msg=site)
    # and the logp is PyMC's of the second model (float32 constant folding of PyMC's own graph: 1e-7)
    x = point(second_model)
    assert float(second["logp"].fn(x, second["logp"].values)[0]) == pytest.approx(pymc_logp(second_model, x), rel=1e-7)


@pytest.mark.parametrize("name", sorted(EXTRA))
def test_extra_model_rebuilt_with_new_data_and_priors_makes_no_rewrite(name, counts):
    """The universality models of test_lifted: pm.Data, int-written priors, an MvNormal, a Potential, a Poisson
    GLM, a truncated Normal. Seeds 1 and 2 give other data AND other prior values at the same shapes."""
    first = lift_sites(EXTRA[name](1))
    counts.reset()
    second_model = EXTRA[name](2)
    second = lift_sites(second_model)
    assert (counts.rewrite, counts.funcify, counts.lowering) == (0, 0, 0), name
    for site in SITES:
        assert second[site].fn is first[site].fn and second[site].key == first[site].key, site
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(first["logp"].values, second["logp"].values))
    x = point(second_model)
    assert float(second["logp"].fn(x, second["logp"].values)[0]) == pytest.approx(pymc_logp(second_model, x), rel=1e-7)


def test_a_hit_still_reads_everything_that_is_the_models_from_the_current_model(counts):
    """`values`, `sources`, `kept_static` and `static_shared` are the current model's, not the memo's."""
    def build(seed):
        rng = np.random.default_rng(seed)
        with pm.Model() as m:
            k = pm.Data("k", np.array(3))                                  # a shape: kept static
            w = pm.Data("w", rng.uniform(0.5, 2.0, size=3))
            x = pm.Normal("x", float(rng.uniform(-1, 1)), 1.3, shape=3)
            pm.Potential("p", -pt.sum(w * x**2) + pt.sum(pt.ones(k)))
        return m

    a, b = build(1), build(2)
    la = lift_logp(a)
    counts.reset()
    lb = lift_logp(b)
    assert counts.rewrite == 0 and lb.fn is la.fn
    assert lb.kept_static == la.kept_static and lb.kept_static, "the static-shared case is not exercised"
    assert len(lb.static_shared) == len(la.static_shared) == 1
    assert lb.static_shared[0][0] is b["k"] and la.static_shared[0][0] is a["k"]
    assert [s for s in lb.sources if isinstance(s, SharedVariable)] == [b["w"]]
    assert not lb.is_stale()
    b["k"].set_value(np.array(4, dtype=b["k"].dtype))
    assert lb.is_stale()                                                     # the check is the current model's


# ------------------------------------------------------------------------------------ miss
def _logp_models():
    def with_extra_term(seed):
        rng = np.random.default_rng(seed)
        with pm.Model() as m:
            mu = pm.Normal("mu", 0.3, 1.7)
            pm.Normal("obs", mu, 1.9, observed=rng.normal(size=12))
            pm.Potential("p", -pt.sum(mu**2))
        return m

    def plain(seed):
        rng = np.random.default_rng(seed)
        with pm.Model() as m:
            mu = pm.Normal("mu", 0.3, 1.7)
            pm.Normal("obs", mu, 1.9, observed=rng.normal(size=12))
        return m

    def other_size(seed):
        rng = np.random.default_rng(seed)
        with pm.Model() as m:
            mu = pm.Normal("mu", 0.3, 1.7)
            pm.Normal("obs", mu, 1.9, observed=rng.normal(size=13))
        return m

    def literal(power):
        def build(seed):
            rng = np.random.default_rng(seed)
            with pm.Model() as m:
                mu = pm.Normal("mu", 0.3, 1.7)
                pm.Normal("obs", mu, 1.9, observed=rng.normal(size=12))
                pm.Potential("p", -pt.abs(mu) ** power)       # a scalar constant of the model's own expression: static
            return m
        return build

    return {"extra term": (plain, with_extra_term), "data size": (plain, other_size),
            "static exponent": (literal(3), literal(4))}


@pytest.mark.parametrize("what", sorted(_logp_models()))
def test_another_structure_shape_or_static_constant_does_not_hit(what, counts):
    a, b = _logp_models()[what]
    la = lift_logp(a(0))
    counts.reset()
    model_b = b(0)
    lb = lift_logp(model_b)
    assert counts.rewrite == 1 and counts.funcify == 1, "this must be a build"
    assert lb.fn is not la.fn and lb.key != la.key
    x = point(model_b)
    assert float(lb.fn(x, lb.values)[0]) == pytest.approx(pymc_logp(model_b, x), rel=1e-7)
    counts.reset()
    lift_logp(b(5))                                                          # and b itself is a hit now
    assert counts.rewrite == 0


def test_a_static_shared_value_is_part_of_the_signature():
    """A pm.Data that sets a shape is baked in as a constant: another value is another program."""
    def build(n):
        with pm.Model() as m:
            k = pm.Data("k", np.array(n))
            x = pm.Normal("x", 0.3, 1.7, shape=3)
            pm.Potential("p", -pt.sum(x**2) + pt.sum(pt.ones(k)))
        return m

    m = build(3)
    la = lift_logp(m)
    assert la.static_shared
    m["k"].set_value(np.array(4, dtype=m["k"].dtype))
    assert la.is_stale()
    lb = lift_logp(m)
    assert lb.fn is not la.fn and lb.key != la.key and not lb.is_stale()
    m["k"].set_value(np.array(3, dtype=m["k"].dtype))                                            # back: the first entry again
    lc = lift_logp(m)
    assert lc.fn is la.fn and lc.key == la.key


# ------------------------------------------------------------------------------------ the key
def test_key_depends_on_the_argument_shapes_not_only_on_the_graph(counts, monkeypatch):
    """A `pm.Data` has an unknown static size, so models of N=30 and N=31 have the SAME graph (one memo, no
    rewrite) and different lowered programs (other keys). `set_data` to another size does the same."""
    def build(n, seed=0):
        rng = np.random.default_rng(seed)
        with pm.Model() as m:
            y = pm.Data("y", rng.normal(size=n))
            mu = pm.Normal("mu", 0.3, 1.7)
            pm.Normal("obs", mu, 1.9, observed=y)
        return m

    a = build(30)
    la = lift_logp(a)
    counts.reset()
    b = build(31, seed=1)
    lb = lift_logp(b)
    assert (counts.rewrite, counts.funcify) == (0, 0)
    assert counts.lowering == 1, "a shape never lowered before is lowered"
    assert lb.fn is la.fn and lb.key != la.key
    assert sum(len(v) for v in lifted._MEMO.values()) == 1
    # the key is what the memo-free lowering says
    fn, values = build_without_memo(b, "logp", monkeypatch)
    assert lifted._key(fn, lifted._abstract(b, list(b.value_vars)), values) == lb.key
    # a size seen before: nothing to lower
    counts.reset()
    assert lift_logp(build(30, seed=2)).key == la.key and lift_logp(build(31, seed=3)).key == lb.key
    assert (counts.rewrite, counts.funcify, counts.lowering) == (0, 0, 0)
    # set_data to a third size, on a model that already holds a sampler's lift
    sampler = PfCheesHMC(a)
    with a:
        pm.set_data({"y": np.zeros(40)})
    sampler._refresh_values()
    assert sampler._lifted.key not in (la.key, lb.key) and sampler._lifted.fn is la.fn
    assert float(sampler._lifted.fn(point(a), sampler._values)[0]) == pytest.approx(pymc_logp(a, point(a)), rel=1e-7)


def test_unknown_static_shape_of_an_input_is_read_off_the_current_model(counts):
    """A random variable sized by a `pm.Data` has a value variable of unknown static shape, which
    `lifted._abstract` reads off the model: the same graph at J=5 and J=7 is one memo and two keys. (With
    `dims=` a dim length is a shape, hence static, and J=7 is another program, as it must be.)"""
    def build(J, seed):
        rng = np.random.default_rng(seed)
        with pm.Model() as m:
            x = pm.Data("x", rng.normal(size=J))
            y = pm.Data("y", rng.normal(size=J))
            th = pm.Normal("th", 0.3, 1.3, shape=x.shape[0])
            pm.Normal("obs", th + x, 1.9, observed=y)
        return m

    a, b = build(5, 1), build(7, 2)
    la = lift_logp(a)
    assert a.value_vars[0].type.shape == (None,)
    counts.reset()
    lb = lift_logp(b)
    assert (counts.rewrite, counts.funcify) == (0, 0) and counts.lowering == 1
    assert lb.fn is la.fn and lb.key != la.key
    x = point(b)
    assert x[0].shape == (7,)
    assert float(lb.fn(x, lb.values)[0]) == pytest.approx(pymc_logp(b, x), rel=1e-7)
    counts.reset()
    assert lift_logp(build(5, 3)).key == la.key and lift_logp(build(7, 4)).key == lb.key
    assert (counts.rewrite, counts.funcify, counts.lowering) == (0, 0, 0)


def test_another_jax_enable_x64_is_another_function(counts):
    """`jax_funcify` makes the constants arrays of the default float width at the time, so a function made
    with x64 on must not serve a call with x64 off (and the reverse)."""
    def build(seed):
        rng = np.random.default_rng(seed)
        with pm.Model() as m:
            mu = pm.Normal("mu", 0.3, 1.7)
            pm.Normal("obs", mu, 1.9, observed=rng.normal(size=12))
        return m

    assert jax.config.jax_enable_x64
    on = lift_logp(build(0))
    counts.reset()
    with jax.enable_x64(False):
        m = build(1)
        off = lift_logp(m)
        assert counts.rewrite == 1 and off.fn is not on.fn and off.key != on.key
        assert all(v.dtype == jnp.float32 for v in off.values)
        x = (jnp.float32(0.7),)
        assert float(off.fn(x, off.values)[0]) == pytest.approx(pymc_logp(m, x), rel=1e-5)
        counts.reset()
        assert lift_logp(build(2)).fn is off.fn and counts.rewrite == 0
    counts.reset()
    assert lift_logp(build(3)).fn is on.fn and counts.rewrite == 0


# ------------------------------------------------------------------------------------ collisions
def test_forced_signature_collision_never_returns_a_wrong_function(counts, monkeypatch):
    """Every graph gets the same signature, so verification (`equal_computations`) alone keeps two different
    models apart: each gets a function that computes its own logp, a rebuild of each finds its own."""
    monkeypatch.setattr(lifted, "_signature", lambda inputs, outputs: 0)
    models = _logp_models()
    pairs = [models["extra term"][0](0), models["extra term"][1](0), models["data size"][1](0), models["static exponent"][1](0)]
    got = []
    for m in pairs:
        lg = lift_logp(m)
        x = point(m)
        assert float(lg.fn(x, lg.values)[0]) == pytest.approx(pymc_logp(m, x), rel=1e-7)
        got.append(lg)
    assert len({id(g.fn) for g in got}) == len(pairs), "different graphs must not share a function"
    assert [len(v) for v in lifted._MEMO.values()] == [len(pairs)], "they are filed in one bucket"
    counts.reset()
    for m_new, lg in zip((models["extra term"][0](9), models["extra term"][1](9), models["data size"][1](9), models["static exponent"][1](9)), got):
        again = lift_logp(m_new)
        x = point(m_new)
        assert again.fn is lg.fn and again.key == lg.key
        assert float(again.fn(x, again.values)[0]) == pytest.approx(pymc_logp(m_new, x), rel=1e-7)
    assert counts.rewrite == 0


def test_a_failing_signature_or_comparison_is_a_miss_not_an_error(counts, monkeypatch):
    m = _logp_models()["extra term"][0](0)
    la = lift_logp(m)

    def boom(*a, **k):
        raise RecursionError("a very deep graph")

    with monkeypatch.context() as mp:
        mp.setattr(lifted, "equal_computations", boom)
        counts.reset()
        lb = lift_logp(_logp_models()["extra term"][0](1))
        assert counts.rewrite == 1 and lb.fn is la.fn, "the build runs; the registry still hands out the first function"
    with monkeypatch.context() as mp:
        mp.setattr(lifted, "_signature", boom)
        counts.reset()
        lc = lift_logp(_logp_models()["extra term"][0](2))
        assert counts.rewrite == 1 and lc.key == la.key


# ------------------------------------------------------------------------------------ order of the values
@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (2, 1, 0), (1, 2, 0), (0, 2, 1), (2, 0, 1)])
def test_data_arrays_in_swapped_roles_each_give_pymcs_logp(order, counts):
    """Three same-shape, same-dtype data arrays in every assignment of roles (x1*a + x2*b + x3*c, x2*a + ...),
    each rebuilt with new values: a function reused with the values in the wrong slots would compute another
    model. Exact (priors are not float32-representable, so PyMC folds no constant in float32)."""
    def build(seed, order):
        rng = np.random.default_rng(seed)
        xs = [rng.normal(size=20) * (1 + i) for i in range(3)]
        with pm.Model() as m:
            a = pm.Normal("a", 0.3, 1.7)
            b = pm.Normal("b", -0.7, 2.3)
            c = pm.Normal("c", 0.9, 1.1)
            mu = xs[order[0]] * a + xs[order[1]] * b + xs[order[2]] * c
            pm.Normal("y", mu, 1.9, observed=rng.normal(size=20))
        return m

    first = build(0, (0, 1, 2))
    lift_logp(first)
    for seed in (1, 2):
        m = build(seed, order)
        lg = lift_logp(m)
        for s in range(3):
            x = point(m, s)
            assert float(lg.fn(x, lg.values)[0]) == pytest.approx(pymc_logp(m, x), rel=1e-12)


def test_value_variables_in_swapped_roles_each_give_pymcs_logp():
    """The model's own value variables are inputs of the graph by position too."""
    def build(swap, seed):
        rng = np.random.default_rng(seed)
        with pm.Model() as m:
            a = pm.Normal("a", 0.3, 1.7)
            b = pm.Normal("b", -0.7, 2.3)
            u, v = (b, a) if swap else (a, b)
            pm.Normal("y", u * 3.7 + v, 1.9, observed=rng.normal(size=9))
        return m

    for swap in (False, True, False, True):
        m = build(swap, seed=int(swap) + 3)
        lg = lift_logp(m)
        x = point(m)
        assert float(lg.fn(x, lg.values)[0]) == pytest.approx(pymc_logp(m, x), rel=1e-12)


# ------------------------------------------------------------------------------------ stale shared variable
def test_set_data_on_a_kept_static_shared_variable_relifts_correctly_with_the_memo_on(counts):
    def build():
        with pm.Model() as m:
            n = pm.Data("n", np.array(3))
            w = pm.Data("w", np.array([1.2, 2.2, 3.2]))
            x = pm.Normal("x", 0.3, 1.7, shape=3)
            pm.Potential("p", -pt.sum(w * x**2) + pt.sum(pt.ones(n)))        # ones(n): n is a shape
        return m

    m = build()
    sampler = PfCheesHMC(m)
    first = sampler._lifted
    assert first.static_shared and first.rule_used == "model_inputs"
    x = point(m)
    assert float(first.fn(x, sampler._values)[0]) == pytest.approx(pymc_logp(m, x), rel=1e-7)

    with m:
        pm.set_data({"n": np.array(5)})
    assert first.is_stale()
    counts.reset()
    sampler._refresh_values()
    assert sampler._lifted is not first and not sampler._lifted.is_stale()
    assert counts.rewrite == 1, "another static value is another graph"
    assert float(sampler._lifted.fn(x, sampler._values)[0]) == pytest.approx(pymc_logp(m, x), rel=1e-7)
    assert pymc_logp(m, x) != pytest.approx(float(first.fn(x, first.values)[0]), rel=1e-9), "the value must matter"

    with m:
        pm.set_data({"n": np.array(3)})                                       # back to the first value
    counts.reset()
    sampler._refresh_values()
    assert counts.rewrite == 0 and sampler._lifted.fn is first.fn
    assert float(sampler._lifted.fn(x, sampler._values)[0]) == pytest.approx(pymc_logp(m, x), rel=1e-7)


# ------------------------------------------------------------------------------------ what the memo keeps
def test_the_memo_holds_no_data():
    """The stored graph has the data replaced by placeholders: the 1000 x 25 design matrix of the logit model
    is lifted, so it is not kept alive (the reason the memo can be an unbounded dict, like the registry)."""
    m = rb.build_model("logit_1000x25")
    lg = lift_sites(m)["logp"]
    assert any(v.size >= 25_000 for v in lg.values)
    memo = next(memo for bucket in lifted._MEMO.values() for memo in bucket)
    sizes = [np.asarray(v.data).size for v in ancestors(memo.outputs) if isinstance(v, TensorConstant)]
    assert sizes and max(sizes) < 100, sizes


# ------------------------------------------------------------------------------------ end to end
def test_second_sampler_call_on_a_rebuilt_model_makes_no_rewrite(counts):
    chains, draws, warmup = 16, 5, 10
    kw = dict(draws=draws, chains=chains, progressbar=False, chees_kwargs={"draws": warmup})
    first = sample_pf_chees_hmc(hier(1, 1), random_seed=1, **kw)
    assert counts.rewrite >= 3
    counts.reset()
    second = sample_pf_chees_hmc(hier(2, 2), random_seed=2, **kw)
    assert (counts.rewrite, counts.funcify, counts.lowering) == (0, 0, 0)
    mu1, mu2 = first.posterior["mu"].values, second.posterior["mu"].values
    assert np.all(np.isfinite(mu2)) and not np.allclose(mu1, mu2)


@pytest.mark.parametrize("name", ["8schools", "logit_1000x25"])
def test_draws_are_bit_identical_with_and_without_the_memo(name, monkeypatch):
    """A sampler call whose lifts are all memo hits and one whose lifts are all builds have the same draws for a
    fixed seed, every array of the InferenceData: same function (the registry names the first for a key), same
    values. (That a build from scratch computes the same numbers as the reused function is checked, bit for
    bit, in the ladder test above.)"""
    kw = dict(draws=5, chains=16, progressbar=False, random_seed=3, chees_kwargs={"draws": 10})
    sample_pf_chees_hmc(rb.build_model(name), **kw)                          # fills the memo and the registry
    filed = sum(len(v) for v in lifted._MEMO.values())
    with_memo = sample_pf_chees_hmc(rb.build_model(name), **kw)
    assert sum(len(v) for v in lifted._MEMO.values()) == filed, "the second call must have been all hits"

    monkeypatch.setattr(lifted, "_recall", lambda inputs, outputs: (None, None))     # every lift builds
    without = sample_pf_chees_hmc(rb.build_model(name), **kw)
    for group in ("posterior", "sample_stats"):
        a, b = getattr(with_memo, group), getattr(without, group)
        assert set(a.data_vars) == set(b.data_vars)
        for var in a.data_vars:
            np.testing.assert_array_equal(a[var].values, b[var].values, err_msg=f"{group}.{var}")
