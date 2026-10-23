"""F2a: `pfcheeshmc.lifted` -- the log-density with model inputs as runtime arguments (CPU, fast).

What a wrong answer here would silently corrupt:
  * the logp (a lifted constant fed in the wrong slot, a lost rewrite, a wrong dtype);
  * the compile-once promise (a data or prior value that leaks into the lowered program makes every new
    dataset a recompile, and the structure-only key stops being structure-only);
  * the stability rewrites (`log(1 + x) -> log1p`, `x**2 -> sqr`) that lifting EVERY constant would block.
Models are built fresh from a seed at fixed shapes, so "same structure, other values" is a real rebuild.
"""
import pathlib
import re
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.special as sps
from pymc.sampling.jax import get_jaxified_logp

# PyMC's own C backend (the reference below) cannot link a Dot under this repo's path with a space; the
# same workaround as tests/test_pf_closed_form.py, set on the live config because pytest has already
# imported pytensor.
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas -lm"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import r1_bias as rb  # noqa: E402

from pfcheeshmc import lifted  # noqa: E402
from pfcheeshmc.lifted import lift_graph, lift_logp  # noqa: E402

LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")


# ---------------------------------------------------------------- models drawn from a seed

def m_pm_data(seed):
    """pm.Data for the observed data; floats written as floats."""
    rng = np.random.default_rng(seed)
    with pm.Model() as m:
        y = pm.Data("y", rng.normal(rng.uniform(1, 3), 1.0, size=30))
        mu = pm.Normal("mu", float(rng.uniform(-1, 1)), float(rng.uniform(2, 6)))
        sigma = pm.HalfNormal("sigma", float(rng.uniform(1, 3)))
        pm.Normal("obs", mu, sigma, observed=y)
    return m


def m_int_priors(seed):
    """Integer-written priors, pm.Normal("mu", 0, 10): int8 constants, direct inputs of the RV node."""
    rng = np.random.default_rng(seed)
    with pm.Model() as m:
        mu = pm.Normal("mu", int(rng.integers(0, 5)), int(rng.integers(5, 100)))
        sd = pm.HalfNormal("sd", int(rng.integers(1, 10)))
        pm.Normal("obs", mu, sd, observed=rng.normal(size=25))
    return m


def m_mvnormal(seed):
    """MvNormal with a constant covariance matrix (a Blockwise Cholesky/solve consumer)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + 3.0 * np.eye(3)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0.0, 2.0, shape=3)
        pm.MvNormal("obs", mu=mu, cov=cov, observed=rng.normal(size=(12, 3)))
    return m


def m_potential(seed):
    """A pm.Potential that uses a data array (a weight vector)."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=5)
    with pm.Model() as m:
        x = pm.Normal("x", 0.0, 1.0, shape=5)
        pm.Potential("pot", -pt.sum(w * x**2))
    return m


def m_poisson(seed):
    """Poisson GLM with integer observations (int64 data) and a float design matrix."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 3))
    y = rng.poisson(np.exp(0.3 * X[:, 0] + 0.5))
    with pm.Model() as m:
        beta = pm.Normal("beta", 0.0, 1.0, shape=3)
        b0 = pm.Normal("b0", 0.0, 1.0)
        pm.Poisson("y", pt.exp(pt.dot(X, beta) + b0), observed=y)
    return m


def m_truncnorm(seed):
    """TruncatedNormal with a free mu: its logp has log(erfc(.)) in a parameter-dependent normaliser,
    which PyTensor rewrites to `erfcx`. The old JAX path has no `erfcx` without TFP."""
    rng = np.random.default_rng(seed)
    with pm.Model() as m:
        mu = pm.Normal("mu", float(rng.uniform(-0.5, 0.5)), 1.0)
        pm.TruncatedNormal("x", mu=mu, sigma=float(rng.uniform(0.8, 1.2)), lower=0.0,
                           observed=np.abs(rng.normal(size=20)) + 0.05)
    return m


EXTRA = {
    "pm_data": m_pm_data,
    "int_priors": m_int_priors,
    "mvnormal": m_mvnormal,
    "potential": m_potential,
    "poisson": m_poisson,
    "truncnorm": m_truncnorm,
}


def _point(model, seed=0):
    ip = model.initial_point()
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(ip[v.name] + 0.3 * rng.standard_normal(np.shape(ip[v.name])))
                 for v in model.value_vars)


def _hlo(model):
    """Lowered text of the lifted logp of a FRESH build (bypassing the registry, which would hand two
    builds the same function object and make an equality test vacuous)."""
    inputs, outputs = list(model.value_vars), [model.logp()]
    fn, values, *_ = lifted._build(model, inputs, outputs, "model_inputs")
    structs = tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values)
    return jax.jit(lambda x, v: fn(x, v)).lower(lifted._abstract(model, inputs), structs).as_text()


def _point_dict(model, x):
    return {v.name: np.asarray(a) for v, a in zip(model.value_vars, x)}


def _widened_c_logp(model, x):
    """The widened logp graph (what rule "model_inputs" lifts) on PyTensor's C backend: the exactness
    reference for the JAX path, since both start from the same graph."""
    (out,), _ = lifted._widen_model_inputs(model, [model.logp()])
    f = pytensor.function(list(model.value_vars), out, on_unused_input="ignore")
    return float(f(*[np.asarray(a) for a in x]))


# The unwidened graph (get_jaxified_logp, compile_logp) differs from the widened one by float32 constant
# folding: on main a literal that happens to be float32-representable (0.5, 10.0, 1) has its normalising
# constants (log sigma, ...) folded by PyTensor in float32, 1e-7 relative per constant. Widening makes
# them float64 (more accurate), so "close to today" is 1e-7, not 1e-12.
TODAY_RTOL = 1e-7


@pytest.fixture(scope="module")
def ladder():
    return {name: rb.build_model(name) for name in LADDER}


# ---------------------------------------------------------------- logp equality

@pytest.mark.parametrize("name", LADDER)
def test_ladder_logp_is_exact_on_the_widened_graph_and_close_to_todays(ladder, name):
    model = ladder[name]
    x = _point(model)
    lg = lift_logp(model)
    assert lg.rule_used == "model_inputs"
    assert lg.n_lifted == len(lg.values) > 0
    new = float(lg.fn(x, lg.values)[0])
    # exactness of the JAX path: the same widened graph on PyTensor's C backend
    assert new == pytest.approx(_widened_c_logp(model, x), rel=1e-12)
    # closeness to today's unwidened graph; sign and dtype are get_jaxified_logp's (= model.logp())
    old = get_jaxified_logp(model)(x)
    assert new == pytest.approx(float(old), rel=TODAY_RTOL)
    assert np.asarray(lg.fn(x, lg.values)[0]).dtype == np.asarray(old).dtype


@pytest.mark.parametrize("name", LADDER)
def test_ladder_gradient_matches_todays(ladder, name):
    model = ladder[name]
    x = _point(model)
    lg = lift_logp(model)
    g_new = jax.grad(lambda xs: lg.fn(xs, lg.values)[0])(x)
    g_old = jax.grad(get_jaxified_logp(model))(x)
    for gn, go in zip(g_new, g_old):
        gn, go = np.asarray(gn), np.asarray(go)
        np.testing.assert_allclose(gn, go, rtol=TODAY_RTOL, atol=TODAY_RTOL * np.abs(go).max())


@pytest.mark.parametrize("name", sorted(EXTRA))
def test_extra_model_logp_is_exact_on_the_widened_graph_and_close_to_pymc_c_backend(name):
    model = EXTRA[name](0)
    x = _point(model)
    lg = lift_logp(model)
    assert lg.rule_used == "model_inputs"
    new = float(lg.fn(x, lg.values)[0])
    assert new == pytest.approx(_widened_c_logp(model, x), rel=1e-12)
    assert new == pytest.approx(float(model.compile_logp()(_point_dict(model, x))), rel=TODAY_RTOL)


def test_uniform_with_int8_literals_has_a_finite_logp():
    """PyMC 5.28.5: pm.Uniform("u", -100, 100) has logp NaN on the C and the JAX backend, because both
    literals are int8 and `upper - lower` overflows. Widening the parameters repairs it."""
    def build(lower, upper):
        with pm.Model() as m:
            pm.Uniform("u", lower, upper)
        return m

    ints, floats = build(-100, 100), build(-100.0, 100.0)
    x = _point(ints)
    lg_ints, lg_floats = lift_logp(ints), lift_logp(floats)
    got, ref = float(lg_ints.fn(x, lg_ints.values)[0]), float(lg_floats.fn(x, lg_floats.values)[0])
    assert np.isfinite(got)
    assert got == pytest.approx(ref, rel=1e-12)
    # the same through the shaped form (ExpandDims-folded constants), which is where most models are
    with pm.Model() as shaped:
        pm.Uniform("u", -100, 100, shape=3)
    xs = _point(shaped)
    lg_shaped = lift_logp(shaped)
    assert np.isfinite(float(lg_shaped.fn(xs, lg_shaped.values)[0]))
    # the native-dtype rule is what main computes
    native = lift_graph(ints, list(ints.value_vars), [ints.logp()], rule="model_inputs_native_dtype")
    assert native.rule_used == "model_inputs_native_dtype"
    if np.isfinite(float(native.fn(x, native.values)[0])):
        warnings.warn("PyMC no longer overflows on int8 Uniform bounds: this test's premise is gone")


def test_truncated_normal_crashes_on_the_old_jax_path_without_tfp():
    """The old-code check: PyTensor's own `Erfcx` dispatch (TFP) with this repo's registration put back
    for the duration, then the stock `get_jaxified_logp`. Skipped where TFP is installed."""
    from pytensor.link.jax.dispatch.scalar import jax_funcify_from_tfp, try_import_tfp_jax_op
    from pytensor.scalar.math import Erfcx, erfcx as erfcx_op

    try:
        try_import_tfp_jax_op(erfcx_op)
    except NotImplementedError:
        pass
    else:
        pytest.skip("tensorflow-probability is installed: the old path works")
    model = m_truncnorm(0)
    x = _point(model)
    ours = lifted.jax_funcify.dispatch(Erfcx)
    assert ours is lifted.jax_funcify_Erfcx
    lifted.jax_funcify.register(Erfcx, jax_funcify_from_tfp)
    try:
        with pytest.raises(NotImplementedError, match="No JAX implementation for Op erfcx"):
            get_jaxified_logp(model)(x)
    finally:
        lifted.jax_funcify.register(Erfcx, ours)
    lg = lift_logp(model)
    assert lg.rule_used == "model_inputs"
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(_widened_c_logp(model, x), rel=1e-12)
    # the registration also repairs the stock entry point
    ref = float(model.compile_logp()(_point_dict(model, x)))
    assert float(get_jaxified_logp(model)(x)) == pytest.approx(ref, rel=1e-12)
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(ref, rel=TODAY_RTOL)


# ---------------------------------------------------------------- no leaks

@pytest.mark.parametrize("name", sorted(EXTRA))
def test_rebuilt_model_with_other_values_lowers_to_identical_hlo_and_key(name):
    a, b = EXTRA[name](1), EXTRA[name](2)
    ha, hb = _hlo(a), _hlo(b)
    assert ha == hb, "a data or prior value leaked into the lowered program"
    assert "loc(" not in ha, "source-location metadata in the key text"
    la, lb = lift_logp(a), lift_logp(b)
    assert la.key == lb.key
    # ...and the values really differ, so the equality above is not two copies of one model
    assert any(not np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(la.values, lb.values))


@pytest.mark.parametrize("name", LADDER)
def test_ladder_rebuilt_from_scratch_lowers_to_identical_hlo_and_key(ladder, name):
    """A second, fresh pm.Model of the same notebook (new variable objects, auto-generated names): the
    lowered text and the key must not depend on object identity, names or build order."""
    again = rb.build_model(name)
    assert again is not ladder[name]
    assert _hlo(again) == _hlo(ladder[name])
    assert lift_logp(again).key == lift_logp(ladder[name]).key


def test_shape_change_changes_the_key():
    def build(n):
        rng = np.random.default_rng(0)
        with pm.Model() as m:
            mu = pm.Normal("mu", 0.0, 5.0)
            pm.Normal("obs", mu, 1.0, observed=rng.normal(size=n))
        return m

    assert lift_logp(build(30)).key != lift_logp(build(31)).key
    assert lift_logp(build(30)).key == lift_logp(build(30)).key


def _literal_models():
    """One model per kind of prior literal: shape () and shaped RVs (whose parameter reaches the logp as a
    folded one-element constant), float- and int-valued RVs, and a SymbolicRV."""
    def scalar(mu, sd):
        with pm.Model() as m:
            x = pm.Normal("mu", mu, sd)
            pm.Normal("obs", x, 1.0, observed=np.linspace(-1, 1, 10))
        return m

    def shaped(mu, sd):
        with pm.Model() as m:
            x = pm.Normal("x", mu, sd, shape=3)
            pm.Normal("obs", x.sum(), 1.0, observed=np.linspace(-1, 1, 10))
        return m

    def poisson(rate, n):
        with pm.Model() as m:
            lam = pm.Gamma("lam", rate, 1.0)
            pm.Poisson("c", lam, observed=np.array([1, 4, 2]))
            pm.Binomial("k", n=n, p=0.3, observed=np.array([3, 5, 7]))
        return m

    def truncated(mu, sd):
        with pm.Model() as m:
            a = pm.Normal("a", mu, sd)
            pm.TruncatedNormal("t", mu=a, sigma=1.0, lower=-0.37, observed=np.abs(np.linspace(-1, 1, 7)) + 0.1)
        return m

    return {"scalar": scalar, "shaped": shaped, "poisson": poisson, "truncated": truncated}


# Literal pairs that PyTensor autocasts to different dtypes: int8 -> float32 -> float64 -> int16 -> int32.
# Equal constants are merged by PyMC's own logp rewrites, so a prior literal that equals another constant
# of the model (mu=0 next to lower=0, or a formula constant) shares one graph node with it and gives a
# different program once it stops being equal. The models below keep their other literals clear of these.
LITERALS = [(0, 10), (0.0, 10.0), (2.5, 2.7), (0.5, 3.0), (0.1, 10.1), (1, 1000), (100, 40000), (-3, 12.5)]
# (rate of a Gamma, n of a Binomial): an int-valued RV's integer parameter stays an integer
LITERALS_INT_RV = [(2.5, 10), (0.1, 1000), (7, 40000), (-3, 12), (3.7, 5), (0.75, 99)]


@pytest.mark.parametrize("kind", ["scalar", "shaped", "poisson", "truncated"])
def test_prior_literal_dtype_does_not_change_the_key(kind):
    """Changing a prior value must not recompile, whatever literal the user typed (int8 / int16 / int32 /
    float32 / float64). The scalar prior constants are widened first, so the dtype of the lifted input
    is the model's, not the value's."""
    build = _literal_models()[kind]
    models = [build(*pair) for pair in (LITERALS_INT_RV if kind == "poisson" else LITERALS)]
    keys = {lift_logp(m).key for m in models}
    assert len(keys) == 1, f"{len(keys)} different programs for {len(models)} literal pairs"
    assert all(lift_logp(m).rule_used == "model_inputs" for m in models)
    assert len({_hlo(m) for m in models}) == 1
    # the lifted values do differ, and every lifted prior value has the same dtype
    dtypes = {tuple(str(v.dtype) for v in lift_logp(m).values) for m in models}
    assert len(dtypes) == 1


def test_prior_literal_equal_to_another_constant_is_a_different_program():
    """KNOWN LIMITATION, pinned so a change is noticed: PyMC's logp rewrites merge equal constants, so a
    prior literal that equals another constant of the model (here mu=0 and lower=0, both int8) is one
    graph node with it, and stops being one when it changes. Two programs, each cached by its key."""
    def build(mu):
        with pm.Model() as m:
            a = pm.Normal("a", mu, 3.3)
            pm.TruncatedNormal("t", mu=a, sigma=1.7, lower=0, observed=np.abs(np.linspace(-1, 1, 7)) + 0.1)
        return m

    assert lift_logp(build(0)).key != lift_logp(build(4)).key
    assert lift_logp(build(4)).key == lift_logp(build(5)).key


@pytest.mark.parametrize("floatx, src, rv, want", [
    ("float64", "float32", "float64", "float64"),     # a float narrower than floatX -> floatX
    ("float64", "float16", "int64", "float64"),
    ("float64", "float64", "float64", None),           # already wide
    ("float64", "int8", "float64", "float64"),         # int param of a float RV -> floatX
    ("float64", "int16", "float64", "float64"),
    ("float64", "uint8", "float64", "float64"),
    ("float64", "int64", "float64", "float64"),
    ("float64", "int8", "int64", "int64"),             # int param of an int RV -> int64
    ("float64", "uint32", "int64", "int64"),
    ("float64", "int64", "int64", None),
    ("float64", "uint64", "int64", None),              # int64 cannot hold it: never narrow
    ("float64", "bool", "float64", None),              # bool is left alone
    ("float32", "float64", "float64", None),           # never narrow, whatever floatX says
    ("float32", "float16", "float32", "float32"),
    ("float32", "int8", "float32", "float32"),
    ("float32", "int64", "float32", None),             # int64 does not fit float32
])
def test_widened_dtype_never_narrows(floatx, src, rv, want):
    with pytensor.config.change_flags(floatX=floatx, warn_float64="ignore"):
        assert lifted._widened_dtype(np.dtype(src), np.dtype(rv)) == want


def test_native_dtype_rule_keeps_the_literals_dtype():
    """The plan's rule C unwidened (the first fallback): an int-written prior and a float-written one are
    different programs, as are literals across a float32/float64 boundary."""
    def build(mu, sd):
        with pm.Model() as m:
            x = pm.Normal("mu", mu, sd)
            pm.Normal("obs", x, 1.0, observed=np.linspace(-1, 1, 10))
        return m

    def key(*pair):
        m = build(*pair)
        lg = lift_graph(m, list(m.value_vars), [m.logp()], rule="model_inputs_native_dtype")
        assert lg.rule_used == "model_inputs_native_dtype"
        return lg.key

    assert key(0, 10) == key(0, 12)                 # int8 -> int8
    assert key(0, 10) != key(0.0, 10.0)             # int-written -> float-written
    assert key(1, 10) != key(1, 1000)               # int8 -> int16
    assert key(0.5, 10.0) == key(0.25, 3.0)         # float32 -> float32
    assert key(0.5, 10.0) != key(0.5, 10.1)         # float32 -> float64


def test_static_constant_change_changes_the_key():
    """A static (kept) constant is part of the program, as the plan says: it recompiles once."""
    def build(shift):
        with pm.Model() as m:
            x = pm.Normal("x", 0.0, 1.0, shape=3)
            pm.Potential("p", -pt.sum(pt.exp(x * shift)))   # scalar literal inside a user expression
        return m

    assert lift_logp(build(0.37)).key != lift_logp(build(0.38)).key
    assert lift_logp(build(0.37)).key == lift_logp(build(0.37)).key


def test_static_constants_one_ulp_apart_give_different_keys():
    """The key is the printed StableHLO: it must print doubles round-trip, or two programs differing in a
    static constant would collide and the registry would hand back the wrong function."""
    x0 = 0.37

    def build(shift):
        with pm.Model() as m:
            x = pm.Normal("x", 0.0, 1.0, shape=3)
            pm.Potential("p", -pt.sum(pt.exp(x * shift)))
        return m

    assert lift_logp(build(x0)).key != lift_logp(build(np.nextafter(x0, 1.0))).key


# ---------------------------------------------------------------- rewrites survive

def _primitives(lg, model):
    """Primitive names in the jaxpr of the lifted function (nested jaxprs print inline)."""
    x = _point(model)
    text = str(jax.make_jaxpr(lambda xs, v: lg.fn(xs, v))(x, lg.values))
    return set(re.findall(r"= ([a-z_0-9]+)\[?[ \n]", text)) | set(re.findall(r"\b([a-z_0-9]+)\[", text))


def test_log1p_rewrite_survives_lifting():
    """log(1 + x) is rewritten to Log1p because the 1 is a visible constant. Lifting every constant
    would hide it."""
    with pm.Model() as m:
        x = pm.HalfNormal("x", float(np.random.default_rng(0).uniform(1, 2)), shape=4)
        pm.Potential("p", -pt.sum(pt.log(1 + x)))
    lg = lift_logp(m)
    old = _primitives(lift_graph(m, list(m.value_vars), [m.logp()], rule="none"), m)
    new = _primitives(lg, m)
    assert "log1p" in old, "premise: PyTensor's JAX rewrite produces Log1p on the constant path"
    assert "log1p" in new


def test_square_rewrite_survives_lifting():
    """x**2 with a literal exponent is rewritten to Sqr (-> `square`), not a general `pow`."""
    with pm.Model() as m:
        x = pm.Normal("x", 0.0, 1.0, shape=4)
        pm.Potential("p", -pt.sum(pt.as_tensor(np.arange(1.0, 5.0)) * x**2))
    lg = lift_logp(m)
    old = _primitives(lift_graph(m, list(m.value_vars), [m.logp()], rule="none"), m)
    new = _primitives(lg, m)
    assert "square" in old and "pow" not in old, f"premise: {sorted(old)}"
    assert "square" in new and "pow" not in new, sorted(new)


# ---------------------------------------------------------------- diagnostics and fallback

def test_value_use_filter_keeps_shape_positions_static_and_reports_them():
    """A pm.Data used as a shape is not a value use: replaced by its current value, named in kept_static."""
    with pm.Model() as m:
        n = pm.Data("n", np.array(3))
        w = pm.Data("w", np.array([1.0, 2.0, 3.0]))
        x = pm.Normal("x", 0.0, 1.0, shape=3)
        pm.Potential("p", -pt.sum(w * x**2) + pt.sum(pt.zeros(n)))
    lg = lift_logp(m)
    assert lg.rule_used == "model_inputs"
    assert any(op == "Alloc" for op, _, _ in lg.kept_static), lg.kept_static
    names = {op for op, _, _ in lg.kept_static}
    assert all(isinstance(op, str) for op in names)
    x = _point(m)
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(_widened_c_logp(m, x), rel=1e-12)


def test_lifted_value_that_feeds_a_shape_falls_back_with_a_warning():
    """A natural failure of rule C: `n * 2` is a value use of the shared `n`, but it then sets a shape.
    On the constant path PyTensor folds it; a lifted `n` is a tracer, so the lowering raises. Rule 2 lifts
    the shared variable again and fails the same way; rule 3 (today's constants) works."""
    with pm.Model() as m:
        n = pm.Data("n", np.array(3))
        x = pm.Normal("x", 0.0, 1.0, shape=3)
        pm.Potential("p", -pt.sum(x**2) + pt.sum(pt.arange(n * 2)))
    x0 = _point(m)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        lg = lift_logp(m)
    msgs = [str(w.message) for w in rec]
    assert msgs and "rule 'model_inputs' failed" in msgs[0]
    assert [m.split("'")[1] for m in msgs] == ["model_inputs", "model_inputs_native_dtype", "arrays_and_shared"], msgs
    assert lg.rule_used == "none" and lg.n_lifted == 0 and lg.values == ()
    ref = float(m.compile_logp()({v.name: np.asarray(a) for v, a in zip(m.value_vars, x0)}))
    assert float(lg.fn(x0, lg.values)[0]) == pytest.approx(ref, rel=1e-12)


def test_widening_failure_falls_back_to_the_native_dtype_rule():
    """A natural failure of the widening: LKJCholeskyCov's `n` is an int parameter of a float-valued RV
    that must stay an integer, so rebuilding the graph with a float `n` raises. The next rule is the
    plan's rule C with the literals' own dtypes, which works."""
    with pm.Model() as m:
        chol, _, _ = pm.LKJCholeskyCov("chol", n=3, eta=2.0, sd_dist=pm.Exponential.dist(1.0))
        pm.MvNormal("y", mu=np.zeros(3), chol=chol, observed=np.random.default_rng(0).normal(size=(5, 3)))
    x = _point(m)
    with pytest.warns(RuntimeWarning, match=r"rule 'model_inputs' failed with \w+: .*falling back to 'model_inputs_native_dtype'"):
        lg = lift_logp(m)
    assert lg.rule_used == "model_inputs_native_dtype" and lg.n_lifted > 0
    ref = float(m.compile_logp()(_point_dict(m, x)))
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(ref, rel=TODAY_RTOL)


def test_forced_fallback_to_native_dtype(monkeypatch):
    m = m_int_priors(0)
    n_rule_c = lift_logp(m).n_lifted

    def failing(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(lifted, "_widen_model_inputs", failing)
    with pytest.warns(RuntimeWarning, match=r"rule 'model_inputs' failed with RuntimeError: boom; falling back to 'model_inputs_native_dtype'"):
        lg = lift_logp(m)
    assert lg.rule_used == "model_inputs_native_dtype" and lg.n_lifted == n_rule_c
    x = _point(m)
    # no widening: this is main's graph, so it agrees with PyMC's C backend to rounding
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(float(m.compile_logp()(_point_dict(m, x))), rel=1e-12)


def test_forced_fallback_to_arrays_and_shared(monkeypatch):
    """Both model-input rules raise (patched); the next rule runs and says so."""
    m = m_pm_data(0)
    n_rule_c = lift_logp(m).n_lifted
    real = lifted._build

    def failing(model, inputs, outputs, rule):
        if rule in lifted._MODEL_INPUT_RULES:
            raise RuntimeError("boom")
        return real(model, inputs, outputs, rule)

    monkeypatch.setattr(lifted, "_build", failing)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        lg = lift_logp(m)
    texts = [str(w.message) for w in rec]
    assert [t.split("'")[1] for t in texts] == ["model_inputs", "model_inputs_native_dtype"]
    assert "RuntimeError: boom; falling back to 'arrays_and_shared'" in texts[1]
    assert lg.rule_used == "arrays_and_shared"
    # this rule lifts the data (a shared variable), not the prior-value constants
    assert 0 < lg.n_lifted < n_rule_c
    x = _point(m)
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(float(m.compile_logp()(_point_dict(m, x))), rel=1e-12)


def test_forced_fallback_all_the_way_to_no_lifting(monkeypatch):
    m = m_poisson(0)
    real = lifted._build

    def failing(model, inputs, outputs, rule):
        if rule != "none":
            raise ValueError(f"no {rule}")
        return real(model, inputs, outputs, rule)

    monkeypatch.setattr(lifted, "_build", failing)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        lg = lift_logp(m)
    texts = [str(w.message) for w in rec]
    assert [t.split("'")[1] for t in texts] == ["model_inputs", "model_inputs_native_dtype", "arrays_and_shared"]
    assert lg.rule_used == "none"
    x = _point(m)
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(float(m.compile_logp()(_point_dict(m, x))), rel=1e-12)


def test_rule_can_be_chosen_and_unknown_rules_are_rejected():
    m = m_pm_data(0)
    c = lift_graph(m, list(m.value_vars), [m.logp()])
    b = lift_graph(m, list(m.value_vars), [m.logp()], rule="arrays_and_shared")
    n = lift_graph(m, list(m.value_vars), [m.logp()], rule="none")
    assert (c.rule_used, b.rule_used, n.rule_used) == ("model_inputs", "arrays_and_shared", "none")
    assert c.n_lifted > b.n_lifted > n.n_lifted == 0
    x = _point(m)
    vals = [float(g.fn(x, g.values)[0]) for g in (c, b, n)]
    assert vals[1] == pytest.approx(vals[2], rel=1e-12)       # neither of these widens
    assert vals[0] == pytest.approx(vals[2], rel=TODAY_RTOL)  # rule C does
    with pytest.raises(ValueError, match="unknown rule"):
        lift_graph(m, list(m.value_vars), [m.logp()], rule="everything")


def test_graph_with_a_shared_rng_is_refused_like_pymc_does():
    with pm.Model() as m:
        pm.Normal("x", 0.0, 1.0)
    draw = pt.random.normal(0.0, 1.0)                  # reads a shared RandomType variable
    with pytest.warns(RuntimeWarning, match="RandomType"):
        with pytest.raises(ValueError, match="shared RandomType variables"):
            lift_graph(m, [], [draw])


def test_last_rule_failure_is_not_swallowed(monkeypatch):
    def failing(*a, **k):
        raise ValueError("last")

    monkeypatch.setattr(lifted, "_build", failing)
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="last"):
            lift_logp(m_pm_data(0))


def test_lifted_order_is_deterministic_across_builds():
    """Argument order comes from the graph structure, not from set or id order: two fresh builds list
    the same dtypes and shapes in the same order (with values that differ)."""
    sig = [tuple((v.dtype.name, v.shape) for v in lift_logp(m_poisson(s)).values) for s in range(4)]
    assert len(set(sig)) == 1


# ---------------------------------------------------------------- sampler wiring

def _m_with_deterministic(seed):
    rng = np.random.default_rng(seed)
    with pm.Model() as m:
        w = pm.Data("w", rng.uniform(0.5, 2.0, size=4))
        sigma = pm.HalfNormal("sigma", float(rng.uniform(1, 3)))          # log-transformed value var
        p = pm.Beta("p", 2.0, 3.0)                                         # logodds-transformed
        z = pm.Normal("z", 0.0, 1.0, shape=4)                              # untransformed
        pm.Deterministic("scaled", w * z * sigma)                          # depends on data and a value var
        pm.Normal("obs", z.sum() * p, sigma, observed=rng.normal(size=6))
    return m


def test_value_to_rv_transform_equals_todays_and_keeps_its_calling_convention():
    """`_compile_value_to_rv_fn` still returns (fn(*value_arrays) -> list of outputs, vars), as the
    sampler calls it under a nested vmap."""
    from pymc.sampling.jax import get_jaxified_graph
    from pfcheeshmc.sampler import PfCheesHMC

    m = _m_with_deterministic(0)
    s = PfCheesHMC(m, seed=0)
    for keep in (False, True):
        fn, outs = s._compile_value_to_rv_fn(keep_untransformed=keep)
        old = get_jaxified_graph(inputs=m.value_vars, outputs=outs)
        rng = np.random.default_rng(1)
        ip = m.initial_point()
        raw = [jnp.asarray(ip[v.name] + 0.2 * rng.standard_normal((2, 3) + np.shape(ip[v.name]))) for v in m.value_vars]
        got = jax.vmap(jax.vmap(fn))(*raw)
        want = jax.vmap(jax.vmap(old))(*raw)
        assert len(got) == len(want) == len(outs)
        for g, w, v in zip(got, want, outs):
            np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-13, atol=0, err_msg=v.name)
    assert any(v.name == "scaled" for v in outs)


def test_sampler_log_density_is_the_lifted_one_and_draws_match_the_old_path():
    """End to end on CPU: PfCheesHMC with its lifted log-density against the same sampler with the old
    `get_jaxified_logp` closure swapped in. Draws are NOT bit-identical in general (the lifted program
    leaves prior-value arithmetic to XLA instead of PyTensor's constant folding, which moves the logp by
    an ulp and the trajectory by ~1e-8); they agree far inside any statistical tolerance."""
    from pfcheeshmc.sampler import PfCheesHMC

    m = m_pm_data(0)
    kw = dict(chains=4, draws=4, keep_untransformed=True, chees_kwargs={"draws": 8, "progressbar": False},
              dynamic_hmc_kwargs={"progressbar": False}, pathfinder_kwargs={"num_paths": 2, "draws": 16})
    new = PfCheesHMC(m, seed=3)
    assert isinstance(new._lifted, lifted.LiftedGraph) and new._lifted.rule_used == "model_inputs"
    old = PfCheesHMC(m, seed=3)
    f = get_jaxified_logp(m)
    old._ld_fn = lambda seq: f(seq)
    a, b = new.sample(**kw), old.sample(**kw)
    for name in ("mu", "sigma_log__", "sigma"):
        np.testing.assert_allclose(a.posterior[name].values, b.posterior[name].values, rtol=0, atol=1e-6)


# ---------------------------------------------------------------- registry

def test_structurally_identical_models_share_one_function_object():
    a, b = lift_logp(m_poisson(10)), lift_logp(m_poisson(11))
    assert a.key == b.key and a.fn is b.fn
    c = lift_logp(m_mvnormal(10))
    assert c.key != a.key and c.fn is not a.fn
    # the shared function works on each model's own values
    for seed, lg in ((10, a), (11, b)):
        m = m_poisson(seed)
        x = _point(m)
        assert float(lg.fn(x, lg.values)[0]) == pytest.approx(_widened_c_logp(m, x), rel=1e-12)


# ---------------------------------------------------------------- erfcx

def _band_grid():
    return np.unique(np.concatenate([
        np.linspace(-20, 40, 60001),
        np.linspace(26.4, 26.8, 40001),          # the guard band (26.5433 .. 26.6417) and both sides
        np.geomspace(40, 1e5, 2001),
        [26.55, 26.63, 26.5432584, 26.6417476, 1e5],
    ]))


def test_erfcx_matches_scipy_everywhere_including_the_guard_band():
    x = _band_grid()
    got = np.asarray(jax.jit(lifted._erfcx_guarded)(jnp.asarray(x)))
    want = sps.erfcx(x)
    assert np.all(np.isfinite(want))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    # the premise: stock JAX is wrong (exactly 0) in the band, which is what the guard repairs
    stock = np.asarray(jax.jit(jax.scipy.special.erfcx)(jnp.asarray(x)))
    wrong = np.abs(stock - want) > 1e-10 * want
    assert wrong.any() and x[wrong].min() > 26.54 and x[wrong].max() < 26.65 and np.all(stock[wrong] == 0)


def test_erfcx_float32_guard_band():
    x = np.linspace(8.0, 10.5, 25001).astype(np.float32)
    got = np.asarray(jax.jit(lifted._erfcx_guarded)(jnp.asarray(x)))
    np.testing.assert_allclose(got, sps.erfcx(x.astype(np.float64)), rtol=3e-6)
    assert got.dtype == np.float32


def test_erfcx_gradient_is_finite_and_correct():
    x = _band_grid()
    g = np.asarray(jax.jit(jax.vmap(jax.grad(lifted._erfcx_guarded)))(jnp.asarray(x)))
    assert np.all(np.isfinite(g))
    # d/dx erfcx = 2 x erfcx(x) - 2/sqrt(pi); that difference cancels to ~1 / (sqrt(pi) x^2) for large x,
    # so the formula itself is only good to ~1e-16 absolute there.
    want = 2 * x * sps.erfcx(x) - 2 / np.sqrt(np.pi)
    np.testing.assert_allclose(g, want, rtol=1e-9, atol=1e-14)
    band = (x > 26.54) & (x < 26.65)
    # inside the band the gradient comes from the series: it must be the true one (about -1/(sqrt(pi) x^2))
    xb = x[band]
    np.testing.assert_allclose(g[band], -1.0 / (np.sqrt(np.pi) * xb**2) * (1 - 3 / (2 * xb**2) + 15 / (4 * xb**4)),
                               rtol=1e-6)


def test_erfcx_is_registered_for_pytensor_including_inside_a_composite():
    from pytensor.scalar.basic import Composite
    from pytensor.compile.mode import JAX

    x = pt.vector("x")
    y = pt.erfcx(x) * 3.0 + pt.exp(x * 0.0)           # elementwise chain that fusion turns into one Composite
    fg = pytensor.graph.fg.FunctionGraph([x], [y])
    JAX.optimizer.rewrite(fg)
    composites = [n.op.scalar_op for n in fg.apply_nodes if isinstance(getattr(n.op, "scalar_op", None), Composite)]
    assert any("Erfcx" in str(c.fgraph.apply_nodes) or "erfcx" in str(c.fgraph.apply_nodes).lower()
               for c in composites), "premise: the rewrite put erfcx inside a Composite"
    f = pytensor.function([x], y, mode="JAX")
    grid = np.array([-3.0, 0.0, 5.0, 26.58, 26.6, 100.0])
    np.testing.assert_allclose(f(grid), sps.erfcx(grid) * 3.0 + 1.0, rtol=1e-12)


def test_gev_prior_normaliser_needs_erfcx_and_now_works(ladder):
    model = ladder["gev"]
    x = _point(model)
    lg = lift_logp(model)
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(float(get_jaxified_logp(model)(x)), rel=TODAY_RTOL)
