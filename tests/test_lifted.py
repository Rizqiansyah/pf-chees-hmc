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
    fn, values, _ = lifted._build(model, inputs, outputs, "model_inputs")
    structs = tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values)
    return jax.jit(lambda x, v: fn(x, v)).lower(lifted._abstract(model, inputs), structs).as_text()


@pytest.fixture(scope="module")
def ladder():
    return {name: rb.build_model(name) for name in LADDER}


# ---------------------------------------------------------------- logp equality

@pytest.mark.parametrize("name", LADDER)
def test_ladder_logp_equals_todays_jaxified_logp(ladder, name):
    model = ladder[name]
    x = _point(model)
    lg = lift_logp(model)
    assert lg.rule_used == "model_inputs"
    assert lg.n_lifted == len(lg.values) > 0
    old = float(get_jaxified_logp(model)(x))
    new = float(lg.fn(x, lg.values)[0])
    assert new == pytest.approx(old, rel=1e-12)
    # sign and dtype: PyMC's default is the positive log density, a float64 scalar
    assert np.asarray(lg.fn(x, lg.values)[0]).dtype == np.asarray(get_jaxified_logp(model)(x)).dtype
    assert new == pytest.approx(float(model.compile_logp()(model.initial_point() | {
        v.name: np.asarray(a) for v, a in zip(model.value_vars, x)})), rel=1e-10)


@pytest.mark.parametrize("name", sorted(EXTRA))
def test_extra_model_logp_equals_pymc_c_backend(name):
    model = EXTRA[name](0)
    x = _point(model)
    lg = lift_logp(model)
    assert lg.rule_used == "model_inputs"
    point = {v.name: np.asarray(a) for v, a in zip(model.value_vars, x)}
    ref = float(model.compile_logp()(point))
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(ref, rel=1e-10)


def test_truncated_normal_crashes_on_the_old_jax_path_without_tfp():
    """The mutation: the pre-F2a path. `get_jaxified_logp` needs `erfcx`, which PyTensor only has via TFP.
    Skipped where this pytensor has a JAX `erfcx` of its own (then there is nothing to repair)."""
    from pytensor.link.jax.dispatch.scalar import try_import_tfp_jax_op
    from pytensor.scalar.math import erfcx as _erfcx_op

    try:
        try_import_tfp_jax_op(_erfcx_op)
    except NotImplementedError:
        pass
    else:
        pytest.skip("tensorflow-probability is installed: the old path works")
    model = m_truncnorm(0)
    x = _point(model)
    # Our registration is process-wide once lifted is imported; put PyTensor's own back for the check.
    ours = lifted.jax_funcify.dispatch(type(_erfcx_op))
    from pytensor.link.jax.dispatch.scalar import jax_funcify_from_tfp
    lifted.jax_funcify.register(type(_erfcx_op), jax_funcify_from_tfp)
    try:
        with pytest.raises(NotImplementedError, match="No JAX implementation for Op erfcx"):
            get_jaxified_logp(model)(x)
    finally:
        lifted.jax_funcify.register(type(_erfcx_op), ours)
    # with the registration in place both the old entry point and the lifted one work
    ref = float(model.compile_logp()({v.name: np.asarray(a) for v, a in zip(model.value_vars, x)}))
    assert float(get_jaxified_logp(model)(x)) == pytest.approx(ref, rel=1e-10)
    lg = lift_logp(model)
    assert lg.rule_used == "model_inputs"
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(ref, rel=1e-10)


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


def test_dtype_change_of_a_prior_changes_the_key():
    """DOCUMENTED, ACCEPTED (plan): an integer-written prior changed to a float-written one is a dtype
    change, so a new program. The same holds for any Python literal that PyTensor autocasts to a
    different dtype: it picks the smallest dtype that holds the value exactly (0, 10 -> int8; 1000 ->
    int16; 0.5, 10.0 -> float32; 0.1, 10.1 -> float64). A change across such a boundary recompiles."""
    def build(mu, sd):
        with pm.Model() as m:
            x = pm.Normal("mu", mu, sd)
            pm.Normal("obs", x, 1.0, observed=np.linspace(-1, 1, 10))
        return m

    base = lift_logp(build(0, 10)).key
    assert lift_logp(build(0, 12)).key == base                    # int8 -> int8
    assert lift_logp(build(0.0, 10.0)).key != base                # integer-written -> float-written
    assert lift_logp(build(1, 1000)).key != base                  # int8 -> int16
    assert lift_logp(build(0.5, 10.0)).key == lift_logp(build(0.25, 3.0)).key  # float32 -> float32
    assert lift_logp(build(0.5, 10.0)).key != lift_logp(build(0.5, 10.1)).key  # float32 -> float64


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
    ref = float(m.compile_logp()({v.name: np.asarray(a) for v, a in zip(m.value_vars, x)}))
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(ref, rel=1e-12)


def test_lifted_value_that_feeds_a_shape_falls_back_with_a_warning():
    """A natural failure of rule C: `n * 2` is a value use of the shared `n`, but it then sets a shape.
    On the constant path PyTensor folds it; a lifted `n` is a tracer, so the lowering raises. Rule 2 lifts
    the shared variable again and fails the same way; rule 3 (today's constants) works."""
    with pm.Model() as m:
        n = pm.Data("n", np.array(3))
        x = pm.Normal("x", 0.0, 1.0, shape=3)
        pm.Potential("p", -pt.sum(x**2) + pt.sum(pt.arange(n * 2)))
    x0 = _point(m)
    with pytest.warns(RuntimeWarning, match="rule 'model_inputs' failed") as rec:
        lg = lift_logp(m)
    msgs = [str(w.message) for w in rec]
    assert len(msgs) == 2 and "falling back to 'arrays_and_shared'" in msgs[0] and "'arrays_and_shared' failed" in msgs[1]
    assert lg.rule_used == "none" and lg.n_lifted == 0 and lg.values == ()
    ref = float(m.compile_logp()({v.name: np.asarray(a) for v, a in zip(m.value_vars, x0)}))
    assert float(lg.fn(x0, lg.values)[0]) == pytest.approx(ref, rel=1e-12)


def test_forced_fallback_to_arrays_and_shared(monkeypatch):
    """Rule C raises (patched); the next rule runs and says so."""
    m = m_pm_data(0)
    real = lifted._build

    def failing(model, inputs, outputs, rule):
        if rule == "model_inputs":
            raise RuntimeError("boom")
        return real(model, inputs, outputs, rule)

    monkeypatch.setattr(lifted, "_build", failing)
    with pytest.warns(RuntimeWarning, match=r"rule 'model_inputs' failed with RuntimeError: boom"):
        lg = lift_logp(m)
    assert lg.rule_used == "arrays_and_shared"
    # the formula/prior constants are not lifted by this rule; the data array is
    assert 0 < lg.n_lifted < lift_logp_rule_c_count(m, monkeypatch, real)
    x = _point(m)
    ref = float(m.compile_logp()({v.name: np.asarray(a) for v, a in zip(m.value_vars, x)}))
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(ref, rel=1e-12)


def lift_logp_rule_c_count(m, monkeypatch, real):
    monkeypatch.setattr(lifted, "_build", real)
    return lift_logp(m).n_lifted


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
    assert len(texts) == 2 and "'model_inputs'" in texts[0] and "'arrays_and_shared'" in texts[1]
    assert lg.rule_used == "none"
    x = _point(m)
    ref = float(m.compile_logp()({v.name: np.asarray(a) for v, a in zip(m.value_vars, x)}))
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(ref, rel=1e-12)


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
        ref = float(m.compile_logp()({v.name: np.asarray(u) for v, u in zip(m.value_vars, x)}))
        assert float(lg.fn(x, lg.values)[0]) == pytest.approx(ref, rel=1e-10)


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
    assert float(lg.fn(x, lg.values)[0]) == pytest.approx(float(get_jaxified_logp(model)(x)), rel=1e-12)
