"""F2d: the chains' initial points with no PyTensor compile.

What this replaces. `_pymc_adapt_diag_start` ran `pm.init_nuts` on every call: a whole `pm.NUTS` step (the
log-density and its gradient compiled on PyTensor's C backend, plus an initial-point function per chain
count), thrown away, because the sampler keeps only `init_points`. 0.5-1.6 s per call on CPU, 0.35-0.66 s on
the GPU box, whatever the model. For `"adapt_diag"` and `"jitter+adapt_diag"` those points are PyMC's own
initial-point graph (`make_initial_point_expression`) with, for the second, a U(-1, 1) jitter in the space of
the value variables, redrawn while the log-density is not finite. The sampler now evaluates that graph in JAX
through F2a's lifting (`lifted.lift_initial_point`) and a module-level jit (`compiled.initial_point`,
`compiled.jittered_initial_points`).

What is pinned here:
* `adapt_diag` points EQUAL PyMC's (<= 1e-12) on the ladder models and on models with a user `initval`, bounded
  and simplex transforms, a `pm.Data` and `dims`;
* `jitter+adapt_diag` is PyMC's distribution (per coordinate, `start - unjittered` is U(-1, 1), KS at C=4096,
  and two-sample KS against PyMC's own), with PyMC's retry and its error when the retries run out;
* a second call with other prior or data values compiles no XLA program and builds no PyTensor function;
* what cannot be built in JAX (a random initial point) or is not one of the two modes goes to `pm.init_nuts`,
  with a warning naming the exception when it was an error;
* `pm_tune` is unused (PyMC reads `tune` only for `jitter+adapt_diag_grad`);
* `lift_logp` of a model with `dims=` (unknown static shapes) builds no PyTensor function.
"""
import pathlib
import sys
import warnings

import jax
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy import stats

# PyMC's own C backend (the reference below) cannot link a Dot under this repo's path with a space.
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas -lm"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import r1_bias as rb  # noqa: E402
from compile_counter import count_compiles, count_pytensor_functions  # noqa: E402
from seeded_models import hier  # noqa: E402

from pfcheeshmc import compiled, lifted, sampler as sampler_module  # noqa: E402
from pfcheeshmc.lifted import lift_initial_point, lift_logp  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402

LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")


# ------------------------------------------------------------------------------------------- models
def m_initval():
    """A user `initval`: a scalar, an array, and one on a transformed variable."""
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1, initval=0.3)
        b = pm.Normal("b", 0, 1, shape=3, initval=np.array([0.1, -0.2, 0.7]))
        s = pm.HalfNormal("s", 2.0, initval=1.7)
        pm.Normal("y", a + b.sum(), s, observed=[0.1, 0.2, 0.3])
    return m


def m_transforms():
    """Interval (Uniform), log-odds (Beta), log (Gamma), stick-breaking (Dirichlet): every finite for any jitter."""
    with pm.Model() as m:
        u = pm.Uniform("u", -3, 5, shape=2)
        be = pm.Beta("be", 2, 5)
        g = pm.Gamma("g", 2, 3)
        d = pm.Dirichlet("d", np.array([1.0, 2.0, 3.0]))
        h = pm.HalfNormal("h", 2.0, shape=3)
        pm.Normal("y", u.sum() + be + g + d[0] + h.sum(), 1.0, observed=[0.1, 0.2])
    return m


def m_data():
    """The support point depends on a `pm.Data` array and a `pm.Data` scalar."""
    with pm.Model() as m:
        x = pm.Data("x", np.array([1.0, 2.0, 3.0]))
        k = pm.Data("k", 2.5)
        th = pm.Normal("th", x * k, 1.0)
        pm.Normal("y", th, 1.0, observed=np.zeros(3))
    return m


def m_dims(values=(0.0, 1.0, 2.0, 3.0, 4.0), scale=2.5):
    """`dims=`: the value variable has an unknown static dimension (`(None,)`)."""
    with pm.Model(coords={"g": list("abcde")}) as m:
        x = pm.Data("x", np.asarray(values), dims="g")
        th = pm.Normal("th", x, 1, dims="g")
        s = pm.HalfNormal("s", scale)
        pm.Normal("y", th, s, observed=np.arange(5.0), dims="g")
    return m


def m_random_point():
    """No `support_point` for LKJCorr: PyMC's initial point is a draw from the prior."""
    with pm.Model() as m:
        pm.LKJCorr("lc", n=3, eta=2.0)
        pm.Normal("a", 0, 1)
    return m


def m_half_line():
    """Finite log-density only for x >= 0: half of the jittered points around 0 are infeasible."""
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        pm.Potential("fence", pt.switch(x >= 0, 0.0, -np.inf))
    return m


EXTRA = {"initval": m_initval, "data": m_data, "dims": m_dims}


@pytest.fixture(scope="module")
def ladder():
    return {name: rb.build_model(name) for name in LADDER}


def pymc_points(model, init, chains, seed=5, **kw):
    points, _ = pm.init_nuts(init=init, chains=chains, model=model, random_seed=seed, progressbar=False,
                             quiet=True, **kw)
    return points


def stack(points, names):
    return {n: np.stack([np.asarray(p[n]) for p in points]) for n in names}


def start(model, init, chains, **kw):
    s = PfCheesHMC(model, seed=1)
    return s, s._pymc_adapt_diag_start(chains, init=init, **kw)


# ----------------------------------------------------------------------- 1. adapt_diag equals PyMC's
def check_equal_to_pymc(model, atol=1e-12):
    s, got = start(model, "adapt_diag", 3, random_seed=5)
    want = stack(pymc_points(model, "adapt_diag", 3), s.rvs)
    assert list(got) == s.rvs                                           # self.rvs order
    for n in s.rvs:
        assert got[n].shape == want[n].shape == (3,) + np.shape(model.initial_point()[n]), n
        assert np.asarray(got[n]).dtype == want[n].dtype, n
        np.testing.assert_allclose(np.asarray(got[n]), want[n], rtol=0, atol=atol, err_msg=n)


@pytest.mark.parametrize("name", LADDER)
def test_adapt_diag_points_equal_pymcs_on_the_ladder(ladder, name):
    check_equal_to_pymc(ladder[name])


@pytest.mark.parametrize("name", ["initval", "data", "dims"])
def test_adapt_diag_points_equal_pymcs_with_initvals_data_and_dims(name):
    check_equal_to_pymc(EXTRA[name]())


def m_bounded():
    """Interval, log-odds, log and stick-breaking transforms, the parameters written as floats (and 2/7 etc. as 2.0/7.0)."""
    with pm.Model() as m:
        pm.Uniform("u", -3.0, 5.0, shape=2)
        pm.Beta("be", 2.0, 5.0)
        pm.Gamma("g", 2.0, 3.0)
        pm.Dirichlet("d", np.array([1.0, 2.0, 3.0]))
        pm.Normal("y", 0.0, 1.0, observed=[0.1, 0.2])
    return m


def test_adapt_diag_points_equal_pymcs_with_bounded_and_simplex_transforms_up_to_pymcs_float32():
    """PyMC's own value is the one that is off here: the parameters are float32 literals (Python floats and ints
    are autocast to the narrowest dtype that holds them), so `log(alpha / beta)` is a float32 computation and
    differs from the float64 value by ~1e-7. `lift_graph` widens scalar prior constants to float64 on purpose
    (the dtype of a program must not depend on the value of a prior), so the point here is the exact one."""
    m = m_bounded()
    check_equal_to_pymc(m, atol=1e-6)
    s, got = start(m, "adapt_diag", 2, random_seed=5)
    want = stack(pymc_points(m, "adapt_diag", 2), s.rvs)
    np.testing.assert_allclose(np.asarray(got["g_log__"]), np.log(2.0 / 3.0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(got["be_logodds__"]), np.log(2.0 / 5.0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(got["u_interval__"]), 0.0, rtol=0, atol=1e-15)
    assert 0 < np.abs(np.asarray(got["g_log__"]) - want["g_log__"]).max() < 1e-6            # PyMC's float32 noise


def test_adapt_diag_points_do_not_depend_on_the_seed_or_on_tune():
    """`pm_tune` is unused: `init_nuts` reads `tune` only for `jitter+adapt_diag_grad`. Checked on PyMC itself."""
    m = hier(1, 1)
    base = start(m, "adapt_diag", 2, random_seed=1, tune=500)[1]
    for kw in (dict(random_seed=9, tune=500), dict(random_seed=1, tune=1), dict(random_seed=1, tune=20000)):
        other = start(m, "adapt_diag", 2, **kw)[1]
        assert all(np.array_equal(base[n], other[n]) for n in base), kw
    s = PfCheesHMC(m, seed=1)
    for init in ("adapt_diag", "jitter+adapt_diag"):                       # PyMC itself, the claim to pin
        a = stack(pymc_points(m, init, 2, tune=1), s.rvs)
        b = stack(pymc_points(m, init, 2, tune=20000), s.rvs)
        assert all(np.array_equal(a[n], b[n]) for n in s.rvs), init
    j1 = start(m, "jitter+adapt_diag", 4, random_seed=2, tune=1)[1]
    j2 = start(m, "jitter+adapt_diag", 4, random_seed=2, tune=20000)[1]
    assert all(np.array_equal(j1[n], j2[n]) for n in j1)


def test_dict_to_position_is_the_models_initial_point():
    m = m_dims()
    s = PfCheesHMC(m, seed=1)
    with count_pytensor_functions() as pf:
        got = s.dict_to_position(m, s.rvs)
    ip = m.initial_point()
    assert pf.count == 0, str(pf)
    for n, a in zip(s.rvs, got):
        np.testing.assert_allclose(np.asarray(a), ip[n], rtol=0, atol=1e-12)


# ---------------------------------------------------- 2. jitter+adapt_diag: PyMC's distribution
C_KS = 4096


@pytest.fixture(scope="module")
def jittered():
    m = m_transforms()
    s = PfCheesHMC(m, seed=1)
    base = s._pymc_adapt_diag_start(1, init="adapt_diag")
    got = s._pymc_adapt_diag_start(C_KS, init="jitter+adapt_diag", random_seed=3)
    return m, s, base, got


def flat(points, names):
    return np.concatenate([np.asarray(points[n]).reshape(len(points[n]), -1) for n in names], axis=1)


def test_jitter_is_uniform_minus_one_one_per_coordinate_and_independent(jittered):
    """start - unjittered ~ U(-1, 1): KS per coordinate at C=4096 (all finite on this model, so no retry)."""
    m, s, base, got = jittered
    assert all(a.shape[0] == C_KS for a in got.values()) and list(got) == s.rvs
    u = flat(got, s.rvs) - flat(base, s.rvs)
    assert u.shape[1] == 2 + 1 + 1 + 2 + 3                                # u, be, g, d (2 free), h
    assert np.abs(u).max() <= 1.0
    for j in range(u.shape[1]):
        p = stats.kstest(u[:, j], "uniform", args=(-1.0, 2.0)).pvalue
        assert p > 1e-3, (j, p)
    corr = np.corrcoef(u.T) - np.eye(u.shape[1])
    assert np.abs(corr).max() < 0.07                                       # 4.5 sd of a null correlation at n=4096


def test_jitter_matches_pymcs_own_distribution(jittered):
    """Two-sample KS of our jittered points against `pm.init_nuts`'s, per coordinate (different RNG streams)."""
    m, s, base, got = jittered
    theirs = flat(stack(pymc_points(m, "jitter+adapt_diag", 2048, seed=11), s.rvs), s.rvs)
    ours = flat(got, s.rvs)
    for j in range(ours.shape[1]):
        p = stats.ks_2samp(ours[:, j], theirs[:, j]).pvalue
        assert p > 1e-3, (j, p)


def test_jitter_is_a_function_of_pm_seed_only():
    m = m_transforms()
    s = PfCheesHMC(m, seed=1)
    a = s._pymc_adapt_diag_start(16, init="jitter+adapt_diag", random_seed=3)
    b = PfCheesHMC(m, seed=99)._pymc_adapt_diag_start(16, init="jitter+adapt_diag", random_seed=3)
    c = s._pymc_adapt_diag_start(16, init="jitter+adapt_diag", random_seed=4)
    assert all(np.array_equal(a[n], b[n]) for n in a)
    assert not any(np.array_equal(a[n], c[n]) for n in a)
    rows = np.asarray(a["u_interval__"])
    assert len({tuple(r) for r in rows}) == 16                             # chains differ from one another


def test_jitter_redraws_chains_with_a_non_finite_log_density():
    """The fence is x >= 0: a jitter around 0 is infeasible half the time; the retried points are U(0, 1)."""
    m = m_half_line()
    s = PfCheesHMC(m, seed=1)
    got = s._pymc_adapt_diag_start(2048, init="jitter+adapt_diag", random_seed=3, jitter_max_retries=60)
    x = np.asarray(got["x"])
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert stats.kstest(x, "uniform").pvalue > 1e-3
    # without retries the same draw is infeasible half the time: the retry is what removes those
    ld = s._logdensity()
    _, finite = compiled.jittered_initial_points(
        ld.fn, ld.names, ld.values, (np.zeros(()),), jax.random.key(0), 0, chains=2048)
    assert 0.4 < float(np.mean(np.asarray(finite))) < 0.6


def test_jitter_that_runs_out_of_retries_raises_pymcs_error():
    m = m_half_line()
    s = PfCheesHMC(m, seed=1)
    with pytest.raises(pm.exceptions.SamplingError):
        pymc_points(m, "jitter+adapt_diag", 64, jitter_max_retries=0)         # PyMC does
    with pytest.raises(pm.exceptions.SamplingError):
        s._pymc_adapt_diag_start(64, init="jitter+adapt_diag", random_seed=1, jitter_max_retries=0)


# ------------------------------------------------------ 3. nothing compiled on a second call
@pytest.mark.parametrize("init", ["adapt_diag", "jitter+adapt_diag"])
def test_other_prior_and_data_values_compile_no_xla_program_and_no_pytensor_function(init):
    first = start(hier(1, 1), init, 4, random_seed=1)[1]
    for data_seed, prior_seed in [(2, 1), (1, 2), (3, 3)]:
        m = hier(data_seed, prior_seed)
        with count_compiles() as cc, count_pytensor_functions() as pf:
            s, got = start(m, init, 4, random_seed=2)
        assert cc.count == 0 and pf.count == 0, (data_seed, prior_seed, str(cc), str(pf))
        if init == "adapt_diag":                                          # and the values follow the priors
            want = stack(pymc_points(m, init, 4), s.rvs)
            for n in s.rvs:
                np.testing.assert_allclose(np.asarray(got[n]), want[n], rtol=0, atol=1e-12)
            assert (prior_seed == 1) == np.array_equal(np.asarray(got["mu"]), np.asarray(first["mu"]))


def test_the_pytensor_counter_sees_the_fallback_it_is_meant_to_catch():
    with count_pytensor_functions() as pf:
        start(hier(1, 1), "adapt_full", 2, random_seed=1)
    assert pf.count >= 2, str(pf)


def test_the_lifted_initial_point_is_one_function_for_one_structure():
    a, b, c = lift_initial_point(hier(1, 1)), lift_initial_point(hier(2, 3)), lift_initial_point(m_data())
    assert a.key == b.key and a.fn is b.fn and a.key != c.key
    assert a.rule_used == "model_inputs" and a.n_lifted >= 1
    assert not np.allclose(np.asarray(a.values[0]), np.asarray(b.values[0]))        # the priors are values


# ----------------------------------------------------------------------------- 4. the fallback
def test_a_random_initial_point_falls_back_to_pymc_with_a_warning_naming_the_exception():
    m = m_random_point()
    s = PfCheesHMC(m, seed=1)
    with pytest.warns(RuntimeWarning, match=r"NotImplementedError.*falling back to pm\.init_nuts"):
        got = s._pymc_adapt_diag_start(2, init="adapt_diag", random_seed=5)
    assert list(got) == s.rvs and got[s.rvs[0]].shape == (2, 3)
    assert all(np.all(np.isfinite(np.asarray(a))) for a in got.values())


def test_a_failure_to_lift_falls_back_to_pymc_with_a_warning_naming_the_exception(monkeypatch):
    m = hier(1, 1)
    s = PfCheesHMC(m, seed=1)

    def boom(model):
        raise RuntimeError("cannot lift")

    monkeypatch.setattr(sampler_module, "lift_initial_point", boom)
    with pytest.warns(RuntimeWarning, match=r"RuntimeError: cannot lift; falling back to pm\.init_nuts"):
        got = s._pymc_adapt_diag_start(2, init="adapt_diag", random_seed=5)
    want = stack(pymc_points(m, "adapt_diag", 2), s.rvs)
    assert all(np.array_equal(np.asarray(got[n]), want[n]) for n in s.rvs)


@pytest.mark.parametrize("init", ["adapt_full", "jitter+adapt_full", "jitter+adapt_diag_grad"])
def test_other_init_modes_go_to_pymc_without_a_warning(init, monkeypatch):
    calls = []
    real = pm.init_nuts
    monkeypatch.setattr(pm, "init_nuts", lambda **kw: calls.append(kw["init"]) or real(**kw))
    s = PfCheesHMC(hier(1, 1), seed=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = s._pymc_adapt_diag_start(2, init=init, random_seed=5)
    assert calls == [init] and list(got) == s.rvs
    assert not [w for w in caught if "falling back" in str(w.message)]


def test_a_model_with_a_discrete_variable_still_gets_pymcs_error():
    with pm.Model() as m:
        pm.Poisson("k", 3.0)
    s = PfCheesHMC(m, seed=1)
    with pytest.raises(ValueError, match="continuous"):
        s._pymc_adapt_diag_start(2, init="adapt_diag", random_seed=1)


# --------------------------------------------- 5. the abstract shapes of a model with `dims=`
def test_lifting_a_model_with_dims_builds_no_pytensor_function():
    """`dims=` makes the value variable's static shape `(None,)`; `_abstract` used to compile `model.initial_point()`."""
    m = m_dims()
    assert any(None in v.type.shape for v in m.value_vars)
    with count_pytensor_functions() as pf:
        lg = lift_logp(m)
        lifted._abstract(m, list(m.value_vars))
    assert pf.count == 0, str(pf)
    ip = m.initial_point()
    assert [tuple(s.shape) for s in lifted._abstract(m, list(m.value_vars))] == [np.shape(ip[v.name]) for v in m.value_vars]
    assert lg.rule_used == "model_inputs"


def test_the_shapes_of_the_initial_point_are_right_after_a_dim_length_changes():
    with pm.Model(coords={"g": list("abc")}) as m:
        th = pm.Normal("th", 0, 1, dims="g")
        pm.Normal("y", th, 1.0, observed=np.zeros(3), dims="g")
    assert lifted._initial_point_shapes(m) == {"th": (3,)}
    m.set_dim("g", 5, coord_values=list("abcde"))
    assert lifted._initial_point_shapes(m) == {"th": (5,)}
    assert lifted._initial_point_shapes(m)["th"] == np.shape(m.initial_point()["th"])


def test_the_shape_fallback_still_reads_the_initial_point(monkeypatch):
    """If folding the shapes raises, `_abstract` falls back to `model.initial_point()` (a PyTensor compile)."""
    m = m_dims()

    def boom(model):
        raise RuntimeError("no folding")

    monkeypatch.setattr(lifted, "_initial_point_shapes", boom)
    with count_pytensor_functions() as pf:
        specs = lifted._abstract(m, list(m.value_vars))
    assert pf.count == 1 and [tuple(s.shape) for s in specs] == [(5,), ()]
