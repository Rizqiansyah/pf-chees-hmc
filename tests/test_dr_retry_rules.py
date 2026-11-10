"""Retry rules of delayed rejection (``delayed_rejection={"factor": a, "stages": k, "retry": ...}``).

Modi, Barnett & Carpenter (2023), eq. (30): any retry probability p_(j+1) that is a function of the state and its
proposals keeps the kernel exact if the ghosts evaluate the same p. Rules: "divergent" (the default, the indicator of a
divergence; test_delayed_rejection.py), {"energy_above": c} (the indicator dH > c) and "probabilistic"
(p = 1 - alpha_j, the paper's eq. 31, with its own retry uniform). Guards:
  1. invariance per rule, two and three stages, on the hard-edge target of test_delayed_rejection.py and on Neal's
     funnel (the paper's case, where ordinary rejections in the neck are what retries help);
  2. power: the invariance test sees a wrong ghost construction under each rule; the threshold rule's p at the ghosts
     is seen on the funnel (the probabilistic rule's p is not testable at this size: see the test's docstring);
  3. "divergent" given explicitly is the default's program; the option is checked and recorded.
CPU.
"""
import inspect
import json
import textwrap
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st

from pfcheeshmc import compiled
from pfcheeshmc.sampler import PfCheesHMC

STATIC = ("fn", "names", "chunk", "jitter_amount", "dr_factor", "dr_batch", "dr_stages", "dr_retry")
N, K = 20_000, 20

# ---- the hard edge (test_delayed_rejection.py): Gamma(3.7), log density -inf below 0
EDGE = st.gamma(3.7)


def _edge_logp(pos):
    x = pos["x"]
    return jnp.where(x > 0, 2.7 * jnp.log(jnp.abs(x) + 1e-300) - x, -jnp.inf)


# ---- Neal's funnel in two dimensions: v ~ N(0, 3), x | v ~ N(0, exp(v / 2))
V = st.norm(0.0, 3.0)


def _funnel_logp(pos):
    v, x = pos["v"], pos["x"]
    return -v ** 2 / 18.0 - 0.5 * x ** 2 * jnp.exp(-v) - 0.5 * v


def _funnel_start(n):
    rng = np.random.default_rng(0)
    v = V.rvs(n, random_state=rng)
    return {"v": jnp.asarray(v), "x": jnp.asarray(rng.normal(size=n) * np.exp(v / 2))}


def _run(chunk, target, retry, stages=2, factor=4, eps=None, L=3.0):
    if target == "edge":
        ld = compiled.plain_log_density(_edge_logp, ["x"])
        start, eps = {"x": jnp.asarray(EDGE.rvs(N, random_state=np.random.default_rng(0)))}, eps or 1.2
        imm = jnp.ones(1)
    else:
        ld = compiled.plain_log_density(_funnel_logp, ["v", "x"])
        start, eps = _funnel_start(N), eps or 0.9
        imm = jnp.ones(2)
    state = compiled.batch_init(ld.fn, ld.names, ld.values, start, 0)
    s, (_, info) = chunk(ld.fn, ld.names, ld.values, state, imm, eps, L, jax.random.key(7), np.int32(0), np.int32(K),
                         chunk=K, jitter_amount=1.0, dr_factor=factor, dr_stages=stages, dr_retry=retry)
    return {k: np.asarray(v) for k, v in s.position.items()}, info


def _check(target, pos):
    """KS and the lower-tail share (the edge; the funnel's neck), in binomial standard errors."""
    x, law = (pos["x"], EDGE) if target == "edge" else (pos["v"], V)
    q = 0.005
    z = (np.mean(x <= law.ppf(q)) - q) / np.sqrt(q * (1 - q) / len(x))
    return st.kstest(x, law.cdf).pvalue, z


RULES = [("energy_above", 2.0), "probabilistic"]


@pytest.mark.parametrize("target", ["edge", "funnel"])
@pytest.mark.parametrize("stages", [2, 3])
@pytest.mark.parametrize("retry", RULES, ids=["energy_above", "probabilistic"])
def test_invariance(target, stages, retry):
    pos, info = _run(compiled.sampling_chunk, target, retry, stages=stages)
    retried = np.asarray(info.retried)
    assert int(np.sum(info.retry_accepted)) > 500, int(np.sum(info.retry_accepted))
    # every rule retries rejected proposals only, and more than the divergent ones
    assert not np.any(retried & np.asarray(info.is_accepted))
    assert retried.sum() > np.sum(np.asarray(info.is_divergent) & ~np.asarray(info.is_accepted))
    p, z = _check(target, pos)
    assert p > 1e-3 and abs(z) < 4, (p, z)


def test_funnel_retries_help():
    """The neck: retries of ordinary rejections reach it; without them the chains started there stay longer."""
    _, off = _run(compiled.sampling_chunk, "funnel", "divergent")
    _, on = _run(compiled.sampling_chunk, "funnel", "probabilistic")
    assert np.mean(np.asarray(on.is_accepted) | np.asarray(on.retry_accepted)) > \
        np.mean(np.asarray(off.is_accepted) | np.asarray(off.retry_accepted)) + 0.02


def _wrong(monkeypatch, old, new, name):
    src = textwrap.dedent(inspect.getsource(compiled.delayed_rejection_step))
    bad = src.replace(old, new)
    assert bad != src
    ns = dict(vars(compiled))
    exec(bad, ns)
    monkeypatch.setattr(compiled, "delayed_rejection_step", ns["delayed_rejection_step"])
    f = compiled.sampling_chunk.__wrapped__
    fresh = types.FunctionType(f.__code__, f.__globals__, name, f.__defaults__, f.__closure__)
    fresh.__kwdefaults__ = f.__kwdefaults__
    return jax.jit(fresh, static_argnames=STATIC)


@pytest.mark.parametrize("retry", RULES, ids=["energy_above", "probabilistic"])
def test_power_without_the_ghosts(monkeypatch, retry):
    """The invariance test sees a wrong ghost construction under each rule (all ghost terms dropped; 50,000 chains,
    100 transitions)."""
    global N, K
    saved = N, K
    N, K = 50_000, 100
    try:
        wrong = _wrong(monkeypatch, "log_a = jnp.minimum(0.0, -dh + ghost_terms_from(proposal_tree(k)[1], dhs, energy(y).dtype) - s_x)",
                       "log_a = jnp.minimum(0.0, -dh)", f"no_ghosts_{retry[0]}")
        pos, _ = _run(wrong, "edge", retry)
    finally:
        N, K = saved
    p, z = _check("edge", pos)
    assert p < 1e-6 or abs(z) > 8, (p, z)


def test_power_without_the_ghosts_threshold(monkeypatch):
    """{"energy_above": c}: the ghosts' indicator p dropped (a ghost retried whatever its energy error) is seen on the
    funnel with a large step (50,000 chains, 100 transitions: KS p about 1e-5; the correct rule's p over 8 seeds was
    0.17-0.94). On the hard edge the reverse proposals nearly always diverge, so the indicator is 1 there anyway.

    The probabilistic rule's p factor has no such test: dropping it, at the ghosts, at the current state or both, moved
    the KS p to 0.005-0.13 at 200,000 chains, against 0.05-0.69 for the correct rule. Its exactness rests on eq. (30)
    and on the invariance tests above. For an indicator rule p at the current state is 1 wherever a retry is made, so
    leaving it out changes nothing (not a test either)."""
    global N, K
    saved = N, K
    N, K = 50_000, 100
    try:
        wrong = _wrong(monkeypatch, "total = total + _log1m_exp(log_a) + log_p(dh, log_a)",
                       "total = total + _log1m_exp(log_a)", "no_ghost_threshold")
        pos, _ = _run(wrong, "funnel", ("energy_above", 2.0), eps=1.5)
    finally:
        N, K = saved
    p, _ = _check("funnel", pos)
    assert p < 1e-3, p


def test_divergent_given_is_the_default():
    a, ia = _run(compiled.sampling_chunk, "edge", "divergent")
    ld = compiled.plain_log_density(_edge_logp, ["x"])
    state = compiled.batch_init(ld.fn, ld.names, ld.values,
                                {"x": jnp.asarray(EDGE.rvs(N, random_state=np.random.default_rng(0)))}, 0)
    s, (_, ib) = compiled.sampling_chunk(ld.fn, ld.names, ld.values, state, jnp.ones(1), 1.2, 3.0, jax.random.key(7),
                                         np.int32(0), np.int32(K), chunk=K, jitter_amount=1.0, dr_factor=4)
    assert np.array_equal(a["x"], np.asarray(s.position["x"]))
    assert np.array_equal(np.asarray(ia.retried), np.asarray(ib.retried))


def test_retry_uniforms_have_their_own_stream():
    key = jax.random.key(3)
    first = [np.asarray(jax.random.key_data(k)) for k in jax.random.split(key, 2)]
    acc = [np.asarray(jax.random.key_data(jax.random.fold_in(key, compiled.DR_STREAM + k))) for k in range(2, 8)]
    for k in range(2, 8):
        own = np.asarray(jax.random.key_data(jax.random.fold_in(key, compiled.DR_RETRY_STREAM + k)))
        assert not any(np.array_equal(own, f) for f in first + acc), k


def _sampler():
    import pymc as pm
    with pm.Model() as m:
        pm.Normal("x")
    s = PfCheesHMC(m, seed=1)
    s._resolved = {}
    return s


@pytest.mark.parametrize("bad", [{"factor": 4, "retry": "always"}, {"factor": 4, "retry": {"energy_above": 0}},
                                 {"factor": 4, "retry": {"energy_above": "2"}}, {"factor": 4, "retry": {"above": 2}}])
def test_option_is_checked(bad):
    with pytest.raises(ValueError):
        _sampler()._run_dynamic_hmc(None, {"step_size": 0.1, "frozen_L_nominal": 4.0}, np.ones(1),
                                    chains=2, draws=1, delayed_rejection=bad, progressbar=False)


def test_recorded_in_config():
    import pymc as pm
    from pfcheeshmc import sample_pf_chees_hmc
    with pm.Model() as m:
        pm.Normal("x", shape=2)
    kw = dict(draws=5, chains=8, random_seed=1, progressbar=False, chees_kwargs={"draws": 10})
    cfg = json.loads(sample_pf_chees_hmc(m, **kw, dynamic_hmc_kwargs={"delayed_rejection": {
        "factor": 4, "retry": {"energy_above": 3}}}).attrs["config"])
    assert cfg["delayed_rejection"] == {"factor": 4, "stages": 2, "retry": {"energy_above": 3.0}}
    cfg = json.loads(sample_pf_chees_hmc(m, **kw, dynamic_hmc_kwargs={"delayed_rejection": {
        "factor": 4, "retry": "divergent"}}).attrs["config"])
    assert cfg["delayed_rejection"] == {"factor": 4, "stages": 2}
