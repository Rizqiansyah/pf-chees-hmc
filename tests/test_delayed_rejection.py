"""Delayed rejection on divergence (opt-in, `dynamic_hmc_kwargs={"delayed_rejection": a}`).

Modi, Barnett & Carpenter (2023), eq. (30) with p2 = 1{first proposal diverged}. Guards:
  1. invariance: chains started at exact draws of a target with a hard edge, run with a step large enough that
     divergences are frequent, keep the exact law; the same test with the ghost terms dropped fails (power);
  2. off by default: the option absent gives the sampler's own kernel (no retry fields, no config key);
  3. the option is checked (an integer >= 2).
CPU, about a minute.
"""
import inspect
import textwrap
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st

from pfcheeshmc import compiled
from pfcheeshmc.sampler import PfCheesHMC

N, K, EPS, L = 20_000, 20, 1.2, 3.0
EXACT = st.gamma(3.7)


def _logp(pos):
    x = pos["x"]
    return jnp.where(x > 0, 2.7 * jnp.log(jnp.abs(x) + 1e-300) - x, -jnp.inf)


def _final_draws(sampling_chunk, dr):
    ld = compiled.plain_log_density(_logp, ["x"])
    x0 = EXACT.rvs(N, random_state=np.random.default_rng(0))
    state = compiled.batch_init(ld.fn, ld.names, ld.values, {"x": jnp.asarray(x0)}, 0)
    s, (_, info) = sampling_chunk(ld.fn, ld.names, ld.values, state, jnp.ones(1), EPS, L, jax.random.key(7),
                                  np.int32(0), np.int32(K), chunk=K, jitter_amount=1.0, dr_factor=dr)
    return np.asarray(s.position["x"]), info


def _tail_z(x, q=0.005):
    """Share of draws below the exact q-quantile, in binomial standard errors (the edge is where errors show)."""
    return (np.mean(x <= EXACT.ppf(q)) - q) / np.sqrt(q * (1 - q) / len(x))


def test_invariance_with_frequent_retries():
    x, info = _final_draws(compiled.sampling_chunk, 4)
    assert int(np.sum(info.retry_accepted)) > 1000          # the retry branch really ran and moved chains
    assert np.array_equal(np.asarray(info.retried), np.asarray(info.is_divergent & ~info.is_accepted))
    assert st.kstest(x, EXACT.cdf).pvalue > 1e-3
    assert abs(_tail_z(x)) < 4


def test_invariance_test_has_power(monkeypatch):
    src = textwrap.dedent(inspect.getsource(compiled.delayed_rejection_step))
    bad = src.replace("log_a = (-dh_y + _log1m_accept(dh_g) - jnp.log1p(-p1)\n"
                      "                     + jnp.where(dh_g > divergence_threshold, 0.0, -jnp.inf))", "log_a = -dh_y")
    assert bad != src
    ns = dict(vars(compiled))
    exec(bad, ns)
    # sampling_chunk looks the step up in the module at trace time; a new function object (jit caches by
    # identity) traces with the wrong one
    monkeypatch.setattr(compiled, "delayed_rejection_step", ns["delayed_rejection_step"])
    f = compiled.sampling_chunk.__wrapped__
    fresh = types.FunctionType(f.__code__, f.__globals__, "wrong_chunk", f.__defaults__, f.__closure__)
    fresh.__kwdefaults__ = f.__kwdefaults__
    wrong_chunk = jax.jit(fresh, static_argnames=("fn", "names", "chunk", "jitter_amount", "dr_factor"))
    x, _ = _final_draws(wrong_chunk, 4)
    assert _tail_z(x) > 10      # the wrong rule over-weights the edge (z about 30 here)


def test_off_by_default_has_no_retry_fields():
    x, info = _final_draws(compiled.sampling_chunk, None)
    assert info.retried is None and info.retry_accepted is None


@pytest.mark.parametrize("bad", [1, 0, 2.5, True, "4"])
def test_option_is_checked(bad):
    import pymc as pm
    with pm.Model() as m:
        pm.Normal("x")
    sampler = PfCheesHMC(m, seed=1)
    sampler._resolved = {}
    with pytest.raises((ValueError, TypeError)):
        sampler._run_dynamic_hmc(None, {"step_size": 0.1, "frozen_L_nominal": 4.0}, np.ones(1),
                                 chains=2, draws=1, delayed_rejection=bad, progressbar=False)
