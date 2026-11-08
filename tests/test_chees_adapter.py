"""CPU tests for the ChEES adapter (`compiled._adapter`: BlackJAX's `base` plus a per-step trace).

The load-bearing one is test_the_adapter_state_is_bit_for_bit_upstream: it feeds the adapter and
BlackJAX's base() the same deterministic inputs for 64 updates and requires every state field to
match exactly, so the trace beside the state changes nothing in it.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

jax.config.update("jax_enable_x64", True)

import blackjax.adaptation.chees_adaptation as upstream_chees

from pfcheeshmc.compiled import ChEESTrace, _adapter, chees_harmonic_acceptance, halton_jitter


def _inputs(n_steps, n_chains=8, dim=3, seed=0):
    """Deterministic per-update inputs for the adapter.

    Acceptances are in (0.2, 0.98] with a few divergent chains mixed in; the
    positions/momentums are smooth functions of the step index so the ChEES
    gradient is non-trivial and time-varying.
    """
    rng = np.random.default_rng(seed)
    for t in range(n_steps):
        phase = t * 0.37
        acc = rng.uniform(0.2, 0.98, size=n_chains)
        div = np.zeros(n_chains, dtype=bool)
        if t % 5 == 0:
            div[0] = True
            acc[0] = 0.0  # what a rejected/divergent trajectory looks like
        pos_prop = {
            "z": (0.5 * np.sin(phase + np.arange(dim)[None, :]) * np.tile(
                (1 + 0.1 * np.arange(n_chains))[:, None], (1, dim)
            ) + 0.01 * t).astype(np.float64)
        }
        pos_init = {
            "z": (pos_prop["z"] - 0.05 * np.cos(phase) + 0.02 * np.arange(n_chains)[:, None]).astype(
                np.float64
            )
        }
        mom = {
            "z": (0.1 * np.cos(phase + 2 * np.arange(dim)[None, :]) * np.tile(
                (1 + 0.1 * np.arange(n_chains))[:, None], (1, dim)
            )).astype(np.float64)
        }
        yield (
            {k: jnp.asarray(v) for k, v in pos_prop.items()},
            {k: jnp.asarray(v) for k, v in mom.items()},
            {k: jnp.asarray(v) for k, v in pos_init.items()},
            jnp.asarray(acc),
            jnp.asarray(div),
        )


def _state_leaves(state):
    return jax.tree_util.tree_map(jnp.asarray, state)


def _assert_states_equal(a, b, label):
    la, lb = _state_leaves(a), _state_leaves(b)
    for name, (xa, xb) in zip(
        jax.tree_util.tree_structure(la).flatten_up_to(la),
        zip(jax.tree_util.tree_leaves(la), jax.tree_util.tree_leaves(lb)),
    ):
        assert xa.shape == xb.shape, f"{label}: shape mismatch {xa.shape} vs {xb.shape}"
        if xa.size == 1 and xa.shape == ():
            assert float(xa) == float(xb), f"{label}: scalar {float(xa)} != {float(xb)}"
        else:
            assert bool(jnp.all(jnp.equal(xa, xb))), f"{label}: arrays differ"


def _upstream():
    """BlackJAX's base() with the adapter's jitter (Halton, jitter_amount 1) and Adam(0.05)."""
    return upstream_chees.base(lambda i: halton_jitter(i, 1.0), lambda i: i + 1, optax.adam(0.05), 0.651, 0.5, 1000)


def test_the_adapter_state_is_bit_for_bit_upstream():
    imm = jnp.array([4.0, 1.0, 0.25], dtype=jnp.float64)
    init_up, upd_up = _upstream()
    init_my, upd_my = _adapter(0.05, 0.651, 0.5, 1.0, 1000)

    su = init_up(jnp.int32(0), jnp.float64(0.1))
    sm = init_my(jnp.int32(0), jnp.float64(0.1))
    _assert_states_equal(su, sm, "init")

    for t, (pp, pm, pi, acc, div) in enumerate(_inputs(64)):
        su = upd_up(su, pp, pm, pi, acc, div, imm)
        sm, trace = upd_my(sm, pp, pm, pi, acc, div, imm)
        _assert_states_equal(su, sm, f"step {t}")
        assert isinstance(trace, ChEESTrace)
        assert int(trace.step) == int(su.step)
        assert float(trace.step_size) == float(su.step_size)
        assert float(trace.trajectory_length) == float(su.trajectory_length)
        assert float(trace.harmonic_acceptance) == float(chees_harmonic_acceptance(acc, div))
        assert 0.0 <= float(trace.harmonic_acceptance) <= 1.0


def test_learning_rate_matches_optax_adam():
    """A learning rate builds optax.adam(lr): the same states as passing optax.adam(0.05) itself."""
    init_a, upd_a = _adapter(0.05, 0.651, 0.5, 1.0, 1000)
    init_b, upd_b = _adapter(optax.adam(0.05), 0.651, 0.5, 1.0, 1000)
    sa, sb = init_a(jnp.int32(0), 0.1), init_b(jnp.int32(0), 0.1)
    for pp, pm, pi, acc, div in _inputs(5):
        imm = jnp.ones(3, dtype=jnp.float64)
        sa, _ = upd_a(sa, pp, pm, pi, acc, div, imm)
        sb, _ = upd_b(sb, pp, pm, pi, acc, div, imm)
    _assert_states_equal(sa, sb, "learning rate")


def test_harmonic_pooling_excludes_divergent_chains():
    """A divergent chain contributes nothing to the step-size adaptation, no
    matter its acceptance value: (acc=0.5, div=True) and (acc=0.0, div=True)
    must drive the step-size state identically."""
    init_my, upd_my = _adapter(0.05, 0.651, 0.5, 1.0, 1000)
    imm = jnp.ones(3, dtype=jnp.float64)

    results = []
    for acc_val in (0.5, 0.0):
        s = init_my(jnp.int32(0), 0.1)
        for pp, pm, pi, acc, div in _inputs(10):
            acc = jnp.where(div, jnp.asarray(acc_val, dtype=acc.dtype), acc)
            s, tr = upd_my(s, pp, pm, pi, acc, div, imm)
        results.append(s)
    # The two runs differ only in the divergent chain's acceptance value, so
    # every state field must be identical at the end.
    _assert_states_equal(results[0], results[1], "divergent exclusion")

    # And step by step, including the trace's harmonic acceptance:
    runs = []
    for acc_val in (0.5, 0.0):
        s = init_my(jnp.int32(0), 0.1)
        trace = []
        for pp, pm, pi, acc, div in _inputs(10):
            acc = jnp.where(div, jnp.asarray(acc_val, dtype=acc.dtype), acc)
            s, tr = upd_my(s, pp, pm, pi, acc, div, imm)
            trace.append((float(s.step_size), float(s.trajectory_length), float(tr.harmonic_acceptance)))
        runs.append(trace)
    for i, (a, b) in enumerate(zip(runs[0], runs[1])):
        assert a == b, f"step {i}: divergent-chain acceptance leaked into the state: {a} vs {b}"


def _aniso_gaussian(sd):
    """log-density of an anisotropic 2-D Gaussian, dict-of-arrays pytree.

    Called per chain under jax.vmap, so each leaf is a single position (2,).
    """
    import math
    logdet = math.log(2 * math.pi * sd[0] * sd[1])

    def ld(pos):
        z = jnp.asarray(pos["theta"])
        return -0.5 * (z[0] ** 2 / sd[0] ** 2 + z[1] ** 2 / sd[1] ** 2) - 0.5 * logdet

    return ld


def _run_warmup(n_steps=300, chains=8, seed=0):
    from pfcheeshmc.utils import chees_warmup_with_progress

    sd = (1e-2, 1e2)  # one stiff, one soft direction
    rng = np.random.default_rng(seed)
    pos = {"theta": jnp.asarray(rng.normal(0.0, 1.0, size=(chains, 2)))}
    imm = jnp.asarray(1.0 / np.array(sd) ** 2)  # diagonal inverse mass (d,)
    (states, params), warm = chees_warmup_with_progress(
        _aniso_gaussian(sd),
        jax.random.key(seed),
        pos,
        0.05,
        optax.adam(0.05),
        n_steps,
        imm,
        progressbar=False,
        max_leapfrog_steps=100,
    )
    tr = warm.trace
    L = tr.trajectory_length / tr.step_size
    return L, tr, params


def test_trace_matches_state_and_counts():
    L, tr, params = _run_warmup(n_steps=50)
    assert tr.step.shape[0] == 50
    assert tr.step_size.shape == (50,)
    assert tr.trajectory_length.shape == (50,)
    assert tr.harmonic_acceptance.shape == (50,)
    assert bool(jnp.all(jnp.isfinite(tr.harmonic_acceptance)))
    assert float(params["warmup_steps_used"]) == 50.0
    # frozen_L_nominal = exp(logT_ma - logEps_ma): same trajectory MA the trace
    # consumed, divided by the step-size MA rather than the raw step size, so
    # on a target where eps is still moving the two can differ by the MA lag.
    # Pin the order of magnitude, not the exact value.
    assert 0.25 * float(L[-1]) <= float(params["frozen_L_nominal"]) <= 4.0 * float(L[-1]) + 1e-6


def test_metric_shape_guard_raises():
    from pfcheeshmc.utils import chees_warmup_with_progress

    rng = np.random.default_rng(0)
    pos = {"theta": jnp.asarray(rng.normal(size=(4, 2)))}
    bad_imm = jnp.ones((3, 3))  # model spans 2 flat coordinates
    with pytest.raises(ValueError, match="flat coordinates"):
        chees_warmup_with_progress(
            _aniso_gaussian((1.0, 1.0)), jax.random.key(0), pos, 0.05,
            optax.adam(0.05), 20, bad_imm, progressbar=False,
        )
    bad_imm1 = jnp.ones(5)
    with pytest.raises(ValueError, match="flat coordinates"):
        chees_warmup_with_progress(
            _aniso_gaussian((1.0, 1.0)), jax.random.key(0), pos, 0.05,
            optax.adam(0.05), 20, bad_imm1, progressbar=False,
        )
