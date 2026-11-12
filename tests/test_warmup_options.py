"""Opt-in warmup settings: ``chees_kwargs["acceptance_floor"]`` and ``chees_kwargs["trajectory_length"]``.

1. The floor is defined against BlackJAX's own update, bit for bit: the step-size fields are those of the update fed
   max(acceptance, floor), the trajectory-length fields those of the update fed the acceptances, the length clipped
   at the new step size. A floor below every acceptance changes nothing.
2. What the floor is for: a few stuck chains among many no longer pull the step-size statistic to zero; the trace
   still shows them (it records the harmonic mean of the acceptances themselves).
3. A fixed trajectory length: held at the value (clipped to [step size, max_leapfrog_steps x step size] at the start,
   each step and after a metric update's restart) and frozen at it, the frozen L capped at max_leapfrog_steps with a
   warning; the step-size fields are untouched.
4. Another value of either setting compiles nothing.
5. Bad values raise before any stage runs; the settings are recorded in the config only when set.
"""
import json

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pymc as pm
import pytest

jax.config.update("jax_enable_x64", True)

import pathlib
import sys

import blackjax.adaptation.chees_adaptation as upstream_chees

from pfcheeshmc import sample_pf_chees_hmc
from pfcheeshmc.compiled import _adapter, chees_harmonic_acceptance, halton_jitter, restart_step_size
from pfcheeshmc.utils import chees_warmup_with_progress

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.compile_counter import count_compiles  # noqa: E402

from test_chees_adapter import _assert_states_equal, _inputs

STEP_FIELDS = ("step_size", "log_step_size_moving_average", "da_state")


def _upstream(max_leapfrog_steps=1000):
    return upstream_chees.base(lambda i: halton_jitter(i, 1.0), lambda i: i + 1, optax.adam(0.05), 0.651, 0.5,
                               max_leapfrog_steps)


def test_floor_is_blackjax_with_the_floored_step_size_statistic():
    floor = 0.3                                       # inside the inputs' acceptance range (0.2, 0.98]
    imm = jnp.array([4.0, 1.0, 0.25], dtype=jnp.float64)
    init_up, upd_up = _upstream()
    init_my, upd_my = _adapter(0.05, 0.651, 0.5, 1.0, 1000, acceptance_floor=floor)
    ref = init_up(jnp.int32(0), jnp.float64(0.1))
    mine = init_my(jnp.int32(0), jnp.float64(0.1))
    _assert_states_equal(ref, mine, "init")
    changed = 0
    for t, (pp, pm_, pi, acc, div) in enumerate(_inputs(64)):
        raw = upd_up(ref, pp, pm_, pi, acc, div, imm)
        fl = upd_up(ref, pp, pm_, pi, jnp.maximum(acc, floor), div, imm)
        ref = raw._replace(**{f: getattr(fl, f) for f in STEP_FIELDS},
                           trajectory_length=jnp.clip(jnp.exp(raw.log_trajectory_length_moving_average),
                                                      max=1000 * fl.step_size, min=fl.step_size))
        mine, trace = upd_my(mine, pp, pm_, pi, acc, div, imm)
        _assert_states_equal(ref, mine, f"step {t}")
        assert float(trace.harmonic_acceptance) == float(chees_harmonic_acceptance(acc, div))   # the raw statistic
        changed += float(fl.step_size) != float(raw.step_size)
    assert changed > 0, "the floor never bound: the test would not see a wrong field"


def test_floor_below_every_acceptance_changes_nothing():
    imm = jnp.ones(3, dtype=jnp.float64)
    init_a, upd_a = _adapter(0.05, 0.651, 0.5, 1.0, 1000)
    init_b, upd_b = _adapter(0.05, 0.651, 0.5, 1.0, 1000, acceptance_floor=1e-6)
    sa, sb = init_a(jnp.int32(0), 0.1), init_b(jnp.int32(0), 0.1)
    for t, (pp, pm_, pi, acc, div) in enumerate(_inputs(32)):
        sa, ta = upd_a(sa, pp, pm_, pi, acc, div, imm)
        sb, tb = upd_b(sb, pp, pm_, pi, acc, div, imm)
        _assert_states_equal(sa, sb, f"step {t}")
        assert float(ta.harmonic_acceptance) == float(tb.harmonic_acceptance)


def test_floor_holds_the_statistic_up_when_a_few_chains_are_stuck():
    """1,024 chains at acceptance 0.95, two at 1e-30 without diverging: BlackJAX's harmonic mean is ~0, below the
    target; with the floor 0.01 it is 1024 / (1022 / 0.95 + 2 / 0.01) = 0.80, above it. Dual averaging's first step
    starts from 10x the initial step size and moves down for a statistic below the target, up for one above. The
    trace shows the stuck chains either way."""
    n, dim = 1024, 3
    acc = jnp.full(n, 0.95).at[:2].set(1e-30)
    div = jnp.zeros(n, dtype=bool)
    x = jax.random.normal(jax.random.key(0), (n, dim))
    pos, mom = {"z": x}, {"z": jax.random.normal(jax.random.key(1), (n, dim))}
    imm = jnp.ones(dim)
    out = {}
    for floor in (None, 0.01):
        init, upd = _adapter(0.05, 0.651, 0.5, 1.0, 1000, acceptance_floor=floor)
        s0 = init(jnp.int32(0), 0.1)
        s1, tr = upd(s0, pos, mom, {"z": x * 0.9}, acc, div, imm)
        out[floor] = (float(tr.harmonic_acceptance), float(s1.step_size))
    assert out[None][0] < 1e-20 and out[0.01][0] < 1e-20          # the trace: the raw harmonic mean
    assert float(chees_harmonic_acceptance(jnp.maximum(acc, 0.01), div)) == pytest.approx(
        1024 / (1022 / 0.95 + 2 / 0.01), rel=1e-9)
    assert out[None][1] < 1.0 < out[0.01][1]


@pytest.mark.parametrize("length, max_steps", [(2.0, 1000), (20.0, 10)])
def test_fixed_trajectory_length(length, max_steps):
    imm = jnp.array([4.0, 1.0, 0.25], dtype=jnp.float64)
    init_a, upd_a = _adapter(0.05, 0.651, 0.5, 1.0, max_steps)
    init_b, upd_b = _adapter(0.05, 0.651, 0.5, 1.0, max_steps, trajectory_length=length)
    sa, sb = init_a(jnp.int32(0), 0.1), init_b(jnp.int32(0), 0.1)
    assert float(sb.trajectory_length) == float(np.clip(length, 0.1, max_steps * 0.1))   # clipped from the start
    assert float(sb.log_trajectory_length_moving_average) == pytest.approx(np.log(length), abs=0)
    clipped = 0
    for t, (pp, pm_, pi, acc, div) in enumerate(_inputs(32)):
        sa, _ = upd_a(sa, pp, pm_, pi, acc, div, imm)
        sb, tr = upd_b(sb, pp, pm_, pi, acc, div, imm)
        for f in STEP_FIELDS:                         # the step size does not see the length
            _assert_states_equal(getattr(sa, f), getattr(sb, f), f"step {t} {f}")
        eps = float(sb.step_size)
        assert float(sb.trajectory_length) == float(np.clip(length, eps, max_steps * eps)), f"step {t}"
        assert float(sb.log_trajectory_length_moving_average) == float(jnp.log(length))
        assert float(tr.trajectory_length) == float(sb.trajectory_length)
        clipped += length > max_steps * eps
    if max_steps == 10:
        assert clipped > 0, "the cap never bound: the test would not see a missing clip"


def _gauss_ld(pos):
    return -0.5 * jnp.sum((pos["x"] / jnp.array([0.1, 1.0, 3.0])) ** 2)


def _warmup(**kw):
    pos = {"x": jax.random.normal(jax.random.key(0), (64, 3))}
    return chees_warmup_with_progress(_gauss_ld, jax.random.key(1), pos, 0.1, 0.05, 60, jnp.ones(3),
                                      progressbar=False, **kw)


def test_warmup_freezes_the_fixed_length():
    (_, params), warm = _warmup(trajectory_length=1.7)
    eps = np.asarray(warm.trace.step_size)
    np.testing.assert_array_equal(np.asarray(warm.trace.trajectory_length), np.clip(1.7, eps, 1000 * eps))
    # the sampling phase runs L = T / eps at the frozen step size
    assert float(params["traj_len_adj"]) * float(params["step_size"]) == pytest.approx(1.7, rel=1e-12)


def test_frozen_length_is_capped_at_max_leapfrog_steps():
    """A fixed T longer than max_leapfrog_steps x the frozen step size: the draws run max_leapfrog_steps steps."""
    with pytest.warns(RuntimeWarning, match="above max_leapfrog_steps 5"):
        (_, params), warm = _warmup(trajectory_length=50.0, max_leapfrog_steps=5)
    assert float(params["traj_len_adj"]) == 5.0
    assert int(np.max(np.asarray(warm.info.num_integration_steps))) <= 5
    (_, params), _ = _warmup(trajectory_length=0.3, max_leapfrog_steps=5)        # under the cap: unchanged
    assert float(params["traj_len_adj"]) * float(params["step_size"]) == pytest.approx(0.3, rel=1e-12)


def test_restart_clips_a_fixed_length_at_the_new_step_size():
    init, upd = _adapter(0.05, 0.651, 0.5, 1.0, 1000, trajectory_length=4.0)
    s = init(jnp.int32(0), jnp.float64(0.5))
    assert float(s.trajectory_length) == 4.0
    r = restart_step_size(s, 0.001, 4.0, max_leapfrog_steps=1000)               # 4 / 0.001 = 4,000 steps > 1,000
    assert float(r.step_size) == 0.001 and float(r.trajectory_length) == pytest.approx(1.0, rel=1e-12)
    assert float(r.log_trajectory_length_moving_average) == float(s.log_trajectory_length_moving_average)
    r = restart_step_size(s, 0.001)                                              # no fixed length: carried on
    assert float(r.trajectory_length) == 4.0


@pytest.mark.parametrize("key, values", [("trajectory_length", (1.7, 2.3)), ("acceptance_floor", (0.01, 0.02))])
def test_another_value_compiles_nothing(key, values):
    _warmup(**{key: values[0]})
    with count_compiles() as again:
        _warmup(**{key: values[1]})
    assert again.count == 0, again.names


def test_warmup_default_is_unchanged_by_none():
    (_, pa), wa = _warmup()
    (_, pb), wb = _warmup(acceptance_floor=None, trajectory_length=None)
    np.testing.assert_array_equal(np.asarray(wa.trace.step_size), np.asarray(wb.trace.step_size))
    np.testing.assert_array_equal(np.asarray(wa.trace.trajectory_length), np.asarray(wb.trace.trajectory_length))
    assert float(pa["traj_len_adj"]) == float(pb["traj_len_adj"])


@pytest.mark.parametrize("kw, match", [
    ({"acceptance_floor": 0.0}, "acceptance_floor"),
    ({"acceptance_floor": -0.1}, "acceptance_floor"),
    ({"acceptance_floor": 0.651}, "acceptance_floor"),
    ({"acceptance_floor": 0.9}, "acceptance_floor"),
    ({"trajectory_length": 0.0}, "trajectory_length"),
    ({"trajectory_length": -1.0}, "trajectory_length"),
    ({"trajectory_length": float("nan")}, "trajectory_length"),
    ({"trajectory_length": float("inf")}, "trajectory_length"),
    ({"acceptance_floor": True}, "acceptance_floor"),
    ({"acceptance_floor": "0.01"}, "acceptance_floor"),
    ({"acceptance_floor": np.array([0.01, 0.02])}, "acceptance_floor"),
    ({"trajectory_length": True}, "trajectory_length"),
    ({"trajectory_length": "2"}, "trajectory_length"),
])
def test_bad_values_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        _warmup(**kw)
    _sampler_raises(kw, match)


@pytest.mark.parametrize("value", [np.float32(2.5), np.array(2.5), jnp.asarray(2.5), 2])
def test_numbers_of_any_scalar_type_are_accepted(value):
    (_, params), _ = _warmup(trajectory_length=value)
    assert float(params["traj_len_adj"]) * float(params["step_size"]) == pytest.approx(float(value), rel=1e-6)


def _sampler_raises(kw, match):
    with pm.Model() as model:
        pm.Normal("x")
    with pytest.raises(ValueError, match=match):              # before Pathfinder runs
        sample_pf_chees_hmc(model, draws=5, chains=8, random_seed=1, progressbar=False,
                            chees_kwargs={"draws": 5, **kw})


def test_floor_is_checked_against_the_target_given():
    with pm.Model() as model:
        pm.Normal("x")
    with pytest.raises(ValueError, match="acceptance_floor"):
        sample_pf_chees_hmc(model, draws=5, chains=8, random_seed=1, progressbar=False,
                            chees_kwargs={"draws": 5, "target_accept": 0.5, "acceptance_floor": 0.6})


def test_end_to_end_recorded_in_config_and_default_absent():
    with pm.Model() as model:
        pm.Normal("x", 0.0, np.array([0.1, 1.0, 10.0]), shape=3)
    kw = dict(draws=20, chains=16, random_seed=1, progressbar=False)
    idata = sample_pf_chees_hmc(model, **kw, chees_kwargs={"draws": 30, "acceptance_floor": 0.01,
                                                           "trajectory_length": 2.5})
    cfg = json.loads(idata.attrs["config"])
    assert cfg["acceptance_floor"] == 0.01 and cfg["trajectory_length"] == 2.5
    # frozen L is ceil(T / eps) at the frozen step size (either side of an integer, to rounding)
    ratio = 2.5 / idata.attrs["frozen_step_size"]
    assert np.ceil(ratio * (1 - 1e-9)) <= idata.attrs["frozen_L_nominal"] <= np.ceil(ratio * (1 + 1e-9))
    idata = sample_pf_chees_hmc(model, **kw, chees_kwargs={"draws": 30})
    cfg = json.loads(idata.attrs["config"])
    assert "acceptance_floor" not in cfg and "trajectory_length" not in cfg
