"""Tests for the chunked warmup stop rule.

The rule exists because warmup is now the largest stage (48% of a radon run at 2,000 steps) and the
number of steps is a constant that has nothing to do with the model. Three properties must hold:

1. Absent stop_rule, behaviour is byte-for-byte the previous path -- same keys, one scan.
2. An impossible threshold must never truncate. A stop rule that stops short when it is not
   satisfied is worse than no stop rule: it silently ships under-adapted hyperparameters.
3. Stopping early must leave a usable state, with and without the progress bar.

The gate is split-R-hat plus adapter quietness, because adapter stability alone declared the radon
centered funnel converged while its trajectory was frozen at 4.4 leapfrog steps.
"""
import jax
import jax.numpy as jnp
import optax

from pfcheeshmc.utils import chees_warmup_with_progress


def _run(num_steps, stop_rule=None, progressbar=False, d=4, chains=8):
    ld = lambda p: -0.5 * jnp.sum(p["x"] ** 2)
    positions = {"x": jnp.full((chains, d), 0.3)}
    imm = jnp.diag(jnp.ones(d))
    (states, params), _info = chees_warmup_with_progress(
        ld, jax.random.key(0), positions, jnp.float64(0.2), optax.adam(0.05), num_steps, imm,
        progressbar=progressbar, stop_rule=stop_rule)
    return states, params


def test_no_stop_rule_runs_every_step():
    _, params = _run(300)
    assert params["warmup_steps_used"] == 300


def test_easy_target_stops_early():
    """The gate is split-R-hat over a 250-draw window plus step-size quietness.

    Both parts were measured: a 50-draw window reads rhat 1.035-1.058 on a unit Normal forever, and
    the harmonic-mean acceptance reads ~0.0003 for a well-tuned sampler because one fully rejected
    proposal zeroes it. Neither is a convergence signal at that window size, which is why the window
    defaults to the whole block and acceptance is opt-in only.
    """
    _, params = _run(1000, {"block": 250, "window": 250, "rhat": 1.01})
    used = params["warmup_steps_used"]
    assert 250 <= used < 1000, f"expected an early stop, used={used}"


def test_window_too_short_never_stops():
    """Guards the reason the default window is the full block: at 50 draws nothing passes."""
    _, params = _run(1000, {"block": 250, "window": 50, "rhat": 1.01})
    assert params["warmup_steps_used"] == 1000, "criteria are inert today; see docstring"


def test_blocked_scan_does_not_restart_adaptation():
    """Block resumption must not rewind the adapter or the Halton sequence.

    A single 400-step scan and four resumed 100-step blocks use different random streams (blocks
    re-derive keys with fold_in), so exact equality is not the property. The property is that a
    restarted block leaves the adapter where a continuous run would: if the step index were reset,
    jitter would replay from zero and the tuned step size would land far away.
    """
    _, single = _run(400)
    _, blocked = _run(400, {"block": 250, "window": 250, "rhat": 0.999})   # never fires -> full run
    eps_single = float(jnp.mean(single["step_size"]))
    eps_blocked = float(jnp.mean(blocked["step_size"]))
    assert 0.5 < eps_blocked / eps_single < 2.0, f"{eps_single=} {eps_blocked=}"
    assert blocked["warmup_steps_used"] == 400


def test_impossible_threshold_never_truncates():
    _, params = _run(400, {"block": 200, "window": 200, "rhat": 0.999})
    assert params["warmup_steps_used"] == 400
def test_early_stop_leaves_a_usable_state():
    states, params = _run(1000, {"block": 250, "window": 250, "rhat": 1.01}, progressbar=True)
    assert params["warmup_steps_used"] < 1000
    assert jnp.all(jnp.isfinite(states.logdensity))
    eps = jnp.atleast_1d(params["step_size"])
    assert jnp.all(jnp.isfinite(eps)) and jnp.all(eps > 0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} PASS")
