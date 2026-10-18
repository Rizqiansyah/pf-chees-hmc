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
import numpy as np
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


def _az_rhat(x):
    """arviz 0.23 returns a Dataset even from a DataArray; unwrap to a scalar."""
    import xarray as xr
    import arviz as az
    r = az.rhat(xr.DataArray(x, dims=("chain", "draw")))
    if isinstance(r, xr.Dataset):
        r = r.to_array()
    return float(np.asarray(r).ravel()[0])


def test_block_rhat_matches_arviz_and_detects_width_differences():
    """The gate's R-hat must be arviz's R-hat, computed in one call, and sensitive to what the
    funnel actually looks like.

    The width case is the one that matters: a hand-written pooled-z version of split-R-hat passed the
    offset case and read 1.000 where arviz read 1.059-1.086, because the library standardises ranks
    per chain rather than pooled. A gate blind to differing chain widths is blind to radon centered.
    """
    from pfcheeshmc.utils import block_max_rhat

    k = jax.random.split(jax.random.key(7), 4)
    # (window, chains, dim) -- the shape the warmup scan emits
    base = jnp.stack([jax.random.normal(k[i], (120, 8)) for i in range(4)], axis=-1)
    cases = {
        "converged": base,
        "offset chains": base.at[:, 1::2, :].add(0.6),
        "uneven width chains": base.at[:, 1::2, :].multiply(2.0),
    }
    # .at[...].add is required: base[:, 1::2, :] + 0.6 selects only the odd chains and returns an
    # all-offset array, which is exactly what R-hat cannot see (a uniform shift of every chain).
    assert float(block_max_rhat(cases["converged"])) < float(block_max_rhat(cases["offset chains"]))
    for label, W in cases.items():
        mine = block_max_rhat(W)
        # W is (window, chains, dim); arviz wants (chain, draw) -> transpose the slice.
        theirs = max(_az_rhat(np.asarray(W[:, :, j], dtype=np.float64).T) for j in range(4))
        assert abs(mine - theirs) < 1e-3, (label, mine, theirs)
        if label != "converged":
            assert mine > 1.03, f"{label} must not look converged: {mine}"


def test_block_rhat_rejects_inconsistent_windows():
    import pytest
    from pfcheeshmc.utils import block_max_rhat
    with pytest.raises(ValueError, match="shape mismatch"):
        block_max_rhat({"a": jnp.zeros((50, 8)), "b": jnp.zeros((40, 8))})


def test_no_stop_rule_runs_every_step():
    _, params = _run(300)
    assert params["warmup_steps_used"] == 300


def test_easy_target_stops_early():
    """The gate is split-R-hat over the block plus step-size quietness.

    Threshold 1.005 rather than the usual 1.01: our estimator is the plain split-R-hat without
    arviz's small-sample correction, so it reads a little low (see the arviz test above).
    """
    _, params = _run(1000, {"block": 250, "window": 250, "rhat": 1.005})
    used = params["warmup_steps_used"]
    assert 250 <= used < 1000, f"expected an early stop, used={used}"


def test_harmonic_acceptance_is_not_in_the_default_gate():
    """Opting back into the acceptance band must make the gate strictly harder to satisfy.

    Pinned because the band looks reasonable and is not: on a well-tuned unit Normal the harmonic
    mean reads 0.0000-0.0126, since one fully rejected proposal -- routine during warmup -- sends a
    harmonic mean to zero. Keeping it in the gate means warmup always runs to the cap.
    """
    _, free = _run(1000, {"block": 250, "window": 250, "rhat": 1.005})
    _, banded = _run(1000, {"block": 250, "window": 250, "rhat": 1.005,
                            "accept_band": (0.601, 0.701)})
    assert banded["warmup_steps_used"] > free["warmup_steps_used"]


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
    states, params = _run(1000, {"block": 250, "window": 250, "rhat": 1.005}, progressbar=True)
    assert params["warmup_steps_used"] < 1000
    assert jnp.all(jnp.isfinite(states.logdensity))
    eps = jnp.atleast_1d(params["step_size"])
    assert jnp.all(jnp.isfinite(eps)) and jnp.all(eps > 0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} PASS")
