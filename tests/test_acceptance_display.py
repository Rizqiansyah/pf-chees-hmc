"""Tests for `chees_harmonic_acceptance`, the number the progress bars display.

Failure mode this guards: both bars showed `jnp.mean(acceptance)` while ChEES adapts the step
size on the harmonic mean over non-divergent chains. On heterogeneous trajectories the
arithmetic mean is systematically higher (Jensen), so a run can look like it is sitting on the
0.651 target while adaptation is driving the step size somewhere else. This misread cost a
diagnosis once already.
"""
import jax
import jax.numpy as jnp

from pfcheeshmc.utils import chees_harmonic_acceptance


def test_matches_blackjax_rule_on_a_mixed_batch():
    acc = jnp.array([0.9, 0.5, 0.1, 0.02])
    div = jnp.array([False, False, False, True])
    # BlackJAX: 1 / mean(1/acc, where=~div) over the three non-divergent chains.
    expected = 1.0 / jnp.mean(1.0 / acc[:3])
    assert float(chees_harmonic_acceptance(acc, div)) == float(expected)


def test_below_the_arithmetic_mean_when_trajectories_differ():
    acc = jnp.array([0.95, 0.30, 0.55, 0.80])
    div = jnp.zeros(4, dtype=bool)
    hm = float(chees_harmonic_acceptance(acc, div))
    assert hm < float(jnp.mean(acc))


def test_equal_acceptances_agree_with_arithmetic():
    acc = jnp.full(8, 0.651)
    div = jnp.zeros(8, dtype=bool)
    assert abs(float(chees_harmonic_acceptance(acc, div)) - 0.651) < 1e-6


def test_all_divergent_is_zero_not_nan():
    acc = jnp.array([0.4, 0.2])
    div = jnp.ones(2, dtype=bool)
    val = float(chees_harmonic_acceptance(acc, div))
    assert jnp.isfinite(val) and val == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} PASS")
