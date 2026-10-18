"""Tests for the ε0 stage: `PfCheesHMC._run_step_size_tune`.

These cover the two failure modes found while reviewing the parallelised rewrite.

1. The search inside `blackjax.adaptation.step_size.find_reasonable_step_size` is a
   `lax.while_loop` whose condition is per-probe. Under `jax.vmap` a batched condition is
   OR-reduced while the body updates every element, so probes that already crossed the
   target acceptance keep doubling/halving and return a value set by the batch's trip
   count rather than their own. Detected here by requiring the result to be invariant to
   removing or reordering unrelated probes.
2. A probe at a position with non-finite log-density yields nan acceptance, which the
   library turns into "halve forever" (`step_size.py:239`), dragging the median to the
   float floor. Detected by feeding one such position.
"""
import numpy as np
import pymc as pm

from pfcheeshmc.sampler import PfCheesHMC


def _stage(n_probes=16, d=5, seed=0):
    rng = np.random.default_rng(seed)
    with pm.Model() as model:
        pm.Normal("x", shape=(d,))
    sampler = PfCheesHMC(model, seed=1234)
    pf_samples = {"x": rng.normal(size=(n_probes, d))}
    return sampler, pf_samples, np.ones(d)


def test_eps0_is_a_dyadic_step_size_in_range():
    sampler, pf, imm = _stage()
    eps0 = sampler._run_step_size_tune(pf, imm, chains=128, seed=1234)
    assert np.isfinite(eps0) and eps0 > 0, eps0
    # values come from 1e-2 * 2**k; the geometric median of such values has 2*log2 in Z
    k = np.log2(eps0 / 1e-2)
    assert abs(2 * k - round(2 * k)) < 1e-4, f"not from the doubling search: {eps0}"
    # one leapfrog step of a unit Normal crossing 0.651 acceptance; the search returns the
    # first value past the target, so it overshoots (d=5 lands on 1.28, d=10 on 0.64)
    assert 0.05 <= eps0 <= 4.0, eps0


def test_eps0_independent_of_which_probes_are_batched():
    sampler, pf, imm = _stage(n_probes=16)
    full = sampler._run_step_size_tune(pf, imm, chains=128, seed=1234)

    order = np.random.default_rng(7).permutation(16)
    shuffled = sampler._run_step_size_tune({"x": pf["x"][order]}, imm, chains=128, seed=1234)
    subset = sampler._run_step_size_tune({"x": pf["x"][:4]}, imm, chains=4, seed=1234)

    for other in (shuffled, subset):
        ratio = max(full, other) / min(full, other)
        assert ratio < 2.0, f"eps0 depends on the batch composition: {full} vs {other}"


def test_eps0_survives_a_nonfinite_probe():
    sampler, pf, imm = _stage(n_probes=8)
    pf["x"][0] = 1e300  # log-density is -inf here
    eps0 = sampler._run_step_size_tune(pf, imm, chains=8, seed=1234)
    assert np.isfinite(eps0) and eps0 > 1e-6, eps0


if __name__ == "__main__":
    for name in sorted(k for k in globals() if k.startswith("test_")):
        globals()[name]()
        print("PASS:", name)
    print("PASS: all step-size tests")
