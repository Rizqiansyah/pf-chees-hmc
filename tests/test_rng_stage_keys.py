"""Tests for per-stage RNG key derivation in `PfCheesHMC`.

Failure mode this guards: warmup derived its key as `split(jax.random.key(seed))[1]` and the
sampling stage derived `split(jax.random.key(seed))[1]` from the same parent, so the two stages
shared one key. Nothing raised, nothing looked wrong, and the posterior was still valid -- the
sampling stream simply replayed the warmup stream's child, which makes seed-to-seed behaviour
couple to warmup length. The fix scopes each stage with `jax.random.fold_in(key(seed), N)`.

These are structural on purpose: the bug is invisible in the output, so the test asserts the
shape of the derivation and then verifies the derived key streams really are disjoint.
"""
import inspect
import re

import jax
import numpy as np
import pymc as pm

from pfcheeshmc.sampler import PfCheesHMC

_STAGE_RE = re.compile(r"fold_in\(jax\.random\.key\(kwargs\[\"seed\"\]\),\s*(\d+)\)")


def _stage_keys():
    """Stage numbers read back out of the source, so the test cannot drift from the code."""
    stages = {}
    for name in ("_run_chees", "_run_dynamic_hmc"):
        src = inspect.getsource(getattr(PfCheesHMC, name))
        hits = _STAGE_RE.findall(src)
        assert len(hits) == 1, f"{name}: expected one fold_in stage key, found {hits}"
        stages[name] = int(hits[0])
    return stages


def _bits(key, n=8):
    return np.asarray(jax.random.bits(key, shape=(n,), dtype="uint32"), dtype=np.uint64)


def test_warmup_and_sampling_use_different_stages():
    stages = _stage_keys()
    assert stages["_run_chees"] != stages["_run_dynamic_hmc"], stages


def test_derived_key_streams_are_disjoint():
    """The actual collision shape: warmup key vs the sampling stage's first child key."""
    stages = _stage_keys()
    seed = 42
    key_warm = jax.random.fold_in(jax.random.key(seed), stages["_run_chees"])
    rng = jax.random.fold_in(jax.random.key(seed), stages["_run_dynamic_hmc"])
    _, key_sample = jax.random.split(rng)
    assert not np.array_equal(_bits(key_warm), _bits(key_sample))


def test_no_hardcoded_probe_key_in_sampling_stage():
    """The field-discovery probe used jax.random.key(0) regardless of the run seed."""
    src = inspect.getsource(PfCheesHMC._run_dynamic_hmc)
    assert "jax.random.key(0)" not in src


def test_seed_changes_the_sampled_chain():
    """End-to-end sanity on CPU: a different seed must give a different realisation."""
    with pm.Model() as model:
        pm.Normal("x", shape=(2,))

    def first_draw(seed):
        import pfcheeshmc

        idata = pfcheeshmc.sample_pf_chees_hmc(
            model, draws=4, chains=2, random_seed=seed,
            chees_kwargs={"draws": 2}, pathfinder_kwargs={"num_paths": 2, "draws": 8},
        )
        return float(np.asarray(idata.posterior["x"].values).ravel()[0])

    assert first_draw(11) != first_draw(12)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} PASS")
