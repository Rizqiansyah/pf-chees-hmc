"""Tests for stratified chain-start selection.

What could silently break: the rank arithmetic running off the end of the subset, duplicates in the
scored log-densities collapsing the picks onto one row, or the pool index being applied to the wrong
axis after the (chains, draws) -> flat change. Each has a test. The behaviour tests use a tiny model
because the property worth checking is "distinct, spread-out starts drawn from the pool", which needs
no expensive fit.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

import pfcheeshmc
from pfcheeshmc.sampler import _stratified_pool_idx


def test_picks_are_distinct_in_range_and_spread_out():
    subset = jnp.arange(1000)
    logp = -jnp.abs(subset - 500).astype(float)      # peak at the middle of the pool
    idx = np.asarray(_stratified_pool_idx(logp, subset, 32))
    assert len(idx) == 32
    assert idx.min() >= 0 and idx.max() < 1000
    assert len(set(idx.tolist())) == 32, "stratified picks must not repeat"
    gaps = np.diff(np.sort(idx))
    assert gaps.max() < 5 * max(gaps.min(), 1), f"picks clustered: gaps {gaps[:5]}"


def test_duplicate_scores_still_give_distinct_picks():
    subset = jnp.arange(64)
    logp = jnp.zeros(64)                              # every draw equally good
    idx = np.asarray(_stratified_pool_idx(logp, subset, 8))
    assert len(set(idx.tolist())) == 8


def test_more_picks_than_subset_rows_stays_in_bounds():
    subset = jnp.arange(8)
    idx = np.asarray(_stratified_pool_idx(-subset.astype(float), subset, 16))
    assert len(idx) == 16
    assert idx.min() >= 0 and idx.max() < 8


def _fit(init_pick):
    with pm.Model() as model:
        pm.Normal("x", shape=(4,))
        return pfcheeshmc.sample_pf_chees_hmc(
            model, draws=30, chains=6, random_seed=3, progressbar=False,
            chees_kwargs={"draws": 10}, pathfinder_kwargs={"draws": 500, "init_pick": init_pick})


def test_stratified_and_iid_both_produce_the_contract():
    for pick in ("stratified", "iid"):
        idata = _fit(pick)
        assert idata.posterior["x"].shape == (6, 30, 4)
        assert set(["acceptance_rate", "diverging", "energy", "is_accepted", "lp",
                    "num_integration_steps"]) <= set(idata.sample_stats.data_vars)


def test_bad_pick_name_raises():
    with pytest.raises(ValueError, match="init_pick"):
        _fit("psis_with_replacement")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} PASS")
