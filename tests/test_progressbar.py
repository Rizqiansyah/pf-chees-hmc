"""Tests for the `progressbar` flag.

Two reasons this is not cosmetic:

1. The bars are updated by the host between the chunks of the warmup and sampling loops, from each chunk's
   own outputs. No `jax.debug.callback` is traced into either program: a jaxpr containing a host callback
   cannot be written to JAX's on-disk compilation cache, and a cached program would call the FIRST call's
   bar. (tests/test_chunked_loops.py checks the jaxprs and the bar values; tests/test_compile_once.py checks
   that on and off run the same programs.)
2. With the bar off nothing is created or printed, not even a hidden bar.

Tests assert the user-visible contract holds both ways, and that the bars really are absent when
asked -- a flag that silently left a bar on would pass a results-only test.
"""
import arviz as az
import pymc as pm
import pytest

import pfcheeshmc


def _fit(capsys, progressbar=True, chees=None, dyn=None):
    """One tiny fit; returns (idata, captured_stderr). chees / dyn override the per-stage flags."""
    with pm.Model() as model:
        pm.Normal("x", shape=(3,))
        idata = pfcheeshmc.sample_pf_chees_hmc(
            model, draws=15, chains=2, random_seed=7,
            chees_kwargs={"draws": 6, **(chees or {})},
            dynamic_hmc_kwargs=dict(dyn or {}),
            progressbar=progressbar)
        err = capsys.readouterr().err
    return idata, err


def test_default_shows_both_bars(capsys):
    _, err = _fit(capsys)
    assert "ChEES Adaptation" in err
    assert "Sampling Dynamic HMC" in err


def test_off_is_silent(capsys):
    _, err = _fit(capsys, progressbar=False)
    assert "ChEES Adaptation" not in err
    assert "Sampling Dynamic HMC" not in err


def test_contract_identical_both_ways(capsys):
    on, _ = _fit(capsys)
    off, _ = _fit(capsys, progressbar=False)
    assert sorted(on.sample_stats.data_vars) == sorted(off.sample_stats.data_vars)
    for name in on.sample_stats.data_vars:
        assert on.sample_stats[name].shape == off.sample_stats[name].shape
    assert az.rhat(off)["x"].max() < 2.0          # sanity only: 15 draws x 2 chains proves nothing


def test_per_stage_override_wins(capsys):
    # Sampling bar off, warmup bar left on through the explicit bucket.
    _, err = _fit(capsys, progressbar=False, dyn={"progressbar": False}, chees={"progressbar": True})
    assert "ChEES Adaptation" in err
    assert "Sampling Dynamic HMC" not in err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
