"""Tests for the `progressbar` flag.

Two reasons this is not cosmetic:

1. Both progress bars are driven by `jax.debug.callback` inside the warmup and sampling scans.
   A jaxpr containing a host callback cannot be written to JAX's on-disk compilation cache, so
   leaving the bar on costs every fresh process a full recompile of those two scans.
2. With the bar off the callback is removed at *trace* time, not skipped at run time, so the
   per-step host round-trip is gone from the compiled code entirely.

Tests assert the user-visible contract holds both ways, and that the bars really are absent when
asked -- a flag that silently leaves a callback in the graph would pass a results-only test.
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
