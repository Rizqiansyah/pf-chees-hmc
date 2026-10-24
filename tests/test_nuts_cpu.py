"""The noise floor is squared, and the CPU-NUTS driver (`benchmarks/nuts_cpu.py`) scores like the GPU rows.

An earlier NUTS driver compared b2 with MCSE/sd (not squared) and scored constrained `x__` keys. These
tests pin the instrument it must share with the gate: `bias.score_ensemble`'s floor on i.i.d. draws, the same
number through `nuts_cpu`'s scoring entry point, and the pieces of the driver that a wrong answer
would silently corrupt (the sampler settings, the value-var names, the gradient count).
No reference files are needed; the one sampling test uses a 2-variable toy model.
"""

import pathlib
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks import bias as rb  # noqa: E402
from benchmarks import nuts_cpu as t3  # noqa: E402

C, D, DIM = 256, 40, 5


def _iid(seed):
    return np.random.default_rng(seed).normal(size=(C, D, DIM))


def _ref():
    return {"x": np.random.default_rng(99).normal(size=(80_000, DIM))}


def _expected_floor():
    """2 ln(2d) / (C*D) for i.i.d. N(0,1): mcse^2 = 1/(C*D) and var_ref = 1 for x; for x^2 both are 2."""
    return 2.0 * np.log(2.0 * DIM) / (C * D)


def _idata(unc, tuned=None):
    """InferenceData-alike with PyMC's group layout: posterior = sampling phase only."""
    import arviz as az

    ds = xr.Dataset({k: (("chain", "draw", f"{k}_dim_0"), v) for k, v in unc.items()})
    groups = {"posterior": ds}
    if tuned is not None:
        groups["warmup_posterior"] = tuned
    return az.InferenceData(**groups)


# ---------------------------------------------------------------- floor regression (plan test 1)

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_floor_is_squared_through_score_ensemble(seed):
    """Within +-30% of 2 ln(10)/(C*D) in SQUARED units. The T3 bug (MCSE/sd, unsquared) gives
    2 ln(10)/sqrt(C*D) ~ 0.0455, about 100x this, so the band cannot pass it."""
    sc = rb.score_ensemble({"x": _iid(seed)}, _ref(), C, D)
    exp = _expected_floor()
    assert 0.7 * exp <= sc["b2_floor"] <= 1.3 * exp, (sc["b2_floor"], exp, sc["b2_floor"] / exp)
    assert not 0.7 * exp <= 2.0 * np.log(2.0 * DIM) / np.sqrt(C * D) <= 1.3 * exp   # band has teeth


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_floor_is_squared_through_the_nuts_scorer(seed):
    """The same check through nuts_cpu's entry point, starting from an InferenceData."""
    sc = t3.score_idata(_idata({"x": _iid(seed)}), ["x"], _ref(), C, D)
    exp = _expected_floor()
    assert 0.7 * exp <= sc["b2_floor"] <= 1.3 * exp, (sc["b2_floor"], exp, sc["b2_floor"] / exp)
    assert sc["g_r1"] == "pass"            # b2 ~ floor ~ 5e-4 on identical distributions


def test_nuts_scorer_calls_score_ensemble_and_g_r1_not_its_own_scorer(monkeypatch):
    seen = {}

    def fake_score(unc, ref, chains, draws):
        seen["score"] = (sorted(unc), chains, draws)
        return {"b2_max": 0.5, "b2_floor": 0.123, "d": 5, "b2_argmax": "x", "labels": ["x"]}

    monkeypatch.setattr(rb, "score_ensemble", fake_score)
    monkeypatch.setattr(rb, "g_r1", lambda b2, floor, cd: seen.setdefault("gate", (b2, floor, cd)) and "G")
    sc = t3.score_run({"x": _iid(0)}, _ref(), C, D)
    assert seen["score"] == (["x"], C, D)
    assert seen["gate"] == (0.5, 0.123, C * D)
    assert (sc["b2_floor"], sc["g_r1"]) == (0.123, "G")


# ---------------------------------------------------------------- driver pieces

def test_sample_kwargs_are_the_planned_cpu_nuts_settings():
    kw = t3.sample_kwargs(draws=100, tune=200, seed=7)
    assert (kw["chains"], kw["cores"], kw["target_accept"]) == (32, 32, 0.8)
    assert kw["discard_tuned_samples"] is False
    assert kw["idata_kwargs"] == {"include_transformed": True}
    assert (kw["draws"], kw["tune"], kw["random_seed"]) == (100, 200, 7)


def test_value_var_names_are_scored_from_the_posterior_and_a_missing_one_is_an_error():
    unc = {"tau_log__": np.zeros((4, 6, 1)), "theta": np.zeros((4, 6, 3))}
    tuned = xr.Dataset({"tau_log__": (("chain", "draw", "t"), np.full((4, 9, 1), 99.0))})
    got = t3.unconstrained_from_idata(_idata(unc, tuned), ["tau_log__", "theta"], 4, 6)
    assert got["tau_log__"].shape == (4, 6, 1) and got["theta"].shape == (4, 6, 3)
    assert np.all(got["tau_log__"] == 0.0)                      # not the 99s in warmup_posterior
    with pytest.raises(KeyError, match="include_transformed"):
        t3.unconstrained_from_idata(_idata({"theta": unc["theta"]}), ["tau_log__", "theta"], 4, 6)
    with pytest.raises(ValueError, match="tuned draws leaking"):
        t3.unconstrained_from_idata(_idata(unc), ["theta"], 4, 7)


def test_gradients_include_tuning_and_are_exact_per_chain():
    import arviz as az

    ss = xr.Dataset({"n_steps": (("chain", "draw"), np.full((2, 5), 3.0))})
    ws = xr.Dataset({"n_steps": (("chain", "draw"), np.full((2, 7), 4.0))})
    idata = az.InferenceData(sample_stats=ss, warmup_sample_stats=ws)
    g = t3.grads_per_chain(idata, chains=2, tune=7, draws=5)
    assert (g["warmup"], g["sampling"], g["total"]) == (28.0, 15.0, 43.0)
    with pytest.raises(RuntimeError, match="omit tuning"):
        t3.grads_per_chain(az.InferenceData(sample_stats=ss), chains=2, tune=7, draws=5)


def test_run_cell_scores_the_transformed_coordinate_and_counts_tuning_gradients():
    """Real (tiny) pm.sample through the driver's own path, 2 chains, sequential."""
    import pymc as pm

    with pm.Model() as m:
        pm.Normal("x", 0.0, 1.0)
        pm.HalfNormal("s", 1.0)
    names = [v.name for v in m.value_vars]
    assert names == ["x", "s_log__"]
    rng = np.random.default_rng(5)
    ref = {"x": rng.normal(size=(20_000, 1)), "s_log__": np.log(np.abs(rng.normal(size=(20_000, 1))))}
    row = t3.run_cell(m, "toy", names, ref, tune=60, draws=40, seed=3, chains=2, cores=1)
    assert row["error"] is None, row["error"]
    assert row["scored_vars"] == ["s_log__", "x"] and row["d"] == 2
    assert row["schema"] == "pf-nuts/v1" and row["target_accept"] == 0.8
    assert row["tune"] == 60 and row["draws"] == 40 and row["chains"] == 2
    assert row["grads_per_chain_total"] == pytest.approx(
        row["grads_per_chain_warmup"] + row["grads_per_chain_sampling"])
    assert row["grads_per_chain_warmup"] > row["grads_per_chain_sampling"] * 0.5 > 0
    assert row["g_r1"] == "not eligible (C*D < 2000)" and np.isfinite(row["b2_floor"])
    assert row["wall_s"] > 0 and row["core_seconds"] > 0
    assert len(row["loadavg_start"]) == 3 and len(row["loadavg_end"]) == 3
