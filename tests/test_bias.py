"""Unit tests of the bias gate, `benchmarks/bias.py` (CPU; the last block runs the real sampler on a 2-D Gaussian).

These pin the parts of the gate that a wrong answer would silently corrupt: the bias statistic, its noise floor, the G-R1
verdict wording, the leaf flattening, the reference reader and its sha256 check against `benchmarks/references.json`, and the
row the gate writes.
"""

import hashlib
import json
import os
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks import bias as rb  # noqa: E402


# ---------------------------------------------------------------- bias + floor

def test_identical_samples_give_zero_bias():
    ref = np.random.default_rng(0).normal(size=(5000, 4))
    ens = ref.copy()
    b2, k, floor, _, _ = rb.bias(ens, ref, np.zeros(4), np.zeros(4))
    assert b2 == pytest.approx(0.0, abs=1e-12)
    assert floor == 0.0


def test_half_sd_shift_in_one_leaf_reads_b2_point_25_at_that_leaf():
    """b2 is (dmu)^2 / var_ref, so a 0.5 sd shift must give 0.25 and must name the leaf."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(40000, 3))
    ens = ref.copy()
    ens[:, 1] += 0.5
    b2, k, _, b2m, _ = rb.bias(ens, ref, np.zeros(3), np.zeros(3))
    assert b2 == pytest.approx(0.25, rel=0.05)
    assert k == 1
    assert b2m[0] < 1e-3 and b2m[2] < 1e-3


def test_second_moment_bias_is_caught_even_with_an_unmoved_mean():
    """A sampler that gets the mean right and the variance wrong must fail G-R1."""
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(40000, 2))
    ens = rng.normal(scale=1.3, size=(40000, 2))
    b2, k, _, b2m, b2v = rb.bias(ens, ref, np.zeros(2), np.zeros(2))
    assert np.max(b2m) < 0.01              # means are fine
    assert np.max(b2v) > 0.05              # x^2 is not
    assert k >= 2                          # the argmax lives in the x^2 half of the index space


def test_floor_is_the_mc_noise_of_b2_itself():
    """Floor = max_f mcse_f^2 / var_ref_f * 2 ln(2d). Independent draws, C*D = 10,000, d=10 gives
    ~6e-4; multiply by the integrated autocorrelation time and a 10x-correlated run sits near 0.006,
    which is why plan §6 refuses to call b2_max < 0.01 a pass without printing the floor."""
    d, n = 10, 10_000
    ref = np.random.default_rng(3).normal(size=(n, d))
    ens = np.random.default_rng(4).normal(size=(n, d))
    mcse = ref.std(0, ddof=1) / np.sqrt(n)
    _, _, floor, _, _ = rb.bias(ens, ref, mcse, mcse)
    assert floor == pytest.approx(float(np.max(mcse ** 2 / ref.var(0, ddof=1))
                                        * 2.0 * np.log(2.0 * d)), rel=1e-9)
    assert 1e-4 < floor < 0.01


# ---------------------------------------------------------------- G-R1 wording

def test_g_r1_refuses_to_call_an_unresolvable_cell_a_pass():
    assert rb.g_r1(0.0005, 0.02, 1024 * 10).startswith("not resolvable")
    assert rb.g_r1(0.0005, 0.001, 1024 * 10) == "pass"
    assert rb.g_r1(0.05, 0.001, 1024 * 10) == "fail"
    assert rb.g_r1(0.0005, 0.001, 128 * 10) == "not eligible (C*D < 2000)"


def test_a_coarse_floor_does_not_neutralise_a_bias_far_above_it():
    """The old rule returned 'not resolvable' on floor > 0.005 without reading b². Measured
    consequence: 6 radon cells, worst b² 63.23 against a floor of 0.00558, read neutral."""
    assert rb.g_r1(63.23, 0.00558, 1024 * 10) == "fail"
    assert rb.g_r1(0.4279, 0.00528, 1024 * 10) == "fail"
    # a bias inside its own noise band stays unresolvable, which is not a pass
    assert rb.g_r1(0.00877, 0.00544, 1024 * 10).startswith("not resolvable")
    assert rb.g_r1(0.01364, 0.00559, 1024 * 10).startswith("not resolvable")


# ---------------------------------------------------------------- leaf flattening

def test_leaves_are_ordered_by_name_and_indexed_within_a_vector():
    by_name = {"theta": np.zeros((5, 1, 3)), "mu": np.ones((5, 1, 1))}
    flat, labels = rb.flatten_run(by_name, 5, 1)
    assert labels == ["mu", "theta[0]", "theta[1]", "theta[2]"]
    assert flat.shape == (5, 4)


def test_argmax_names_a_parameter_and_which_moment_failed():
    """b2_argmax must survive the 2d-long index space: it is the only place a reader learns whether
    the sampler got the mean wrong or the spread wrong."""
    rng = np.random.default_rng(7)
    unc = {"a": rng.normal(size=(6, 200, 1)), "b": rng.normal(size=(6, 200, 3))}
    ref = {"a": rng.normal(size=(1200, 1)), "b": rng.normal(scale=1.4, size=(1200, 3))}
    sc = rb.score_ensemble(unc, ref, chains=6, draws=200)
    assert sc["d"] == 4
    name, moment = sc["b2_argmax"].rsplit(" ", 1)
    assert name in sc["labels"], (name, sc["labels"])   # vector leaves are named b[0], b[1], ...
    assert moment == "(x^2)", sc["b2_argmax"]


def test_run_and_reference_are_flattened_to_the_same_leaves():
    """The run is (chains, draws, *shape) and the reference is (rows, leaves). Folding the chain axis
    into the leaf axis on one side only gives a plausible-looking b2 with the wrong leaves, so the two
    flatteners must agree on the leaf names and on chains*draws rows."""
    run = {"mu": np.zeros((4, 3, 1)), "theta": np.zeros((4, 3, 8))}
    ref = {"mu": np.zeros((12, 1)), "theta": np.zeros((12, 8))}
    fr, lr = rb.flatten_run(run, 4, 3)
    ff, lf = rb.flatten_ref(ref)
    assert lr == lf == ["mu", "theta[0]", "theta[1]", "theta[2]", "theta[3]", "theta[4]",
                        "theta[5]", "theta[6]", "theta[7]"]
    assert fr.shape == ff.shape == (12, 9)


def test_the_reference_reader_takes_the_unconstrained_arrays_by_variable_name(tmp_path, monkeypatch):
    """The reference file is flat: `u__<name>` is the unconstrained draws of a value variable, (chains, draws, *shape). G-R1 is
    scored in that space, so `x__<name>` (constrained), `mean__<name>` and the metadata are never read."""
    p = tmp_path / "flat.npz"
    np.savez(p, **{"u__tau_log__": np.zeros((8, 100, 1)), "u__theta": np.zeros((8, 100, 3)), "u__mu": np.zeros((8, 100)),
                   "x__tau_log__": np.ones((8, 100, 1)), "mean__tau_log__": np.float64(7.0), "meta__gate": np.array("PASS")})
    monkeypatch.setattr(rb, "reference_path", lambda model: str(p))
    out = rb.load_reference("m", ["tau_log__", "theta", "mu"])
    assert {k: v.shape for k, v in out.items()} == {"tau_log__": (800, 1), "theta": (800, 3), "mu": (800, 1)}
    assert all(np.allclose(v, 0.0) for v in out.values())          # u__, never x__ / mean__
    assert all(v.dtype == np.float64 for v in out.values())


def test_reference_missing_a_value_var_is_an_error_not_a_silent_subset(tmp_path, monkeypatch):
    """A short reference must fail loudly: b2_max over a subset of leaves can pass G-R1 while an
    unscored variable is badly biased."""
    p = tmp_path / "short.npz"
    np.savez(p, **{"u__mu": np.zeros((8, 100, 1))})
    monkeypatch.setattr(rb, "reference_path", lambda model: str(p))
    with pytest.raises(KeyError) as e:
        rb.load_reference("m", ["mu", "tau_log__"])
    assert "tau_log__" in str(e.value)


def test_missing_reference_is_a_named_gate_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "reference_path", lambda model: str(tmp_path / "nope.npz"))
    with pytest.raises(FileNotFoundError) as e:
        rb.verify_reference("8schools")
    assert "make_reference" in str(e.value)


def test_a_reference_the_manifest_does_not_describe_is_refused(tmp_path, monkeypatch):
    """The gate is only as good as its reference: a file rebuilt without updating the manifest, or edited, must not be scored against."""
    p = tmp_path / "8schools.npz"
    np.savez(p, **{"u__mu": np.zeros((8, 100, 1))})
    monkeypatch.setattr(rb, "reference_path", lambda model: str(p))
    with pytest.raises(ValueError, match="not the file the manifest describes"):
        rb.verify_reference("8schools")
    monkeypatch.setattr(rb, "load_manifest", lambda: {"models": {"8schools": {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}}})
    assert rb.verify_reference("8schools") == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(KeyError, match="not in"):
        rb.verify_reference("not_a_model")


def test_the_manifest_describes_the_four_ladder_models():
    from benchmarks.models import MODELS

    models = rb.load_manifest()["models"]
    assert tuple(models) == MODELS
    for name, entry in models.items():
        assert len(entry["sha256"]) == 64 and entry["gate"] == "PASS", name
        assert entry["path"] == f"~/pf-runs/references/{name}.npz"


def test_real_reference_files_are_keyed_shaped_and_hashed_right():
    """The gate depends on the reference, so its shape and hash are asserted, not assumed. Covers the non-ASCII value-var
    names (GEV) and the transformed-name suffixes that end in `__`. Radon (223 MB) is read too: the sha256 and the
    draws of every ladder model must be what the manifest says."""
    from benchmarks.models import MODELS

    missing = [m for m in MODELS if not os.path.exists(rb.reference_path(m))]
    if missing:
        pytest.skip(f"reference files not built on this host: {missing}")
    for model in MODELS:
        rb.verify_reference(model)
        names = rb.value_var_names(model)
        ref = rb.load_reference(model, names)
        assert sorted(ref) == sorted(names), (model, sorted(ref), sorted(names))
        for n, a in ref.items():
            assert a.shape[0] == 80_000, (model, n, a.shape)   # 8 chains x 10,000 draws
            assert np.isfinite(a).all(), (model, n)


def test_the_draws_digest_is_a_function_of_the_draws_and_their_names():
    a = {"mu": np.arange(6.0).reshape(2, 3, 1), "tau": np.ones((2, 3, 1))}
    same = {"tau": np.ones((2, 3, 1)), "mu": np.arange(6.0).reshape(2, 3, 1)}
    assert rb.draws_digest(a) == rb.draws_digest(same)             # by sorted name, not by insertion order
    nudged = {**a, "mu": np.nextafter(a["mu"], 1e9)}
    assert rb.draws_digest(nudged) != rb.draws_digest(a)           # one ulp is a different digest
    assert rb.draws_digest({"mu": a["mu"], "sigma": a["tau"]}) != rb.draws_digest(a)


# ---------------------------------------------------------------- the scalar value variables

def test_scalar_value_vars_contribute_exactly_chain_x_draw_rows():
    """The population a wrong fold silently skips is the SCALAR value vars: all 3 of GEV's, and 5 of
    radon NC's 7 (sigma_y_log__, sigma_alpha_log__, sigma_beta_log__, mu_alpha, mu_beta). A fold that
    reads (chain, draw) as (rows, leaves) scores 4 draws for those parameters while the vector
    variables score chain*draw, and nothing downstream notices -- b2_max is a max over leaves, so a
    silently under-sampled scalar just stops being the maximum. Asserted by COUNT, not by shape:
    a 2-D (chain, draw) variable must contribute chain*draw rows and a (chain, draw, 85) variable
    chain*draw*85 elements, and both branch must agree when the scalar arrives in either form."""
    import xarray as xr

    C, D = 4, 25
    rng = np.random.default_rng(11)
    post = xr.Dataset({
        "sigma_y_log__": (("chain", "draw"), rng.normal(size=(C, D))),          # 2-D scalar var
        "mu_alpha": (("chain", "draw", "unused"), rng.normal(size=(C, D, 1))),  # 3-D scalar var
        "alpha_raw": (("chain", "draw", "alpha_raw_dim_0"),
                      rng.normal(size=(C, D, 85))),                              # vector var
    })
    unc = {str(v): np.asarray(post[v].values, dtype="float64") for v in post.data_vars}
    ref = {n: rng.normal(size=(C * D, a.shape[2] if a.ndim == 3 and a.shape[2] > 1 else 1))
           for n, a in unc.items()}

    # rows and leaves: 1 scalar from the 2-D form, 1 from the 3-D form, 85 from the vector
    ens, labels = rb.flatten_run(unc, C, D)
    assert ens.shape == (C * D, 87), (ens.shape, labels)
    assert len(labels) == 87 and labels[0] == "alpha_raw[0]"
    n_scalar = sum(1 for l in labels if "[" not in l)
    assert n_scalar == 2, labels                                  # sigma_y_log__, mu_alpha
    assert sum(1 for l in labels if l.startswith("alpha_raw[")) == 85

    sc = rb.score_ensemble(unc, ref, chains=C, draws=D)
    assert sc["d"] == 87
    total_elements = sum(int(np.asarray(v).size) for v in unc.values())
    assert total_elements == C * D * 87, total_elements           # 25 + 100 + 8500

    # The point of the count: a scalar variable must actually be scored. Bias ONE scalar leaf by a
    # whole sd and the gate has to name it -- a fold that dropped scalars would name a vector leaf
    # (or nothing) and the run would still look fine.
    biased = {n: v.copy() for n, v in unc.items()}
    biased["sigma_y_log__"] = np.asarray(biased["sigma_y_log__"]).reshape(C, D) + 1.0
    sc2 = rb.score_ensemble(biased, ref, chains=C, draws=D)
    assert sc2["b2_argmax"] == "sigma_y_log__ (x)", sc2["b2_argmax"]
    assert sc2["b2_max"] > 0.5, sc2["b2_max"]


def test_fold_agrees_whether_a_scalar_arrives_as_2d_or_3d():
    """The exact regression: (chain, draw) and (chain, draw, 1) must score the same parameter set."""
    C, D = 6, 40
    x = np.random.default_rng(12).normal(size=(C, D))
    a = rb.flatten_run({"mu": x.reshape(C, D), "v": np.zeros((C, D, 2))}, C, D)
    b = rb.flatten_run({"mu": x.reshape(C, D, 1), "v": np.zeros((C, D, 2))}, C, D)
    assert a[0].shape == b[0].shape == (C * D, 3)
    assert np.allclose(a[0][:, a[1].index("mu")], x.reshape(C * D))
    assert a[1] == b[1] == ["mu", "v[0]", "v[1]"]




def test_shape_contract_rejects_a_wrong_axis_order_instead_of_reshaping_it():
    """A (draws, chains) transposition is the classic silent error: the product is the same, so a
    reshape-based reader accepts it and every leaf gets the wrong samples. chains and draws are
    checked separately, so this raises."""
    with pytest.raises(ValueError) as e:
        rb.flatten_run({"mu": np.zeros((3, 4))}, chains=4, draws=3)
    assert "expected (chains=4, draws=3" in str(e.value)
    with pytest.raises(ValueError):
        rb.flatten_ref({"mu": np.zeros((4, 3)), "theta": np.zeros((5, 3))})


def test_the_reference_directory_is_pf_reference_dir_or_the_home_default(monkeypatch, tmp_path):
    monkeypatch.setenv("PF_REFERENCE_DIR", str(tmp_path))
    assert rb.reference_path("gev") == str(tmp_path / "gev.npz")
    monkeypatch.delenv("PF_REFERENCE_DIR")
    assert rb.reference_path("gev") == os.path.expanduser("~/pf-runs/references/gev.npz")


@pytest.mark.parametrize("given, expected", [
    (None, "blas__ldflags=-llapack -lblas -lcblas -lm"),
    ("floatX=float32", "floatX=float32,blas__ldflags=-llapack -lblas -lcblas -lm"),
    ("floatX=float32,blas__ldflags=-lopenblas", "floatX=float32,blas__ldflags=-lopenblas"),
])
def test_the_blas_link_line_for_the_path_with_a_space_is_added_to_the_users_flags_not_instead_of_them(given, expected):
    """The repo lives under a path with a space, where PyTensor's own link line fails for any model with a Dot. The gate sets the
    one that works, in a child (import time), and leaves a user's own `blas__ldflags` and other flags alone."""
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTENSOR_FLAGS"}
    env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[1])
    if given is not None:
        env["PYTENSOR_FLAGS"] = given
    out = subprocess.run([sys.executable, "-c", "import os, benchmarks.bias; print(os.environ['PYTENSOR_FLAGS'])"],
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == expected


# ---------------------------------------------------------------- the gate, on a 2-D Gaussian (the real sampler)

def _tiny_model():
    """Value variables `x` (2 leaves) and `s_log__` (the transformed one: the gate scores the unconstrained space)."""
    import pymc as pm

    with pm.Model() as m:
        pm.Normal("x", 0.0, 1.0, size=2)
        pm.HalfNormal("s", 1.0)
    return m


@pytest.fixture
def tiny_ladder(monkeypatch):
    """`benchmarks.models.build_model` returns a small model, and the reference is a synthetic one: a real sampler run on CPU, no files."""
    import benchmarks.models as bm

    rng = np.random.default_rng(0)
    monkeypatch.setattr(bm, "build_model", lambda name: _tiny_model())
    return {"x": rng.normal(size=(800, 2)), "s_log__": np.log(np.abs(rng.normal(size=(800, 1))))}


def _cell(ref, pf_kw=None, ch_kw=None, seed=1, cold=False):
    return rb.run_cell("tiny", 8, 4, 6, seed, cold, ref, "0" * 64, pf_kw, ch_kw)


# keys of attrs["config"] that are results of the run rather than knobs
_RESULT_KEYS = ("frozen_step_size", "frozen_L_nominal", "eps0", "warmup_steps_used")


def test_a_cell_is_a_row_with_the_score_the_hash_of_its_draws_and_the_samplers_own_default_config(tiny_ladder):
    """Nothing is passed beyond the warmup length, so the config the sampler resolves is the one a direct call with no kwargs
    resolves: the gate has no defaults of its own. The row's draws hash is that of the direct call's draws."""
    import pfcheeshmc

    row = _cell(tiny_ladder)
    assert row["error"] is None, row["error"]
    assert row["schema"] == "pf-bias/v1" and row["reference_sha256"] == "0" * 64
    assert row["kwargs"] == {"pathfinder_kwargs": {}, "chees_kwargs": {"draws": 6}}
    assert (row["model"], row["chains"], row["draws_per_chain"], row["warmup"], row["seed"]) == ("tiny", 8, 4, 6, 1)
    assert row["d"] == 3 and row["g_r1"] == "not eligible (C*D < 2000)"
    assert row["b2_max"] >= 0 and row["b2_floor"] > 0 and row["b2_argmax"].split(" (")[0] in ("s_log__", "x[0]", "x[1]")
    assert row["floatX"] == "float64" and row["x64"] is True and row["cold"] is False
    assert row["frozen_step_size"] > 0 and row["steps_mean"] > 0 and row["wall_s"] > 0

    direct = pfcheeshmc.sample_pf_chees_hmc(_tiny_model(), chains=8, draws=4, random_seed=1, progressbar=False,
                                            keep_untransformed=True, chees_kwargs={"draws": 6})
    knobs = lambda c: {k: v for k, v in c.items() if k not in _RESULT_KEYS}
    assert knobs(row["effective"]) == knobs(json.loads(direct.attrs["config"])), "a gate default differs from the sampler's"
    unc = {n: np.asarray(direct.posterior[n].values, dtype="float64").reshape(8, 4, -1) for n in ("x", "s_log__")}
    assert row["draws_sha256"] == rb.draws_digest(unc)
    assert row["frozen_step_size"] == direct.attrs["frozen_step_size"]
    assert row["grads_per_chain_warmup"] == direct.attrs["warmup_grad_evals_per_chain"]


def test_extra_pathfinder_and_chees_options_are_passed_and_land_in_the_row_and_in_effective(tiny_ladder):
    row = _cell(tiny_ladder, {"draws": 3, "imm_source": "pf_closed", "pf_pool_cap": 20}, {"adam_lr": 0.05})
    assert row["error"] is None, row["error"]
    assert row["kwargs"] == {"pathfinder_kwargs": {"draws": 3, "imm_source": "pf_closed", "pf_pool_cap": 20},
                             "chees_kwargs": {"adam_lr": 0.05, "draws": 6}}
    eff = row["effective"]
    assert (eff["imm_source"], eff["pf_draws"], eff["pf_pool_cap"], eff["num_warmup"]) == ("pf_closed", 3, 20, 6)


def test_chees_draws_is_the_warmup_length_and_cannot_be_given_twice(tiny_ladder):
    with pytest.raises(ValueError, match="--warmups"):
        _cell(tiny_ladder, None, {"draws": 5})


def test_an_option_the_sampler_does_not_know_is_a_row_with_the_error_not_a_gap(tiny_ladder):
    """Every stage rejects a key it does not read (a TypeError that names it), so a typo cannot silently run the default."""
    row = _cell(tiny_ladder, {"imm_sorce": "mc"})
    assert row["error"] is not None and "TypeError" in row["error"] and "imm_sorce" in row["error"], row["error"]
    assert "b2_max" not in row and row["wall_s"] > 0


def test_main_writes_one_row_per_model_warmup_and_seed_and_only_the_first_is_cold(tiny_ladder, monkeypatch, tmp_path):
    monkeypatch.setattr(rb, "verify_reference", lambda model: "f" * 64)
    monkeypatch.setattr(rb, "load_reference", lambda model, names: tiny_ladder)
    out = tmp_path / "rows.jsonl"
    rb.main(["--models", "tiny", "--chains", "8", "--draws", "4", "--warmups", "6,8", "--seeds", "1,2", "--out", str(out)])
    rows = [json.loads(l) for l in open(out)]
    assert [(r["warmup"], r["seed"], r["cold"]) for r in rows] == [(6, 1, True), (6, 2, False), (8, 1, False), (8, 2, False)]
    assert all(r["error"] is None and r["reference_sha256"] == "f" * 64 for r in rows)
    assert len({r["draws_sha256"] for r in rows}) == 4                 # other seed, other warmup: other draws
    with pytest.raises(SystemExit, match="--warmups"):
        rb.main(["--models", "tiny", "--out", str(out)])


def test_check_references_only_passes_a_described_file_and_names_the_failure_of_the_others(monkeypatch, tmp_path, capsys):
    import benchmarks.models as bm

    p = tmp_path / "8schools.npz"
    np.savez(p, **{"u__mu": np.zeros((8, 100)), "u__tau_log__": np.zeros((8, 100)), "u__theta": np.zeros((8, 100, 8))})
    monkeypatch.setattr(rb, "reference_path", lambda model: str(p) if model == "8schools" else str(tmp_path / f"{model}.npz"))
    monkeypatch.setattr(rb, "load_manifest", lambda: {"models": {"8schools": {"sha256": rb.sha256_file(str(p))},
                                                                   "gev": {"sha256": "0" * 64}}})
    monkeypatch.setattr(bm, "build_model", lambda name: bm.build_8schools() if name == "8schools" else _tiny_model())
    rb.main(["--models", "8schools,gev", "--check-references-only"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("[ref-guard] 8schools: PASS") and "3 value vars, 10 leaves, 800 reference draws" in lines[0], lines
    assert lines[1].startswith("[ref-guard] gev: FAIL  FileNotFoundError"), lines
