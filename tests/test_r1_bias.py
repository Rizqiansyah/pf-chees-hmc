"""WP3 harness unit tests (CPU, no reference files; the R2-WP3 block at the end runs the real sampler
on a 2-D Gaussian).

These pin the parts of tools/r1_bias.py that a wrong answer would silently corrupt: the bias
statistic, its noise floor, the G-R1 verdict wording, the frozen-L reconstruction the legacy arm
depends on, and the reference reader's tolerance of whatever container WP0.2 writes.
"""

import json
import os
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import r1_bias as rb  # noqa: E402
import wp3_summarize as rb2  # noqa: E402


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


# ---------------------------------------------------------------- frozen L reconstruction

def test_nominal_length_recovers_the_frozen_trajectory_length():
    """The legacy arm exposes only integration_steps_fn(i) = ceil(h_i * L), so L must be bracketed.

    Each realised count contributes one bound of width 1/h_i, so the intersection pins L to well
    under one integration step (the Halton jitter gets arbitrarily close to 1, but never exceeds it).
    Half a step of uncertainty on L is stated, not hidden: if this drifts, every
    `frozen_L_nominal` in the R1 tables is wrong, and frozen L is the number the plan's whole
    trajectory-length argument rests on.
    """
    import numpy as np
    import blackjax.mcmc.dynamic_hmc as dh

    import jax.numpy as jnp

    W, L_true = 300, 21.7
    bits = int(np.ceil(np.log2(W + 1000)))

    def fn(i):
        i = jnp.asarray(i, dtype=jnp.int32)
        return float(np.ceil(float(dh.halton_sequence(i, bits)) * L_true))

    assert rb.nominal_length(fn, W) == pytest.approx(L_true, abs=0.6)

    W2, L2 = 1000, 60.0
    bits2 = int(np.ceil(np.log2(W2 + 1000)))
    fn2 = lambda i: float(np.ceil(float(dh.halton_sequence(
        jnp.asarray(i, dtype=jnp.int32), bits2)) * L2))
    assert rb.nominal_length(fn2, W2) == pytest.approx(L2, abs=0.6)


def test_nominal_length_refuses_to_guess_when_the_bounds_are_empty():
    """Realised counts that cannot come from one L (a non-monotone or impossible trace) must return
    None, not a midpoint of an empty interval."""
    assert rb.nominal_length(lambda i: 0.0, 0) is None
    assert rb.nominal_length(lambda i: 1.0 if i == 0 else 1e6, 4) is None


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
    sc = rb.score_ensemble(unc, ref, chains=6, draws=200, superchains=2, nested_limit=1.01)
    assert sc["d"] == 4
    name, moment = sc["b2_argmax"].rsplit(" ", 1)
    assert name in sc["labels"], (name, sc["labels"])   # vector leaves are named b[0], b[1], ...
    assert moment == "(x^2)", sc["b2_argmax"]
    assert sc["nested_rhat_max"] is not None


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


def test_reference_reader_accepts_grouped_or_flat_containers(tmp_path, monkeypatch):
    """WP0.2 writes a flat npz with `u__<name>` / `x__<name>` / `mean__<name>` keys; the reader keys
    on the *variable name* and takes the unconstrained set, because G-R1 is scored in that space.
    The nested-dict form is supported too so a format change does not silently mis-score runs."""
    import _common

    p = tmp_path / "flat.npz"
    np.savez(p, **{"u__tau_log__": np.zeros((8, 100, 1)), "x__tau": np.ones((8, 100, 1)),
                   "mean__tau_log__": np.float64(0.0), "meta__gate": np.array("PASS")})
    monkeypatch.setattr(_common, "reference_path", lambda model: str(p))
    out = rb.load_reference("m", ["tau_log__"], require_stats=False)
    assert out["tau_log__"].shape == (800, 1)
    assert np.allclose(out["tau_log__"], 0.0)          # u__, never x__ / mean__

    grouped = {"unconstrained": {"mu": np.zeros((8, 100, 1))},
               "constrained": {"mu": np.ones((8, 100, 1))}}
    p2 = tmp_path / "g.npz"
    np.savez(p2, data=np.array(grouped, dtype=object), allow_pickle=True)
    monkeypatch.setattr(_common, "reference_path", lambda model: str(p2))
    assert np.allclose(rb.load_reference("m", ["mu"], require_stats=False)["mu"], 0.0)

    bare = {"tau_log__": np.zeros((800, 1))}
    p3 = tmp_path / "b.npz"
    np.savez(p3, **bare)
    monkeypatch.setattr(_common, "reference_path", lambda model: str(p3))
    assert rb.load_reference("m", ["tau_log__"], require_stats=False)["tau_log__"].shape == (800, 1)


def test_reference_missing_a_value_var_is_an_error_not_a_silent_subset(tmp_path, monkeypatch):
    """A short reference must fail loudly: b2_max over a subset of leaves can pass G-R1 while an
    unscored variable is badly biased."""
    import _common

    p = tmp_path / "short.npz"
    np.savez(p, **{"u__mu": np.zeros((8, 100, 1))})
    monkeypatch.setattr(_common, "reference_path", lambda model: str(p))
    with pytest.raises(KeyError) as e:
        rb.load_reference("m", ["mu", "tau_log__"], require_stats=False)
    assert "tau_log__" in str(e.value)


def test_missing_reference_is_a_named_gate_failure(tmp_path, monkeypatch):
    import _common

    monkeypatch.setattr(_common, "reference_path", lambda model: str(tmp_path / "nope.npz"))
    with pytest.raises(FileNotFoundError) as e:
        rb.load_reference("nope", ["mu"])
    assert "make_reference" in str(e.value)


# ---------------------------------------------------------------- superchains start option

def test_superchains_replicates_K_distinct_starts_and_validates_K():
    """pathfinder_kwargs={'superchains': K}: K stratified Pathfinder draws, each used C/K times."""
    import pymc as pm
    from pfcheeshmc.sampler import PfCheesHMC

    with pm.Model():
        x = pm.Normal("x", 0.0, 1.0, size=2)
        s = PfCheesHMC(pm.modelcontext(None), seed=11)
        pos, imm = s._run_pathfinder(chains=4, draws=8, init_pick="stratified", superchains=2)
        flat = np.stack([np.concatenate([np.ravel(pos[n][c]) for n in s.rvs]) for c in range(4)])
        assert np.array_equal(flat[0], flat[1]) and np.array_equal(flat[2], flat[3]), \
            "within a superchain the start must be replicated exactly"
        assert not np.array_equal(flat[0], flat[2]), "different superchains must start apart"

        with pytest.raises(ValueError):
            s._run_pathfinder(chains=4, draws=8, superchains=3)
        with pytest.raises(ValueError):
            s._run_pathfinder(chains=4, draws=8, superchains=5)


def test_real_reference_files_are_keyed_and_shaped_right():
    """The gate depends on the reference, so its shape is asserted, not assumed. Covers the
    non-ASCII value-var names (GEV) and the transformed-name suffixes that end in `__`.
    radon (223 MB) is skipped here on purpose; the harness prints the leaf labels every run and
    raises on any run/reference mismatch."""
    import _common

    for model in ("8schools", "gev"):
        p = _common.reference_path(model)
        if not os.path.exists(p):
            pytest.skip(f"WP0.2 reference not built yet: {p}")
        import pymc as pm  # noqa: F401  (bench_ladder's loader needs PyMC importable)

        names = rb.value_var_names(model)
        ref = rb.load_reference(model, names)   # strict: the real files must satisfy the contract
        assert sorted(ref) == sorted(names), (model, sorted(ref), sorted(names))
        for n, a in ref.items():
            assert a.shape[0] == 80_000, (model, n, a.shape)   # 8 chains x 10,000 draws
            assert np.isfinite(a).all(), (model, n)


def test_collapsed_reference_statistics_are_rejected_at_load(tmp_path, monkeypatch):
    """WP0.2's aggregates must keep the parameter axis. Live defect (2026-10-19): the first
    8schools.npz stored mean__theta / mcse_x__theta as single numbers, so any later WP that reads
    them would score 8 schools against one scalar and never notice."""
    import _common

    theta = np.zeros((8, 100, 8))
    good = {"u__theta": theta, "u__mu": np.zeros((8, 100)),
            "mean__theta": np.zeros(8), "var__theta": np.ones(8),
            "mcse_x__theta": np.full(8, 0.01), "mcse_x2__theta": np.full(8, 0.01),
            "mean__mu": np.float64(0.0), "var__mu": np.float64(1.0),
            "mcse_x__mu": np.float64(0.01), "mcse_x2__mu": np.float64(0.01)}
    p = tmp_path / "good.npz"
    np.savez(p, **good)
    monkeypatch.setattr(_common, "reference_path", lambda model: str(p))
    assert set(rb.load_reference("m", ["mu", "theta"])) == {"mu", "theta"}

    bad = dict(good, mean__theta=np.float64(0.03), mcse_x__theta=np.float64(0.01))
    p2 = tmp_path / "bad.npz"
    np.savez(p2, **bad)
    monkeypatch.setattr(_common, "reference_path", lambda model: str(p2))
    with pytest.raises(ValueError) as e:
        rb.load_reference("m", ["mu", "theta"])
    assert "mean__theta has shape (), sample leaf shape (8,)" in str(e.value)
    assert "mcse_x__theta" in str(e.value)

    # the exemption is explicit and the sample arrays still load (so b2 stays computable)
    out = rb.load_reference("m", ["mu", "theta"], require_stats=False)
    assert out["theta"].shape == (800, 8)


def test_superchains_off_leaves_the_pick_unchanged():
    import pymc as pm
    from pfcheeshmc.sampler import PfCheesHMC

    with pm.Model():
        pm.Normal("x", 0.0, 1.0, size=2)
        s = PfCheesHMC(pm.modelcontext(None), seed=12)
        a, _ = s._run_pathfinder(chains=4, draws=8, init_pick="stratified")
        b, _ = s._run_pathfinder(chains=4, draws=8, init_pick="stratified", superchains=4)
        for n in s.rvs:
            assert np.allclose(np.asarray(a[n]), np.asarray(b[n])), \
                "K == chains must reproduce the plain stratified pick"


# ---------------------------------------------------------------- W-selection (the deliverable)

def _row(model="m", arm="legacy", W=100, seed=1, rep=1, b2=0.001, floor=0.002,
         chains=1024, draws=10, gw=1000.0, gs=100.0, error=None):
    return {"model": model, "arm": arm, "warmup": W, "seed": seed, "rep": rep, "b2_max": b2,
            "b2_floor": floor, "chains": chains, "draws_per_chain": draws,
            "grads_per_chain_warmup": gw, "grads_per_chain_sampling": gs, "error": error}


def test_smallest_passing_w_must_pass_on_every_seed():
    rows = [_row(W=50, seed=1, b2=0.001), _row(W=50, seed=2, b2=0.09),      # one seed fails
            _row(W=100, seed=1, b2=0.001), _row(W=100, seed=2, b2=0.004),
            _row(W=200, seed=1, b2=0.001), _row(W=200, seed=2, b2=0.001)]
    t = rb2.summarise(rows)[0]
    assert t["smallest_passing_W"] == 100
    assert t["grads_per_chain_at_that_W"] == [1100.0, 1100.0]
    states = {W: st for W, st, _, _ in t["cells"]}
    assert states[50].startswith("fail") and states[200] == "PASS"


def test_unresolvable_and_ineligible_cells_are_never_reported_as_passes():
    rows = [_row(W=100, seed=1, b2=0.0001, floor=0.02), _row(W=100, seed=2, b2=0.0001, floor=0.02),
            _row(W=200, seed=1, b2=0.0001, floor=0.001, chains=128),
            _row(W=200, seed=2, b2=0.0001, floor=0.001, chains=128)]
    t = rb2.summarise(rows)[0]
    states = {W: st for W, st, _, _ in t["cells"]}
    assert states[100].startswith("not resolvable")
    assert states[200].startswith("not eligible")
    assert t["smallest_passing_W"] is None


def test_cold_rows_starts_rows_and_error_rows_are_excluded():
    rows = [_row(W=100, seed=1, rep=0, b2=0.001), _row(W=100, seed=2, rep=0, b2=0.001),
            _row(model="m2", W=0, seed=1),                      # pf_starts
            _row(W=300, seed=1, b2=None)]
    # nothing left to summarise: no table entry, so no W can be claimed
    assert rb2.summarise(rows) == []
    # a re-run replaces the earlier row for the same cell (tagging or contention recovery)
    dup = [_row(W=100, seed=1, b2=0.5), _row(W=100, seed=1, b2=0.001), _row(W=100, seed=2, b2=0.001)]
    t = rb2.summarise(dup)[0]
    assert t["cells"][0][2] == 0.001, "the later row for the same cell must win"
    assert t["smallest_passing_W"] == 100


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

    sc = rb.score_ensemble(unc, ref, chains=C, draws=D, superchains=2, nested_limit=1.01)
    assert sc["d"] == 87
    total_elements = sum(int(np.asarray(v).size) for v in unc.values())
    assert total_elements == C * D * 87, total_elements           # 25 + 100 + 8500

    # The point of the count: a scalar variable must actually be scored. Bias ONE scalar leaf by a
    # whole sd and the gate has to name it -- a fold that dropped scalars would name a vector leaf
    # (or nothing) and the run would still look fine.
    biased = {n: v.copy() for n, v in unc.items()}
    biased["sigma_y_log__"] = np.asarray(biased["sigma_y_log__"]).reshape(C, D) + 1.0
    sc2 = rb.score_ensemble(biased, ref, chains=C, draws=D, superchains=2, nested_limit=1.01)
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


# ---------------------------------------------------------------- R2-WP3: passthrough + pooled seeds

def _tiny_model():
    import pymc as pm

    with pm.Model() as m:
        pm.Normal("x", 0.0, 1.0, size=2)
    return m


@pytest.fixture
def tiny_harness(monkeypatch):
    """r1_bias pointed at a 2-D Gaussian and a synthetic reference: a real sampler run on CPU, no files."""
    ref = np.random.default_rng(0).normal(size=(800, 2))
    monkeypatch.setattr(rb, "build_model", lambda name: _tiny_model())
    monkeypatch.setattr(rb, "load_reference", lambda name, names, require_stats=True: {"x": ref})


def _run_tiny(tmp_path, *flags):
    args = rb.parse_args(["--models", "tiny", "--chains", "8", "--draws", "4", "--warmups", "6",
                          "--seeds", "1", "--repeat", "1", "--arm", "t", "--step", "T",
                          "--out", str(tmp_path / "rows.jsonl"), *flags])
    import bench_ladder as bl

    return args, rb.run_cell(bl, "tiny", args, 6, 1, 0, False)


# keys of attrs["config"] that are results of the run rather than knobs
_RESULT_KEYS = ("frozen_step_size", "frozen_L_nominal", "eps0", "warmup_steps_used")


def test_with_no_flags_the_harness_passes_nothing_and_effective_is_the_samplers_default(
        tiny_harness, tmp_path):
    """Run 1 injected --pf-draws 1 / --orientation precision / --iterate ma, so the 'default arm' was
    never the sampler's default (plan Part B 1.3). With no flags, nothing is passed beyond the warmup
    length, and the config the sampler resolves is the one a direct call with no kwargs resolves."""
    import pfcheeshmc

    args, row = _run_tiny(tmp_path)
    kw = json.loads(row["kwargs"])
    assert kw["pathfinder_kwargs"] == {} and kw["chees_kwargs"] == {"draws": 6}, kw
    assert row["schema"] == "pf-r2/v1" and row["step"] == "T" and row["pooled"] is False
    assert row["error"] is None, row["error"]

    eff = row["effective"]
    assert eff["imm_orientation"] == "covariance"                 # the sampler's default
    direct = json.loads(pfcheeshmc.sample_pf_chees_hmc(
        _tiny_model(), chains=8, draws=4, random_seed=1, progressbar=False,
        chees_kwargs={"draws": 6}).attrs["config"])
    assert eff["iterate"] == direct["iterate"]
    knobs = lambda c: {k: v for k, v in c.items() if k not in _RESULT_KEYS}
    assert knobs(eff) == knobs(direct), "a harness default differs from the sampler's"
    # no refresh: the two refresh scalars exist in the row and are null
    assert row["metric_refresh_T0"] is None and row["metric_refresh_lmax_whitened"] is None


def test_flags_that_are_given_are_passed_and_land_in_effective(tiny_harness, tmp_path):
    args, row = _run_tiny(tmp_path, "--iterate", "raw", "--orientation", "precision",
                          "--pf-draws", "3", "--imm-source", "pf_closed", "--start-scale", "1.5")
    kw = json.loads(row["kwargs"])
    assert kw["pathfinder_kwargs"] == {"draws": 3, "imm_orientation": "precision",
                                       "imm_source": "pf_closed", "pf_start_scale": 1.5}
    assert kw["chees_kwargs"] == {"draws": 6, "iterate": "raw"}
    eff = row["effective"]
    assert (eff["iterate"], eff["imm_orientation"], eff["imm_source"], eff["pf_draws"],
            eff["pf_start_scale"]) == ("raw", "precision", "pf_closed", 3, 1.5)
    assert row["error"] is None, row["error"]


def test_a_flag_that_effective_contradicts_aborts_the_run_and_keeps_the_row(
        tiny_harness, tmp_path, monkeypatch):
    """The run-1 `iterate` bug (LEDGER E13): single-phase _run_chees ignored chees_kwargs['iterate'],
    so cells labelled raw ran ma and nothing in the row said so. Simulate it by making the sampler's
    config record claim 'ma' while the harness asked for 'raw'."""
    from pfcheeshmc.sampler import PfCheesHMC

    orig = PfCheesHMC._config_record

    def lying(self, *a, **k):
        cfg = json.loads(orig(self, *a, **k))
        cfg["iterate"] = "ma"
        return json.dumps(cfg)

    monkeypatch.setattr(PfCheesHMC, "_config_record", lying)
    with pytest.raises(rb.ConfigMismatch, match=r"iterate: asked 'raw', sampler ran 'ma'"):
        _run_tiny(tmp_path, "--iterate", "raw")
    rows = [json.loads(l) for l in open(tmp_path / "rows.jsonl")]
    assert len(rows) == 1 and "ConfigMismatch" in rows[0]["error"]
    assert rows[0]["effective"]["iterate"] == "ma"                 # the evidence is on disk


def test_check_effective_unit_semantics():
    eff = {"iterate": "raw", "pf_draws": 4000, "pf_start_scale": 1.0, "superchains": None,
           "metric_refresh": "none", "num_warmup": 50, "metric_refresh_warmup": None}
    rb.check_effective(eff, {"iterate": "raw", "pf_start_scale": 1, "superchains": None})   # 1 == 1.0
    rb.check_effective(None, {"iterate": "ma"})              # a tree without the attr: nothing to check
    with pytest.raises(rb.ConfigMismatch, match="pf_draws: asked 1, sampler ran 4000"):
        rb.check_effective(eff, {"pf_draws": 1})
    with pytest.raises(rb.ConfigMismatch, match="absent"):   # a missing key is a mismatch, not a skip
        rb.check_effective(eff, {"imm_source": "mc"})
    with pytest.raises(rb.ConfigMismatch):                   # False is not 0 is not None
        rb.check_effective({"metric_refresh_reseed": None}, {"metric_refresh_reseed": False})


def test_none_flags_add_no_key_and_the_refresh_warmup_is_resolved_and_clamped():
    a = rb.parse_args([])
    assert rb.pathfinder_kwargs_from(a) == {} and rb.chees_kwargs_from(a, 100) == {"draws": 100}
    a = rb.parse_args(["--metric-refresh", "sample_diag"])
    assert rb.chees_kwargs_from(a, 100) == {"draws": 100, "metric_refresh": "sample_diag",
                                            "metric_refresh_warmup": 50}
    assert rb.chees_kwargs_from(a, 3)["metric_refresh_warmup"] == 2          # max(2, W // 2)
    a = rb.parse_args(["--metric-refresh", "sample_diag", "--metric-refresh-warmup", "30",
                       "--no-metric-refresh-reseed"])
    kw = rb.chees_kwargs_from(a, 100)
    assert kw["metric_refresh_warmup"] == 30 and kw["metric_refresh_reseed"] is False
    # the sampler clamps phase A to [2, W - 2]; the guard expects the clamped value
    want = rb.expected_config({}, kw, W=25)
    assert want["metric_refresh_warmup"] == 23 and want["metric_refresh"] == "sample_diag"
    with pytest.raises(SystemExit):                                          # needs a refresh
        rb.parse_args(["--metric-refresh-warmup", "30"])


def _draws(seed, chains=64, draws=10):
    rng = np.random.default_rng(seed)
    return {"a": rng.normal(size=(chains, draws, 1)) + 0.1, "b": rng.normal(size=(chains, draws, 3))}


def test_pooling_three_identical_seeds_keeps_b2_and_divides_the_floor_by_three():
    unc = _draws(5)
    ref = {"a": np.random.default_rng(6).normal(size=(4000, 1)),
           "b": np.random.default_rng(7).normal(size=(4000, 3))}
    one = rb.score_ensemble(unc, ref, chains=64, draws=10, superchains=None, nested_limit=1.01)
    pooled, chains = rb.pool_seed_scores({1: unc, 2: unc, 3: unc}, ref, 10, 1.01)
    assert chains == 192
    assert pooled["b2_max"] == pytest.approx(one["b2_max"], rel=1e-9)
    assert pooled["b2_argmax"] == one["b2_argmax"]
    assert pooled["b2_floor"] == pytest.approx(one["b2_floor"] / 3.0, rel=0.05)
    assert pooled["nested_rhat_max"] is None                 # not defined across seeds


def test_pooling_independent_seeds_lowers_the_floor_about_as_one_over_S():
    ref = {"a": np.random.default_rng(6).normal(size=(4000, 1)),
           "b": np.random.default_rng(7).normal(size=(4000, 3))}
    per = [rb.score_ensemble(_draws(s), ref, 64, 10, None, 1.01)["b2_floor"] for s in (1, 2, 3, 4)]
    pooled, _ = rb.pool_seed_scores({s: _draws(s) for s in (1, 2, 3, 4)}, ref, 10, 1.01)
    assert pooled["b2_floor"] == pytest.approx(np.mean(per) / 4.0, rel=0.35)


def test_run_pooled_writes_one_pooled_row_that_smallest_pass_reads(tmp_path, monkeypatch):
    import argparse
    import summarise_rows as sr

    ref = {"a": np.random.default_rng(6).normal(size=(4000, 1)),
           "b": np.random.default_rng(7).normal(size=(4000, 3))}
    monkeypatch.setattr(rb, "load_reference", lambda name, names, require_stats=True: ref)
    out = tmp_path / "p.jsonl"
    args = argparse.Namespace(wp="WP3", arm="A", draws=10, chains=64, out=str(out), step="M2a",
                              nested_limit=1.01, cpu_contended=False,
                              allow_collapsed_reference_stats=False)
    seed_rows = []
    for s in (1, 2, 3):
        eff = {"iterate": "raw", "seed": s, "frozen_step_size": 0.1 * s, "num_draws": 10}
        seed_rows.append({"model": "m", "arm": "A", "warmup": 50, "seed": s, "rep": 0, "cold": False,
                          "b2_max": 0.001, "b2_floor": 0.002, "chains": 1024, "draws_per_chain": 10,
                          "grads_per_chain_warmup": 100.0, "grads_per_chain_sampling": 20.0,
                          "error": None, "effective": eff, "kwargs": json.dumps({"k": 1}),
                          "pooled": False})
    kept = {s: (_draws(s), r) for s, r in zip((1, 2, 3), seed_rows)}
    row = rb.run_pooled("m", args, 50, 0, False, kept)
    assert row["pooled"] is True and row["seed"] == "pooled" and row["chains"] == 192
    assert row["schema"] == "pf-r2/v1" and row["step"] == "M2a" and row["pooled_seeds"] == [1, 2, 3]
    assert row["effective"] == {"iterate": "raw", "num_draws": 10}   # what all seeds share
    assert row["error"] is None and row["warmup"] == 50 and row["cold"] is False
    assert row["g_r1_pool"] in ("pass", "fail") or row["g_r1_pool"].startswith("not")
    assert len([l for l in open(out)]) == 1

    # one seed alone cannot be pooled
    assert rb.run_pooled("m", args, 50, 0, False, {1: kept[1]}) is None

    # smallest-pass reads it as the diagnostic, never as a fourth seed
    res, _ = sr.smallest_pass(seed_rows + [json.loads(l) for l in open(out)])
    assert res[0]["pt"] == 50 and res[0]["seeds"] == [1, 2, 3]
    assert res[0]["pooled"] == (row["b2_max"], row["b2_floor"])


def test_gev_probe_start_scale_is_forwarded_and_rescales_the_start_dispersion(
        tiny_harness, tmp_path, monkeypatch):
    """M2c reads the start sd / reference sd table at --start-scale 2 and 3: the starts are rescaled
    about the Pathfinder mean AFTER the picks, so the same seed gives exactly S times the ratio."""
    import _common
    import gev_divergence_probe as probe

    monkeypatch.setattr(_common, "require_gpu_lock", lambda what="": None)   # CPU stand-in for the lock
    out = tmp_path / "probe.jsonl"
    base = ["--models", "tiny", "--chains", "32", "--out", str(out)]
    probe.main(base)
    probe.main(base + ["--start-scale", "2"])
    one, two = [json.loads(l) for l in open(out)]
    assert one["start_scale"] is None and one["effective"]["pf_start_scale"] == 1.0
    assert two["start_scale"] == 2.0 and two["effective"]["pf_start_scale"] == 2.0
    assert one["effective"]["pf_draws"] == 4000                     # the sampler's, not a probe default
    assert two["median_ratio"] == pytest.approx(2.0 * one["median_ratio"], rel=2e-3)
