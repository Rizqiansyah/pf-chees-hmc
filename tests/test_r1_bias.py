"""WP3 harness unit tests (CPU, no sampler, no reference files).

These pin the parts of tools/r1_bias.py that a wrong answer would silently corrupt: the bias
statistic, its noise floor, the G-R1 verdict wording, the frozen-L reconstruction the legacy arm
depends on, and the reference reader's tolerance of whatever container WP0.2 writes.
"""

import os
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import r1_bias as rb  # noqa: E402


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
    flat, labels = rb.flatten_by_leaf(by_name)
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


def test_run_and_reference_are_flattened_by_the_same_rule():
    """The run is (chains, draws, *shape) and the reference is (rows, leaves). Folding the chain
    axis into the leaf axis on one side only gives a plausible-looking b2 with the wrong leaves, so
    both sides go through one function and their labels must agree."""
    run = {"mu": np.zeros((4, 3, 1)), "theta": np.zeros((4, 3, 8))}
    ref = {"mu": np.zeros((12, 1)), "theta": np.zeros((12, 8))}
    fr, lr = rb.flatten_by_leaf(run)
    ff, lf = rb.flatten_by_leaf(ref)
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
