"""F2b-2: the warmup and sampling loops as host-driven chunks of one jitted program (CPU). Correctness side.

The compile counts are in `test_compile_once.py`. What those tests cannot see is whether the draws are still
right, because a chunked loop with a dynamic active count can be fast and wrong in ways nothing else
notices: a key that depends on the chunk size, a Halton index that restarts with each chunk, a skipped step
that is not skipped, a progress bar that stops at the first chunk. Here:

* determinism, and CHUNK-SIZE INVARIANCE: the per-step key is `fold_in(stage_key, global_index)` and the
  Halton index runs in the carried state, so the same seed gives bit-identical draws whatever the chunk
  sizes (even 1) and the stop-rule block;
* the discrete bookkeeping (gradient counts, divergences, steps used) against an independent recomputation
  of the trajectory-length rule from the warmup trace;
* a conjugate Normal-Normal model with a known posterior, against the analytic mean and variance;
* the progress bars (host-driven, at chunk granularity) and the absence of any host callback in the programs;
* the Halton depth (fixed at 30 bits) against the old depth, and the plumbing (`run_chunks`, `HostStack`).
"""
import contextlib
import json
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
import blackjax.mcmc.dynamic_hmc as dynamic_hmc
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest
from tqdm.auto import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.seeded_models import hier  # noqa: E402

import pfcheeshmc.sampler as sampler_module
import pfcheeshmc.utils as utils_module
from pfcheeshmc import compiled
from pfcheeshmc.sampler import PfCheesHMC, sample_pf_chees_hmc

CH = 8
D_THETA = 10
PRIOR_MU = np.linspace(-1.0, 1.0, D_THETA)
PRIOR_SD = np.linspace(0.5, 2.0, D_THETA)
OBS = np.linspace(1.0, -1.0, D_THETA)
OBS_SE = np.linspace(1.0, 0.4, D_THETA)


def analytic():
    """Posterior mean and variance of each coordinate: theta_j ~ N(mu_j, s_j^2), y_j ~ N(theta_j, se_j^2)."""
    precision = 1.0 / PRIOR_SD ** 2 + 1.0 / OBS_SE ** 2
    var = 1.0 / precision
    return var * (PRIOR_MU / PRIOR_SD ** 2 + OBS / OBS_SE ** 2), var


def conjugate_model():
    with pm.Model() as model:
        theta = pm.Normal("theta", PRIOR_MU, PRIOR_SD, shape=D_THETA)
        pm.Normal("y", theta, OBS_SE, observed=OBS)
    return model


def run(seed=5, warmup=40, draws=23, chains=CH, model=None, progressbar=False, **chees):
    return sample_pf_chees_hmc(
        model if model is not None else hier(1, 1), draws=draws, chains=chains, random_seed=seed,
        progressbar=progressbar, chees_kwargs={"draws": warmup, **chees})


def arrays(idata):
    out = {f"posterior/{k}": np.asarray(v.values) for k, v in idata.posterior.data_vars.items()}
    out.update({f"stats/{k}": np.asarray(v.values) for k, v in idata.sample_stats.data_vars.items()})
    return out


def set_chunks(monkeypatch, warmup, sample, transform):
    monkeypatch.setattr(compiled, "WARMUP_CHUNK", warmup)
    monkeypatch.setattr(compiled, "SAMPLE_CHUNK", sample)
    monkeypatch.setattr(compiled, "TRANSFORM_CHUNK", transform)


# ------------------------------------------------------------------------------------ keys
def test_under_the_default_prng_config_fold_in_is_the_old_split_stream():
    """Why the default path's draws did not change: `split(key, N)[i]` IS `fold_in(key, i)` (public API only).

    The loops used `split(stage_key, num_steps)`, whose shape depends on W and D. `fold_in(stage_key, i)` does
    not, and under `jax_threefry_partitionable` (JAX's default) it is the same key. If a JAX default ever
    changes this, the loops stay correct and chunk-invariant, but their draws stop matching the old ones.
    """
    assert jax.config.jax_threefry_partitionable
    key = jax.random.fold_in(jax.random.key(3), 1)
    for n in (1, 7, 50, 1000):
        split = jax.random.key_data(jax.random.split(key, n))
        folded = jax.random.key_data(jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(n)))
        assert np.array_equal(split, folded), n


# ----------------------------------------------------------------------------------- Halton
@pytest.mark.parametrize("W", [1, 7, 24, 25, 100, 1000, 3000])
def test_fixed_halton_depth_equals_the_old_depth_on_the_old_range(W):
    """`max_bits` was ceil(log2(W + 1000)); it is 30 now. The values agree wherever i + 1 < 2**old_bits.

    The old range of a run is i < W + D with D <= 1000, i.e. i + 1 <= W + 1000 <= 2**old_bits. The single
    index i + 1 == 2**old_bits (possible only when W + 1000 is a power of two, e.g. W = 24, D = 1000) was
    the old code's defect: every bit below old_bits is zero there, so its jitter was exactly 0 and the draw
    took 0 leapfrog steps. The fixed depth gives a positive jitter there.
    """
    old_bits = int(np.ceil(np.log2(W + 1000)))
    i = jnp.arange(0, W + 1000)
    old = jax.vmap(lambda k: dynamic_hmc.halton_sequence(k, old_bits))(i)
    new = jax.vmap(lambda k: dynamic_hmc.halton_sequence(k, compiled.HALTON_BITS))(i)
    below = np.asarray(i + 1 < 2 ** old_bits)
    assert below.sum() >= len(i) - 1
    assert np.array_equal(np.asarray(old)[below], np.asarray(new)[below])
    assert np.all(np.asarray(new) > 0) and compiled.HALTON_BITS == 30
    if not below.all():
        assert float(np.asarray(old)[~below][0]) == 0.0


def test_halton_depth_fits_the_integer_width_with_and_without_x64():
    assert dynamic_hmc.halton_sequence(jnp.asarray(5, dtype=jnp.int32), compiled.HALTON_BITS) > 0
    with pytest.raises(ValueError, match="bit width"):
        dynamic_hmc.halton_sequence(jnp.asarray(5, dtype=jnp.int32), 32)


# ------------------------------------------------------------------------------ determinism
def test_the_same_seed_gives_bit_identical_draws_and_another_seed_does_not():
    a, b, other = arrays(run(seed=5)), arrays(run(seed=5)), arrays(run(seed=6))
    assert a.keys() == b.keys()
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["posterior/mu"], other["posterior/mu"])


# ------------------------------------------------------------------- chunk-size invariance
CHUNK_PATHS = {
    "default": dict(warmup=40),
}


@pytest.mark.parametrize("path", list(CHUNK_PATHS))
def test_draws_do_not_depend_on_the_chunk_sizes(monkeypatch, path):
    """The key of step i and the Halton index depend only on i: chunks of 50 or 7 or 1 are the same run."""
    kw = CHUNK_PATHS[path]
    set_chunks(monkeypatch, 50, 100, 100)
    reference = arrays(run(**kw))
    for sizes in [(7, 3, 4), (1, 1, 1), (50, 7, 23)]:
        set_chunks(monkeypatch, *sizes)
        got = arrays(run(**kw))
        for k in reference:
            assert np.array_equal(reference[k], got[k]), f"{path} chunks {sizes}: {k} differs"


def test_draws_do_not_depend_on_the_stop_rule_block():
    """The block is when the rule looks, not part of the stream: a rule that never fires is the plain run."""
    plain = arrays(run(warmup=48))
    for block in (8, 16, 48, 100):
        got = arrays(run(warmup=48, warmup_stop={"block": block, "window": block, "rhat": 0.0}))
        for k in plain:
            assert np.array_equal(plain[k], got[k]), f"block {block}: {k} differs"


def test_a_stop_rule_that_fires_stops_at_a_block_boundary_and_leaves_a_usable_state():
    idata = run(warmup=64, warmup_stop={"block": 16, "window": 16, "rhat": 100.0, "eps_stability": 100.0})
    assert idata.attrs["warmup_steps_used"] == 16
    assert np.all(np.isfinite(idata.posterior["mu"].values))


# ------------------------------------------------------------------------- bookkeeping
@contextlib.contextmanager
def spy_on_warmup(monkeypatch):
    seen = {}
    real = PfCheesHMC._run_chees

    def spy(self, *a, **k):
        out = real(self, *a, **k)
        seen["trace"], seen["states"], seen["params"], seen["info"] = self._warmup_trace, *out
        return out

    monkeypatch.setattr(PfCheesHMC, "_run_chees", spy)
    yield seen


@pytest.mark.parametrize("chunks", [(50, 100, 100), (7, 3, 4)])
def test_discrete_bookkeeping_matches_an_independent_recomputation(monkeypatch, chunks):
    """Gradient counts, divergences and steps used agree with the per-step arrays and with the rule that made them.

    The leapfrog count of step i is `ceil(halton(i) * L_i)` with `L_i = T / eps` as the adapter left it after
    step i - 1 (`L_0 = 1`), the same for every chain. Recomputing it from the warmup TRACE (a separate output
    of the program) pins the Halton index to the global step (it must not restart with a chunk) and checks that
    skipped steps leave no trace in the returned rows. The sampling stage continues the index at W.
    """
    set_chunks(monkeypatch, *chunks)
    W, D = 31, 23
    with spy_on_warmup(monkeypatch) as seen:
        idata = run(warmup=W, draws=D)
    info, trace = seen["info"], seen["trace"]
    assert info.num_integration_steps.shape == (W, CH) and info.is_divergent.shape == (W, CH)
    assert trace.step_size.shape == (W,) and np.array_equal(trace.step, np.arange(2, W + 2))
    assert isinstance(info.num_integration_steps, np.ndarray) and isinstance(trace.step_size, np.ndarray)
    L = np.concatenate([[1.0], trace.trajectory_length[:-1] / trace.step_size[:-1]])
    h = np.asarray(jax.vmap(lambda i: dynamic_hmc.halton_sequence(i, compiled.HALTON_BITS))(jnp.arange(W + D)))
    expect = np.ceil(h[:W] * L).astype(np.int64)
    assert np.array_equal(info.num_integration_steps, np.broadcast_to(expect[:, None], (W, CH)))
    L_frozen = float(seen["params"]["frozen_L_nominal"])
    drawn = idata.sample_stats["num_integration_steps"].values                 # (chains, draws)
    assert np.array_equal(drawn, np.broadcast_to(np.ceil(h[W:] * L_frozen).astype(np.int64)[None], (CH, D)))
    # the totals the artifact carries
    assert idata.attrs["warmup_steps_used"] == W == json.loads(idata.attrs["config"])["warmup_steps_used"]
    assert idata.attrs["warmup_grad_evals_per_chain"] == int(info.num_integration_steps.sum()) // CH
    assert idata.attrs["sampling_grad_evals_per_chain"] == int(drawn.sum()) // CH
    assert int(info.num_integration_steps.sum()) % CH == 0 and int(drawn.sum()) % CH == 0
    assert json.loads(idata.attrs["config"])["num_draws"] == D == idata.posterior["mu"].shape[1]


# ------------------------------------------------------------------ a known posterior
@pytest.fixture(scope="module")
def known_posterior():
    """The conjugate model (C=256, W=200, D=50), `theta` of shape (256, 50, 10)."""
    idata = sample_pf_chees_hmc(conjugate_model(), draws=50, chains=256, random_seed=1, progressbar=False,
                                chees_kwargs={"draws": 200})
    return np.asarray(idata.posterior["theta"].values)


def mc_check(theta, label):
    """Per-coordinate mean and variance within 4 Monte-Carlo standard errors of the analytic values.

    The standard error comes from the spread of the per-chain estimates (chains are independent), so it
    includes the autocorrelation of the draws within a chain.
    """
    mean, var = analytic()
    chains = theta.shape[0]
    chain_mean = theta.mean(axis=1)                                      # (chains, 10)
    chain_sq = ((theta - mean) ** 2).mean(axis=1)                        # unbiased for the variance: known mean
    for name, est, truth in (("mean", chain_mean, mean), ("variance", chain_sq, var)):
        se = est.std(axis=0, ddof=1) / np.sqrt(chains)
        z = (est.mean(axis=0) - truth) / se
        assert np.all(np.abs(z) < 4.0), f"{label} {name}: z = {np.round(z, 2)}"


def test_known_posterior_mean_and_variance_within_4_standard_errors(known_posterior):
    assert known_posterior.shape == (256, 50, D_THETA)
    mc_check(known_posterior, "known posterior")


# ------------------------------------------------------------------------ progress bars
class RecordingBar(tqdm):
    """A real tqdm bar that remembers every refresh: (description, position, postfix)."""

    log = []

    def refresh(self, *args, **kwargs):
        RecordingBar.log.append((self.desc, self.n, self.postfix))
        return super().refresh(*args, **kwargs)


@pytest.mark.parametrize("chunks", [(50, 100, 100), (7, 4, 5)])
def test_progress_bars_reach_the_end_and_report_the_divergences(monkeypatch, chunks):
    """Both bars are driven by the host between chunks: they advance chunk by chunk, end at W and D, and
    their divergence total is the one in the arrays (the old warmup bar only added every 25th step)."""
    set_chunks(monkeypatch, *chunks)
    monkeypatch.setattr(utils_module, "tqdm", RecordingBar)
    monkeypatch.setattr(sampler_module, "tqdm", RecordingBar)
    RecordingBar.log = []
    W, D = 31, 23
    with spy_on_warmup(monkeypatch) as seen:
        idata = run(warmup=W, draws=D, progressbar=True)
    for desc, total, expected_div in (
            ("ChEES Adaptation", W, int(seen["info"].is_divergent.sum())),
            ("Sampling Dynamic HMC", D, int(idata.sample_stats["diverging"].values.sum()))):
        rows = [(n, post) for d, n, post in RecordingBar.log if d == desc and post]
        assert rows, f"{desc}: the bar was never updated"
        ns = [n for n, _ in rows]
        assert ns[-1] == total and ns == sorted(ns), (desc, ns)
        assert f"Divergence: {expected_div}" in rows[-1][1], (desc, rows[-1], expected_div)
        assert "cceptance" in rows[-1][1]
    assert "ε:" in [p for d, n, p in RecordingBar.log if d == "ChEES Adaptation"][-1]
    size = chunks[0]
    assert sorted({n for d, n, p in RecordingBar.log if d == "ChEES Adaptation" and p}) == \
        sorted({min(k, W) for k in range(size, W + size, size)})


def test_a_bar_that_is_off_is_silent(capsys):
    run(warmup=9, draws=5, progressbar=False)
    err = capsys.readouterr().err
    assert "ChEES Adaptation" not in err and "Sampling Dynamic HMC" not in err


# ------------------------------------------------------------ no host callback in any program
def _jaxprs():
    """The jaxprs of the loop programs, built the way the sampler calls them."""
    s = PfCheesHMC(hier(1, 1), seed=0)
    ld = s._logdensity()
    names = ld.names
    pos = {n: jnp.zeros((CH,) + np.shape(v)) for n, v in s.model.initial_point().items() if n in names}
    d = sum(int(np.prod(np.shape(v))) for v in pos.values()) // CH
    settings = dict(optim=0.05, target_accept=0.651, decay_rate=0.5, jitter_amount=1.0, max_leapfrog_steps=100)
    states, astate = compiled.warmup_init(ld.fn, ld.names, ld.values, pos, jnp.asarray(0.1), **settings)
    key, imm = jax.random.key(0), jnp.eye(d)
    z = np.int32(0)
    out = {}
    out["warmup_init"] = jax.make_jaxpr(lambda p: compiled.warmup_init(
        ld.fn, ld.names, ld.values, p, jnp.asarray(0.1), **settings))(pos)
    for stack in (False, True):
        out[f"warmup_chunk stack={stack}"] = jax.make_jaxpr(lambda c: compiled.warmup_chunk(
            ld.fn, ld.names, ld.values, c, imm, key, z, np.int32(3), chunk=5, stack_positions=stack, **settings)
        )((states, astate))
    out["sampling_chunk"] = jax.make_jaxpr(lambda st: compiled.sampling_chunk(
        ld.fn, ld.names, ld.values, st, imm, jnp.asarray(0.3), jnp.asarray(4.0), key, z, np.int32(3),
        chunk=5, jitter_amount=1.0))(states)
    return out


def test_no_host_callback_in_any_loop_program():
    for name, jaxpr in _jaxprs().items():
        text = str(jaxpr)
        assert "callback" not in text, name
        assert "cond[" in text or name == "warmup_init", f"{name}: the skipped steps must be a real conditional"


# ---------------------------------------------------------------------------- plumbing
def test_run_chunks_dispatches_the_next_chunk_before_it_converts_the_previous_one():
    events = []

    def call(carry, start, n):
        events.append(f"dispatch {start}+{n}")
        return carry + n, (start, n)

    def consume(start, n, out):
        events.append(f"consume {start}+{n}")
        assert out == (start, n)

    final = compiled.run_chunks(call, 0, total=23, size=10, consume=consume)
    assert final == 23
    assert events == ["dispatch 0+10", "dispatch 10+10", "consume 0+10", "dispatch 20+3", "consume 10+10",
                      "consume 20+3"]
    events.clear()
    compiled.run_chunks(call, 0, total=0, size=10, consume=consume)
    assert events == ["dispatch 0+0", "consume 0+0"]                  # one empty chunk: the shapes are known


def test_host_stack_cuts_the_real_rows_with_numpy():
    out = lambda start: {"a": np.arange(start, start + 4)[:, None] * np.ones((1, 2)), "b": np.arange(start, start + 4)}
    full = compiled.HostStack(10)
    for start, n in ((0, 4), (4, 4), (8, 2)):
        full.add(start, n, out(start))
    f = full.result()
    assert f["b"].tolist() == list(range(10)) and f["a"].shape == (10, 2)
    empty = compiled.HostStack(0)
    empty.add(0, 0, out(0))
    assert empty.result()["b"].shape == (0,)


def test_the_numpy_harmonic_acceptance_is_the_jax_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        acc = rng.uniform(0.0, 1.0, 12)
        div = rng.uniform(size=12) < 0.3
        assert compiled.chees_harmonic_acceptance_np(acc, div) == pytest.approx(
            float(compiled.chees_harmonic_acceptance(jnp.asarray(acc), jnp.asarray(div))), rel=1e-12)
    acc = np.array([0.0, 0.5])                                       # a zero acceptance: 1/0 -> inf -> 0
    assert compiled.chees_harmonic_acceptance_np(acc, np.zeros(2, bool)) == 0.0
    assert compiled.chees_harmonic_acceptance_np(acc, np.ones(2, bool)) == 0.0     # all divergent


def test_the_transform_pads_the_last_chunk_and_equals_one_big_call(monkeypatch):
    s = PfCheesHMC(hier(1, 1), seed=0)
    rng = np.random.default_rng(1)
    hist = type("H", (), {})()
    hist.position = {n: rng.normal(size=(13, 4) + np.shape(v)) for n, v in s.model.initial_point().items()
                     if n in s.rvs}
    set_chunks(monkeypatch, 50, 100, 5)                                # 13 draws: 5 + 5 + 3 (padded)
    chunked = s._transform_value_samples_to_problem_space(hist)
    set_chunks(monkeypatch, 50, 100, 100)                              # one padded call of 100
    whole = s._transform_value_samples_to_problem_space(hist)
    assert chunked.keys() == whole.keys()
    for k in whole:
        assert whole[k].shape[:2] == (4, 13)
        assert np.array_equal(chunked[k], whole[k]), k
