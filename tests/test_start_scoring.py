"""F2c: the stratified start pick scores at most START_SCORE_PER_CHAIN * C pool rows.

Under the default `mc` pool, `max(4C, 4d^2)` rows, the stratified pick used to score `min(pool, 64 C)` of them:
the whole pool when d > sqrt(C), i.e. 4d^2 log-density evaluations per call (122,500 at d=175, 1,000,000 at
d=500; 6.4 s for radon in the F2 plan's evidence). The pool and the covariance are unchanged; only the subset
that is ranked by log-density to pick C equally spaced starts is capped.

Pinned here:
* the stride rounds up, so `ceil(total / stride) <= 4C` for every pool size, and is 1 (the whole pool is
  scored, as before) for every pool of at most 4C rows, which includes the default pool for d <= sqrt(C);
* `_run_pathfinder` scores no more than 4C rows, counted at `compiled.score_pool`, for the default and for an
  explicit pool;
* with the whole pool scored the picks are the stratified picks over the whole pool (bit-identical to the
  code before the cap), and with a stride they are the stratified picks over exactly that strided subset;
* a larger pool than the cap still gives distinct, ordered, finite starts.

The start-dispersion ratio of radon against the reference posterior (REPORT run-1 section 3j), old against new,
is `tools/start_dispersion.py`; its numbers are in results/f2/F2cd/SUMMARY.md.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

from pfcheeshmc import compiled
from pfcheeshmc import sampler as sm
from pfcheeshmc.sampler import PfCheesHMC, default_pool_cap, start_score_stride


def old_stride(total, chains):
    return max(total // max(chains * 64, 1), 1)


# ------------------------------------------------------------------------------- the stride
def test_the_stride_caps_the_scored_rows_at_four_per_chain():
    assert sm.START_SCORE_PER_CHAIN == 4
    for chains in (1, 2, 8, 32, 100, 1024):
        for total in sorted({1, 2, chains, 4 * chains - 1, 4 * chains, 4 * chains + 1, 5 * chains, 7 * chains,
                             8 * chains - 1, 8 * chains, 9 * chains, 1000 * chains, 122_500, 1_000_000}):
            stride = start_score_stride(total, chains)
            rows = math.ceil(total / stride)
            assert rows <= 4 * chains or stride == 1 and total <= 4 * chains, (total, chains, stride, rows)
            if total <= 4 * chains:
                assert stride == 1, (total, chains)                        # the whole pool, as before
            else:                                                          # as little striding as keeps the cap
                assert math.ceil(total / (stride - 1)) > 4 * chains, (total, chains, stride)


@pytest.mark.parametrize("d", [3, 10, 25])
def test_the_stride_is_one_and_unchanged_for_d_up_to_sqrt_C_at_C_1024(d):
    """8 Schools (d=10), GEV (d=3) and logit (d=25) at C=1,024: total = 4C, so nothing changes."""
    total = default_pool_cap(1024, d)
    assert total == 4096
    assert start_score_stride(total, 1024) == old_stride(total, 1024) == 1


def test_the_stride_caps_radon_and_the_d_500_pool():
    for d, chains in [(175, 1024), (500, 1024), (175, 32)]:
        total = default_pool_cap(chains, d)
        assert old_stride(total, chains) < start_score_stride(total, chains)
        assert math.ceil(total / start_score_stride(total, chains)) <= 4 * chains
    assert start_score_stride(default_pool_cap(1024, 175), 1024) == 30         # 122,500 -> 4,084 rows


# ------------------------------------------------------------------ what _run_pathfinder scores
def normal_model(d):
    with pm.Model() as m:
        pm.Normal("x", 1.0, 2.0, shape=d)
    return m


class Spy:
    """Record the pool and the scored indices of one `_run_pathfinder` call."""

    def __init__(self, monkeypatch):
        self.pool, self.scored = None, []
        real_draw, real_score = compiled.pathfinder_draw, compiled.score_pool

        def draw(sample, key, state, num_samples):
            self.pool = real_draw(sample, key, state, num_samples)
            return self.pool

        def score(fn, names, values, pool, idx):
            self.scored.append(np.asarray(idx))
            return real_score(fn, names, values, pool, idx)

        monkeypatch.setattr(compiled, "pathfinder_draw", draw)
        monkeypatch.setattr(compiled, "score_pool", score)

    @property
    def idx(self):
        return np.concatenate(self.scored)


def run(model, chains, monkeypatch, **kw):
    spy = Spy(monkeypatch)
    s = PfCheesHMC(model, seed=1)
    pf, imm = s._run_pathfinder(chains=chains, seed=1, score_chunk=64, **kw)
    return s, spy, pf


def reference_picks(s, spy, chains, stride):
    """The stratified pick over the pool rows `0, stride, 2 stride, ...`, written out from the library calls."""
    ld = s._logdensity()
    total = jax.tree_util.tree_leaves(spy.pool)[0].shape[0]
    idx = jnp.arange(0, total, stride)
    lp = compiled.score_pool(ld.fn, ld.names, ld.values, spy.pool, idx)
    pick = sm._stratified_pool_idx(lp, idx, chains)
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[np.asarray(pick)], spy.pool)


def test_the_default_pool_is_scored_whole_when_d_is_at_most_sqrt_C_and_the_pick_is_the_whole_pool_pick(monkeypatch):
    chains = 128                                                   # d = 10 <= sqrt(128): pool = 4C = 512 rows
    s, spy, pf = run(normal_model(10), chains, monkeypatch)
    total = jax.tree_util.tree_leaves(spy.pool)[0].shape[0]
    assert total == 4 * chains and np.array_equal(spy.idx, np.arange(total))
    want = reference_picks(s, spy, chains, stride=1)
    assert np.array_equal(np.asarray(pf["x"]), want[0] if isinstance(want, tuple) else want)


def test_a_pool_larger_than_four_C_is_scored_on_at_most_four_C_rows_and_picks_from_them(monkeypatch):
    chains = 16                                                    # d = 12 > sqrt(16): pool = 4 * 144 = 576 rows
    s, spy, pf = run(normal_model(12), chains, monkeypatch)
    total = jax.tree_util.tree_leaves(spy.pool)[0].shape[0]
    assert total == default_pool_cap(chains, 12) == 576
    stride = start_score_stride(total, chains)
    assert stride > 1 and old_stride(total, chains) == 1           # the old code scored all 576
    assert np.array_equal(spy.idx, np.arange(0, total, stride)) and len(spy.idx) <= 4 * chains
    want = reference_picks(s, spy, chains, stride)
    assert np.array_equal(np.asarray(pf["x"]), want[0] if isinstance(want, tuple) else want)


def test_an_explicit_pool_cap_is_scored_on_at_most_four_C_rows(monkeypatch):
    chains = 8
    s, spy, pf = run(normal_model(5), chains, monkeypatch, pf_pool_cap=20000)
    assert jax.tree_util.tree_leaves(spy.pool)[0].shape[0] == 20000
    assert 0 < len(spy.idx) <= 4 * chains
    starts = np.asarray(pf["x"])
    assert starts.shape == (chains, 5) and np.all(np.isfinite(starts))
    assert len({tuple(r) for r in starts}) == chains                           # distinct starts


def test_the_picks_are_spread_over_the_log_density_ranks(monkeypatch):
    """Equally spaced ranks of the scored subset: the lowest and the highest log-density of the subset are picked."""
    chains = 16
    s, spy, pf = run(normal_model(12), chains, monkeypatch)
    ld = s._logdensity()
    lp = np.asarray(compiled.score_positions(ld.fn, ld.names, ld.values, {"x": pf["x"]}))
    scored = np.asarray(compiled.score_pool(ld.fn, ld.names, ld.values, spy.pool, jnp.asarray(spy.idx)))
    assert lp.min() == scored.min()                                            # rank 0 of the subset
    assert lp.max() <= scored.max() and lp.max() >= np.sort(scored)[-1 - len(scored) // chains]
