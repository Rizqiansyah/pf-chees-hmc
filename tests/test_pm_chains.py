"""F2h: `pm_chains` PyMC points are shared by the chains, chain c taking repaired row c mod pm_chains.

What a wrong answer here costs: `_run_pathfinder` repaired the (pm_chains, ...) stack and then broadcast it to
(chains, ...) as if it had one row, so any 1 < pm_chains < chains raised "Incompatible types for broadcasting" (it
did so before F2 too). The two counts that worked must not change: pm_chains = 1 (every chain on row 0, the default)
and pm_chains = chains (the identity). Their draws for a fixed seed are pinned by `test_golden.py`.

The cyclic assignment is read off the methods `_run_pathfinder` calls (`_ensure_finite_inits`, `compiled.score_positions`,
`compiled.pathfinder_fit`), not off a log. The PyMC points must differ from row to row for that to mean anything:
`pm_init="jitter+adapt_diag"` gives every chain its own U(-1, 1) jitter (PyMC's `_init_jitter` seeds each chain
separately, and `compiled.jittered_initial_points` does the same), where "adapt_diag" gives every chain the model's test
point.
"""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

from pfcheeshmc import compiled, sample_pf_chees_hmc
from pfcheeshmc.sampler import PfCheesHMC

CHAINS = 8                       # not a multiple of 3, 5 or 7: the points are shared unevenly


def _model():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        tau = pm.HalfNormal("tau", 1.0)
        theta = pm.Normal("theta", mu, tau, shape=3)
        pm.Normal("y", theta, 1.0, observed=np.array([0.3, -0.2, 0.5]))
    return model


def _host(stack):
    return {n: np.asarray(a) for n, a in stack.items()}


def _flat(stack, names, rows):
    """{name: (rows, ...)} -> (rows, D), the names in order."""
    return np.concatenate([np.asarray(stack[n]).reshape(rows, -1) for n in names], axis=1)


class Probe:
    """Records what `_run_pathfinder` hands from one step to the next, by wrapping the methods it calls."""

    def __init__(self, monkeypatch):
        self.raw = self.repaired = self.chain_starts = self.fit_start = None
        self.blocks, self.scores, self.repairs = [], [], []
        probe = self
        real_ensure, real_find = PfCheesHMC._ensure_finite_inits, PfCheesHMC._find_finite_start
        real_run, real_score, real_fit = PfCheesHMC._run_pathfinder, compiled.score_positions, compiled.pathfinder_fit

        def ensure(sampler, init_stack, key, **kw):
            probe.raw = _host(init_stack)
            out = real_ensure(sampler, init_stack, key, **kw)
            probe.repaired = _host(out)
            return out

        def find(sampler, x0, key, **kw):
            probe.repairs.append((_host(x0), np.asarray(jax.random.key_data(key))))
            return real_find(sampler, x0, key, **kw)

        def score(fn, names, values, stack):
            out = real_score(fn, names, values, stack)
            probe.blocks.append(_host(stack))
            probe.scores.append(np.asarray(out))
            return out

        def fit(approximate, fn, names, values, key, initial_position):
            probe.fit_start = [np.asarray(a) for a in initial_position]
            return real_fit(approximate, fn, names, values, key, initial_position)

        def run(sampler, **kwargs):
            out = real_run(sampler, **kwargs)
            probe.chain_starts = _host(out[0])
            return out

        monkeypatch.setattr(PfCheesHMC, "_ensure_finite_inits", ensure)
        monkeypatch.setattr(PfCheesHMC, "_find_finite_start", find)
        monkeypatch.setattr(PfCheesHMC, "_run_pathfinder", run)
        monkeypatch.setattr(compiled, "score_positions", score)
        monkeypatch.setattr(compiled, "pathfinder_fit", fit)

    @property
    def scored(self):
        """The (chains, ...) stack the best-start scoring saw: the stack the Pathfinder stage starts from."""
        return {n: np.concatenate([b[n] for b in self.blocks], axis=0) for n in self.blocks[0]}


def _assert_cyclic(probe, pm_chains, chains):
    names = list(probe.raw)                   # the model's value variables, in the order the sampler keeps them
    rows = np.arange(chains) % pm_chains
    # the premise: pm_chains points were built and screened, and they differ from row to row
    assert all(a.shape[0] == pm_chains for a in probe.raw.values()), {n: a.shape for n, a in probe.raw.items()}
    rep = _flat(probe.repaired, names, pm_chains)
    if pm_chains > 1:
        gaps = np.linalg.norm(rep[:, None] - rep[None], axis=-1)[np.triu_indices(pm_chains, 1)]
        assert gaps.min() > 0.05, f"the {pm_chains} points do not differ, so the assignment is not tested: {gaps}"
    # chain c sits on repaired row c mod pm_chains, exactly
    assert len(probe.blocks) == 1, "one scoring block expected: the chains fit in score_chunk"
    for n in names:
        assert probe.scored[n].shape[0] == chains
        np.testing.assert_array_equal(probe.scored[n], probe.repaired[n][rows], err_msg=n)
    # Pathfinder starts from the highest-logp point among the pm_chains rows
    best = int(np.argmax(np.where(np.isfinite(probe.scores[0][:pm_chains]), probe.scores[0][:pm_chains], -np.inf)))
    np.testing.assert_array_equal(np.concatenate([a.ravel() for a in probe.fit_start]), rep[best])


@pytest.mark.parametrize("pm_chains", [1, 2, 3, 5, 7, CHAINS])
def test_chain_c_starts_from_the_repaired_point_c_mod_pm_chains(monkeypatch, pm_chains):
    """init_source="adapt_diag": the chains' starts are the points themselves, plus a tiny per-chain jitter when shared."""
    sampler = PfCheesHMC(_model(), seed=5)
    probe = Probe(monkeypatch)
    sampler._run_pathfinder(chains=CHAINS, draws=40, seed=5, pm_chains=pm_chains, pm_init="jitter+adapt_diag",
                            init_source="adapt_diag")
    _assert_cyclic(probe, pm_chains, CHAINS)

    names = list(probe.raw)
    rep = _flat(probe.repaired, names, pm_chains)
    starts = _flat(probe.chain_starts, names, CHAINS)
    rows = np.arange(CHAINS) % pm_chains
    if pm_chains == CHAINS:
        np.testing.assert_array_equal(starts, rep)                  # one point per chain: no jitter
    else:
        dist = np.linalg.norm(starts[:, None] - rep[None], axis=-1)    # (chains, pm_chains)
        np.testing.assert_array_equal(dist.argmin(axis=1), rows)       # the nearest point is the chain's own
        own = dist[np.arange(CHAINS), rows]
        assert (own < 1e-2 * (1.0 + np.linalg.norm(rep[rows], axis=1))).all(), own
        assert len({tuple(r) for r in starts.tolist()}) == CHAINS, "chains sharing a point must not start alike"


@pytest.mark.parametrize("init_source", ["pf", "adapt_diag"])
def test_sample_runs_end_to_end_with_pm_chains_between_one_and_chains(monkeypatch, init_source):
    """pm_chains=3, chains=8 (a ragged cycle): finite draws, and the same assignment as above."""
    model = _model()
    probe = Probe(monkeypatch)
    idata = sample_pf_chees_hmc(
        model, draws=10, chains=CHAINS, random_seed=7, progressbar=False, chees_kwargs={"draws": 6},
        pathfinder_kwargs={"pm_chains": 3, "pm_init": "jitter+adapt_diag", "init_source": init_source})
    assert idata.posterior.sizes["chain"] == CHAINS and idata.posterior.sizes["draw"] == 10
    for name, v in idata.posterior.data_vars.items():
        assert np.isfinite(v.values).all(), name
    config = json.loads(idata.attrs["config"])
    assert config["pm_chains"] == 3 and config["init_source"] == init_source      # the points built, not the chains
    _assert_cyclic(probe, 3, CHAINS)


def test_a_point_that_needs_repair_is_repaired_once_and_shared(monkeypatch):
    """The repair runs on the pm_chains rows (one key each), never once per chain: chains 1, 4 and 7 share one result."""
    with pm.Model() as model:
        pm.Normal("x", shape=(4,))
    sampler = PfCheesHMC(model, seed=7)

    def bad_sinv_flat(x):                 # finite value, NaN gradient at 0: the GEV xi = 0 shape
        x = x[0] if isinstance(x, (tuple, list)) else x
        return -jnp.sum(jnp.where(x == 0.0, 0.0, x * x * jnp.sin(1.0 / x)))
    sampler._ld_fn = bad_sinv_flat
    given = np.array([[0.4, -0.3, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0], [-0.2, 0.5, 0.3, -0.4]])
    monkeypatch.setattr(sampler, "_pymc_adapt_diag_start", lambda chains, **kw: {"x": jnp.asarray(given[:chains])})
    probe = Probe(monkeypatch)
    sampler._run_pathfinder(chains=CHAINS, draws=40, seed=7, pm_chains=3, init_source="adapt_diag")

    assert len(probe.repairs) == 1, f"{len(probe.repairs)} repairs for 3 points"
    x0, key_data = probe.repairs[0]
    np.testing.assert_array_equal(x0["x"], given[1])
    expected_key = jax.random.split(jax.random.split(jax.random.key(7), 5)[0], 3)[1]
    np.testing.assert_array_equal(key_data, np.asarray(jax.random.key_data(expected_key)))
    repaired = probe.repaired["x"]
    np.testing.assert_array_equal(repaired[[0, 2]], given[[0, 2]])
    assert (np.abs(repaired[1]) > 0).all() and np.isfinite(repaired[1]).all()
    scored = probe.scored["x"]
    for c in range(CHAINS):
        np.testing.assert_array_equal(scored[c], repaired[c % 3])
    assert np.isfinite(np.asarray(probe.chain_starts["x"])).all()


@pytest.mark.parametrize("pm_chains", [0, CHAINS + 1, -2])
def test_pm_chains_outside_one_to_chains_is_refused(pm_chains):
    with pytest.raises(ValueError, match=r"pm_chains must be in \[1, chains=8\]"):
        PfCheesHMC(_model(), seed=1)._run_pathfinder(chains=CHAINS, draws=40, pm_chains=pm_chains)
