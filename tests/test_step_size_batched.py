"""eps0: the step-size search is ONE vmapped dispatch, and it is the per-probe search.

`_run_step_size_tune` used to dispatch `compiled.search_step_size` once per probe (32 sequential calls: about
0.65 s on the GPU for radon), after a separate feasibility dispatch, with a Python filter and a Python
geometric median. `compiled.search_step_sizes` is one program for a fixed number of probes: the feasibility
filter, the compaction of the feasible probes to the front, the per-probe keys, the vmapped search and the
geometric median are all inside it, so the number of feasible probes can differ between calls with no
recompile. This file pins that it is the same computation:

* eps0 and every per-probe step size are BIT-identical to the per-probe version (the code of cdc4a55, kept
  below as the reference) on the four ladder models, with and without infeasible probes;
* a second call with another number of infeasible probes compiles no XLA program;
* an all-infeasible batch still returns 1e-2.

(`jax.vmap` of a `while_loop` whose predicate is batched keeps the previous carry for finished elements, so
the converged probes are frozen: tests/test_step_size_tune.py's first docstring item describes an older,
different vmapped version.)
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytensor
import pytest

pytensor.config.blas__ldflags = "-llapack -lblas -lcblas -lm"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.models import build_model  # noqa: E402
from compile_counter import count_compiles  # noqa: E402
from seeded_models import hier  # noqa: E402

from pfcheeshmc import compiled  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402

LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")


def per_probe_eps0(sampler, pf_samples, inverse_mass_matrix, seed, num_probes=32):
    """`_run_step_size_tune` as it was at cdc4a55: a feasibility dispatch, then one search dispatch per probe.

    Returns ``(eps0, [the step size of each feasible probe])``.
    """
    ld = sampler._logdensity()
    take = min(num_probes, int(len(next(iter(pf_samples.values())))))
    probes = [{k: jnp.asarray(v)[i] for k, v in pf_samples.items()} for i in range(take)]
    feasible = compiled.feasible_mask(
        ld.fn, ld.names, ld.values, {k: jnp.stack([p[k] for p in probes]) for k in probes[0]})
    probes = [p for p, k in zip(probes, [bool(k) for k in feasible]) if k]
    if not probes:
        return 1e-2, []
    keys = jax.random.split(jax.random.key(seed), len(probes))
    eps_list = [float(compiled.search_step_size(
        ld.fn, ld.names, ld.values, inverse_mass_matrix, keys[i], probes[i])) for i in range(len(probes))]
    eps_ok = [e for e in eps_list if jnp.isfinite(e) and e > 0.0]
    if not eps_ok:
        return 1e-2, eps_list
    return float(jnp.exp(jnp.median(jnp.log(jnp.asarray(eps_ok))))), eps_list


def probes_around_the_initial_point(model, n=16, scale=0.3, infeasible=(), seed=0):
    """``(sampler, {name: (n, ...)} positions, metric)``; the rows in ``infeasible`` are at a non-finite log-density."""
    s = PfCheesHMC(model, seed=1)
    base = s._pymc_adapt_diag_start(1, init="adapt_diag")
    rng = np.random.default_rng(seed)
    pf = {k: np.asarray(v) + scale * rng.normal(size=(n,) + v.shape[1:]) for k, v in base.items()}
    for i in infeasible:
        for k in pf:
            pf[k][i] = 1e300
    d = sum(int(np.prod(v.shape[1:])) for v in pf.values())
    return s, pf, np.ones(d)


@pytest.fixture(scope="module")
def ladder():
    return {name: build_model(name) for name in LADDER}


@pytest.mark.parametrize("name", LADDER)
def test_eps0_is_bit_identical_to_the_per_probe_search_on_the_ladder(ladder, name):
    for infeasible in [(), (0, 5, 6, 15)]:
        s, pf, imm = probes_around_the_initial_point(ladder[name], infeasible=infeasible)
        for seed in (1, 2):
            want, per_probe = per_probe_eps0(s, pf, imm, seed)
            got = s._run_step_size_tune(pf, imm, seed=seed)
            assert got == want, (name, infeasible, seed, got, want)
            assert len(per_probe) == 16 - len(infeasible)
            ld = s._logdensity()
            _, eps, n_ok = compiled.search_step_sizes(
                ld.fn, ld.names, ld.values, imm, jax.random.key(seed), pf, take=16)
            assert int(n_ok) == len(per_probe)
            assert np.array_equal(np.asarray(eps)[:int(n_ok)], np.asarray(per_probe)), (name, infeasible, seed)


def test_probes_differ_so_the_equality_above_is_not_trivial(ladder):
    """Within a batch the probes stop at different step sizes, which is what the frozen-carry batching must keep."""
    s, pf, imm = probes_around_the_initial_point(ladder["gev"], n=32)
    _, per_probe = per_probe_eps0(s, pf, imm, 1)
    assert len(set(per_probe)) >= 2, per_probe


def test_a_different_number_of_infeasible_probes_compiles_nothing_and_changes_nothing_else():
    m = hier(1, 1)
    s, pf, imm = probes_around_the_initial_point(m, n=16)
    s._run_step_size_tune(pf, imm, seed=1)                                    # the program of 16 probes
    for infeasible in [(3,), (0, 1, 2, 3, 4), tuple(range(1, 16)), (7, 9)]:
        s, pf, imm = probes_around_the_initial_point(hier(2, 3), n=16, infeasible=infeasible)
        with count_compiles() as cc:
            got = s._run_step_size_tune(pf, imm, seed=2)
        assert cc.count == 0, (infeasible, str(cc))
        assert got == per_probe_eps0(s, pf, imm, 2)[0], infeasible


def test_eps0_depends_on_the_probes_only_through_the_feasible_ones():
    """Moving the infeasible rows around, or replacing their values, does not change eps0."""
    s, pf, imm = probes_around_the_initial_point(hier(1, 1), n=16, infeasible=(2, 11))
    a = s._run_step_size_tune(pf, imm, seed=3)
    pf2 = {k: v.copy() for k, v in pf.items()}
    for k in pf2:
        pf2[k][2], pf2[k][11] = np.nan, np.inf
    assert s._run_step_size_tune(pf2, imm, seed=3) == a


def test_no_feasible_probe_returns_the_fallback_step_size():
    s, pf, imm = probes_around_the_initial_point(hier(1, 1), n=8, infeasible=range(8))
    assert s._run_step_size_tune(pf, imm, seed=1) == 1e-2
    assert per_probe_eps0(s, pf, imm, 1)[0] == 1e-2


def test_fewer_probes_than_the_probe_count_use_all_of_them():
    s, pf, imm = probes_around_the_initial_point(hier(1, 1), n=5)
    assert s._run_step_size_tune(pf, imm, seed=1) == per_probe_eps0(s, pf, imm, 1)[0]
    assert s._run_step_size_tune(pf, imm, seed=1, eps0_probes=3) == per_probe_eps0(s, pf, imm, 1, num_probes=3)[0]
